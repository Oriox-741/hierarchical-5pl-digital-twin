"""Stable-Baselines3 callback that streams sampled training transitions to PostgreSQL."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from queue import Empty, Full, Queue
from threading import Thread
from typing import Any
from uuid import UUID, uuid4

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from src.learn.episode_store import EpisodeStore, EpisodeTransition
from src.learn.model_registry import Algorithm
from src.sense.db_pool import DatabaseConfig, DatabasePool


class TrainingTraceCallback(BaseCallback):
    """Persist sampled SB3 transitions without turning PostgreSQL into a step firehose.

    Stable-Baselines3 training is synchronous, while the project database stack is
    asyncpg-based. This callback bridges those worlds with source-side sampling,
    local batch coalescing, a bounded queue, and a writer thread that owns its
    asyncio event loop and asyncpg pool.
    """

    def __init__(
        self,
        *,
        algorithm: Algorithm,
        policy_id: str,
        environment_id: str,
        agent_id: str = "global_controller",
        agent_role: str = "control_tower",
        team_id: str = "joint_from_scratch",
        db_config: DatabaseConfig | None = None,
        flush_batch_size: int = 5_000,
        sample_every_n_steps: int = 50,
        log_episode_boundaries: bool = True,
        queue_max_batches: int = 2_048,
        shutdown_timeout_seconds: float | None = 300.0,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        if flush_batch_size <= 0:
            raise ValueError("flush_batch_size must be greater than zero.")
        if sample_every_n_steps <= 0:
            raise ValueError("sample_every_n_steps must be greater than zero.")
        if queue_max_batches <= 0:
            raise ValueError("queue_max_batches must be greater than zero.")
        if shutdown_timeout_seconds is not None and shutdown_timeout_seconds <= 0.0:
            raise ValueError("shutdown_timeout_seconds must be greater than zero when provided.")

        self.algorithm: Algorithm = algorithm
        self.policy_id = policy_id
        self.environment_id = environment_id
        self.agent_id = agent_id
        self.agent_role = agent_role
        self.team_id = team_id
        self.db_config = db_config or DatabaseConfig()
        self.flush_batch_size = flush_batch_size
        self.sample_every_n_steps = sample_every_n_steps
        self.log_episode_boundaries = log_episode_boundaries
        self.shutdown_timeout_seconds = shutdown_timeout_seconds

        self._queue: Queue[Sequence[EpisodeTransition] | None] = Queue(maxsize=queue_max_batches)
        self._pending: list[EpisodeTransition] = []
        self._writer_thread: Thread | None = None
        self._writer_error: BaseException | None = None
        self._previous_obs: np.ndarray | None = None
        self._episode_ids: list[UUID] = []
        self._step_ids: list[int] = []
        self._dropped_batches = 0
        self._dropped_transitions = 0

    def _on_training_start(self) -> None:
        initial_obs = getattr(self.model, "_last_obs", None)
        if initial_obs is None:
            initial_obs = self.locals.get("new_obs")
        if initial_obs is None:
            raise RuntimeError("Cannot initialize trace callback: SB3 did not expose initial observations.")

        self._previous_obs = _ensure_observation_batch(initial_obs)
        env_count = int(self._previous_obs.shape[0])
        self._episode_ids = [uuid4() for _ in range(env_count)]
        self._step_ids = [0 for _ in range(env_count)]
        self._pending.clear()
        self._dropped_batches = 0
        self._dropped_transitions = 0

        self._writer_thread = Thread(
            target=self._run_writer,
            name="sb3-training-trace-writer",
            daemon=False,
        )
        self._writer_thread.start()

    def _on_step(self) -> bool:
        if self._writer_error is not None:
            raise RuntimeError("Training trace writer failed.") from self._writer_error
        if self._previous_obs is None:
            raise RuntimeError("Training trace callback was not initialized.")

        new_obs = _ensure_observation_batch(self.locals["new_obs"])
        env_count = int(new_obs.shape[0])
        actions = _ensure_action_batch(self.locals["actions"], env_count)
        rewards = np.asarray(self.locals["rewards"], dtype=np.float64).reshape(-1)
        dones = np.asarray(self.locals["dones"], dtype=np.bool_).reshape(-1)
        infos = _ensure_infos(self.locals.get("infos"), env_count)

        previous_obs = _align_previous_obs(self._previous_obs, env_count)
        recorded_at = datetime.now(timezone.utc)
        transitions: list[EpisodeTransition] = []

        for env_index in range(env_count):
            info = infos[env_index]
            truncated = bool(info.get("TimeLimit.truncated", False))
            done = bool(dones[env_index])
            terminated = bool(done and not truncated)
            should_log = self._should_log_transition(done)
            terminal_observation = info.get("terminal_observation")
            next_observation = (
                _single_observation_payload(terminal_observation)
                if terminal_observation is not None
                else _single_observation_payload(new_obs[env_index])
            )

            if should_log:
                reward_components = _reward_components(info)
                global_reward = reward_components.get("global", float(rewards[env_index]))
                ppo_local_reward = reward_components.get("ppo_local")
                dqn_local_reward = reward_components.get("dqn_local")
                local_reward = (
                    ppo_local_reward
                    if self.algorithm == "ppo"
                    else dqn_local_reward if self.algorithm == "dqn" else None
                )
                transitions.append(
                    EpisodeTransition(
                        episode_id=self._episode_ids[env_index],
                        step_id=self._step_ids[env_index],
                        recorded_at=recorded_at,
                        observation=_single_observation_payload(previous_obs[env_index]),
                        action=_action_payload(actions[env_index], self.algorithm),
                        next_observation=next_observation,
                        projected_action=_projected_action_payload(info),
                        reward=float(rewards[env_index]),
                        terminated=terminated,
                        truncated=truncated,
                        policy_id=self.policy_id,
                        environment_id=self.environment_id,
                        agent_id=self.agent_id,
                        agent_role=self.agent_role,
                        team_id=self.team_id,
                        local_reward=float(local_reward if local_reward is not None else rewards[env_index]),
                        global_reward=float(global_reward),
                        ppo_local_reward=ppo_local_reward,
                        dqn_local_reward=dqn_local_reward,
                        info=_info_payload(info),
                    )
                )

            if done:
                self._episode_ids[env_index] = uuid4()
                self._step_ids[env_index] = 0
            else:
                self._step_ids[env_index] += 1

        if self._writer_thread is not None and not self._writer_thread.is_alive():
            if self._writer_error is not None:
                raise RuntimeError("Training trace writer failed.") from self._writer_error
            raise RuntimeError("Training trace writer stopped unexpectedly.")
        if transitions:
            self._pending.extend(transitions)
            if len(self._pending) >= self.flush_batch_size:
                self._enqueue_pending()
        self._previous_obs = new_obs.copy()
        return True

    def _on_training_end(self) -> None:
        if self._writer_error is not None:
            raise RuntimeError("Training trace writer failed during shutdown.") from self._writer_error
        if self._writer_thread is None:
            return
        if not self._writer_thread.is_alive():
            return

        self._enqueue_pending()
        try:
            self._queue.put(None, timeout=self.shutdown_timeout_seconds)
        except Full as exc:
            raise RuntimeError("Training trace queue did not accept shutdown signal.") from exc

        self._writer_thread.join(timeout=self.shutdown_timeout_seconds)
        if self._writer_thread.is_alive():
            raise TimeoutError("Training trace writer did not flush and stop before shutdown timeout.")
        if self._writer_error is not None:
            raise RuntimeError("Training trace writer failed during shutdown.") from self._writer_error
        if self.verbose > 0 and self._dropped_transitions > 0:
            print(
                "TrainingTraceCallback dropped "
                f"{self._dropped_transitions} transitions across {self._dropped_batches} queued batches "
                "because PostgreSQL could not keep up."
            )

    def _run_writer(self) -> None:
        try:
            asyncio.run(self._write_loop())
        except BaseException as exc:  # pragma: no cover - surfaced by training thread
            self._writer_error = exc

    async def _write_loop(self) -> None:
        db = DatabasePool(self.db_config)
        await db.startup()
        store = EpisodeStore(db, batch_size=self.flush_batch_size)
        try:
            while True:
                item = await asyncio.to_thread(self._queue.get)
                try:
                    if item is None:
                        return
                    await store.write_many(item)
                finally:
                    self._queue.task_done()
        finally:
            await db.shutdown()

    def _should_log_transition(self, done: bool) -> bool:
        return (self.n_calls % self.sample_every_n_steps == 0) or (self.log_episode_boundaries and done)

    def _enqueue_pending(self) -> None:
        if not self._pending:
            return
        batch = tuple(self._pending)
        self._pending.clear()
        try:
            self._queue.put_nowait(batch)
            return
        except Full:
            self._drop_oldest_batch()
        try:
            self._queue.put_nowait(batch)
        except Full:
            self._dropped_batches += 1
            self._dropped_transitions += len(batch)

    def _drop_oldest_batch(self) -> None:
        try:
            dropped = self._queue.get_nowait()
        except Empty:
            return
        try:
            if dropped is not None:
                self._dropped_batches += 1
                self._dropped_transitions += len(dropped)
        finally:
            self._queue.task_done()


def _ensure_observation_batch(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        return array.reshape(1, 1)
    if array.ndim == 1:
        return array.reshape(1, -1)
    return array.reshape(array.shape[0], -1)


def _ensure_action_batch(value: Any, env_count: int) -> np.ndarray:
    array = np.asarray(value)
    if env_count == 1:
        return array.reshape(1, -1)
    if array.ndim == 1 and array.shape[0] == env_count:
        return array.reshape(env_count, 1)
    return array.reshape(env_count, -1)


def _ensure_infos(value: Any, env_count: int) -> list[Mapping[str, Any]]:
    if isinstance(value, Sequence):
        infos = list(value)
    else:
        infos = []

    normalized: list[Mapping[str, Any]] = []
    for index in range(env_count):
        info = infos[index] if index < len(infos) else {}
        normalized.append(info if isinstance(info, Mapping) else {})
    return normalized


def _align_previous_obs(previous_obs: np.ndarray, env_count: int) -> np.ndarray:
    if previous_obs.shape[0] == env_count:
        return previous_obs
    if previous_obs.shape[0] == 1:
        return np.repeat(previous_obs, env_count, axis=0)
    raise RuntimeError(
        f"Previous observation batch has {previous_obs.shape[0]} envs, but current batch has {env_count}."
    )


def _single_observation_payload(value: Any) -> dict[str, list[float]]:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    return {"vector": [float(component) for component in vector]}


def _action_payload(value: Any, algorithm: Algorithm) -> dict[str, Any]:
    array = np.asarray(value)
    if algorithm == "dqn":
        return {"discrete": int(array.reshape(-1)[0])}
    return {"vector": [float(component) for component in array.astype(np.float32).reshape(-1)]}


def _projected_action_payload(info: Mapping[str, Any]) -> dict[str, Any] | None:
    action = info.get("projected_action", info.get("action"))
    if isinstance(action, Mapping):
        return dict(action)
    return None


def _info_payload(info: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "step": info.get("step"),
        "simulation_time": info.get("simulation_time"),
        "projected": bool(info.get("projected", False)),
        "blocked": bool(info.get("blocked", False)),
        "projection_reasons": list(info.get("projection_reasons", ())),
        "reward_components": dict(info.get("reward_components", {}))
        if isinstance(info.get("reward_components"), Mapping)
        else {},
        "snapshot": info.get("snapshot", {}),
    }


def _reward_components(info: Mapping[str, Any]) -> dict[str, float]:
    raw = info.get("reward_components", {})
    if not isinstance(raw, Mapping):
        return {}
    output: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float, str)):
            continue
        output[str(key)] = float(value)
    return output
