"""Filesystem-backed model registry for checkpoint paths, metrics, and active models."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4


Algorithm = Literal["ppo", "dqn"]
ModelStatus = Literal["candidate", "active", "archived"]


@dataclass(frozen=True, slots=True)
class ModelRecord:
    model_id: str
    algorithm: Algorithm
    path: str
    status: ModelStatus
    created_at: str
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def score(self) -> float:
        if "mean_reward" in self.metrics:
            return float(self.metrics["mean_reward"])
        if "service_level" in self.metrics:
            return float(self.metrics["service_level"])
        return float("-inf")


class ModelRegistry:
    """Registry API used by orchestration to load explicit active model paths."""

    def __init__(self, registry_dir: Path | str = Path("models/registry")) -> None:
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.registry_dir / "models.jsonl"
        self.active_path = self.registry_dir / "active_models.json"

    def register(
        self,
        *,
        algorithm: Algorithm,
        path: Path | str,
        agent_role: str | None = None,
        metrics: dict[str, float] | None = None,
        metadata: dict[str, Any] | None = None,
        status: ModelStatus = "candidate",
    ) -> ModelRecord:
        checkpoint_path = Path(path)
        if checkpoint_path.suffix not in {".zip", ".pt"}:
            raise ValueError("registered model path must point to a .zip or .pt checkpoint.")

        record_metadata = dict(metadata or {})
        if agent_role is not None:
            record_metadata["agent_role"] = agent_role

        record = ModelRecord(
            model_id=str(uuid4()),
            algorithm=algorithm,
            path=str(checkpoint_path),
            status=status,
            created_at=datetime.now(timezone.utc).isoformat(),
            metrics=metrics or {},
            metadata=record_metadata,
        )
        self._append(record)
        if status == "active":
            self.set_active(algorithm, checkpoint_path, agent_role=agent_role)
        return record

    def records(
        self,
        *,
        algorithm: Algorithm | None = None,
        agent_role: str | None = None,
        status: ModelStatus | None = None,
    ) -> list[ModelRecord]:
        if not self.index_path.exists():
            return []
        output: list[ModelRecord] = []
        for line in self.index_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = ModelRecord(**json.loads(line))
            role_matches = agent_role is None or record.metadata.get("agent_role") == agent_role
            status_matches = status is None or record.status == status
            if (algorithm is None or record.algorithm == algorithm) and role_matches and status_matches:
                output.append(record)
        return output

    def candidate_records(
        self,
        *,
        algorithm: Algorithm | None = None,
        agent_role: str | None = None,
    ) -> list[ModelRecord]:
        """Return candidate records for explicit review flows only."""

        return self.records(algorithm=algorithm, agent_role=agent_role, status="candidate")

    def best(
        self,
        algorithm: Algorithm,
        *,
        agent_role: str | None = None,
        status: ModelStatus = "active",
    ) -> ModelRecord:
        candidates = [
            record
            for record in self.records(algorithm=algorithm, agent_role=agent_role, status=status)
            if Path(record.path).exists()
        ]
        if not candidates and agent_role is not None:
            candidates = [
                record
                for record in self.records(algorithm=algorithm, status=status)
                if Path(record.path).exists()
            ]
        if not candidates:
            raise FileNotFoundError(f"No registered {status} {algorithm} model checkpoints found.")
        return max(candidates, key=lambda record: record.score)

    def best_path(
        self,
        algorithm: Algorithm,
        *,
        agent_role: str | None = None,
        status: ModelStatus = "active",
    ) -> Path:
        return Path(self.best(algorithm, agent_role=agent_role, status=status).path)

    def set_active(self, algorithm: Algorithm, path: Path | str, *, agent_role: str | None = None) -> None:
        checkpoint_path = Path(path)
        if checkpoint_path.suffix not in {".zip", ".pt"}:
            raise ValueError("active model path must point to a .zip or .pt checkpoint.")
        active = self._read_active()
        active[self._active_key(algorithm, agent_role=agent_role)] = str(checkpoint_path)
        self.active_path.write_text(json.dumps(active, indent=2), encoding="utf-8")

    def active_path_for(self, algorithm: Algorithm, *, agent_role: str | None = None) -> Path:
        active = self._read_active()
        value = active.get(self._active_key(algorithm, agent_role=agent_role))
        if value is None and agent_role is not None:
            value = active.get(algorithm)
        if value is None:
            role_suffix = f" for role {agent_role}" if agent_role is not None else ""
            raise FileNotFoundError(f"No explicit active {algorithm} model configured{role_suffix}.")
        path = Path(value)
        if not path.exists():
            raise FileNotFoundError(f"Active {algorithm} model does not exist: {path}")
        return path

    def _append(self, record: ModelRecord) -> None:
        with self.index_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(record), separators=(",", ":")) + "\n")

    def _read_active(self) -> dict[str, str]:
        if not self.active_path.exists():
            return {}
        return json.loads(self.active_path.read_text(encoding="utf-8"))

    @staticmethod
    def _active_key(algorithm: Algorithm, *, agent_role: str | None = None) -> str:
        if agent_role is None:
            return algorithm
        return f"{algorithm}:{agent_role}"
