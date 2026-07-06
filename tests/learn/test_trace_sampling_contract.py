from __future__ import annotations

import argparse
import unittest

from src.learn.train_joint_torch import JointTraceRecorder, TraceLoggingConfig, trace_logging_config_from_args


class TraceSamplingContractTest(unittest.TestCase):
    def test_trace_logging_config_loads_sample_interval(self) -> None:
        config = trace_logging_config_from_args(
            argparse.Namespace(disable_trace_logging=False),
            flush_interval=5_000,
            sample_interval=100,
            environment_id="joint_rolling_5pl_288",
            team_id="joint_from_scratch",
        )

        self.assertTrue(config.enabled)
        self.assertEqual(config.flush_interval, 5_000)
        self.assertEqual(config.sample_interval, 100)

    def test_sample_interval_one_preserves_every_step_logging(self) -> None:
        recorder = JointTraceRecorder(buffer=object(), config=TraceLoggingConfig(sample_interval=1))  # type: ignore[arg-type]

        self.assertTrue(recorder.should_record_step(0))
        self.assertTrue(recorder.should_record_step(1))
        self.assertTrue(recorder.should_record_step(99))

    def test_positive_sample_interval_records_only_matching_steps(self) -> None:
        recorder = JointTraceRecorder(buffer=object(), config=TraceLoggingConfig(sample_interval=100))  # type: ignore[arg-type]

        self.assertTrue(recorder.should_record_step(0))
        self.assertFalse(recorder.should_record_step(1))
        self.assertFalse(recorder.should_record_step(99))
        self.assertTrue(recorder.should_record_step(100))

    def test_disabled_trace_recorder_never_records(self) -> None:
        recorder = JointTraceRecorder.disabled(config=TraceLoggingConfig(enabled=False, sample_interval=100))

        self.assertFalse(recorder.should_record_step(0))
        self.assertFalse(recorder.should_record_step(100))


if __name__ == "__main__":
    unittest.main()
