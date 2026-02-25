"""Tests for run-level observability helpers."""

from __future__ import annotations

import json

from spriteforge.gates import GateVerdict
from spriteforge.observability import RunMetricsCollector, write_run_summary


def test_run_metrics_collector_snapshot_includes_gate_and_retry_counts() -> None:
    collector = RunMetricsCollector()

    collector.record_gate_verdict(
        GateVerdict(
            gate_name="gate_0",
            passed=False,
            confidence=0.2,
            feedback="failed",
        )
    )
    collector.record_gate_verdict(
        GateVerdict(
            gate_name="gate_0",
            passed=True,
            confidence=0.9,
            feedback="passed",
        )
    )
    collector.record_retry("guided")
    collector.record_retry("guided")
    collector.finish()

    snapshot = collector.snapshot(call_tracker=None)

    assert snapshot["retries_total"] == 2
    assert snapshot["retries_by_tier"]["guided"] == 2
    assert snapshot["gate_fail_count"]["gate_0"] == 1
    assert snapshot["gate_pass_count"]["gate_0"] == 1
    assert snapshot["last_failed_gate"] == "gate_0"
    assert snapshot["run_finished_at_epoch"] is not None


def test_write_run_summary_writes_valid_json(tmp_path) -> None:  # type: ignore[no-untyped-def]
    summary_path = tmp_path / "summary" / "run.json"
    payload = {
        "character": "Theron",
        "metrics": {"retries_total": 3, "llm_calls_total": 42},
    }

    write_run_summary(summary_path, payload)

    assert summary_path.exists()
    loaded = json.loads(summary_path.read_text())
    assert loaded == payload
