"""Run-level observability helpers for SpriteForge.

Provides lightweight, in-process metrics aggregation that can be surfaced
in CLI output and exported as JSON after a run.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from spriteforge.gates import GateVerdict


@dataclass
class RunMetricsCollector:
    """Collect run-level counters for retries and gate outcomes."""

    run_started_at_epoch: float = field(default_factory=time.time)
    run_finished_at_epoch: float | None = None

    _gate_pass_count: Counter[str] = field(default_factory=Counter)
    _gate_fail_count: Counter[str] = field(default_factory=Counter)
    _retry_count_by_tier: Counter[str] = field(default_factory=Counter)
    _retry_count_total: int = 0
    _last_failed_gate: str | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def record_gate_verdict(self, verdict: GateVerdict) -> None:
        """Record pass/fail counts for a gate verdict."""
        with self._lock:
            if verdict.passed:
                self._gate_pass_count[verdict.gate_name] += 1
            else:
                self._gate_fail_count[verdict.gate_name] += 1
                self._last_failed_gate = verdict.gate_name

    def record_retry(self, tier: str) -> None:
        """Record a retry event by escalation tier."""
        with self._lock:
            self._retry_count_total += 1
            self._retry_count_by_tier[tier] += 1

    def finish(self) -> None:
        """Mark the run as finished."""
        with self._lock:
            if self.run_finished_at_epoch is None:
                self.run_finished_at_epoch = time.time()

    def snapshot(self, call_tracker: Any | None = None) -> dict[str, Any]:
        """Build a JSON-serializable snapshot of collected metrics."""
        with self._lock:
            now = time.time()
            finished_at = self.run_finished_at_epoch
            duration_seconds = max(
                0.0,
                (finished_at if finished_at is not None else now)
                - self.run_started_at_epoch,
            )

            snapshot: dict[str, Any] = {
                "run_started_at_epoch": self.run_started_at_epoch,
                "run_finished_at_epoch": finished_at,
                "duration_seconds": duration_seconds,
                "retries_total": self._retry_count_total,
                "retries_by_tier": dict(self._retry_count_by_tier),
                "gate_pass_count": dict(self._gate_pass_count),
                "gate_fail_count": dict(self._gate_fail_count),
                "last_failed_gate": self._last_failed_gate,
            }

        if call_tracker is not None:
            snapshot["llm_calls_total"] = int(call_tracker.count)
            snapshot["token_usage"] = dict(call_tracker.token_usage)

        return snapshot


def write_run_summary(path: Path, payload: dict[str, Any]) -> None:
    """Write a run summary payload to disk as UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
