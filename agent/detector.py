"""Detection. Deliberately no model in the hot path.

Two independent triggers, because they catch different failures. Latency
alone misses a node that is stalled and timing requests out — that incident
barely moves p99 at all and shows up almost entirely as errors.
"""

from __future__ import annotations

import itertools
import statistics
from dataclasses import dataclass, field
from typing import List, Optional

from fleet.cluster import Cluster, WINDOW_S

_ids = itertools.count(1)


@dataclass
class Incident:
    id: str
    opened_t: float
    trigger: str                 # "latency_slo" | "error_rate"
    opening_note: str
    p99_at_open_ms: float
    error_rate_at_open: float
    detected_after_s: float      # time from first bad window to firing
    closed_t: Optional[float] = None
    resolution: Optional[str] = None
    attempts: List[dict] = field(default_factory=list)

    @property
    def open(self) -> bool:
        return self.closed_t is None


@dataclass
class Detector:
    breach_windows_required: int = 2      # sustained, not a single spike
    error_rate_threshold: float = 0.02
    error_windows_required: int = 2
    # Gray failures never cross an absolute threshold. A node serving at 60%
    # of fleet speed pushes p99 to ~1.7x baseline while staying under SLO,
    # and an absolute-threshold detector never fires at all.
    regression_multiple: float = 1.5
    regression_windows_required: int = 3
    cooldown_s: float = 30.0
    _last_close_t: float = -1e9

    def poll(self, cluster: Cluster) -> Optional[Incident]:
        hist = list(cluster.history)
        if len(hist) < self.breach_windows_required:
            return None
        if cluster.t - self._last_close_t < self.cooldown_s:
            return None

        latest = hist[-1]

        lat_breach = [w.p99 * 1000 > cluster.slo_p99_ms
                      for w in hist[-self.breach_windows_required:]]
        err_breach = [w.error_rate > self.error_rate_threshold
                      for w in hist[-self.error_windows_required:]]

        reference = self._reference_p99(hist)
        reg_breach = []
        if reference:
            reg_breach = [w.p99 * 1000 > reference * self.regression_multiple
                          for w in hist[-self.regression_windows_required:]]

        trigger = None
        if all(lat_breach):
            trigger = "latency_slo"
        elif all(err_breach):
            trigger = "error_rate"
        elif reg_breach and len(reg_breach) >= self.regression_windows_required \
                and all(reg_breach):
            trigger = "latency_regression"
        if trigger is None:
            return None

        # how long the condition had been true before we fired
        run = 0
        for w in reversed(hist):
            if trigger == "latency_slo":
                bad = w.p99 * 1000 > cluster.slo_p99_ms
            elif trigger == "error_rate":
                bad = w.error_rate > self.error_rate_threshold
            else:
                bad = w.p99 * 1000 > (reference or 0) * self.regression_multiple
            if bad:
                run += 1
            else:
                break

        note = (
            f"Trigger: {trigger}. "
            f"p99 {latest.p99 * 1000:.0f}ms against an SLO of {cluster.slo_p99_ms:.0f}ms, "
            f"error rate {latest.error_rate:.1%}, "
            f"throughput {latest.throughput_rps:.1f} rps. "
            f"Condition has held for {run * WINDOW_S:.0f}s."
            + (f" Baseline p99 for this fleet is around {reference:.0f}ms."
               if reference else "")
        )
        return Incident(
            id=f"INC-{next(_ids):03d}",
            opened_t=cluster.t,
            trigger=trigger,
            opening_note=note,
            p99_at_open_ms=latest.p99 * 1000,
            error_rate_at_open=latest.error_rate,
            detected_after_s=run * WINDOW_S,
        )

    @staticmethod
    def _reference_p99(hist) -> Optional[float]:
        """Median p99 over the oldest third of history: a stand-in for normal."""
        if len(hist) < 9:
            return None
        older = hist[: max(3, len(hist) // 3)]
        return statistics.median(w.p99 * 1000 for w in older)

    def mark_closed(self, t: float) -> None:
        self._last_close_t = t
