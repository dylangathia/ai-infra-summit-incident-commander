"""Executor and commander.

The commander is what makes this a closed loop rather than an alerting tool:
detect, investigate, validate, act, then *verify* and escalate itself if the
action did not work. The escalation carries the failed hypothesis back to the
investigator so the second attempt is informed rather than a retry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from agent.detector import Detector, Incident
from agent.investigator import Investigation, investigate
from agent.policy import Decision, Policy
from fleet.cluster import Cluster, WINDOW_S
from fleet.node import NodeState
from providers.base import Provider

MAX_ATTEMPTS = 3
# Verification needs patience. A correct remedy often makes p99 *worse* first:
# capping context or rate-limiting a tenant stops the bleeding immediately but
# the queue built during the incident still has to drain. Declaring failure at
# a fixed 30s punished correct diagnoses and sent the agent chasing ghosts.
MAX_VERIFY_WINDOWS = 30   # 150s of patience
GRACE_WINDOWS = 6         # backlog is allowed to get worse before it gets better
STALL_WINDOWS = 8         # no improvement for this long means the action failed


def execute(cluster: Cluster, action: str, args: Dict[str, Any]) -> str:
    """Apply a validated action. Nothing reaches here unvalidated."""
    if action == "drain_node":
        return cluster.drain(args["node_id"])
    if action == "rollback_node":
        return cluster.rollback(args["node_id"])
    if action == "rate_limit_tenant":
        return cluster.rate_limit_tenant(args["tenant"],
                                         args.get("admit_fraction", 0.25))
    if action == "cap_context":
        return cluster.cap_context(args.get("max_tokens", 4096))
    if action == "scale_up":
        return cluster.scale_up(int(args.get("count", 1)), warm=True)
    if action == "no_action":
        return "no action taken"
    return f"executor does not implement '{action}'"


@dataclass
class Attempt:
    n: int
    investigation: dict
    decision: dict
    executed: List[str] = field(default_factory=list)
    p99_before_ms: float = 0.0
    p99_after_ms: float = 0.0
    verified: bool = False
    verdict: str = ""


@dataclass
class Commander:
    cluster: Cluster
    provider: Provider
    policy: Policy = field(default_factory=Policy)
    detector: Detector = field(default_factory=Detector)
    on_event: Optional[Callable[[dict], None]] = None

    incident: Optional[Incident] = None
    attempts: List[Attempt] = field(default_factory=list)
    postmortem: str = ""

    def _emit(self, kind: str, **payload) -> None:
        if self.on_event:
            self.on_event({"t": round(self.cluster.t, 1), "kind": kind, **payload})

    # ---------- helpers ----------

    def _p99_ms(self, windows: int = 3) -> float:
        hist = list(self.cluster.history)[-windows:]
        if not hist:
            return 0.0
        return sum(w.p99 for w in hist) / len(hist) * 1000

    def _error_rate(self, windows: int = 3) -> float:
        hist = list(self.cluster.history)[-windows:]
        if not hist:
            return 0.0
        return sum(w.error_rate for w in hist) / len(hist)

    def _backlog(self) -> int:
        return sum(len(n.queue) for n in self.cluster.nodes)

    def _healthy(self) -> bool:
        return (self._p99_ms() <= self.cluster.slo_p99_ms
                and self._error_rate() <= self.detector.error_rate_threshold)

    def _verify(self, advance, incident_id: str, attempt: int):
        """Watch until recovery, or until improvement stalls.

        Returns (verified, p99_after_ms, note).
        """
        best = self._p99_ms(1)
        best_backlog = self._backlog()
        stall = 0
        for i in range(MAX_VERIFY_WINDOWS):
            advance(WINDOW_S)
            if self._healthy():
                return True, self._p99_ms(2), (
                    f"recovered after {(i + 1) * WINDOW_S:.0f}s of watching")
            cur = self._p99_ms(2)
            backlog = self._backlog()
            # p99 lags the fix: requests already queued carry their old wait
            # time to completion. A shrinking backlog means the remedy IS
            # working even while latency looks flat.
            latency_better = cur < best * 0.95
            backlog_better = backlog < best_backlog * 0.95
            if latency_better or backlog_better:
                best = min(best, cur)
                best_backlog = min(best_backlog, backlog)
                stall = 0
                self._emit("verify_progress", incident_id=incident_id,
                           attempt=attempt, p99_ms=round(cur),
                           backlog=backlog,
                           driver="latency" if latency_better else "backlog")
            elif i >= GRACE_WINDOWS:
                stall += 1
                if stall >= STALL_WINDOWS:
                    return False, cur, (
                        f"no improvement for {stall * WINDOW_S:.0f}s; "
                        f"p99 stuck near {cur:.0f}ms")
        return False, self._p99_ms(2), (
            f"still breaching after {MAX_VERIFY_WINDOWS * WINDOW_S:.0f}s")

    # ---------- the loop ----------

    def poll(self, advance: Callable[[float], None]) -> Optional[dict]:
        """Run one full incident lifecycle if the detector fires.

        ``advance(seconds)`` steps the simulation; the caller owns the clock so
        this works identically in a batch eval and behind a live websocket.
        """
        incident = self.detector.poll(self.cluster)
        if incident is None:
            return None

        self.incident = incident
        self.attempts = []
        self._emit("incident_opened", incident_id=incident.id,
                   trigger=incident.trigger, note=incident.opening_note)

        note = incident.opening_note
        for n in range(1, MAX_ATTEMPTS + 1):
            p99_before = self._p99_ms()

            inv: Investigation = investigate(self.cluster, self.provider,
                                             incident_note=note)
            self._emit("investigation", incident_id=incident.id, attempt=n,
                       root_cause=inv.root_cause, confidence=inv.confidence,
                       evidence=inv.evidence, ruled_out=inv.ruled_out,
                       tool_calls=inv.tool_calls_used, trace=inv.to_dict()["trace"])

            proposed = inv.recommended_action
            decision: Decision = self.policy.evaluate(
                self.cluster, proposed.get("action", "no_action"),
                proposed.get("args", {}))
            self._emit("policy", incident_id=incident.id, attempt=n,
                       proposed=proposed, decision=decision.to_dict())

            executed: List[str] = []
            if decision.verdict != "refused":
                if decision.companion:
                    executed.append(execute(self.cluster,
                                            decision.companion["action"],
                                            decision.companion["args"]))
                executed.append(execute(self.cluster, decision.action, decision.args))
            self._emit("executed", incident_id=incident.id, attempt=n,
                       results=executed)

            verified, p99_after, verify_note = self._verify(advance, incident.id, n)
            attempt = Attempt(
                n=n, investigation=inv.to_dict(), decision=decision.to_dict(),
                executed=executed, p99_before_ms=round(p99_before),
                p99_after_ms=round(p99_after), verified=verified,
                verdict=("recovered" if verified else "still breaching"),
            )
            self.attempts.append(attempt)
            self._emit("verification", incident_id=incident.id, attempt=n,
                       p99_before_ms=attempt.p99_before_ms,
                       p99_after_ms=attempt.p99_after_ms,
                       verified=verified, note=verify_note)

            if verified:
                incident.closed_t = self.cluster.t
                incident.resolution = inv.root_cause or "resolved"
                self.detector.mark_closed(self.cluster.t)
                break

            # Escalate: hand the failed hypothesis back rather than retrying blind.
            note = (
                f"{incident.opening_note}\n\n"
                f"ATTEMPT {n} FAILED. You previously concluded the cause was "
                f"'{inv.root_cause}' and the action taken was "
                f"{decision.action} {decision.args} "
                f"({'; '.join(executed) or 'nothing executed'}). "
                f"Policy verdict: {decision.verdict} — {decision.reason}. "
                f"p99 went from {p99_before:.0f}ms to {p99_after:.0f}ms and the "
                f"SLO is still breached ({verify_note}). "
                f"That hypothesis is either wrong or "
                f"incomplete. Re-investigate, and treat '{inv.root_cause}' as "
                f"ruled out unless you find evidence the remedy was simply "
                f"insufficient."
            )

        if self.incident and self.incident.open:
            self.incident.resolution = "unresolved — escalated to human"
            self._emit("escalated_to_human", incident_id=incident.id,
                       attempts=len(self.attempts))

        self.postmortem = self._write_postmortem()
        self._emit("postmortem", incident_id=incident.id, text=self.postmortem)
        return self.summary()

    # ---------- reporting ----------

    def _write_postmortem(self) -> str:
        inc = self.incident
        if inc is None:
            return ""
        lines = [
            f"# {inc.id} — {inc.resolution or 'open'}",
            "",
            f"**Detected** at t={inc.opened_t:.0f}s via {inc.trigger}, "
            f"{inc.detected_after_s:.0f}s after the condition began.",
            f"**p99 at open**: {inc.p99_at_open_ms:.0f}ms "
            f"(SLO {self.cluster.slo_p99_ms:.0f}ms). "
            f"**Error rate**: {inc.error_rate_at_open:.1%}.",
            "",
        ]
        for a in self.attempts:
            inv = a.investigation
            lines.append(f"## Attempt {a.n} — {a.verdict}")
            lines.append(f"- Root cause: **{inv.get('root_cause')}** "
                         f"(confidence {inv.get('confidence')}, "
                         f"{inv.get('tool_calls_used')} tool calls)")
            if inv.get("summary"):
                lines.append(f"- {inv['summary']}")
            for e in inv.get("evidence", [])[:4]:
                lines.append(f"  - evidence: {e}")
            for r in inv.get("ruled_out", [])[:3]:
                if isinstance(r, dict):
                    lines.append(f"  - ruled out {r.get('cause')}: {r.get('why')}")
            d = a.decision
            lines.append(f"- Policy: **{d['verdict']}** — {d['reason']}")
            if d.get("companion"):
                lines.append(f"  - companion action: {d['companion']}")
            lines.append(f"- Executed: {'; '.join(a.executed) or 'nothing'}")
            lines.append(f"- p99 {a.p99_before_ms:.0f}ms -> {a.p99_after_ms:.0f}ms")
            lines.append("")
        if inc.closed_t is not None:
            lines.append(f"**Resolved** at t={inc.closed_t:.0f}s "
                         f"({inc.closed_t - inc.opened_t:.0f}s after detection).")
        else:
            lines.append("**Not resolved.** Escalated to a human operator.")
        return "\n".join(lines)

    def summary(self) -> dict:
        inc = self.incident
        return {
            "incident_id": inc.id if inc else None,
            "trigger": inc.trigger if inc else None,
            "detected_after_s": inc.detected_after_s if inc else None,
            "resolution": inc.resolution if inc else None,
            "time_to_resolution_s": (
                round(inc.closed_t - inc.opened_t, 1)
                if inc and inc.closed_t else None),
            "attempts": [
                {"n": a.n, "root_cause": a.investigation.get("root_cause"),
                 "action": a.decision.get("action"),
                 "policy_verdict": a.decision.get("verdict"),
                 "p99_before_ms": a.p99_before_ms, "p99_after_ms": a.p99_after_ms,
                 "verified": a.verified}
                for a in self.attempts
            ],
            "postmortem": self.postmortem,
        }
