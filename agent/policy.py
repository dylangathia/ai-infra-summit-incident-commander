"""Policy: the deterministic gate between the model and the fleet.

Every rule here exists because the measured behaviour of the simulator
demanded it, not because a policy layer sounded like good practice:

- Draining the throttled node without replacing it moved p99 from 5250ms to
  9350ms. Removing a quarter of the capacity from a fleet at ~65% utilisation
  overloads what remains. So a drain that breaches the headroom floor is
  rewritten to bring up a replacement alongside it.
- Scaling to match demand exactly left a permanent backlog: throughput
  matched arrivals, errors went to zero, and p99 sat at 9000ms indefinitely
  with 800 requests standing in queue. So scale-ups are floored at a surplus.

The model proposes. This decides. It can approve, rewrite, or refuse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fleet.cluster import Cluster
from fleet.node import NodeState

ALLOWED = {"drain_node", "rollback_node", "rate_limit_tenant",
           "cap_context", "scale_up", "no_action"}

DESTRUCTIVE = {"drain_node", "rollback_node"}


@dataclass
class Decision:
    verdict: str                       # "approved" | "rewritten" | "refused"
    action: str
    args: Dict[str, Any] = field(default_factory=dict)
    companion: Optional[Dict[str, Any]] = None   # e.g. scale_up beside a drain
    reason: str = ""
    requires_approval: bool = False

    def to_dict(self) -> dict:
        return {
            "verdict": self.verdict, "action": self.action, "args": self.args,
            "companion": self.companion, "reason": self.reason,
            "requires_approval": self.requires_approval,
        }


@dataclass
class Policy:
    min_healthy_nodes: int = 2
    headroom_floor: float = 0.20       # keep 20% of serving capacity spare
    max_scale_step: int = 12
    auto_approve_destructive: bool = True   # demo mode; UI can flip this

    # ---------- capacity model ----------

    @staticmethod
    def _serving_capacity(cluster: Cluster, excluding: Optional[str] = None) -> float:
        """Effective nodes, weighting each by how much work it can actually do."""
        total = 0.0
        for n in cluster.nodes:
            if n.id == excluding or n.state in (NodeState.OFFLINE, NodeState.DRAINING):
                continue
            if n.state == NodeState.LOADING:
                continue
            total += n.clock_factor * n.quant_speed_factor
        return total

    @staticmethod
    def _demand(cluster: Cluster) -> float:
        """Rough effective-node demand implied by recent throughput."""
        hist = list(cluster.history)[-6:]
        if not hist:
            return 0.0
        rps = sum(w.throughput_rps for w in hist) / len(hist)
        queued = sum(len(n.queue) for n in cluster.nodes)
        # calibrated against the simulator: ~16 rps of sustainable work per node
        return rps / 16.0 + queued / 200.0

    # ---------- evaluation ----------

    def evaluate(self, cluster: Cluster, action: str,
                 args: Optional[dict] = None) -> Decision:
        args = dict(args or {})

        if action not in ALLOWED:
            return Decision("refused", "no_action", {},
                            reason=f"'{action}' is not on the allow-list")

        if action == "no_action":
            return Decision("approved", action, args, reason="no change requested")

        if action in ("drain_node", "rollback_node"):
            node_id = args.get("node_id")
            node = next((n for n in cluster.nodes if n.id == node_id), None)
            if node is None:
                return Decision("refused", "no_action", {},
                                reason=f"node '{node_id}' does not exist")

            healthy_after = sum(
                1 for n in cluster.nodes
                if n.id != node_id and n.state == NodeState.HEALTHY)
            if healthy_after < self.min_healthy_nodes:
                return Decision(
                    "refused", "no_action", {},
                    reason=(f"would leave {healthy_after} healthy node(s), "
                            f"below the floor of {self.min_healthy_nodes}"))

            if action == "rollback_node":
                # A rollback only means anything for a node that is serving a
                # different build. A node hung loading weights is not serving
                # at all — rolling it back leaves it in rotation, still
                # attracting traffic it cannot answer.
                if node.state == NodeState.LOADING:
                    return Decision(
                        "rewritten", "drain_node", {"node_id": node_id},
                        reason=(f"{node_id} is still loading weights, not "
                                f"serving a bad build — a rollback would leave "
                                f"it in rotation; draining instead"),
                        requires_approval=not self.auto_approve_destructive)
                if (node.quantization == "fp8"
                        and node.model_version == "llama-3.1-70b@v4"):
                    return Decision(
                        "refused", "no_action", {},
                        reason=(f"{node_id} is already on the baseline build "
                                f"(fp8 @v4); there is nothing to roll back"))

            if action == "drain_node":
                remaining = self._serving_capacity(cluster, excluding=node_id)
                demand = self._demand(cluster)
                if remaining < demand * (1 + self.headroom_floor):
                    return Decision(
                        "rewritten", "drain_node", {"node_id": node_id},
                        companion={"action": "scale_up", "args": {"count": 1}},
                        reason=(f"draining alone leaves {remaining:.1f} effective "
                                f"nodes against demand of {demand:.1f}; bringing up "
                                f"a replacement alongside the drain"),
                        requires_approval=not self.auto_approve_destructive)

            return Decision("approved", action, args,
                            reason="capacity floor satisfied",
                            requires_approval=not self.auto_approve_destructive)

        if action == "scale_up":
            requested = int(args.get("count", 1) or 1)
            demand = self._demand(cluster)
            capacity = self._serving_capacity(cluster)
            # matching demand leaves a standing queue forever; require surplus
            needed = max(0.0, demand * (1 + self.headroom_floor) - capacity)
            floor = max(1, int(needed + 0.999))
            count = min(self.max_scale_step, max(requested, floor))
            if count != requested:
                return Decision(
                    "rewritten", "scale_up", {"count": count},
                    reason=(f"{requested} node(s) would only match demand "
                            f"({demand:.1f} effective nodes against {capacity:.1f} "
                            f"serving); scaling to {count} to leave surplus and "
                            f"actually drain the backlog"))
            return Decision("approved", action, {"count": count},
                            reason="scale-up leaves surplus capacity")

        if action == "rate_limit_tenant":
            tenant = args.get("tenant")
            frac = float(args.get("admit_fraction", 0.25) or 0.25)
            if not tenant:
                return Decision("refused", "no_action", {},
                                reason="no tenant specified")
            if frac < 0.02:
                return Decision("rewritten", action,
                                {"tenant": tenant, "admit_fraction": 0.02},
                                reason="floor of 2% admission; a full block is "
                                       "an outage for that tenant, not a remedy")
            return Decision("approved", action,
                            {"tenant": tenant, "admit_fraction": frac},
                            reason="within limits")

        if action == "cap_context":
            cap = int(args.get("max_tokens", 4096) or 4096)
            if cap < 1024:
                return Decision("rewritten", action, {"max_tokens": 1024},
                                reason="capping below 1024 tokens breaks ordinary "
                                       "requests, not just the pathological ones")
            return Decision("approved", action, {"max_tokens": cap},
                            reason="within limits")

        return Decision("refused", "no_action", {}, reason="unhandled action")