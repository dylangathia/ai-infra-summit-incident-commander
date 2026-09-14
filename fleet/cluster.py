"""The cluster: router, tick loop, and rolling telemetry.

The router is deliberately naive — least queue depth. It has no idea a node
is sick. That is what lets a single degraded node poison cluster p99: its
queue drains slowly, so it *looks* attractive right up until it doesn't.
"""

from __future__ import annotations

import statistics
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional

from .node import GPUNode, NodeConfig, NodeState, Request, TICK_S

WINDOW_S = 5.0


def pct(values: List[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]


@dataclass
class Window:
    t: float
    p50: float
    p95: float
    p99: float
    ttft_p95: float
    throughput_rps: float
    error_rate: float
    completed: int
    nodes: List[dict]
    per_tenant: Dict[str, dict]

    def to_dict(self) -> dict:
        return {
            "t": round(self.t, 1),
            "p50_ms": round(self.p50 * 1000),
            "p95_ms": round(self.p95 * 1000),
            "p99_ms": round(self.p99 * 1000),
            "ttft_p95_ms": round(self.ttft_p95 * 1000),
            "throughput_rps": round(self.throughput_rps, 2),
            "error_rate": round(self.error_rate, 4),
            "completed": self.completed,
            "nodes": self.nodes,
            "per_tenant": self.per_tenant,
        }


@dataclass
class Cluster:
    n_nodes: int = 4
    slo_p99_ms: float = 4000.0
    t: float = 0.0
    nodes: List[GPUNode] = field(default_factory=list)
    history: Deque[Window] = field(default_factory=lambda: deque(maxlen=720))
    events: List[dict] = field(default_factory=list)
    _window_reqs: List[Request] = field(default_factory=list)
    _window_shed: int = 0
    _acc: float = 0.0
    # agent-controlled ingress policy
    tenant_limits: Dict[str, float] = field(default_factory=dict)  # tenant -> admit fraction
    max_context_tokens: Optional[int] = None
    context_caps_applied: int = 0
    policy_rejected: int = 0
    seed: int = 3

    def __post_init__(self):
        import random as _r
        self._rng = _r.Random(self.seed)
        if not self.nodes:
            self.nodes = [
                GPUNode(id=f"gpu-{i:02d}", cfg=NodeConfig())
                for i in range(self.n_nodes)
            ]

    # ---------- routing ----------

    def route(self, req: Request) -> Optional[GPUNode]:
        candidates = [n for n in self.nodes if n.accepting]
        if not candidates:
            return None
        # Least outstanding requests. A stalled node has no batch, so it looks
        # like the *least* loaded box in the fleet and attracts traffic it
        # cannot serve. This is the trap, and it is how real balancers behave.
        return min(candidates, key=lambda n: len(n.queue) + len(n.batch))

    def log_event(self, kind: str, detail: str, actor: str = "system") -> None:
        self.events.append({
            "t": round(self.t, 1), "kind": kind,
            "detail": detail, "actor": actor,
        })

    # ---------- tick ----------

    def tick(self, incoming: List[Request]) -> Optional[Window]:
        for req in incoming:
            # --- ingress policy (agent-controlled) ---
            if req.tenant in self.tenant_limits:
                if self._rng.random() > self.tenant_limits[req.tenant]:
                    self.policy_rejected += 1
                    req.failed = True
                    continue
            if self.max_context_tokens and req.prompt_tokens > self.max_context_tokens:
                req.prompt_tokens = self.max_context_tokens
                self.context_caps_applied += 1

            node = self.route(req)
            if node is None or not node.enqueue(req, self.t):
                self._window_shed += 1
            # a shed request is still a completed unit of work for error rate

        for node in self.nodes:
            node.tick(self.t)

        self.t += TICK_S
        self._acc += TICK_S

        if self._acc >= WINDOW_S:
            self._acc = 0.0
            return self._close_window()
        return None

    def _close_window(self) -> Window:
        completed: List[Request] = []
        node_snaps = []
        shed_total = self._window_shed
        for node in self.nodes:
            node_snaps.append(node.snapshot(WINDOW_S))
            shed_total += node.shed
            completed.extend(node.reset_window())

        lat = [r.latency for r in completed if r.latency is not None]
        ttft = [r.ttft for r in completed if r.ttft is not None]

        per_tenant: Dict[str, dict] = {}
        for r in completed:
            b = per_tenant.setdefault(
                r.tenant, {"count": 0, "prompt_tokens": 0, "lat": []})
            b["count"] += 1
            b["prompt_tokens"] += r.prompt_tokens
            b["lat"].append(r.latency)
        for name, b in per_tenant.items():
            lats = b.pop("lat")
            b["p99_ms"] = round(pct(lats, 99) * 1000)
            b["avg_prompt_tokens"] = round(b["prompt_tokens"] / max(1, b["count"]))

        total_units = len(completed) + shed_total
        w = Window(
            t=self.t,
            p50=pct(lat, 50), p95=pct(lat, 95), p99=pct(lat, 99),
            ttft_p95=pct(ttft, 95),
            throughput_rps=len(completed) / WINDOW_S,
            error_rate=shed_total / max(1, total_units),
            completed=len(completed),
            nodes=node_snaps,
            per_tenant=per_tenant,
        )
        self._window_shed = 0
        self.history.append(w)
        return w

    # ---------- actions the agent may take ----------

    def drain(self, node_id: str) -> str:
        for n in self.nodes:
            if n.id == node_id:
                n.state = NodeState.DRAINING
                for r in n.queue:
                    tgt = self.route(r)
                    if tgt:
                        tgt.enqueue(r, self.t)
                n.queue.clear()
                self.log_event("action", f"drained {node_id}", "agent")
                return f"{node_id} draining; queue redistributed"
        return f"no such node {node_id}"

    def restore(self, node_id: str) -> str:
        for n in self.nodes:
            if n.id == node_id:
                n.state = NodeState.HEALTHY
                n.clock_factor = 1.0
                self.log_event("action", f"restored {node_id}", "agent")
                return f"{node_id} healthy"
        return f"no such node {node_id}"

    def scale_up(self, count: int = 1, warm: bool = True,
                 actor: str = "agent") -> str:
        for i in range(count):
            node = GPUNode(id=f"gpu-{len(self.nodes):02d}")
            if not warm:
                node.state = NodeState.LOADING
                node.load_remaining_s = 45.0
            self.nodes.append(node)
        kind = "action" if actor == "agent" else "scale"
        self.log_event(kind, f"scaled up by {count}", actor)
        return f"added {count} node(s)"

    def rate_limit_tenant(self, tenant: str, admit_fraction: float = 0.25) -> str:
        self.tenant_limits[tenant] = max(0.0, min(1.0, admit_fraction))
        self.log_event("action", f"rate-limited {tenant} to {admit_fraction:.0%}", "agent")
        return f"{tenant} admitting {admit_fraction:.0%} of requests"

    def cap_context(self, max_tokens: int = 4096) -> str:
        self.max_context_tokens = max_tokens
        self.log_event("action", f"context capped at {max_tokens} tokens", "agent")
        return f"prompts truncated to {max_tokens} tokens"

    def rollback(self, node_id: str) -> str:
        for n in self.nodes:
            if n.id == node_id:
                n.quantization = "fp8"
                n.model_version = "llama-3.1-70b@v4"
                self.log_event("action", f"rolled back {node_id}", "agent")
                return f"{node_id} rolled back to fp8 @v4"
        return f"no such node {node_id}"
