"""Fault injection.

Each fault changes a *physical* parameter. None of them touch latency
directly — the symptoms are downstream consequences. Scenario 6 is not a
fault at all, and telling it apart from scenario 1 is the whole thesis.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Optional

from .cluster import Cluster
from .node import NodeState
from .workload import Workload


@dataclass
class Fault:
    key: str
    label: str
    correct_action: str
    apply: Callable[[Cluster, Workload, random.Random], str]


def _pick_node(cluster: Cluster, rng: random.Random):
    healthy = [n for n in cluster.nodes if n.state == NodeState.HEALTHY]
    return rng.choice(healthy) if healthy else None


def thermal_throttle(cluster, workload, rng) -> str:
    n = _pick_node(cluster, rng)
    if not n:
        return "no healthy node"
    n.clock_factor = rng.uniform(0.34, 0.48)
    n.temp_c = rng.uniform(89, 95)
    return f"{n.id} throttled to {n.clock_factor:.2f}x"


def kv_exhaustion(cluster, workload, rng) -> str:
    # RAG tenant starts sending far longer contexts
    p = workload.tenants["globex-rag"]
    # Same request rate, far longer contexts. This is what separates it from
    # a surge: demand in requests is flat, demand in KV tokens explodes.
    p.prompt_mu = 9.9
    p.prompt_sigma = 0.3
    return "globex-rag context length inflated (rate unchanged)"


def noisy_neighbour(cluster, workload, rng) -> str:
    p = workload.tenants["initech-batch"]
    p.rps_multiplier = 11.0
    return "initech-batch flooding"


def version_skew(cluster, workload, rng) -> str:
    n = _pick_node(cluster, rng)
    if not n:
        return "no healthy node"
    n.quantization = "fp16"
    n.model_version = "llama-3.1-70b@v5"
    cluster.log_event("deploy", f"{n.id} redeployed to v5 (fp16)", "ci")
    return f"{n.id} skewed to fp16/v5"


def cold_start_stall(cluster, workload, rng) -> str:
    cluster.scale_up(1, warm=False, actor="autoscaler")
    node = cluster.nodes[-1]
    node.load_remaining_s = 10_000.0  # hangs indefinitely
    cluster.log_event("scale", f"{node.id} provisioned, loading weights", "autoscaler")
    return f"{node.id} stalled in LOADING while accepting traffic"


def traffic_surge(cluster, workload, rng) -> str:
    workload.global_multiplier = rng.uniform(2.1, 2.8)
    return f"uniform surge x{workload.global_multiplier:.1f}"


CATALOGUE = [
    Fault("thermal_throttle", "Thermal throttling on one node",
          "drain_node", thermal_throttle),
    Fault("kv_exhaustion", "KV cache exhaustion from long contexts",
          "cap_context", kv_exhaustion),
    Fault("noisy_neighbour", "Noisy neighbour flooding the fleet",
          "rate_limit_tenant", noisy_neighbour),
    Fault("version_skew", "Version/quantization skew after deploy",
          "rollback_node", version_skew),
    Fault("cold_start_stall", "Cold-start stall taking live traffic",
          "remove_from_rotation", cold_start_stall),
    Fault("traffic_surge", "Genuine traffic surge (NOT a fault)",
          "scale_up", traffic_surge),
]

BY_KEY = {f.key: f for f in CATALOGUE}


def inject(cluster: Cluster, workload: Workload, key: Optional[str] = None,
           rng: Optional[random.Random] = None) -> dict:
    """Inject a named fault, or a uniformly random one if key is None."""
    rng = rng or random.Random()
    fault = BY_KEY[key] if key else rng.choice(CATALOGUE)
    detail = fault.apply(cluster, workload, rng)
    cluster.log_event("fault_injected", detail, "chaos")
    return {"key": fault.key, "label": fault.label,
            "detail": detail, "correct_action": fault.correct_action}
