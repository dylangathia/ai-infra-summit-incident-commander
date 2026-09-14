"""Request arrival process.

Three tenants with genuinely different shapes, because "who is sending
what" is the signal that separates a noisy neighbour from a surge.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Dict, List

from .node import Request, TICK_S


@dataclass
class TenantProfile:
    name: str
    rps: float
    prompt_mu: float        # lognormal mu over prompt tokens
    prompt_sigma: float
    out_mu: float
    out_sigma: float
    rps_multiplier: float = 1.0   # faults scale this


def default_tenants() -> Dict[str, TenantProfile]:
    return {
        # chat traffic: short prompts, short answers, high volume
        "acme-chat": TenantProfile("acme-chat", rps=42.0, prompt_mu=6.2,
                                   prompt_sigma=0.55, out_mu=4.6, out_sigma=0.5),
        # RAG traffic: long prompts, moderate answers
        "globex-rag": TenantProfile("globex-rag", rps=15.0, prompt_mu=7.9,
                                    prompt_sigma=0.6, out_mu=5.0, out_sigma=0.45),
        # batch summarisation: very long prompts, low volume
        "initech-batch": TenantProfile("initech-batch", rps=4.0, prompt_mu=8.7,
                                       prompt_sigma=0.5, out_mu=5.6, out_sigma=0.4),
    }


@dataclass
class Workload:
    tenants: Dict[str, TenantProfile] = field(default_factory=default_tenants)
    global_multiplier: float = 1.0
    seed: int = 7
    _next_id: int = 0
    _rng: random.Random = field(init=False)

    def __post_init__(self):
        self._rng = random.Random(self.seed)

    def _sample(self, mu: float, sigma: float, lo: int, hi: int) -> int:
        return int(min(hi, max(lo, self._rng.lognormvariate(mu, sigma))))

    def arrivals(self, t: float) -> List[Request]:
        out: List[Request] = []
        for prof in self.tenants.values():
            rate = prof.rps * prof.rps_multiplier * self.global_multiplier
            expected = rate * TICK_S
            n = 0
            # Poisson via Knuth for small lambda
            limit = self._rng.random()
            p = 1.0
            import math
            target = math.exp(-expected)
            while p > target and n < 50:
                p *= self._rng.random()
                n += 1
            n = max(0, n - 1)
            for _ in range(n):
                self._next_id += 1
                out.append(Request(
                    id=self._next_id,
                    tenant=prof.name,
                    prompt_tokens=self._sample(prof.prompt_mu, prof.prompt_sigma, 32, 32_000),
                    max_output_tokens=self._sample(prof.out_mu, prof.out_sigma, 16, 2_000),
                    arrival_t=t,
                ))
        return out
