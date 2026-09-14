"""A single GPU inference node.

The point of this module: latency is never assigned, it is *computed*.
Every metric a node reports is a consequence of its queue, its batch, and
its KV cache occupancy. Faults change physical parameters (clock speed,
cache capacity, model version) and the symptoms emerge on their own.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

TICK_S = 0.1        # simulation resolution
REQUEST_TIMEOUT_S = 30.0   # client gives up and we count it as an error


class NodeState(str, Enum):
    HEALTHY = "healthy"
    LOADING = "loading"      # cold start: weights not resident yet
    DRAINING = "draining"    # finishing in-flight work, no new admissions
    OFFLINE = "offline"


@dataclass
class Request:
    id: int
    tenant: str
    prompt_tokens: int
    max_output_tokens: int
    arrival_t: float

    admitted_t: Optional[float] = None
    first_token_t: Optional[float] = None
    finished_t: Optional[float] = None
    generated: int = 0
    node_id: Optional[str] = None
    preemptions: int = 0
    failed: bool = False

    # prefill accounting
    prefill_remaining: float = 0.0

    @property
    def kv_tokens(self) -> int:
        return self.prompt_tokens + self.generated

    @property
    def latency(self) -> Optional[float]:
        if self.finished_t is None:
            return None
        return self.finished_t - self.arrival_t

    @property
    def ttft(self) -> Optional[float]:
        if self.first_token_t is None:
            return None
        return self.first_token_t - self.arrival_t


@dataclass
class NodeConfig:
    peak_decode_tps: float = 3200.0      # aggregate decode tokens/sec at saturation
    batch_halfpoint: float = 5.0         # batch size at which we reach ~50% of peak
    prefill_tps: float = 55_000.0        # prefill is compute-bound and fast
    kv_capacity_tokens: int = 160_000
    max_batch: int = 24
    queue_limit: int = 240               # beyond this we shed load (503)


@dataclass
class GPUNode:
    id: str
    cfg: NodeConfig = field(default_factory=NodeConfig)

    state: NodeState = NodeState.HEALTHY
    clock_factor: float = 1.0            # 1.0 = nominal; thermal throttle drops this
    temp_c: float = 61.0
    model_version: str = "llama-3.1-70b@v4"
    quantization: str = "fp8"
    load_remaining_s: float = 0.0        # for LOADING state

    queue: List[Request] = field(default_factory=list)
    batch: List[Request] = field(default_factory=list)

    # rolling counters, reset by the metrics collector each window
    completed: List[Request] = field(default_factory=list)
    shed: int = 0
    preemptions_window: int = 0
    tokens_this_window: int = 0
    events: List[tuple] = field(default_factory=list)  # (t, kind, detail)

    # ---------- capability ----------

    @property
    def quant_speed_factor(self) -> float:
        # fp8 is the fleet norm. A node redeployed at fp16 is materially slower.
        return {"fp8": 1.0, "int8": 1.08, "fp16": 0.62}.get(self.quantization, 1.0)

    @property
    def kv_used(self) -> int:
        return sum(r.kv_tokens for r in self.batch)

    @property
    def kv_util(self) -> float:
        return self.kv_used / max(1, self.cfg.kv_capacity_tokens)

    @property
    def accepting(self) -> bool:
        return self.state in (NodeState.HEALTHY, NodeState.LOADING)

    def decode_tps(self) -> float:
        """Aggregate decode throughput, saturating with batch size."""
        b = len(self.batch)
        if b == 0:
            return 0.0
        saturation = b / (b + self.cfg.batch_halfpoint)
        return (
            self.cfg.peak_decode_tps
            * saturation
            * self.clock_factor
            * self.quant_speed_factor
        )

    # ---------- admission ----------

    def enqueue(self, req: Request, t: float) -> bool:
        if not self.accepting or len(self.queue) >= self.cfg.queue_limit:
            self.shed += 1
            req.failed = True
            return False
        req.node_id = self.id
        self.queue.append(req)
        return True

    def _can_admit(self, req: Request) -> bool:
        if len(self.batch) >= self.cfg.max_batch:
            return False
        # reserve headroom for the tokens this request will generate
        projected = self.kv_used + req.prompt_tokens + min(req.max_output_tokens, 256)
        return projected <= self.cfg.kv_capacity_tokens

    def _admit_from_queue(self, t: float) -> None:
        while self.queue and self._can_admit(self.queue[0]):
            req = self.queue.pop(0)
            req.admitted_t = t
            req.prefill_remaining = req.prompt_tokens
            self.batch.append(req)

    def _expire_queue(self, t: float) -> None:
        """Clients do not wait forever. A node that never serves its queue
        bleeds errors rather than silently absorbing traffic."""
        if not self.queue:
            return
        keep = []
        for r in self.queue:
            if t - r.arrival_t > REQUEST_TIMEOUT_S:
                r.failed = True
                self.shed += 1
            else:
                keep.append(r)
        self.queue = keep

    def _preempt_if_saturated(self, t: float) -> None:
        """Under cache pressure, evict the largest in-flight request.

        Its generated tokens are lost and it must re-prefill. This is the
        recompute storm that makes KV exhaustion so much worse than it looks.
        """
        guard = 0
        while self.kv_util > 0.97 and len(self.batch) > 1 and guard < 8:
            victim = max(self.batch, key=lambda r: r.kv_tokens)
            self.batch.remove(victim)
            victim.generated = 0
            victim.first_token_t = None
            victim.preemptions += 1
            victim.prefill_remaining = victim.prompt_tokens
            self.queue.insert(0, victim)
            self.preemptions_window += 1
            guard += 1

    # ---------- tick ----------

    def tick(self, t: float) -> None:
        if self.state == NodeState.OFFLINE:
            return

        # expire before anything else: even a loading node bleeds timeouts
        self._expire_queue(t)

        if self.state == NodeState.LOADING:
            # A stalled cold start keeps accepting traffic it cannot serve.
            self.load_remaining_s -= TICK_S
            if self.load_remaining_s <= 0:
                self.state = NodeState.HEALTHY
                self.events.append((t, "node_ready", self.id))
            return

        if self.state != NodeState.DRAINING:
            self._admit_from_queue(t)
        self._preempt_if_saturated(t)

        if not self.batch:
            self._cool(t)
            return

        # --- prefill phase for requests that still need it ---
        prefill_budget = self.cfg.prefill_tps * self.clock_factor * TICK_S
        for req in list(self.batch):
            if req.prefill_remaining > 0 and prefill_budget > 0:
                spend = min(req.prefill_remaining, prefill_budget)
                req.prefill_remaining -= spend
                prefill_budget -= spend

        # --- decode phase, shared across the batch ---
        decoding = [r for r in self.batch if r.prefill_remaining <= 0]
        if decoding:
            total_tokens = self.decode_tps() * TICK_S
            per_req = total_tokens / len(decoding)
            carry = 0.0
            for req in decoding:
                whole = int(per_req + carry)
                carry = (per_req + carry) - whole
                if whole <= 0:
                    continue
                if req.first_token_t is None:
                    req.first_token_t = t
                req.generated += whole
                self.tokens_this_window += whole
                if req.generated >= req.max_output_tokens:
                    req.finished_t = t
                    self.batch.remove(req)
                    self.completed.append(req)

        self._heat(t)

    # ---------- thermal model ----------

    def _heat(self, t: float) -> None:
        load = len(self.batch) / max(1, self.cfg.max_batch)
        target = 58 + 34 * load
        self.temp_c += (target - self.temp_c) * 0.02

    def _cool(self, t: float) -> None:
        self.temp_c += (55 - self.temp_c) * 0.02

    # ---------- telemetry ----------

    def snapshot(self, window_s: float) -> dict:
        return {
            "node_id": self.id,
            "state": self.state.value,
            "queue_depth": len(self.queue),
            "batch_size": len(self.batch),
            "kv_util": round(self.kv_util, 4),
            "kv_used_tokens": self.kv_used,
            "tokens_per_s": round(self.tokens_this_window / window_s, 1),
            "temp_c": round(self.temp_c, 1),
            "clock_factor": round(self.clock_factor, 3),
            "model_version": self.model_version,
            "quantization": self.quantization,
            "preemptions": self.preemptions_window,
            "shed": self.shed,
        }

    def reset_window(self) -> List[Request]:
        done, self.completed = self.completed, []
        self.tokens_this_window = 0
        self.preemptions_window = 0
        self.shed = 0
        return done
