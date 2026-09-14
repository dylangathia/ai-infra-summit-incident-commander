"""Live engine.

The fleet ticks on a wall clock so a judge can watch degradation happen. The
commander runs in a worker thread and its ``advance`` callback simply sleeps,
which means the fleet keeps degrading while the agent is thinking — the same
way a real incident does not pause for the on-call engineer.
"""

from __future__ import annotations

import asyncio
import random
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from agent.commander import Commander
from agent.detector import Detector
from agent.policy import Policy
from fleet import faults
from fleet.cluster import Cluster
from fleet.node import TICK_S
from fleet.workload import Workload
from providers.base import get_provider

SPEED = 8.0          # sim seconds per wall-clock second
FRAME_S = 0.1        # wall-clock cadence of the tick loop


class Engine:
    def __init__(self, provider_name: str = "echo", n_nodes: int = 4):
        self.provider_name = provider_name
        self.n_nodes = n_nodes
        self.speed = SPEED
        self.events: Deque[dict] = deque(maxlen=400)
        self.windows: Deque[dict] = deque(maxlen=240)
        self._lock = threading.Lock()
        self._agent_thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.agent_enabled = True
        self.active_fault: Optional[str] = None
        self.reset()

    # ---------- lifecycle ----------

    def reset(self) -> None:
        self._stop.set()
        if self._agent_thread and self._agent_thread.is_alive():
            self._agent_thread.join(timeout=2.0)
        self._stop.clear()
        seed = random.randrange(1, 10_000)
        self.cluster = Cluster(n_nodes=self.n_nodes, seed=seed)
        self.workload = Workload(seed=seed)
        self.events.clear()
        self.windows.clear()
        self.active_fault = None
        self.commander = Commander(
            cluster=self.cluster, provider=get_provider(self.provider_name),
            policy=Policy(), detector=Detector(), on_event=self._record)
        self._start_agent()
        self._record("reset", seed=seed, nodes=self.n_nodes)

    def _record(self, kind: str, **payload) -> None:
        if isinstance(kind, dict):            # called as on_event({...})
            event = kind
        else:
            event = {"t": round(self.cluster.t, 1), "kind": kind, **payload}
        with self._lock:
            self.events.append(event)

    # ---------- the agent thread ----------

    def _start_agent(self) -> None:
        def advance(seconds: float) -> None:
            """Wait in wall-clock terms; the tick loop moves the fleet."""
            deadline = time.monotonic() + seconds / self.speed
            while time.monotonic() < deadline:
                if self._stop.is_set():
                    raise SystemExit
                time.sleep(0.02)

        def loop() -> None:
            while not self._stop.is_set():
                try:
                    if self.agent_enabled:
                        self.commander.poll(advance)
                    time.sleep(0.5)
                except SystemExit:
                    return
                except Exception as exc:            # never kill the demo
                    self._record("agent_error", error=str(exc))
                    time.sleep(2.0)

        self._agent_thread = threading.Thread(target=loop, daemon=True)
        self._agent_thread.start()

    # ---------- the fleet tick loop ----------

    async def run(self) -> None:
        while True:
            ticks = max(1, int(self.speed * FRAME_S / TICK_S))
            for _ in range(ticks):
                w = self.cluster.tick(self.workload.arrivals(self.cluster.t))
                if w:
                    with self._lock:
                        self.windows.append(w.to_dict())
            await asyncio.sleep(FRAME_S)

    # ---------- control ----------

    def inject(self, key: Optional[str] = None) -> dict:
        info = faults.inject(self.cluster, self.workload, key)
        self.active_fault = info["key"]
        # deliberately NOT added to self.events: the UI must not reveal ground
        # truth before the agent has reached its own conclusion
        return info

    def set_agent(self, enabled: bool) -> None:
        self.agent_enabled = enabled
        self._record("agent_toggled", enabled=enabled)

    # ---------- state for the client ----------

    def state(self) -> Dict[str, Any]:
        with self._lock:
            windows = list(self.windows)[-90:]
            events = list(self.events)[-60:]
        latest = windows[-1] if windows else None
        return {
            "t": round(self.cluster.t, 1),
            "speed": self.speed,
            "slo_p99_ms": self.cluster.slo_p99_ms,
            "agent_enabled": self.agent_enabled,
            "latest": latest,
            "p99_series": [w["p99_ms"] for w in windows],
            "rps_series": [w["throughput_rps"] for w in windows],
            "err_series": [w["error_rate"] for w in windows],
            "nodes": latest["nodes"] if latest else [],
            "tenants": latest["per_tenant"] if latest else {},
            "events": events,
            "catalogue": [{"key": f.key, "label": f.label}
                          for f in faults.CATALOGUE],
        }