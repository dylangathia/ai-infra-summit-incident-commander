"""Agent off vs agent on, identical scenarios, identical seeds.

This is the business-value artifact. Every number here is measured from the
simulator, not asserted. The control arm is not "do nothing forever" — it is
a human on-call who pages, investigates and acts after a realistic delay,
because beating a strawman proves nothing.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from agent.commander import Commander
from agent.detector import Detector
from agent.policy import Policy
from fleet import faults
from fleet.cluster import Cluster, WINDOW_S
from fleet.workload import Workload
from providers.base import get_provider

# Cost basis. Stated openly so a judge can substitute their own number.
GPU_HOURLY_USD = 2.60          # one accelerator node, on-demand, list-ish
SLO_P99_MS = 4000.0
HUMAN_RESPONSE_S = 300.0       # page -> engineer at keyboard, acting
WARMUP_S = 60.0
HORIZON_S = 900.0              # total observation window per run


@dataclass
class RunResult:
    scenario: str
    seed: int
    arm: str
    detected_after_s: Optional[float] = None
    resolved_after_s: Optional[float] = None
    degraded_s: float = 0.0
    slo_violating_requests: int = 0
    failed_requests: int = 0
    wasted_gpu_hours: float = 0.0
    root_cause: Optional[str] = None
    correct_action: bool = False
    attempts: int = 0


def _make(seed: int):
    cluster = Cluster(n_nodes=4, seed=seed)
    cluster.slo_p99_ms = SLO_P99_MS
    return cluster, Workload(seed=seed)


def _degradation_stats(cluster: Cluster, from_t: float) -> Dict[str, float]:
    """Degraded time, requests served in violation, and capacity wasted."""
    degraded_s = 0.0
    violating = 0
    failed = 0
    wasted_node_seconds = 0.0
    for w in cluster.history:
        if w.t < from_t:
            continue
        breached = w.p99 * 1000 > SLO_P99_MS
        if breached:
            degraded_s += WINDOW_S
            violating += int(w.completed)
        err = min(0.99, w.error_rate)
        failed += int(round(w.completed * err / max(1e-9, 1 - err)))
        # Accelerator time spent serving traffic that missed its objective.
        # Throughput shortfall is the wrong measure here: queues absorb the
        # backlog, so the fleet keeps completing requests at close to normal
        # rate while every one of them is late. The cost is that the capacity
        # was paid for and produced responses that breached the SLO.
        if breached:
            wasted_node_seconds += len(w.nodes) * WINDOW_S
    return {
        "degraded_s": degraded_s,
        "slo_violating_requests": violating,
        "failed_requests": failed,
        "wasted_gpu_hours": wasted_node_seconds / 3600.0,
    }


def run_agent(scenario: str, seed: int, provider_name: str) -> RunResult:
    cluster, wl = _make(seed)

    def advance(d: float):
        for _ in range(int(d / 0.1)):
            cluster.tick(wl.arrivals(cluster.t))

    advance(WARMUP_S)
    truth = faults.inject(cluster, wl, scenario, random.Random(seed))
    fault_t = cluster.t

    cmd = Commander(cluster=cluster, provider=get_provider(provider_name),
                    policy=Policy(), detector=Detector())
    summary = None
    while cluster.t < WARMUP_S + HORIZON_S and summary is None:
        summary = cmd.poll(advance)
        if summary is None:
            advance(10.0)
    if cluster.t < WARMUP_S + HORIZON_S:
        advance(min(120.0, WARMUP_S + HORIZON_S - cluster.t))

    stats = _degradation_stats(cluster, fault_t)
    res = RunResult(scenario=scenario, seed=seed, arm="agent", **stats)
    if summary:
        res.detected_after_s = summary["detected_after_s"]
        res.resolved_after_s = summary["time_to_resolution_s"]
        res.attempts = len(summary["attempts"])
        first = summary["attempts"][0] if summary["attempts"] else {}
        res.root_cause = first.get("root_cause")
        res.correct_action = any(a.get("verified") for a in summary["attempts"])
    return res


def run_baseline(scenario: str, seed: int, delay_s: float = HUMAN_RESPONSE_S) -> RunResult:
    """Control arm: a competent human who arrives after a realistic delay."""
    from efficacy import REMEDY

    cluster, wl = _make(seed)

    def advance(d: float):
        for _ in range(int(d / 0.1)):
            cluster.tick(wl.arrivals(cluster.t))

    advance(WARMUP_S)
    faults.inject(cluster, wl, scenario, random.Random(seed))
    fault_t = cluster.t

    detector = Detector()
    detected_at = None
    while cluster.t < fault_t + delay_s:
        advance(WINDOW_S)
        if detected_at is None and detector.poll(cluster):
            detected_at = cluster.t

    REMEDY[scenario](cluster, wl)          # the human does the right thing
    advance(min(240.0, WARMUP_S + HORIZON_S - cluster.t))

    stats = _degradation_stats(cluster, fault_t)
    res = RunResult(scenario=scenario, seed=seed, arm="human_baseline", **stats)
    res.detected_after_s = (detected_at - fault_t) if detected_at else None
    res.resolved_after_s = delay_s
    res.correct_action = True
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default="echo")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    seeds = [11 + 7 * i for i in range(args.seeds)]
    rows: List[RunResult] = []

    for f in faults.CATALOGUE:
        for seed in seeds:
            rows.append(run_baseline(f.key, seed))
            rows.append(run_agent(f.key, seed, args.provider))

    def arm(name: str, scenario: Optional[str] = None) -> List[RunResult]:
        return [r for r in rows if r.arm == name
                and (scenario is None or r.scenario == scenario)]

    print(f"\nprovider={args.provider}  seeds={seeds}  "
          f"GPU cost basis=${GPU_HOURLY_USD}/node-hour\n")
    print(f"{'scenario':>20} {'MTTD':>14} {'degraded min':>22} "
          f"{'wasted GPU-h':>20} {'correct':>8}")
    print(f"{'':>20} {'human / agent':>14} {'human -> agent':>22} "
          f"{'human -> agent':>20}")
    print("-" * 92)

    for f in faults.CATALOGUE:
        h, a = arm("human_baseline", f.key), arm("agent", f.key)
        hd = statistics.fmean(r.degraded_s for r in h) / 60
        ad = statistics.fmean(r.degraded_s for r in a) / 60
        hg = statistics.fmean(r.wasted_gpu_hours for r in h)
        ag = statistics.fmean(r.wasted_gpu_hours for r in a)
        hm = [r.detected_after_s for r in h if r.detected_after_s is not None]
        am = [r.detected_after_s for r in a if r.detected_after_s is not None]
        correct = sum(1 for r in a if r.correct_action)
        print(f"{f.key:>20} "
              f"{(statistics.fmean(hm) if hm else 0):5.0f}s /{(statistics.fmean(am) if am else 0):5.0f}s "
              f"{hd:9.1f} -> {ad:<9.1f} "
              f"{hg:8.2f} -> {ag:<8.2f} "
              f"{correct:>4}/{len(a)}")

    h, a = arm("human_baseline"), arm("agent")
    hd = sum(r.degraded_s for r in h) / 60
    ad = sum(r.degraded_s for r in a) / 60
    hg = sum(r.wasted_gpu_hours for r in h)
    ag = sum(r.wasted_gpu_hours for r in a)
    hv = sum(r.slo_violating_requests for r in h)
    av = sum(r.slo_violating_requests for r in a)
    correct = sum(1 for r in a if r.correct_action)

    print("-" * 92)
    print(f"\nAcross {len(a)} incidents ({len(faults.CATALOGUE)} scenarios x "
          f"{len(seeds)} seeds):\n")
    print(f"  Degraded serving time      {hd:8.1f} min  ->  {ad:8.1f} min   "
          f"({(1 - ad / hd) * 100 if hd else 0:.0f}% reduction)")
    print(f"  Requests served over SLO   {hv:8d}      ->  {av:8d}       "
          f"({(1 - av / hv) * 100 if hv else 0:.0f}% reduction)")
    print(f"  Degraded GPU-hours         {hg:8.2f}      ->  {ag:8.2f}       "
          f"(accelerator time spent serving traffic that missed SLO)")
    print(f"  Correct resolution          {'n/a (given)':>12}  ->  {correct}/{len(a)}")

    per_inc_min = (hd - ad) / max(1, len(a))
    per_inc_gpuh = (hg - ag) / max(1, len(a))
    print(f"\n  Per incident: {per_inc_min:.1f} fewer minutes degraded, "
          f"{per_inc_gpuh:.3f} GPU-hours recovered on a 4-node fleet.")

    # Extrapolation, labelled as such. The simulated fleet is 4 nodes; the
    # audience runs hundreds.
    FLEET, PER_MONTH = 200, 15
    scale = FLEET / 4.0
    monthly = per_inc_gpuh * scale * PER_MONTH * GPU_HOURLY_USD
    print(f"\n  Extrapolated (NOT simulated): a {FLEET}-node fleet at "
          f"{PER_MONTH} incidents/month\n"
          f"  would recover ~{per_inc_gpuh * scale * PER_MONTH:.0f} GPU-hours/month "
          f"= ${monthly:,.0f}/month at ${GPU_HOURLY_USD}/node-hour,\n"
          f"  before counting the {(1 - av / hv) * 100 if hv else 0:.0f}% "
          f"reduction in requests served over SLO.\n")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump([r.__dict__ for r in rows], fh, indent=2)
        print(f"  wrote {args.json}")


if __name__ == "__main__":
    main()