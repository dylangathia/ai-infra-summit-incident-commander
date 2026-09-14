"""Does the correct action actually fix the incident?

If a scenario cannot be remediated, the agent-on vs agent-off comparison
shows nothing for it and the business-value claim is empty. Run this before
building the agent, not after.
"""

import random
from fleet.cluster import Cluster
from fleet.workload import Workload
from fleet import faults


def sick_node(cluster):
    """Whichever node the fault actually landed on."""
    for n in cluster.nodes:
        if n.clock_factor < 0.9 or n.quantization == "fp16" or n.state.value == "loading":
            return n.id
    return cluster.nodes[0].id


def drain_and_replace(c, w):
    """Draining removes capacity. On a fleet with no headroom that makes the
    incident worse, so a replacement must come up alongside the drain."""
    nid = sick_node(c)
    c.scale_up(1, warm=True)
    return c.drain(nid)


REMEDY = {
    "thermal_throttle":  drain_and_replace,
    "kv_exhaustion":     lambda c, w: c.cap_context(2048),
    "noisy_neighbour":   lambda c, w: c.rate_limit_tenant("initech-batch", 0.05),
    "version_skew":      lambda c, w: c.rollback(sick_node(c)),
    "cold_start_stall":  drain_and_replace,
    "traffic_surge":     lambda c, w: c.scale_up(8, warm=True),
}


def run(key, seed=11, warmup=60.0, degrade=60.0, recover=150.0):
    cluster = Cluster(n_nodes=4, seed=seed)
    wl = Workload(seed=seed)
    rows = []

    def step(d):
        for _ in range(int(d / 0.1)):
            w = cluster.tick(wl.arrivals(cluster.t))
            if w:
                rows.append(w)

    step(warmup)
    base = sum(r.p99 for r in rows[-6:]) / 6 * 1000
    faults.inject(cluster, wl, key, random.Random(seed))
    step(degrade)
    during = sum(r.p99 for r in rows[-6:]) / 6 * 1000
    err_during = sum(r.error_rate for r in rows[-6:]) / 6
    REMEDY[key](cluster, wl)
    step(recover)
    after = sum(r.p99 for r in rows[-6:]) / 6 * 1000
    err_after = sum(r.error_rate for r in rows[-6:]) / 6
    return base, during, after, err_during, err_after


if __name__ == "__main__":
    SLO = 4000
    print(f"{'scenario':>20} {'base':>8} {'during':>9} {'after':>9} "
          f"{'err→':>7} {'verdict':>10}")
    for f in faults.CATALOGUE:
        b, d, a, ed, ea = run(f.key)
        fixed = a < SLO and a < d * 0.75
        # cold start is an error-rate incident, not a latency one
        if f.key == "cold_start_stall":
            fixed = ea <= ed
        print(f"{f.key:>20} {b:7.0f}ms {d:8.0f}ms {a:8.0f}ms "
              f"{ea:6.1%} {'RECOVERED' if fixed else 'STILL SICK':>10}")
