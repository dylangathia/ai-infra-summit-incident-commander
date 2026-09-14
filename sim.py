"""Smoke test: prove that p99 climbs on its own after a physical fault."""

import sys
from fleet.cluster import Cluster, WINDOW_S
from fleet.workload import Workload
from fleet import faults


def run(fault_key=None, warmup_s=60.0, post_s=120.0, quiet=False):
    cluster = Cluster(n_nodes=4)
    wl = Workload(seed=11)
    rows = []

    def step(duration):
        ticks = int(duration / 0.1)
        for _ in range(ticks):
            w = cluster.tick(wl.arrivals(cluster.t))
            if w:
                rows.append(w)

    step(warmup_s)
    baseline = [r.p99 * 1000 for r in rows[-6:]]
    info = faults.inject(cluster, wl, fault_key)
    step(post_s)
    after = [r.p99 * 1000 for r in rows[-6:]]

    if not quiet:
        print(f"\n=== {info['label']} ===")
        print(f"injected: {info['detail']}")
        print(f"{'t':>7} {'p99_ms':>8} {'p95_ms':>8} {'rps':>7} {'err':>7}  worst node")
        for r in rows[::4]:
            worst = max(r.nodes, key=lambda n: n["queue_depth"])
            flag = "  <-- fault" if r.t > warmup_s and r.t <= warmup_s + 1 else ""
            print(f"{r.t:7.0f} {r.p99*1000:8.0f} {r.p95*1000:8.0f} "
                  f"{r.throughput_rps:7.1f} {r.error_rate:7.2%}  "
                  f"{worst['node_id']} q={worst['queue_depth']:3d} "
                  f"kv={worst['kv_util']:.0%} clk={worst['clock_factor']:.2f}{flag}")
        print(f"baseline p99 {sum(baseline)/len(baseline):.0f}ms "
              f"-> post-fault p99 {sum(after)/len(after):.0f}ms")
    return sum(baseline) / len(baseline), sum(after) / len(after)


if __name__ == "__main__":
    key = sys.argv[1] if len(sys.argv) > 1 else None
    if key == "all":
        print(f"{'fault':>22} {'baseline':>10} {'after':>10} {'ratio':>8}")
        for f in faults.CATALOGUE:
            b, a = run(f.key, quiet=True)
            print(f"{f.key:>22} {b:9.0f}ms {a:9.0f}ms {a/max(b,1):7.1f}x")
    else:
        run(key)
