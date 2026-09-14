"""Can the tools actually discriminate the six scenarios? Agent-free check."""
import random, json
from fleet.cluster import Cluster
from fleet.workload import Workload
from fleet import faults
from telemetry import tools

def observe(key, seed=11):
    c=Cluster(n_nodes=4,seed=seed); w=Workload(seed=seed)
    def step(d):
        for _ in range(int(d/0.1)): c.tick(w.arrivals(c.t))
    step(60); faults.inject(c,w,key,random.Random(seed)); step(75)
    summary=tools.call_tool(c,"get_cluster_summary")
    talkers=tools.call_tool(c,"get_top_talkers")
    sig={}
    for m in ("clock_factor","kv_util","queue_depth","batch_size"):
        r=tools.call_tool(c,"compare_nodes",{"metric":m})
        sig[m]=(r["dispersion_verdict"].split(" —")[0], r["outliers"])
    ev=tools.call_tool(c,"get_recent_events")
    return c,summary,talkers,sig,ev

for f in faults.CATALOGUE:
    c,s,tk,sig,ev=observe(f.key)
    print(f"\n=== {f.key} ===")
    print(f"  p99={s['current']['p99_ms']}ms vs baseline {s['earlier_baseline']['p99_ms']}ms  "
          f"rps={s['current']['throughput_rps']} (was {s['earlier_baseline']['throughput_rps']})  "
          f"err={s['current']['error_rate']:.2%}")
    for m,(v,o) in sig.items():
        print(f"  {m:14} {v:12} outliers={o}")
    for t in tk["tenants"]:
        print(f"  tenant {t['tenant']:16} reqs x{t["request_rate_change"]} "
              f"promptsize x{t["avg_prompt_size_change"]} avg={t['avg_prompt_tokens']}")
    print(f"  events: {[e['kind'] for e in ev['events']]}")
