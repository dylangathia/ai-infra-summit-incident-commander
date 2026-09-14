"""End-to-end: does the agent loop reach the right root cause?"""
import random, sys
from fleet.cluster import Cluster
from fleet.workload import Workload
from fleet import faults
from providers.base import get_provider
from agent.investigator import investigate

EXPECTED_ACTION = {
    "thermal_throttle":"drain_node","kv_exhaustion":"cap_context",
    "noisy_neighbour":"rate_limit_tenant","version_skew":"rollback_node",
    "cold_start_stall":"drain_node","traffic_surge":"scale_up",
}

def scenario(key, seed=11):
    c=Cluster(n_nodes=4,seed=seed); w=Workload(seed=seed)
    def step(d):
        for _ in range(int(d/0.1)): c.tick(w.arrivals(c.t))
    step(60); faults.inject(c,w,key,random.Random(seed)); step(75)
    return c

if __name__=="__main__":
    prov=get_provider(sys.argv[1] if len(sys.argv)>1 else "echo")
    print(f"provider={prov.name}\n")
    ok=0
    for f in faults.CATALOGUE:
        c=scenario(f.key)
        inv=investigate(c,prov)
        act=inv.recommended_action["action"]
        hit=act==EXPECTED_ACTION[f.key]
        ok+=hit
        print(f"{f.key:>20} -> {str(inv.root_cause):>22} "
              f"action={act:<20} tools={inv.tool_calls_used} "
              f"{'OK' if hit else 'WRONG'}")
        if inv.error: print(f"{'':>20}    error: {inv.error}")
    print(f"\n{ok}/{len(faults.CATALOGUE)} correct actions")