"""Full incident lifecycle, no model required (echo provider)."""
import random, sys
from fleet.cluster import Cluster
from fleet.workload import Workload
from fleet import faults
from providers.base import get_provider
from agent.commander import Commander

def run(key, seed=11, provider_name="echo", verbose=True):
    c=Cluster(n_nodes=4,seed=seed); w=Workload(seed=seed)
    def advance(d):
        for _ in range(int(d/0.1)): c.tick(w.arrivals(c.t))
    advance(60)
    truth=faults.inject(c,w,key,random.Random(seed))
    advance(25)
    events=[]
    cmd=Commander(cluster=c, provider=get_provider(provider_name),
                  on_event=lambda e: events.append(e))
    res=None
    for _ in range(8):           # marginal breaches take longer to trip
        res=cmd.poll(advance)
        if res is not None: break
        advance(20)
    if verbose:
        print(f"\n{'='*66}\nGROUND TRUTH: {truth['label']}\n{'='*66}")
        for e in events:
            if e["kind"]=="incident_opened": print(f"[{e['t']:6.1f}] OPEN   {e['note'][:100]}")
            elif e["kind"]=="investigation": print(f"[{e['t']:6.1f}] DIAG   {e['root_cause']} (conf {e['confidence']}, {e['tool_calls']} tools)")
            elif e["kind"]=="policy": print(f"[{e['t']:6.1f}] POLICY {e['decision']['verdict']}: {e['decision']['reason'][:90]}")
            elif e["kind"]=="executed": print(f"[{e['t']:6.1f}] ACT    {'; '.join(e['results'])}")
            elif e["kind"]=="verification": print(f"[{e['t']:6.1f}] VERIFY {e['p99_before_ms']}ms -> {e['p99_after_ms']}ms  {'RECOVERED' if e['verified'] else 'FAILED'} ({e['note']})")
            elif e["kind"]=="escalated_to_human": print(f"[{e['t']:6.1f}] ESCALATE after {e['attempts']} attempts")
    return res

if __name__=="__main__":
    pn = sys.argv[1] if len(sys.argv)>1 else "echo"
    for f in faults.CATALOGUE:
        r=run(f.key, provider_name=pn)
        if r is None:
            print("  -> detector never fired"); continue
        print(f"  -> {r['resolution']} in {r['time_to_resolution_s']}s, {len(r['attempts'])} attempt(s)")
