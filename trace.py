"""Print one full investigation so you can see what the model actually saw."""
import json, sys
from diagnose import scenario
from providers.base import get_provider
from agent.investigator import investigate

key = sys.argv[1] if len(sys.argv) > 1 else "noisy_neighbour"
prov = get_provider(sys.argv[2] if len(sys.argv) > 2 else "openai")
c = scenario(key)
inv = investigate(c, prov)

print(f"=== GROUND TRUTH: {key} ===\n")
for s in inv.to_dict()["trace"]:
    if s["kind"] == "tool":
        r = s["result"] or {}
        brief = {k: r[k] for k in
                 ("dispersion_verdict","outliers","throughput_direction",
                  "slo_breached","tenants","events","lines") if k in r}
        print(f"TOOL  {s['tool']}({s['args']})")
        print(f"      {json.dumps(brief, default=str)[:400]}\n")
    elif s["kind"] == "thought" and s["content"]:
        print(f"SAID  {s['content'][:300]}\n")
print("--- conclusion ---")
print(json.dumps({k: inv.to_dict()[k] for k in
                  ("root_cause","confidence","evidence","ruled_out",
                   "recommended_action","error")}, indent=2, default=str))