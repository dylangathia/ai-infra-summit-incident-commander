"""A scripted provider that implements the same interface as a real model.

Two uses:

1. The agent loop can be exercised with no API key and no cost, which matters
   when you are debugging the loop rather than the reasoning.
2. It is the **heuristic baseline** for the eval. Hand-written rules were
   given the answer key; the model was not. If the model matches the baseline
   on the six known scenarios and beats it on compound ones, that is the
   evidence that reasoning is doing work a decision tree cannot.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from .base import ModelResponse, ToolCall

PLAN = [
    ("get_cluster_summary", {}),
    ("compare_nodes", {"metric": "tokens_per_s"}),
    ("compare_nodes", {"metric": "clock_factor"}),
    ("compare_nodes", {"metric": "batch_size"}),
    ("get_top_talkers", {}),
    ("get_recent_events", {}),
]


class EchoProvider:
    name = "echo"

    # ---------- interface ----------

    def complete(self, system, messages, tools=None, max_tokens=1500,
                 temperature=0.0) -> ModelResponse:
        results = self._results_so_far(messages)
        step = len(results)
        if step < len(PLAN):
            name, args = PLAN[step]
            return ModelResponse(
                text="",
                tool_calls=[ToolCall(id=f"call_{step}", name=name, arguments=args)],
            )
        return ModelResponse(text=json.dumps(self._conclude(results)))

    def assistant_message(self, response: ModelResponse) -> Dict[str, Any]:
        return {"role": "assistant", "content": [
            {"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments}
            for c in response.tool_calls
        ] or [{"type": "text", "text": response.text}]}

    def tool_result_message(self, call: ToolCall, result: Any) -> Dict[str, Any]:
        return {"role": "user", "content": [{
            "type": "tool_result", "tool_use_id": call.id,
            "content": json.dumps(result, default=str),
        }]}

    # ---------- internals ----------

    @staticmethod
    def _results_so_far(messages) -> List[dict]:
        out = []
        for m in messages:
            if m.get("role") != "user" or not isinstance(m.get("content"), list):
                continue
            for block in m["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    try:
                        out.append(json.loads(block["content"]))
                    except (json.JSONDecodeError, KeyError, TypeError):
                        out.append({})
        return out

    def _conclude(self, r: List[dict]) -> dict:
        summary, tps, clock, batch, talkers, events = (r + [{}] * 6)[:6]

        tps_out = tps.get("outliers") or []
        clock_out = clock.get("outliers") or []
        batch_rows = {n["node_id"]: n["value"] for n in batch.get("nodes", [])}
        idle = [nid for nid, v in batch_rows.items() if v < 0.5]

        cur = summary.get("current", {})
        base = summary.get("earlier_baseline", {})
        rps_now = cur.get("throughput_rps", 0)
        rps_before = base.get("throughput_rps", rps_now) or rps_now
        tenants = talkers.get("tenants", [])

        def ratio(t, key):
            v = t.get(key)
            return v if isinstance(v, (int, float)) else 1.0

        rate_spike = max(tenants, key=lambda t: max(ratio(t, "request_rate_change"),
                                                   ratio(t, "share_change")),
                         default=None)
        size_spike = max(tenants, key=lambda t: ratio(t, "avg_prompt_size_change"),
                         default=None)

        def out(cause, summary_text, action, args, evidence, ruled=None):
            return {
                "root_cause": cause, "summary": summary_text, "confidence": 0.7,
                "evidence": evidence, "ruled_out": ruled or [],
                "recommended_action": {"action": action, "args": args},
            }

        if idle:
            nid = idle[0]
            return out("cold_start_stall",
                       f"{nid} is accepting traffic while not serving any.",
                       "drain_node", {"node_id": nid},
                       [f"{nid} batch size ~0 with a standing queue"])

        if tps_out:
            nid = tps_out[0]
            if nid in clock_out:
                return out("thermal_throttle", f"{nid} is thermally throttled.",
                           "drain_node", {"node_id": nid},
                           [f"{nid} clock and throughput both below fleet median"])
            return out("version_skew", f"{nid} is serving a different build.",
                       "rollback_node", {"node_id": nid},
                       [f"{nid} throughput low but clock nominal"],
                       [{"cause": "thermal_throttle", "why": "clock_factor uniform"}])

        if rps_now > rps_before * 1.15:
            return out("traffic_surge", "Demand rose across all tenants.",
                       "scale_up", {"count": 8},
                       [f"throughput {rps_before} -> {rps_now} rps, no node anomalous"],
                       [{"cause": "node_fault", "why": "all node metrics uniform"}])

        if rate_spike and max(ratio(rate_spike, "request_rate_change"),
                              ratio(rate_spike, "share_change")) > 2.5:
            return out("noisy_neighbour",
                       f"{rate_spike['tenant']} is flooding the fleet.",
                       "rate_limit_tenant",
                       {"tenant": rate_spike["tenant"], "admit_fraction": 0.05},
                       [f"{rate_spike['tenant']} request rate "
                        f"x{ratio(rate_spike, 'request_rate_change')}, "
                        f"fleet share x{ratio(rate_spike, 'share_change')}"])

        if size_spike and ratio(size_spike, "avg_prompt_size_change") > 2:
            return out("kv_exhaustion",
                       f"{size_spike['tenant']} prompt sizes exploded.",
                       "cap_context", {"max_tokens": 2048},
                       [f"{size_spike['tenant']} avg prompt "
                        f"x{ratio(size_spike, 'avg_prompt_size_change')} "
                        f"at flat request rate"],
                       [{"cause": "traffic_surge", "why": "request rate did not rise"}])

        return out("undetermined", "No clear cause found.", "no_action", {},
                   ["all metrics within normal dispersion"])