"""The investigator.

Deliberately NOT given the fault catalogue or any signature table. It gets
domain knowledge about how inference fleets behave and a set of tools, and
has to reach a conclusion the same way an on-call engineer would. If it is
handed the answer key it is a lookup table with extra steps, and a judge who
reads the system prompt will see that immediately.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fleet.cluster import Cluster
from providers.base import ModelResponse, Provider, ToolCall
from telemetry import tools as telemetry_tools

MAX_STEPS = 8

ACTIONS = [
    "drain_node",           # args: {"node_id": str}
    "rollback_node",        # args: {"node_id": str}
    "rate_limit_tenant",    # args: {"tenant": str, "admit_fraction": float}
    "cap_context",          # args: {"max_tokens": int}
    "scale_up",             # args: {"count": int}
    "no_action",            # args: {}
]

SYSTEM = f"""You are the on-call SRE for a GPU inference fleet serving LLM
traffic to several tenants. An SLO breach has been detected and you must find
the root cause.

How this fleet works:
- Each node serves requests with continuous batching. Throughput per request
  falls as batch size rises.
- Each node has a fixed KV cache. Long prompts consume it fast. When it
  saturates, in-flight sequences are preempted and must recompute from
  scratch, which is far more expensive than it sounds.
- The load balancer routes to the node with the fewest outstanding requests.
  A node that is not serving anything therefore looks like the emptiest node
  in the fleet and attracts traffic.
- Nodes can thermally throttle, be redeployed at a different quantization,
  or hang while loading weights. The balancer does not know any of this.

How to investigate:
1. Establish what changed at the fleet level, including the direction of
   throughput. Latency rising with throughput means something different from
   latency rising while throughput falls.
2. Determine whether the anomaly is CONCENTRATED on particular nodes or
   UNIFORM across all of them. This single distinction rules out half the
   possible causes. Do not skip it.
3. If uniform, the cause is demand-side or fleet-wide: look at per-tenant
   behaviour, and distinguish more requests from bigger requests.
4. If concentrated, identify the node and find out what is different about
   it: logs, recent deploys, its physical state.
5. Two different causes can produce the same symptom on one metric. Before
   concluding, check a second metric that would separate them.

Be skeptical of the obvious reading. A fleet where every node is equally
loaded is not a fleet with a broken node, however bad the latency looks.

When you are confident, reply with ONLY a JSON object, no prose around it:

{{
  "root_cause": "<short slug, e.g. thermal_throttle_on_node>",
  "summary": "<one sentence an on-call engineer would write>",
  "confidence": <0.0 to 1.0>,
  "evidence": ["<specific observation with numbers>", "..."],
  "ruled_out": [{{"cause": "<what you considered>", "why": "<what refuted it>"}}],
  "recommended_action": {{"action": "<one of: {', '.join(ACTIONS)}>", "args": {{}}}}
}}

Action arguments:
- drain_node / rollback_node: {{"node_id": "gpu-XX"}}
- rate_limit_tenant: {{"tenant": "<name>", "admit_fraction": 0.05}}
- cap_context: {{"max_tokens": 2048}}
- scale_up: {{"count": <int>}}  — remember that matching demand exactly leaves
  a permanent backlog; recovery needs surplus capacity.
- no_action: {{}}

Your "ruled_out" list is not optional. State what you considered and what
specifically refuted it."""


@dataclass
class Step:
    kind: str                 # "thought" | "tool" | "conclusion"
    content: str
    tool_name: Optional[str] = None
    tool_args: Optional[dict] = None
    tool_result: Optional[dict] = None


@dataclass
class Investigation:
    steps: List[Step] = field(default_factory=list)
    root_cause: Optional[str] = None
    summary: str = ""
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    ruled_out: List[dict] = field(default_factory=list)
    recommended_action: Dict[str, Any] = field(
        default_factory=lambda: {"action": "no_action", "args": {}})
    tool_calls_used: int = 0
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "root_cause": self.root_cause,
            "summary": self.summary,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "ruled_out": self.ruled_out,
            "recommended_action": self.recommended_action,
            "tool_calls_used": self.tool_calls_used,
            "error": self.error,
            "trace": [
                {"kind": s.kind, "content": s.content, "tool": s.tool_name,
                 "args": s.tool_args, "result": s.tool_result}
                for s in self.steps
            ],
        }


def _extract_json(text: str) -> Optional[dict]:
    """Models sometimes wrap JSON in prose or fences. Recover it."""
    if not text:
        return None
    cleaned = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    depth, start = 0, None
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    return json.loads(cleaned[start:i + 1])
                except json.JSONDecodeError:
                    start = None
    return None


def investigate(cluster: Cluster, provider: Provider,
                incident_note: str = "", max_steps: int = MAX_STEPS) -> Investigation:
    inv = Investigation()

    opening = (
        "An SLO breach has been detected on the inference fleet.\n"
        f"{incident_note}\n"
        "Investigate and determine the root cause. Start by establishing what "
        "changed at the fleet level."
    )
    messages: List[Dict[str, Any]] = [{"role": "user", "content": opening}]

    for _ in range(max_steps):
        try:
            resp: ModelResponse = provider.complete(
                system=SYSTEM,
                messages=messages,
                tools=telemetry_tools.TOOL_SCHEMAS,
            )
        except Exception as exc:  # provider/network failure must not crash the loop
            inv.error = f"provider error: {exc}"
            return inv

        if resp.text:
            inv.steps.append(Step(kind="thought", content=resp.text))

        if not resp.wants_tools:
            parsed = _extract_json(resp.text)
            if parsed is None:
                messages.append(provider.assistant_message(resp))
                messages.append({
                    "role": "user",
                    "content": "Reply with only the JSON object described in your instructions.",
                })
                continue
            inv.root_cause = parsed.get("root_cause")
            inv.summary = parsed.get("summary", "")
            inv.confidence = float(parsed.get("confidence", 0.0) or 0.0)
            inv.evidence = parsed.get("evidence", []) or []
            inv.ruled_out = parsed.get("ruled_out", []) or []
            action = parsed.get("recommended_action") or {}
            if action.get("action") in ACTIONS:
                inv.recommended_action = {
                    "action": action["action"],
                    "args": action.get("args", {}) or {},
                }
            else:
                inv.error = f"proposed unknown action: {action.get('action')!r}"
            inv.steps.append(Step(kind="conclusion", content=inv.summary))
            return inv

        messages.append(provider.assistant_message(resp))
        for call in resp.tool_calls:
            result = telemetry_tools.call_tool(cluster, call.name, call.arguments)
            inv.tool_calls_used += 1
            inv.steps.append(Step(
                kind="tool", content=f"{call.name}({call.arguments})",
                tool_name=call.name, tool_args=call.arguments, tool_result=result,
            ))
            messages.append(provider.tool_result_message(call, result))

    inv.error = f"no conclusion within {max_steps} steps"
    return inv