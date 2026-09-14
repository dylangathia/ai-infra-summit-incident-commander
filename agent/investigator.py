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

MAX_STEPS = 9
MAX_METRIC_COMPARISONS = 4    # beyond this, more metrics stop adding evidence
# Tool payloads are small — all six together are ~775 tokens. The dominant
# per-call cost is resending the system prompt, which compaction cannot help
# with. Truncating results therefore saved almost nothing and destroyed the
# early evidence (throughput direction, first dispersion check) that the
# conclusion depends on. These bounds only catch pathological growth; rate
# limits are handled by provider backoff instead.
KEEP_FULL_RESULTS = 12
TRUNCATE_TO = 1200

ACTIONS = [
    "drain_node",           # args: {"node_id": str}
    "rollback_node",        # args: {"node_id": str}
    "rate_limit_tenant",    # args: {"tenant": str, "admit_fraction": float}
    "cap_context",          # args: {"max_tokens": int}
    "scale_up",             # args: {"count": int}
    "no_action",            # args: {}
]

SYSTEM = f"""You are the on-call SRE for a GPU inference fleet serving several
tenants. An SLO breach has been detected. Find the root cause.

Fleet mechanics:
- Continuous batching; per-request throughput falls as batch size rises.
- Fixed KV cache per node. Long prompts fill it; when it saturates, sequences
  are preempted and must recompute, which is far costlier than it looks.
- The balancer routes to fewest outstanding requests, so a node serving
  nothing looks emptiest and attracts traffic it cannot handle.
- Nodes thermally throttle, get redeployed at a different quantization, or
  hang loading weights. The balancer knows none of this.

Method:
1. get_cluster_summary first. Note throughput_direction: rising means
   demand-side, falling means supply-side, flat means the fleet got slower at
   unchanged load.
2. Establish dispersion with compare_nodes. Read the `dispersion_verdict` and
   `outliers` fields and trust them. Do NOT eyeball per-node numbers — four
   loaded nodes always have spread, and judging by eye is how a busy fleet
   gets mistaken for a broken one. Three or four metrics is enough; more will
   not change the picture.
3. UNIFORM everywhere means no sick node, so the cause is demand-side or
   fleet-wide. get_top_talkers decides it — read its `demand_verdict` the
   same way you read dispersion_verdict, and do not infer from the tenant
   list yourself. BROAD means everyone is sending more: scale up, and never
   rate-limit, which would punish tenants who did nothing wrong. CONCENTRATED
   means one tenant is crowding out the others: rate-limit that one. A tenant
   sending *bigger* prompts at an unchanged rate is a third thing again and
   needs cap_context. Check get_recent_events for a deploy too.
4. CONCENTRATED means find what is different about that node — its logs,
   a recent deploy, its physical state.
5. Two causes can share one symptom. Check a second metric that separates
   them before concluding.

Call several tools in one turn when you know what you need — it is faster and
cheaper than one at a time.

Your evidence list must state the throughput_direction and the
dispersion_verdict for each metric you compared. Your ruled_out list is not
optional: name what you considered and what specifically refuted it.

When confident, reply with ONLY this JSON, no prose around it:

{{
  "root_cause": "<short slug>",
  "summary": "<one sentence>",
  "confidence": <0.0-1.0>,
  "evidence": ["<observation with numbers>"],
  "ruled_out": [{{"cause": "<considered>", "why": "<refuted by>"}}],
  "recommended_action": {{"action": "<one of: {', '.join(ACTIONS)}>", "args": {{}}}}
}}

Action args:
- drain_node / rollback_node: {{"node_id": "gpu-XX"}}
- rate_limit_tenant: {{"tenant": "<name>", "admit_fraction": 0.05}}
- cap_context: {{"max_tokens": 2048}}
- scale_up: {{"count": <int>}} — matching demand exactly leaves a permanent
  backlog; recovery needs surplus.
- no_action: {{}}"""


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


def _compact(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Truncate older tool results in place.

    Every step resends the whole history, so context grows quadratically and a
    single investigation can exceed a free tier's tokens-per-minute budget.
    Older results are truncated rather than removed, because dropping a tool
    message without its matching tool_call breaks both providers' validation.
    """
    idx = [i for i, m in enumerate(messages) if _is_tool_result(m)]
    for i in idx[:-KEEP_FULL_RESULTS] if len(idx) > KEEP_FULL_RESULTS else []:
        m = messages[i]
        if isinstance(m.get("content"), str):
            if len(m["content"]) > TRUNCATE_TO:
                m["content"] = m["content"][:TRUNCATE_TO] + " …(earlier result truncated)"
        elif isinstance(m.get("content"), list):
            for block in m["content"]:
                if isinstance(block, dict) and isinstance(block.get("content"), str) \
                        and len(block["content"]) > TRUNCATE_TO:
                    block["content"] = block["content"][:TRUNCATE_TO] + " …(earlier result truncated)"
    return messages


def _is_tool_result(m: Dict[str, Any]) -> bool:
    if m.get("role") == "tool":
        return True
    content = m.get("content")
    return (m.get("role") == "user" and isinstance(content, list)
            and any(isinstance(b, dict) and b.get("type") == "tool_result"
                    for b in content))


def _missing_evidence(inv: "Investigation") -> Optional[str]:
    """Refuse a conclusion that skipped the check its own findings demanded.

    A model that establishes UNIFORM dispersion everywhere has ruled out a
    sick node — which means the cause is demand-side, and it cannot know
    which without looking at per-tenant behaviour. Concluding at that point
    is a guess dressed as a finding.
    """
    used = {s.tool_name for s in inv.steps if s.kind == "tool"}
    verdicts = [(s.tool_result or {}).get("dispersion_verdict", "")
                for s in inv.steps if s.tool_name == "compare_nodes"]
    all_uniform = bool(verdicts) and all(v.startswith("UNIFORM") for v in verdicts)

    if all_uniform and "get_top_talkers" not in used:
        return ("Every metric you compared came back UNIFORM, so you have "
                "ruled out a single degraded node. That makes this demand-side "
                "or fleet-wide. You have not yet called get_top_talkers, so you "
                "cannot know whether a tenant changed its request rate or its "
                "request size — and those are different incidents with "
                "different remedies. Call it before concluding.")
    if all_uniform and "get_recent_events" not in used:
        return ("Before concluding a fleet-wide cause, check get_recent_events "
                "for a deploy or scaling action that would explain it. If there "
                "is none, say so in your evidence.")
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
    nudged = False
    steered = False

    for step in range(max_steps):
        final_step = step == max_steps - 1
        if final_step:
            messages.append({
                "role": "user",
                "content": ("You have used your tool budget. Do not call any more "
                            "tools. Conclude now with the JSON object, using the "
                            "evidence you already have. If you are genuinely "
                            "uncertain, say so in the summary and give your best "
                            "hypothesis with a low confidence value."),
            })
        try:
            resp: ModelResponse = provider.complete(
                system=SYSTEM,
                messages=(_compact(messages)
                          if getattr(provider, "compact_history", True)
                          else messages),
                tools=None if final_step else telemetry_tools.TOOL_SCHEMAS,
            )
        except Exception as exc:  # provider/network failure must not crash the loop
            inv.error = f"provider error: {exc}"
            return inv

        if resp.text:
            inv.steps.append(Step(kind="thought", content=resp.text))

        if not resp.wants_tools and not final_step:
            gap = _missing_evidence(inv)
            if gap and not nudged:
                nudged = True
                messages.append(provider.assistant_message(resp))
                messages.append({"role": "user", "content": gap})
                continue

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

        compared = sum(1 for st in inv.steps
                       if st.tool_name == "compare_nodes")
        incoming = sum(1 for c in resp.tool_calls if c.name == "compare_nodes")
        if compared + incoming > MAX_METRIC_COMPARISONS and not steered:
            steered = True
            messages.append(provider.assistant_message(resp))
            for call in resp.tool_calls:
                result = telemetry_tools.call_tool(cluster, call.name, call.arguments)
                inv.tool_calls_used += 1
                inv.steps.append(Step(kind="tool",
                                      content=f"{call.name}({call.arguments})",
                                      tool_name=call.name, tool_args=call.arguments,
                                      tool_result=result))
                messages.append(provider.tool_result_message(call, result))
            messages.append({"role": "user", "content": (
                f"You have compared {compared + incoming} metrics "
                "across nodes. Comparing more will not change the dispersion "
                "picture. Spend your remaining calls on the other tools: "
                "get_top_talkers shows whether a tenant changed its behaviour, "
                "get_recent_events shows deploys and scaling, get_node_logs "
                "explains a specific node.")})
            continue

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