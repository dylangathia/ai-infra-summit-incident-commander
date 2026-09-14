"""The only window the investigator has onto the fleet.

Two rules govern this module:

1. **No ground truth leaks.** The chaos layer logs what it injected. The agent
   must never see it. Every event of kind ``fault_injected`` is filtered out
   here, and node attributes are exposed only as an operator would see them
   (temperature, clock, quantization) — never as "this node is the faulty one".

2. **Comparison over recitation.** Dumping every metric invites the model to
   pattern-match on magnitude. These tools return *dispersion* — is this
   anomaly concentrated on one node or spread across all of them — because
   that is the question that separates a fault from a surge.
"""

from __future__ import annotations

import statistics
from typing import Any, Dict, List, Optional

from fleet.cluster import Cluster, WINDOW_S

HIDDEN_EVENT_KINDS = {"fault_injected"}

COMPARABLE_METRICS = [
    "queue_depth", "batch_size", "kv_util", "tokens_per_s",
    "temp_c", "clock_factor", "preemptions", "shed",
]


MAX_WINDOW_S = 60.0


def _clamp(seconds: float) -> float:
    """A window wider than the incident averages the fault into the baseline.

    Models reach for large windows expecting more signal and get less: a
    tenant flooding for 40s inside a 300s average barely moves.
    """
    try:
        seconds = float(seconds)
    except (TypeError, ValueError):
        return 30.0
    return max(5.0, min(seconds, MAX_WINDOW_S))


def _recent(cluster: Cluster, seconds: float) -> List:
    # No clamping here: internal callers legitimately ask for long windows
    # (the baseline lookup wants 5x the analysis window). Clamping applies
    # only to values the model supplies, at the public tool boundary.
    n = max(1, int(seconds / WINDOW_S))
    return list(cluster.history)[-n:]


def _avg(rows, attr) -> float:
    vals = [getattr(r, attr) for r in rows]
    return sum(vals) / len(vals) if vals else 0.0


# --------------------------------------------------------------------------


def get_cluster_summary(cluster: Cluster, window_s: float = 60.0) -> Dict[str, Any]:
    """Fleet-level health: is the SLO breached, by how much, and for how long."""
    window_s = _clamp(window_s)
    recent = _recent(cluster, window_s)
    if not recent:
        return {"error": "no telemetry yet"}

    baseline = _recent(cluster, window_s * 5)[: max(1, len(recent))]
    now = recent[-1]

    breach_windows = 0
    for w in reversed(list(cluster.history)):
        if w.p99 * 1000 > cluster.slo_p99_ms:
            breach_windows += 1
        else:
            break

    return {
        "slo_breached": now.p99 * 1000 > cluster.slo_p99_ms,
        "throughput_direction": (
            "RISING — demand-side; more work is arriving"
            if now.throughput_rps > _avg(baseline, "throughput_rps") * 1.12
            else "FALLING — supply-side; the fleet is serving less than it was"
            if now.throughput_rps < _avg(baseline, "throughput_rps") * 0.88
            else "FLAT — demand unchanged; the fleet is slower at the same load"),
        "slo_p99_ms": cluster.slo_p99_ms,
        "current": {
            "p50_ms": round(now.p50 * 1000),
            "p95_ms": round(now.p95 * 1000),
            "p99_ms": round(now.p99 * 1000),
            "ttft_p95_ms": round(now.ttft_p95 * 1000),
            "throughput_rps": round(now.throughput_rps, 1),
            "error_rate": round(now.error_rate, 4),
        },
        "window_avg": {
            "p99_ms": round(_avg(recent, "p99") * 1000),
            "throughput_rps": round(_avg(recent, "throughput_rps"), 1),
            "error_rate": round(_avg(recent, "error_rate"), 4),
        },
        "earlier_baseline": {
            "p99_ms": round(_avg(baseline, "p99") * 1000),
            "throughput_rps": round(_avg(baseline, "throughput_rps"), 1),
        },
        "breach_duration_s": round(breach_windows * WINDOW_S, 1),
        "node_count": len(cluster.nodes),
        "nodes_accepting": sum(1 for n in cluster.nodes if n.accepting),
        "hint": ("Throughput rising with latency suggests demand-side; "
                 "throughput flat or falling suggests supply-side."),
    }


def compare_nodes(cluster: Cluster, metric: str = "queue_depth",
                  window_s: float = 30.0) -> Dict[str, Any]:
    """Per-node values for one metric, with dispersion.

    This is the tool that separates a degraded node from a loaded fleet.
    """
    if metric not in COMPARABLE_METRICS:
        return {"error": f"unknown metric '{metric}'",
                "available": COMPARABLE_METRICS}

    window_s = _clamp(window_s)
    recent = _recent(cluster, window_s)
    if not recent:
        return {"error": "no telemetry yet"}

    per_node: Dict[str, List[float]] = {}
    for w in recent:
        for snap in w.nodes:
            per_node.setdefault(snap["node_id"], []).append(snap.get(metric, 0))

    values = {nid: sum(v) / len(v) for nid, v in per_node.items()}
    if not values:
        return {"error": "no nodes"}

    nums = list(values.values())
    # Median and MAD, not mean and stdev: with four nodes a single sick node
    # drags the mean toward itself and hides its own deviation.
    median = statistics.median(nums)
    mad = statistics.median([abs(v - median) for v in nums]) * 1.4826
    mean = statistics.fmean(nums)

    # MAD is exactly zero when most nodes are identical, which is the normal
    # case in a healthy fleet and precisely when a sick node must be caught.
    # Floor the scale so the degenerate case still produces a usable z.
    spread = max(nums) - min(nums)
    scale = max(mad, 0.05 * abs(median), 1e-9)
    # Practical significance: deviate by a fifth of the largest magnitude here.
    significant = 0.20 * max(abs(median), max(abs(v) for v in nums), 1e-9)

    rows = []
    for nid, val in sorted(values.items(), key=lambda kv: -kv[1]):
        dev = val - median
        z = dev / scale
        outlier = len(nums) > 2 and abs(z) > 2.5 and abs(dev) >= significant
        rows.append({
            "node_id": nid,
            "value": round(val, 3),
            "deviation_from_median": round(dev, 3),
            "robust_z": round(z, 2),
            "outlier": outlier,
        })

    cv = (spread / abs(median)) if abs(median) > 1e-9 else (1.0 if spread > 0 else 0.0)
    outliers = [r["node_id"] for r in rows if r["outlier"]]
    if outliers:
        verdict = "CONCENTRATED — one or more nodes deviate from the fleet"
    elif cv < 0.30:
        verdict = "UNIFORM — every node looks alike; this is fleet-wide, not a single sick node"
    else:
        verdict = "MIXED — spread is elevated but no clear outlier"

    return {
        "dispersion_verdict": verdict,
        "outliers": outliers,
        "metric": metric,
        "fleet_median": round(median, 3),
        "window_s_used": _clamp(window_s),
        "robust_dispersion": round(cv, 3),
        "window_s": window_s,
        "nodes": rows,
    }


def get_recent_events(cluster: Cluster, limit: int = 15) -> Dict[str, Any]:
    """Deploys, scale events and actions taken. Chaos events are filtered."""
    visible = [e for e in cluster.events if e["kind"] not in HIDDEN_EVENT_KINDS]
    return {
        "events": visible[-limit:],
        "note": "Correlate timestamps against when latency changed.",
    }


def get_top_talkers(cluster: Cluster, window_s: float = 30.0,
                    lookback_s: float = 150.0) -> Dict[str, Any]:
    """Per-tenant demand now versus earlier, in requests and in tokens."""
    window_s = _clamp(window_s)
    n = max(1, int(window_s / WINDOW_S))
    history = list(cluster.history)
    recent = history[-n:]
    # The earlier window must sit BEFORE the incident began, or the ratios
    # compare the incident against itself. A tenant flooding at 11x showed as
    # 1.8x when the baseline window overlapped the flood.
    back = max(1, int(lookback_s / WINDOW_S))
    start = max(0, len(history) - back - n)
    earlier = history[start:start + n] or history[:n]

    def agg(rows):
        out: Dict[str, Dict[str, float]] = {}
        for w in rows:
            for tenant, b in w.per_tenant.items():
                acc = out.setdefault(tenant, {"count": 0, "prompt_tokens": 0,
                                              "arrivals": 0})
                acc["count"] += b["count"]
                acc["prompt_tokens"] += b["prompt_tokens"]
                acc["arrivals"] += b.get("arrivals", 0)
        return out

    now, before = agg(recent), agg(earlier)
    now_windows = max(1, len(recent))
    before_windows = max(1, len(earlier))
    total_now = sum(v["count"] for v in now.values()) or 1
    total_before = sum(v["count"] for v in before.values()) or 1

    tenants = []
    for name, v in sorted(now.items(), key=lambda kv: -kv[1]["count"]):
        prev = before.get(name, {"count": 0, "prompt_tokens": 0, "arrivals": 0})
        prev_avg = prev["prompt_tokens"] / max(1, prev["count"])
        cur_avg = v["prompt_tokens"] / max(1, v["count"])
        # rate is measured on arrivals (offered load), not completions
        cur_rate = v.get("arrivals", v["count"]) / now_windows
        prev_rate = prev.get("arrivals", prev["count"]) / before_windows
        prev_share = prev["count"] / max(1, total_before)
        cur_share = v["count"] / total_now
        tenants.append({
            "tenant": name,
            "requests_offered": int(v.get("arrivals", v["count"])),
            "requests_served": int(v["count"]),
            "share_of_fleet": round(cur_share, 3),
            "share_of_fleet_before": round(prev_share, 3),
            "share_change": round(cur_share / prev_share, 2) if prev_share > 0 else None,
            "avg_prompt_tokens": round(cur_avg),
            "request_rate_change": (
                round(cur_rate / prev_rate, 2) if prev_rate > 0 else None),
            "avg_prompt_size_change": (
                round(cur_avg / prev_avg, 2) if prev_avg > 0 else None),
        })

    # Same idea as dispersion_verdict: compute the distinction rather than
    # hoping the model infers it from a list. One tenant growing while the
    # others shrink is a noisy neighbour. Every tenant growing together is
    # fleet demand, and rate-limiting anyone there punishes the innocent.
    grew = [t for t in tenants
            if (t.get("request_rate_change") or 1.0) > 1.3]
    shrank = [t for t in tenants
              if (t.get("request_rate_change") or 1.0) < 0.85]
    if len(grew) >= max(2, len(tenants) - 1):
        demand_verdict = ("BROAD — every tenant is sending more; this is "
                          "fleet-wide demand, not one tenant misbehaving")
    elif len(grew) == 1 and shrank:
        demand_verdict = (f"CONCENTRATED — {grew[0]['tenant']} grew while the "
                          "others shrank; one tenant is crowding out the rest")
    elif len(grew) == 1:
        demand_verdict = f"CONCENTRATED — only {grew[0]['tenant']} grew"
    else:
        demand_verdict = ("FLAT — no tenant materially changed its request "
                          "rate; if latency rose, the cause is not demand")

    size_spikes = [t["tenant"] for t in tenants
                   if (t.get("avg_prompt_size_change") or 1.0) > 1.8]

    return {
        "demand_verdict": demand_verdict,
        "tenants_sending_bigger_prompts": size_spikes,
        "window_s": window_s,
        "recent_window": [round(recent[0].t - window_s, 1), round(recent[-1].t, 1)] if recent else None,
        "baseline_window": [round(earlier[0].t - window_s, 1), round(earlier[-1].t, 1)] if earlier else None,
        "tenants": tenants,
        "note": ("request_count_change and avg_prompt_size_change are ratios "
                 "against an earlier window. A tenant sending the same number "
                 "of much larger requests is a different incident from one "
                 "sending many more requests. share_change is more robust than "
                 "request_rate_change when the incident has been running a "
                 "while: check the baseline_window timestamps against when the "
                 "breach started, and raise lookback_s if they overlap."),
    }


def get_node_logs(cluster: Cluster, node_id: str, limit: int = 12) -> Dict[str, Any]:
    """Operator-visible log lines for one node, derived from its actual state."""
    node = next((n for n in cluster.nodes if n.id == node_id), None)
    if node is None:
        return {"error": f"no such node '{node_id}'",
                "known_nodes": [n.id for n in cluster.nodes]}

    lines: List[str] = []
    t = round(cluster.t, 1)

    if node.state.value == "loading":
        lines.append(f"[{t}] WARN  weight load in progress, "
                     f"{node.load_remaining_s:.0f}s remaining (elapsed far beyond p99 load time)")
        lines.append(f"[{t}] WARN  readiness probe passing but no batch scheduled")
    if node.state.value == "draining":
        lines.append(f"[{t}] INFO  draining, no new admissions")
    if node.temp_c > 85:
        lines.append(f"[{t}] WARN  package temperature {node.temp_c:.1f}C above 85C limit")
    if node.clock_factor < 0.9:
        lines.append(f"[{t}] WARN  SM clock reduced to {node.clock_factor:.2f} of nominal")
    if node.kv_util > 0.9:
        lines.append(f"[{t}] WARN  kv cache {node.kv_util:.0%} occupied")
    if node.preemptions_window > 0:
        lines.append(f"[{t}] WARN  {node.preemptions_window} sequences preempted, "
                     f"recompute required on resume")
    if node.quantization != "fp8":
        lines.append(f"[{t}] INFO  serving {node.model_version} "
                     f"quantization={node.quantization}")
    if node.shed > 0:
        lines.append(f"[{t}] ERROR {node.shed} requests timed out in queue")
    if not lines:
        lines.append(f"[{t}] INFO  nominal: batch={len(node.batch)} "
                     f"queue={len(node.queue)} temp={node.temp_c:.0f}C")

    return {"node_id": node_id, "state": node.state.value, "lines": lines[-limit:]}


# --------------------------------------------------------------------------
# Tool-calling schema + dispatch

TOOL_SCHEMAS = [
    {
        "name": "get_cluster_summary",
        "description": ("Fleet-wide latency, throughput, error rate and SLO status, "
                        "with an earlier baseline for comparison. Start here."),
        "input_schema": {
            "type": "object",
            "properties": {
                "window_s": {"type": "number",
                             "description": "Seconds to average over. Default 60."}
            },
        },
    },
    {
        "name": "compare_nodes",
        "description": ("Compare one metric across every node and report dispersion. "
                        "Use this to determine whether a problem is concentrated on "
                        "specific nodes or uniform across the fleet. Metrics: "
                        + ", ".join(COMPARABLE_METRICS)),
        "input_schema": {
            "type": "object",
            "properties": {
                "metric": {"type": "string", "enum": COMPARABLE_METRICS},
                "window_s": {"type": "number"},
            },
            "required": ["metric"],
        },
    },
    {
        "name": "get_recent_events",
        "description": ("Recent deploys, autoscaler activity and operator actions, "
                        "with timestamps. Use to correlate a change against a "
                        "config or deployment event."),
        "input_schema": {
            "type": "object",
            "properties": {"limit": {"type": "integer"}},
        },
    },
    {
        "name": "get_top_talkers",
        "description": ("Per-tenant demand, with ratios against an earlier window "
                        "for both request count and average prompt size."),
        "input_schema": {
            "type": "object",
            "properties": {"window_s": {"type": "number"}},
        },
    },
    {
        "name": "get_node_logs",
        "description": "Recent log lines for one node. Use after identifying a suspect.",
        "input_schema": {
            "type": "object",
            "properties": {"node_id": {"type": "string"}},
            "required": ["node_id"],
        },
    },
]

_DISPATCH = {
    "get_cluster_summary": get_cluster_summary,
    "compare_nodes": compare_nodes,
    "get_recent_events": get_recent_events,
    "get_top_talkers": get_top_talkers,
    "get_node_logs": get_node_logs,
}


def call_tool(cluster: Cluster, name: str, args: Optional[dict] = None) -> Dict[str, Any]:
    fn = _DISPATCH.get(name)
    if fn is None:
        return {"error": f"unknown tool '{name}'",
                "available": list(_DISPATCH)}
    try:
        return fn(cluster, **(args or {}))
    except TypeError as exc:
        return {"error": f"bad arguments for {name}: {exc}"}