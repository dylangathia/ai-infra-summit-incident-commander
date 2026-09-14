# AI Incident Commander — fleet simulator (part 1)

Latency is never assigned here. It is computed from queues, batching and KV
cache occupancy. Faults change physical parameters only; the symptoms emerge.

## Run

```bash
python3 sim.py                    # random fault
python3 sim.py thermal_throttle   # a specific one
python3 sim.py all                # every scenario, baseline vs after
```

## Verified behaviour (SLO: p99 < 4000ms, baseline ~2200ms)

| scenario | post-fault p99 | primary signature |
|---|---|---|
| thermal_throttle | ~4900ms (2.2x) | one node, clock_factor down, temp >89C, queue grows |
| kv_exhaustion | ~27000ms (12.6x) | kv_util >95% fleet-wide, preemptions climbing |
| noisy_neighbour | ~20000ms (9.2x) | one tenant's request count and prompt size explode |
| version_skew | ~3550ms (1.6x) | one node slower, quantization=fp16, deploy event |
| cold_start_stall | ~2500ms (1.2x) | node state=loading, batch=0, kv=0, timeout bursts |
| traffic_surge | ~15000ms (7.1x) | ALL nodes loaded uniformly, no node anomalous |

The last two rows are the interesting ones. `traffic_surge` is not a fault:
draining a node there makes it worse. `cold_start_stall` barely moves p99 and
must be caught on error rate instead. An agent that only watches p99 and only
knows how to drain will get both wrong.

## Layout

- `fleet/node.py` — queueing, continuous batching, KV cache, preemption/recompute, thermal model
- `fleet/workload.py` — three tenants with different prompt-shape distributions
- `fleet/cluster.py` — least-outstanding-requests router, rolling 5s metric windows, agent actions
- `fleet/faults.py` — six injectors, randomisable
- `sim.py` — smoke test

Next: `telemetry/tools.py` (the five tools the investigator may call), then the agent loop.

## Verified tool signatures (`python3 probe.py`)

Baseline p99 ~2050ms, throughput ~60rps. All six are separable, but none by a
single metric — every one needs at least two tool calls.

| scenario | node-level dispersion | throughput | decisive evidence |
|---|---|---|---|
| thermal_throttle | `clock_factor` CONCENTRATED, `tokens_per_s` CONCENTRATED | flat | temp >89C, clock 0.46x on one node |
| version_skew | `clock_factor` UNIFORM, `tokens_per_s` CONCENTRATED | flat | deploy event + `quantization=fp16` in node log |
| cold_start_stall | `queue_depth`/`batch_size`/`kv_util` CONCENTRATED on newest node | flat | state=loading, batch=0, timeout bursts |
| kv_exhaustion | all UNIFORM | **falls** (60→42rps) | one tenant's avg prompt size x2.6 at *lower* request rate |
| noisy_neighbour | all UNIFORM | flat | one tenant's request rate x35 |
| traffic_surge | all UNIFORM | **rises** (60→84rps) | every tenant's rate up together, none anomalous |

Two pairs are deliberately near-identical and force multi-step reasoning:

- **thermal vs version_skew** — same `tokens_per_s` outlier shape. Only
  `clock_factor` dispersion plus the deploy event tells them apart.
- **surge vs kv_exhaustion vs noisy_neighbour** — all UNIFORM at node level.
  Separated only by throughput direction and per-tenant rate-vs-size ratios.

### Tool bugs caught by the probe (worth knowing, they were not obvious)

- `get_top_talkers` compared a 30s window against a 5s one, so every tenant
  appeared to grow ~6x in every scenario, including ones where demand was flat.
  Windows are now equal-length and rate-normalised.
- Mean/stdev outlier detection failed both ways: it flagged random nodes as
  outliers from ordinary jitter, and a switch to median/MAD then missed a node
  at 0.46x clock entirely, because MAD is exactly **zero** when the other three
  nodes are identical — which is the normal healthy case. The scale is now
  floored and paired with a practical-significance threshold.
- The autoscaler's scale-up was logged as an agent action, implying to the
  investigator that something had already been done about the incident.

`fault_injected` events are filtered out of `get_recent_events`. The agent
never sees ground truth.

## Agent loop (`python3 diagnose.py echo` / `python3 diagnose.py anthropic`)

`agent/investigator.py` runs a tool-calling loop against `telemetry/tools.py`
and returns a root cause, evidence, an explicit `ruled_out` list, and one
action from the allow-list. The loop is provider-agnostic: `providers/base.py`
defines the interface, `anthropic_provider.py` implements it, and adding a
sponsor provider is one new file.

The system prompt contains **no fault catalogue and no signature table**. It
describes how inference fleets behave and instructs the investigator to
establish dispersion before concluding. The answer key lives only in the eval.

`providers/echo_provider.py` is a hand-written decision tree over the same
tools, used as the heuristic baseline and to run the loop with no API key.
It scores 6/6 on the known scenarios — as it should, it was written with the
answer key in hand. The interesting comparison is compound and unseen faults,
where a fixed tree has no branch to take.

### A third tool bug, caught only by running the loop end to end

`get_top_talkers` compared the recent window against a baseline window that
**overlapped the incident**. A tenant flooding at 11x its normal rate appeared
as 1.8x, and the noisy-neighbour scenario was misdiagnosed as undetermined.
The baseline window now sits 150s back by default, exposes its own timestamps
so the caller can check for overlap, takes a `lookback_s` argument, and
reports `share_change` alongside rate — share is far more robust when the
incident has been running a while.