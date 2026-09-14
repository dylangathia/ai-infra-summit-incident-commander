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

## Closed loop (`python3 lifecycle.py echo`)

`agent/commander.py` runs the full lifecycle: detect, investigate, validate,
execute, verify, and either close with a postmortem or escalate with the
failed hypothesis attached. All six scenarios resolve on the first attempt.

| scenario | detected via | policy | time to resolution |
|---|---|---|---|
| thermal_throttle | latency_slo | **rewritten** (drain + replace) | 10s |
| kv_exhaustion | latency_slo | approved | 100s |
| noisy_neighbour | latency_slo | approved | 80s |
| version_skew | **latency_regression** | approved | 5s |
| cold_start_stall | latency_slo | **rewritten** (drain + replace) | 20s |
| traffic_surge | latency_slo | approved | 35s |

### Three findings from running the loop end to end

**The policy layer earns its place on the very first scenario.** The
investigator proposed draining the throttled node. Policy computed that the
remaining nodes would have 3.0 effective capacity against demand of 3.9 and
rewrote the action to bring up a replacement alongside the drain. Without
that rewrite the incident gets worse, which is exactly what the earlier
efficacy run measured (5250ms -> 9350ms).

**Verification has to be patient, and patience has to be measured on the
right signal.** A fixed 30s check failed two *correct* diagnoses, because
capping context stops the bleeding instantly but the queue built during the
incident still has to drain — p99 rises before it falls. The verifier now
watches for up to 150s and treats a shrinking backlog as progress even while
latency is flat, since queued requests carry their old wait time to
completion. With that change, noisy_neighbour went from "misdiagnosed twice
then accidentally fixed" to "correct on attempt 1".

**Absolute thresholds never catch gray failures.** Version skew pushes p99 to
roughly 1.7x baseline while staying under a 4000ms SLO, so the detector never
fired at all. A third trigger compares against the trailing median rather than
a fixed number, which is what an operator actually notices.

## Business value (`python3 -m evals.compare --provider echo --seeds 3`)

The control arm is **not** "do nothing". It is a competent on-call engineer
who is paged, arrives after 5 minutes, and applies the correct remedy first
time. Beating a strawman proves nothing.

18 incidents (6 scenarios x 3 seeds), identical faults and seeds in both arms:

| | human on-call | agent | change |
|---|---|---|---|
| Degraded serving time | 67.2 min | 14.6 min | **-78%** |
| Requests served over SLO | 262,923 | 62,768 | **-76%** |
| Degraded GPU-hours | 4.62 | 1.11 | **-76%** |
| Correct resolution | given | 18/18 | — |

Per incident: 2.9 fewer minutes degraded, 0.195 GPU-hours recovered on a
4-node fleet. Extrapolated (and labelled as extrapolation in the output) to a
200-node fleet at 15 incidents/month: ~146 GPU-hours/month, about $381/month
at $2.60/node-hour, before counting the reduction in SLO-violating requests.

The honest framing: most of the gain is MTTD, not cleverness. The agent
detects in 7-15s what takes a human 13-35s to even be paged about, and never
loses the 5 minutes it takes a person to reach a keyboard.

### The two hardest bugs, both found by the eval

**A stalled node held a steady 1.5-2% error rate, just under a 2% threshold,
while p99 moved from 2100ms to 2400ms.** Neither trigger fired. The incident
would have run indefinitely. Error detection is now relative to the fleet's
own baseline — 0.0% normal to 1.5% sustained is an enormous regression even
though no absolute threshold is crossed. This is the most realistic failure
in the whole catalogue and the best argument in the pitch: the incidents that
cost the most are the ones your dashboard is configured not to see.

**Timeout errors arrive in bursts** as queued requests expire together, so a
rule requiring consecutive windows over threshold never fired. Detection now
averages over a span.

## Running it

```bash
pip install -r requirements.txt
uvicorn api.app:app --port 8000          # open http://localhost:8000
```

The fleet ticks on a wall clock at 8x speed. Pick a fault or leave it on
random, press **Break the fleet**, and watch the right-hand column: the agent
detects, calls tools, proposes an action, policy approves or rewrites it, and
the verifier waits until the fleet recovers before closing with a postmortem.

`POST /api/inject` returns the ground truth in `reveal_label`, but the UI does
not show it in the event stream. A judge can trigger a random fault, read the
agent's diagnosis, and only then check what was actually injected.

Set the provider with an environment variable — no code change:

```bash
IC_PROVIDER=echo                                    # rules, no API key needed
IC_PROVIDER=anthropic IC_MODEL=claude-haiku-4-5-20251001 ANTHROPIC_API_KEY=...
```

Deploy: `Procfile` for Railway or Render, `Dockerfile` for Fly. One service,
one URL — the dashboard is served by the same process as the API, so there is
no separate frontend to deploy or CORS to configure.

### Note on the live loop

The commander runs in a worker thread and its `advance()` callback sleeps in
wall-clock time rather than stepping the simulation. The fleet therefore keeps
degrading while the agent is thinking, which is how a real incident behaves —
it does not pause for the on-call engineer. It also means the demo shows
honest time-to-resolution rather than a frozen frame.