# AI Incident Commander

**An autonomous on-call engineer for GPU inference fleets.** It watches live
telemetry, detects when a latency SLO breaks, investigates by choosing what to
look at, proposes a fix, has that fix validated by a deterministic policy
layer, executes it, then verifies whether it actually worked — and escalates
itself if it didn't.

**Live demo:** https://ai-infra-summit-incident-commander.onrender.com
Pick a fault, press *Break the fleet*, and watch the right-hand column. The
free tier sleeps, so allow ~40s to wake on first load.

Built for the AI Infra Summit Hackathon. Target track: **AI Data Centers**.

---

## The problem

Running inference at scale is an operations problem that looks nothing like
web ops. A GPU node can thermally throttle without crashing anything. A KV
cache can saturate and start preempting sequences that must then recompute
from scratch. A node can hang loading weights and keep accepting traffic —
because the load balancer routes to whoever has the fewest outstanding
requests, and a node serving nothing looks like the emptiest box in the fleet.

Six different root causes. On a dashboard, all of them look like *p99 is
climbing*.

Two are worth staring at. **A thermally throttled node and a genuine traffic
surge produce near-identical latency curves.** The correct response to one is
to drain a node; the correct response to the other is to scale up. Drain
during a surge and you make the incident materially worse.

Telling those apart, with the reasoning shown, is what this is.

---

## Results

### Does it find the right cause?

Six scenarios. The system prompt contains **no fault catalogue and no
signature table** — it describes how inference fleets behave and what order to
investigate in, and nothing about what the faults are.

| provider | model | correct action |
|---|---|---|
| openai-compatible | `openai/gpt-oss-120b` via Groq | **6/6** |
| rules baseline | hand-written decision tree | 6/6 |

An earlier run on the same code scored 5/6, with the sixth scenario failing on
provider quota exhaustion rather than a wrong diagnosis.

The rules baseline scoring 6/6 is not a result — it was written with the
answer key in hand, and exists to run the loop without an API key and to give
the model something to beat. The number that matters is from a model that was
told nothing about the faults.

### Does it fix them?

All six resolve on the first attempt through the full loop.

| scenario | detected via | policy | time to resolution |
|---|---|---|---|
| thermal_throttle | latency_slo | **rewritten** (drain + replace) | 10s |
| kv_exhaustion | latency_slo | approved | 100s |
| noisy_neighbour | latency_slo | approved | 80s |
| version_skew | **latency_regression** | approved | 5s |
| cold_start_stall | **error_rate** | **rewritten** (drain + replace) | 20s |
| traffic_surge | latency_slo | approved | 35s |

Two of those detections would never fire on a conventional dashboard.
`version_skew` pushes p99 to ~1.7x baseline while staying *under* the 4000ms
SLO. `cold_start_stall` holds a steady 1.5–2% error rate against a 2% alert
threshold while p99 moves only 2100ms → 2400ms. Both are caught by triggers
that compare against the fleet's own baseline rather than a fixed number.

**The incidents that cost the most are the ones your dashboard is configured
not to see.**

### Is it worth anything?

The control arm is **not** "do nothing". It is a competent on-call engineer
who is paged, arrives after five minutes, and applies the correct remedy first
time. 18 incidents, 6 scenarios × 3 seeds, identical faults and seeds in both
arms:

| | human on-call | agent | change |
|---|---|---|---|
| Degraded serving time | 67.2 min | 14.6 min | **−78%** |
| Requests served over SLO | 262,923 | 62,768 | **−76%** |
| Degraded GPU-hours | 4.62 | 1.11 | **−76%** |
| Correct resolution | given | 18/18 | — |

Most of that gain is time-to-detection, not cleverness. The agent detects in
7–15s what takes a human 13–35s to be paged about, and never loses the five
minutes it takes a person to reach a keyboard.

Two rows are worth reading honestly. On `noisy_neighbour` the agent's MTTD is
*worse* than the human's (15s vs 13s), because the detector deliberately waits
an extra window rather than paging on a single spike. And `cold_start_stall`
shows almost no change in degraded minutes, because it barely breaches latency
at all — its real cost is 2% of one tenant's traffic failing silently for as
long as nobody notices.

Extrapolated — and labelled as extrapolation in the tool's own output — a
200-node fleet at 15 incidents/month would recover ~146 GPU-hours/month, about
$381/month at $2.60/node-hour, before counting the reduction in SLO-violating
requests.

---

## How it works

### The simulator: model the queue, not the metrics

The shortcut would be a generator that emits plausible numbers and flips them
when a fault fires. This doesn't do that. It models request arrival, per-node
queues, continuous batching, and service time as a function of batch size and
KV cache occupancy.

So when a fault is injected, the downstream effects **emerge**: the queue backs
up on its own, the balancer keeps routing to a sick node on its own, latency
cascades to neighbours on its own. Faults change *physical parameters* — clock
speed, cache capacity, quantization — and never touch latency directly.

That matters twice over. It gives the agent a genuinely hard problem instead of
a lookup exercise, and it means the demo survives a judge asking "what if you
also do X?" — because you can just do X and watch.

### The control loop

Four stages. The model is deliberately absent from two of them.

**Detector** *(deterministic)* — sustained SLO breach, error-rate regression,
or latency regression against the fleet's own trailing median. No model in the
hot path.

**Investigator** *(model, tool-calling)* — given an incident, decides what to
look at. Five tools: `get_cluster_summary`, `compare_nodes`,
`get_recent_events`, `get_top_talkers`, `get_node_logs`. Forms a hypothesis,
confirms or refutes it, and outputs a root cause with evidence and an explicit
`ruled_out` list. **This trace is the demo.**

**Planner + Executor** *(model proposes, policy validates)* — an allow-list of
actions with independent blast-radius checks. Policy can approve, **rewrite**,
or refuse.

**Verifier** *(deterministic)* — watches until the fleet recovers or
improvement stalls. On failure, hands the failed hypothesis back to the
investigator so the retry is informed rather than blind.

Closing with an auto-written postmortem.

### Ground truth never leaks

The chaos layer logs what it injected; every `fault_injected` event is filtered
out of `get_recent_events`. Node attributes are exposed only as an operator
would see them — temperature, clock, quantization — never as "this is the
faulty one". `POST /api/inject` returns the answer so *you* can check it, but
the UI never shows it in the event stream. Trigger a random fault, read the
agent's diagnosis, then check what it actually was.

---

## What we learned by watching it fail

Every improvement below came from a failed run, and almost all were fixed in
the **measurement** rather than the prompt. That distinction is the main
engineering claim here: *an agent reasoning over a misleading metric is not a
reasoning problem.*

**Demand was measured from completions.** `get_top_talkers` counted requests
the fleet *finished*. Under saturation short requests complete and long ones
time out, so during a genuine fleet-wide surge the tenant with the smallest
prompts appeared to grow while the others appeared to shrink — turning a surge
into a textbook noisy neighbour. The cluster now meters arrivals at ingress,
before routing or policy. This one would have misled a human operator exactly
as it misled the model.

**The baseline window overlapped the incident.** A tenant flooding at 11x its
normal rate measured as 1.8x, because "before" included the flood. The window
now sits 150s back, reports its own timestamps so the caller can check for
overlap, and exposes fleet-share change alongside rate.

**The model asked for a 300-second window and got less signal, not more.** A
40-second fault averaged into five minutes of healthy traffic disappears.
Windows are clamped at 60s at the tool boundary, with the clamped value
reported back.

**Outlier detection failed in both directions.** Mean and standard deviation
flagged random nodes from ordinary jitter. Median and MAD then missed a node
running at 0.46x clock — because MAD is exactly **zero** when the other three
nodes are identical, which is the normal healthy case. The scale is now floored
and paired with a practical-significance threshold.

**Tool payloads buried their own conclusions.** `compare_nodes` returned four
per-node rows before its `dispersion_verdict`, and the model reliably eyeballed
the numbers instead of reading the verdict — deciding a uniformly loaded fleet
had a broken node. Payloads now lead with the verdict, and `get_top_talkers`
computes BROAD vs CONCENTRATED rather than leaving it to be inferred.

**Verification was impatient, and watching the wrong signal.** A fixed 30s
check failed two *correct* diagnoses. Capping context stops the bleeding
instantly, but the queue built during the incident still has to drain, so p99
rises before it falls. The verifier now watches up to 150s and treats a
shrinking backlog as progress even while latency is flat, because queued
requests carry their old wait time to completion.

**The investigator spent its whole budget on one tool.** It compared six
metrics, established UNIFORM everywhere, ran out of calls before looking at
per-tenant behaviour, then guessed. It is now steered off metric comparison
after four — but only once every verdict is UNIFORM, because a CONCENTRATED
result means there is still a node worth chasing.

### Naive remediation is worse than nothing

`efficacy.py` verifies that each correct action actually restores the SLO. Two
findings shaped the policy layer:

**Draining a node without replacing it made the incident worse** — 5250ms →
9350ms. Removing a quarter of the capacity from a fleet at ~65% utilisation
overloads what remains. Policy refuses a drain that breaches the headroom floor
and rewrites it to bring up a replacement alongside.

**Scaling to parity does not clear a backlog.** Under surge, eight nodes
matched arrivals exactly, errors went to zero, and p99 sat at 9000ms
indefinitely with ~800 requests permanently queued. Recovery needs surplus, not
balance.

---

## Running it

```bash
pip install -r requirements.txt
uvicorn api.app:app --port 8000        # http://localhost:8000
```

The fleet ticks on a wall clock at 8x speed. The commander runs in a worker
thread whose `advance()` sleeps in real time, so **the fleet keeps degrading
while the agent is thinking** — a real incident does not pause for the on-call
engineer.

### Verification suite — no API key needed

```bash
python sim.py all          # every fault, baseline vs after
python probe.py            # can the tools discriminate the six scenarios?
python efficacy.py         # does each correct action restore the SLO?
python diagnose.py echo    # investigator, rules provider
python lifecycle.py echo   # full detect → verify → postmortem loop
python -m evals.compare --provider echo --seeds 3
python trace.py traffic_surge openai   # one full investigation, printed
```

### Providers

Copy `.env.example` to `.env`. Everything model-facing goes through
`providers/base.py`; adding a provider is one file.

| provider | env | cost |
|---|---|---|
| `echo` | none | free — rules baseline, runs the whole loop |
| `anthropic` | `ANTHROPIC_API_KEY` | prepaid credits |
| `openai` | `OPENAI_API_KEY` + `OPENAI_BASE_URL` | depends on host |

Because `OPENAI_BASE_URL` is configurable, that one file covers Groq, Cerebras,
Together and OpenRouter:

```dotenv
IC_PROVIDER=openai
OPENAI_BASE_URL=https://api.groq.com/openai/v1
OPENAI_API_KEY=gsk_...
IC_MODEL=openai/gpt-oss-120b
```

`.env` is gitignored. No key is committed.

### Deploy

`Procfile` for Railway or Render, `Dockerfile` for Fly. One service, one URL —
the dashboard is served by the same process as the API, so there is no separate
frontend and no CORS.

---

## Layout

```
fleet/        node.py  workload.py  cluster.py  faults.py
telemetry/    tools.py
agent/        detector.py  investigator.py  policy.py  commander.py
providers/    base.py  anthropic_provider.py  openai_provider.py  echo_provider.py
evals/        compare.py
api/          app.py  engine.py
web/          index.html
```

- `fleet/` — queueing, continuous batching, KV cache, preemption/recompute,
  thermal model, ingress metering, six fault injectors
- `telemetry/` — the five tools, with leakage guards and window clamping
- `agent/` — the control loop
- `providers/` — model adapters; `echo_provider.py` is the rules baseline
- `evals/` — agent vs human-baseline comparison
- `api/` + `web/` — live engine, websocket, single-file dashboard

MIT licensed.