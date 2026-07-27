# 10. Putting it together: the complete build

Sections 1 through 6 taught each stage with its options and tradeoffs; section 7
showed where real teams diverge. What none of them show is a single system with
every decision made. This capstone does three things: it gives you an opinionated
default stack so option paralysis never blocks a first build, it walks the
chapter's scenario end to end with every choice committed and costed, and it
shows how the same decisions flip when the constraints change. It closes with
the smallest runnable batching scheduler, one file, no installs.

## The default stack: start here, deviate with reason

Every stage in this chapter has two to five credible options, and a first-time
builder can burn a week comparing engines before serving a single token. Skip
that. The stack below is a sane default for a first production build; each row
names when to deviate and which section explains why. Engines change yearly,
but the interface of each stage (schedule, prefill, cache, decode, shard,
quantize, scale, admit) does not, so pick per stage by interface and treat any
specific engine as replaceable.

| Stage | Default | Deviate when | Why (section) |
|---|---|---|---|
| Serving engine | vLLM-class: continuous batching plus PagedAttention, on by default | Never back to static batching; the GPU-idle problem is universal | [3](03-batching.md) |
| Batching policy | Iteration-level, token-budget packing, KV admission with headroom for future tokens | Uniform short outputs: admission headroom can shrink | [3](03-batching.md) |
| Prefill scheduling | Chunked prefill on a single pool | Prefill and decode SLOs conflict at fleet scale and NVLink-class fabric exists: disaggregate | [3](03-batching.md) |
| Parallelism | Smallest TP degree that fits the model, within one node; replicate whole copies for throughput | Model exceeds a node: PP across nodes; MoE experts exceed a GPU: EP | [5](05-parallelism-and-quantization.md) |
| Weight precision | FP8 on H100 or newer, behind a quality-eval gate | Pre-H100 hardware: INT8; fitting is the problem: 4-bit, higher quality risk | [5](05-parallelism-and-quantization.md) |
| KV cache | Paged, INT8 KV when concurrency (not weights) fills HBM | KV quality eval fails; attention-structure changes (MQA, sharing) need training, not a serving flag | [5](05-parallelism-and-quantization.md), [2](02-the-throughput-problem.md) |
| Speculative decoding | Off until per-workload acceptance is measured | Output echoes input: n-gram drafting, no second model to host | [4](04-speculative-decoding.md) |
| Autoscaling | Queue-depth leading signal, warm buffer, snapshot-restore cold start | Traffic genuinely flat: fixed fleet, no autoscaler to tune | [6](06-autoscaling-and-cost.md) |
| Admission | SLO gate, per-sequence KV reservation, 429 with retry-after under saturation | Never. Admitting everything under overload makes everyone miss | [6](06-autoscaling-and-cost.md) |

The last row is the one beginners skip and regret: without SLO-aware admission,
the first real spike turns into a retry storm and every request in flight misses
its target together. Wiring the 429 path before launch is an afternoon of work
that pays for itself the first morning peak.

## The complete build

Return to the scenario from [section 1](01-clarifying-requirements.md): a 70B
dense model, 500 QPS average with 3x spikes, p99 TTFT under 500 ms, p99
inter-token latency under 50 ms, a mixed workload of 8k-token RAG prompts and
long-output agent calls, H100s, a paid and a free tier, minimizing cost per
million output tokens. Here is the whole system with every choice committed and
the reason it won.

| Decision | Choice | Why it won |
|---|---|---|
| Engine | Continuous batching plus PagedAttention | Retiring finished sequences per step keeps the GPU saturated; paging cuts KV fragmentation below a few percent |
| Parallelism | TP=8 within one NVLink node; replicate nodes behind a load balancer | 140 GB of BF16 weights cannot fit one 80 GB H100; TP in-node also cuts per-token latency, which the 50 ms TPOT SLO needs |
| Weight precision | FP8, eval-gated | H100-native; halves weight bytes read per decode step on a bandwidth-bound phase |
| KV precision | INT8 KV cache | 8k-token RAG prompts make KV, not weights, the concurrency limit; halving KV bytes doubles the headroom |
| Prefill scheduling | Chunked prefill, one pool | 8k prefills would otherwise stall every in-flight decode and blow the TPOT SLO; disaggregation is not needed at this scale |
| Speculative decoding | Off at launch; revisit per workload | The mixed traffic cannot be predicted at admission; at a packed batch the spare compute speculation needs does not exist |
| Admission | SLO gate, per-sequence KV reservation, two priority queues | The paid tier must be protected under overload; free tier is shed first with a 429 and retry hint |
| Autoscaling | Queue-depth and wait-time leading signals, warm buffer, snapshot restore | Cold start is minutes, spikes are seconds; lagging signals react after the SLO is already broken |
| Launch gate | Quality, cost, and safety axes together | An FP8 or INT8-KV win that regresses quality, or a config that OOMs in-flight requests, does not ship on a cost number alone |

**GPU count and memory sizing.** The model forces the first decision: 70B
parameters in BF16 is about 140 GB of weights against 80 GB of HBM per H100
([section 1](01-clarifying-requirements.md)), so sharding precedes everything.
One TP=8 node holds 640 GB of HBM; FP8 weights take about 70 GB of it, roughly
9 GB per GPU, leaving the bulk of HBM for KV cache. With the chapter's KV
arithmetic ([section 2](02-the-throughput-problem.md)), a token costs about
320 KB in BF16 and about 160 KB with INT8 KV, so a mixed-traffic sequence
averaging 4,000 context tokens (Illustrative) holds about 640 MB; 64 live
sequences use about 41 GB, comfortably inside the node with room for the
8k-prompt outliers.

**Throughput and fleet size.** At an average of 300 output tokens per request
(Illustrative), 500 QPS is 150,000 output tokens per second fleet-wide. The
roofline bound for one node is generous: about 70 GB of FP8 weights plus 41 GB
of KV per step over roughly 27 TB/s of aggregate HBM bandwidth is near 4 ms per
step, but real engines run several times off roofline, so budget 25 ms achieved
(Illustrative). That is a TPOT of 25 ms, inside the 50 ms SLO, and a node
throughput of 64 sequences per 25 ms step, about 2,560 tokens/s per node or 320
tokens/s per GPU. The steady-state fleet is 150,000 / 2,560, about 60 nodes
(480 H100s). The 3x spike needs roughly 180; the difference is the autoscaler's
problem, below.

**Latency.** TTFT budget: a worst-case 8k-token prefill runs near 300 ms on the
node (Illustrative), and the autoscaler triggers when mean queue wait crosses
200 ms of the 500 ms budget ([section 6](06-autoscaling-and-cost.md)), so the
two components fit the SLO only when the queue signal fires early; this is why
the leading signal is load-bearing, not an optimization. Chunked prefill
stretches that prefill across several iterations, trading a little TTFT for a
smooth token stream on every other user's request.

**Cost per million output tokens.** From [section 6](06-autoscaling-and-cost.md),
cost = (GPU hourly rate x 10^6) / (tokens/s/GPU x 3600). At $3 per H100-hour
and 320 tokens/s/GPU that is about $2.60 per million output tokens; equivalently,
60 nodes at $24 per node-hour is about $35,000 per day (Illustrative) serving
about 13 billion output tokens. Compare the section's own worked example at 80
tokens/s/GPU: $10.40 per million. The entire gap is the denominator, which is
why continuous batching, FP8, and INT8 KV are business decisions, not tuning
details. Every remaining lever (speculation on a low-batch tier, better packing)
is judged by whether it moves that denominator without failing the quality gate.

**What breaks in month one.** Three failure modes dominate early operations, so
wire their signals before launch: preemption and swap counters (a rising count
means admission is outrunning the KV budget and the pool is heading into the
thrashing cliff from [section 3](03-batching.md)), mean queue wait time against
the 200 ms trigger (the leading signal for TTFT; if it alarms after p99 TTFT
moves, the autoscaler is wired to a lagging metric), and the 429-shed rate by
tier (free-tier shedding during spikes is the design working; paid-tier
shedding means the warm buffer is undersized or the cold start got slower).

## The same techniques under different constraints

The review question that matters in practice is not "which engine is best" but
"which stack is right under my constraints." Here is the same system built
three times. Only the middle column is the build above; the other two keep the
identical stage interfaces and swap nearly every implementation choice.

| | Single-node chat product | 70B two-tier API (this chapter) | Overnight batch generation |
|---|---|---|---|
| Model / traffic | 8B model, ~5 QPS interactive chat | 70B dense, 500 QPS with 3x spikes | 70B dense; millions of requests nightly, no interactivity |
| Latency budget | Sub-second TTFT feels good; no hard SLO | p99 TTFT < 500 ms, p99 TPOT < 50 ms | None; throughput and cost only |
| Parallelism | Fits one H100; two replicas for availability, none for scale | TP=8 in-node, ~60 replicated nodes | TP=8 in-node; as many nodes as the budget or spot market allows |
| Quantization | FP8 weights; BF16 KV is fine at this concurrency | FP8 weights plus INT8 KV | 4-bit weights if the eval gate passes; fewer bytes also means faster cold starts on spot |
| Batching | Continuous (always); chunked prefill barely matters with short prompts | Continuous, token-budget packing, chunked prefill | Continuous, packed to the compute roofline; latency of any single sequence is irrelevant |
| Speculative decoding | n-gram drafting if output echoes input (code, templates); measure acceptance first | Off; packed batches have no spare compute for verification | No; the batch is compute-saturated, exactly the regime where speculation loses |
| Autoscaling | Fixed two replicas; an autoscaler is more config than the fleet | Leading-signal scaling, warm buffer, tiered shedding | Scale-to-zero and spot capacity; cold starts are free when nothing is waiting |
| What would be over-engineering | TP, disaggregation, priority queues, warm fleets | Disaggregation (one pool still meets both SLOs) | Warm buffers, chunked prefill, priority tiers, streaming |

Two lessons fall out. First, the single-node column is mostly deletions: when
the model fits one GPU and traffic is light, parallelism, admission tiers, and
autoscaling all disappear, and the whole build is one engine flag away from
default. Second, the batch column shows the latency-throughput trade collapsing
to one side: with no SLO, every latency protection (chunked prefill, warm
buffer, speculation) becomes pure overhead, and the right move is the largest
batch the KV budget sustains on the cheapest interruptible hardware.

## What each constraint decides

The compressed decision guide. Read the left column off your requirements; the
right columns say which lever it moves before you compare any engines.

| Your constraint | Lever it moves | Rule of thumb |
|---|---|---|
| Model size vs one GPU's HBM | Parallelism | Fits: replicate for throughput. Does not fit: TP within a node. Exceeds a node: PP across nodes, accept the bubble |
| TPOT budget | Batch ceiling and precision | Pack until bandwidth-saturated, never past the KV budget; FP8 or INT8 halves bytes per step and buys TPOT directly |
| TTFT budget under mixed load | Prefill scheduling | Chunked prefill first; disaggregate only when the two SLOs conflict at fleet scale and fast fabric exists |
| Concurrency and long contexts | KV cache | Paged always; quantize KV when the cache, not the weights, fills HBM |
| Output echoes input | Speculative decoding | n-gram drafts with no second model; measure acceptance per workload, speedup goes below 1 when it is low |
| Spike speed vs cold start | Autoscaling signal and buffer | Scale on queue depth or wait time, never on latency; warm buffer sized to spike magnitude times cold-start duration |
| Priority tiers | Admission | Reserve the paid capacity slice first; shed the free tier with 429 and retry-after, never by silent queueing |
| Cost per million output tokens | Tokens/s/GPU, the denominator | Any 2x throughput lever halves cost; none of them ship without the quality eval gate |

## The smallest runnable scheduler

The review of every serving-engine tutorial is the same: the reader configures
flags and still cannot see the scheduler. So here is the batching decision in
one file with zero installs, a discrete-event simulation that runs the same
seeded request stream through static and continuous batching. Every production
component is swapped for the smallest thing with the same interface: a decode
step becomes one tick, the KV cache concurrency budget becomes eight slots, and
output-length variance, the villain of [section 3](03-batching.md), is a random
integer. The shape is the lesson; every section of this chapter upgrades one
piece of this file.

```python
"""Static vs continuous batching on one seeded request stream, no installs."""
import random

random.seed(7)
SLOTS = 8      # concurrent sequences the KV cache can hold
N = 200        # requests in the stream

# One shared workload: arrival step and output length per request.
arrive, out_len, t = [], [], 0.0
for _ in range(N):
    t += random.expovariate(0.08)               # bursty arrivals, ~0.08 req/step
    arrive.append(int(t))
    out_len.append(random.randint(8, 160))      # output-length variance is the villain

def simulate(policy):
    """One GPU; each tick is one decode step; every active slot emits one token."""
    remaining = out_len[:]
    finish, batch = {}, set()
    idle_slot_steps, busy_steps, step, next_req = 0, 0, 0, 0
    while len(finish) < N:
        # admission: continuous refills every step; static only when the batch retires
        if policy == "continuous" or not batch:
            while len(batch) < SLOTS and next_req < N and arrive[next_req] <= step:
                batch.add(next_req)
                next_req += 1
        active = [i for i in batch if remaining[i] > 0]
        if active:
            busy_steps += 1
            idle_slot_steps += SLOTS - len(active)   # held-but-finished slots waste here
            for i in active:
                remaining[i] -= 1
                if remaining[i] == 0:
                    finish[i] = step + 1
                    if policy == "continuous":
                        batch.discard(i)             # slot freed immediately
        if policy == "static" and batch and all(remaining[i] == 0 for i in batch):
            batch.clear()                            # whole batch retires together
        step += 1
    lat = sorted(finish[i] - arrive[i] for i in range(N))
    return {"tokens/step": sum(out_len) / max(finish.values()),
            "p50 latency": lat[N // 2],
            "p99 latency": lat[min(N - 1, int(N * 0.99))],
            "idle fraction": idle_slot_steps / (busy_steps * SLOTS)}

for policy in ("static", "continuous"):
    r = simulate(policy)
    print(f"{policy:>10}: " + "  ".join(f"{k}={v:.2f}" if isinstance(v, float)
                                        else f"{k}={v}" for k, v in r.items()))
```

Run it and the seeded stream reports static at 4.58 tokens per step with a 0.43
idle-slot fraction, p50 latency 720 steps and p99 1024, against continuous at
6.13 tokens per step, 0.23 idle, p50 87 and p99 199. The arrival rate was
chosen so demand sits between the two capacities: continuous batching keeps up
and latency stays near pure service time, while static batching's held-but-idle
slots (43% of its capacity, burned waiting for each batch's longest member) put
it into permanent saturation, so its latency is mostly queueing and grows with
the stream. That is the chapter's core claim in under fifty lines: same GPU, same
requests, and the scheduler alone decides whether the system is saturated. Each
toy piece stands in for a production one: the tick is a decode step whose cost
is flat because the weight read dominates and is amortized across slots, SLOTS
is the KV-cache concurrency budget that PagedAttention manages, the admission
loop is iteration-level scheduling, and the idle fraction is the GPU waste that
[section 3](03-batching.md)'s 8x scheduling win recovers. What it deliberately
omits is the rest of the chapter: prefill cost and chunking, KV growth and
preemption, quantization, and the autoscaler that adds nodes when even the
continuous line cannot keep up.
