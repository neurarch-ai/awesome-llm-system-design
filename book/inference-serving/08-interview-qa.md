# 8. Interview Q and A

The questions an interviewer actually asks about LLM inference serving, grouped by
how they are used. The commonly-missed ones are where interviews are won or lost.

## Commonly asked

**Q: Why is decode bandwidth-bound but prefill compute-bound?**

A: Prefill processes all prompt tokens in parallel in one forward pass. The GPU
performs many operations (attention, feed-forward, layer norms) per byte of weights
loaded, so arithmetic intensity is high and it runs on the compute side of the
roofline. Decode emits one token per pass. The GPU reads the entire model and the
entire KV cache to produce a single output, so operations per byte loaded is near
1 and it runs on the bandwidth side. Increasing batch size helps decode (amortizes
the fixed weight-read across more tokens per step) but barely helps prefill (it
is already compute-bound). Different bottlenecks; different levers.

**Q: What is continuous batching and why does it help?**

A: Static batching waits for a fixed group of requests, runs them all to completion,
then starts the next group. When requests finish at different times, the GPU sits
idle waiting for the longest member. Continuous batching schedules at the token
step: after each decode step the scheduler retires finished sequences and admits
waiting ones into the freed slots immediately. The GPU stays saturated because
there is no idle window between requests. Anyscale measured up to 23x throughput
over naive static batching (roughly 8x from the scheduling win and the rest from
PagedAttention's memory efficiency).

**Q: What is the KV cache, and why is it the binding memory constraint?**

A: The KV cache stores the key and value projections from attention for every
token a sequence has processed so far. These are read at every decode step to
compute attention over the context. For a 70B model with 80 layers, 8 KV heads,
128 head dim in BF16, each token consumes roughly 320 KB of KV cache. A sequence
of 4,000 output tokens fills about 1.3 GB, and 50 concurrent sequences of that
length need 64 GB, which exceeds H100 HBM before weights are counted. KV cache
size, not model weights, usually limits concurrency.

**Q: Why does a long prefill spike inter-token latency for other users?**

A: A prefill step on a large prompt occupies the GPU for an entire iteration:
every in-flight decode sequence is blocked until that step finishes. If a 32k-token
prefill takes 400 ms, every other user sees a 400 ms gap between tokens at that
moment. Chunked prefill fixes this by breaking the prefill into short chunks
interleaved with ongoing decode steps, so the pause is distributed across many
small steps rather than landing in one large one.

**Q: When should you use tensor parallelism vs. pipeline parallelism?**

A: TP splits each layer's matrices across GPUs with an all-reduce per layer. It
reduces per-GPU memory and can cut per-request latency, but needs fast interconnect
(NVLink) because the all-reduce fires at every layer for every token. Keep TP
within a single node. PP splits the model by layer groups across nodes, with
communication only at stage boundaries. It tolerates slower inter-node links but
adds a pipeline bubble that hurts single-request latency. The rule of thumb: TP
within a node to fit the model or cut latency; PP across nodes to scale past a
node's GPU count; replicate whole copies once a single copy fits.

## Tricky (the follow-ups that separate people)

**Q: Your p99 TTFT looks fine under normal load but blows up during spikes. What do you do?**

A: The problem is that you are scaling on a lagging signal (latency after it has
already risen). Switch to a leading signal: queue depth (requests waiting to start
prefill) or mean queue wait time. Trigger autoscaling when the queue builds, not
when TTFT has already missed. In parallel, add a warm-buffer replica or two so
spikes are absorbed while new instances boot. If the spike is fast enough that
new replicas cannot help, the only protection is SLO-aware admission: shed load
with a 429-style signal rather than admitting requests that will miss their target
anyway.

**Q: A team claims speculative decoding gives 4x throughput. Is that credible?**

A: It depends on the workload. Speculative speedup follows:
$\text{speedup} = (1 - \alpha^{k+1}) / ((1-\alpha)(1+ck))$.
At high acceptance rates ($\alpha \approx 0.8$, $k=4$, $c=0.1$) the formula gives
roughly 3-4x. LinkedIn achieved nearly 4x on the Hiring Assistant because output
heavily echoes the prompt, so n-gram drafts get very high acceptance. On free-form
creative generation, acceptance collapses and a generic draft can make inference
1.5x slower (Fireworks measured exactly this). The claim is credible for that
workload; it would not hold for arbitrary text generation.

**Q: When does increasing batch size stop helping throughput?**

A: There are two ceilings. First, the bandwidth ceiling: decode throughput grows
with batch size because the fixed weight-read cost is amortized across more tokens,
but once the GPU is saturated (all bandwidth used), adding more sequences does not
increase tokens/s/GPU. Second, the KV cache ceiling: as more concurrent sequences
fill HBM with their KV caches, eventually there is no room for more sequences and
the scheduler starts preempting or rejecting. Past that point, bigger "batches"
mean the scheduler is actually degrading. Measure your actual roofline and KV
occupancy before assuming bigger is always better.

**Q: Speculative decoding and large-batch throughput are both "make decode faster." Why can you not just stack them?**

A: They fight over the same resource. Speculative decoding trades spare compute for
fewer sequential target passes: each step runs the target on $k+1$ candidate tokens
in parallel and keeps the longest verified prefix. That parallel verification is
cheap only when the GPU has idle compute, which is true at low-to-moderate batch
size where decode is bandwidth-bound. Large-batch continuous batching (from Orca,
OSDI 2022) does the opposite: it packs enough concurrent sequences to push decode
toward the compute roofline, so the "free" verification compute no longer exists and
each speculative step's extra work now competes with real tokens. The mechanism-level
consequence is that the speculative speedup formula's overhead term $ck$ stops being
negligible once the batch is compute-saturated. In practice you pick one regime:
speculation for latency-sensitive low-batch traffic, big batches for
throughput-sensitive bulk traffic, and route between them rather than layering both.

**Q: How do you hold p99 latency when you retrain and redeploy the model?**

A: The main risk during a redeploy is the transition window where some replicas
run the old model and some run the new one. If the model changes output format or
behavior, clients receiving tokens from different versions may see inconsistent
streams. Use a rolling deploy with canary: route a small fraction of traffic to the
new version, validate both latency and output quality on live traffic, then
progressively shift. Keep old replicas available as a rollback target until the new
version is confirmed healthy. For disaggregated setups, also verify the prefill
and decode pools are updated together and that KV cache format is compatible between
versions.

## Commonly answered wrong (the traps)

**Q: Should you always pick the biggest batch size to maximize throughput?**

A: No. Past the bandwidth saturation point, adding more sequences fills HBM with
KV cache and the scheduler starts preempting running sequences, which hurts both
throughput and latency. The right target is the batch size where the GPU is
saturated without exceeding KV cache budget. Profile and set a hard limit; do not
assume "more is more."

**Q: Speculative decoding changes the output distribution, so you need extra eval before shipping?**

A: This gets the direction right but the reason wrong. Done correctly with rejection
sampling, speculative decoding is provably equivalent to sampling from the target
model; the output distribution is identical, not approximated. You need quality
eval not because the distribution is different, but because a buggy accept rule or
misconfigured verification can silently regress quality. Measure parity on a held-out
set before shipping, then treat it as a correctness check rather than a quality
trade.

**Q: Disaggregated prefill and decode is the right solution for any large-scale deployment.**

A: Only when prefill and decode SLOs genuinely conflict and fast interconnect is
available for the KV handoff. A single pool with chunked prefill is simpler, lower
latency, and sufficient for most workloads. Disaggregation adds the KV transfer
cost as a potential new bottleneck: without NVLink-speed interconnect between the
prefill and decode pools, the handoff dominates and you lose the win. Size the
interconnect first, or do not disaggregate.

**Q: You can apply quantization at serving time to any model without retraining.**

A: Weight quantization can often be applied post-training with minimal quality
loss (FP8 and INT8 on modern hardware), but KV cache quantization and, especially,
changes to the attention structure (MQA, GQA, cross-layer sharing) must be trained
into the model from the start. Switching a model that was trained with full KV heads
to MQA at serving time degrades quality significantly. The Character.AI approach
works because those reductions are baked into their training runs, not applied as
a serving-time overlay. Mechanism: MQA (Google, 2019) and GQA (Google, 2023)
collapse many key/value heads down to one or a few, so every query head must attend
through shared KV projections. If you mean-pool the trained per-head KV projections
at serving time, the query heads were never optimized to read from that averaged
subspace, and attention resolves to the wrong tokens; only a training run (or at
least a short uptraining pass) lets the query heads re-learn to share. Weight
quantization has no analogous problem because it perturbs each stored value slightly
rather than changing which parameters the attention computation reads.
