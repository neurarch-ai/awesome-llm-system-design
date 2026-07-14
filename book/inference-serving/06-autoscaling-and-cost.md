# 6. Autoscaling and cost

Maximizing throughput per GPU only matters if you are not paying for idle GPUs
when load is low, and not letting requests miss their SLO when load spikes.
Autoscaling is the bridge. It is also where LLM serving introduces a wrinkle that
CPU services do not face: the cold start is measured in minutes, not milliseconds.

## The cold-start problem

Spinning up a new GPU replica requires scheduling a GPU node (often a scarce
resource pool), pulling a model image (potentially hundreds of gigabytes), loading
weights into HBM, and warming the engine (compiling CUDA graphs, building attention
caches). This can take two to five minutes. A traffic spike that doubles QPS arrives
in seconds. By the time new replicas are ready, the spike may already be over, and
the SLO was broken at the worst possible moment.

The standard autoscaling loop in CPU services reacts to CPU utilization or request
latency after it has already risen. In GPU LLM serving, these signals are lagging:
by the time p99 TTFT has risen, you have already missed SLOs for dozens of
requests and the new replicas are still booting.

## Leading-signal autoscaling

Scale on a **leading signal** that predicts SLO violation before it occurs:

- **Queue depth:** the number of requests waiting to start prefill is the clearest
  predictor of TTFT degradation. If the queue grows faster than the server drains
  it, TTFT will rise. Scale when the queue exceeds a threshold, not when TTFT
  already has.
- **Wait time in queue:** the time a request has spent waiting without beginning
  prefill is a direct predictor of TTFT. Trigger scale-up when mean wait time
  crosses a fraction of the TTFT budget (say, 200 ms of the 500 ms SLO).
- **GPU memory utilization:** KV cache occupancy is a leading indicator for
  throughput saturation. High KV utilization means the scheduler is turning away
  new sequences or preempting existing ones, which will shortly manifest as TTFT
  spikes.

Use CPU or GPU utilization as a secondary check, not the primary signal.

## The warm-buffer pattern

Keep a small number of **pre-warmed idle replicas** at all times. They absorb
traffic spikes while new replicas boot. The trade-off is paying for idle GPU time
during off-peak hours. The right buffer size is a function of spike magnitude,
cold-start duration, and SLO tolerance. If spikes are 3x average and cold start is
3 minutes, you need enough warm capacity to handle the difference for 3 minutes.
Many teams keep one or two warm replicas and accept a modest idle cost.

**Speeding the cold start** reduces how large a warm buffer you need:

- Cache the model image close to the GPUs (local NVMe, regional model registry).
- Stream weights into HBM during warm-up rather than waiting for a full copy.
- Snapshot a warmed process so the next boot can restore from a checkpoint instead
  of reinitializing. Modal's Memory Snapshots claim a 10x cold-start reduction.
- Use aggressive quantization (INT4) just for the cold-start load path: fewer bytes
  to fetch, faster to reach serving state.

Scale to zero only for models that serve cold paths where some latency on the first
request is acceptable. Never for the hot path.

## SLO-aware admission and load shedding

Under saturation, admitting every request makes all of them miss their SLO. The
correct behavior is **controlled load shedding**: when the server is saturated,
reject new requests with a clear 429-style signal and a retry hint rather than
queueing them indefinitely. Requests already admitted keep their slots and their
reserved KV-cache budget; the SLO is protected for those already in flight.

Two things must work together for this to be safe:

**Per-sequence KV reservation:** when a request is admitted, reserve its maximum
KV-cache budget immediately. This prevents a new admission from causing an
out-of-memory event that kills requests already mid-decode. vLLM's PagedAttention
addresses this with block-based allocation; naive contiguous allocation can OOM
mid-stream.

**Priority queues:** in a two-tier system (paid and free), maintain separate queues
or token budgets. Paid-tier requests are admitted first and see a lower effective
utilization ceiling; free-tier requests are the first to be shed. Assign the
guaranteed capacity slice before accepting anything from the lower tier.

## Cost per million output tokens

Throughput per GPU directly determines cost. The formula is:

$$\text{cost per million tokens} = \frac{\text{GPU hourly rate} \times 10^6}{\text{tokens/s/GPU} \times 3600}$$

For an H100 at \$3 per GPU-hour and 80 tokens/s/GPU throughput:

$$\text{cost} = \frac{3 \times 10^6}{80 \times 3600} \approx 10.4 \text{ USD per million tokens}$$

Doubling throughput per GPU halves cost. This is why continuous batching,
quantization, and speculative decoding have disproportionate business impact: each
raises the denominator of the above formula.

## Serving is a provider choice, not just a model choice

For a model you do not self-host, the same weights are served by many providers,
and they are not interchangeable: output tokens per second and price per token vary
several-fold across hosts for the identical open model, because each runs a
different engine, batching policy, quantization, and hardware. Do not quote a
single "the model costs X"; benchmark the **median (p50) speed and price per
provider** over a rolling window, since a one-shot measurement is noisy. Independent
trackers such as [Artificial Analysis](https://artificialanalysis.ai/) publish
exactly this, per-provider p50 output speed and price for each model, and it is the
right number to bring to a build-versus-buy or provider-selection decision. The
practical rule: choose the provider on the speed-and-price frontier that meets your
latency SLO, and re-check it periodically, because the frontier moves.

## Bottlenecks table

| Bottleneck | First sign | Root cause | Fix |
|---|---|---|---|
| Low GPU utilization | tokens/s/GPU well below roofline | Static batching or small effective batch size | Continuous (iteration-level) batching |
| TPOT spikes under mixed load | Inter-token latency spikes when new requests arrive | Long prefills blocking in-flight decode steps | Chunked prefill or disaggregated serving |
| KV cache OOM at high concurrency | Requests preempted or rejected mid-decode | KV cache fills HBM before compute is saturated | Paged KV, KV quantization, reduce max-sequence budget |
| Model does not fit on one GPU | Cannot start serving without sharding | Weights plus activation memory exceed HBM | Tensor parallelism within a node |
| Decode latency floor | Per-token latency irreducible even at batch 1 | One token per expensive target-model pass | Speculative decoding; verify acceptance before enabling |
| Tail latency under overload | p99 TTFT explodes at high QPS | Admitting all requests into a saturated queue | SLO-aware admission with 429 backpressure |
| Spike latency | New replicas not ready during a QPS surge | Cold-start duration longer than spike rise time | Warm buffer, leading-signal autoscaling, fast weight load |
| Memory bandwidth on decode | Throughput below bandwidth roofline even at large batch | Full-precision weight reads per step | Weight and KV quantization (FP8, INT8) |
