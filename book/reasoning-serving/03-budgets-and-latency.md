# 3. Budgets, latency, and the tail

## Output length is a random variable now

For a non-thinking model, response length is roughly predictable and its spread is
modest. For a thinking model, the same prompt can produce 400 tokens or 12,000, and
the distribution is right-skewed: most requests are near the mode, a minority run
several times longer, and those dominate everything downstream.

Two numbers describe the damage. Let $S$ be service time (output tokens divided by
decode rate). The mean $E[S]$ sets capacity. The **squared coefficient of
variation** $C^2 = \text{Var}(S)/E[S]^2$ sets the tail, because queueing delay in a
single-server queue grows with the second moment:

$$E[W] = \frac{\rho}{1-\rho} \cdot \frac{E[S]\,(1 + C^{2})}{2}$$

This is the Pollaczek-Khinchine result, and its practical content is blunt: **two
workloads with the same mean service time can have wildly different p99s, and the
thinking workload is always the one with the fatter tail.** Doubling variance
doubles the queueing component of latency at fixed utilization. It is also why
"just add capacity" is expensive here: the $\rho/(1-\rho)$ factor means you have to
run the fleet well below saturation to keep the tail civil.

## What that implies for the design

- **Budget every request.** A maximum thinking budget is a latency guarantee, not a
  cost preference. Without one, a single pathological request occupies a slot for
  minutes.
- **Truncation must be a designed behavior.** The wrong version silently cuts the
  output mid-thought and returns garbage (and, in evaluation, scores as a wrong
  answer, see [benchmarking, section 3](../benchmark-eval/03-the-harness.md)). The
  right version is a forced-answer step at the budget boundary ("you have used your
  thinking budget, give your best answer now"), or an explicit decline with a
  fallback path.
- **Admission control beats infinite queues.** Under overload, shed or downgrade:
  serve the non-thinking path rather than queueing a thinking request behind twenty
  others. A fast worse answer is usually better than a slow correct one that arrives
  after the user left.
- **Separate the tails you promise.** Time-to-first-token is now a poor proxy for
  anything, because the model may think for a long time before the first visible
  token. Promise, measure, and alert on time-to-first-visible-token, total time, and
  the fraction of requests that hit the budget cap.

## Head-of-line blocking is the real operational failure

A long thinking request holds a KV slot for its entire duration. With continuous
batching, that means:

- **Slot occupancy is skewed.** A handful of long generations can occupy most of
  the batch, so the effective concurrency for everyone else collapses.
- **Short requests queue behind long ones.** Classic head-of-line blocking, and the
  usual fix (shortest-job-first) is unavailable because you do not know the length
  in advance.
- **Memory pressure spikes with context.** Long thinking is long context: the KV
  cache for a 12,000-token generation is what the [KV cache
  chapter](../kv-cache/) sizes, and the cost lands on the same fleet.

Three mitigations, in the order to reach for them:

1. **Separate queues (or fleets) by budget class.** The cheapest structural fix:
   short requests never wait behind long ones because they are not in the same
   queue. It costs some efficiency in exchange for a predictable tail.
2. **Predict the length.** A cheap classifier over the prompt gives a rough class
   (short, medium, long), which restores approximate shortest-job-first scheduling
   and, usefully, is the same classifier that drives effort routing in
   [section 4](04-allocation-and-routing.md).
3. **Preemption.** Pause and evict a long generation when the queue backs up,
   resuming later. This is real work in the serving layer and is only worth it at
   scale, but it is what turns a hard tail into a soft one.

## Streaming and the perceived tail

The user does not experience the token distribution; they experience uncertainty.
Two cheap changes buy more than a lot of engineering:

- **Stream a progress signal during thinking** (a step count, a summary of the
  current step, or a plain indicator). The real latency is unchanged; the
  abandonment rate is not.
- **Support cancellation, and account for it.** A cancelled request should stop
  generating and should still be recorded, tokens and all, or your cost model
  quietly undercounts.

Whether to stream the reasoning trace itself is a product and safety decision, not
a systems one: traces can contain intermediate content the product should not show,
and they are the raw material for distillation by whoever reads them.

## When to use which control

| Reach for | When | Instead of |
|---|---|---|
| A hard token budget per request | Always, in production | An unbounded generation that can hold a slot indefinitely |
| Forced-answer at the boundary | The task must return something | Silent truncation, which returns malformed output and scores as failure |
| Separate queues by budget class | Mixed traffic with a latency promise on the short path | One queue, where short requests inherit the long tail |
| Length prediction plus priority | You have enough traffic to train a classifier and a tail problem | Pure FIFO, which is optimal for nothing here |
| Preemption | Large fleet, strict tail SLO, engineering budget | Adding capacity to paper over a variance problem |
| Admission control and downgrade | Overload periods | Queueing indefinitely and failing the SLO for everyone |
| Progress streaming | Any user-facing thinking path | A spinner and a hope |

## Implementation pitfalls

| Problem | Symptom | Fix |
|---|---|---|
| No budget cap | Occasional multi-minute requests; slots held hostage | Cap per request; alert on the fraction hitting the cap |
| Silent truncation at the cap | Malformed answers, and evaluation scores them as wrong reasoning | Forced-answer step or explicit decline |
| p99 tracked but not modelled | Fleet sized on mean service time; tail collapses under load | Size from $E[S]$ and $C^2$; keep utilization well below saturation |
| One queue for all budgets | Short interactive requests inherit the long tail | Budget-class queues or a priority scheme |
| TTFT used as the latency SLO | Metric looks fine while users wait | Measure time-to-first-visible-token and total time separately |
| Cancellations not accounted | Cost model drifts from the invoice | Record tokens generated for cancelled requests |
| Budget set by taste | Either the flat part of the curve or a truncation rate | Measure the accuracy-versus-budget curve on your own traffic and set the budget at the knee |
