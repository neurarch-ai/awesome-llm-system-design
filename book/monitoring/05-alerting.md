# 5. Alerting

## The core rule: alert on rates and deltas, not on events

At any scale, a system that pages on-call for every single flagged answer will
train the team to ignore pages. A single ungrounded claim, a single judge
disagreement, a single slow request are noise. The signal is a **rate shift after
a change**, and the right instrument is a z-score computed against a recent
baseline.

For the ungrounded rate:

$$z_t = \frac{r_t - r_{\text{ref}}}{\sqrt{\,r_{\text{ref}}(1 - r_{\text{ref}}) / n_t\,}}$$

where $r_t$ is the ungrounded rate in the current window of $n_t$ judged traces
and $r_{\text{ref}}$ is the baseline rate from a reference window. Page when
$z_t$ exceeds your threshold (typically $z \geq 3$). This approach means a true
hallucination spike, one that moves the rate by several percentage points, fires
quickly while day-to-day noise does not.

Notice the role of $n_t$ in the denominator: a smaller window (heavier sampling)
narrows the confidence interval and makes it easier to detect small rate shifts,
but it also raises the minimum detectable effect size. Tune the sampling rate and
window size together, not independently.

## On-call tiers: matching alert speed to answer cost

Not every quality signal deserves a three-in-the-morning page. Match the alert
urgency to what a bad answer costs:

| Signal | Detection latency | Response tier |
|---|---|---|
| Guardrail block rate spike (safety regression) | minutes | page immediately (PagerDuty or equivalent) |
| Ungrounded rate z-score spike | minutes to hours | page within the hour |
| Judge score rolling average below threshold | hours | ticket, next business day |
| Input-distribution drift (embedding distance rising) | hours to days | weekly dashboard review, proactive investigation |
| Frozen eval replay regression on a scheduled run | depends on schedule (hourly to daily) | block the pending deploy, not a page |
| Cost per request p95 spike | minutes | page if above a cost-budget threshold |

The key distinction is between **block** (stops a deploy from reaching users) and
**page** (wakes someone up). Frozen eval regressions block; a gradual drift in
judge scores generates a ticket.

## Replay on every change

Every model swap and every prompt edit is a moment when quality can move silently.
Wire the frozen eval replay to trigger automatically on each such event, before
traffic shifts to the new configuration. This is the continuous version of the
deploy gate from pre-ship evaluation: the gate now runs forever, not just once.

The two-step on a change:

1. Run the frozen eval set against the candidate. If scores regress below the
   acceptable band, block the rollout.
2. If the frozen eval passes, route a canary slice to the candidate and monitor
   the live proxy scores, feedback, and latency for at least one traffic cycle
   (twenty-four hours covers a diurnal pattern) before expanding.

## Avoiding the tail-sampling trap

Uniform random sampling spends the human-review and judge budget on common, easy
requests and rarely encounters the rare failure pattern that is doing damage.
Stratify the sampling:

- Oversample requests with low or negative explicit feedback (discards,
  thumbs-down).
- Oversample requests where the edit or retry rate is high.
- Oversample requests where the retrieval score was low (the model answered from
  weak or empty context).
- Oversample requests where a guardrail fired or nearly fired.
- Include a uniform baseline slice so you have an unbiased estimate of overall
  quality.

This stratified design is what keeps the human-review queue pointed at failures
rather than burning budget on examples that would score well anyway.
