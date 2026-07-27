# 10. Putting it together: the complete build

Sections 1 through 6 taught each layer with its options and tradeoffs; section 7
showed where real teams diverge. What none of them show is a single system with
every decision made. This capstone does three things: it gives you an opinionated
default stack so option paralysis never blocks a first build, it walks the
chapter's scenario end to end with every choice committed and costed, and it
shows how the same decisions flip when the constraints change. It closes with
the smallest runnable monitor, one file, no installs.

## The default stack: start here, deviate with reason

Every layer in this chapter has several credible options, and a first-time
builder can burn a week comparing observability vendors before logging a single
trace. Skip that. The stack below is a sane default for a first production
build; each row names when to deviate and which section explains why. Tools
change yearly, but the interface of each layer (trace, meter, proxy-score,
drift-check, alert) does not, so pick per layer by interface and treat any
specific platform as replaceable.

| Layer | Default | Deviate when | Why (section) |
|---|---|---|---|
| Tracing | OTel-style span per step; verbatim inputs, retrieved context, output; model id and prompt version on the generation span | Single-shot endpoint with no retrieval or tools: one generation span carries everything | [2](02-what-to-observe.md) |
| Metrics on all traffic | p50/p95/p99 and TTFT (not the mean), cost and tokens per request, error rate by class, derived from span attributes | Never. They are nearly free once the trace exists | [2](02-what-to-observe.md), [6](06-serving-and-scaling.md) |
| Online eval without labels | Sampled LLM judge (faithfulness + relevance) plus grounding check against logged context plus implicit user signals; calibrate kappa against human labels before alerting on any of them | Answers are not supposed to be grounded in documents: drop the grounding check, keep the judge and behavioral signals | [3](03-online-eval-without-labels.md) |
| Drift detection | Input-embedding cosine distance vs a reference window (cheap encoder, all traffic) as the leading indicator; output proxies confirm | Traffic too thin for stable windows: lean on scheduled frozen eval replay instead | [4](04-detecting-drift-and-regressions.md) |
| Regression gates | Frozen eval replay on every model or prompt change; canary 5 to 10 percent for a 24-hour traffic cycle before full rollout | Mistakes are irreversible (an agent executing actions): shadow only until the diff evidence justifies exposure | [4](04-detecting-drift-and-regressions.md) |
| Alerting policy | z >= 3 on rate deltas over windows, never on single events; tiers: guardrail spike pages, ungrounded z-spike pages, judge decay tickets, input drift goes to a dashboard | Strong weekly seasonality: matched hour-of-week baseline so you alert on the residual, not the periodicity | [5](05-alerting.md), [8](08-interview-qa.md) |
| Sampling | Stratified, not uniform: oversample discards, heavy edits, low retrieval scores, guardrail near-misses, plus a uniform baseline slice | High-stakes change window: raise the rate temporarily, then drop it once confidence is established | [5](05-alerting.md), [6](06-serving-and-scaling.md) |

The calibration clause in the eval row is the one beginners skip and regret: an
uncalibrated judge is a confident guess, and paging on a confident guess is how
a monitoring channel gets muted in its first month. A few hundred human labels
pay for themselves the first time the judge and the users disagree.

## The complete build

Return to the scenario from [section 1](01-clarifying-requirements.md): an
enterprise support RAG copilot, no labels on production traffic, prompt edited
several times a week, model swapped quarterly, an observation budget of fifteen
percent of serving cost, a human agent accepting or discarding each answer with
a five-to-thirty-minute lag, and regressions that must surface within hours.
Here is the whole system with every choice committed and the reason it won.

| Decision | Choice | Why it won |
|---|---|---|
| Tracing | Span per step, retrieved context logged verbatim, conversation id across turns | Without the logged context, grounding checks are impossible after the fact; multi-turn sessions must be reconstructable |
| Quality proxies | Judge on a 10 percent sample, grounding check on the same slice, accept/discard rate on all traffic | The agent's accept/discard is a dense, honest, free label proxy; the budget caps judging at fifteen percent, so leave headroom |
| Judge design | Two-stage (free reasoning, then small-model reformat), model and prompt version pinned, kappa measured on a few hundred human labels before any alert | Forcing structured output mid-reasoning costs accuracy; an unpinned judge manufactures fake regressions |
| Drift monitor | Cheap encoder embeds every query; cosine distance vs a reference window reset after each deliberate change | Leading indicator at a fraction of generation cost; a stale reference fires on every intended change |
| Regression gates | Frozen replay wired to every prompt edit; quarterly model swaps get shadow diff, then a 5 to 10 percent canary for 24 hours | Prompt edits are the frequent risk here; the eval set is refreshed from flagged production traces so it does not go stale |
| Alerting | Guardrail spike pages immediately; ungrounded z >= 3 pages within the hour; judge rolling-average decay tickets; input drift goes to weekly review | Match alert speed to what a bad answer costs; input drift alone never pages, because it predicts trouble without confirming it |
| Sampling | Stratified weights: discards, heavy edits, low retrieval scores, guardrail near-misses, plus a uniform slice | Uniform sampling burns the judge and human budget on easy requests and misses the rare failure doing damage |
| Retention | Full fidelity for flagged traces, truncated for clean ones after a short window, PII redacted before long-term storage | Otherwise the trace store becomes the largest pool of sensitive data in the infrastructure |

**Sampling rate and judge cost.** Take an illustrative 100,000 requests per day.
A judge call costs roughly as much as the generation call itself
([section 3](03-online-eval-without-labels.md)), so judging everything doubles
the serving bill. At a 10 percent sample the judge adds about 10 percent of
serving cost; the grounding check rides the same sampled slice, and the drift
encoder and span-derived metrics run on all traffic for another point or two.
Total: roughly 12 percent, inside the fifteen-percent budget with headroom
reserved for temporarily raising the rate around a model swap.

**Detection latency.** The formula from [section 6](06-serving-and-scaling.md),
$t_{\text{detect}} \approx k / (s \cdot \lambda \cdot r_{\text{fail}})$, with
$s = 0.10$, $\lambda$ = 100,000 requests per day, a failure rate of 2 percent,
and $k = 50$ flagged traces for confidence, gives 50 / (0.10 x 100,000 x 0.02)
= 0.25 days, about six hours. That satisfies the "within hours, not days"
requirement, and it makes the tradeoff concrete: halving the sample to 5
percent halves the observation bill and pushes detection to half a day.

**Alert thresholds.** At a 10 percent sample the judge scores about 10,000
traces per day, so a 500-trace alerting window fills in roughly 72 minutes.
With a baseline ungrounded rate of 5 percent, the z-score from
[section 5](05-alerting.md) reaches 3 when the windowed rate hits about 8
percent: a three-point rate shift pages within about an hour of onset, while
day-to-day noise does not. Window size and sampling rate were tuned together,
because $n_t$ sits in the denominator of the same formula.

**Storage.** An LLM trace with verbatim prompt, retrieved context, and output
runs about 20 KB (illustrative). At 100,000 requests per day that is 2 GB per
day, roughly 60 GB per month of raw traces. Tiered retention keeps full
fidelity only for the flagged ~10 percent and truncates the rest after a short
window, cutting steady-state storage several-fold and shrinking the
redaction surface at the same time.

**What breaks in month one.** Three failure modes dominate early operations, so
wire their signals before launch: a silent quality regression under stable
latency (a Friday prompt edit ships, latency and error dashboards stay green,
and the ungrounded rate climbs all weekend because frozen replay was wired to
model swaps but not prompt edits); judge drift (the faithfulness score trends
up while kappa against fresh human labels falls, meaning the instrument is
flattering, not the product improving; recalibrate on a schedule, not on
suspicion); and alert fatigue (a stationary-baseline z-score pages every Monday
morning on seasonal traffic until the channel gets muted; switch to a matched
hour-of-week baseline before trust is gone, per
[section 8](08-interview-qa.md)).

## The same techniques under different constraints

The review question that matters in practice is not "which observability
platform is best" but "which monitoring design is right under my constraints."
Here is the same layer stack built three times. Only the middle column is the
build above; the other two keep the identical layer interfaces and swap nearly
every implementation choice.

| | Internal docs bot | Enterprise support copilot (this chapter) | Regulated consumer assistant |
|---|---|---|---|
| Traffic / stakes | ~2k requests/day; a wrong answer wastes minutes | ~100k requests/day (illustrative); a confident wrong answer damages customer trust | Millions of requests/day; a harmful answer is regulatory exposure |
| Observation budget | Absolute judge cost is trivial: judge 100 percent of traffic if you like | 15 percent of serving cost; judge ~10 percent, stratified | Larger budget, but 100 percent judging is still unaffordable; cheap triage model scores everything, full judge takes the flagged tail |
| Quality signal | Thumbs plus a weekly skim of judged traces | Judge + grounding + agent accept/discard joined on trace id | Judge + grounding + mandatory safety re-scan sampled from allowed traffic |
| Drift and gates | Frozen replay on deploy; no drift monitor, traffic too thin for stable windows | Replay on every prompt edit; shadow then 5 to 10 percent canary on model swaps | Shadow-first always; long canaries; human sign-off gates on top of automated ones |
| Alerting | No pager. Weekly dashboard review | Tiered: page / ticket / dashboard, z >= 3 on rate deltas | Guardrail and safety-re-scan spikes page immediately, around the clock |
| Human review | The developer reads flagged traces over coffee | Stratified queue calibrating the judge monthly | Standing review team; labels feed both calibration and compliance audit |
| What would be over-engineering | Canary infrastructure, stratified sampling, on-call tiers | 100 percent judging, standing review team | Nothing on the safety side; uniform sampling is what would be negligent |

Two lessons fall out. First, the docs-bot column is mostly deletions: at 2k
requests per day the judge bill is pocket change even at full coverage, the
traffic is too thin for windowed drift statistics to be stable, and a pager
would fire more often on variance than on truth. The frozen replay on deploy
plus a weekly skim is the whole system, and that is correct, not lazy. Second,
the regulated column shows the observation budget and the alerting tier moving
in opposite directions from cost: when a miss is a regulatory event, sampling
gets stratified harder rather than cheaper, the safety re-scan on allowed
traffic stops being optional ([section 8](08-interview-qa.md)), and the
expensive judge is protected by a cheap first-pass triage model instead of a
lower sampling rate.

## What each constraint decides

The compressed decision guide. Read the left column off your requirements; the
right columns say which lever it moves before you compare any tools.

| Your constraint | Lever it moves | Rule of thumb |
|---|---|---|
| Observation budget | Judge sampling rate $s$ | Judge cost is linear in $s$; halving $s$ halves the bill and doubles detection time. Spend the savings on stratification, not coverage |
| Detection-latency target | $s$ and window size, together | $t \approx k / (s \lambda r_{\text{fail}})$; raise $s$ temporarily around a high-stakes change, then lower it |
| Cost of a wrong answer | Alert tier | Confirmed output regression pages; gradual judge decay tickets; input drift alone never pages |
| Change frequency | Replay trigger | Prompt edits several times a week mean replay fires on the edit event, not on a nightly schedule |
| Ground-truth lag | Calibration cadence | Minutes-lag behavioral labels: join on trace id and trend daily. Human labels: recalibrate the judge on a schedule and report kappa next to the score |
| Traffic volume | Window size $n_t$ | Thin traffic with small windows trips on variance; size $n_t$ so the z-test can see the smallest rate shift you care about |
| Grounded-answer promise | The one span field | Log retrieved context verbatim at the retrieval span; no downstream check can recover it later |
| Seasonal traffic | Baseline choice | Matched hour-of-week baseline; alert on the residual after known periodicity, not on the periodicity |
| Privacy and compliance | Retention and redaction | Full fidelity for flagged traces only; redact PII before long-term storage; gate raw-trace access separately from dashboards |

## The smallest runnable monitor

The review of every observability tutorial is the same: the reader assembles a
vendor SDK and a dashboard and still cannot see the mechanism. So here is the
chapter's core detection loop in one file with zero installs. The sampled judge
becomes a seeded stream of daily quality scores, the regression becomes a known
injection day, and the two alerting philosophies from
[section 5](05-alerting.md) run side by side: a static threshold ("alert if the
score looks bad") against a rolling-window drift detector (current window mean
vs a frozen reference, in standard-error units). The shape is the lesson; every
section of this chapter upgrades one function of this file.

```python
"""Static threshold vs rolling-window drift detection, runnable with no installs."""
import random, statistics

random.seed(13)

# --- simulate a daily quality proxy (sampled judge faithfulness score) -------

BASELINE_MEAN, NOISE_STD = 0.90, 0.02
REGRESSION_DAY, DAILY_DECAY = 31, 0.004          # slow regression: a bad prompt edit

def score(day):
    """Mean judge score for one day; production: aggregated from sampled traces."""
    drift = max(0, day - REGRESSION_DAY + 1) * DAILY_DECAY
    return BASELINE_MEAN - drift + random.gauss(0, NOISE_STD)

scores = [score(d) for d in range(1, 61)]        # days 1..60

# --- detector 1: static threshold (what most teams wire first) ---------------

STATIC_THRESHOLD = 0.82                          # "alert if the score looks bad"

def static_alert(scores):
    for day, s in enumerate(scores, 1):
        if s < STATIC_THRESHOLD:
            return day
    return None

# --- detector 2: rolling-window drift (baseline window vs current window) ----

BASE_WINDOW, CUR_WINDOW, Z_PAGE = 14, 7, 3.0     # page at z >= 3, as in alerting

def drift_alert(scores):
    base = scores[:BASE_WINDOW]                  # frozen reference window
    mu, sd = statistics.mean(base), statistics.stdev(base)
    for day in range(BASE_WINDOW + CUR_WINDOW, len(scores) + 1):
        cur = statistics.mean(scores[day - CUR_WINDOW:day])
        z = (mu - cur) / (sd / CUR_WINDOW ** 0.5)   # std-error units of the mean
        if z >= Z_PAGE:
            return day
    return None

# --- report ------------------------------------------------------------------

s_day, d_day = static_alert(scores), drift_alert(scores)
print(f"regression injected at day {REGRESSION_DAY} "
      f"(-{DAILY_DECAY:.3f}/day, noise std {NOISE_STD})")
print(f"static threshold < {STATIC_THRESHOLD}: "
      + (f"fires day {s_day}, lag {s_day - REGRESSION_DAY} days" if s_day else "never fires"))
print(f"rolling drift z >= {Z_PAGE:.0f}:        "
      + (f"fires day {d_day}, lag {d_day - REGRESSION_DAY} days" if d_day else "never fires"))

# rerun with a shallower regression: static goes blind, drift still catches it
DAILY_DECAY = 0.002
random.seed(13)
scores = [score(d) for d in range(1, 61)]
s_day, d_day = static_alert(scores), drift_alert(scores)
print(f"shallower regression (-{DAILY_DECAY:.3f}/day): "
      f"static {'day ' + str(s_day) if s_day else 'never fires'}, "
      f"drift fires day {d_day}, lag {d_day - REGRESSION_DAY} days")
```

Run it and the output demonstrates the chapter's core alerting claim in about
sixty lines. The regression is injected at day 31, decaying the score by 0.004
per day under 0.02 noise. The static threshold does not fire until day 51, a
20-day lag, because a slow decay has to travel all the way from the healthy
baseline down to wherever the line was drawn. The rolling drift detector fires
at day 38, a 7-day lag, because it compares the current window against the
frozen reference and pages on the shift itself, not on an absolute level. The
second run makes the sharper point: halve the decay to 0.002 per day and the
static threshold never fires inside the 60-day horizon while the drift detector
still catches it at day 42. This is why [section 5](05-alerting.md) alerts on
rates and deltas rather than levels, and why
[section 4](04-detecting-drift-and-regressions.md) insists the reference window
be reset after each deliberate change: the detector's power comes entirely from
the baseline being right. Swap the simulated scores for the judged-trace
aggregates on your trace stream, the reference window for a matched
hour-of-week baseline, and the print for a pager, and you have rebuilt this
chapter.
