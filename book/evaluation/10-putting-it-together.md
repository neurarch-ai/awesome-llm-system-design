# 10. Putting it together: the complete build

Sections 1 through 6 taught each stage with its options and tradeoffs; section 7
showed where real teams diverge. What none of them show is a single eval system
with every decision made. This capstone does three things: it gives you an
opinionated default stack so option paralysis never blocks a first build, it
walks the chapter's scenario end to end with every choice committed and costed,
and it shows how the same decisions flip when the constraints change. It closes
with the smallest runnable judge experiment, one file, no installs.

## The default stack: start here, deviate with reason

Every stage in this chapter has several credible options, and a first-time
builder can burn a week comparing harnesses before gating a single change. Skip
that. The stack below is a sane default for a first production eval system; each
row names when to deviate and which section explains why. Tools change yearly,
but the interface of each stage (build the golden set, score, judge, calibrate,
gate, confirm online, recalibrate) does not, so pick per stage by interface and
treat any specific harness as replaceable.

| Stage | Default | Deviate when | Why (section) |
|---|---|---|---|
| Golden dataset | ~1,000 versioned rows: common paths, known-hard cases, one row per fixed bug; a held-out slice never tuned against | Day one: start with a few hundred well-chosen rows and grow from production traffic | [3](03-offline-eval.md) |
| Labels | Human annotation as the anchor; production signals (thumbs, edits) nominate refresh candidates | Rare or adversarial slices are uncovered: add synthetic rows, calibrated to human labels | [3](03-offline-eval.md) |
| Task metrics | Exact match, F1, or pass-fail wherever the answer is checkable; reframe tasks to expose checkable signals | Never. Use them first; the judge covers only what they cannot | [3](03-offline-eval.md) |
| Judge mode | Pairwise candidate vs production baseline, both orderings averaged | You need a per-dimension breakdown: add a pointwise rubric per axis | [4](04-llm-as-judge.md) |
| Judge calibration | Kappa vs human labels above bar; different model family; pinned model and prompt version | Kappa below bar: fix the rubric, do not gate yet | [4](04-llm-as-judge.md) |
| Regression gate | Automatic on every prompt, model, or config change; gate the worst slice; tolerance from measured judge sigma | Never gate on the average alone | [5](05-online-eval.md) |
| Safety gate | Separate binary policy-compliance set with its own threshold | Never merged into the capability score | [3](03-offline-eval.md) |
| Cost control | Cache unchanged (input, output, judge-version) triples; 50-row smoke subset locally, full suite at the gate | Cadence is low (solo project, weekly changes): run the full suite every time | [6](06-serving-and-scaling.md) |
| Online proof | Canary first, then A/B on behavioral metrics with latency, cost, and refusal guardrails | Output cannot be shown yet: shadow mode; regulated output: human expert A/B | [5](05-online-eval.md) |
| Feedback loop | On every offline-online disagreement, recalibrate the suite, not the tolerance | Never | [5](05-online-eval.md), [2](02-frame-the-eval.md) |

The first row is the one beginners skip and regret: without a versioned golden
set, every prompt edit and model swap is a vibe, and you cannot tell whether a
change helped. One afternoon of labeling pays for itself the first time a
teammate proposes a "drop-in" model upgrade.

## The complete build

Return to the scenario from [section 1](01-clarifying-requirements.md): a
customer-facing assistant answering product questions, open-ended output with no
exact-match metric, daily prompt edits plus roughly monthly model upgrades,
hallucinated product facts as the worst failure, automated gates on every change
with human review reserved for the uncertain cases, and a budget of a few
hundred dollars per candidate evaluation. Here is the whole system with every
choice committed and the reason it won.

| Decision | Choice | Why it won |
|---|---|---|
| Golden set | 1,000 rows sampled from production, versioned in source control, 15% held out (Illustrative split) | Coverage over volume; the held-out slice is what separates measurement from optimization theater |
| Slices | Language, query type, customer tier | A change that lifts the average while tanking one language must still block |
| Checkable slice | Product-fact claims reframed as extractable fields scored by exact match | A task metric is free and unfoolable; the judge is reserved for what it cannot check |
| Judge mode | Pairwise candidate vs production baseline, both orderings averaged | Relative judgment is more reliable than a 1-to-10 scale; averaging cancels position bias |
| Dimension scoring | Pointwise rubric on accuracy and helpfulness separately | The scenario names both; they can regress independently and a single winner hides which moved |
| Judge calibration | Different model family, pinned version, kappa vs human labels above 0.6 before gating | An uncalibrated judge is a second opinion, not an instrument |
| Gate | Worst-slice delta, tolerance set at measured judge sigma (1 to 2 points) | A guessed tolerance flaps; a zero tolerance blocks on noise |
| Safety | Binary adversarial set, separate gate | A more capable but less safe candidate must still block |
| Harness | Parallel workers, result cache, 50-row smoke subset for local iteration | The gate must finish in minutes or engineers route around it |
| Online | Internal canary, then A/B on task completion, edit rate, thumbs; latency, cost, refusal guardrails | Offline is structurally blind to these signals; they only exist on real traffic |
| Human review | Near-threshold verdicts and monthly model upgrades only | Automation handles the volume; humans handle the judgment calls |

**Suite size.** 1,000 rows across three slice axes gives roughly 80 rows per
segment at a dozen segments (Illustrative). That matters because the gate fires
on the worst slice: a segment much below ~50 rows has judge noise wider than
the tolerance, so it can flap the gate on its own. Size the golden set from the
smallest slice you must protect, not from the total.

**Judge cost per gate.** 1,000 rows judged pairwise in both orderings is
roughly 2,000 judge calls per candidate ([section 6](06-serving-and-scaling.md)).
At a few hundred tokens per judgment and ten cents per thousand tokens, a full
gate run lands near $80 (Illustrative), inside the few-hundred-dollar
per-candidate budget from [section 1](01-clarifying-requirements.md). The smoke
subset is 50 rows, about 100 calls and $4 per local iteration (Illustrative).
The cache is what makes daily cadence survivable: outputs unchanged since the
last run are never re-judged, so an iterating engineer pays only for the rows
their edit actually moved.

**Gate latency.** A serially-executed 1,000-row suite at two seconds per call
takes over thirty minutes; at thirty-way parallelism it takes about one minute
([section 6](06-serving-and-scaling.md)). One minute wires into CI like a test
suite. Thirty minutes gets skipped, and a skipped gate is not a gate.

**Statistical power online.** A preference is a proportion, so the 95%
confidence interval is $\hat{p} \pm 1.96\sqrt{p(1-p)/n}$, and the interval must
exclude 0.5 to declare a winner. At a true win rate of 54% that takes roughly
2,400 comparisons ([section 5](05-online-eval.md)); halving the effect size
roughly quadruples the sample ([section 6](06-serving-and-scaling.md)). This
arithmetic is why the offline suite exists at all: A/B slots are expensive and
slow, so the cheap offline gate filters weak candidates first and the A/B
budget goes only to likely winners.

**What breaks in month one.** Three failure modes dominate early operations, so
wire their signals before launch: judge drift (re-score a fixed calibration set
of stored (input, output) pairs on a schedule; a score shift with no candidate
change means the instrument moved, per [section 4](04-llm-as-judge.md)),
offline-online disagreement (log the offline verdict next to every canary and
A/B outcome; repeated disagreement means the suite measures the wrong thing and
must be recalibrated, per [section 5](05-online-eval.md)), and gate flapping
(the same candidate blocking and passing across reruns means the tolerance sits
below the judge's noise floor; re-measure sigma on identical inputs instead of
loosening the gate by hand).

## The same techniques under different constraints

The review question that matters in practice is not "which judge is best" but
"which eval stack is right under my constraints." Here is the same system built
three times. Only the middle column is the build above; the other two keep the
identical stage interfaces and swap nearly every implementation choice.

| | Coding assistant (checkable task) | Customer assistant (this chapter) | Regulated legal drafting |
|---|---|---|---|
| Output checkability | High: code either passes the tests or it does not | Low: open-ended answers, no reference | Low, and the cost of an error is severe |
| Golden set | Thousands of cases plus broken-repo fixtures with executable CI | 1,000 versioned rows sliced by language, query type, tier | Hundreds of expert-authored cases; slow to grow, every row costly |
| Scoring | Unit-test pass rate and pass@k for almost everything; judge only for the chat-explanation slice | Task metric on the checkable fact slice; validated pairwise judge for the rest | Semi-automated task eval; the judge is advisory, never the final word |
| Judge calibration | Light: the judge covers a minority slice | Kappa above bar vs human labels, pinned version, drift re-scoring | Calibrated, but subordinate to subject-matter-expert review |
| Online proof | Internal canary (dogfood) plus daily regression vs production | Canary, then A/B on completion, edit rate, thumbs | Human expert A/B as the final arbiter; no live experimentation on clients |
| Gate cadence | Every commit, CI-shaped | Every prompt or model change, minutes per run | Per release; human sign-off dominates the timeline |
| What would be over-engineering | A pairwise judge on rows a unit test already scores | Human sign-off on every daily prompt edit | High-cadence automation; the bottleneck is expert review, not compute |

Two lessons fall out. First, the coding-assistant column is mostly deletions:
when the task is checkable, the judge, its calibration, its drift monitoring,
and its bias corrections all shrink to a corner of the system, and the eval
bill collapses because a test run is free where a judge call is not. This is
the strongest form of the chapter's rule: use task metrics wherever the task
allows. Second, the legal column shows the gate's trust anchor moving: the
automated suite still runs, but it filters candidates for human experts rather
than replacing them, cadence drops from daily to per-release, and the online
loop is human preference instead of live traffic, because an A/B on regulated
output is not an available instrument.

## What each constraint decides

The compressed decision guide. Read the left column off your requirements; the
right columns say which lever it moves before you compare any tools.

| Your constraint | Lever it moves | Rule of thumb |
|---|---|---|
| Output checkability | Scoring instrument | Checkable: task metric, no judge to calibrate. Open-ended: validated judge, and reframe every checkable sub-part out of its scope |
| Change cadence | Cost controls | Daily edits: cache, smoke subset, cheapest judge that clears kappa. Monthly upgrades: full suite every time is fine |
| Eval budget | Judge size and suite depth | Measure kappa for several judges and pick the cheapest above bar; both orderings doubles calls, so budget for it |
| Traffic volume | Online instrument | High traffic: A/B. Low traffic: internal canary. None yet: offline gate now, validate it against online outcomes when traffic arrives |
| Blast radius | Human involvement | Regulated or irreversible: human sign-off gate. Otherwise: humans only for near-threshold verdicts |
| Smallest slice you must protect | Golden set size | Size from the smallest slice, not the total; a slice thinner than the judge noise floor cannot be gated |
| Effect size to detect | Online sample size | Sample grows with the inverse square of the effect; halve the effect, quadruple the comparisons |
| Judge-human agreement | Gate trustworthiness | Kappa below bar: fix the rubric. Never widen the tolerance to excuse a bad instrument |

## The smallest runnable judge experiment

The review of every eval framework tutorial is the same: the reader assembles a
harness and still cannot see why the protocol matters. So here is the chapter's
core measurement problem in one file with zero installs. Every production
component is swapped for the smallest thing with the same interface: the LLM
judge becomes a seeded coin flip with a known quality signal and a deliberate
first-position preference, the eval suite becomes a loop, and the significance
test becomes a percentile bootstrap. The shape is the lesson; sections 4 and 5
of this chapter upgrade one function each of this file.

```python
"""A biased pairwise judge, the swap fix, and why small eval sets lie. Stdlib only."""
import random

random.seed(7)
TRUE_WIN = 0.54          # candidate A's true win rate over baseline B
POSITION_BIAS = 0.12     # fixed extra preference for whichever answer is shown first

def judge(first, second):
    """Simulated pairwise judge: sees the true quality signal plus a fixed
    first-position preference. Production: one LLM call with a rubric."""
    p_first = (TRUE_WIN if first == "A" else 1 - TRUE_WIN) + POSITION_BIAS
    return first if random.random() < p_first else second

def win_rate_a_first(n):
    """Naive protocol: A always shown first. The bias lands entirely on A."""
    return sum(judge("A", "B") == "A" for _ in range(n)) / n

def paired_verdicts(n):
    """Debiased protocol: judge each pair in both orderings and average,
    so every pair yields 1, 0.5, or 0 wins for A and the offset cancels."""
    return [((judge("A", "B") == "A") + (judge("B", "A") == "A")) / 2
            for _ in range(n)]

def bootstrap_ci(verdicts, iters=2000, alpha=0.05):
    """Percentile bootstrap CI on the mean win rate. Same math on real verdicts."""
    n = len(verdicts)
    means = sorted(sum(random.choices(verdicts, k=n)) / n for _ in range(iters))
    return means[int(alpha / 2 * iters)], means[int((1 - alpha / 2) * iters) - 1]

# -- part 1: position bias, measured and cancelled --------------------------
N = 5000
print(f"true win rate of A over B:           {TRUE_WIN:.3f}")
print(f"judged with A always first (n={N}): {win_rate_a_first(N):.3f}")
swapped = paired_verdicts(N)
print(f"both orderings, averaged   (n={N}): {sum(swapped) / N:.3f}")

# -- part 2: how many pairs before the gate can call a 54% winner? ----------
print("\npairs   estimate   95% bootstrap CI    verdict")
for n in (100, 400, 2400):
    v = paired_verdicts(n)
    est = sum(v) / n
    lo, hi = bootstrap_ci(v)
    call = "A wins" if lo > 0.5 else ("B wins" if hi < 0.5 else "cannot separate")
    print(f"{n:5d}   {est:.3f}      [{lo:.3f}, {hi:.3f}]    {call}")
```

Run it and the output demonstrates the chapter's two protocol claims in about
forty lines. Part 1: with A always shown first, the judge reports a 0.659 win
rate for a candidate whose true rate is 0.540, a fabricated eleven-point lift
that would sail through a naive gate; running both orderings and averaging
recovers 0.533, within sampling noise of the truth, which is exactly the
order-swap fix from [section 4](04-llm-as-judge.md). Part 2: at 100 debiased
pairs the bootstrap interval is [0.465, 0.590] and at 400 pairs [0.497, 0.565],
both spanning 0.5, so the suite cannot separate a genuinely better candidate
from a tie; at 2,400 pairs the interval is [0.529, 0.557] and A finally wins,
matching the roughly 2,400 comparisons that [section 5](05-online-eval.md)
derives for a 54% effect. Swap the coin flip for a real judge call, the loop
for your golden set, and the printed verdict for a per-slice gate, and you have
rebuilt this chapter.
