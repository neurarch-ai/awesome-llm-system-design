# 10. Putting it together: the complete build

Every earlier section presented options. This one commits: one stack, decided, with
the run costed, then re-derived under three different constraint sets, and closed
with a runnable reference you can execute with nothing but Python 3.

The scenario is the one from [section 1](01-clarifying-requirements.md): choose a
base model for a product family from three candidates (two API models, one
open-weight model we host), while tracking our own fine-tunes on the same protocol,
with the requirement that a 2-point difference be callable.

## The default stack

| Decision | Committed choice | Why, in one line |
|---|---|---|
| Portfolio | 3 public suites with headroom, 1 live suite, 2 agentic suites, 1 private internal set | Capability coverage plus a contamination check plus a decision-maker of record |
| Scoring mode | Generative plus answer matching everywhere; log-likelihood only for base-model training telemetry | Matches how the model is used and works on API candidates |
| Open-ended grading | Per-item rubric criteria, model grader, certified against expert labels | Holistic scores do not reproduce; criteria are debuggable |
| Judged numbers | PPI-corrected with 300 expert labels per candidate | Unbiased regardless of grader quality, at a labeling cost you can size |
| Decode policy | One policy per benchmark applied to all candidates, vendor-recommended settings checked on a subset | Comparability first, with a sanity check that the policy is not penalizing anyone |
| Test-time compute | Two effort levels for the reasoning-capable candidates | Quality is a curve against spend, not a scalar |
| Seeds | 5 on suites under 300 items, 3 on agentic, 1 on suites over 1,000 items | Variance where variance dominates, items where items dominate |
| Statistics | Paired per-item comparison, bootstrap interval on the difference, BH correction across the pre-registered comparisons | The gap, not the scores, is the quantity of interest |
| Contamination evidence | Live-suite window after every candidate's cutoff, plus the internal set | Runnable by anyone, unlike a decontamination claim |
| Artifact | Report card with score, interval, cost, protocol hash, and per-comparison verdict | A number without its protocol is not reproducible |

## The run, costed

Item counts and sample budget per candidate:

| Suite | Items | Seeds | Runs | Kind |
|---|---|---|---|---|
| GPQA Diamond | 198 | 5 | 990 | Short-answer reasoning |
| MMLU-Pro subset | 1,000 | 1 | 1,000 | Broad knowledge |
| LiveCodeBench (post-cutoff window) | 300 | 3 | 900 | Code, contamination check |
| Internal set | 800 | 3 | 2,400 | Rubric-graded open-ended |
| SWE-bench Verified | 500 | 1 | 500 | Agentic, repo environment |
| tau2-bench | 300 | 3 | 900 | Agentic, tool plus user simulation |

Item counts here are the slices this run evaluates, not the full published size of
each suite: the MMLU-Pro row is a sampled subset, the LiveCodeBench row is a
release-date window after every candidate's cutoff, and the agentic rows are the
task sets we chose to run. GPQA Diamond and SWE-bench Verified are run whole.

The cost is dominated by two things people underestimate: agentic episodes, where
the context is re-sent on every tool turn, and the effort curve, which multiplies
the reasoning suites. Using illustrative prices of \$3 per million input tokens and
\$15 per million output tokens:

```text
non-agentic   5,290 runs x (1.5k in + 3k out)   ~=  7.9M in + 15.9M out  ~= $262
SWE-bench       500 episodes x (120k in + 15k out) ~= 60M in +  7.5M out ~= $292
tau2-bench      900 episodes x (40k in + 6k out)   ~= 36M in +  5.4M out ~= $189
                                                       per candidate     ~= $743
3 candidates                                                             ~= $2,230
high-effort curve on 2 candidates (reasoning suites, 3x output)          ~=   $400
rubric grading 7,200 grader calls on a cheap model                       ~=    $30
                                                       total compute     ~= $2,700
```

The other budget line is human labeling: 300 expert judgments per candidate for the
PPI correction, roughly 900 judgments at a few minutes each, which is one to two
expert-weeks and the real constraint on cadence. Wall clock at 30-way parallelism is
a few hours for the non-agentic suites and most of a day for the agentic ones, so
the full grid is an overnight job and the smoke subset is a coffee break.

That ratio, two thousand dollars of compute against two expert-weeks of labeling, is
the reason PPI matters: the cheap resource is model calls and the scarce one is
judgment, so the design should spend the former to stretch the latter.

## The report card

What lands in the document, per benchmark, with the aggregate kept separate:

| Candidate | Internal set (PPI) | 95% CI | GPQA-D | LiveCodeBench | tau2 pass^1 | tau2 pass^3 | Out tokens per item | \$ per 1k items |
|---|---|---|---|---|---|---|---|---|
| A (medium effort) | 0.678 | +/- 0.031 | 0.61 | 0.44 | 0.68 | 0.31 | 3,100 | \$48 |
| A (high effort) | 0.702 | +/- 0.030 | 0.69 | 0.52 | 0.71 | 0.36 | 9,400 | \$142 |
| B (default) | 0.671 | +/- 0.032 | 0.64 | 0.49 | 0.66 | 0.29 | 1,400 | \$23 |
| C (open weight, self-hosted) | 0.639 | +/- 0.033 | 0.55 | 0.41 | 0.58 | 0.20 | 1,900 | \$9 |

Illustrative numbers, but the shape is the point. Read as scores, A-high wins. Read
as the report card: A-high costs six times B for a gap on the internal set whose
paired interval very likely includes zero, C is a third of B's quality gap away at a
fraction of the cost and might win once serving volume is priced in, and every
candidate's pass^3 on the agentic suite is roughly half its pass^1, which is the
number the product team needs to see before promising an autonomous flow.

The verdicts that go with it: **B as the default**, **A-high as an escalation tier
for hard items** (which is the [cost-optimization chapter's](../cost-optimization/)
cascade, now justified with numbers), **C revisited if volume grows**, and **the
A-versus-B internal-set gap reported as not distinguishable** with the item count
that would settle it.

## The same system under three constraint sets

**Frontier lab training its own models.** The bottleneck moves from cost to cadence:
you need a signal per checkpoint, not per quarter. Split into a cheap proxy suite
(a few hundred items, log-likelihood scored, run every few thousand steps) and the
full grid at milestones only. Decontamination becomes real work rather than a claim,
because you own the corpus: n-gram and near-duplicate removal of benchmark items and
their paraphrases, plus the caveat that distillation from an external teacher can
reintroduce leakage you cannot see. Add a sealed slice with a logged query budget,
because with hundreds of checkpoints selection leakage is the dominant risk, not
training-set overlap. Deterministic serving matters more here than anywhere else,
since a training-telemetry curve that moves with batch composition is unreadable.

**Seed-stage startup on API models.** Drop to one seed on the large suites, keep
multiple seeds only where the item count is small, and cut the public portfolio to
two suites plus one live suite. Spend everything you save on the internal set,
because it is the only thing that decides your product, and on 200 human labels for
the PPI correction. Skip self-hosting comparisons until volume justifies them. The
honest posture in a fundraising deck is a protocol hash and an interval next to the
number, which is cheap and immediately distinguishes you from a screenshot of a
leaderboard.

**Regulated or safety-critical domain.** The rubric is written by domain experts and
versioned like code; the grader is certified against expert labels per rubric
version and re-certified on a schedule. A safety suite runs as a separate blocking
gate with its own threshold, paired with a benign set so over-refusal is measured
rather than rewarded. Every run is retained for audit with full provenance
(protocol hash, container digest, raw outputs), and a human sign-off sits after the
automated gate, because the automated pipeline reduces the volume that needs expert
attention rather than eliminating it.

## The smallest runnable experiment

One file, standard library only. It answers the four questions a benchmark report
has to answer, and it reproduces the central lesson of
[section 6](06-statistics-and-leaderboards.md): a 2-point gap on a 500-item
benchmark is not a result.

```python
"""Benchmark-eval statistics on one page. Python 3, standard library only."""

import random
from math import comb, sqrt

random.seed(10)

# Per-item correctness for two candidates on the SAME 500 items (paired by index).
# Items carry a shared difficulty, so the two models agree on most of them; only a
# thin band plus per-run noise separates them. That correlation is exactly what the
# paired analysis exploits and the unpaired one throws away.
N, SKILL_A, SKILL_B, NOISE = 500, 0.72, 0.70, 0.05
difficulty = [random.random() for _ in range(N)]          # shared across candidates


def run(skill):
    return [int((d < skill) != (random.random() < NOISE)) for d in difficulty]


a, b = run(SKILL_A), run(SKILL_B)


def wilson(k, n, z=1.96):
    """95% interval for a proportion. Correct at small n, unlike the normal approx."""
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return center - half, center + half


def mcnemar(x, y):
    """Paired comparison: only items where the two disagree carry information."""
    b_wins = sum(1 for u, v in zip(x, y) if u == 1 and v == 0)   # x right, y wrong
    c_wins = sum(1 for u, v in zip(x, y) if u == 0 and v == 1)   # y right, x wrong
    delta = (b_wins - c_wins) / len(x)
    se = sqrt(b_wins + c_wins) / len(x)
    z = (b_wins - c_wins) / sqrt(b_wins + c_wins) if b_wins + c_wins else 0.0
    return delta, se, z, b_wins, c_wins


def paired_bootstrap(x, y, reps=5000):
    """Resample items (not scores) to get an interval on the difference."""
    n = len(x)
    diffs = []
    for _ in range(reps):
        idx = [random.randrange(n) for _ in range(n)]
        diffs.append(sum(x[i] - y[i] for i in idx) / n)
    diffs.sort()
    return diffs[int(0.025 * reps)], diffs[int(0.975 * reps)]


def items_needed(discordance, delta, z_sum=2.80):
    """Items required to detect `delta` at 5% significance and 80% power, paired."""
    return discordance * (z_sum / delta) ** 2


def pass_hat_k(n, c, k):
    """Reliability: P(all k independent attempts succeed), unbiased from n trials."""
    return 0.0 if c < k else comb(c, k) / comb(n, k)


def ppi(judge_all, judge_labeled, human_labeled):
    """Judge mean, rectified by the judge's measured bias on the human-labeled slice."""
    bias = sum(h - j for h, j in zip(human_labeled, judge_labeled)) / len(human_labeled)
    return sum(judge_all) / len(judge_all) + bias


lo_a, hi_a = wilson(sum(a), N)
lo_b, hi_b = wilson(sum(b), N)
delta, se, z, bw, cw = mcnemar(a, b)
blo, bhi = paired_bootstrap(a, b)

print(f"A = {sum(a)/N:.3f}  95% CI [{lo_a:.3f}, {hi_a:.3f}]   (unpaired)")
print(f"B = {sum(b)/N:.3f}  95% CI [{lo_b:.3f}, {hi_b:.3f}]   (unpaired)")
print(f"paired: delta={delta:+.3f}  se={se:.3f}  mcnemar z={z:.2f}  "
      f"(A-only {bw}, B-only {cw})")
print(f"paired bootstrap 95% CI on the difference: [{blo:+.3f}, {bhi:+.3f}]")
print("verdict:", "distinguishable" if blo > 0 or bhi < 0 else "NOT distinguishable")
d = (bw + cw) / N
print(f"discordance={d:.3f} -> items needed for a 2-point call: "
      f"{items_needed(d, 0.02):,.0f}")

print()
for p in (0.9, 0.68):
    n_trials, c_ok = 100, int(round(100 * p))
    print(f"per-attempt {p:.0%}: pass^3={pass_hat_k(n_trials, c_ok, 3):.2f}  "
          f"pass^8={pass_hat_k(n_trials, c_ok, 8):.2f}")

print()
judged = [1 if random.random() < 0.74 else 0 for _ in range(5000)]   # judge, all items
sub = list(range(300))                                               # human-labeled slice
judge_sub = [judged[i] for i in sub]
# humans are stricter than the judge on ~8% of the items it passed
human_sub = [0 if (j == 1 and random.random() < 0.08) else j for j in judge_sub]
print(f"judge-only estimate : {sum(judged)/len(judged):.3f}   (precise, biased)")
print(f"humans-only (n=300) : {sum(human_sub)/len(human_sub):.3f}   (unbiased, wide)")
print(f"PPI-corrected       : {ppi(judged, judge_sub, human_sub):.3f}   "
      f"(unbiased, narrow)")
```

Output:

```text
A = 0.712  95% CI [0.671, 0.750]   (unpaired)
B = 0.690  95% CI [0.648, 0.729]   (unpaired)
paired: delta=+0.022  se=0.015  mcnemar z=1.46  (A-only 34, B-only 23)
paired bootstrap 95% CI on the difference: [-0.006, +0.052]
verdict: NOT distinguishable
discordance=0.114 -> items needed for a 2-point call: 2,234

per-attempt 90%: pass^3=0.73  pass^8=0.42
per-attempt 68%: pass^3=0.31  pass^8=0.04

judge-only estimate : 0.742   (precise, biased)
humans-only (n=300) : 0.683   (unbiased, wide)
PPI-corrected       : 0.678   (unbiased, narrow)
```

Three things to take from twenty lines of output. **A is 2.2 points ahead and that
is not a result**: the paired bootstrap interval crosses zero, and it would take
roughly 2,200 items to settle a gap that size. **Pairing bought most of the
precision available**: the unpaired intervals are about 4 points wide each, the
paired standard error is 1.5 points. **The judge's number is precise and wrong**:
0.742 against a corrected 0.678, a 6-point bias that 300 honest labels both revealed
and removed. Every one of those is a sentence you can say in an interview, and each
one is the kind of claim that only sounds credible when you can show where the
number came from.
