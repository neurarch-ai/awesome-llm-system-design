# 5. Scoring and autoraters

Scoring is where a benchmark decides what counts as right. The three questions are:
what the metric is, who applies it, and how you know the applier is trustworthy.

## Pick the metric from the task, not from habit

| Task type | Default metric | Why | Failure mode to watch |
|---|---|---|---|
| Closed-form short answer (math, extraction, entity) | Free-form generation plus equivalence-aware answer matching | Matches how the model is used; avoids option-pattern shortcuts | Equivalence bugs ("1/2" vs "0.5", units, LaTeX); measure the parse-failure rate |
| Multiple choice (legacy comparability only) | Accuracy, with the normalization stated | Comparable to published numbers | Saturation, option-order sensitivity, answerable-without-the-question items |
| Code generation | Unit-test pass, reported as pass@k | Executable, unfoolable by surface plausibility | Weak tests accept wrong code; flaky tests; container drift |
| Agent trajectories | Task success plus steps, tokens, dollars, and a reliability metric | A single success rate hides both cost and variance | Inaction counted as success; partial credit inflating scores |
| Open-ended generation with expert criteria | Rubric grading against per-item criteria | Decomposes a fuzzy judgment into checkable claims | Rubric quality and grader agreement become the bottleneck |
| Open-ended generation, relative | Pairwise preference by a certified judge | Relative judgments are more stable than absolute ones | Position, verbosity, and self-preference bias (see [evaluation chapter section 4](../evaluation/04-llm-as-judge.md)) |
| Long-context tasks | Task-specific accuracy at each context length, not one aggregate | Degradation is length-dependent and non-monotonic | Needle-style retrieval saturating while real aggregation fails ([RULER](https://arxiv.org/abs/2404.06654), [HELMET](https://arxiv.org/abs/2410.02694)) |
| Instruction following | Programmatically verifiable constraints | No judge needed, so no judge bias | Only covers constraints you can verify ([IFEval](https://arxiv.org/abs/2311.07911)) |

## Answer matching: the default for closed-form tasks

Give the model the question without options, let it generate, then decide whether
the generated answer is equivalent to the reference. Equivalence is the hard part
and it is a pipeline component with its own error rate:

1. **Normalize** (strip formatting, canonicalize units, parse LaTeX).
2. **Symbolic check** where the domain allows it (a computer algebra comparison for
   math, running the extracted SQL for a query task).
3. **Model-based fallback** only for what steps 1 and 2 cannot decide, using a small
   model with the reference answer in the prompt.
4. **Audit** a sample of both accepted and rejected answers by hand, and report the
   matcher's own error rate.

That fourth step is what separates a measurement from a guess. If the matcher
disagrees with a human on 3 percent of items, no comparison finer than 3 points is
supportable, and you should say so out loud when reporting.

## pass@k and pass^k measure opposite things

Both draw $k$ samples per task; they answer different questions, and confusing them
is one of the most common technical mistakes in agent evaluation.

**pass@k** is coverage: the probability that *at least one* of $k$ samples is
correct. It is the right metric when something downstream can verify and select, a
test suite, a compiler, or a human reviewer. The unbiased estimator over $n \ge k$
drawn samples of which $c$ pass is in the [evaluation chapter](../evaluation/03-offline-eval.md).

**pass^k** is reliability: the probability that *all* $k$ independent trials
succeed, introduced with tau-bench for tool-agent-user interaction
([tau-bench](https://arxiv.org/abs/2406.12045)). It is the right metric when the
user gets one attempt and a failure is a failure.

$$\text{pass}^k = p^k \qquad\text{so}\qquad p = 0.9 \implies \text{pass}^8 \approx 0.43$$

A 90 percent success rate looks strong and is not: run the same task eight times and
the agent gets all eight right less than half the time. The unbiased estimator from
$n$ samples with $c$ successes mirrors the pass@k one:

```python
from math import comb
def pass_hat_k(n, c, k):      # n trials drawn, c successes, reliability at k
    if c < k:                 # fewer than k successes -> no k-subset is all-success
        return 0.0
    return comb(c, k) / comb(n, k)     # P(all k sampled trials succeeded)
# pass_hat_k(n=10, c=9, k=8) -> 0.2  (a 90% agent is "always right" 8 times in 10 rarely)
```

Report both when the deployment allows retries and the reliability one when it does
not. Naming this distinction unprompted is a strong signal in an agent-eval
interview.

## Rubric grading: how open-ended tasks got measurable

The state of the art for expert open-ended tasks is per-item, expert-written rubric
criteria rather than a holistic 1-to-10 score. OpenAI's HealthBench is the
reference design: physician-written criteria attached to each conversation, each
carrying a point value (positive for behaviors that should be rewarded, negative for
ones that should be penalized), with a model grader deciding criterion by criterion
whether the response meets it ([HealthBench](https://arxiv.org/abs/2505.08775)).

$$s(\text{response}) = \frac{\sum_{c \in C} w_c \cdot \mathbb{1}[\text{criterion } c \text{ met}]}{\sum_{c \in C, w_c \gt 0} w_c}$$

Why this beats a holistic score: each criterion is a small, checkable, nearly binary
claim, which is the kind of judgment models and humans agree on. The rubric is also
an artifact you can debug, version, and hand to a domain expert. The cost is
building it, which is annotation work that does not scale by prompt engineering.

The same idea appears as checklists in agent evaluation and as rubric-conditioned
reward models in post-training; the design lesson transfers: **decompose the
judgment until each piece is checkable, then aggregate with stated weights.**

## Certifying the autorater

Once a model grades your benchmark, the grader is part of the measurement
instrument, and an uncertified instrument produces numbers of unknown bias. The
certification steps, in order:

1. **Build a meta-eval set.** A few hundred (item, response) pairs with expert
   labels, deliberately oversampled near the decision boundary rather than on easy
   extremes.
2. **Measure agreement**, and report it: exact agreement, Cohen's kappa, and, for
   pairwise judges, swap consistency. HealthBench's own design includes physician
   meta-evaluation of the model grader for exactly this reason.
3. **Benchmark the judge against known-hard cases.** [JudgeBench](https://arxiv.org/abs/2410.12784)
   exists because strong general models are mediocre judges on objectively-checkable
   pairs where the wrong answer sounds better.
4. **Probe for gaming.** Run degenerate baselines through the grader: an empty
   answer, a constant answer, a long padded answer, and an answer containing an
   instruction aimed at the grader. Any of these scoring well is a blocking defect,
   and constant-output "null models" have been shown to win non-trivial rates on
   automatic judge benchmarks.
5. **Pin and re-certify.** Judge model version and judge prompt are versioned
   dependencies; re-score a frozen calibration set on a schedule to detect drift.

If agreement is below bar, fix the rubric before you touch anything else. A
sharpened criterion buys more agreement than a bigger judge model, and it is
cheaper.

## Correcting the judge statistically instead of trusting it

Certification tells you the judge is biased; it does not remove the bias. The modern
answer is to keep the cheap judge on the full set, keep a small human-labeled
subset, and use the subset to *correct* the judge's estimate. This is
prediction-powered inference (PPI), and it is the single most useful recent
technique in this chapter.

Let $f(X_i)$ be the judge's score on item $i$, available for all $N$ items, and
$Y_j$ the human label on the $n \ll N$ items that were labeled by both. The
rectified estimator is:

$$\hat{\theta}_{\text{PPI}} = \underbrace{\frac{1}{N}\sum_{i=1}^{N} f(X_i)}_{\text{cheap judge estimate}} + \underbrace{\frac{1}{n}\sum_{j=1}^{n}\bigl(Y_j - f(X_j)\bigr)}_{\text{measured judge bias}}$$

The first term uses every judged item; the second term is an unbiased estimate of
the judge's systematic error. The result is unbiased regardless of how bad the judge
is, and its variance falls as the judge gets more accurate, so a better judge buys
tighter intervals rather than a different answer. Stratifying the human-labeled
sample across slices tightens it further ([Stratified Prediction-Powered Inference
for Hybrid Language Model Evaluation](https://arxiv.org/abs/2406.04291)); the same
correction can be framed through the judge's sensitivity and specificity, with
confidence intervals that account for uncertainty in both the test and the
calibration set ([How to Correctly Report LLM-as-a-Judge
Evaluations](https://arxiv.org/abs/2511.21140)).

```python
from statistics import mean
def ppi_estimate(judge_all, judge_labeled, human_labeled):
    # judge_all: judge scores on every item (length N)
    # judge_labeled / human_labeled: aligned scores on the human-labeled subset (length n)
    bias = mean(h - j for h, j in zip(human_labeled, judge_labeled))  # systematic judge error
    return mean(judge_all) + bias                                    # rectified, unbiased
# judge says 0.72 overall; on 200 human-labeled items it runs 0.05 high -> corrected 0.67
```

Two consequences worth saying out loud in an interview. First, this **changes the
labeling budget question** from "can we afford to label everything" to "how many
labels buy an acceptable interval," which is a sizing calculation rather than a
funding fight. Second, it **removes the excuse for an uncalibrated judge**: you no
longer need the judge to be unbiased, you need a few hundred honest labels and the
discipline to keep collecting them as the distribution moves.

## When to use which scoring approach

| Reach for | When | Instead of |
|---|---|---|
| Executable checks (tests, symbolic equivalence, verifiable constraints) | The task admits a machine-checkable answer | A judge you would then have to certify and maintain |
| Answer matching with a small matcher model | Short free-form answers where surface form varies | Multiple choice, which measures a narrower construct |
| Expert rubric plus certified model grader | Open-ended expert tasks where holistic scores do not reproduce | A 1-to-10 judge score with no rubric and no meta-eval |
| Pairwise preference by a certified judge | Ranking two candidates on open-ended quality | Absolute scores that drift across rubric versions |
| PPI-corrected judge estimate | Any judged benchmark where the number will be reported | Raw judge means, which carry the judge's bias into the headline |
| pass^k plus cost | User-facing agent reliability | pass@k alone, which measures coverage under retries |
