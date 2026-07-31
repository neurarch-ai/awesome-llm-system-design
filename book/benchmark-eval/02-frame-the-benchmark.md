# 2. Framing the benchmark

## What a benchmark score actually is

A benchmark score is three commitments, and naming them is the fastest way to sound
like someone who has run evals rather than read about them.

- **A construct.** The latent ability you claim to measure ("can it do competition
  math", "can it resolve a real GitHub issue"). The construct is never observed
  directly.
- **An item population.** The finite sample of tasks you drew as a proxy for the
  construct. The score is a sample statistic, so it carries sampling error, and the
  population may not represent the construct (competition math is not engineering
  math).
- **A protocol.** How each item is turned into a prompt, how the model is run, and
  how the output is turned into a number. Two protocols on the same items and the
  same model give different scores, routinely by 10 points or more.

Most benchmark arguments are really disagreements about one of the three while both
sides talk about the number. "Is this benchmark saturated" is an item-population
argument; "why is my MMLU 8 points below yours" is almost always a protocol
argument; "does SWE-bench predict engineering productivity" is a construct argument.

## Model-level vs system-level evaluation

| Dimension | Benchmark eval (this chapter) | System eval ([evaluation chapter](../evaluation/)) |
|---|---|---|
| Unit under test | A model, at a pinned revision | A candidate: prompt plus model plus config plus retrieval plus tools |
| Item source | Public or internal benchmark suites, fixed | Golden set sampled from your own traffic |
| What it decides | Which model to build on; whether a training run improved | Whether this change ships today |
| Comparability target | Other people's published numbers, other checkpoints | Your own production baseline |
| Dominant risk | Contamination, protocol drift, construct mismatch | Judge miscalibration, slice regressions, stale golden set |
| Cadence | Per checkpoint, per candidate model, per quarter | Per commit, per prompt edit |

They share machinery (harness, judges, statistics) and answer different questions.
The failure that costs the most is using one where the other belongs: gating a
product change on MMLU, or picking a base model on a 200-row product golden set that
only exercises one prompt.

## The capability portfolio

Pick benchmarks the way you pick a test suite: by what each one would catch if it
broke, not by what is famous. As of 2026, the 2021 to 2023 multiple-choice suites
(MMLU, HellaSwag, GSM8K, HumanEval) are saturated at the frontier, which means their
remaining signal is mostly label noise, and the meaningful families are these.

| Capability | Benchmark families | What it really measures | How it breaks |
|---|---|---|---|
| Knowledge and reasoning | GPQA Diamond, MMLU-Pro, Humanity's Last Exam | Recall plus multi-step inference on expert questions | Small item counts (GPQA Diamond is 198 items) make error bars wide; contamination from public mirrors |
| Math | AIME-style competition sets, FrontierMath | Symbolic multi-step derivation under a format constraint | 30-item sets: one item is over 3 points; answer-equivalence parsing errors |
| Code (unit-test verified) | SWE-bench Verified, LiveCodeBench, HumanEval-style pass@k | Producing a patch or function that passes tests | Weak test suites accept wrong patches; harness and container drift; contamination on old problems |
| Agentic and tool use | tau-bench and tau2-bench, BFCL, GAIA, terminal and computer-use suites | Multi-step tool calls under a policy, with state | Inaction scored as success; partial credit hiding failures; environment flakiness |
| Long context | RULER, HELMET, multi-round coreference retrieval sets | Retrieval and aggregation across a long input, not just a needle | Near-perfect needle-in-a-haystack scores that do not transfer to real tasks |
| Instruction following | IFEval-style verifiable-constraint sets | Obeying explicit output constraints | Easy to overfit with formatting-focused post-training |
| Multimodal | MMMU-Pro and successors | Cross-modal reasoning, not caption matching | Text-only shortcuts solve many items |
| Safety and refusal | Jailbreak and policy suites, adversarial red-team sets | Behavior under adversarial pressure | Over-refusal is not measured by the same set; needs a paired benign set |
| Economically valuable work | GDPval-style expert-task sets, METR time-horizon suites | Whether long, realistic tasks complete at all | Expensive; small n; human baselining required |

Two selection rules do most of the work. **Prefer benchmarks with headroom**: a
benchmark where every candidate scores above 90 percent cannot rank them, because
the remaining gap is inside the label-error rate. And **always include at least one
private internal benchmark** built from your own workload, because it is the only
set you can prove was never trained on and the only one whose construct matches the
product.

## Saturation, headroom, and discriminative power

Two models at 94.1 and 94.6 on a saturated benchmark are not 0.5 apart; they are
tied inside label noise. A benchmark's usefulness for *ranking* depends on the
spread of item difficulty relative to the models under test: items that every
candidate solves and items that no candidate solves both contribute variance and no
information. That is the same intuition item-response theory formalizes, and it is
why frontier labs keep retiring benchmarks rather than reporting them forever.

Practical test before adopting a benchmark: run two models you already know differ
and check whether the benchmark separates them by more than its own confidence
interval. If it cannot separate a known pair, it will not separate an unknown one.

## Construct validity: does the score mean what you will use it for

The score can be perfectly measured and still not support the decision.

- **Multiple-choice shortcuts.** Many multiple-choice items can be answered above
  chance without the question at all, from option patterns alone, so the format
  measures discrimination among given options rather than the ability to produce an
  answer. Recent work finds free-form generation plus reference matching a better
  measure of the same construct ([Answer Matching Outperforms Multiple Choice for
  Language Model Evaluation](https://arxiv.org/abs/2507.02856)).
- **Item validity.** Benchmarks contain broken items. A manual re-annotation of
  thousands of MMLU questions found a single-digit-percent error rate across
  subjects (bad parsing, multiple correct options, no correct option, missing
  context), released as MMLU-Redux ([Are We Done with MMLU?](https://arxiv.org/abs/2406.04127)),
  and validity audits of adversarially filtered commonsense sets found a large
  fraction of items with grammatical or semantic problems ([What the HellaSwag? On
  the Validity of Common-Sense Reasoning Benchmarks](https://arxiv.org/abs/2504.07825)).
  Label errors cap the maximum achievable score and add noise exactly where models
  are separated.
- **Agentic overestimation.** Weak outcome validation lets an agent "pass" without
  solving the task. A systematic audit found SWE-bench-Verified style setups can
  overestimate skill because test suites are too weak to reject wrong patches, and
  that lax criteria in tau-bench can count inaction as success ([Establishing Best
  Practices for Building Rigorous Agentic Benchmarks](https://arxiv.org/abs/2507.02825)).
- **Population mismatch.** Competition math predicts competition math. If your
  product is retrieval over technical documents, a math score is a proxy for a proxy.

## Compare and contrast: five instruments people all call "the benchmark"

| Instrument | What it gives you | What it cannot give you | Cost |
|---|---|---|---|
| Static public benchmark (GPQA, MMLU-Pro) | Comparability with published numbers; cheap, repeatable | Freedom from contamination; construct match to your task | Low |
| Live or time-gated benchmark (LiveBench, LiveCodeBench) | Items released after the model's cutoff, so contamination is bounded by construction | Stable cross-time comparability; the item pool changes under you | Low to medium |
| Private internal benchmark | Construct match plus a provable no-leak story | External comparability; a public number to cite | Medium (annotation) |
| Human preference arena | Aggregate human taste at scale, on real prompts | Task-level correctness; immunity to style and best-of-N submission effects | Medium (traffic) |
| Human uplift or expert trial | Whether the model changes real work outcomes | Throughput; you cannot run it per checkpoint | High |

A complete answer runs a portfolio: two or three public benchmarks with headroom for
external comparability, one live benchmark as a contamination check, and one private
internal benchmark as the decision-maker of record.

## Inputs and outputs of the pipeline

**Input:** a set of candidates, where a candidate is (model identifier, revision or
snapshot, serving stack, decode configuration, test-time-compute budget). Two budget
settings of the same model are two candidates, not one.

**Input per run:** a pinned benchmark version, including item release dates where the
benchmark has them.

**Output:** for each (candidate, benchmark, slice): a point estimate, an interval, a
sample count, a cost in tokens and dollars, and a protocol hash. Plus a verdict on
each pairwise comparison you were asked to make: better, worse, or not
distinguishable at this sample size. "Not distinguishable" is a legitimate and
frequently correct answer, and being willing to give it is a senior signal.
