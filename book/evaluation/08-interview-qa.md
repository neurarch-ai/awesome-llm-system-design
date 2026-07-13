# 8. Interview Q&A

The questions an interviewer actually asks about LLM evaluation systems, grouped
by how they are used. The commonly-missed ones are where interviews are won or lost.

## Commonly asked

**Q: How do you evaluate an LLM output when there is no single correct answer?**

A: Two approaches, used together. First, try to reframe part of the task as
checkable: instead of judging a free-text answer, ask the model to output a
structured field you can compare exactly (a category, a JSON blob, a resolved
entity). Use a task metric there. For the genuinely open-ended dimensions
(helpfulness, tone, faithfulness to context), use an LLM-as-judge with a sharp
rubric, but validate it first against human labels and report judge-human agreement
(Cohen's kappa) before gating anything on it. The combination of task metrics
where possible and a validated judge where necessary is the honest answer; neither
alone is complete.

**Q: What is the difference between an offline eval and an online eval, and why do
you need both?**

A: The offline eval runs a candidate (prompt plus model) against a fixed golden
dataset and produces a score. It is fast, cheap, and repeatable. The online eval
routes real traffic to the candidate and measures behavioral signals: did users
complete their task, did they edit the output, did the session continue. You need
both because offline has no access to the signals that actually reflect user value
(task completion, retention, cost), and online has no way to catch a regression
before it ships. The offline gate is the guardrail; the online measurement is the
ground truth check that tells you whether the guardrail is calibrated correctly.

**Q: How do you wire the eval into the deploy path so a regression cannot ship by
accident?**

A: Treat the offline suite like unit tests: run it automatically on any change to
a prompt, model identifier, or inference config, fail the build if any segment
drops more than a tolerance below baseline, and gate per slice rather than just on
the average. A model swap is a versioned artifact change and goes through the same
gate. Only after the gate passes does the candidate reach a canary; only after the
canary does it reach full rollout. The key wiring points are the CI trigger (so it
runs without anyone remembering to run it) and the per-slice gate (so a segment
regression cannot hide behind an improving average).

**Q: How do you know your eval is any good?**

A: Online-offline correlation. Run the offline suite, let the candidate reach an
online A/B, and compare the offline verdict to the online outcome. If they agree
consistently, the suite earns trust. If they disagree (offline predicts a win but
the A/B shows a loss, or vice versa), the suite is measuring the wrong thing and
you fix the suite, not the tolerance. Spotify names this explicitly: the gap
between offline eval and A/B outcome is the primary calibration signal. A suite
that has not been validated against at least a few online outcomes is not a gate;
it is a guess.

**Q: How do you evaluate a RAG system?**

A: Separately for retrieval and generation, because a RAG answer can fail for two
completely different reasons and you cannot tell which half to fix if you score
only the final answer. Retrieval quality: did the right documents come back? Measure
recall and precision at k against labeled relevant documents. Answer groundedness
(faithfulness): given the retrieved context, is each claim in the answer supported
by it? This is a judge task. Answer relevance: does the answer address the question?
Also a judge task, scored separately from groundedness. Splitting the three lets
you localize a failure: low retrieval recall means you never had the evidence; low
groundedness with high retrieval recall means the model hallucinated from good
context; low relevance with high groundedness means it answered a different question
faithfully.

## Tricky (the follow-ups that separate people)

**Q: Your offline score went up but your online A/B showed a regression. What
happened, and what do you do?**

A: The most common causes: the judge has verbosity bias and is rewarding the
longer, more confident-sounding candidate that users actually find less helpful;
the golden set is stale and does not cover the query patterns where the regression
lives; or the score is averaged in a way that hides the regressing segment. Diagnose
by checking which segments regressed online and whether they were covered in the
golden set, then check whether the judge preferentially scored longer outputs. The
fix is not to adjust the tolerance; it is to recalibrate the suite to match what
the online loop is telling you about real user value.

**Q: A teammate says "let us just check quality manually before every release."
What is wrong with that?**

A: Three things. Manual review does not scale when prompts change daily across a
team. It is not repeatable: two reviewers looking at the same output on different
days may score it differently. And it does not gate per slice; a reviewer eyeballing
examples will miss a regression buried in one language or customer tier. Manual
review is valuable for calibrating the judge and for the uncertain canary cases
where the automated gate is too close to call. It is not a substitute for a wired-
in automated gate that runs on every change and gates per segment.

**Q: Why is position bias a problem in pairwise judgment, and is running both
orderings actually enough to fix it?**

A: Position bias is a systematic preference in the judge for whichever answer is
shown first (sometimes last). It distorts the win-rate estimate by a fixed offset
in one direction. Running both orderings and averaging cancels the offset because
the bias is symmetric across the two presentations; averaging subtracts it out.
The averaging fix is valid under the assumption that the bias is consistent across
the pair, which holds for most model judges on most rubrics. If you detect
residual bias after averaging (for example, by checking swap consistency on a
matched set of known-equal outputs), the rubric may be the problem and needs
revision.

**Q: How do you handle judge drift? The hosted judge model updated silently and
yesterday's scores are not comparable to today's.**

A: Three defenses. First, pin the judge model version explicitly; treat it as a
versioned dependency the way you pin a library version. Second, version the judge
prompt separately from the candidate prompt; a judge-prompt change is as
disruptive as a judge model change. Third, maintain a fixed calibration set of
(input, output) pairs with known scores, and re-score them periodically; a score
shift on the calibration set with no change to the candidate signals judge drift.
When you detect drift, do not retroactively compare scores across the boundary;
re-baseline against the new judge version before the next gate.

## Commonly answered wrong (the traps)

**Q: Can I use a public benchmark like MMLU or HumanEval as the quality gate for
my feature?**

A: No. Public benchmarks measure general capability at the model level, and they
are subject to contamination: if training data included near-duplicates of benchmark
cases, the model looks good on the benchmark and fails in production. They are
useful as a coarse capability filter when selecting a base model, but not as the
quality gate for a specific feature. Your gate needs a private, freshly-sampled
golden set from your own task and traffic, versioned and held out from prompt
iteration.

**Q: Should the LLM judge be the same model as the one you are evaluating, to
keep everything consistent?**

A: No. Self-preference bias is a documented failure mode: a model systematically
prefers outputs from its own family. Using the evaluated model as its own judge
inflates its scores and is equivalent to not having a judge. Use a different model
family as judge where possible, measure its kappa against human labels, and
document the cross-family agreement. If cross-family options are limited, at minimum
validate the within-family judge more carefully against human labels on the specific
rubric before trusting it.

**Q: If my eval score is high, do I still need an online A/B test?**

A: Usually yes, at least to validate the gate the first few times. Offline scores
are a prediction, not proof. The judge is structurally blind to task completion,
user edit rate, session length, cost, and any signal that requires a real user.
The first time you run a new offline suite, you do not know whether its verdicts
predict online outcomes. You earn the right to ship on the offline gate alone only
after the gate has demonstrated it reliably predicts A/B outcomes in the past. Even
then, major changes (new model family, large capability shift) should still reach
an online canary before full rollout.

**Q: Set the gate tolerance to zero: any regression at all blocks the deploy.**

A: Judge noise means a tolerance of zero will flap the gate constantly. The judge
is a noisy instrument; the same input and output will score slightly differently
across runs due to model temperature, slight prompt sensitivity, and the judge
model's own variance. A zero tolerance treats that noise as a real regression and
blocks valid changes. Set the tolerance from the judge's measured score variance
on identical inputs (the standard deviation of repeated scores on the same output),
typically one to two percentage points, so the gate blocks real regressions and
passes through noise.
