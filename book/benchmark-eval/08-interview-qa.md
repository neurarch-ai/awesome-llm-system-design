# 8. Interview Q&A

The questions actually asked about benchmarking a model, grouped by how they are
used. The traps at the bottom are where this interview is usually lost, because each
one has an answer that sounds right and is wrong.

## Commonly asked

**Q: Walk me through the whole pipeline. How do you evaluate a model on benchmarks?**

A: Eight stages, and the interesting content is in the middle three.

1. **Decide what the number is for.** Model selection, training telemetry, a release
   gate, or an external claim. That decides the required rigor and the audience.
2. **Select a portfolio, not a benchmark.** Capabilities matched to the workload,
   each with headroom, plus at least one private internal set that cannot be
   contaminated and does match the construct.
3. **Pin the items.** Benchmark version, item count, release dates, licence.
4. **Fix the protocol.** Chat template, system prompt, few-shot count and order,
   answer-format instruction, scoring mode, decode parameters, output-token budget,
   samples per item, seeds. This is where most of the variance between published
   numbers comes from, so it is written down and hashed, not assumed.
5. **Run it, keeping everything.** Sharded by item, cached on the full protocol hash,
   sandboxed for code and agents, with tokens and dollars accounted per run, and
   every rendered prompt and raw completion stored.
6. **Score.** Executable checks where the task allows, answer matching for short
   free-form, rubric grading with a certified grader for open-ended, unit tests for
   code, task success plus cost plus reliability for agents. Track parse-failure,
   truncation, and refusal rates as first-class outputs.
7. **Analyze like an experiment.** Paired per-item comparisons, intervals, seed
   variance, clustered errors where items are grouped, multiple-comparison control
   over the grid you actually ran.
8. **Report a card, not a number.** Score, interval, sample and seed counts, cost per
   item, protocol hash, contamination evidence, and a verdict per comparison that is
   allowed to be "not distinguishable."

Then say the thing that frames the whole answer: the pipeline's job is not to
produce a high number, it is to produce a number that survives someone trying to
reproduce it.

**Q: How do you choose which benchmarks to run?**

A: Work backwards from the decision. List the capabilities the product depends on
(say retrieval-heavy reasoning, code edits, tool calling, long inputs, two
languages), pick one benchmark per capability that still has headroom, and drop
anything where all candidates already score above about 90 since the remaining gap
is inside label noise. Add one live or time-gated benchmark as a contamination
check, and one private internal set that is the decision-maker of record. Before
adopting a benchmark, validate it can separate two models you already know differ by
more than its own confidence interval; if it cannot, it will not separate the ones
you are actually comparing.

**Q: Your number does not match the vendor's published number. What happened?**

A: The prior is protocol, not model. I check in this order because that is roughly
the order of magnitude of the effects: chat template and system prompt, few-shot
count and order, scoring mode (log-likelihood ranking versus generative plus
parser), output-token budget and truncation rate, decode parameters and sample
count, then benchmark version and item subset. The maintainers of the LM Evaluation
Harness documented cases where prompt formatting alone moves a model between near
random and competent on the same items, which is larger than any model-generation
gap I would be arguing about. The resolution is not to reconcile the numbers but to
re-run every candidate, including the baseline, under one protocol I control.

**Q: How do you know a 3-point difference is real?**

A: Compute it paired. Both models answered the same items, so the concordant items
cancel and only the discordant ones carry information: with $b$ items A wins and $c$
items B wins, the difference is $(b-c)/n$ with standard error about
$\sqrt{b+c}/n$. On a 200-item benchmark the unpaired interval on each score is
around 7 points, so a 3-point gap from one run is not resolvable, but paired it may
be. If the interval still includes zero, the answer is "not distinguishable at this
sample size," plus the item count that would settle it, which at a 2-point target
and typical discordance is on the order of a couple of thousand items. Report seed
variance too: on small reasoning benchmarks, run-to-run swings exceed the effect
people are claiming.

**Q: How do you evaluate a model with a variable thinking budget?**

A: Treat the budget as part of the candidate, not part of the model. The same model
at low and high effort are two candidates, and quality is a curve against spend, so
I report score against mean output tokens and cost at two or three effort levels.
Two operational points: set the output-token cap from the observed length
distribution and report the truncation rate, because a truncated derivation is
scored as a wrong answer for a reason unrelated to ability; and either apply one
decode policy to every candidate or use each vendor's recommended settings and say
which, since mixing the two silently advantages whichever model got its preferred
configuration.

**Q: How do you handle contamination?**

A: Prevention where I control training, detection where I do not. On the training
side, decontamination is a data-pipeline job: n-gram and near-duplicate removal of
benchmark items and their paraphrases, with the caveat that it cannot catch
distillation from a contaminated teacher. On the evaluation side, since I usually
cannot see someone else's corpus, I rely on black-box evidence: score post-cutoff
items against difficulty-matched pre-cutoff items, run a time-gated benchmark like
LiveBench or a release-date window in LiveCodeBench, and where I can get token
probabilities use membership-inference statistics such as Min-K% against a reference
distribution of text the model could not have seen. The strongest single check is a
functional twin, a benchmark rebuilt to the same specification with new items; a
large drop on the twin is the contamination signature. And I keep a private internal
set, which is the only one where the no-leak story is provable rather than
argued.

**Q: How is evaluating an agent different?**

A: Three differences. The environment becomes part of the measurement, so the
container is pinned by digest, the network policy is stated, and retries are counted
(a suite where 5 percent of items needed a retry is reporting infrastructure, not
capability). Outcome validation has to be strong enough to reject a wrong solution;
audits of widely used agentic benchmarks found test suites weak enough to accept
incorrect patches and success criteria that can score inaction as success, so I
manually audit a sample of passing trajectories before publishing anything. And the
metric changes: a single success rate hides both cost and variance, so I report task
success plus steps and dollars, and reliability as pass^k, the probability that all
$k$ independent attempts succeed, rather than pass@k, which is coverage under
retries.

## Tricky (the follow-ups that separate people)

**Q: Our fine-tune gained 6 points on the target benchmark and users say it got
worse. What happened?**

A: Most likely one of three, and they are distinguishable. **Format overfitting**:
the fine-tune learned the benchmark's answer shape rather than the capability, which
shows up as a gain that vanishes when the same construct is asked free-form instead
of multiple choice. **Selection leakage**: the benchmark was used to choose the
checkpoint and the hyperparameters, so the reported number is a maximum over many
looks rather than an unbiased estimate; the tell is how many times the set was
queried during development. **Narrow gain with broad regression**: the benchmark
moved and everything else quietly dropped, which is why a regression portfolio
including instruction following, refusal behavior, and the internal set runs on every
checkpoint. The confirming experiment is cheap: rebuild the same construct in a
different format, evaluate on a held-out slice that was never queried, and diff the
full portfolio rather than the headline.

**Deeper:** These three have different fixes and only one of them is a model
problem. Format overfitting is fixed in the data mix, selection leakage is fixed by
process (a sealed slice with a query budget), and a broad regression is fixed by
training. Diagnosing which you have before proposing a fix is most of the value.

**Q: We are grading an open-ended benchmark with a model. How do you make that
number trustworthy?**

A: Certify, then correct, and do not skip the second step. Certification is a
meta-eval set of a few hundred expert-labeled pairs oversampled near the decision
boundary, reporting agreement and swap consistency, plus adversarial probes: empty
answers, constant answers, padded answers, and answers containing an instruction
aimed at the grader. Correction is statistical: keep the cheap grader on all $N$
items, keep human labels on a small $n$, and use prediction-powered inference, which
adds the measured judge bias back to the judge's mean and is unbiased no matter how
poor the judge is. That reframes the budget question from "can we label everything"
to "how many labels buy an acceptable interval." Also report the grader's own error
rate, because no comparison finer than that rate is supportable.

**Deeper:** The reason certification alone is not enough: agreement statistics tell
you the judge is wrong at some rate, but a gate needs an unbiased estimate of the
quantity, not a characterization of the instrument. PPI converts the instrument's
measured error into a correction term, which is why a mediocre judge plus 200 honest
labels can beat a great judge with none.

**Q: Same model, same prompt, temperature zero, and two runs disagree. Why?**

A: Greedy decoding is deterministic in principle and not in a serving system.
Kernels for normalization, matmul, and attention are not batch-invariant, so the
floating-point reduction order depends on what else was batched with your request,
and a single token flip early in a derivation changes everything after it. Other
candidates for the same symptom: the provider silently updated the model behind an
alias, a different serving engine or quantization, or a nondeterministic parser or
sandbox. The remedies in order: pin revisions and engine versions, use
batch-invariant kernels if reproducibility is a requirement and accept the
throughput cost, or accept run variance and measure it, which you need for the error
bars anyway. What you must not do is report a single-run number as exact.

**Q: How would you build an internal benchmark that is still useful in two years?**

A: Design for refresh and for a budget of looks. Source items continuously from
production traffic and from experts, timestamp every item, and retire items that
every candidate solves (they cost compute and carry no information). Split into a
public-internal portion people iterate against and a sealed portion with a query
budget, where every evaluation against the sealed set is logged and a checkpoint
chosen on it consumes a look. Store items with per-item metadata for slicing
(language, length, domain, difficulty), keep a rubric rather than a single reference
answer for open-ended items so the grading criteria are debuggable, and re-certify
the grader on a schedule. The failure mode this design targets is not going stale in
content, it is going stale in difficulty, which is why difficulty-targeted item
addition matters more than volume.

**Q: The eval grid is 12 candidates by 15 benchmarks by 5 seeds. Make it affordable.**

A: Four levers, roughly in order of savings. **Stage it**: a cheap smoke subset (a
few hundred items across benchmarks) ranks candidates coarsely, and only the
survivors get the full grid, which usually cuts the grid by half or more. **Cache on
the protocol hash** so re-runs after a scoring or parser change re-score stored
outputs instead of regenerating them, which is the single biggest saver during
iteration. **Right-size the sample counts per benchmark from the statistics**: small
benchmarks need many seeds and large ones need few, so a flat 5 seeds everywhere
overspends on the large suites and underspends on the small ones. **Shard by item
across providers and GPUs** to convert cost pressure into wall-clock pressure. And
report what you cut: if the safety suite ran at 200 of 2,000 items, that belongs in
the report, because a silently truncated suite reads as full coverage.

**Q: Which do you trust when the public benchmarks and the internal benchmark
disagree?**

A: The internal one, and I can say why in one sentence: it matches the construct we
care about and it is the only set I can prove was not trained on. The public
disagreement is still information, though. If the model wins publicly and loses
internally, the likely causes are distribution mismatch (our prompts, our documents,
our tools) or contamination inflating the public number. If it loses publicly and
wins internally, the model may be specialized in a way that suits us, which is a
reason to check whether our internal set is too narrow. Either way the internal set
decides the deployment and the public numbers stay as the external comparability
story.

## Commonly answered wrong (the traps)

**Q: We removed exact duplicates of the benchmark from the training data, so
contamination is handled. True?**

A: No. Exact-match deduplication catches the easiest case and misses the four that
matter: near-duplicates and paraphrases, translations and reformatted mirrors,
format leakage where the model learned the test's answer style without seeing its
items, and distillation from a teacher that was itself contaminated. It also does
nothing about selection leakage, where nothing leaked into training but the test set
was consulted hundreds of times while choosing checkpoints. A defensible answer
pairs a decontamination claim with its parameters (n-gram size, normalization,
near-duplicate threshold) and, more importantly, with black-box evidence:
performance on post-cutoff items, on a functional twin, or on a private set.

**Deeper:** The asymmetry is what makes this trap dangerous. Decontamination
evidence is a claim about a corpus only the trainer can inspect, while the
time-split and functional-twin tests are runnable by anyone. In an interview, the
person who reaches for the runnable test is demonstrating that they know the claim
is unverifiable by the reader.

**Q: MMLU and HumanEval are the standard, so report those. Right?**

A: They were the standard, and at the frontier they are saturated, which means the
gap between candidates is inside the benchmark's own label-error and sampling noise.
Multiple-choice formats carry a second problem: a nontrivial fraction of items can
be answered above chance without seeing the question at all, so the format partly
measures option discrimination rather than the ability to produce an answer. Re-run
them if you need comparability with published numbers, but the decision should rest
on benchmarks with headroom, a free-form scored version of the same construct, and
your internal set. And publish the no-question baseline for any multiple-choice set
you rely on.

**Q: The agent solves 90 percent of tasks, so users will see it work nine times out
of ten. Right?**

A: Only if pass@1 was measured the way the user experiences it, and usually it is
not. pass@k with $k \gt 1$ measures coverage: the chance that at least one of $k$
attempts succeeds, which is the right metric when a verifier or a human picks the
winner and the wrong one for a user with a single shot. Reliability is pass^k, all
$k$ attempts succeeding, and it decays geometrically: a 90 percent per-attempt rate
is about 43 percent at $k = 8$. If the reported number was a best-of-k or came from
a suite with weak outcome validation, the real per-attempt rate is lower still.

**Deeper:** The two metrics also imply different products. A high pass@k with a low
pass^k argues for a verifier-in-the-loop design, retries plus a checker, while a
system that cannot verify its own output needs the pass^k number to be the one on
the slide.

**Q: We ran greedy decoding once per item, so the number is deterministic and
comparable. Right?**

A: Neither, usually. Greedy is not reproducible in a batched serving system because
kernels are not batch-invariant, and it is not comparable across candidates when
vendors recommend different decode settings. A single greedy run also gives you no
variance estimate, so you cannot say anything about whether a gap is real. The
minimum honest protocol is multiple seeds with reported spread, or a demonstrated
reproduction of the exact number on a re-run.

**Q: Average all the benchmark scores into one number so leadership has a single
metric.**

A: An index is legitimate, and a naive average is not. Benchmarks have different
chance floors (25 percent on a 4-option set, 0 percent free-form) and different
ceilings, so averaging raw percentages weights by scale rather than importance;
saturated components contribute noise; and an unweighted mean is still a weighting,
just an unexamined one. Build the index properly: normalize, state the weights,
drop saturated components, and bootstrap over items to put an interval on the index
and on the *rank*, which is what people actually read. Then keep the per-benchmark
table next to it, because the index cannot tell you which capability moved.

**Q: Use the strongest available model as the judge; it will be the most accurate.**

A: Strongest is not the same as certified. Judge-specific benchmarks exist because
strong general models are mediocre judges on objectively checkable pairs where the
wrong answer sounds better, and a strong judge from the same family as a candidate
brings self-preference bias. The right procedure is to measure several candidate
judges against a human-labeled meta-eval set, pick the cheapest that clears the
agreement bar, probe it with degenerate inputs to make sure a constant or padded
answer cannot score well, and then correct its remaining bias statistically with a
small human-labeled sample rather than assuming it away.

**Q: Model A scores higher across the board, so we should build on A.**

A: Higher on what, at what cost, and with what interval. Three checks before that
becomes a decision: are the gaps outside the paired confidence intervals, or is this
a tie; was the comparison cost-matched, because a candidate given five times the
output tokens should be compared on the frontier rather than on the score; and does
the benchmark portfolio match our workload, since a model that wins on competition
math and loses on our internal retrieval set is the wrong choice for a retrieval
product. Benchmarks filter candidates; the internal set and the online loop decide.

**Q: Our fine-tune beats the published base-model number, so it is an improvement.**

A: Not established, because those two numbers came from different pipelines. The
published number used the vendor's prompt, few-shot setting, token budget, and
parser, usually chosen favorably, and possibly a different benchmark version. The
only valid comparison re-runs the base model under exactly your protocol, on the
same items, at the same time, and then compares paired per item. Nine times out of
ten the re-run baseline lands somewhere other than the published figure, and the
sign of your improvement is not always preserved.

**Deeper:** This trap also generalizes to comparing across time within your own
team. Benchmarks get patched, harness defaults change, providers update models
behind aliases, so a number from six months ago is a different protocol even if the
config file looks the same. Baselines are re-run, not remembered, and the protocol
hash is what tells you whether a re-run was needed.
