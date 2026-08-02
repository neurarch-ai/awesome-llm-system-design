# 8. Interview Q&A

## Commonly asked

**Q: We switched to a reasoning model, quality is up, p99 latency is unpredictable
and cost tripled. Walk me through what you do.**

A: Three moves, in order. **Instrument outcomes first**: log tokens, wall clock,
whether a verifier accepted, and whether the task was ultimately solved, because
every policy comparison from here is on cost per solved task and I cannot compute it
otherwise. **Bound the tail**: a hard thinking budget per request with a
forced-answer step at the boundary (never a silent truncation), separate queues by
budget class so short requests do not wait behind long ones, and admission control
that downgrades rather than queues under overload. **Then allocate**: extraction and
formatting go to a non-thinking path, verifiable tasks go to a cascade (cheap
attempt, run the checker, escalate on failure), and unverifiable tasks keep a fixed
budget set at the knee of the measured accuracy-versus-budget curve. The headline
report is cost per solved task and p99, as a pair.

**Q: Why does p99 get so much worse than the mean when you turn on thinking?**

A: Because queueing delay depends on the second moment of service time, not the
first. For a queue at utilization $\rho$, the waiting term scales with
$\frac{\rho}{1-\rho}\cdot\frac{E[S]\,(1+C^2)}{2}$, where $C^2$ is the squared
coefficient of variation. Thinking makes the output-length distribution long-tailed,
so $C^2$ rises, and the tail rises with it even at unchanged mean service time.
There is a second, structural effect on top: long generations hold KV slots, so the
effective batch collapses and short requests queue behind them, which is
head-of-line blocking. The fixes are correspondingly two: run at lower utilization
and cap budgets to shrink the tail, and separate queues or predict length so short
requests are not stuck behind long ones.

**Q: Sequential or parallel test-time compute?**

A: Sequential (one longer chain) needs no extra machinery and costs latency,
because tokens are produced serially. Parallel (k samples) costs tokens and not much
latency, since samples run concurrently, but it is useless without a way to pick the
winner: repeated sampling raises the chance that *some* sample is right, and only a
verifier converts that into an answer you can ship. So the answer is driven by
whether a checker exists. Verifiable task, latency-sensitive: parallel with an
executor. No verifier: sequential with a measured budget. And the mix should be
adaptive by difficulty rather than fixed, which is the finding from the test-time
scaling literature.

**Q: How do you set the thinking budget?**

A: Measure, do not guess. Run the accuracy-versus-budget curve on your own traffic
at several budget levels, find where it flattens, and set the cap at the knee. Then
watch two operational numbers: the fraction of requests hitting the cap (rising
means the distribution shifted or the cap is too tight) and the truncation behavior
at the cap (it must be a forced answer, not a cut-off trace). Different task classes
get different budgets, because the knee is task-dependent: competition-style math
keeps improving far longer than extraction does.

**Q: What is the right way to compare a reasoning model against a non-reasoning
one?**

A: Cost-matched, or at least cost-annotated. A model given ten times the output
tokens should not be compared on score alone, so the comparison is a frontier:
score against tokens and dollars per task, with latency reported alongside. Two
budget settings of the same model are two candidates. And the comparison should be
paired per item on your own data with intervals, because reasoning benchmarks are
small and noisy enough that single-run gaps are frequently not real (see
[benchmarking](../benchmark-eval/06-statistics-and-leaderboards.md)).

## Tricky

**Q: Your cascade escalates 20 percent of requests. Traffic shifts and it starts
escalating 60 percent. What happens and what do you do?**

A: The long path saturates, and because the long path is the high-variance one, the
p99 for everyone degrades fast: this is a compound failure where a quality signal
(the cheap path failing more) turns into a capacity incident. Immediate mitigation
is to cap the escalation rate, which means some requests knowingly get the cheap
answer, and to shed or downgrade at the admission layer rather than queue. Then
diagnose: did the input distribution change, did the cheap model change underneath
you, or did the verifier get stricter? The structural fix is that escalation should
be a budgeted resource with its own quota, not an unbounded consequence of a
verifier verdict.

**Deeper:** This is why the escalation rate belongs on a dashboard next to latency.
It is simultaneously the earliest quality signal you have (the cheap path is failing
more) and a leading indicator of a capacity problem, and most teams discover it as
the second rather than the first.

**Q: Best-of-16 with a reward model gives worse answers than best-of-4. How?**

A: Optimization pressure against an imperfect verifier. Selecting the argmax of a
learned reward over 16 samples searches harder for whatever the reward model
overvalues, so beyond some k you are increasingly selecting reward-model artifacts
rather than correct answers. The diagnostics: check whether the selected sample's
reward is rising while human or executor agreement is falling, and compare against a
random pick from the same samples. The fixes are a stronger or executable verifier,
a KL-style penalty against the base policy's likelihood, ensembling verifiers, or
simply capping k where the delivered quality curve peaks rather than where coverage
does.

**Q: How do you schedule when you do not know how long a request will run?**

A: You cannot do shortest-job-first without job lengths, so you either approximate
them or avoid needing them. Approximate: a cheap classifier over the prompt predicts
a length class, which restores an approximate priority order and doubles as the
effort-routing signal. Avoid: partition by budget class into separate queues, so
each queue has low internal variance and short requests never sit behind long ones,
at some cost in fleet efficiency. At larger scale, add preemption so a long
generation can be paused when the queue backs up. In an interview, naming that FIFO
is optimal for nothing in this setting and that the fix is either prediction or
partitioning is the answer they are looking for.

**Q: The product wants to show the reasoning trace. Systems view?**

A: Streaming a progress signal is close to free and materially improves perceived
latency; streaming the raw trace is a different decision with three consequences.
Safety: traces contain exploratory content, including reasoning about things the
final answer correctly refuses, so it needs its own moderation pass. Competitive:
published traces are the raw material for distilling your behavior into another
model. Product: users read the trace as a promise, and a trace that contradicts the
answer is worse than no trace. The middle path most teams take is a summarized or
step-labelled trace rather than the raw one, which keeps the latency benefit without
the exposure.

## Commonly answered wrong

**Q: Reasoning models are better, so we should route everything through one.**

A: Not uniformly. Thinking helps where the model can find and *recognize* a better
answer than its first one. On factual recall it adds nothing (the knowledge is in
the weights or it is not, so the fix is retrieval), on extraction and formatting it
adds drift and cost, and on latency-bound interactive surfaces the tail is the
product. It also makes evaluation misleading: a reasoning model wins benchmarks
partly because it spent more, which is why the comparison must be cost-matched. The
defensible position is a mixed workload where the majority path does not think and
the budget is spent where it changes outcomes.

**Q: We sample 8 candidates and take the best, so quality should be near the pass@8
number.**

A: Only with a selector that good. pass@8 measures whether *any* of the 8 is
correct; delivered quality is that coverage multiplied by how often your selection
picks a correct one when it exists. With a mediocre selector, most of what you paid
for evaporates: coverage near 0.98 with a selector that is right 60 percent of the
time delivers roughly 0.6. That gap is the argument for investing in the verifier
rather than in more samples, and for preferring an executor over a model-based
judge wherever the output is runnable.

**Deeper:** The same distinction explains why teams with unit tests get so much more
out of test-time compute than teams without: the executor is a near-perfect selector,
so their delivered quality tracks coverage, while everyone else is bounded by their
judge.

**Q: We cap the thinking budget, so the tail is handled.**

A: A cap bounds the tail of *service time*; it does not bound queueing delay, which
is driven by variance and utilization, and it does not fix head-of-line blocking. A
fleet running at high utilization with capped-but-variable service times still
misses its p99, because the $\rho/(1-\rho)$ factor dominates. The cap is one of
three controls, alongside running at a lower target utilization and partitioning
queues by budget class. And a cap without a forced-answer step converts latency
outliers into malformed answers, which is a worse failure than a slow one.

**Q: Just add GPUs until the p99 is fine.**

A: You can buy your way down the $\rho/(1-\rho)$ curve, but it is the most expensive
lever available, because the variance term means you have to run substantially below
saturation to move the tail. Every other lever is cheaper: budget caps shrink the
tail directly, budget-class queues remove the blocking, routing removes the requests
that never needed to think, and a cascade removes the long path from the majority of
traffic entirely. Capacity is the right answer only after those, and the capstone's
numbers show why: policy changes moved cost per solved task by more than a factor of
two on the same fleet.

**Q: Cost per request went down, so the change was a win.**

A: Only if the solve rate held. Cost per request is trivially minimized by failing
faster, which is exactly what an over-aggressive effort router does: it sends hard
requests down the cheap path, the bill drops, and users retry or leave. The
governing metric is cost per solved task, with latency reported alongside, and it
requires per-request outcome logging. A policy that halves cost per request while
dropping the solve rate by a third is worse on the metric that matters, and the
capstone shows a case where the cheapest-per-request policy is the worst product.
