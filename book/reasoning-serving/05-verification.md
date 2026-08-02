# 5. Verification

Parallel sampling raises the chance that *some* attempt is right. A verifier is what
converts that into an answer you can ship. Everything expensive in this chapter gets
cheaper when the verifier gets better, which is why verification, not thinking, is
usually the highest-leverage investment.

## The ladder of verifiers

| Verifier | Signal | Cost | Gameable? | Use when |
|---|---|---|---|---|
| Execution (unit tests, a compiler, a SQL run) | Ground truth on the tested behavior | Sandbox time | Only by overfitting to the tests | Code, queries, anything runnable |
| Symbolic or programmatic check | Exact, for the property checked | Negligible | Rarely | Math answers, schema validity, constraint satisfaction |
| Deterministic self-consistency (majority over canonical answers) | Agreement, not correctness | k samples | It cannot tell confident wrongness from correctness | Short canonical answers with no checker |
| Outcome reward model | A learned score for the final answer | One model call | Yes, and it is the classic reward-hacking target | Open-ended tasks with training data |
| Process reward model (step-level) | A score per reasoning step | Several calls | Yes, and more subtly | Long derivations where the failure is localizable |
| LLM judge with a rubric | A model's opinion | One model call | Yes: verbosity, self-preference, injection | Open-ended, and only after certification |

The ordering is the advice: **prefer a checker you can execute over a model you have
to trust.** Step-level supervision is materially stronger than outcome-only
supervision when you have to use a learned verifier
([Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)), and the general
finding that selection quality caps the value of sampling is what makes coverage
without a verifier an illusion
([Large Language Monkeys](https://arxiv.org/abs/2407.21787)).

## The economics: a verifier is a budget multiplier

With per-attempt success $p$ and $k$ attempts, coverage is
$1-(1-p)^k$, but delivered accuracy is coverage times the probability the selector
picks a correct sample when one exists. Two consequences:

- **A weak selector wastes attempts.** At $p = 0.4$ and $k = 8$, coverage is about
  0.98 while a selector that is right 60 percent of the time delivers around 0.6.
  You paid for eight samples and received the quality of roughly one and a half.
- **Verifier improvements beat sample-count improvements.** Doubling $k$ moves
  coverage along a flattening curve; improving the selector moves the whole product.

This is the same distinction as pass@k versus delivered quality in
[benchmarking, section 5](../benchmark-eval/05-scoring-and-autoraters.md), and it is
worth naming in an interview: **pass@k is what you can buy; the verifier decides how
much of it you get to keep.**

## How verifiers fail

- **Weak tests accept wrong answers.** The agentic-benchmark literature documents
  exactly this failure in evaluation harnesses, where test suites too weak to reject
  incorrect patches inflate scores ([Establishing Best Practices for Building
  Rigorous Agentic Benchmarks](https://arxiv.org/abs/2507.02825)). In serving, the
  same weakness lets a wrong answer through the accept test.
- **Optimization pressure finds the gap.** Best-of-n against a learned reward model
  is optimization *against the verifier*, so samples drift toward whatever the
  verifier rewards rather than what is correct. The larger $k$, the harder the
  pressure.
- **Judges can be attacked through the content.** When the verifier is a model
  reading model output, an instruction embedded in that output is an injection
  vector (see [safety](../safety/)).
- **The verifier is a cost line.** A process reward model scoring every step of a
  10,000-token trace can cost more than the generation it is checking. Account for
  verification tokens explicitly in the cost per solved task.

## Practical accept tests

For a cascade, the accept test does not have to be perfect, it has to be
**conservative in the right direction**. Missing a wrong answer is the expensive
error; escalating an answer that was actually fine costs only the escalation.

| Task | Cheap accept test |
|---|---|
| Code change | Run the repository's tests; accept only on a green run |
| SQL or query generation | Execute against a test database and compare row sets |
| Structured extraction | Schema validation plus a field-level consistency check against the source |
| Math or numeric answer | Recompute symbolically, or check the answer satisfies the stated constraints |
| Free-form explanation with sources | Every claim must map to a retrieved span (a groundedness check) |
| Anything else | A certified rubric judge, with the certification discipline from [benchmarking, section 5](../benchmark-eval/05-scoring-and-autoraters.md) |

The last row carries a warning: an uncertified judge as an accept test converts a
quality problem into a silent quality problem, because the cascade will confidently
accept whatever the judge likes.

## When to use which

| Reach for | When | Instead of |
|---|---|---|
| Execution-based verification | The output is runnable | A model judging whether code looks correct |
| Symbolic or schema checks | The output has a checkable form | Sampling more and hoping |
| Self-consistency voting | Short canonical answers, no checker available | Best-of-n, which needs a selector you do not have |
| Outcome reward model | Open-ended output with training data and modest $k$ | Large-$k$ best-of-n, where optimization pressure breaks it |
| Process reward model | Long derivations, and the budget supports step scoring | Outcome-only scoring on traces where the error is localizable |
| Certified rubric judge | Nothing else applies | An uncertified judge, which makes failures invisible |
| Investing in the verifier instead of tokens | Coverage is high and delivered quality is not | Doubling $k$ again |
