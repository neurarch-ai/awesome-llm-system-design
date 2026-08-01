# 1. Clarifying the requirements

Before running a single benchmark, pin down what the number is *for*. Here is a
typical exchange. Every question either removes work or changes the protocol.

**Candidate:** What decision does this number drive? Picking a base model, tracking
our own training runs, gating a release, or publishing an external claim?

**Interviewer:** Two of those. We are choosing a base model for a product family,
and we continue-pretrain and fine-tune our own variants, so we need to track whether
each training run actually improved anything. Some of the numbers end up on a model
card.

**Candidate:** Then I need to know who the audience is for each. An internal
selection number can be noisy and cheap; a model-card number has to be reproducible
by someone outside the company.

**Interviewer:** Correct, and assume someone will try to reproduce it.

**Candidate:** Which capabilities actually matter for the product? Benchmark
selection is where most of the error comes from, and a portfolio that does not match
the workload measures the wrong thing precisely.

**Interviewer:** Reasoning over technical documents, code generation, tool calling
with a few internal APIs, and long inputs. Multilingual matters for two markets.

**Candidate:** Are the candidates open-weight, API-only, or both? That decides
whether I can score by log-likelihood, whether I can pin a revision, and whether I
control the serving stack.

**Interviewer:** Mixed. Two API models and one open-weight model we host.

**Candidate:** Do the candidates have variable test-time compute, a reasoning or
thinking mode with a budget I can turn up?

**Interviewer:** Yes, two of them do.

**Candidate:** Then quality is not a scalar; it is a curve against spend. I will
report score against tokens and dollars, not score alone, otherwise the comparison
silently rewards whichever model I let think longest.

**Interviewer:** Fine. What does this cost us?

**Candidate:** Depends on the item count, the sample count per item, and the number
of seeds. I will size it from the smallest difference you need to detect rather than
picking a round number.

**Interviewer:** We need to be able to call a 2-point difference.

**Candidate:** That constrains the design more than anything else you have said.

Let us summarize the problem statement. **We are asked to design a benchmark
evaluation pipeline that ranks a mixed set of candidate models (API and
open-weight, some with variable test-time compute) on a capability portfolio
matched to a product workload, tracks our own training runs against the same
protocol, and produces numbers defensible enough to publish.** The pipeline must
resolve differences of about 2 points, control for contamination, and report cost
alongside quality.

Two consequences fall out immediately, and stating them early accounts for most of
the signal in this question.

**Consequence 1: the score is an estimate produced by a protocol, so the protocol
is part of the result.** The same model on the same benchmark moves by more than a
model generation depending on prompt format, few-shot count, scoring mode, answer
parser, decoding parameters, and output-token budget. The reported artifact is
therefore never a bare number; it is a number, an interval, and a protocol hash
that pins the harness commit, prompt template, model revision, decode parameters,
and sample count. If a colleague cannot re-derive the number from that record, you
did not measure anything, you generated one.

**Consequence 2: published numbers are not comparable to yours, so you re-run every
baseline yourself under one protocol.** A vendor's model card number was produced by
a different harness with a different prompt and a different token budget, usually
tuned favorably. Comparing your fine-tune against their published baseline is the
single most common way a benchmark report gets an internal decision wrong. The rule
is: one harness, one protocol, all candidates, including the baseline you think you
already know.

A third point is worth stating in the interview even though it is a scoping
decision rather than a consequence: **a benchmark measures a model, not your
product.** It filters candidates and tracks training. It does not gate a feature,
because it does not run your prompts, your retrieval, your tools, or your users.
The gate lives in the [evaluation chapter](../evaluation/), and a strong answer
names that boundary before the interviewer has to.
