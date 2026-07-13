# 1. Clarifying the requirements

Before designing anything, pin down what the problem actually is. The most common
interview mistake here is hearing "the model is not good enough" and immediately
proposing a training run. The real work is to ask five questions that either kill
the need for fine-tuning entirely or sharply scope what it must do.

Here is a typical exchange.

**Candidate:** When you say the model is not good enough, what does that look like
concretely? Wrong facts, wrong format, wrong tone, or a skill the base model
genuinely lacks?
**Interviewer:** The model handles the domain knowledge fine, but its output format
is inconsistent and its tone does not match our brand voice.

**Candidate:** Do we have labeled data, and who produces it? Because if we have
no labeled examples and no clear path to getting them, the pipeline does not start.
**Interviewer:** We have about four thousand human-labeled (prompt, ideal response)
pairs from the past six months. We can get more if we need them.

**Candidate:** How fast does the domain change? If the facts churn weekly, baking
them into weights is a treadmill.
**Interviewer:** The facts are fairly stable. The brand tone is what needs to be
consistent, and that does not change.

**Candidate:** What are the latency and cost constraints, and do we control the
serving infrastructure or are we on a closed API?
**Interviewer:** We self-host on open weights, so we control everything. Latency
is under two seconds per response.

**Candidate:** What is our quality floor, and how will we measure whether a new
model cleared it?
**Interviewer:** You tell me -- that is part of what we want you to design.

Let us summarize the problem statement. **We are asked to adapt a general-purpose
base model so that its output format is consistent and its tone matches our brand,
given about four thousand labeled examples, a self-hosted open-weights
infrastructure, and a stable domain.** Every candidate model must clear a quality
gate before it reaches a user, and we need to design that gate.

Two consequences fall out immediately, and stating them early is most of the
signal in this question.

First, **this is a behavior problem, not a knowledge problem.** The model already
knows the domain facts; it just does not behave the way we want. That points
directly at supervised fine-tuning (or at better prompt engineering), and rules
out retrieval-augmented generation as the primary fix. The distinction matters
because it tells the interviewer you know what each tool is for.

Second, **fine-tuning is the last lever, not the first.** The honest ordering is:
prompt engineering, retrieval, supervised fine-tuning, preference alignment. We
should try the earlier, cheaper rungs before committing to a training run, and we
should be ready to argue why we skipped them. The next section covers this ladder
in detail.
