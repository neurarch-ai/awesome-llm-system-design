# 1. Clarifying the requirements

Before designing anything, pin down what the system must do. Here is a typical
exchange between a candidate and an interviewer. Notice that every question either
removes work or changes the design.

**Candidate:** What kind of output is the LLM producing? A closed answer with a
checkable correct response, or something open-ended like a summary or a chat reply?

**Interviewer:** Primarily open-ended. It is a customer-facing assistant that
answers product questions. There are no single correct answers.

**Candidate:** Is there existing human judgment we can compare against, or are we
starting fresh?

**Interviewer:** We have some thumbs-up/thumbs-down signals from users in
production, and we run occasional human annotation studies, but nothing systematic.

**Candidate:** How often does the prompt or model change? Daily prompt edits,
or more like quarterly model upgrades?

**Interviewer:** Both. The product team edits prompts constantly. We also plan to
upgrade the base model roughly monthly.

**Candidate:** What does a failure cost? A bad answer in a legal pipeline is very
different from a slightly worse tone in a chatbot.

**Interviewer:** Unhelpful answers erode trust and drive users away. Hallucinated
product facts are worse. We care most about factual accuracy and helpfulness.

**Candidate:** Do we need to catch regressions automatically, or is manual review
acceptable?

**Interviewer:** Manual review cannot keep up. We need automated gates on every
change, with human review reserved for the uncertain cases.

**Candidate:** What is the budget for running the eval itself? Every judged example
costs a model call.

**Interviewer:** Reasonable cost. We can spend a few hundred dollars per candidate
evaluation, but not thousands.

Let us summarize the problem statement. **We are asked to design an evaluation
system for a customer-facing LLM assistant.** The output is open-ended natural
language; no exact-match metric (a check that the output equals a reference string
exactly) exists. The system must automatically gate every
prompt or model change before it ships, catch regressions per segment, run cheaply
enough to trigger on every change, and distinguish factual accuracy from general
helpfulness. Human review is reserved for the uncertain cases; everything else must
clear an automated bar.

Two consequences fall out of this immediately, and stating them early accounts for
most of the signal in this question.

**Consequence 1: we cannot use exact-match metrics alone.** The output is
open-ended. That forces us to either design a task metric (reframe part of the
problem as checkable) or rely on a judge model for the genuinely subjective
dimensions. A judge introduces calibration overhead: a judge you have not validated
against human labels is just a second opinion you have no reason to believe. That
validation step is not optional; it is the thing that separates a trustworthy eval
system from a false sense of measurement.

**Consequence 2: the gate must be automatic and per-slice.** A single aggregate
score hides regressions. A change that lifts the average while tanking one
language, one customer tier, or one query type can still ship. The gate must fire
on the worst segment, not the mean, and it must fire automatically on every change
the way unit tests fire on every commit.
