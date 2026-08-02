# 1. Clarifying the requirements

**Candidate:** What kind of task is this? Is there something that can check the
answer, or is it open-ended?

**Interviewer:** Mixed. A lot of it is code and structured data extraction where we
can run a check. Some of it is free-form explanation.

**Candidate:** That split matters more than anything else here, because thinking
pays off most where a verifier exists. Without one, extra samples cannot be selected
between, and the only lever is a longer single chain.

**Interviewer:** Assume both exist and tell me how you would spend on each.

**Candidate:** What does the latency budget look like, and is it a mean or a tail?
Reasoning models make output length a random variable, so a p50 target and a p99
target are almost different products.

**Interviewer:** Users wait for the answer. We promised a p95 under 30 seconds.

**Candidate:** Does the user see the thinking, or only the answer? Streaming a
progress signal changes the perceived latency budget without changing the real one.

**Interviewer:** We show a "thinking" indicator, not the trace.

**Candidate:** What is the cost ceiling per request, and do we have per-request
outcome data? To choose a policy I need to know whether each request was solved, not
just what it cost.

**Interviewer:** We log cost. Outcomes only for the code path, where tests run.

**Candidate:** Then the first piece of work is instrumenting outcomes on the rest,
because otherwise we can compare policies on price and not on value.

**Interviewer:** Fair. Anything else?

**Candidate:** Two things. Do we control the serving fleet or is this a hosted API?
If we host it, thinking tokens occupy KV slots for a long time and the queueing
behavior becomes our problem. And is quality actually better with thinking **on this
task**? It is not uniformly better, and I would want the cost-matched comparison
before designing around it.

Let us summarize. **We are asked to serve a reasoning model for a mixed workload
(verifiable code and data tasks, plus open-ended explanation) under a p95 latency
promise, with a cost ceiling per request, on a fleet we operate, where outcome data
exists for part of the traffic and must be built for the rest.**

Two consequences fall out immediately.

**Consequence 1: output length is now a random variable, so capacity and SLO
planning move from means to tails.** A non-thinking model produces a few hundred
tokens with modest spread. A thinking model produces a distribution whose mean might
be several thousand tokens and whose tail runs far beyond it. Queueing delay depends
on the *second* moment of service time, not just the first, so a distribution with
the same mean and a fatter tail produces a dramatically worse p99. Everything in
[section 3](03-budgets-and-latency.md) follows from that one fact: budgets,
truncation policy, admission control, and how you size the fleet.

**Consequence 2: thinking is a purchasable quality axis, so the design question is
allocation, not on-off.** Every request can be given a different budget, and the
correct budget depends on how hard the request is and how expensive being wrong is.
That turns the system into a router with a spend dimension, and it makes the
governing metric **cost per solved task** rather than cost per request. A policy
that halves cost per request while dropping the solve rate by a third is worse, and
only the solved-task denominator shows it.

A third point is worth stating because it prevents an expensive mistake:
**reasoning models are not uniformly better.** On recall-style questions, formatting
and extraction, and anything latency-bound, extended thinking buys little and costs
a lot. The honest first experiment is a cost-matched comparison on your own traffic
(see [benchmarking, section 6](../benchmark-eval/06-statistics-and-leaderboards.md)),
because a model given ten times the output tokens should not be compared against one
that was not.
