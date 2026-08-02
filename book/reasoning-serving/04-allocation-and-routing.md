# 4. Allocating the budget

Once thinking is a per-request purchase, the system is a router with a spend
dimension. This section is the arithmetic for spending it.

## The three policies, and when each wins

**Always think.** Every request gets a large budget. Simple, expensive, and it
inflates the latency tail for the majority of requests that never needed it.

**Effort routing.** A classifier assigns each request a budget class before
generation. Cheap, and its ceiling is the classifier's accuracy: requests routed
short that needed long are simply lost quality, and you never find out unless you
measure outcomes.

**Cascade.** Run the cheap path, check the answer, escalate only on failure. It is
the strongest policy when a verifier exists, and it has a property the other two do
not: the escalated request effectively gets two independent attempts, so the solve
rate can exceed the always-think policy at a fraction of the cost.

The arithmetic that decides between routing and cascading, per request:

$$C_{\text{cascade}} = c_{\text{short}} + c_{\text{verify}} + (1 - a)\,c_{\text{long}}, \qquad C_{\text{always}} = c_{\text{long}}$$

where $a$ is the fraction of requests the cheap path answers acceptably. The
cascade wins when

$$a \gt \frac{c_{\text{short}} + c_{\text{verify}}}{c_{\text{long}}}$$

With a short path at roughly a tenth the cost of the long one and a cheap verifier,
that threshold sits near 15 percent: **if the cheap path can handle even one request
in five, cascading beats always-thinking on cost.** The catch is entirely in the
accept test, which the next section is about.

## Deciding difficulty before you have the answer

Effort routing needs a difficulty signal at admission time. Ranked by cost:

| Signal | How | Cost | Caveat |
|---|---|---|---|
| Declared by the caller | The product marks a surface as high-stakes | Free | Callers over-declare; needs a quota |
| Rules over the prompt | Task type, length, presence of code or math | Near free | Brittle, but a strong baseline and fully explainable |
| Small classifier | Trained on logged (prompt, was-it-solved-cheaply) pairs | Cheap | Needs the outcome logging from section 1 |
| Model self-report | Ask the model to rate difficulty first | One extra call | Poorly calibrated; treat as a weak feature |
| Cheap-path attempt | Just try it and check | The cheap path itself | This is the cascade, and it is usually the better answer |

The last row is the punchline: **a cheap attempt plus a verifier is a better
difficulty classifier than any difficulty classifier**, because it measures the
thing you care about (can the cheap path solve it) rather than predicting it. Use
prediction when there is no verifier, or when the escalation latency is
unacceptable.

## Adaptive stopping

Budgets can also be spent adaptively within a request:

- **Stop when the answer stabilizes.** Sample a few short continuations; if they
  agree, stop. This is self-consistency used as a stopping rule rather than a
  voting rule, and it saves the most on the easy majority.
- **Stop when the verifier passes.** For verifiable tasks, generate until a sample
  passes or the budget is exhausted, which converts the budget into an expected
  number of attempts.
- **Force the answer at the boundary.** Covered in
  [section 3](03-budgets-and-latency.md), and it is the difference between a
  degraded answer and a broken one.

Explicit budget control at inference time (forcing more thinking by suppressing the
end-of-thinking token, or forcing less by injecting an answer prompt) is a
documented technique rather than a hack ([s1: Simple test-time
scaling](https://arxiv.org/abs/2501.19393)), and it is what you fall back on when
the provider exposes no budget parameter.

## Model routing composes with budget routing

There are two dimensions, not one: **which model** and **how much thinking**. They
interact, and the common mistake is collapsing them.

| | Small model | Large model |
|---|---|---|
| **No thinking** | Extraction, classification, formatting | Knowledge-heavy single-shot answers |
| **Thinking** | Often the best value: a small reasoning model with a verifier | Hard multi-step problems where being wrong is expensive |

The bottom-left cell is where a lot of production value sits and where interviews
reward you for noticing: a small model given room to think, with a verifier to catch
its mistakes, frequently beats a large model answering immediately, at a fraction of
the cost. The general routing machinery lives in the
[cost-optimization chapter](../cost-optimization/03-routing-and-cascades.md); what
this chapter adds is that the escalation axis is now spend-per-request as well as
model size.

## When to use which

| Reach for | When | Instead of |
|---|---|---|
| Always think | High-stakes, low-volume, no verifier, latency tolerant | Paying the tail on every request in a mixed workload |
| Effort routing by rules | You need something today and have obvious task-type signals | Waiting for a trained classifier before doing anything |
| Effort routing by classifier | Enough traffic and outcome logging to train on | Model self-reported difficulty, which is poorly calibrated |
| Cascade with a verifier | Anything checkable, and the cheap path solves a meaningful share | Effort prediction, which guesses at what the cascade measures |
| Best-of-n with a verifier | Latency budget allows parallel sampling and the verifier is strong | Sequential thinking, when wall-clock matters more than tokens |
| Adaptive stopping | Long tail of easy requests inside a thinking path | A fixed budget applied uniformly |
| No thinking at all | Extraction, formatting, recall, latency-bound UX | A reasoning model applied because it benchmarks better |

**Worked example.** A mixed workload (code fixes, data extraction, explanations)
starts on always-think and is both slow and expensive. The team instruments outcomes
first, because no policy can be compared without them. Extraction moves to a
non-thinking small model with constrained decoding, since deliberation buys nothing
there. Code moves to a cascade: a short attempt, the repository's own tests as the
verifier, escalate on failure, which raises the solve rate above the always-think
baseline (the escalated requests get two attempts) while cutting cost per solved
task by more than half. Explanations have no verifier, so they keep a moderate fixed
budget chosen at the knee of the measured accuracy-versus-budget curve, with a
forced-answer step at the boundary. The classifier that decides which surface goes
where is retrained monthly on logged outcomes, and the whole thing is reported as
cost per solved task rather than cost per request.
