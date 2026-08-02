# A mock interview, end to end

Every chapter's Q&A is a set of isolated answers. A real loop is 45 continuous
minutes where the interviewer decides what to probe based on what you just said.
This is one full transcript, with the interviewer's private scoring notes in
quoted blocks, so you can see what each turn was actually being graded on.

The question is a common senior one because it starts as a design and turns into
a measurement problem: **"We shipped a customer support assistant on top of our
docs. Leadership wants to upgrade to a newer base model next month. Design the
system that tells us whether to do it."**

Read it once as the candidate, then once as the interviewer.

---

## Minutes 0 to 6: clarify, and get the frame right

**Interviewer:** We have a support assistant running on our docs. There is a newer
base model out. How would you decide whether to switch?

**Candidate:** Let me scope it first. Is the question "is the new model better in
general" or "is our assistant better with the new model in it"? Those are different
systems and I would build them differently.

**Interviewer:** Say both. We also publish some numbers internally.

**Candidate:** Then I will build two things and keep them separate: a **model-level
benchmark pipeline** that ranks candidate base models, and a **system-level eval**
that gates our actual assistant. The benchmark pipeline filters candidates and the
system eval decides the launch. What I want to avoid is gating a product change on a
public benchmark, because those measure general capability on someone else's task
distribution and they can be contaminated.

> **Interviewer note.** The first thing being graded is whether the candidate
> separates model evaluation from system evaluation. Roughly half of candidates
> collapse them and spend the rest of the interview designing one thing that does
> neither job. Strong signal: naming the boundary unprompted and saying which one
> gates the launch.

**Candidate:** A few scoping questions. What does a failure cost? A wrong answer
about a refund policy is different from a slightly worse tone.

**Interviewer:** Wrong policy answers are the expensive ones. We have had two
escalations from them.

**Candidate:** Then groundedness is a first-class metric, not a subjective one, and
I will want it separable from general helpfulness. Second: do we have traffic logs
with outcomes, or only transcripts?

**Interviewer:** Transcripts, thumbs up and down, and whether the ticket was later
escalated to a human.

**Candidate:** Escalation is the best signal in that list, because it is behavioral
and it correlates with the failure you care about. Third: how often does the prompt
change? If the product team edits prompts weekly, the eval has to run on every
change, which puts a hard ceiling on what it can cost.

**Interviewer:** Weekly at least.

**Candidate:** Last one: are we self-hosting or on an API? It changes whether the
upgrade is also a serving change.

**Interviewer:** API today.

> **Interviewer note.** Four questions, each of which changes the design: cost of
> failure (metric choice), available labels (what the eval can be built from),
> change cadence (cost ceiling), deployment (blast radius). Candidates who ask
> generic questions ("how many users?") without connecting them to a design
> consequence get no credit for asking.

---

## Minutes 6 to 18: the system-level eval

**Candidate:** Starting with the gate, because it is what decides the launch. It is
two loops. Offline: a versioned golden set of real support questions with reference
answers and the source passages, scored per slice, wired into CI so it runs on any
change to the prompt, the model id, or the retrieval config. Online: an A/B or a
canary that measures escalation rate, thumbs, and follow-up "that is wrong" messages.

**Interviewer:** What is on the golden set?

**Candidate:** A few hundred cases, chosen for coverage rather than volume: the
common paths, the known-hard ones, and one row for every incident we have ever had,
so a fixed bug cannot silently return. Sliced by product area, language, and
question type, because a single average hides a segment collapse. Versioned in the
repo, with a held-out portion I do not look at while iterating on prompts.

**Interviewer:** How do you score a support answer? There is no exact match.

**Candidate:** I split it. Anything checkable gets a task metric: did it cite the
right document, is the quoted figure identical to the source, did it call the
right tool. For the rest I use an LLM judge, but I would not gate on a judge I have
not calibrated. So before it gates anything I collect a few hundred human labels,
measure agreement with the judge, and fix the rubric until agreement clears a bar.
Kappa around 0.6 is the usual threshold.

**Interviewer:** Suppose it clears. Are you comfortable putting the judge's number
in the internal report?

**Candidate:** For ranking two candidates on the same rubric, yes. As an absolute
number, no, because clearing an agreement bar does not make the judge unbiased. If
that number leaves the team I would correct it: keep the judge on everything, keep
human labels on a few hundred items, and add the measured judge bias back to the
judge's mean. That estimator is unbiased regardless of how good the judge is.

> **Interviewer note.** This is the question that separates senior from mid. Most
> candidates stop at "calibrate the judge with kappa." The distinction between
> relative claims (which survive a biased judge) and absolute claims (which do not),
> plus a concrete correction, is a strong-hire signal. It is fine if they do not
> know the name; the mechanism is what counts.

**Interviewer:** How do you decide the gate threshold?

**Candidate:** From measured noise, not taste. I score the same inputs repeatedly to
get the judge's variance, then set the tolerance around one sigma above it. A
zero-tolerance gate flaps on every build and gets disabled within a week, which is
worse than a slightly loose gate. And the gate fires on the worst slice, not the
mean, since a 2 percent segment can collapse entirely and move the average by
2 points, which is inside the noise.

---

## Minutes 18 to 30: the model-level benchmark

**Interviewer:** Now the other half. How do you evaluate the new base model itself?

**Candidate:** A portfolio, not a benchmark. Capabilities that match our workload:
retrieval-heavy reasoning, instruction following, tool calling, and the two
languages we serve. For each, a public suite with headroom, so nothing where every
candidate is above 90, plus one live or time-gated suite as a contamination check,
plus our internal set which is the decision-maker of record.

**Interviewer:** Why the internal set if you already have public benchmarks?

**Candidate:** Two reasons. It matches our construct, and it is the only set I can
prove was not trained on. Public numbers are for external comparability, not for
deciding.

**Interviewer:** The vendor's model card says 78 on a benchmark. Your harness gets
66 on the same benchmark. What happened?

**Candidate:** The prior is protocol, not model. I would check in descending order
of effect size: chat template and system prompt, then scoring mode (log-likelihood
over options versus generating and parsing), then the output-token budget and the
truncation rate, since a truncated reasoning trace scores as a wrong answer, then
few-shot count and the answer-format instruction, then decode parameters and sample
count, then benchmark version. Usually printing one rendered prompt and one raw
completion finds it in a minute. But the resolution is not to reconcile with their
number, it is to re-run every candidate, including the current model, under one
protocol I control.

> **Interviewer note.** Asked verbatim in most loops that touch evaluation. The
> answer being looked for has two parts: an ordered checklist (shows they have
> debugged this), and the conclusion that published baselines are re-run rather
> than trusted. Candidates who answer only "different prompt maybe" get partial
> credit.

**Interviewer:** You run it and the new model is 3 points ahead on your internal
set of 400 items. Ship it?

**Candidate:** Not yet, that might not be a real difference. I would compute it
paired, since both models answered the same items: only the discordant items carry
information, so the difference is (b minus c) over n with standard error root(b plus
c) over n. On 400 items an unpaired interval on each score is around 5 points, so a
3-point gap alone is not resolvable; paired it might be. If the interval includes
zero I say "not distinguishable at this sample size" and give the item count that
would settle it.

**Interviewer:** And if leadership does not like that answer?

**Candidate:** Then I give them the decision they actually need, which is not
"better or worse" but "what would we lose by waiting a week to add items, versus
what do we gain by shipping now." I would also point out that the internal set is
the cheapest thing to grow, and that if we plan to make this decision every quarter,
a set that cannot resolve a 2-point difference will be the bottleneck every time.

> **Interviewer note.** The follow-up is a pressure test, not a statistics question.
> Two failure modes: caving and calling the 3 points a win, or repeating the
> statistics louder. The signal is converting an inconclusive result into a decision
> the business can act on while being honest about what is known.

---

## Minutes 30 to 40: the change lands

**Interviewer:** Assume it passes. What happens on the day of the switch?

**Candidate:** It is a versioned artifact change, so it goes through the same gate as
a prompt edit: CI runs the offline suite per slice, then a canary. I would ship to
internal users first, then a small traffic fraction, watching escalation rate,
thumbs, and the guardrails: latency, cost per request, and refusal rate. A candidate
that wins quality and doubles cost is not obviously a win, and I would rather find
that at 5 percent than at 100.

**Interviewer:** Two weeks in, offline says it is better and escalation rate is up.
What do you do?

**Candidate:** Believe the online signal and fix the offline suite. The gap is the
calibration signal. Concretely I would check which slices escalated, whether those
slices were represented in the golden set, and whether the judge rewarded something
users penalized. The most common version of this is verbosity: the newer model
writes longer, the judge likes longer, users do not. That is testable by regressing
judge score on answer length across a matched set.

**Interviewer:** And if it is verbosity?

**Candidate:** Fix the instrument, not the tolerance. Penalize padding in the rubric
or control for length in scoring, add the escalating cases into the golden set as
new rows, then re-run. I would not widen the gate to make the number pass.

> **Interviewer note.** The offline-online disagreement is the single most
> informative question in an eval interview. Weak answers defend the offline number.
> Strong answers treat the disagreement as data about the eval, name a specific
> plausible mechanism, and describe a test for it.

**Interviewer:** Anything you would have done differently up front, knowing this
happened?

**Candidate:** Yes, two things. I would have added answer length as a tracked
dimension from day one, because it is cheap and it is the most common judge
confound. And I would have run the first canary with a larger fraction for a shorter
time, since we had the traffic and the earlier the online signal arrives, the less
the offline suite gets to be wrong in production.

---

## Minutes 40 to 45: the fast follow-ups

**Interviewer:** Quickly. Someone proposes running the eval only before releases
instead of on every change.

**Candidate:** Then it is not a gate, it is a report. The value comes from running
automatically on every change to a prompt, model id, or retrieval config, the same
way unit tests do. If cost is the concern the fix is a smoke subset for local
iteration and the full suite at the gate, plus caching judged pairs that have not
changed, not a lower cadence.

**Interviewer:** Someone else wants the judge to be the strongest available model.

**Candidate:** Strongest is not the same as certified, and if it is the same family
as the model we are evaluating it self-prefers. I would measure a few candidate
judges against human labels and pick the cheapest that clears the bar, then probe it
with degenerate inputs (empty answers, padded answers, an answer containing an
instruction aimed at the judge) before trusting it.

**Interviewer:** Last one. What is the number you put at the top of the report?

**Candidate:** For the launch decision, the worst-slice delta against the current
production system with its interval, and the online escalation rate from the canary.
Not an aggregate score. If I have to give one number I would give the per-solved-task
cost alongside it, because quality without cost is half a decision.

> **Interviewer note.** Closing rapid-fire is for confirming that the earlier answers
> were understood rather than recited. A candidate who has been consistent all the
> way through will answer these in a sentence each; one who assembled the design
> from memorized parts tends to contradict something they said in minute 12.

---

## The scoring rubric

What the interviewer writes down afterwards.

| Dimension | No hire | Hire | Strong hire |
|---|---|---|---|
| Framing | One eval for everything | Separates model-level from system-level | Names which one gates the launch and why the other cannot |
| Metric design | A single quality score | Splits checkable metrics from judged ones | Separates groundedness from helpfulness and ties each to a business consequence |
| Judge discipline | Trusts the judge | Calibrates against human labels, reports agreement | Distinguishes relative from absolute claims and corrects the bias statistically |
| Statistics | Reports a difference | Puts an interval on it | Compares paired, sizes the set, and is willing to say "not distinguishable" |
| Protocol awareness | Compares against published numbers | Knows numbers differ across harnesses | Gives an ordered debugging checklist and re-runs baselines |
| Operations | Manual eval before release | Wired into CI, canary before rollout | Guardrails alongside quality, and the rollback path is a deploy they have thought about |
| Handling disagreement | Defends the offline number | Believes the online signal | Names a mechanism, tests it, fixes the instrument rather than the tolerance |
| Communication | Answers what was asked | Structured and clear | Converts an inconclusive result into an actionable decision without overclaiming |

## How to use this

Do it as a drill, not a read. Cover the candidate turns, answer out loud, then
compare. The gap between your answer and the transcript is not a knowledge gap most
of the time, it is a **structure** gap: the transcript answers in the order the
interviewer scores, and it says the boundary conditions out loud instead of assuming
them.

Then run the same drill against a different question from the
[question bank](../questions.md) using the [answer framework](../framework/answer-framework.md),
and read the relevant chapter's Q&A afterwards to find what you missed.
