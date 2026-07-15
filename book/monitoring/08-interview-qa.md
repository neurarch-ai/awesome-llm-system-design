# 8. Interview Q&A

The questions an interviewer actually asks about production LLM monitoring,
grouped by how they are used. The commonly-missed ones are where interviews are
won or lost.

## Commonly asked

**Q: You have no labels in production. How do you measure quality?**

A: You do not measure it; you estimate it from three kinds of proxy. First, an
LLM-as-judge running on a sampled slice of traces scores faithfulness (is the
answer supported by the retrieved context) and relevance (does it address the
question). Second, a grounding check decomposes each answer into atomic claims
and verifies them against the logged retrieved context. Third, implicit user
signals, accept, discard, heavy edit, immediate rephrase, provide a dense and
honest behavioral signal at zero extra cost. None of these is accuracy. They are
estimates that lie until you calibrate each one against human labels and report a
kappa or F1 agreement score.

**Q: How do you catch a hallucination spike specifically?**

A: Compute a per-response groundedness score against the retrieved context that
was logged on the span. Aggregate to an ungrounded rate over the current window
and compare it to a baseline rate using a z-score. Alert when $z_t$ exceeds your
threshold, not when a single response is flagged. Trend the rate after every
retrieval or model change and alert on the delta. A single flagged answer is noise;
a rate shift is the signal.

**Q: The team is swapping the model next week. How do you prevent a silent
regression?**

A: Three steps, in order of risk. First, run the frozen eval set against the
candidate; if it regresses below the acceptable band, block the rollout before any
user sees it. Second, if the frozen eval passes, route five to ten percent of live
traffic to the candidate (canary) and compare proxy scores, latency, cost, and
user feedback against the control for at least one full traffic cycle. Third,
optionally run the candidate in shadow mode in parallel for zero-risk output
diffing before the canary. Coordinate the rollout schedule so humans are available
to respond if the canary goes wrong.

**Q: Why not just run the judge on every request?**

A: Cost. An LLM judge call costs roughly as much as the generation call itself,
so judging every request roughly doubles the serving bill. At our fifteen-percent
observation budget, we can judge at most fifteen percent of traffic. The right
move is to sample strategically (oversample the suspicious tail) rather than
uniformly, and to use a cheap encoder or NLI model as a first-pass filter that
routes only the flagged fraction to the expensive judge.

**Q: What are the first-class dashboards you would build?**

A: Six dashboards I would consider non-negotiable: latency percentiles (p50, p95,
p99, and TTFT separately, not the mean); cost and tokens per request; error and
timeout rates by error class; judge score rolling average with a trend line;
ungrounded rate with z-score alert; and guardrail firing rates (block rate,
jailbreak hit rate, refusal rate). The guardrail rates matter because a rising
refusal rate is a silent degradation: blocked answers never get quality-scored,
so the quality dashboard says healthy while good traffic is being rejected.

## Tricky (the follow-ups that separate people)

**Q: Your judge score is trending up month over month. Is that good news?**

A: Not necessarily. The judge may be drifting: it was calibrated against a human
sample collected three months ago, and since then the domain, the prompt, or the
traffic mix has shifted. A judge that was accurate at kappa 0.82 in January may
be systematically flattering by March. Report the judge-human agreement alongside
the score; if kappa is falling while the score rises, the instrument is lying.
Recalibrate regularly by collecting fresh human labels on real traffic.

**Q: You are sampling five percent of traffic for judging. Your CEO asks if
quality is improving. What do you tell them?**

A: You tell them the judge score trend is your best available estimate, along with
the kappa agreement score that quantifies how much to trust it. You also report
the implicit behavioral signal (accept rate, retry rate) which covers all traffic
at zero extra cost and is harder to game than a judge. If those two signals agree,
you have corroborating evidence. If they diverge, investigate before claiming
improvement.

**Q: How does an agent system differ from a RAG system in what you log and
check?**

A: For an agent, the retrieved context is replaced by a sequence of tool calls
with arguments, results, and errors. You need a span per tool call, not just a
span per request, because the failure might be in step 3 of a 6-step plan and you
cannot find it without step-level granularity. Grounding checks become plan
coherence checks: did the agent call the right tools with the right arguments, and
did its conclusions follow from the tool results? Frozen evals are harder to
maintain because agent traces are longer and more variable than single-shot QA.

**Q: Your ungrounded-rate z-score alert fires at the same time every Monday
morning. Is the model broken?**

A: Probably not; this is the classic failure of a stationary-baseline z-score. The
z-score $z_t = (\text{rate}_t - \mu) / \sigma$ assumes the baseline mean $\mu$ and
standard deviation $\sigma$ describe a single stable distribution, but production
traffic is seasonal: weekday-morning traffic can carry a different query mix (more
first-time users, different topics) than the weekend window the baseline averaged in.
The rate genuinely shifts, so the z-score genuinely spikes, but the model did not
change. The fix is mechanism-level, not threshold-tuning: compute the baseline over a
matched window (same hour-of-week) or difference against a seasonally-adjusted
expectation, so you are alerting on the residual after known periodicity is removed
rather than on the periodicity itself. Interviewers use this to see whether you
understand that drift detection is a stationarity assumption, not a magic threshold.

**Q: The grounding score looks fine but users are still unhappy. Where do you
look?**

A: Grounding only checks whether claims are supported by the retrieved context. It
cannot catch: wrong context retrieved (the model answered from the right form but
the wrong document), irrelevant answers (technically faithful but off-topic),
tone or format failures (too long, too technical, wrong language register), or
outdated context (the retrieved document was accurate six months ago). Cross-check
with explicit feedback, retry rates, and edit rates to localize the failure mode.

## Commonly answered wrong (the traps)

**Q: Can you just report mean response latency on the dashboard?**

A: No. The mean hides the tail, and the tail is what users feel in a streaming UI.
A "drop-in better" model that raises p99 latency from 800ms to 3000ms and doubles
TTFT is a regression for most users even if the mean barely moves. Always report
p50, p95, p99, and TTFT separately. TTFT is the streaming-perceived latency;
p99 is what your worst-served users experience.

**Q: A high thumbs-up rate means quality is good, right?**

A: No. Only a tiny, self-selected fraction of users ever clicks the thumbs widget,
biased toward the very pleased and the very angry. Low click-through means
nothing. Cross-check with implicit signals: edit rate, retry rate, and abandon
rate cover all users. Cross-check with the judge and grounding scores. A high
thumbs-up rate with a high retry rate is a contradiction to investigate, not a
victory to celebrate.

**Q: You ran the frozen eval set and scores look fine after the model swap. Ship
it?**

A: Not yet. The frozen eval set goes stale: it was built from past traffic and
may not cover the failure modes of current traffic. A fine-grained regression on a
new topic cluster, a language the team added support for, or a prompt pattern that
emerged in the last month will not appear in a stale eval set. The frozen eval is
a necessary gate, not a sufficient one. Follow it with a canary on live traffic
before full rollout.

**Q: My guardrail block rate is stable, so safety is fine?**

A: Not necessarily. A stable block rate could mean the dangerous outputs are still
getting through uncaught. The dangerous case is the harmful output that no
guardrail flagged. Supplement the block-rate dashboard with a sampled safety
re-scan on allowed traffic (outputs that were not blocked) to catch the failures
the guardrail missed. The mechanism-level reason block rate is the wrong instrument:
it measures the guardrail's recall against what it can already detect, not against
the true harm base rate, so it is blind to any novel attack the classifier was never
trained to catch. A jailbreak family the filter does not recognize produces zero
blocks and a perfectly stable block rate while harmful outputs flow. Only sampling
the allowed stream into an independent re-scan (ideally a different model or a human)
estimates the miss rate on the traffic the primary guardrail called safe.
