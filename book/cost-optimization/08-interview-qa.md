# 8. Interview Q&A

The questions an interviewer actually asks about cost optimization and model
routing, grouped by how they are typically used. The commonly-missed ones are
where interviews are won or lost.

## Commonly asked

**Q: Where would you start on a system you have never seen before?**
A: Put a gateway in front so spend is observable, then profile where the money
goes: is it input tokens (over-retrieval, long prompts), output tokens (long
generation), or request count (high-QPS on a large model)? That single
measurement decides which lever to reach for first: trimming/compression for
input cost, routing/right-sizing for output cost, caching for request count. A
design that skips the profiling step and jumps straight to routing or caching
often optimizes the wrong term.

**Q: When do you use a router versus a cascade?**
A: It is a latency trade, not a quality trade. A router decides blind, before
any generation, so it fits a tight latency SLO but cannot know when it mis-routes.
A cascade runs the cheap model first, scores its own answer, and escalates only
when unsure; it catches its own mistakes at the cost of a first call plus a
scorer. If you have latency slack and a trustworthy confidence signal (verifiable
tasks: code, SQL, citations), use a cascade. If the SLO is tight, route. They
compose: route first to divide traffic into buckets, cascade within the uncertain
middle bucket.

**Q: What is the right similarity threshold for a semantic cache?**
A: There is no single right answer; you sweep it on a labeled dataset of
should-hit and should-not pairs and pick the threshold that maximizes hit rate
while keeping the rate of wrong-neighbor answers below your quality tolerance. The
common mistake is tuning on raw hit rate: a looser threshold raises hit rate and
can simultaneously increase the fraction of served responses that are wrong.

**Q: Estimate the savings from a router.**
A: Router savings per request: $S = f_{\text{weak}} \cdot (c_{\text{big}} - c_{\text{small}}) - c_{\text{router}}$.

```python
def router_savings(f_weak, c_big, c_small, c_router):
    # savings = big-vs-small gap captured on the weak-handled fraction, minus router cost
    return f_weak * (c_big - c_small) - c_router
# e.g. router_savings(0.5, 10.0, 2.0, 0.25) -> 3.75
```

Start with a measured $f_{\text{weak}}$ from a sample of traffic (what fraction
of queries would a small model answer at bar?), not a hoped-for one. A common
trap is assuming the cheap model handles 70% of traffic when the actual number,
measured on a hard eval set, is 40%.

**Q: What can go wrong with a semantic cache?**
A: Three failure modes. First, a loose threshold returns a near-neighbor's answer
to a genuinely different question (confidently, cheaply wrong). Second, volatile
facts served from cache become stale; set TTLs. Third, a shared cache for scoped
or personalized answers leaks user A's response to user B; scope the cache key
or skip caching for personalized content entirely.

## Tricky (the follow-ups that separate people)

**Q: The bill dropped 40% after you shipped the router. Did quality hold?**
A: Unanswerable without per-bucket quality tracking. A cost drop with no paired
quality number is a silent regression, not a win. A green cost dashboard is
the exact signature of a router dumping newly-hard queries on the small model.
The right thing to report is cost saved and quality retained, per routing bucket,
including the hard tail. If you did not measure it, the right answer is "I do not
know, and that is a problem."

**Deeper:** The blind spot is structural, not accidental. An aggregate quality metric is dominated by the easy majority of traffic, so a regression concentrated in the hard tail (a small fraction of requests) barely moves the mean. Only per-bucket quality measured on a fixed hard-tail slice can surface it, which is why the aggregate dashboard staying green is the signature of the failure rather than evidence against it.

**Q: Everyone uses semantic caching. Should you always deploy one?**
A: Only if the hit rate clears the break-even: $h^{\ast} = c_{\text{embed}} / (c_{\text{model}} - c_{\text{hit}})$.

```python
def cache_breakeven(c_hit, c_embed, c_model):
    return c_embed / (c_model - c_hit)   # hit rate above which caching nets positive
# e.g. cache_breakeven(0.005, 0.02, 1.0) -> 0.020100502512562814  (about 2%)
```

On typical numbers this is around 2%, but on traffic that is nearly all unique
free-text queries (e.g., a general-purpose assistant), even a tuned semantic
cache may not reach break-even. Measure the organic hit rate on a sample before
shipping the cache.

**Deeper:** The break-even $h^{\ast}$ rises as the served model gets cheaper, because the denominator $c_{\text{model}} - c_{\text{hit}}$ shrinks while the embedding cost $c_{\text{embed}}$ (paid on every request, hit or miss) stays fixed. So a workload already routed to a cheap small model can make the cache net-negative even at a respectable hit rate, which is the opposite of the intuition that caching is always free money.

**Q: Bigger LLMs are slower and cost more per token. Can you always swap in a
smaller model for a subtask?**
A: Not always. On a narrow, stable, labeled task a fine-tuned small model often
matches or beats the frontier model. On tasks that require broad reasoning, recent
world knowledge, or instruction following across long context, a small model
genuinely regresses. The test is running both on your eval set, not assuming
either direction. The operational cost is real: each additional model in the
fleet must be evaluated, monitored, and updated.

**Deeper:** The regression shows up specifically on inputs outside the fine-tune's training distribution, so a small model that passes the offline eval today can still fail on novel phrasings tomorrow. That makes distribution drift the real risk: the eval set has to be refreshed against live traffic, or the gap silently reopens while the offline number stays green.

**Q: When does LLMLingua compression actually net out as a win?**
A: When the input-token saving exceeds the compression pass cost: $c_{\text{big}}
\cdot (n_{\text{orig}} - n_{\text{comp}}) \gt c_{\text{small}} \cdot n_{\text{orig}}$.

```python
def compression_net_win(c_big, c_small, n_orig, n_comp):
    # gain from tokens removed on the big model vs cost of the small-LM pass over the full prompt
    return c_big * (n_orig - n_comp) > c_small * n_orig
# e.g. compression_net_win(10.0, 1.0, 1000, 200) -> True
```

On short prompts or output-dominated workloads the small-LM pass is pure overhead.
On long, verbose, redundant RAG context (20 retrieved chunks with lots of
repeated boilerplate) it can pay 5-10x. Profile the bill before applying
compression; trim to top-3 chunks first since that is zero-cost and often
achieves most of the input-token reduction.

**Deeper:** The compression pass is itself a small-LM forward over the full original context to score token importance, so its cost scales with $n_{\text{orig}}$, not $n_{\text{comp}}$. That is the mechanism behind the break-even: on short-input or output-dominated workloads the pass is pure overhead because the tokens it removes were cheap to begin with; the win only materializes when the removed input tokens were both numerous and expensive on the big model.

**Q: Context trimming and LLMLingua-style compression look similar, both shrink
the prompt; when does the difference actually matter?**

A: Both reduce input tokens before the big-model call, and on a padded RAG
prompt either one lowers the bill. The mechanics differ in granularity and in
what they cost. Trimming drops whole retrieved chunks using a relevance score
that the pipeline often already computes, so it is near-free and the surviving
text is untouched. Compression pays a small-LM forward pass over the entire
original context to score every token, then deletes low-information tokens
inside the text the model will actually read. The difference matters in two
regimes. If the waste is between chunks (top-20 retrieval where 17 chunks are
padding), trimming captures almost all the saving at zero quality risk and
compression adds cost for little extra gain. If the waste is inside the chunks
(verbose boilerplate the answer must be extracted from), trimming cannot touch
it, and only token-level compression helps, at the price of the scoring pass
and the risk of deleting the one token the answer hinged on. That risk profile
is why trimming is safe on extraction tasks and compression is not.

## Commonly answered wrong (the traps)

**Q: Can you add a frontier model call to the router to make it more accurate?**
A: No. The router must be strictly cheaper than the cheapest model it gates.
Using a frontier call to route means you pay frontier prices on every request
plus the model call on the routed subset. The router's cost $c_{\text{router}}$
is subtracted from savings in the formula; make it large enough and savings go
negative. Use a classifier, a heuristic, or a small fine-tuned model.

**Deeper:** Even setting price aside, a frontier router adds its full latency to every request before any useful generation begins, so it worsens time-to-first-token on exactly the easy queries routing was supposed to speed up. A router must be cheaper and faster than what it gates, on both axes.

**Q: Should the router optimize for cost minimization?**
A: No. Optimizing the router purely for cost causes it to dump newly-hard queries
on the small model whenever the cost function dominates the training signal. The
correct objective is cost minimization subject to a quality floor per bucket.
Load the eval set with the hard tail so a router that mis-handles hard queries
shows as a quality regression, not a cost win.

**Deeper:** This is a reward-specification failure. Cost is fully and cheaply observable at every request, while quality on the hard tail is sparsely sampled, so an unconstrained objective drifts toward the term it can see. Encoding the per-bucket quality floor as a hard constraint rather than a soft penalty is what stops the optimizer from trading away tail quality for a legible cost number.

**Q: FP8 quantization improved throughput. Does this mean the API got cheaper?**
A: Only if you are self-hosting. On a per-token API the provider controls the
serving infrastructure; quantization on your side does not change the bill.
FP8 is a self-hosting lever: it reduces VRAM footprint and increases tokens per
GPU-second, which translates to lower cost per token only when you own the GPU.

**Deeper:** On an API the per-token price is set by the provider's contract and is decoupled from their serving efficiency, so any throughput gain they get from quantization accrues to their margin, not your bill. Your quantization only moves cost when you are the one paying the GPU-hour, which is why the lever exists solely above the self-hosting QPS break-even.

**Q: Should I route offline batch jobs through the same routing logic as
interactive traffic?**
A: No. Offline bulk jobs (backfills, nightly classification, offline eval
generation) have no user waiting and no latency SLO. Route them to a provider
batch API (about half the per-token price of the sync endpoint) or to a
self-hosted model running at maximum batch size on spot capacity. The interactive
routing logic is designed around a latency budget that offline jobs do not have;
apply it to offline traffic and you pay online prices for offline work.

**Deeper:** The batch endpoint earns its lower price by deferring execution and packing requests into large batches that maximize GPU utilization, which is exactly the throughput-for-latency trade an interactive SLO forbids. So the roughly half price is the mechanism of batching showing up in the bill, not an arbitrary discount, and routing interactive traffic through it would blow the latency budget outright.
