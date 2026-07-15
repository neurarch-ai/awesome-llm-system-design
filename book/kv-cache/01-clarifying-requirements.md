# 1. Clarifying the requirements

Before touching an architecture, pin down what the system must do. Every question
below either removes work or changes the answer completely.

**Candidate:** What does the workload look like: long prompts with short answers,
or short prompts with long answers?

**Interviewer:** Mixed, but the main pain point is a RAG chatbot with a large
shared instruction block and 10 to 15 rounds of history per session. Prompts can
reach 32k tokens; answers are usually under 500 tokens.

**Candidate:** What context lengths are we targeting? "Long context" means
something different at 8k versus 200k.

**Interviewer:** 32k today, but the product team wants to get to 128k.

**Candidate:** What latency targets matter most: first-token latency, inter-token
latency, or both?

**Interviewer:** Users notice both. First token under 2 seconds; inter-token
under 100 ms. Right now p99 decode latency (decode is the phase that generates
output tokens one at a time) blows past that under load.

**Candidate:** What is the quality floor? Specifically, are we allowed to
quantize the KV cache, change the attention architecture, or are we serving a
fixed third-party model?

**Interviewer:** We run our own weights, so architectural changes at training time
are on the table. Quantization is fine if quality holds on our evals.

**Candidate:** What is the concurrency target: how many simultaneous sessions?

**Interviewer:** 500 to 1000 concurrent sessions per GPU node, with traffic
spikes to 3x.

**Candidate:** Is the system prompt shared across most requests, or does it vary
per user?

**Interviewer:** There is a 4k-token system prompt shared by every request. User
history after that is per-session.

---

Let us summarize. **We are serving a RAG chatbot at 32k to 128k context,
targeting first-token latency under 2 s and inter-token latency under 100 ms at
500 to 1000 concurrent sessions, with a 4k shared system prompt and a model we
control.** Quality can be traded against memory as long as it passes a long-context
eval.

Two consequences fall out of this immediately:

**The KV cache (the saved keys and values from every past token, kept so the model
does not recompute them each step) will dominate GPU memory, not the model
weights.** At 32k tokens with a typical 7B model in GQA configuration
(grouped-query attention, where several query heads share one set of keys and
values to shrink the cache), the cache for a single session already costs over 1 GB. At 128k and 1000 concurrent sessions, this dwarfs the
weight footprint. Understanding why is the entire cost model, and it is covered in
the next section.

**The 4k shared system prompt is a gift.** Every request that reuses that prefix
can skip its prefill (the one-time forward pass that builds the KV cache for all
prompt tokens) entirely if the serving stack supports prefix caching (reusing a
shared prefix's already-cached KV instead of recomputing it). That
one fact alone changes first-token latency more than almost any other lever.
Naming it early is a strong signal that you have built or studied these systems.
