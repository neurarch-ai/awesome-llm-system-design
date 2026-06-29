# The answer framework

LLM system design interviews reward the same thing every system design interview
rewards: a structured answer that starts broad, commits to a design, and then
stress-tests it. The content is new; the shape is not.

Use these six steps. They map onto a 45-minute loop with time to spare.

## 1. Clarify and scope (5 min)

Do not start designing. The fastest way to fail is to build the wrong system
confidently. Pin down:

- **The task.** "Answer questions over docs" and "summarize tickets" and "write
  code" are three different systems. What exactly goes in and comes out?
- **Scale.** Queries per second at peak. Corpus size. Number of users. Context
  length. These set every downstream number.
- **Latency budget.** A chat UI tolerates streaming first-token latency under a
  second. A batch enrichment job tolerates minutes. This decides your whole
  serving strategy.
- **Quality bar and how it is measured.** "Good answers" is not a spec. Is there
  ground truth? A human review loop? An acceptable hallucination rate?
- **Cost sensitivity.** Are you optimizing for a research demo or a product with
  a per-query margin? This decides whether you reach for the biggest model or
  the cheapest one that clears the bar.

State your assumptions out loud and write them down. The interviewer will correct
the ones that matter.

## 2. Define functional and non-functional requirements (3 min)

Turn the clarifications into a short list.

- **Functional:** what the system must do (retrieve, generate, cite sources,
  call tools, stream).
- **Non-functional:** latency p50/p99, throughput, cost per query, freshness,
  availability, safety constraints.

Name the one or two non-functional requirements that will dominate the design.
For most LLM systems it is the cost-versus-latency tradeoff in serving.

## 3. Sketch the high-level data flow (8 min)

Draw the boxes and the arrows. For an LLM system there are almost always two
distinct paths, and conflating them is a classic mistake:

- **The offline / write path:** ingesting and preparing data. Document parsing,
  chunking, embedding, index building, fine-tuning. Runs on a schedule or on
  ingest, not per query.
- **The online / read path:** what happens per request. Retrieve, assemble the
  prompt, call the model, post-process, stream back.

Separating them lets you reason about freshness (a write-path concern) and
latency (a read-path concern) independently.

## 4. Go deep on the components the interviewer cares about (15 min)

This is where most of the signal is. You cannot go deep on everything, so read
the room and pick. The high-value deep dives in LLM systems are usually:

- **Retrieval quality.** Chunking strategy, embedding model choice, hybrid
  search, re-ranking. Recall is usually the ceiling on RAG quality.
- **The model and its serving cost.** Which model, and why. Then the cost model:
  the [KV cache](../topics/02-long-context-and-kv-cache.md), batching,
  quantization. This is where knowing the actual architecture pays off.
- **The eval loop.** How you know it works and how you catch regressions. Strong
  candidates bring this up unprompted.

When you discuss the model, be concrete about the architecture. "We use an MoE
model so only a fraction of parameters activate per token" is a stronger answer
than "we use a big model," and it invites the follow-up you want.

## 5. Identify bottlenecks and scale (8 min)

Walk the request path and ask where it breaks first as load grows:

- Embedding the query on every request: cache it.
- Vector search latency at corpus scale: approximate index, sharding.
- Model throughput: continuous batching, more replicas, a smaller model with
  speculative decoding.
- The KV cache filling GPU memory on long contexts: this is often the real wall.
  See topic 02.

For each bottleneck, name the fix and its tradeoff. There is no free lunch and
interviewers want to see that you know it.

## 6. Address failure modes and quality (5 min)

The thing that separates senior answers:

- **Hallucination and grounding.** Citations, abstention, confidence
  thresholds.
- **Safety.** Input and output filtering, prompt-injection defense (especially
  for agents and RAG over untrusted content).
- **Graceful degradation.** What happens when the model is down, the index is
  stale, a tool times out.
- **Cost runaway.** Token budgets, max iterations on agent loops, caching.

---

## The cost model you should be able to do in your head

Almost every LLM serving question reduces to this. Be ready to estimate it:

- Cost and latency scale with **tokens processed**, split into **prefill**
  (the prompt, processed in parallel) and **decode** (the output, generated one
  token at a time).
- Decode is memory-bandwidth bound and dominated by the **KV cache**, which
  grows with context length, batch size, and model size.
- The big levers: a smaller or sparser (MoE) model, a smaller KV cache (GQA,
  latent attention), bigger effective batches (continuous batching), fewer
  decode steps (speculative decoding), and caching repeated prefixes.

If you can reason about prefill versus decode and why the KV cache dominates long
contexts, you are ahead of most candidates. Topic 02 builds this out with real
architectures.

---

## Driving the conversation (delivery counts as much as content)

Interviewers grade how you communicate, not just what you know. A correct design
delivered as a wall of monologue scores worse than a slightly thinner one
delivered as a led conversation. Concretely:

- **Lead, do not wait.** State your plan for the next 40 minutes up front ("I will
  scope, sketch the high-level design, then go deep on retrieval and serving"),
  then drive it. The interviewer should steer, not push.
- **Think out loud and signpost.** Say which step you are on and why. "I am going
  to assume X because Y; tell me if that is wrong" turns silence into signal.
- **Make assumptions explicit and write them down.** It lets the interviewer
  correct the ones that matter instead of watching you build on sand.
- **Manage time.** Do not spend 20 minutes on chunking. Cover the whole system at
  some depth, then go deep where the interviewer leans in. If you are running long,
  say "I will note monitoring exists and move on unless you want it."
- **Read the steer.** When they ask a question, it is usually a hint about where
  they want depth. Follow it instead of finishing your prepared thread.
- **State tradeoffs, not verdicts.** "Option A is cheaper but staler; I would pick
  A because the freshness budget is an hour" beats asserting A with no reasoning.

## Common mistakes

- Jumping to a solution before scoping.
- Designing one path and forgetting the other (online versus offline).
- Naming a model but having no cost model for serving it.
- No eval story. "It works" is not an answer.
- Ignoring safety on systems that ingest untrusted text.
- Reaching for the biggest model when a smaller one clears the bar cheaper.
