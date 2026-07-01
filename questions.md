# Question bank

This repo is organized by **component** (RAG, the KV cache, agents, serving, and
so on) because those are the reusable building blocks. But interviews are not
posed that way. An interviewer asks an **end-to-end question** ("design a customer
support agent"), and you are expected to assemble the right components yourself.

This page is the bridge. It lists the LLM system design questions that actually
get asked, what each one is really testing, and which topics to draw on. Practice
by picking a question, drafting an answer with the [framework](framework/answer-framework.md),
then reading the linked topics to find your gaps.

| Question | What it is really testing | Topics to draw on |
|---|---|---|
| Make our LLM feature cheaper without tanking quality | Right-sizing and routing, cascades, semantic caching, prompt compression, quantization | [04 inference serving](topics/04-inference-serving-at-scale.md), [02 KV cache](topics/02-long-context-and-kv-cache.md) |
| How do you know a live LLM app still works, and catch a hallucination spike? | Online eval without ground truth, LLM-as-judge, grounding checks, regression detection | [06 evaluation](topics/06-evaluation-system.md) |
| Design a coding / support / research assistant (agent) | Tool-calling loops, planning, retrieval, cost and latency control, verification | [03 agents](topics/03-agent-orchestration.md), [01 RAG](topics/01-rag-serving.md), [06 evaluation](topics/06-evaluation-system.md) |
| Add images or audio to our assistant | The projector, image-token budget, multimodal serving cost | [09 multimodal](topics/09-multimodal-serving.md) |
| Keep our chatbot safe and on-policy | Input and output guardrails, jailbreak and injection defense, PII, human review | [07 safety and guardrails](topics/07-safety-and-guardrails.md), [06 evaluation](topics/06-evaluation-system.md) |
| Design a Q&A system over our internal docs / wikis | Retrieval quality, grounding, freshness, access control | [01 RAG](topics/01-rag-serving.md), [08 search](topics/08-semantic-search-and-embeddings.md), [06 eval](topics/06-evaluation-system.md), [07 safety](topics/07-safety-and-guardrails.md) |
| Design a customer support agent that resolves tickets | Tool-calling loop, bounded cost, safe writes, injection defense | [03 agents](topics/03-agent-orchestration.md), [07 safety](topics/07-safety-and-guardrails.md), [06 eval](topics/06-evaluation-system.md) |
| Our LLM serving is too slow and expensive, fix it | The cost model, KV cache, batching, where to spend | [02 KV cache](topics/02-long-context-and-kv-cache.md), [04 serving](topics/04-inference-serving-at-scale.md) |
| Design an LLM inference platform at high QPS | Throughput per GPU vs p99, batching, parallelism, autoscaling | [04 serving](topics/04-inference-serving-at-scale.md), [02 KV cache](topics/02-long-context-and-kv-cache.md) |
| Design a ChatGPT-style chat product | Streaming, session state, multi-turn cost, degradation | [10 streaming](topics/10-realtime-streaming-chat.md), [02 KV cache](topics/02-long-context-and-kv-cache.md), [04 serving](topics/04-inference-serving-at-scale.md), [07 safety](topics/07-safety-and-guardrails.md) |
| Design a code-generation assistant | Decode-heavy serving, latency, eval by execution | [04 serving](topics/04-inference-serving-at-scale.md), [02 KV cache](topics/02-long-context-and-kv-cache.md), [06 eval](topics/06-evaluation-system.md) |
| Design enterprise / semantic search | Embeddings, ANN index, hybrid search, re-ranking | [08 search](topics/08-semantic-search-and-embeddings.md), [01 RAG](topics/01-rag-serving.md) |
| Adapt a base model to our domain task | Fine-tune vs RAG vs prompt, LoRA, the eval gate | [05 post-training](topics/05-post-training-pipeline.md), [01 RAG](topics/01-rag-serving.md), [06 eval](topics/06-evaluation-system.md) |
| Build an LLM evaluation / experimentation platform | Offline suites, LLM-as-judge, regression gates, A/B | [06 eval](topics/06-evaluation-system.md), [05 post-training](topics/05-post-training-pipeline.md) |
| Design the safety / moderation layer for an LLM product | Layered defense, classifiers, injection, PII, latency budget | [07 safety](topics/07-safety-and-guardrails.md), [06 eval](topics/06-evaluation-system.md) |
| Design a visual assistant that answers questions about images | Vision encoder + projector, image-token budget, serving | [09 multimodal](topics/09-multimodal-serving.md), [02 KV cache](topics/02-long-context-and-kv-cache.md) |
| Design an AI summarization / writing feature | Long-context cost, grounding, eval of open-ended output | [01 RAG](topics/01-rag-serving.md), [02 KV cache](topics/02-long-context-and-kv-cache.md), [06 eval](topics/06-evaluation-system.md) |
| Reduce hallucinations in our LLM product | Grounding via retrieval, abstention, eval of faithfulness, output guards | [01 RAG](topics/01-rag-serving.md), [06 eval](topics/06-evaluation-system.md), [07 safety](topics/07-safety-and-guardrails.md) |
| Make our agent cheaper and safer to run | Step/token budgets, model tiering, code-side action gates, injection defense | [03 agents](topics/03-agent-orchestration.md), [02 KV cache](topics/02-long-context-and-kv-cache.md), [07 safety](topics/07-safety-and-guardrails.md) |
| Design a chat product that remembers across sessions | Session state, transcript growth, summarization, prefix caching | [10 streaming](topics/10-realtime-streaming-chat.md), [01 RAG](topics/01-rag-serving.md), [02 KV cache](topics/02-long-context-and-kv-cache.md) |
| Design an LLM gateway / model routing layer | Routing across models and providers, fallback, cost and latency control, eval | [04 serving](topics/04-inference-serving-at-scale.md), [06 eval](topics/06-evaluation-system.md), [07 safety](topics/07-safety-and-guardrails.md) |

## How to use a question

1. **Scope it first.** Almost every question above hides the same first move:
   pin the task, the scale, the latency budget, and how quality is measured before
   you design anything. See [the framework](framework/answer-framework.md).
2. **Name the dominant constraint.** Most LLM questions reduce to a cost-versus-
   latency tradeoff in serving, or a retrieval-quality ceiling. Say which one
   dominates this question early.
3. **Assemble, do not recite.** The components are the topics; the skill is wiring
   the right ones for this question and defending the wiring.
4. **Bring eval and safety unprompted.** "How do you know it works" and "what
   happens with untrusted input" separate senior answers, and they apply to almost
   every question here.

Missing a question you were asked? Open an issue or a PR; see
[topics/README.md](topics/README.md).
