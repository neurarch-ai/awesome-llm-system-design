# Topics

Each topic is a self-contained interview walkthrough. They follow the same
shape, mirroring the [answer framework](../framework/answer-framework.md):

1. The question, as an interviewer would pose it
2. Clarifying questions and scope
3. Requirements
4. High-level data flow (offline and online paths)
5. Deep dives on the components that carry the signal
6. Bottlenecks and scaling
7. Failure modes, safety, and eval
8. Likely follow-up questions
9. The relevant architecture, opened live

## Topics by pipeline stage

All twelve topics are written. They are grouped by where they sit in a production LLM
system. To navigate by use case instead, start from the
[question bank](../questions.md).

**Retrieval and knowledge**
- [01 - RAG serving](01-rag-serving.md)
- [08 - Semantic search and embedding service](08-semantic-search-and-embeddings.md)

**Inference and serving (the cost layer)**
- [02 - Long-context inference and the KV cache](02-long-context-and-kv-cache.md)
- [04 - LLM inference serving at scale](04-inference-serving-at-scale.md)
- [10 - Realtime streaming chat](10-realtime-streaming-chat.md)
- [11 - Cost optimization and model routing](11-cost-optimization-and-model-routing.md)

**Building applications on top**
- [03 - Agent orchestration](03-agent-orchestration.md)
- [09 - Multimodal serving](09-multimodal-serving.md)

**Adapting the model**
- [05 - Fine-tuning and post-training pipeline](05-post-training-pipeline.md)

**Quality and safety**
- [06 - LLM evaluation system](06-evaluation-system.md)
- [07 - Safety, moderation, and guardrails](07-safety-and-guardrails.md)
- [12 - Production monitoring and observability](12-production-monitoring-and-observability.md)

## Planned

- Speculative decoding deep dive and disaggregated prefill/decode
- Cost modeling and capacity planning end to end
- On-device and edge LLM serving

## Contributing a topic

Open a PR that follows the eight-section shape above. Rules:

- **Be honest about numbers.** If you cite a dimension, a latency, or a cost,
  it should be real or clearly labeled as an illustrative estimate.
- **No vendor pitch in the body.** The walkthrough teaches; the only product
  references are the architecture links at the end, which open the actual graph
  being discussed.
- **Link the architecture where it is real.** If your topic touches a specific
  model mechanism, link the validated graph so readers can trace it.
- **Prefer mechanism over name-dropping.** Explain what GQA does, do not just say
  a model uses it.
