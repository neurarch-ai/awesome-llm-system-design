# LLM System Design: A Practical Guide

This folder holds the book-format edition of the LLM half of this project: the
same production LLM system-design material as the topic walkthroughs, rewritten
as instructional book chapters. Where the top-level repo is organized for
interview practice, these chapters read as a teaching sequence you can work
through front to back.

Each chapter opens a set of **validated reference architectures** from the
Neurarch model zoo as figures. They are not screenshots: every figure is a
shape-checked graph you can open live and inspect layer by layer.

## What you will learn

- Build a production RAG system: chunking, embedding services, ANN indexing at scale, re-ranking, and grounded generation.
- Reason about the real cost of serving an LLM: the KV cache, continuous batching, speculative decoding, and quantization.
- Compose models into products: agent orchestration, multimodal serving, and real-time streaming chat.
- Adapt models with fine-tuning and post-training, and align them with DPO and RLHF.
- Know that a system works and keep it working: offline and online evaluation, safety and guardrails, cost optimization and routing, and production monitoring.
- Connect every design choice to the architecture underneath it, traced on a real graph rather than a box diagram.

## Chapters

| # | Chapter | Covers |
|---|---------|--------|
| 1 | [Serving Retrieval-Augmented Generation](chapter-01-serving-rag.md) | Retrieval, chunking, the embedding service, ANN at scale, re-ranking, grounded generation, eval |
| 2 | [Semantic Search and Embedding Services](chapter-02-semantic-search-and-embeddings.md) | Vector index choice, recall vs latency, quantization, hybrid search, re-ranking |
| 3 | [Long-Context Inference and the KV Cache](chapter-03-long-context-and-kv-cache.md) | The real cost of serving, GQA and MLA, paged attention, prefix caching, batching |
| 4 | [Serving LLM Inference at Scale](chapter-04-inference-serving-at-scale.md) | Continuous batching, speculative decoding, disaggregation, parallelism, autoscaling |
| 5 | [Real-Time Streaming Chat](chapter-05-realtime-streaming-chat.md) | Token streaming, transport, session memory, backpressure, voice pipelines |
| 6 | [Agent Orchestration](chapter-06-agent-orchestration.md) | Tool-calling loops, planning, single vs multi-agent, cost and latency control |
| 7 | [Multimodal Serving](chapter-07-multimodal-serving.md) | Vision-language models, the projector, the image-token budget, serving cost |
| 8 | [Fine-Tuning and Post-Training](chapter-08-fine-tuning-and-post-training.md) | LoRA and QLoRA, SFT, DPO and RLHF, data curation, eval gates |
| 9 | [Evaluating LLM Systems](chapter-09-evaluating-llm-systems.md) | Offline suites, LLM-as-judge, online A/B, regression gates |
| 10 | [Safety, Moderation, and Guardrails](chapter-10-safety-and-guardrails.md) | Input and output filtering, jailbreak and injection defense, PII, policy routing |
| 11 | [Cost Optimization and Model Routing](chapter-11-cost-optimization-and-routing.md) | Routing, cascades, semantic caching, prompt compression, the quality-cost frontier |
| 12 | [Production Monitoring and Observability](chapter-12-production-monitoring.md) | Tracing, online eval without labels, hallucination detection, drift, regression |

A companion book covers the classic-ML half (recommenders, ranking, fraud,
computer vision, speech, NLP, forecasting) in the
[ML System Design Interview](https://github.com/neurarch-ai/awesome-ml-system-design)
repository.

## Technical requirements

You need only a modern web browser to open the validated reference graphs used as
figures throughout the book. Each figure links to the
[Neurarch model zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo), where
the architecture opens live in the editor at real dimensions. The chapters name
the supporting tooling (an approximate-nearest-neighbor index, a re-ranker, an
inference server, a feature or trace store) but do not require you to install
anything to read them.

## How to use this book

Each chapter ends with a **Questions** section for self-review and a **Further
reading** list of first-party production engineering writeups. Read a chapter,
open its figures, attempt the questions, then follow the further reading to see
how real teams shipped the same system. Built by [Neurarch](https://www.neurarch.com).
