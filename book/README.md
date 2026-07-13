# LLM System Design: A Practical Guide

This folder holds the book-format edition of the LLM half of this project: the
same production LLM system-design material as the topic walkthroughs, rewritten as
teach-first book chapters. Where the top-level repo is organized for interview
practice, these chapters read as a teaching sequence you can work through front to
back.

Each chapter is a folder of one file per section. It opens with a
Candidate/Interviewer dialogue to scope the problem, walks the design end to end,
borrows how real teams shipped it in production (with first-party links), and
closes with an interview Q&A and a self-test. Figures are worked matplotlib plots
and mermaid diagrams, plus **validated reference architectures** from the Neurarch
model zoo that open live at real dimensions, not screenshots.

## What you will learn

- Understand how a model comes to exist: the lifecycle, data curation and pretraining, continued pretraining, and post-training with LoRA, SFT, DPO, and RLHF.
- Reason about the real cost of serving an LLM: the KV cache, continuous batching, speculative decoding, quantization, cost optimization, and streaming.
- Ground a model in knowledge: RAG, semantic search, and the embedding and re-ranking stack.
- Compose models into products: agent orchestration and multimodal serving.
- Know that a system works and keep it working: offline and online evaluation, safety and guardrails, and production monitoring.
- Connect every design choice to the architecture underneath it, traced on a real graph rather than a box diagram.

## Chapters

Ordered the way a model comes to life: build it, serve it, ground it, compose it, then assure it.

### Building and adapting the model

| Chapter | Covers |
|---------|--------|
| [The LLM Lifecycle](llm-lifecycle/) | The five-stage map (data, pretraining, mid-training, post-training, deployment) and the cross-stage math |
| [Data Curation and Pretraining](data-and-pretraining/) | Web-scale data pipeline, dedup and decontamination, tokenizer, scaling laws, architecture and parallelism |
| [Continued Pretraining and Long Context](continued-pretraining/) | Domain adaptation, catastrophic forgetting, RoPE scaling, YaRN, long-context evaluation |
| [Fine-Tuning and Post-Training](post-training/) | Prompt vs RAG vs SFT vs LoRA, DPO and RLHF, data curation, eval gates |

### Inference and serving

| Chapter | Covers |
|---------|--------|
| [Long-Context Inference and the KV Cache](kv-cache/) | The real cost of serving, GQA and MLA, paged attention, prefix caching, batching |
| [Serving LLM Inference at Scale](inference-serving/) | Continuous batching, speculative decoding, disaggregation, parallelism, autoscaling |
| [Cost Optimization and Model Routing](cost-optimization/) | Routing, cascades, semantic caching, prompt compression, the quality-cost frontier |
| [Real-Time Streaming Chat](streaming-chat/) | Token streaming, transport, session memory, backpressure, voice pipelines |

### Retrieval and knowledge

| Chapter | Covers |
|---------|--------|
| [Serving Retrieval-Augmented Generation](rag-serving/) | Retrieval, chunking, the embedding service, ANN at scale, re-ranking, grounded generation, eval |
| [Semantic Search and Embedding Services](semantic-search/) | Vector index choice, recall vs latency, quantization, hybrid search, re-ranking |

### Building applications

| Chapter | Covers |
|---------|--------|
| [Agent Orchestration](agents/) | Tool-calling loops, planning, single vs multi-agent, cost and latency control |
| [Multimodal Serving](multimodal/) | Vision-language models, the projector, the image-token budget, serving cost |

### Quality and safety

| Chapter | Covers |
|---------|--------|
| [Evaluating LLM Systems](evaluation/) | Offline suites, LLM-as-judge, online A/B, regression gates |
| [Safety, Moderation, and Guardrails](safety/) | Input and output filtering, jailbreak and injection defense, PII, policy routing |
| [Production Monitoring and Observability](monitoring/) | Tracing, online eval without labels, hallucination detection, drift, regression |

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

Open a chapter folder and read its sections in order (start at the folder's
README). Each chapter ends with an interview Q&A and a self-test, and links a set
of first-party production engineering writeups. Read a chapter, open its figures,
attempt the questions, then follow the further reading to see how real teams
shipped the same system. Built by [Neurarch](https://www.neurarch.com).
