# LLM System Design Interview

A practical guide to the system design questions you actually get asked when
interviewing for roles that build with large language models: applied scientist,
ML engineer, LLM infra, and the growing "AI engineer" bucket.

Most system design prep material predates the LLM era. It teaches you how to
shard a database and design a URL shortener. Useful, but it does not prepare you
for "design a RAG system that answers questions over 50M internal documents" or
"our agent is slow and expensive, walk me through how you would fix it." Those
questions have their own vocabulary: retrieval, KV cache, continuous batching,
speculative decoding, eval harnesses, guardrails, tool-calling loops.

This repo covers that vocabulary, organized as interview-ready walkthroughs.

Every model architecture in here is a **validated reference graph**, not a
hand-drawn box diagram. Each one is checked end to end: tensor shapes,
attention head divisibility, parameter counts within roughly 10 percent of the
published weights. You can open any of them, edit the structure, and watch the
numbers change. More on why that matters [below](#about-the-diagrams).

---

## How to use this repo

1. **Read the [answer framework](framework/answer-framework.md) first.** It is the
   spine every walkthrough hangs off. Interviewers grade structure as much as
   content; a strong framework keeps you from rambling.
2. **Work through the topics in order.** They build on each other. RAG introduces
   retrieval and serving basics; the inference topic goes deep on the cost model;
   agents compose both.
3. **For each topic, try to draft your own answer before reading ours.** Then
   compare. The gaps are your study list.
4. **Open the architectures live.** Where a topic touches a real model (MoE
   routing, latent attention, hybrid SSM), there is a link that loads the actual
   graph so you can trace the data flow yourself instead of trusting a static
   picture.

---

## Topics

| # | Topic | What it teaches | Status |
|---|-------|-----------------|--------|
| 01 | [RAG serving](topics/01-rag-serving.md) | Retrieval, chunking, embedding service, the read path, freshness, eval | Ready |
| 02 | [Long-context inference and the KV cache](topics/02-long-context-and-kv-cache.md) | The real cost of serving, GQA vs MLA, paged attention, prefix caching, batching | Ready |
| 03 | [Agent orchestration](topics/03-agent-orchestration.md) | Tool-calling loops, planning, state, multi-agent, cost and latency control | Ready |
| 04 | [LLM inference serving at scale](topics/04-inference-serving-at-scale.md) | Continuous batching, speculative decoding, tensor parallelism, autoscaling | Ready |
| 05 | [Fine-tuning and post-training pipeline](topics/05-post-training-pipeline.md) | LoRA, SFT, DPO/RLHF, data curation, eval gates | Ready |
| 06 | [LLM evaluation system](topics/06-evaluation-system.md) | Offline suites, LLM-as-judge, online A/B, regression gates | Ready |
| 07 | [Safety, moderation, and guardrails](topics/07-safety-and-guardrails.md) | Input/output filtering, jailbreak defense, PII, policy routing | Ready |
| 08 | [Semantic search and embedding service](topics/08-semantic-search-and-embeddings.md) | Vector index choice, recall vs latency, re-ranking, hybrid search | Ready |
| 09 | [Multimodal serving](topics/09-multimodal-serving.md) | Vision-language models, the projector, image token budget | Ready |
| 10 | [Realtime streaming chat](topics/10-realtime-streaming-chat.md) | Token streaming, session memory, websockets, backpressure | Ready |

See [topics/README.md](topics/README.md) for the full roadmap and how to
contribute a topic.

> A companion [**ML System Design Interview**](https://github.com/neurarch-ai/awesome-ml-system-design)
> repo (recommenders, ranking, feature pipelines, the classic non-LLM stack) is
> also available. This one covers the LLM-era stack; the companion covers the
> foundations underneath it.

---

## About the diagrams

Every architecture diagram links back to [Neurarch](https://www.neurarch.com),
where it lives as a structured graph rather than an image.

This is a deliberate choice, and it matters for interview prep specifically. A
picture of an architecture cannot be wrong in a way you can catch. A validated
graph can. The widely copied Llama-3 diagram floating around, for example, lists
the FFN size as 11008, which is actually Llama-2's number. Llama-3 is 14336.
Nobody noticed for a long time because a `.png` has no ground truth to check
against.

When you are about to explain GQA or MoE routing or latent attention to an
interviewer, you want the dimensions in your head to be the real ones. So the
diagrams here are not screenshots. Each is a link that opens the actual graph,
at real dimensions, that you can inspect and modify:

- **Click to open it live** and the architecture loads onto a canvas where you
  can trace every tensor shape, fold the repeated transformer blocks, and change
  a hyperparameter to see what breaks.
- **The source graphs are open** in the
  [Model Zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo) (87
  architectures, MIT). Each entry is a real graph at real dimensions plus
  verified hyperparameters.

If you have never traced a model's shapes by hand, doing it once before an
interview is worth more than reading ten blog posts. The links are there so you
can.

---

## License

MIT. See [LICENSE](LICENSE). Contributions welcome; see
[topics/README.md](topics/README.md).

Built and maintained by the team behind [Neurarch](https://www.neurarch.com).
