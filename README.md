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

> **Interviews ask end-to-end questions, not "explain RAG."** The
> [**question bank**](questions.md) maps the questions you actually get ("design a
> support agent", "make our serving cheaper") to the component topics below.
> Start there if you are practicing for a specific loop.

---

## Topics

Two ways in. **By question** (how interviews are actually posed): start with the
[**question bank**](questions.md), which maps prompts like "design a customer
support agent" or "make our LLM serving cheaper" to the topics below. **By stage
of the LLM stack** (below): the topics are grouped by where they sit in a
production LLM system. Unlike classic ML, which splits by use case (recommenders,
fraud, vision), the LLM stack is better organized by pipeline stage; these five
groups are where the 800-case database's GenAI, RAG, and AI-agent examples
actually live. The companion [ML System Design Interview](https://github.com/neurarch-ai/awesome-ml-system-design)
repo groups the classic-ML half by use case, mirroring the same
[Evidently AI 800-case database](https://www.evidentlyai.com/ml-system-design).

### Retrieval and knowledge
*Get the right context in front of the model.*

| # | Topic | What it teaches |
|---|-------|-----------------|
| 01 | [RAG serving](topics/01-rag-serving.md) | Retrieval, chunking, embedding service, the read path, freshness, eval |
| 08 | [Semantic search and embedding service](topics/08-semantic-search-and-embeddings.md) | Vector index choice, recall vs latency, re-ranking, hybrid search |

### Inference and serving (the cost layer)
*What an LLM actually costs to run, and how to serve it at scale.*

| # | Topic | What it teaches |
|---|-------|-----------------|
| 02 | [Long-context inference and the KV cache](topics/02-long-context-and-kv-cache.md) | The real cost of serving, GQA vs MLA, paged attention, prefix caching, batching |
| 04 | [LLM inference serving at scale](topics/04-inference-serving-at-scale.md) | Continuous batching, speculative decoding, tensor parallelism, autoscaling |
| 10 | [Realtime streaming chat](topics/10-realtime-streaming-chat.md) | Token streaming, session memory, websockets, backpressure |
| 11 | [Cost optimization and model routing](topics/11-cost-optimization-and-model-routing.md) | Model routing, cascades, semantic caching, prompt compression, right-sizing, the quality-cost frontier |

### Building applications on top
*Compose the model into a product capability.*

| # | Topic | What it teaches |
|---|-------|-----------------|
| 03 | [Agent orchestration](topics/03-agent-orchestration.md) | Tool-calling loops, planning, state, multi-agent, cost and latency control |
| 09 | [Multimodal serving](topics/09-multimodal-serving.md) | Vision-language models, the projector, image token budget |

### Adapting the model
*Change the model's behavior when prompting and retrieval are not enough.*

| # | Topic | What it teaches |
|---|-------|-----------------|
| 05 | [Fine-tuning and post-training pipeline](topics/05-post-training-pipeline.md) | LoRA, SFT, DPO/RLHF, data curation, eval gates |

### Quality and safety
*Know it works, and keep it from misbehaving.*

| # | Topic | What it teaches |
|---|-------|-----------------|
| 06 | [LLM evaluation system](topics/06-evaluation-system.md) | Offline suites, LLM-as-judge, online A/B, regression gates |
| 07 | [Safety, moderation, and guardrails](topics/07-safety-and-guardrails.md) | Input/output filtering, jailbreak defense, PII, policy routing |
| 12 | [Production monitoring and observability](topics/12-production-monitoring-and-observability.md) | Tracing, online eval without labels, LLM-as-judge, hallucination detection, drift, regression |

All twelve topics are written and ready.

**Going deeper than the whiteboard?** [**deep-dives.md**](deep-dives.md) is a
bank of ~135 rapid-fire, depth-probing questions: the follow-ups an interviewer
pulls once the design is on the board (normalization and attention variants, where KL divergence shows up, optimization and gradient descent, decoding and sampling, generative model families, distributed-training failure modes, and quantization tradeoffs). Each answer leads with the mechanism, then the tradeoff.

See [topics/README.md](topics/README.md) for the full roadmap and how to
contribute a topic.

Every topic ends with a **Seen in production** section: real first-party
engineering writeups (Anthropic, DoorDash, Uber, Character.AI, LinkedIn, and
more) tagged by what they illustrate, so the framework is grounded in shipped
systems rather than whiteboard theory. For the broadest index of production ML
and LLM systems, see the
[Evidently AI ML system design database](https://www.evidentlyai.com/ml-system-design)
(800 case studies from 150+ companies).

Prefer to browse the whole set by category in one place? Every entry is rolled
up into [**CASE-STUDIES.md**](CASE-STUDIES.md).

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
