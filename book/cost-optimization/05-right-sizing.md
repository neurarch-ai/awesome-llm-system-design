# 5. Right-sizing

The largest cost mistake in most systems is wiring a single frontier model into
every subtask because it was easiest to set up, then optimizing around it. Most
of the traffic in a real pipeline does not need the frontier model. Match the
model to the task and most of the bill disappears before routing or caching
even runs.

## Matching model size to task

Think of the pipeline as a funnel. At each stage the goal is narrower and the
task is more tractable; a smaller, more specialized model can handle it better
and cheaper than a general frontier model.

| Pipeline stage | What it needs | Right size |
|---|---|---|
| Intent classification / routing | A label (easy / hard / category) | Fine-tuned classifier, 1B or less; or a regex/keyword layer |
| Embedding (for caching, RAG, search) | A dense vector | Dedicated embedding model (e.g., all-MiniLM-L6, 22M params) |
| Reranking retrieved chunks | A relevance score, not text | Small cross-encoder, 100M range |
| Short answer / lookup / extraction | A constrained, formulaic response | Fine-tuned small model (7-13B) or a structured-output call |
| Reasoning, code, long generation | Open-ended, complex output | Frontier model (only these reach here) |

A fine-tuned small model on a narrow task often beats a giant general model on
that task, at a fraction of the cost. Anyscale found that a Llama-3 8B
classifier outperformed GPT-4-as-a-router on routing quality; IBM showed that a
well-matched 13B specialist beat Llama-2 70B on specific benchmark tasks. The
gain comes from task-specific fine-tuning, not from model size.

## Quantization: more throughput at lower cost per token

Quantization reduces the number of bytes per model parameter, which on
bandwidth-bound decode (reading weights from memory per generated token) translates
directly to more tokens per second on the same GPU. At FP8 (8-bit float), Baseten
measured a 33% gain in tokens-per-second and 24% lower cost per million tokens
versus FP16 on a Mistral 7B deployed on H100, with near-zero perplexity change.
INT4 goes further at more quality risk.

This is a **self-hosting lever only**. On a per-token API your pricing is set by
the provider; quantization on their side is already baked in. The self-host versus
API break-even is:

$$Q^* = \frac{c_{\text{gpu/hour}}}{3600 \cdot t_{\text{tok}} \cdot c_{\text{api/tok}}}$$

where $t_{\text{tok}}$ is tokens per request and $c_{\text{api/tok}}$ is the
API price per token. Above QPS $Q^*$, the fixed GPU cost beats per-token API
pricing; below it, the API wins and you are paying for idle GPU time. Do the
arithmetic before self-hosting.

## Distillation: bake the frontier model's knowledge into a small one

Knowledge distillation trains a small student model to mimic a large teacher's
outputs on your specific task distribution. On a well-defined, stable task (a
classification, a template-fill, a constrained extraction) the student often
matches 95-98% of the teacher's quality at a tenth of the cost. The upfront
investment is a labeled dataset (teacher outputs on your traffic) and a training
run; the ongoing savings can be large if QPS is high.

Distillation is worth it when: the task is stable (the teacher's behavior does
not shift week to week), QPS is high enough to justify the training run, and you
can measure quality precisely enough to know when the student regresses.

## Right-sizing the provider, not just the model

Once you have picked the smallest model that clears the quality bar, there is a
second lever that costs nothing to pull: **which provider serves it.** For a model
you do not self-host, the same weights vary several-fold in price per token and in
output speed across hosting providers, because each runs a different engine,
batching, quantization, and hardware. So right-sizing has two axes, the model and
the host, and you pick the point on the price-and-speed frontier that meets your
latency SLO. Bring the **median (p50) price and speed per provider** over a rolling
window to the decision, not a single noisy sample; independent trackers such as
[Artificial Analysis](https://artificialanalysis.ai/) publish exactly that per
provider, and the frontier moves, so re-check it periodically.

## The operational cost of more models

Every additional model in the system has a maintenance cost: it must be evaluated,
deployed, monitored for silent quality regression, and updated when the task
distribution drifts. A system with a classifier router, an embedding model, a
reranker, a small generative model, and a frontier model has five quality surfaces
that can each silently regress. The right-sizing strategy works best when the
subtasks are stable and the quality eval per model is automated.

## When to use which

| Reach for | When | Instead of |
|---|---|---|
| Fine-tuned small model (1-13B) | Narrow, stable task (classify, extract, label) where task-specific training data is available | Frontier model for every subtask, which is correct but expensive |
| Dedicated embedding model | Any vector embedding need (caching, RAG, semantic search) | A generative frontier call to produce embeddings (much more expensive, not better) |
| Small cross-encoder reranker | Reranking a retrieved shortlist (50-100 items) for relevance | A full generative model scoring each chunk individually |
| Quantization (FP8, INT8, INT4) | Self-hosted model above the QPS break-even $Q^*$ where fixed GPU cost beats API price | API pricing, where the lever does not exist on your side |
| Distillation | High-QPS stable task where a training run is justified by the volume of calls | A one-off or low-QPS task where the training investment never pays back |
| Mixture-of-Experts (Mixtral style) | You want large-model capacity but can control your own serving: only 2-3 experts fire per token | Dense models where every parameter pays for every token |
| Batch API (provider-side) | Bulk work with no user waiting (backfills, nightly summarization, offline eval generation) | Interactive endpoint, which charges online prices for offline work |

**Tools.** Small fine-tunes and distilled students are built on PyTorch (Meta) and Hugging Face Transformers, often with PEFT/LoRA adapters for cheap task-specific training. Embeddings come from sentence-transformers (the all-MiniLM family), and rerankers are cross-encoders from the same ecosystem or ColBERT. Self-hosted quantization uses GPTQ, AWQ, or llama.cpp/GGUF served through vLLM or TensorRT-LLM (NVIDIA), while Mixtral-style MoE is served by vLLM and SGLang. The provider-side batch and per-token pricing levers are reached through a gateway such as LiteLLM.

**Worked example.** A content platform runs a moderation and summarization pipeline where nearly every request currently hits one frontier model. Walking the funnel, the team moves intent classification to a fine-tuned small model trained on their own labels rather than paying frontier prices for a label, uses a dedicated embedding model for their cache and search instead of a generative call, and adds a small cross-encoder to rerank retrieved passages rather than scoring each chunk with a large model. The frontier model stays only on the open-ended summary generation at the funnel's tip. Because their nightly summarization backfill has no user waiting, they route that volume through a provider batch API instead of the interactive endpoint, and only consider self-hosted quantization once a subtask's QPS clears the break-even where fixed GPU cost beats per-token pricing.
