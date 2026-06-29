# 02 - Long-context inference and the KV cache

> **Interviewer:** "We serve a chat product. Contexts are getting long, GPU bills
> are climbing, and p99 latency is bad under load. Walk me through what is
> actually expensive about serving an LLM and how you would bring the cost down
> without hurting quality."

This is the question that separates people who have served models from people who
have only called an API. The whole thing hinges on one data structure: the KV
cache. Get that right and the rest follows.

## 1. Clarify and scope

- **Workload shape?** Long prompts and short answers (RAG, summarization) versus
  short prompts and long answers (code generation, agents) have opposite cost
  profiles. Ask.
- **Context lengths?** "Long context" could mean 8k or 200k. The number changes
  the answer completely.
- **Latency target?** First-token latency (driven by prefill) versus
  inter-token latency (driven by decode) are different problems.
- **Quality floor?** How much can you trade? Quantization and a smaller model are
  on the table only if there is headroom.

## 2. The cost model (say this part out loud)

Inference has two phases with different cost characteristics:

- **Prefill:** process the whole prompt at once. Compute-bound, highly parallel,
  fast per token. This sets your **first-token latency**.
- **Decode:** generate output one token at a time. Each step reads the entire
  model and the entire KV cache from memory to produce one token. It is
  **memory-bandwidth bound**, not compute bound. This sets your
  **inter-token latency** and dominates cost for long outputs.

The reason decode is bandwidth-bound is the **KV cache**.

### What the KV cache is and why it dominates

When a transformer generates token by token, it caches the keys and values of
every past token so it does not recompute them. That cache grows with sequence
length, batch size, and model size. For long-context inference, the KV cache, not
the weights, is often what fills your GPU memory.

A rough size formula worth knowing:

```
kv_bytes ≈ 2 (K and V) × layers × seq_len × kv_heads × head_dim × bytes_per_elem × batch
```

The two factors you control architecturally are **kv_heads** and **head_dim**.
That is the whole story behind the attention variants below: they all attack this
formula.

## 3. The levers, from cheapest to deepest

### Lever 1: batching (continuous batching)

The GPU is underused serving one request at a time. **Continuous batching**
(a.k.a. in-flight batching) interleaves many requests, adding and retiring them
at the token level instead of waiting for a whole batch to finish. This is the
single biggest throughput win and the first thing to reach for. The cost: the KV
cache now holds many sequences at once, so memory becomes the binding constraint,
which motivates everything below.

### Lever 2: paged attention

The KV cache for a sequence does not have to be contiguous. **Paged attention**
(the idea behind vLLM) manages the cache in fixed-size blocks like virtual
memory pages, which kills fragmentation and lets you pack far more concurrent
sequences into the same GPU memory. Mention it as the reason modern serving
stacks fit so many requests at once.

### Lever 3: prefix caching

In chat and RAG, many requests share a prefix (the system prompt, a long shared
document). **Prefix caching** reuses the KV cache for that shared prefix across
requests instead of recomputing prefill every time. Huge win for RAG with a
fixed instruction block or multi-turn chat.

### Lever 4: quantization

Quantize weights (and optionally the KV cache) to 8-bit or 4-bit. Shrinks memory
and increases bandwidth headroom, which directly speeds up decode. The tradeoff
is a quality hit that you must measure, not assume. KV-cache quantization
specifically is a clean win for long-context memory pressure.

### Lever 5: the attention architecture itself

This is the deep dive that wins the interview, because it is about *why* models
like Llama-3 and DeepSeek-V3 are shaped the way they are. They all attack the
cache term in the formula above, but in different ways: GQA shrinks `kv_heads`
directly, while MLA does not shrink that term at all, it replaces the cached keys
and values with a single low-rank latent (more below).

**Multi-head attention (MHA):** every query head has its own key and value head.
Maximum quality, maximum cache. This is the baseline the others optimize against.

**Grouped-query attention (GQA):** share one key/value head across a group of
query heads. If you have 32 query heads and 8 KV heads, you have cut the KV cache
4x with little quality loss. This is what Llama-3 uses, and it is the current
default for good reason.

**Multi-head latent attention (MLA):** DeepSeek-V3's approach, and a notch
sharper than GQA. Instead of caching keys and values at all, MLA caches a single
low-rank **latent** vector per token and reconstructs the per-head keys and
values from it on the fly:

1. Project the token down into a small latent (a down-projection to, say, 512
   dims).
2. Cache only that latent.
3. At attention time, up-project the latent back into keys and values for all
   heads.

The keys and values you attend over are never stored at full width. You store the
compressed version and pay a tiny matmul to expand it. The KV cache shrinks by
roughly an order of magnitude, and long-context serving stops being dominated by
cache memory.

There is one wrinkle worth knowing, and naming it signals real depth. Rotary
position embeddings (RoPE) do not commute cleanly with this compression, because
RoPE is applied per position and the latent is position-free. DeepSeek's answer
is to **split the head dimension**: part of it carries RoPE the normal way, part
goes through the compressed latent path. So each head is effectively two
concatenated pieces, one positional and one latent. This is the detail most
casual diagrams of DeepSeek-V3 quietly get wrong or skip.

### Lever 6: fewer or cheaper parameters per token (MoE)

Orthogonal to the cache, but on the same bill. A **mixture-of-experts** model
has many feed-forward "experts" but routes each token to only a couple of them,
so the active parameter count per token is a fraction of the total. Mixtral and
DeepSeek-V3 both do this. It cuts the compute and bandwidth per token while
keeping the model's total capacity high. The cost is memory (all experts live on
the GPU) and routing complexity.

### Lever 7: fewer decode steps (speculative decoding)

A small "draft" model proposes several tokens; the big model verifies them in one
parallel forward pass, accepting the longest correct prefix. When the draft is
often right, you get multiple tokens per big-model step. Net effect: fewer
expensive decode steps for the same output. Covered further in the planned
serving topic.

## 4. Putting it together

A strong closing synthesis:

> Start with continuous batching and paged attention for throughput. Add prefix
> caching because our RAG prompts share a long instruction block. If memory is
> still the wall on long contexts, quantize the KV cache and consider a model
> with GQA or latent attention rather than full MHA. If compute per token is the
> wall, an MoE model gives capacity without paying for every parameter on every
> token. Pick based on whether we are memory-bound or compute-bound, which I
> would confirm by profiling prefill versus decode.

That answer shows you can reason from the cost model to the architecture, which
is exactly the signal.

## 5. Failure modes

- **OOM under load:** the KV cache for concurrent long sequences exceeds GPU
  memory. Cap concurrent sequences, page the cache, quantize it.
- **First-token latency spikes:** prefill of very long prompts. Prefix caching
  and chunked prefill help.
- **Quality regression from quantization:** always gate behind an eval, never
  ship on vibes.

## 6. Likely follow-ups

- "Why is decode bandwidth-bound and prefill compute-bound?" Decode does one
  token of compute but reads the whole model plus cache; prefill does many tokens
  of compute per memory read.
- "GQA vs MLA, when would you pick which?" GQA is simpler and the safe default;
  MLA wins when the KV cache is your binding constraint and you control the model.
- "Estimate the KV cache for 100k context." Plug into the formula; show you can
  do arithmetic about GPU memory.

---

## Seen in production

Real systems that ship the patterns above. Each is a first-party engineering
writeup; read them for what an interview answer skips: who the system serves,
the product design, the eval bar, and the deployment shape.

- **vLLM (UC Berkeley)** [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180): OS-style KV-cache paging cuts fragmentation, boosting throughput 2x to 4x. *(deployment)*
- **Character.AI** [Optimizing AI Inference at Character.AI](https://blog.character.ai/optimizing-ai-inference-at-character-ai-2/): MQA, hybrid local/global attention, and cross-layer KV-sharing cut cost 33x. *(deployment)*
- **DeepSeek** [DeepSeek-V2: a strong, economical, efficient MoE language model](https://arxiv.org/abs/2405.04434): Multi-head Latent Attention compresses the KV cache into a latent vector, shrinking it 93%. *(product design)*
- **Google Research** [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245): Grouped-query attention trades KV heads for speed at near-MHA quality. *(product design)*

- **NVIDIA** [5x faster time to first token with TensorRT-LLM KV cache early reuse](https://developer.nvidia.com/blog/5x-faster-time-to-first-token-with-nvidia-tensorrt-llm-kv-cache-early-reuse/): Early KV reuse, flexible block sizing, and smart eviction cut TTFT. *(deployment)*
- **NVIDIA** [Optimizing inference with NVFP4 KV cache](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/): 4-bit KV cache halves memory vs FP8, doubling context with under 1% loss. *(deployment)*
- **Databricks** [Inference-Friendly Models with MixAttention](https://www.databricks.com/blog/mixattention): Cross-layer KV sharing plus sliding-window attention shrinks the cache. *(product design)*
- **Databricks** [Accelerating LLM inference with prompt caching](https://www.databricks.com/blog/accelerating-llm-inference-prompt-caching-open-source-models-databricks): Automatic prefix KV reuse: 2.5x throughput, 3x lower P50 latency. *(deployment)*
- **llm-d** [KV-Cache wins you can see: prefix caching to distributed scheduling](https://llm-d.ai/blog/kvcache-wins-you-can-see): Single-instance prefix caching breaks in clusters; cache-aware scheduling fixes it. *(deployment)*
- **LMSYS / SGLang** [Fast and expressive LLM inference with RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/): A radix-tree KV cache enables automatic cross-request prefix reuse. *(deployment)*
- **Together AI** [Serving MiniMax-M3: 1M-token context without regrets](https://www.together.ai/blog/serving-minimax-m3-for-efficient-inference-unlocking-1m-token-context-and-multimodality-without-regrets): Paged sparse attention and KV-block-major kernels make 1M-token serving practical. *(deployment)*
- **Hugging Face** [Unlocking longer generation with KV cache quantization](https://huggingface.co/blog/kv-cache-quantization): Per-token int4 KV quantization yields about 2.5x memory savings. *(product design)*
- **KIVI** [A tuning-free asymmetric 2-bit quantization for KV cache](https://arxiv.org/abs/2402.02750): Per-channel keys and per-token values enable 2-bit KV compression. *(product design)*

- **MIT / Meta** [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453): The attention-sink insight lets fixed-window LLMs stream to millions of tokens. *(product design)*
- **Anthropic** [Prompt caching with Claude](https://claude.com/blog/prompt-caching): Caches reused context across API calls, cutting cost up to 90% and latency 85%. *(product design)*
- **Colfax / Together** [FlashAttention-3: fast, accurate attention with asynchrony and low precision](https://arxiv.org/abs/2407.08608): Hopper-optimized attention via warp-specialization and FP8, 1.5-2x faster. *(deployment)*
- **UT Austin / Stanford** [H2O: Heavy-Hitter Oracle for efficient generative inference](https://arxiv.org/abs/2306.14048): KV-cache eviction keeping recent plus heavy-hitter tokens, up to 29x throughput. *(deployment)*
- **UIUC / Cohere** [SnapKV: LLM knows what you are looking for before generation](https://arxiv.org/abs/2404.14469): Fine-tuning-free KV compression via observation-window clustering, 3.6x faster decode. *(product design)*
- **Fireworks AI** [FireAttention V2: long contexts practical for online inference](https://fireworks.ai/blog/fireattention-v2-long-context-inference): Production long-context kernels with FP8 prefill and multi-host deployment. *(deployment)*
- **Microsoft** [MInference 1.0: accelerating pre-filling via dynamic sparse attention](https://arxiv.org/abs/2407.02490): Dynamic sparse attention cuts long-context prefill latency up to 10x. *(deployment)*

More production case studies: the [Evidently AI ML system design database](https://www.evidentlyai.com/ml-system-design) (800 case studies from 150+
companies) is the broadest curated index; this section pulls the ones that map
directly onto this topic.

---
## Trace the architectures

This topic is most of why the diagrams in this repo are validated graphs and not
pictures: the dimensions are the whole argument. "The latent is 512 dims" is the
kind of claim that is wrong half the time it gets copied. Open these and read the
real numbers off the graph.

- **GQA baseline (Llama-3 8B):**
  [open it live](https://www.neurarch.com/?import=https://raw.githubusercontent.com/neurarch-ai/awesome-llm-model-zoo/main/architectures/llama3-8b/model.json).
  Find the attention block and compare the number of query heads to KV heads;
  that ratio is the cache saving.

  ![Llama-3 8B](https://raw.githubusercontent.com/neurarch-ai/awesome-llm-model-zoo/main/architectures/llama3-8b/assets/diagram.png)

- **MLA + MoE (DeepSeek-V3):**
  [open it live](https://www.neurarch.com/?import=https://raw.githubusercontent.com/neurarch-ai/awesome-llm-model-zoo/main/architectures/deepseek-v3/model.json).
  The 60-odd identical transformer blocks are folded into one with a "x N" badge
  so the latent attention and the expert routing are actually visible instead of
  buried in a hundred-layer scroll. Trace the RoPE-versus-latent head split.

  ![DeepSeek-V3](https://raw.githubusercontent.com/neurarch-ai/awesome-llm-model-zoo/main/architectures/deepseek-v3/assets/diagram.png)

- **MoE routing on its own (Mixtral block):**
  [open it live](https://www.neurarch.com/?import=https://raw.githubusercontent.com/neurarch-ai/awesome-llm-model-zoo/main/architectures/mixtral-block/model.json)
  to see the router send each token to a top-k of experts.

  ![Mixtral block](https://raw.githubusercontent.com/neurarch-ai/awesome-llm-model-zoo/main/architectures/mixtral-block/assets/diagram.png)

A good exercise before an interview: open DeepSeek-V3, swap MLA back to plain GQA,
and watch the KV-cache estimate change. The graphs are real dimensions,
shape-checked end to end. All 92 are in the
[Model Zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo)
([gallery](https://neurarch-ai.github.io/awesome-llm-model-zoo)). Built by
[Neurarch](https://www.neurarch.com).
