# 7. How teams do it in production

Every production LLM serving system runs the same two-phase loop: prefill builds
the KV cache once, then decode reuses it one token at a time. What diverges between
companies is where they cut the cache, how they manage memory, whether they reuse
prefill work, and what their primary constraint is. The architecture is shared; the
leverage is in which levers they pulled and in what order.

## Where the real designs diverge

| System | KV reduction strategy | Memory management | Prefix or prompt caching | Primary optimization | When it wins | When to watch out |
|---|---|---|---|---|---|---|
| vLLM (UC Berkeley) | None intrinsic | OS-style paging in fixed blocks | Yes, block-level with copy-on-write | Throughput via packing more concurrent sequences | Many concurrent variable-length sequences; fragmentation is the bottleneck | Per-node cache does not span a cluster; needs cache-aware routing at scale |
| Character.AI | MQA plus cross-layer KV sharing plus native int8 | Sliding-window local/global hybrid | Yes, rolling-hash LRU tree across turns (~95% hit rate) | Memory and cost (33x reduction since 2022) | High-traffic chat with heavy prompt reuse and aggressive cost targets | MQA plus int8 stacks multiple quality risks; validate on your eval |
| DeepSeek-V2/V3 (MLA) | Latent-vector compression (~93% vs MHA) | Standard paged blocks | Not the focus | Long-context KV memory | You train or control the model and KV cache is the binding constraint | RoPE does not commute with the latent path; requires the split-head fix at training time |
| Google (GQA, Llama 3) | Fewer KV heads (4x to 8x reduction) | Standard | Not the focus | Throughput and memory at near-MHA quality | Safe default for any model at most serving scales | Fixed ratio; more aggressive situations call for MLA |
| SGLang (RadixAttention) | None intrinsic | Radix tree with LRU eviction | Yes, automatic cross-request for branching prefixes | Throughput and TTFT for agent trees and few-shot workloads | Many requests with shared branching prefixes | Benefit collapses when prefixes rarely overlap; LRU can thrash under diverse traffic |
| NVIDIA TensorRT-LLM (NVFP4) | 4-bit KV quantization (4x vs FP16) | Dependency-aware block eviction | Yes, early reuse while prefill still running | Long-context memory and TTFT (up to 5x faster) | Long-prompt system-prompt-heavy workloads on Hopper GPUs | Under-1% accuracy claim is benchmark-dependent; validate on your tasks |
| Hugging Face (KV quant) | Per-token int4/int2 quantization | Full-precision recent-token window | Not the focus | Memory for fixed models you cannot retrain | Long-generation memory pressure on a locked checkpoint | Degrades on retrieval-heavy tasks; keep a full-precision window and eval |
| Anthropic (prompt caching) | None intrinsic | TTL-scoped server-side cache per API user | Yes, API-level across calls (Claude 3/3.5/3.7) | Cost and latency for large shared contexts (up to 90% cost reduction) | Large fixed context reused across many API calls | No benefit when context changes every call; cache TTL expiry resets the savings |
| Databricks (prompt caching) | None intrinsic | Volatile per-tenant cache | Yes, automatic prefix matching across requests | Throughput and P50 latency (2.5x, 3x at 30% hit rate) | Repeated system prompts in multi-tenant deployments | Multi-tenant isolation is mandatory; exact-prefix matching misses on any token mismatch |
| StreamingLLM (MIT/Meta) | Fixed sliding window plus sink tokens | Rolling window; first tokens always kept | Not the focus | Unbounded streaming at fixed KV budget | Never-ending streams where losing the middle is acceptable | Evicted tokens are genuinely lost; fails on whole-document recall |
| llm-d (KV-aware scheduling) | None intrinsic | Distributed cache-aware routing | Yes, cross-node prefix routing | Cluster-wide prefix hit rate | Multi-node deployments where single-node prefix caches fragment | Added routing complexity; benefit depends on traffic locality |

The core dividing line: a system either **shrinks each KV entry** (GQA, MLA,
quantization), **reuses entries across requests** (paged, radix, and prefix caching),
or **drops entries entirely** (eviction, sliding windows). Most production systems
layer two or three of these together.

## The systems (first-party write-ups)

- **vLLM (UC Berkeley)** [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180): OS-style KV-cache paging kills fragmentation and boosts throughput 2x to 4x. *(deployment)*
- **Character.AI** [Optimizing AI Inference at Character.AI](https://blog.character.ai/optimizing-ai-inference-at-character-ai-2/): MQA, hybrid local/global attention, cross-layer KV sharing, and int8 cut cost 33x. *(deployment)*
- **DeepSeek** [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434): Multi-head Latent Attention compresses the KV cache by ~93% via a per-token latent vector. *(product design)*
- **Google Research** [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245): Grouped-query attention at near-MHA quality with a smaller cache; uptraining recipe included. *(product design)*
- **NVIDIA** [5x Faster Time to First Token with TensorRT-LLM KV Cache Early Reuse](https://developer.nvidia.com/blog/5x-faster-time-to-first-token-with-nvidia-tensorrt-llm-kv-cache-early-reuse/): Early KV reuse, flexible block sizing, and dependency-aware eviction cut TTFT. *(deployment)*
- **NVIDIA** [Optimizing Inference with NVFP4 KV Cache](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/): 4-bit KV halves memory vs FP8 with under 1% accuracy loss. *(deployment)*
- **Databricks** [Inference-Friendly Models with MixAttention](https://www.databricks.com/blog/mixattention): Cross-layer KV sharing plus sliding-window attention shrinks the cache in a production model. *(product design)*
- **Databricks** [Accelerating LLM Inference with Prompt Caching](https://www.databricks.com/blog/accelerating-llm-inference-prompt-caching-open-source-models-databricks): Automatic prefix KV reuse delivers 2.5x throughput and 3x lower P50 latency. *(deployment)*
- **llm-d** [KV-Cache Wins You Can See](https://llm-d.ai/blog/kvcache-wins-you-can-see): Single-instance prefix caching breaks in clusters; cache-aware scheduling fixes it. *(deployment)*
- **SGLang / LMSYS** [Fast and Expressive LLM Inference with RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/): Radix-tree KV cache enables automatic cross-request prefix reuse for branching workloads. *(deployment)*
- **Hugging Face** [Unlocking Longer Generation with KV Cache Quantization](https://huggingface.co/blog/kv-cache-quantization): Per-token int4 KV quantization yields about 2.5x memory savings on a fixed model. *(product design)*
- **Anthropic** [Prompt Caching with Claude](https://www.anthropic.com/news/prompt-caching): Server-side cache across API calls cuts cost up to 90% and latency 85% for large shared contexts. *(product design)*
- **MIT/Meta** [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453): The attention-sink insight lets fixed-window LLMs stream to millions of tokens stably. *(product design)*
- **Together AI** [Serving MiniMax-M3: 1M-token context without regrets](https://www.together.ai/blog/serving-minimax-m3-for-efficient-inference-unlocking-1m-token-context-and-multimodality-without-regrets): Paged sparse attention and KV-block-major kernels make 1M-token serving practical. *(deployment)*
- **KIVI** [A Tuning-Free Asymmetric 2-bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750): Per-channel keys and per-token values enable 2-bit KV compression with a full-precision recent-token window. *(product design)*
- **UT Austin / Meta** [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048): KV eviction keeping recent tokens plus heavy-hitters, up to 29x throughput. *(deployment)*

## Trace the architectures (Model Zoo)

The numbers in the formulas above are real model dimensions, not invented for
illustration. The graphs below are validated, shape-checked architecture graphs
you can open and inspect live:

- **GQA baseline (Llama 3 8B):** [open it live](https://www.neurarch.com/?import=https://raw.githubusercontent.com/neurarch-ai/awesome-llm-model-zoo/main/architectures/llama3-8b/model.json). Find the attention block; compare query head count to KV head count. That ratio is the cache saving.

  ![Llama-3 8B](https://raw.githubusercontent.com/neurarch-ai/awesome-llm-model-zoo/main/architectures/llama3-8b/assets/diagram.png)

- **MLA plus MoE (DeepSeek-V3):** [open it live](https://www.neurarch.com/?import=https://raw.githubusercontent.com/neurarch-ai/awesome-llm-model-zoo/main/architectures/deepseek-v3/model.json). Trace the RoPE sub-dimension versus the compressed-latent sub-dimension through the attention block.

  ![DeepSeek-V3](https://raw.githubusercontent.com/neurarch-ai/awesome-llm-model-zoo/main/architectures/deepseek-v3/assets/diagram.png)

All validated architecture graphs are in the [Model Zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo)
([gallery](https://neurarch-ai.github.io/awesome-llm-model-zoo)). Built by [Neurarch](https://www.neurarch.com).
