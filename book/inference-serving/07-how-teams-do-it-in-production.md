# 7. How teams do it in production

Every production LLM serving stack converges on the same skeleton: continuous
batching that reshapes the batch every token step, a paged KV cache to prevent
fragmentation, a prefill pass that fills the cache, a bandwidth-bound decode loop
that reads the cache per token, and an SLO-driven autoscaler. What actually differs
between teams is which single stage each one pushed hardest, and that choice follows
directly from their workload.

## Where the real designs diverge

| System | Key optimization | Batching | Speculative decoding | Prefill/decode split | Quantization |
|---|---|---|---|---|---|
| Anyscale (vLLM) | PagedAttention for fragmentation | Continuous, iteration-level | No | One pool | BF16 baseline |
| Character.AI | KV footprint (MQA, cross-layer sharing, int8) | Continuous | No | One pool | INT8 weights and KV |
| LinkedIn | n-gram speculative decoding for templated output | Continuous (vLLM) | Yes, n-gram | One pool | Not stated |
| Baseten (BEI) | Token-budget packing, FP8, TensorRT-LLM | Token-budget continuous | Yes | No | FP8 (H100) |
| NVIDIA Dynamo | Disaggregated prefill/decode with KV-aware routing | Continuous | No | Yes (separate pools, separate machines) | Phase-specific TP |
| Together AI (ATLAS) | Online-adaptive speculative decoding | Continuous | Yes, adaptive | No | Not stated |
| Fireworks (FireOptimizer) | Workload-specialized draft model | Continuous | Yes, trained per workload | No | Per-workload config |
| Moonshot (Mooncake) | Pooled tiered KV cache (CPU/DRAM/SSD/object store) | Continuous | No | Yes | Not stated |
| Microsoft (Splitwise) | Phase-split onto separate hardware per phase | Continuous | No | Yes, separate physical machines | Phase-specific |
| Sarathi-Serve | Chunked prefill for stall-free scheduling | Stall-free chunked | No | One pool | Not stated |
| Modal | Engine selection, memory snapshots, weight streaming | Continuous | Optional | No | FP8 or INT4 per workload |
| Databricks | Hardware selection and batch-size guidance | Continuous | No | Prefill/decode as phases | Hardware-specific |

The core dividing line is disaggregation: Dynamo, Splitwise, DistServe, and
Mooncake split prefill and decode into separate pools or machines; everyone else
keeps one pool and smooths the tension with chunked prefill or speculative
decoding. The second line is speculative decoding: LinkedIn, Together, and
Fireworks invest in drafts; Character.AI and Anyscale invest in the KV footprint
instead. You only need to pick one primary lever that matches your bottleneck.

## The systems (first-party write-ups)

- **Anyscale** [How continuous batching enables 23x throughput in LLM inference](https://www.anyscale.com/blog/continuous-batching-llm-inference): Iteration-level scheduling plus PagedAttention beat static batching up to 23x on Meta OPT-13B on A100. Separates the scheduling win (~8x) from the memory win; the honest comparison against optimized static batching is 5-6x. *(deployment)*

- **Character.AI** [Optimizing AI Inference at Character.AI](https://blog.character.ai/optimizing-ai-inference-at-character-ai/): MQA, cross-layer KV sharing, and INT8 weight plus KV quantization cut serving cost 13.5x versus commercial APIs at roughly 20,000 QPS. Aggressive KV reduction must be trained into the architecture, not switched on at serving. *(deployment)*

- **LinkedIn** [Accelerating LLM inference with speculative decoding](https://www.linkedin.com/blog/engineering/ai/accelerating-llm-inference-with-speculative-decoding-lessons-from-linkedins-hiring-assistant): N-gram speculative decoding on the Hiring Assistant gave nearly 4x throughput and 66% lower P90 latency; high acceptance because output echoes job-description text. *(eval bar)*

- **Baseten** [How we built BEI: high-throughput embedding, reranker, classifier inference](https://www.baseten.co/blog/how-we-built-bei-high-throughput-embedding-inference/): Token-budget batch packing, FP8 on H100 (over 99% cosine similarity, 50%-plus throughput gain), and TensorRT-LLM for 2x over vLLM on embedding workloads. *(deployment)*

- **NVIDIA** [NVIDIA Dynamo: a low-latency distributed inference framework](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/): Disaggregated serving with a KV-aware smart router (radix-tree prefix scoring), phase-specific tensor parallelism, and a tiered KV cache offload manager. 30x throughput on DeepSeek-R1 671B on GB200 NVL72. *(deployment)*

- **Together AI** [ATLAS: runtime-learning speculative decoding](https://www.together.ai/blog/adaptive-learning-speculator-system-atlas): An adaptive speculator that learns from live traffic and blends with a static baseline; up to 4x baseline throughput on DeepSeek-V3.1 with no manual tuning. *(product design)*

- **Fireworks AI** [FireOptimizer: customizing latency and quality](https://fireworks.ai/blog/fireoptimizer): Workload-specialized draft models (generic at alpha=0.29 caused 1.5x slowdown; specialized at 0.76 gave 2x speedup); plus profile-driven quantization and caching. *(product design)*

- **Moonshot AI** [Mooncake: a KVCache-centric disaggregated architecture](https://arxiv.org/abs/2407.00079): Kimi's prefill/decode disaggregation with a pooled CPU/DRAM/SSD/object-store KV cache for shared-prefix-heavy traffic. *(deployment)*

- **Microsoft Research** [Splitwise: efficient generative LLM inference using phase splitting](https://arxiv.org/abs/2311.18677): Splits prefill (compute-heavy) and decode (bandwidth-heavy) onto separate physical machines for cost and throughput. *(deployment)*

- **Peking University / UCSD** [DistServe: disaggregating prefill and decoding for goodput-optimized LLM serving](https://arxiv.org/abs/2401.09670): Disaggregates prefill and decode with per-phase parallelism tuned to goodput (requests meeting SLO) rather than raw throughput. *(deployment)*

- **Microsoft Research** [Sarathi-Serve: taming the throughput-latency tradeoff](https://arxiv.org/abs/2403.02310): Chunked prefill with stall-free scheduling smooths TPOT under mixed workloads without disaggregating. *(deployment)*

- **Modal** [High-performance LLM inference](https://modal.com/docs/guide/high-performance-llm-inference): Engine choice (vLLM for throughput, SGLang for latency), memory snapshots for 10x cold-start reduction, FP8 and weight streaming. *(deployment)*

- **Databricks** [LLM inference performance engineering: best practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices): Prefill/decode breakdown, hardware selection, and batch-size guidance for steady, known workloads. *(eval bar)*

- **Google** [Fast inference from transformers via speculative decoding](https://arxiv.org/abs/2211.17192): The original paper formalizing draft-and-verify decoding with rejection sampling; 2-3x speedup with provably identical output distribution. *(product design)*

- **Baseten** [The Baseten inference stack](https://www.baseten.co/resources/guide/the-baseten-inference-stack/): Multi-cloud autoscaling, routing, custom kernels, and adaptive speculative execution for bursty enterprise traffic. *(deployment)*

- **Snowflake** [Arctic Inference with Shift Parallelism](https://www.snowflake.com/en/blog/engineering/arctic-inference-shift-parallelism/): A vLLM plugin with dynamic shift parallelism that adapts TP degree to live traffic patterns. *(deployment)*
