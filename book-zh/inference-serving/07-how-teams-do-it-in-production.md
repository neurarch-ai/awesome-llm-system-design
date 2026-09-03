# 7. 真实团队在生产环境里怎么做

所有生产级的 LLM serving 栈最后都收敛到同一副骨架：每个 token step 都重组 batch 的连续批处理、防碎片的分页 KV cache、把 cache 填起来的 prefill、每个 token 都要读一遍 cache 的受带宽限制的 decode 循环，再加一个由 SLO 驱动的自动扩缩容。真正拉开差距的，是各家把哪一个环节推到了极致，而这个选择直接由它们各自的负载决定。

## 真实设计在哪里分道扬镳

| 系统 | 关键优化 | 批处理 | 投机解码 | prefill/decode 分离 | 量化 |
|---|---|---|---|---|---|
| Anyscale（vLLM） | 用 PagedAttention 解决碎片 | 连续，迭代级 | 无 | 单一资源池 | BF16 基线 |
| Character.AI | KV 占用（MQA、跨层共享、int8） | 连续 | 无 | 单一资源池 | INT8 权重与 KV |
| LinkedIn | 针对模板化输出的 n-gram 投机解码 | 连续（vLLM） | 有，n-gram | 单一资源池 | 未说明 |
| Baseten（BEI） | token 预算打包、FP8、TensorRT-LLM | 按 token 预算的连续批处理 | 有 | 无 | FP8（H100） |
| NVIDIA Dynamo | prefill/decode 分离部署，配 KV 感知路由 | 连续 | 无 | 有（独立资源池，独立机器） | 分阶段各自的 TP |
| Together AI（ATLAS） | 在线自适应的投机解码 | 连续 | 有，自适应 | 无 | 未说明 |
| Fireworks（FireOptimizer） | 按负载定制的草稿模型 | 连续 | 有，按负载训练 | 无 | 按负载配置 |
| Moonshot（Mooncake） | 池化的分层 KV cache（CPU/DRAM/SSD/对象存储） | 连续 | 无 | 有 | 未说明 |
| Microsoft（Splitwise） | 按阶段拆到不同硬件上 | 连续 | 无 | 有，物理机分离 | 分阶段各自的方案 |
| Sarathi-Serve | 分块 prefill，实现无停顿调度 | 无停顿的分块批处理 | 无 | 单一资源池 | 未说明 |
| Modal | 引擎选型、内存快照、权重流式加载 | 连续 | 可选 | 无 | 按负载选 FP8 或 INT4 |
| Databricks | 硬件选型与 batch 大小的经验指引 | 连续 | 无 | 把 prefill/decode 当作两个阶段 | 随硬件而定 |

最主要的分水岭是分离部署：Dynamo、Splitwise、DistServe 和 Mooncake 把 prefill 和 decode 拆到不同的资源池或机器上；其余各家保持单一资源池，用分块 prefill 或投机解码来化解这两者之间的张力。第二条分界线是投机解码：LinkedIn、Together 和 Fireworks 在草稿模型上投入，Character.AI 和 Anyscale 则把力气花在压缩 KV 占用上。你只需要挑一个跟自己瓶颈对得上的主要杠杆。

## 这些系统（第一手技术文章）

- **Anyscale** [How continuous batching enables 23x throughput in LLM inference](https://www.anyscale.com/blog/continuous-batching-llm-inference)：在 A100 上跑 Meta OPT-13B，迭代级调度加 PagedAttention 相比静态批处理最多快 23 倍。文章把调度带来的收益（约 8 倍）和显存带来的收益分开算；跟优化过的静态批处理诚实对比，是 5 到 6 倍。*（部署）*

- **vLLM** [vLLM V1: a major upgrade to vLLM's core architecture](https://blog.vllm.ai/2025/01/27/v1-alpha-release.html)（2025 年 1 月）：调度器、KV cache 管理器和 worker 的彻底重写，把上面那些优化默认打开。它的常驻 batch 循环缓存输入张量、每步只做增量更新，砍掉 CPU 开销，吞吐比 V0 高约 1.7 倍。2025 年的教训是：规模上去之后，瓶颈从 GPU kernel 转移到了 CPU 调度开销上，所以赢在框架重写，而不是某个新的 attention 花招。*（部署）*

- **Character.AI** [Optimizing AI Inference at Character.AI](https://blog.character.ai/optimizing-ai-inference-at-character-ai/)：MQA、跨层 KV 共享，加上 INT8 的权重与 KV 量化，在约 20,000 QPS 的量级上把 serving 成本压到商用 API 的十三分之一多一点。激进的 KV 压缩必须在架构里训出来，不是 serving 时打个开关就有。*（部署）*

- **LinkedIn** [Accelerating LLM inference with speculative decoding](https://www.linkedin.com/blog/engineering/ai/accelerating-llm-inference-with-speculative-decoding-lessons-from-linkedins-hiring-assistant)：Hiring Assistant 上的 n-gram 投机解码带来接近 4 倍吞吐和低 66% 的 P90 延迟；接受率高，是因为输出本身就在复述职位描述里的文字。*（评测标准）*

- **Baseten** [How we built BEI: high-throughput embedding, reranker, classifier inference](https://www.baseten.co/blog/how-we-built-bei-high-throughput-embedding-inference/)：按 token 预算打包 batch、H100 上用 FP8（余弦相似度 99% 以上，吞吐提升 50% 以上），并用 TensorRT-LLM 在 embedding 负载上做到 vLLM 的 2 倍。*（部署）*

- **NVIDIA** [NVIDIA Dynamo: a low-latency distributed inference framework](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)：分离式 serving，配上 KV 感知的智能路由（基数树前缀打分）、按阶段各自设定的张量并行，以及分层的 KV cache 卸载管理器。在 GB200 NVL72 上跑 DeepSeek-R1 671B，吞吐提升 30 倍。*（部署）*

- **Together AI** [ATLAS: runtime-learning speculative decoding](https://www.together.ai/blog/adaptive-learning-speculator-system-atlas)：一个会从线上流量里学习的自适应 speculator，并与静态基线做融合；在 DeepSeek-V3.1 上无需手工调参就把吞吐做到基线的 4 倍。*（产品设计）*

- **Fireworks AI** [FireOptimizer: customizing latency and quality](https://fireworks.ai/blog/fireoptimizer)：按负载定制的草稿模型（通用草稿在 alpha=0.29 时反而慢 1.5 倍；定制后 alpha 到 0.76，快 2 倍），另有基于负载画像的量化与缓存。*（产品设计）*

- **Moonshot AI** [Mooncake: a KVCache-centric disaggregated architecture](https://arxiv.org/abs/2407.00079)：Kimi 的 prefill/decode 分离方案，配一个池化的 CPU/DRAM/SSD/对象存储 KV cache，面向共享前缀占比很高的流量。*（部署）*

- **Microsoft Research** [Splitwise: efficient generative LLM inference using phase splitting](https://arxiv.org/abs/2311.18677)：把 prefill（吃算力）和 decode（吃带宽）拆到不同的物理机上，同时优化成本和吞吐。*（部署）*

- **Peking University / UCSD** [DistServe: disaggregating prefill and decoding for goodput-optimized LLM serving](https://arxiv.org/abs/2401.09670)：拆开 prefill 和 decode，并按 goodput（满足 SLO 的请求数）而不是裸吞吐来分别调各阶段的并行度。*（部署）*

- **Microsoft Research** [Sarathi-Serve: taming the throughput-latency tradeoff](https://arxiv.org/abs/2403.02310)：分块 prefill 加无停顿调度，在混合负载下把 TPOT 抹平，且不需要做分离部署。*（部署）*

- **Modal** [High-performance LLM inference](https://modal.com/docs/guide/high-performance-llm-inference)：引擎选型（要吞吐用 vLLM，要延迟用 SGLang）、把冷启动缩短到十分之一的内存快照、FP8 与权重流式加载。*（部署）*

- **Databricks** [LLM inference performance engineering: best practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)：拆解 prefill 与 decode，讲硬件选型，以及负载稳定且已知时该怎么定 batch 大小。*（评测标准）*

- **Google** [Fast inference from transformers via speculative decoding](https://arxiv.org/abs/2211.17192)：把起草加验证的解码方式连同拒绝采样一起形式化的原始论文；2 到 3 倍加速，且可证明输出分布完全一致。*（产品设计）*

- **Baseten** [The Baseten inference stack](https://www.baseten.co/resources/guide/the-baseten-inference-stack/)：面向突发性企业流量的多云自动扩缩容、路由、自定义 kernel 和自适应投机执行。*（部署）*

- **Snowflake** [Arctic Inference with Shift Parallelism](https://www.snowflake.com/en/blog/engineering/arctic-inference-shift-parallelism/)：一个带动态 shift parallelism 的 vLLM 插件，能根据线上流量形态调整 TP 的并行度。*（部署）*
