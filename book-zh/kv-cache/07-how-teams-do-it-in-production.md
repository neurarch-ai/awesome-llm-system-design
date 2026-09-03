# 7. 真实团队在生产环境里怎么做

所有生产级的 LLM 服务系统跑的都是同一个两阶段循环：prefill 把 KV cache 建起来，
然后 decode 一个 token 一个 token 地复用它。各家真正分道扬镳的地方在于：
从哪儿动刀砍 cache、怎么管显存、要不要复用 prefill 的成果，以及它们的首要约束是什么。
架构是共通的，杠杆在于拉了哪几根、按什么顺序拉。

## 真实设计的分歧在哪

| 系统 | KV 缩减策略 | 显存管理 | 前缀 / prompt 缓存 | 主要优化目标 | 什么时候它赢 | 要留神什么 |
|---|---|---|---|---|---|---|
| vLLM（UC Berkeley） | 架构上没有 | 操作系统风格的定长块分页 | 有，块级，配写时复制 | 靠塞进更多并发序列提吞吐 | 大量长度不一的并发序列；碎片是瓶颈 | 按节点存的 cache 不跨集群；规模上去要配缓存感知路由 |
| Character.AI | MQA 加跨层 KV 共享加原生 int8 | 滑动窗口的局部 / 全局混合 | 有，跨轮次的滚动哈希 LRU 树（命中率约 95%） | 显存和成本（自 2022 年起降了 33 倍） | 高流量对话，prompt 复用多，成本目标激进 | MQA 叠 int8 是把多重质量风险摞在一起；一定要在自己的评测上验 |
| DeepSeek-V2/V3（MLA） | 潜向量压缩（相比 MHA 约 93%） | 标准分页块 | 不是重点 | 长上下文的 KV 显存 | 模型是自己训的或自己能改，且 KV cache 是真正的约束 | RoPE 和潜向量这条路径不可交换；训练期需要做分头拆分的修正 |
| Google（GQA、Llama 3） | 减少 KV 头数（缩 4 到 8 倍） | 标准 | 不是重点 | 在接近 MHA 的质量下拿到吞吐和显存 | 绝大多数服务规模下任何模型的安全默认值 | 比例是固定的；更激进的场合该上 MLA |
| SGLang（RadixAttention） | 架构上没有 | 基数树配 LRU 淘汰 | 有，对分叉前缀自动跨请求复用 | agent 树和 few-shot 负载的吞吐与首 token 延迟 | 大量请求共享会分叉的前缀 | 前缀很少重叠时收益归零；流量太杂时 LRU 会颠簸 |
| NVIDIA TensorRT-LLM（NVFP4） | 4 bit KV 量化（相比 FP16 缩 4 倍） | 依赖感知的块淘汰 | 有，prefill 还在跑时就能提前复用 | 长上下文显存和首 token 延迟（最高快 5 倍） | Hopper GPU 上长 prompt、system prompt 占大头的负载 | "精度损失低于 1%"这个说法依赖具体 benchmark；在自己的任务上验 |
| Hugging Face（KV 量化） | 逐 token 的 int4/int2 量化 | 近期 token 保持全精度窗口 | 不是重点 | 改不了的固定模型的显存 | checkpoint 锁死、长生成带来显存压力 | 检索重的任务上会退化；保留全精度窗口并做评测 |
| Anthropic（prompt caching） | 架构上没有 | 按 API 用户存、带 TTL 的服务端 cache | 有，API 级别跨调用复用（Claude 3/3.5/3.7） | 大块共享上下文的成本和延迟（成本最多降 90%） | 一大块固定上下文在很多次 API 调用之间反复使用 | 上下文每次都变就没有收益；cache TTL 过期后省下的又回去了 |
| Databricks（prompt caching） | 架构上没有 | 每租户的易失 cache | 有，跨请求自动做前缀匹配 | 吞吐和 P50 延迟（命中率 30% 时分别是 2.5 倍、3 倍） | 多租户部署里反复出现的 system prompt | 多租户隔离是硬要求；精确前缀匹配只要有一个 token 对不上就不命中 |
| StreamingLLM（MIT/Meta） | 固定滑动窗口加 sink token | 滚动窗口；最前面几个 token 永远留着 | 不是重点 | 固定 KV 预算下的无限流式 | 永不结束的流，丢掉中间那段可以接受 | 被淘汰的 token 是真没了；整篇文档召回类任务会挂 |
| llm-d（KV 感知调度） | 架构上没有 | 分布式的缓存感知路由 | 有，跨节点的前缀路由 | 集群级别的前缀命中率 | 多节点部署，单节点前缀缓存被切碎 | 路由复杂度上去了；收益取决于流量的局部性 |

核心的分界线：一个系统要么**把每条 KV 记录压小**（GQA、MLA、量化），
要么**跨请求复用记录**（分页、基数树、前缀缓存），要么**干脆丢掉记录**（淘汰、滑动窗口）。
大多数生产系统会把其中两三种叠着用。

## 这些系统（一手资料）

- **vLLM（UC Berkeley）** [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180)：操作系统风格的 KV cache 分页干掉了碎片，吞吐提升 2 到 4 倍。*(deployment)*
- **Character.AI** [Optimizing AI Inference at Character.AI](https://blog.character.ai/optimizing-ai-inference-at-character-ai-2/)：MQA、局部 / 全局混合注意力、跨层 KV 共享加 int8，把成本砍到原来的三十三分之一。*(deployment)*
- **DeepSeek** [DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model](https://arxiv.org/abs/2405.04434)：Multi-head Latent Attention 用逐 token 的潜向量把 KV cache 压掉约 93%。*(product design)*
- **Google Research** [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245)：分组 query 注意力用更小的 cache 拿到接近 MHA 的质量，还给了 uptraining 的配方。*(product design)*
- **NVIDIA** [5x Faster Time to First Token with TensorRT-LLM KV Cache Early Reuse](https://developer.nvidia.com/blog/5x-faster-time-to-first-token-with-nvidia-tensorrt-llm-kv-cache-early-reuse/)：提前复用 KV、灵活的块大小、依赖感知的淘汰，把首 token 延迟压了下来。*(deployment)*
- **NVIDIA** [Optimizing Inference with NVFP4 KV Cache](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/)：4 bit KV 相比 FP8 把显存又减了一半，精度损失低于 1%。*(deployment)*
- **Databricks** [Inference-Friendly Models with MixAttention](https://www.databricks.com/blog/mixattention)：跨层 KV 共享配上滑动窗口注意力，在一个生产模型上把 cache 压了下来。*(product design)*
- **Databricks** [Accelerating LLM Inference with Prompt Caching](https://www.databricks.com/blog/accelerating-llm-inference-prompt-caching-open-source-models-databricks)：自动的前缀 KV 复用带来 2.5 倍吞吐和三分之一的 P50 延迟。*(deployment)*
- **llm-d** [KV-Cache Wins You Can See](https://llm-d.ai/blog/kvcache-wins-you-can-see)：单实例的前缀缓存在集群里就不灵了，缓存感知调度能补上。*(deployment)*
- **SGLang / LMSYS** [Fast and Expressive LLM Inference with RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/)：基数树 KV cache 让分叉型负载能自动跨请求复用前缀。*(deployment)*
- **Hugging Face** [Unlocking Longer Generation with KV Cache Quantization](https://huggingface.co/blog/kv-cache-quantization)：逐 token 的 int4 KV 量化在固定模型上省下约 2.5 倍显存。*(product design)*
- **Anthropic** [Prompt Caching with Claude](https://www.anthropic.com/news/prompt-caching)：跨 API 调用的服务端 cache 在大块共享上下文上把成本最多降 90%、延迟降 85%。*(product design)*
- **MIT/Meta** [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453)：attention sink 这个洞见让固定窗口的 LLM 能稳定流式生成到数百万 token。*(product design)*
- **Together AI** [Serving MiniMax-M3: 1M-token context without regrets](https://www.together.ai/blog/serving-minimax-m3-for-efficient-inference-unlocking-1m-token-context-and-multimodality-without-regrets)：分页稀疏注意力加 KV 块主序的 kernel，让 1M token 的服务变得可行。*(deployment)*
- **KIVI** [A Tuning-Free Asymmetric 2-bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750)：key 按通道、value 按 token 量化，配一段全精度的近期 token 窗口，实现 2 bit 的 KV 压缩。*(product design)*
- **UT Austin / Meta** [H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models](https://arxiv.org/abs/2306.14048)：KV 淘汰时保留近期 token 加重度命中 token，吞吐最高提升 29 倍。*(deployment)*

## 去看看这些架构（Model Zoo）

上面公式里的数字都是真实的模型维度，不是为了举例编出来的。
下面这些图都是经过校验、做过形状检查的架构图，可以直接打开来看：

- **GQA 基线（Llama 3 8B）：**[打开实时版本](https://www.neurarch.com/?import=https://raw.githubusercontent.com/neurarch-ai/awesome-llm-model-zoo/main/architectures/llama3-8b/model.json)。找到注意力块，把 query 头数和 KV 头数比一比，这个比值就是省下来的 cache。

  ![Llama-3 8B](https://raw.githubusercontent.com/neurarch-ai/awesome-llm-model-zoo/main/architectures/llama3-8b/assets/diagram.png)

- **MLA 加 MoE（DeepSeek-V3）：**[打开实时版本](https://www.neurarch.com/?import=https://raw.githubusercontent.com/neurarch-ai/awesome-llm-model-zoo/main/architectures/deepseek-v3/model.json)。沿着注意力块，把 RoPE 子维度和压缩潜向量子维度分别走一遍。

  ![DeepSeek-V3](https://raw.githubusercontent.com/neurarch-ai/awesome-llm-model-zoo/main/architectures/deepseek-v3/assets/diagram.png)

所有校验过的架构图都在 [Model Zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo)
（[画廊](https://neurarch-ai.github.io/awesome-llm-model-zoo)）。由 [Neurarch](https://www.neurarch.com) 打造。
