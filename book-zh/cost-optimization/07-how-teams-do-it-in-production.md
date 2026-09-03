# 7. 真实团队在生产环境里怎么做

这里的每套系统骨架都一样：网关挡在供应商前面，缓存把重复请求短路掉，路由器或级联挑一个档位，只有活下来的难查询才会走到前沿模型。团队之间真正的差别在于，他们先优化了流水线的哪一段，以及针对自己那个成本大头，哪个杠杆买到的收益最多。

## 真实设计的分岔点在哪里

| 系统 | 主要杠杆 | 模型档位 | 他们要打的成本驱动因素 | 为什么长成这样 |
|---|---|---|---|---|
| Stanford FrugalGPT | 带训练打分器的级联 | 三段链路（从便宜到前沿） | 质量参差的查询上的单请求成本 | 一个按答案可靠性训练出来的打分器，比盲路由器更精准 |
| LMSYS RouteLLM | 基于偏好数据的盲路由器 | 弱模型（Mixtral 级）与强模型（GPT-4 级） | Chatbot Arena 那类流量上的单请求成本 | 55k 条偏好对教会了"难还是简单"，而且能跨模型对迁移 |
| Anyscale | 微调过的分类器路由器 | Mixtral-8x7B 与 GPT-4 | 单请求成本；开源闭源混编的机队 | 五档质量分给出的梯度比二分类的难 / 简单更强 |
| IBM Research | 多模型预测式路由器 | 11 个模型的模型库 | 各类专才任务上的单请求成本 | 有些 13B 专才打得过 70B 通才，路由到对的专才就赢了 |
| Microsoft LLMLingua | prompt 压缩 | 不换模型 | 又长又啰嗦的 RAG 上下文带来的输入 token 成本 | 基于困惑度的 token 打分能压 20 倍，质量几乎不掉 |
| Databricks 批量推理 | batch API / 离线路由 | 任何自建模型 | 批量任务在按在线价付钱 | ai_query 的 SQL 接口加上自动扩缩，让批量工作跑在最大 batch 上 |
| Baseten FP8 | 量化 | H100 上的 Mistral 7B FP8 | 自建的单 token 成本 | FP8 把显存占用砍半，token/s 提升 33%，单 token 成本比 FP16 低 24% |
| Cloudflare AI Gateway | 精确匹配缓存加网关 | 任意供应商 | 重复的相同调用带来的请求量 | Hash(body) 缓存，目前只有精确匹配，语义缓存在计划中；网关负责 fallback |
| Uber GenAI Gateway | 带预算和 fallback 的网关 | 多厂商机队 | 规模化之后的开支可见性与可靠性 | 统一代理取代各服务各自调供应商；预算按团队切 |

分界线在这里：**路由和级联换的是更便宜的模型**（在看到答案之前或之后），**缓存和压缩让这次调用本身更便宜**。Right-sizing 和量化压的是模型档位的成本地板。网关让上面这些都能被强制执行。挑那个和你的成本大头对得上的杠杆。

## 这些系统（一手资料）

- **Stanford** [FrugalGPT: Using LLMs While Reducing Cost and Improving Performance](https://arxiv.org/abs/2305.05176)：一条学出来的级联，只有在便宜答案被判为不可靠时才升级到更贵的模型；成本最多低 98% 而质量追平 GPT-4，或者花同样的钱把准确率抬高 4%。
- **LMSYS** [RouteLLM: an open framework for cost-effective LLM routing](https://www.lmsys.org/blog/2024-07-01-routellm/)：用 55k 条 Chatbot Arena 对比数据训出来的偏好数据路由器；只把 14% 的流量送给 GPT-4，就在 MT Bench 上保住了 GPT-4 的 95% 质量；四种路由器实现，全都能迁移到没见过的模型对上。这四种分别是：在 Arena 对战上做相似度加权排序、在偏好矩阵上做矩阵分解、微调一个 BERT 分类器、以及一个 causal-LLM 分类器，每一种都预测强模型的胜率，于是一个阈值就定下了成本-质量的工作点。
- **Anyscale** [Building an LLM Router for High-Quality and Cost-Effective Responses](https://www.anyscale.com/blog/building-an-llm-router-for-high-quality-and-cost-effective-responses)：把 Llama-3 8B 微调成五档复杂度分类器；在 MT Bench 上砍掉 70% 成本；训练用的 109k 条查询由 GPT-4 当裁判打标。
- **IBM Research** [LLM routing for quality, low-cost responses](https://research.ibm.com/blog/LLM-routers)：跨 11 个模型的模型库做预测式路由；用 benchmark 训练来预测准确率与成本之比；11 个模型的组合打赢了其中任何一个单独的模型；成本最多降 85%。
- **Microsoft Research** [LLMLingua: prompt compression for LLM efficiency](https://www.microsoft.com/en-us/research/blog/llmlingua-innovating-llm-efficiency-with-prompt-compression/)：用小语言模型的困惑度打分，先粗后细地压缩；最高压 20 倍，质量掉约 1.5 个点；已集成进 LlamaIndex 供 RAG 使用。
- **Databricks** [Simple, Fast, Scalable Batch LLM Inference](https://www.databricks.com/blog/introducing-simple-fast-and-scalable-batch-llm-inference-mosaic-ai-model-serving)：用 ai_query 的 SQL 接口在数仓表上做受治理的批量推理；自动扩缩，数据不搬家，重试容错；这是数仓规模上的 batch 与在线成本杠杆。
- **Baseten** [33% faster LLM inference with FP8 quantization](https://www.baseten.co/blog/33-faster-llm-inference-with-fp8-quantization/)：通过 TensorRT-LLM 在 H100 上跑 FP8 的 Mistral 7B；相比 FP16，token/s 高 33%，每百万 token 成本低 24%；困惑度持平，人工评测只看出细微的文风差异。
- **Cloudflare** [Caching in AI Gateway](https://developers.cloudflare.com/ai-gateway/features/caching/)：精确匹配缓存（对供应商、模型和完整 body 做哈希）；cf-aig-cache-ttl 和 cf-aig-cache-key 两个 header 用来按请求定制 TTL 和 key；文本和图像响应都覆盖；语义缓存在计划中。
- **Uber** [Uber's GenAI Gateway](https://www.uber.com/blog/genai-gateway/)：统一的多厂商代理，带团队级预算管理、跨供应商 fallback 和逐次调用的日志；没有这种集中式成本管控，优化就只能靠猜。

完整的对比表、全部数学推导和象限图，见 [topics/11-cost-optimization-and-model-routing.md](../../topics/11-cost-optimization-and-model-routing.md) 里那份高密度参考资料。
