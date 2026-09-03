# 7. 真实团队在生产环境里怎么做

每一次重要的长上下文适配，最后都收敛到同一副骨架：拿一个预训练好的基础模型，
在一个调低并重新升温的学习率下重新进入 next-token 目标，重缩放 RoPE 位置频率，
在上采样过的长文档上分阶段继续训练，进入后训练之前用一道回归检查加一个 RULER
式的评估把关。团队之间真正不同的只有两个决定：**频率重缩放做得有多非均匀**，
以及**怎么证明扩展出来的长度是真的**。骨架大家都一样，杠杆在重缩放配方和评估
纪律上。

## 真正的设计在哪里分岔

| 系统 | 轴线 | 扩展机制 | 达到的长度 | 关键杠杆 | 注意 |
|---|---|---|---|---|---|
| Meta Llama 3 | 长度 | RoPE 重缩放 + 分阶段继续训练 | 分 6 个阶段从 8K 到 128K | 在预训练后期做分阶段扩展 | 405B 的预训练是实验室级别的规模 |
| Meta Code Llama | 领域 + 长度 | NTK-ABF（RoPE base 从 10000 到 1000000） | 训练在 16K，可外推到 100K | 从一个通用基础模型继续预训练出一个代码领域 | 代码会收窄通用能力；需要回放来保住通用语言 |
| 01.AI Yi | 长度 | 在长数据上继续预训练，RoPE 重缩放 | 最长 200K | 数据质量优先于架构上的新意 | 长数据的整理才是卡脖子的约束 |
| Nous YaRN | 长度 | 非均匀频率缩放 + softmax 温度 | 64K 到 128K 以上 | 大约 0.1% 的预训练 token 量 | 斜坡频段和温度系数必须调 |
| Microsoft LongRoPE | 长度 | 逐维度的进化搜索 + 渐进扩展 | 200 万 token 以上 | 搜出来的非均匀重缩放 + 短上下文恢复 | 搜索成本；必须有一步短上下文恢复 |
| Alibaba Qwen2.5 | 领域 + 长度 | 渐进长度 + YaRN 式缩放 + Dual Chunk Attention | 开源模型 128K（turbo 版 1M） | 分阶段的非均匀缩放加 attention 分块 | 有效长度低于配置长度；用 RULER 去验 |
| Mila 持续预训练 | 领域 | 重新升温 + 重新衰减 + 回放（是方法论，不是产品） | 追平从零重训 | 一小部分回放就能大幅减少遗忘 | 适中的重新升温峰值是关键超参 |
| Ai2 DAPT（"Don't Stop Pretraining"） | 领域 | 领域自适应和任务自适应的继续预训练 | 特定任务的 NLP | 两阶段领域内预训练确实划算的经典证据 | 有污染风险；要对着自己的评估集做去污染 |

分界线在于：**重缩放配方和数据质量买来的是有效上下文的天花板，评估纪律才告诉
你实际有没有摸到它。** 几乎每一次发布里，配置长度和有效长度都是分开的；要测的
就是这个差距。

## 这些系统（第一手材料）

- **Meta** [The Llama 3 Herd of Models](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/)：在预训练后期分六个递进阶段把上下文窗口从 8K 扩到 128K，用的是 RoPE 重缩放加分组查询注意力。分阶段扩展更便宜（早期序列短），也比一次跑到超长更稳定。*（长度）*

- **Meta** [Code Llama: Open Foundation Models for Code](https://arxiv.org/abs/2308.12950)：在代码语料上对 Llama 2 做继续预训练，配合 Adjusted Base Frequency（RoPE base 从 10000 提到 1000000），在 16K 序列上训练，可外推到最长 100K token 的输入。这是"提高 RoPE base 是一次真正的非均匀重缩放，而不是拍脑袋"的经典生产案例。*（领域 + 长度）*

- **01.AI** [Yi: Open Foundation Models by 01.AI](https://arxiv.org/abs/2403.04652)：6B 和 34B 的双语基础模型，通过在长数据上继续预训练扩展到 200K。报告把功劳记在数据质量而不是架构上，这对频率缩放那一脉文献是个有用的平衡：一旦 RoPE 重缩放做得够了，长数据的整理就成了卡脖子的约束。*（长度）*

- **Nous Research** [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071)：非均匀 RoPE 频率缩放加上一个 softmax attention 温度修正，只用原预训练 token 量大约 0.1% 就能扩展上下文，比均匀插值便宜得多。它成了激进扩展的默认配方。*（长度）*

- **Microsoft** [LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753)：对逐维度的 RoPE 重缩放因子做进化搜索，加上渐进扩展和一步短上下文恢复，达到 200 万 token 以上。它说明最优的重缩放是非均匀的，而且和输入长度有关；到那么远的长度上，固定的闭式解找不到最好的因子。*（长度）*

- **Meta** [Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595)：线性位置插值把位置索引均匀压缩回训练过的范围，用大约一千步微调就把 LLaMA 扩到了 32K。朴素外推会灾难性失败，这篇论文给出了能跑通的最小替代方案。*（长度）*

- **Ai2** [Don't Stop Pretraining: Adapt Language Models to Domains and Tasks](https://arxiv.org/abs/2004.10964)：领域自适应预训练加任务自适应预训练，在四个领域上都提升了下游任务。这是"第二段领域内预训练确实划算"的经典证据。*（领域）*

- **Mila** [Simple and Scalable Strategies to Continually Pre-train Large Language Models](https://arxiv.org/abs/2403.08763)：学习率重新升温、重新衰减，再加上一小部分回放，就能让继续预训练用零头的算力追平完整的从零重训。它把遗忘与学习之间的权衡量化成了重新升温峰值和回放比例的函数。*（领域）*

- **Alibaba** [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115)：在 18 万亿 token 上预训练，长上下文适配用的是渐进增加长度、YaRN 式的非均匀缩放和 Dual Chunk Attention，开源模型达到 128K（turbo 版最高 1M）。它坦率承认配置长度和有效长度是两回事。*（领域 + 长度）*

- **NVIDIA** [RULER: What's the Real Context Size of Your Long-Context Language Models?](https://arxiv.org/abs/2404.06654)：一个包含多跳变量追踪、聚合和多针任务的合成 benchmark，显示大多数声称 32K 以上的模型，远没到宣传长度就急剧退化。它是衡量有效上下文的标准参考。*（评估）*

- **Meta** [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation (ALiBi)](https://arxiv.org/abs/2108.12409)：在 attention 分数上加一个线性距离惩罚，让在短上下文上训练的模型在测试时不用重训就能外推到更长的输入。它是 RoPE 重缩放之外的另一条路，但必须在预训练时就采用。*（长度，架构层面的替代方案）*
