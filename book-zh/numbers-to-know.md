# 该记住的数字

面试时应该能脱口而出的那些量，每一个都附上推导它的公式，这样可以现场重新推，而不
用死记。这里的每一条在旁边链接的章节里都有完整展开；这一页只是压缩版。

数字都是量级示意，不是 benchmark 结果。真正要养成的习惯是：先说公式，再代入数字，
最后说清什么因素会改变答案。

## 模型大小与显存

| 量 | 公式或数值 | 章节 |
|---|---|---|
| 权重占用显存 | $N \cdot b_w$ 字节；fp16 每参数 2 字节，int8 是 1 字节，int4 分组量化约 0.52（4 bit 加上每 128 个参数一个 fp16 scale） | [模型压缩](model-compression/) |
| 70B 模型，fp16 | 约 141 GB，一张 80GB 的加速卡放不下 | [模型压缩](model-compression/10-putting-it-together.md) |
| 70B 模型，int4 分组量化 | 约 36 GB，放得下 | [模型压缩](model-compression/10-putting-it-together.md) |
| 每 token 的 KV cache | $2 \cdot L \cdot h_{kv} \cdot d_h \cdot b_{kv}$ 字节 | [KV Cache](kv-cache/) |
| KV cache，80 层，8 个 KV head，head 维度 128，fp16 | 每 token 320 KB，所以 32K 上下文每条序列约 10.5 GB | [KV Cache](kv-cache/) |
| 全参数微调的训练显存 | 大约 $2N$（bf16 权重）加上约 $12N$（Adam 状态）字节，还没算激活 | [后训练](post-training/) |
| LoRA 或 QLoRA 的可训练参数占比 | 远低于 1%；QLoRA 把冻结的底座模型压到 4 bit | [后训练](post-training/) |

## 吞吐与延迟

| 量 | 公式或数值 | 章节 |
|---|---|---|
| 每 token 的 decode 时间 | （权重字节数 + 读取的 KV 字节数）/ 显存带宽：decode 受显存带宽限制 | [推理服务](inference-serving/) |
| Prefill 时间 | 跟着 FLOPs 走，每个 prompt token 约 $2N$：prefill 受算力限制 | [推理服务](inference-serving/) |
| 权重字节数减半的效果 | batch 为 1 时 decode 时间几乎减半；batch 32 到 64 时大约只有 1.1 到 1.3 倍 | [模型压缩](model-compression/02-frame-the-compression.md) |
| 排队延迟 | 随 $\frac{\rho}{1-\rho}\cdot\frac{E[S]\,(1+C^2)}{2}$ 增长：看的是二阶矩，不只是均值 | [推理模型服务](reasoning-serving/03-budgets-and-latency.md) |
| 推理路径的方差 | 思考轨迹是长尾的；$C^2$ 约 0.9，而短路径约 0.2 | [推理模型服务](reasoning-serving/10-putting-it-together.md) |
| 投机解码 | 攻的是受带宽限制的 decode，所以对思考量大的负载帮助最大 | [推理服务](inference-serving/04-speculative-decoding.md) |

## 评估统计

| 量 | 公式或数值 | 章节 |
|---|---|---|
| Benchmark 分数的置信区间 | $\pm 1.96\sqrt{\hat p(1-\hat p)/n}$ | [Benchmark](benchmark-eval/06-statistics-and-leaderboards.md) |
| 30 题的 benchmark，$\hat p = 0.5$ | 约 $\pm 18$ 分；一道题就是 3.3 分 | [Benchmark](benchmark-eval/06-statistics-and-leaderboards.md) |
| 198 题的 benchmark（GPQA Diamond） | 约 $\pm 7$ 分 | [Benchmark](benchmark-eval/06-statistics-and-leaderboards.md) |
| 500 题的 benchmark（SWE-bench Verified） | 约 $\pm 4.4$ 分 | [Benchmark](benchmark-eval/06-statistics-and-leaderboards.md) |
| 配对差值的标准误 | 在不一致的题目上算 $\sqrt{b+c}/n$；McNemar 检验 $z = (b-c)/\sqrt{b+c}$ | [Benchmark](benchmark-eval/06-statistics-and-leaderboards.md) |
| 要判定 2 分差距需要的题数 | $n \approx d \cdot 7.85/\delta^2$，不一致率 10% 时大约 2,000 题 | [Benchmark](benchmark-eval/06-statistics-and-leaderboards.md) |
| A/B 胜率的显著性 | 95% 置信区间必须不包含 0.5；真实胜率 54% 时大约需要 2,400 次比较 | [评估](evaluation/05-online-eval.md) |
| 裁判可信的门槛 | Cohen's kappa 到 0.6 左右，才能拿裁判当门禁 | [评估](evaluation/04-llm-as-judge.md) |
| PPI 校正 | $\hat\theta = \text{mean}(\text{judge}) + \text{mean}(\text{human} - \text{judge})$，后一项在有人工标注的子集上算 | [Benchmark](benchmark-eval/05-scoring-and-autoraters.md) |

## 采样与 agent

| 量 | 公式或数值 | 章节 |
|---|---|---|
| pass@k（覆盖率） | $1-(1-p)^k$；无偏估计量 $1 - \binom{n-c}{k}/\binom{n}{k}$ | [评估](evaluation/03-offline-eval.md) |
| pass^k（可靠性） | $p^k$；一个 90% 的 agent 在 $k=8$ 时只有约 43% | [Benchmark](benchmark-eval/05-scoring-and-autoraters.md) |
| 级联的盈亏平衡点 | 当 $a \gt (c_{\text{short}} + c_{\text{verify}})/c_{\text{long}}$ 时级联划算，通常约 15% | [推理模型服务](reasoning-serving/04-allocation-and-routing.md) |
| 采样最终交付的质量 | 覆盖率乘以选择器准确率：0.98 的覆盖率配 0.6 的选择器，交付出来约 0.6 | [推理模型服务](reasoning-serving/05-verification.md) |

## 数据与训练

| 量 | 公式或数值 | 章节 |
|---|---|---|
| 算力最优的 token 数 | Chinchilla scaling：大约每参数 20 个 token | [数据与预训练](data-and-pretraining/) |
| 训练 FLOPs | $N$ 个参数、$D$ 个 token 时约 $6ND$ | [LLM 的生命周期](llm-lifecycle/) |
| 对抗遗忘的回放比例 | 继续预训练的配比里放 5% 到 10% 的通用数据 | [中期训练](mid-training/03-the-mid-training-phase.md) |
| 继续预训练的语料下限 | 数十亿 token 起步；不到这个量，改用 SFT 或 RAG | [中期训练](mid-training/) |
| 重新预热的峰值学习率 | 原预训练峰值的一个分数，然后重新衰减 | [中期训练](mid-training/03-the-mid-training-phase.md) |

## 检索

| 量 | 公式或数值 | 章节 |
|---|---|---|
| Embedding 索引大小 | 向量数乘以维度乘以每个分量的字节数；量化是主要杠杆 | [语义搜索](semantic-search/) |
| 召回与延迟 | ANN 参数在两者之间连续取舍；要报固定延迟下的召回率，而不是单独一个召回率 | [语义搜索](semantic-search/04-vector-index.md) |
| 重排预算 | 用 cross-encoder 只过 top 50 到 100 个候选，不是过整个语料库 | [RAG 服务](rag-serving/04-retrieval-and-reranking.md) |
| 分块大小 | 几百个 token 带重叠；真正的约束是重排器和答案需要什么 | [RAG 服务](rag-serving/03-indexing-and-chunking.md) |

## 值得背下来的五句话

1. **Decode 受显存带宽限制，prefill 受算力限制。** 你在优化的是哪一个，决定了某个
   杠杆有没有可能起作用。
2. **KV cache 是随上下文和 batch 增长的那一项；** 权重是固定的。
3. **Benchmark 分数是一个估计值。** 200 道题上，跑一次得到的 3 分差距是分辨不出来的，
   配对比较才是精度的来源。
4. **排队延迟随服务时间的方差增长，** 所以思考型模型先毁掉的是尾延迟，然后才是均值。
5. **看每解决一个任务的成本，不是每次请求的成本。** 每个便宜的策略之所以便宜，都是
   因为它失败得更多，而这只有在分母里才看得见。
