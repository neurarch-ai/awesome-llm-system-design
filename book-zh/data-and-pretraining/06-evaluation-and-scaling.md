# 6. 评估与规模扩展

评估一个基础模型和评估一个 chat 模型，判断标准不一样，用错指标是个经典错误。这里有两层：训练信号（loss），以及下游能力（benchmark）。两层都得当心。

## 训练信号：perplexity 与 bits-per-byte

主要的训练指标是留出集上的 loss，报出来的形式是 **perplexity**：

$$\text{PPL} = \exp\!\left(\mathcal{L}\right)$$

其中 $\mathcal{L}$ 是留出集上按 token 平均的负对数似然。越低越好。

从机制上讲，$\mathcal{L}$ 就是模型的下一个 token 分布和真实下一个 token 之间的交叉熵，在所有位置上取平均。唯一容易绊倒人的细节是**错开一位**：位置 $t$ 的 logits 要拿位置 $t{+}1$ 的 token 来打分。

```python
import torch.nn.functional as F
# logits: (batch, seq, vocab); targets: (batch, seq) of token ids
def loss_and_perplexity(logits, targets):
    logits = logits[:, :-1, :]          # drop the last position (no next token)
    targets = targets[:, 1:]            # the next token is the label (shift by one)
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    return ce, ce.exp()                 # perplexity = exp(mean cross-entropy)
```

perplexity 是这个平均交叉熵的指数，所以 perplexity 等于 10 的意思是：模型平均而言的不确定程度，相当于在 10 个 token 里均匀乱猜。

**perplexity 只在共用同一个分词器的模型之间可比。** 词表更大的模型每句话吐出的 token 更少（每个 token 覆盖更多文本），这会机械地把 perplexity 拉低，而模型在真实任务上未必更好。这就是分词器 fertility 问题的另一面。

**bits-per-byte（BPB）** 用文本的字节数做归一化，把这个假象消掉：

$$\text{BPB} = \frac{\mathcal{L}}{\ln 2} \cdot \frac{n_{\text{tokens}}}{n_{\text{bytes}}}$$

BPB 是与分词器无关的指标。只要在比较词表不同的模型，就该用它。认真的预训练论文两个都会报。

## Benchmark 评估

在 loss 评估之后（或者同时），把模型放到能力 benchmark 上跑：MMLU（知识广度）、ARC-Challenge（推理）、HellaSwag（常识）、HumanEval（代码生成）、GSM8K（数学推理）。

**要按时间切分，不要随机切分。** 留出未来某个时间窗口的文档，看今天的模型能不能应付。随机切分会把未来泄漏进来，让模型的成绩虚高。

**报任何 benchmark 数字之前先做去污染。** 任何没有附带去污染说明的分数都该被怀疑。成熟的做法是把你发现并移除掉的污染率一并报出来。一个只因为评测集漏进训练数据才好看的分数，是负债，不是资产。

## 评估告诉不了你的东西

基础模型评估衡量的是下一个 token 的预测质量。它并不直接衡量指令跟随、安全性、对齐程度，或者真实世界里的有用性。那些需要后训练阶段的评估（RLHF、DPO、红队测试）。预训练评估定的是能力的下限，后训练决定这份能力以什么方式表现出来。

## 瓶颈

| 瓶颈 | 最先露头的迹象 | 修法 | 取舍 |
|---|---|---|---|
| 抽取不干净 | 重复数量暴涨；启发式过滤器误伤 | 从 WARC 重新抽取；去页面模板噪声；URL 黑名单 | 流水线比直接用 WET 纯文本更重 |
| 近似重复 | 出现记忆问题；评测泄漏；浪费 token | 在单个 dump 内和跨 dump 做 MinHash / LSH 模糊去重 | 去重过头会剥掉合理的常见文本；要做消融来定激进程度 |
| 低质量网页文本 | token 加了很多，benchmark 却不再涨 | 启发式过滤加上学习式质量分类器 | 分类器会把参考语料的偏见固化进来；要在下游评测上验证 |
| 评测污染 | 数据刷新之后 benchmark 突然跳高 | n-gram 和 embedding 去污染；把发现的污染率报出来 | 会误删一些真实收益；诚信优先于分数 |
| 分词器 fertility | 某个语言每个词要花 $3 \times$ 的 token | 在真实混合数据上拟合词表；按各语言的 fertility 定词表大小 | embedding 和 softmax 的参数量更大 |
| 显存墙 | 模型还没塞进单张 GPU 就 OOM | 张量并行、ZeRO / FSDP 切分、激活重计算 | 多出通信；MFU 下降 |
| MFU 偏低 | GPU 在等；实际吞吐远低于标称 | TP 限制在节点内、PP 用很多 micro-batch、把通信重叠起来 | 并行方案变复杂，调优和维护成本高 |
| 硬件故障 | 任务每隔几小时就挂 | 高频的分片 checkpoint；弹性重启 | checkpoint 的 I/O 成本和存储开销 |
| loss 尖峰 | 训练到第 N 步中途 loss 发散 | 回滚到上一个好的 checkpoint；跳过或重新打乱 batch；调低学习率 | 丢掉一些步数；需要自动或人工介入 |
| 算力分配失当 | 预算烧完时还没到算力最优的 token 数 | 开跑之前先用 Chinchilla 算尺寸；服务负载重就过度训练一个更小的模型 | 放弃训练最优，换推理省钱 |

**再展开一点。** 有两行的出处和机制特别清楚。显存墙那一行的修法，是把 ZeRO（Microsoft）的优化器、梯度和参数切分，配上它在 FSDP（Meta）里的原生实现；代价是 ZeRO-3 会在每层前向和反向之前把该层参数 all-gather 一次，这些多出来的通信最后表现为 MFU 下降。算力分配失当那一行的修法是 Chinchilla（DeepMind, 2022）的尺寸规则，大约每参数 20 个 token，但它最小化的只是训练算力：当模型将来要服务几十亿 token 时，有意把一个更小的模型过度训练、越过 Chinchilla 点，是拿一次性的训练算力换永久更便宜的每 token 推理开销。
