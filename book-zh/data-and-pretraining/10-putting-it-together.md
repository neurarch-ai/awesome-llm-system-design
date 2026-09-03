# 10. 把它们拼起来：完整的方案

第 1 到 6 节把每个阶段的选项和取舍讲了一遍，第 7 节展示了真实团队在哪里分道扬镳。它们都没给出的，是一个每个决策都拍板了的完整系统。这一节收官做三件事：给出一套有主张的默认技术栈，让选择困难症不至于卡住第一次搭建；把本章那个场景从头到尾走一遍，每个选择都定下来并算清成本；再展示同样这些决策在约束变了之后怎么翻转。最后用一份最小可运行的去重代码收尾，一个文件，不用装任何东西。

## 默认技术栈：从这里开始，有理由再偏离

这条流水线的每个阶段都有三到六个说得过去的选项，一个第一次上手的人可能花掉一个月比较过滤配方，却还没产出一个干净的 token。别这么干。下面这套栈是第一次认真搭建时的合理默认值；每一行都写明什么时候该偏离，以及哪一节解释了为什么。工具年年在变，但每个阶段的接口（抽取、路由、过滤、去重、去污染、混合、分词、定尺寸、训练）不变，所以按接口逐阶段选型，把任何具体的库都当成可替换的。

| 阶段 | 默认做法 | 什么时候偏离 | 为什么（对应小节） |
|---|---|---|---|
| 抽取 | 用能去页面模板噪声的抽取器从 WARC 重新抽取；不要信 WET | 语料本来就是干净文本（书籍、论文）：直接进语言识别 | [2](02-the-data-pipeline.md) |
| 语言识别 | fastText 一类的分类器，按语言分流，置信度不够的文档丢掉 | 来源已知的单语语料：抽检一下就够了 | [2](02-the-data-pipeline.md) |
| 质量过滤 | 一小撮做过消融的 Gopher/C4 启发式规则，加一个 FineWeb-Edu 式的学习式分类器 | 没有足够丰富的参考语料来训分类器：只用启发式规则 | [3](03-data-quality.md) |
| 去重 | 先精确哈希，再在 dump 内和跨 dump 做 MinHash + LSH | 语料小到能做全量两两比较：省掉 LSH 那套机制 | [3](03-data-quality.md) |
| 去污染 | 对每一个评测集做 n-gram 重叠（13-gram），并报出发现的比例 | 永远不要跳过。在第一个训练 token 之前就跑 | [3](03-data-quality.md) |
| 混合 | 按领域配权重，代码、数学、论文上采样；后期退火到质量最高的数据 | 单领域语料：没什么可加权的 | [2](02-the-data-pipeline.md) |
| 分词器 | 字节级 BPE，32K 到 64K 词表，在最终混合数据上一次性拟合 | 多语言或没有空格的文字：SentencePiece，128K 以上，按语言看 fertility | [4](04-pretraining-choices.md) |
| 定尺寸 | 碰硬件之前先在纸上用 C ~ 6ND 把预算花掉 | 服务负载重：把更小的模型过度训练到远超每参数 20 个 token | [4](04-pretraining-choices.md) |
| 架构 | dense 的 pre-norm decoder，GQA + RoPE + RMSNorm + SwiGLU | 需要的容量超出每 token FLOP 预算：上带负载均衡的 MoE | [4](04-pretraining-choices.md) |
| 并行 | FSDP/ZeRO 切分的数据并行，bf16，高频分片 checkpoint | 单层撑爆一张 GPU：加节点内 TP；整个模型栈放不下：加 PP | [5](05-systems.md) |

去污染那一行是新手最爱跳过、事后最后悔的一行：不做它，你报出的每一个 benchmark 数字都可能是虚构，而且往往等到别人发现你才知道。它是整张表里最便宜的一行，也是唯一一行没有"偏离条件"的。

## 完整的方案

回到[第 1 节](01-clarifying-requirements.md)那个场景：一个通用的、以英文为主的基础模型，从零预训练，算力大约 $6 \times 10^{22}$ FLOPs（约 10,000 个 A100 GPU-day），数据来自网页爬取加上有授权的书籍、代码和论文，在做过去污染的 MMLU / ARC / HellaSwag / HumanEval 套件上评估，将来要以每天几十亿 token 的量级重负载服务。下面是整个系统，每个选择都拍了板，并附上它胜出的理由。

| 决策 | 选择 | 它为什么胜出 |
|---|---|---|
| 从零训的正当性 | 开头就说清楚；只因为没有开放基础模型覆盖目标，才继续往下走 | 几乎没有人应该从零预训练；不用等人问就说出这一点，是第一个资深信号 |
| 抽取 | 从 WARC 重新抽取，去页面模板噪声，配 URL 黑名单 | 抽取出垃圾会把重复数量撑大，并毒害下游每一个过滤器 |
| 语言识别 | fastText 一类的路由；英文主流 + 分流出来的多语言切片 | 按语言分别设阈值，英文才不会把门槛定得饿死其他文字 |
| 质量过滤 | 少量做过消融的 Gopher/C4 规则，加一个教育性分类器 | FineWeb 的消融结论：只有少数几条规则真能动 benchmark；token 少而精胜过多而杂 |
| 去重 | 精确哈希，再在 dump 内和跨 dump 做 MinHash + LSH，激进程度做过消融 | 近似重复往往只差一个时间戳，光靠精确哈希会漏掉大多数；而去重顶到最大会掉分 |
| 去污染 | 对全部四个 benchmark 做 13-gram 重叠；发现的比例公开出来 | 任何没有附带去污染说明的头条数字都该被怀疑 |
| 分词器 | 字节级 BPE，64K 词表，在最终混合数据上一次性拟合 | 以英文为主，带代码和一点多语言：128K 会让稀有 token 训练不足 |
| 定尺寸 | 7B dense，训练约 1.4T token（大约每参数 200 个 token） | 重负载服务让部署最优胜过 Chinchilla 的训练最优点 |
| 架构 | dense，GQA + RoPE + RMSNorm + SwiGLU | 服务端显存和可预测的每 token 成本；RoPE 让后期扩上下文很便宜 |
| 调度 | 线性 warmup，余弦衰减，范数 1.0 的梯度裁剪，用代理模型做 muP 调参 | 每次基础模型训练都在用的稳定性配方；便宜地调峰值学习率，一次放大 |
| 并行 | FSDP（ZeRO-3 级别）切分的数据并行，bf16 配 fp32 主权重 | 7B 的每一层都放得进单张 GPU；完整的优化器占用放不进 |
| 容错 | 分片异步 checkpoint，loss 尖峰回滚加跳过 batch，弹性重启 | 要在每隔几小时就坏一次的硬件上跑好几周；恢复是例行操作，不是事故 |
| 评估 | bits-per-byte 加上按时间切分、做过去污染的 benchmark | perplexity 被分词器绑定；随机切分会泄漏未来、让模型成绩虚高 |

**给这次训练定尺寸。** 预算是 $C \approx 6ND = 6 \times 10^{22}$ FLOPs。Chinchilla 最优（$D = 20N$）解出来大约是一个 22B 模型配 450B token。但[第 1 节](01-clarifying-requirements.md)已经把重负载服务钉死了，而[第 4 节](04-pretraining-choices.md)说这会翻转目标函数：同样的预算改成买一个 7B 模型，配 $6 \times 10^{22} / (6 \times 7 \times 10^9) \approx 1.4$T token，大约每参数 200 个 token，比 Chinchilla 点远出十倍。每单位训练 FLOP 换来的 loss 更差；而每个服务 token 的成本，永久地，比 22B 那个方案大约好 3 倍。

**token 供给。** 这条流水线供得上 1.4T 干净 token 吗？漏斗的保留率是原始字节数的个位数百分比（[第 2 节](02-the-data-pipeline.md)），而这是设计如此，不是失败。示意漏斗：1 PB 原始爬取文本，约 20% 熬过抽取，其中约 60% 在语言识别后属于目标语言，其中约 30% 过了质量过滤，其中约 50% 过了去重，端到端落在 2% 附近，大约 20 TB，按每 token 4 字节算约 5T token。教育性分类器再从中留下最好的约 1.4T，也就是 FineWeb-Edu 那一手：拿体量换质量。先例说明这个量很宽裕：FineWeb 把 96 个 Common Crawl dump 蒸出了 15T token。而万一供给不够，Muennighoff 的结论适用：重复到大约四个 epoch 以内，效果和用新鲜 token 差不多。

**GPU 时间。** 10,000 个 A100 GPU-day，按 bf16 峰值 312 TFLOP/s 算是 $2.7 \times 10^{23}$ 峰值 FLOPs，所以要交付 $6 \times 10^{22}$ 只需要约 22% 的 MFU，低于一次调优良好的训练能拿到的 30% 到 50%（[第 5 节](05-systems.md)）。按 40% MFU 算，这次训练大约需要 5,600 个 GPU-day：在 1,024 张 GPU 上（集群规模为示意值）不到六天，预算的其余部分被消融实验、重启，以及调调度用的代理训练吃掉。真正紧张的预算不是 FLOPs，而是第一个训练 token 之前那几周的流水线工作。

**显存与并行。** 混合精度 Adam 每个参数要 16 字节（[第 5 节](05-systems.md)），所以这个 7B 模型的训练占用是 112 GB，超过 80 GB 的 A100。朴素数据并行因此复制不了它，但切分就能解决，不需要什么英雄主义：光是把 FSDP 铺到 64 个 rank 上，常驻占用就降到每张 GPU 约 $16 \times 7\times10^9 / 64 \approx 1.75$ GB，剩下的显存留给激活。7B 这个尺度上张量并行和流水线并行都不需要；这是全章最简单的并行方案，而这本身就是那个部署最优的定尺寸决策带来的结果。

**第一个月会坏在哪里。** 早期运维由三种失效模式主导，所以在开跑之前就要把它们的信号接好：loss 尖峰（梯度范数和 loss 告警，加上[第 5 节](05-systems.md)那套自动回滚加跳过；第一次尖峰会在训练中途到来，而不是在复盘会上），污染发现得太晚（benchmark 在一次数据刷新之后跳高，说明去污染那一遍没覆盖到新加的切片；重跑一次并重新公布比例，而不是替那个数字辩护），以及去重漏网（样本里出现逐字复述的页面模板或许可证片段，说明近似重复活了下来；先去查跨 dump 的 MinHash 那一遍和它的阈值，再去怪模型）。

## 同样的技术，在不同约束下

实践中真正要紧的复盘问题不是"哪种去重最好"，而是"在我的约束下哪种去重最好"。下面是同一套章法用三遍。只有中间那一列是上面那套方案，另外两列保持完全相同的阶段接口，几乎换掉了每一个实现选择。

| | 领域继续预训练 | 通用 7B 从零训（本章） | 前沿多语言 MoE |
|---|---|---|---|
| 语料 / 算力 | 约 30B 领域 token（示意值）；几十张 GPU 跑几天 | 约 1.4T token；10,000 个 A100 GPU-day（$6 \times 10^{22}$ FLOPs） | 约 15T token；实验室级集群跑几个月 |
| 从零训吗？ | 不：在 Llama 3 或 OLMo 上继续预训练 | 是：由一个真实的能力缺口撑起来 | 是：冲的就是前沿 |
| 分词器 | 从基础模型继承；换了它就等于被迫从零训 | 字节级 BPE，64K，在混合数据上一次性拟合 | SentencePiece，128K 以上；按语言跟踪 fertility |
| 质量过滤 | 人工整理加几条启发式规则；语料小到能人眼过一遍 | 做过消融的启发式规则加教育性分类器 | 按语言的 CCNet 式 perplexity 参考模型，加学习式分类器 |
| 去重 | 精确哈希加一遍 MinHash | 精确 + 在 dump 内和跨 dump 的 MinHash/LSH，做过消融 | 同左，但 $b$ 和 $r$ 按语言各自调，激进程度做过消融 |
| 去污染 | 仍然是必做的：对你将要报的评测集做 n-gram | 对标准套件做 13-gram，比例公开 | n-gram 加 embedding 重叠；公开分数会被人挑刺着读 |
| 架构 / 并行 | 原样沿用基础模型的架构；只用 FSDP | dense GQA 7B；FSDP，bf16，不用 TP 和 PP | MoE（DeepSeek-V3 级别：总参数 671B，激活 37B）；TP + PP + EP + ZeRO，FP8 |
| 什么算过度设计 | 换新分词器、跨 dump 的去重机制、任何 MoE | FP8、专家并行、128K 词表 | 一味最大化全局去重；把 dense 扩到同等容量 |

有两个结论掉出来。第一，左边那一列才是多数读者真实所处的位置，而它主要由删减组成：[第 1 节](01-clarifying-requirements.md)那句"几乎没有人应该从零预训练"落到实处，就是这样一套方案，分词器、架构和并行方案全部继承，所有精力都花在数据整理上。唯一一行永远不缩水的是去污染。第二，右边那一列展示了瓶颈的迁移：到了前沿规模，数据工作只是入场券，真正卡住的约束变成了互连和精度，这也是 FP8 和专家并行只出现在那一列的原因。

## 每个约束各自决定什么

压缩版的决策指南。从需求里对到左边那一列，右边几列会告诉你它先动哪个杠杆，然后你才需要去比较具体工具。

| 你的约束 | 它动的杠杆 | 经验法则 |
|---|---|---|
| 算力预算 | N 和 D 一起 | $C \approx 6ND$；C 固定时，训练最优是 $D \approx 20N$ |
| 服务量 | 定尺寸的目标函数 | 每天几十亿 token：把更小的模型过度训练到远超每参数 20 个 token |
| 独立 token 供给 | 重复次数 vs 过滤的激进程度 | 重复到约 4 个 epoch 以内，效果接近新鲜 token；再往后每多一次几乎不加分 |
| 语言混合 | 词表大小和流水线形态 | 多语言：SentencePiece 128K 以上，按语言设阈值；永远要按语言报 fertility |
| 语料规模 | 去重机制 | 语料小：全量两两比较就行。万亿 token：MinHash + LSH，把拐点放在 $J \approx (1/b)^{1/r}$ 附近 |
| 要公开报 benchmark | 去污染的深度 | 第一个训练 token 之前必做 n-gram；分数要公开时再加 embedding 重叠 |
| 参数量 vs GPU 显存 | 并行维度 | Adam 下每参数 16 字节；先切优化器状态（ZeRO-1/2），再切参数（ZeRO-3/FSDP）；只有单层放不下时才用 TP，而且只在节点内 |
| 训练时长 x 故障率 | checkpoint 间隔 | 定到"平均无故障时间乘以损失比例"可接受为止；写操作要分片、要异步 |
| 稳定性 vs 吞吐 | 精度 | 默认 bf16 配 fp32 主权重；只有在前沿规模、受互连限制时才上 FP8 |

## 最小可运行的去重

每篇流水线文章收到的反馈都一样：读者对着"MinHash 加 LSH"点头，却还是看不出为什么光有精确哈希不够。所以这里把本章杠杆最大的那个阶段放进一个文件，零安装。每个生产组件都被换成接口相同的最小实现：哈希族是加盐的标准库哈希，LSH 桶变成直接比较（在玩具规模上没问题），语料是七篇文档，其中一篇是完全重复，两篇是有小改动的近似重复，另外三篇各自独立。形态才是重点；[第 3 节](03-data-quality.md)会把这个文件里的每个函数都升级一遍。

```python
"""Exact + MinHash near-duplicate dedup in one file, runnable with no installs."""
import hashlib

def shingles(text, n=3):
    """Overlapping word n-grams; production: 5-grams over normalized text."""
    words = text.lower().split()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}

def h(seed, shingle):
    """One of k independent hash functions, made by salting a stdlib hash."""
    return int.from_bytes(hashlib.md5(f"{seed}:{shingle}".encode()).digest()[:8], "big")

def minhash(shingle_set, k=64):
    """Keep the minimum of each salted hash; production: MinHash + LSH banding."""
    return [min(h(seed, s) for s in shingle_set) for seed in range(k)]

def estimate_jaccard(sig_a, sig_b):
    """Fraction of matching signature entries estimates J(A, B)."""
    return sum(a == b for a, b in zip(sig_a, sig_b)) / len(sig_a)

DOCS = [
    ("web/chinchilla-1", "The Chinchilla result says parameters and tokens should "
     "grow together, at roughly twenty tokens per parameter, when minimizing loss "
     "for a fixed training compute budget, and a seventy billion parameter "
     "Chinchilla beat a far larger Gopher at equal compute."),
    ("web/chinchilla-2", "The Chinchilla result says parameters and tokens should "
     "grow together, at roughly twenty tokens per parameter, when minimizing loss "
     "for a fixed training compute budget, and a seventy billion parameter "
     "Chinchilla beat a far larger Gopher at equal compute."),  # exact duplicate
    ("web/chinchilla-3", "Updated 2024: The Chinchilla result says parameters and "
     "tokens should grow together, at about twenty tokens per parameter, when "
     "minimizing loss for a fixed training compute budget, and a seventy billion "
     "parameter Chinchilla beat a far larger Gopher at equal compute."),  # near-dup
    ("blog/dedup-1", "Near-duplicate documents dominate the web because the same "
     "page recurs across crawl snapshots with only a timestamp or header changed, "
     "so exact hashing alone removes almost none of them."),
    ("blog/dedup-2", "Near-duplicate documents dominate the web because the same "
     "page recurs across the crawl snapshots with only a timestamp or a header "
     "changed, so exact hashing alone removes almost none of them."),  # near-dup
    ("paper/tokenizer", "An English-heavy vocabulary fragments other scripts into "
     "many more tokens per word, so fertility must be reported per language."),
    ("code/readme", "This repository trains a small decoder-only transformer with "
     "warmup, cosine decay, and gradient clipping at norm one."),
]

def dedup(docs, threshold=0.7):
    kept, seen_exact = [], set()
    exact_drops = fuzzy_drops = 0
    for doc_id, text in docs:
        fingerprint = hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()
        if fingerprint in seen_exact:                    # stage 1: exact hash
            exact_drops += 1
            print(f"drop {doc_id:<18} exact duplicate (same content hash)")
            continue
        seen_exact.add(fingerprint)
        sig = minhash(shingles(text))                    # stage 2: MinHash
        best_id, best_j = None, 0.0
        for kept_id, kept_sig in kept:                   # toy scale: compare all;
            j = estimate_jaccard(sig, kept_sig)          # production: LSH buckets
            if j > best_j:
                best_id, best_j = kept_id, j
        if best_j >= threshold:
            fuzzy_drops += 1
            print(f"drop {doc_id:<18} near-duplicate of {best_id} "
                  f"(J~{best_j:.2f}, exact hash differs)")
            continue
        kept.append((doc_id, sig))
        print(f"keep {doc_id:<18} best match J~{best_j:.2f}, below threshold")
    print("\nfunnel:")
    print(f"  raw documents       {len(docs)}")
    print(f"  after exact dedup   {len(docs) - exact_drops}  (-{exact_drops} byte-identical)")
    print(f"  after fuzzy dedup   {len(kept)}  (-{fuzzy_drops} near-duplicates exact hashing missed)")
    print(f"  keep rate           {len(kept) / len(docs):.0%}")

dedup(DOCS)
```

跑一遍，输出就是本章去重论点的缩微版。那个完全一样的镜像（`web/chinchilla-2`）在第一阶段就死在内容哈希上。两篇近似重复直接从这个哈希底下溜了过去，因为一个 "Updated 2024" 前缀或者插进去的一个冠词就会改掉每一个字节级指纹，它们只能被 MinHash 阶段抓住，两篇对各自的原文估计出的 J 都在 0.70 左右；而改写过但相关的 `web/chinchilla-3` 跟那些独立文档比，分数接近零。漏斗打印出来是：7 篇原始文档，精确去重后 6 篇，模糊去重后 4 篇，保留率 57%，三次丢弃里有两次是精确哈希看不见的。把加盐哈希换成真正的 MinHash 族，把全量两两比较的循环换成调好 $b$ 和 $r$ 的 LSH 分段，再放到 dump 内和跨 dump 上去跑，你就重建出了本章杠杆最大的那个阶段。
