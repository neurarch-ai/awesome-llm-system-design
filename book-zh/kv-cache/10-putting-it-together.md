# 10. 把它们拼起来：完整的方案

第 1 到 6 节把每根杠杆连同它的选项和取舍都讲了一遍，第 7 节展示了真实团队在哪里分道扬镳。
但它们都没有给出一个把每个决策都做完的完整系统。这一节收官，做三件事：
给一套有主张的默认技术栈，让选择困难不至于卡住第一次搭建；
把本章的场景从头到尾走一遍，每个选择都拍板并算清尺寸；
再展示同样这些决策在约束变了之后会怎么翻转。
最后是一个最小的可运行 KV cache 模型，一个文件，零安装。

## 默认技术栈：从这里开始，有理由再偏离

本章每根杠杆都有两到五个说得通的选项，第一次搭系统的人光是对比服务引擎就能烧掉一周，
一个 token 都还没解出来。别这么干。下面这套栈是第一次做生产搭建时一个合理的默认值，
每一行都写清楚什么时候该偏离、哪一节解释了为什么。引擎每年都在换，
但杠杆本身（压小每条记录、把池子分页、跨请求复用、扩展位置、连续批处理）不变，
所以要按杠杆做决定，任何具体引擎都当成可替换的。

| 杠杆 | 默认值 | 什么时候偏离 | 为什么（章节） |
|---|---|---|---|
| 注意力变体 | GQA，每个 KV 头配 4 到 8 个 query 头（Llama 3、Mistral、Gemma 都是这样） | 在目标上下文下 cache 仍然是墙，而且训练在你手里：上 MLA | [3](03-shrinking-the-cache.md) |
| KV 精度 | 先 FP16；长上下文评测过了再上 FP8 | checkpoint 固定且显存吃紧：逐 token 的 INT4，key 的位宽高于 value | [3](03-shrinking-the-cache.md) |
| 显存管理 | PagedAttention，16 token 一块，写时复制共享 | 单卡上跑单条序列：连续分配就行，还更简单 | [4](04-paged-and-shared.md) |
| 前缀复用 | 前缀缓存，把所有稳定内容放在 prompt 最前面 | 流量每个请求都完全不同：cache 永远不命中 | [4](04-paged-and-shared.md) |
| 分叉复用 | 扁平的前缀缓存 | agent 树或 few-shot 扇出共享会分叉的前缀：上 RadixAttention | [4](04-paged-and-shared.md) |
| 上下文扩展 | YaRN 加一小段微调，扩到训练长度的 4 到 16 倍 | 永久流式且不需要上下文中段召回：滑动窗口加 sink | [5](05-long-context.md) |
| prefill 策略 | 一次并行前向；只在目标 batch 下会 OOM 时才用分块 prefill | prompt 很短：分块白白增加延迟 | [5](05-long-context.md) |
| 批处理 | 第一天就上连续批处理 | 共享服务永远不要偏离；静态批处理只用于离线单租户跑批 | [6](06-serving-and-scaling.md) |
| 投机解码 | 关 | 中低 batch 加结构化输出（代码、模板）：打开 | [6](06-serving-and-scaling.md) |
| 淘汰 | 不做；只用无损杠杆 | cache 真的装不下且忘掉旧上下文可以接受：sink 加窗口；必须保持精确：query 感知的稀疏化（Quest） | [3](03-shrinking-the-cache.md) |

最后一行是最该先记住的：淘汰是这张表里唯一会改变模型能回答什么的杠杆。
它上面的全都是无损的，所以先把无损那几行用尽再碰它；真要碰，
就用大海捞针评测把关，而不是困惑度。

## 完整的方案

回到[第 1 节](01-clarifying-requirements.md)的场景：一个 RAG 聊天机器人，
今天 32k 上下文、目标 128k，首 token 低于 2 秒，token 间隔低于 100 毫秒，
每个 GPU 节点 500 到 1000 条并发会话且有 3 倍的尖峰，一段 4k 的 system prompt 被每个请求共享，
模型是我们自己的。下面是整个系统，每个选择都已拍板，附上它为什么胜出。

| 决策 | 选择 | 为什么它胜出 |
|---|---|---|
| 注意力变体 | GQA，8 个 KV 头，从 MHA checkpoint uptrain 而来 | 花预训练算力的约 5%，就拿到接近 MHA 的质量和小 4 倍的 cache；权重在我们手里，这是最便宜的一笔大收益 |
| KV 精度 | FP8，过评测门禁；key 的精度高于 value | 把 cache 又砍一半；面试官允许在质量守住的前提下量化，而 key 是敏感的那个张量 |
| 显存管理 | PagedAttention，16 token 一块，写时复制 | 4k 到 32k 长短混杂会把连续缓冲区打碎；分页把 20% 到 40% 浪费掉的 HBM 找回来 |
| 前缀复用 | 以 4k system prompt 为键的前缀缓存，稳定内容在前；会话历史做逐轮缓存 | 共享 prompt 是首 token 上最大的一根杠杆；逐轮扩展让多轮续聊变得很便宜 |
| 集群路由 | 缓存感知路由（llm-d 风格） | 按节点存的前缀缓存在机队规模上会被切碎；轮询会悄无声息地毁掉命中率 |
| 上下文扩展 | YaRN 加一小段长序列微调，服务于 128k 的路线图 | 扩 4 倍（训练 32k 服务 128k），质量曲线比普通插值更好 |
| prefill 策略 | 命中前缀的请求走一次前向；冷的 32k prompt 走分块 prefill | 在高并发下把 prefill 峰值显存框住；前缀缓存热起来之后，冷路径很少见 |
| 批处理 | 连续批处理，配准入上限和对闲置会话的 LRU 淘汰，保护在途序列 | 并发目标逼着必须上；上限加淘汰才是吸收 3 倍尖峰而不抢占活跃解码的那一手 |
| 投机解码 | 上线时关闭 | 这个节点跑在高 batch 上，draft 的算力只会加压，解不开显存带宽这个瓶颈 |
| 上下文淘汰 | 不做；只用无损杠杆 | RAG 聊天机器人必须能回答检索文档里任何位置的问题；有损淘汰恰恰在这种负载上失效 |

**每个 token 多少字节。** 取 $L = 32$、$h_{\text{kv}} = 8$、$d_{\text{head}} = 128$、FP16，
每个 token 每层花 $2 \times 8 \times 128 \times 2 = 4096$ 字节，整个栈加起来每 token 128 KB
（[第 2 节](02-the-cost-model.md)）。同一个模型用 MHA 每 token 要付 512 KB，GQA 已经砍掉了 4 倍。
FP8 再把 128 KB 减半到 64 KB。下面每一个数字都是从这一个数推出来的。

**目标上下文下的 cache。** 一条满的 32k 会话（32 768 个 token）在 FP16 下是
$2 \times 32 \times 32768 \times 8 \times 128 \times 2 \approx 4.29$ GB，
跟[第 8 节](08-interview-qa.md)里 Llama 3 8B 那笔账是同一个算式，FP8 下是 2.15 GB。
那段 4k 的共享 system prompt 只存一份，靠写时复制的块共享，
所以每条会话真正独占的大约是 28.7k 个 token，FP8 下接近 1.9 GB（示意）。

**显存预算对节点。** 朴素的需求就是[第 8 节](08-interview-qa.md)里那场灾难：
1000 条会话乘 4.29 GB 等于 4.29 TB，而 H100 节点只有 640 GB。
这套方案从两头把这个缺口收拢。FP8 加上共享前缀，把单会话占用压到约 1.9 GB；
再扣掉模型权重（FP16 下 14 GB）和运行时开销，节点大约还剩 560 GB 留给 KV，
也就是同时驻留约 290 条长满的 32k 会话（示意）。
剩下的那个因子来自负载本身：500 到 1000 条会话里大多数在用户两轮之间是闲着的，
所以闲置会话被 LRU 淘汰，靠逐轮前缀缓存续上，而不是整条重新 prefill。
128k 的路线图把这笔账再乘 4 倍变差，这也是为什么 MLA（在 GQA 之下再省约 4 倍，
[第 3 节](03-shrinking-the-cache.md)）是排在后面待办，而不是被否掉。

**首 token 延迟。** prefill 受算力限制，随 prompt 长度增长（[第 2 节](02-the-cost-model.md)）。
前缀缓存命中时，那 4k system prompt 一分钱不花，只需要 prefill 每条会话各自的后缀；
碰上冷的 32k prompt，分块 prefill 用把这一趟串行化的代价框住峰值显存，
所以块大小要对着 2 秒的预算调，而不是随手定一个。

**第一个月会坏在哪。** 早期运维里有三种失败模式占大头，所以上线前就把它们的信号接好：
集群前缀命中率（[第 6 节](06-serving-and-scaling.md)里那个按节点存 cache 的问题；
命中率远低于单节点数字，说明坏的是路由而不是缓存）、
FP8 下的长上下文召回（[第 5 节](05-long-context.md)那张大海捞针网格，
因为困惑度会看起来一切正常，而中段深度的检索已经在悄悄退化）、
以及 3 倍尖峰下活跃会话被抢占（回来的用户如果 cache 在对话中途被淘汰，
就要莫名其妙地重付一次 prefill，所以要把被淘汰的在途序列，
以及续聊会话的首 token 延迟，跟冷启动的分开来跟踪）。

## 同一套技术在不同约束下

实践中真正重要的复盘问题不是"哪种 cache 技巧最好"，而是"在我的约束下哪种最好"。
下面是同一个服务问题搭三遍。只有中间那一列是上面那套方案，另外两列杠杆完全一样，
但几乎每一个设置都换了。

| | 固定 checkpoint，一张 24 GB 显卡 | 规模化的 RAG 聊天机器人（本章） | 成本目标极端的消费级对话 |
|---|---|---|---|
| 模型控制权 / 负载 | 第三方开源权重，不能重训；并发用户就那么几个 | 权重自有；每节点 500 到 1000 条会话，32k 到 128k 上下文 | 权重自有；海量对话流量，100 轮以上的历史，成本就是产品约束 |
| 注意力变体 | checkpoint 是什么就是什么（通常是 GQA）；这不是你手里的杠杆 | GQA 8 头，uptrain 得到；MLA 排队等 128k | 训练时就做进去的 MQA 加跨层 KV 共享（Character.AI 那个画像，[第 7 节](07-how-teams-do-it-in-production.md)） |
| KV 精度 | 逐 token 的 INT4 配全精度近期窗口；剩下唯一的架构杠杆 | FP8，key 高于 value，过评测门禁 | 训练时就是原生 int8，所以没有事后量化的损失 |
| 显存管理 | 分页块（任何现代引擎都自带） | 分页、写时复制、准入上限加闲置会话 LRU | 分页，再加局部 / 全局混合的滑动窗口层来框住增长 |
| 前缀复用 | 只有 prompt 真的重复才做；个人助手往往根本没有共享前缀 | 4k system prompt 全机队缓存，配缓存感知路由 | 对话轮次上的滚动哈希 LRU 树，命中率约 95% |
| 上下文策略 | 待在 checkpoint 训练过的窗口里；没有微调预算去扩 | YaRN 加一小段微调扩到 128k；要求整篇文档召回，所以不做淘汰 | 滑动窗口加 sink；闲聊忘掉很久以前的内容可以接受 |
| 批处理 / 解码 | batch 很低，所以投机解码在这里是难得的大收益 | 高 batch 下的连续批处理；投机解码关闭 | 极高 batch 下的连续批处理；每根杠杆都叠上，每一根都过评测门禁 |
| 什么算过度设计 | uptraining、缓存感知路由、任何分布式的东西 | MQA（没必要担的质量风险）、INT2 的 KV | 整篇文档召回那一整套机制；这个负载不需要 |

从中掉出来两条经验。第一，左边那一列由你碰不到的东西定义：checkpoint 固定时，
训练期那几行（GQA 比例、MLA、原生 int8）全部消失，整场游戏只剩服务期的杠杆，
这正是[第 3 节](03-shrinking-the-cache.md)里架构和量化的那条分界线。
第二，右边那一列展示的是当质量还有余量、成本才是墙的时候会变成什么样：
MQA 和激进的开窗对 RAG 聊天机器人是错的，因为检索要求精确召回；
对闲聊却是对的，因为它不要求。杠杆清单从头到尾没变，变的是负载对损失的容忍度。

## 每种约束决定什么

压缩版的决策指南。从需求里读出左边那一列，右边几列告诉你，
在你去对比任何引擎之前，它先动的是哪根杠杆。

| 你的约束 | 它动的杠杆 | 经验法则 |
|---|---|---|
| 模型控制权 | 注意力变体 | 权重自有：默认 GQA，cache 仍是墙时上 MLA。checkpoint 固定：KV 量化是剩下唯一的架构杠杆 |
| 上下文长度目标 | 位置方案、prefill 策略 | 超训练长度 2 到 4 倍：PI 加微调。4 到 16 倍：YaRN。prompt 会让 prefill OOM：分块，块大小对着首 token 预算调 |
| 并发目标 | 批处理、显存管理 | 并发序列多过寥寥几条之后，连续批处理加分页就是必选，而且它们是一起上的 |
| token 间延迟预算 | cache 体积（$h_{\text{kv}}$、$b$） | decode 受带宽限制；减少每一步读的字节数，不要去加 FLOPs |
| 首 token 预算 | 前缀缓存、块大小 | 前缀命中就完全跳过 prefill；把稳定内容放最前面且逐字节一致，否则 cache 永远不触发 |
| 请求之间的共享内容 | prompt 排布 | system prompt 和共享文档要排在任何逐用户 token 之前；开头一个变量 token 就能让 cache 从那里开始全废 |
| 检索的质量底线 | KV 位宽、淘汰这一族 | 用大海捞针召回把关，不是困惑度；key 的量化比 value 保守；要整篇文档召回就绝不做有损淘汰 |
| 硬性显存上限 | 淘汰这一族 | 能接受遗忘：sink 加滑动窗口。必须保持精确：query 感知的稀疏化，东西全留着但读得更少 |
| 多节点机队 | 路由 | 按节点存的前缀缓存在规模上会被切碎；缓存感知路由或者分布式缓存能把命中率找回来 |
| 尖峰流量 | 准入和淘汰策略 | 限制准入，只 LRU 淘汰闲置会话，保护在途解码；绝不为了放进一个新请求而抢占一场活跃的对话 |

## 最小的可运行 KV cache

每一份服务引擎教程的读后感都一样：读者装完 CUDA、vLLM 和一个模型 checkpoint，
还是看不到那两个决定一切的数字。所以这里把它们放在一个文件里，零安装。
第 1 部分是[第 2 节](02-the-cost-model.md)的体积公式，
在[第 3 节](03-shrinking-the-cache.md)的四种注意力变体上分别求值。
第 2 部分是[第 4 节](04-paged-and-shared.md)里那场玩具分配器的比赛：
同样 60 GB 的 KV 预算（示意），一边是朴素的连续预留，一边是定长的分页块，
跑在一组固定随机种子的会话长度混合上。形状本身就是这一课；
本章每一节动的都是这个文件里的某一个常数。

```python
"""KV-cache arithmetic and a toy paged allocator, runnable with no installs."""
import random

GB = 1e9

def kv_bytes_per_token(n_layers, h_kv, d_head, bytes_per_elem):
    """Section 2 formula without S and B: 2 (K and V) x L x h_kv x d_head x b."""
    return 2 * n_layers * h_kv * d_head * bytes_per_elem

def mla_bytes_per_token(n_layers, d_c, bytes_per_elem):
    """MLA caches one latent of size d_c per layer instead of full K and V."""
    return n_layers * d_c * bytes_per_elem

# --- part 1: the size formula for a 7B-class model (L=32, d_head=128, FP16) --

L, D, FP16 = 32, 128, 2
variants = {
    "MHA (h_kv=32)": kv_bytes_per_token(L, 32, D, FP16),
    "GQA (h_kv=8) ": kv_bytes_per_token(L, 8, D, FP16),
    "MQA (h_kv=1) ": kv_bytes_per_token(L, 1, D, FP16),
    "MLA (d_c=512)": mla_bytes_per_token(L, 512, FP16),
}
S = 32768  # the chapter's 32k session
print(f"KV bytes per token, and one {S}-token session:")
for name, per_tok in variants.items():
    ratio = per_tok / variants["MHA (h_kv=32)"]
    print(f"  {name}: {per_tok:>7,} B/token  x {S} tokens = "
          f"{per_tok * S / GB:5.2f} GB  ({ratio:.1%} of MHA)")

# --- part 2: contiguous reservation vs paged blocks under one HBM budget ----

BUDGET = 60 * GB      # KV budget: one 80 GB GPU minus weights and overhead
MAX_CTX = 32768       # every request may grow to the 32k cap
PER_TOK = variants["GQA (h_kv=8) "]
BLOCK = 16            # tokens per block, as in PagedAttention

random.seed(0)
def sample_len():
    # RAG chatbot mix: 4k shared system prompt plus a long tail of history
    return min(MAX_CTX, 4096 + int(random.expovariate(1 / 6000)))
lengths = [sample_len() for _ in range(2000)]

# contiguous: every arrival reserves the full max-context buffer up front
contig_cap = int(BUDGET // (MAX_CTX * PER_TOK))
used = sum(lengths[:contig_cap]) * PER_TOK
reserved = contig_cap * MAX_CTX * PER_TOK
print(f"\ncontiguous: {contig_cap:>3} sequences fit; "
      f"{1 - used / reserved:.0%} of reserved bytes sit unused")

# paged: allocate ceil(len/BLOCK) blocks on demand from one shared pool
pool = int(BUDGET // (BLOCK * PER_TOK))
admitted = wasted_toks = used_toks = 0
for n in lengths:
    blocks = -(-n // BLOCK)          # ceil division
    if blocks > pool:
        break
    pool -= blocks
    admitted += 1
    used_toks += n
    wasted_toks += blocks * BLOCK - n
print(f"paged:      {admitted:>3} sequences fit; "
      f"{wasted_toks / (used_toks + wasted_toks):.2%} of allocated bytes sit unused")
print(f"paged admits {admitted / contig_cap:.1f}x more concurrent sequences "
      f"from the same {BUDGET / GB:.0f} GB")
```

跑一下，第 1 部分会打印出本章自己的那些数字：MHA 每 token 512 KB，
GQA 是 128 KB（25.0%），MQA 是 16 KB（3.1%），MLA 的潜向量是 32 KB（6.2%），
于是一条 32k 的会话分别是 17.18 GB、4.29 GB、0.54 GB 和 1.07 GB；
其中 GQA 那个数字，正是上面方案里用的单会话成本。
第 2 部分接着展示了为什么分页到处都在用：连续预留只装下 13 条序列，
预留的字节里有 70% 闲置；分页块装下 40 条，只浪费 0.07%，
并发提升 3.1 倍，落在[第 4 节](04-paged-and-shared.md)所说的 2 到 4 倍区间里。
这个玩具给每条序列都按 32k 上限预留，是朴素做法里的最坏情况；
真实的连续分配器预留得少一些，但仍然会损失该节引用的那 20% 到 40%。
玩具里的每一块都对应一个生产组件：`kv_bytes_per_token` 是每份容量规划的起点公式，
块池加向上取整的分配就是去掉注意力 kernel 之后的 vLLM 块表，
固定种子的长度混合就是让碎片开始咬人的变长流量；
而玩具刻意省掉的那一样东西，序列之间的块共享，就是前缀缓存，
有了它，全部 2000 条序列都可以指向同一份 256 个块的 system prompt。
