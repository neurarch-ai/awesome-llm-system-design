# 10. 把它们拼起来：完整的方案

第 1 到 6 节把每个机制连同它的选项和取舍讲了一遍，第 7 节展示了真实团队在哪里
分岔。这些都没展示的，是一次把每个决定都做完的完整适配。这一节收官，做三件事：
给出一套有主见的默认配方，让人不会因为选择困难而跑不出第一次训练；把本章的临床
场景从头到尾走一遍，每个选择都拍板并算清预算；再展示当约束变了以后，同样这些
决定会怎么翻转。最后以核心机制最小可运行的演示收尾，一个文件，不用装任何东西。

## 默认技术栈：从这里开始，有理由再偏离

这条流水线的每个阶段都有两到五个说得过去的选项，一个第一次上手的人，可以在还没
训练出一步之前就先烧掉一周去比对各家重缩放论文。别这样。下面这套配方是第一次
适配的合理默认值；每一行都写清了什么时候该偏离，以及哪一节解释了原因。方法会
演进，但每个阶段的接口不会（选适配方式、配数据混合、排学习率调度、重缩放位置、
分阶段推长度、给结果把关），所以按阶段来做决定，把任何具体方法都当成可替换的。

| 阶段 | 默认选择 | 什么时候偏离 | 为什么（节） |
|---|---|---|---|
| 适配方法 | 带回放的完整继续预训练（DAPT） | 领域内 token 不到大约 10 亿：改用 SFT 或 RAG；遗忘预算很严：LoRA / QLoRA | [3](03-the-mid-training-phase.md) |
| 领域数据混合 | 领域语料加 10% 的通用数据回放 | 领域本来就贴近预训练分布：降低升温峰值可以顶替额外的回放 | [3](03-the-mid-training-phase.md) |
| LR 调度 | 从衰减后的地板重新升温到一个适中的峰值（预训练峰值的一个零头），cosine 再衰减，在尾部退火最高质量的数据 | 训练停滞：把峰值适度提高；通用 benchmark 回退：先降峰值，再去动数据 | [3](03-the-mid-training-phase.md) |
| RoPE 缩放方法 | YaRN（逐频段的非均匀混合加 attention 温度修正） | 扩展倍数在大约 8 倍以内且微调预算接近零：NTK-ABF；2M 以上的极端目标：LongRoPE 搜索出来的重缩放 | [4](04-context-extension.md) |
| 上下文扩展配方 | 上采样真正的长文档，分阶段增加长度，在长上下文训练之前先做重缩放 | 绝不用拼起来的无关短文档；那是在教模型远处的 token 不重要 | [4](04-context-extension.md) |
| 服务姿态 | GQA 基础模型、FlashAttention、paged attention；64K 及以上上 KV 量化 | 产品永远只会见到短 prompt：整套长上下文服务栈都可以跳过 | [6](06-serving-and-scaling.md) |
| 评估 | 前后各跑一遍完整通用套件，NIAH 召回率随深度热力图，RULER 作为放行门槛 | 没有例外。第一步训练之前就把回退红线定死 | [5](05-evaluation.md) |

最后一行是新手会跳过、然后后悔的那一行：没有前后对比的通用套件，每一个关于回放和
峰值的决定都只是感觉，而遗忘在领域这一片里是无声的。事先把回退红线定下来是一场会
的事；放行之后才发现遗忘，就是一次回滚。

## 完整的方案

回到第 [1](01-clarifying-requirements.md) 节那个场景：一个 8K 窗口的通用基础模型，
400 亿 token 去标识化的临床记录，服务文档长度的 p95 是 60K token，MMLU、GSM8K 和
指令跟随上有两个百分点的回退预算，跑在我们自己的基础设施上，适配后的基础模型之后
要流向后训练。下面是整次适配，每个选择都已拍板，并附上它胜出的理由。

| 决定 | 选择 | 为什么它胜出 |
|---|---|---|
| 两条轴线的顺序 | 先 DAPT，再上下文扩展，分开把关 | 两条轴线相互独立、失败模式不同；顺序两趟（Code Llama 那种形态）能让每道门槛各抓各的回退 |
| 适配方法 | 在 400 亿临床 token 上做完整 DAPT | 这是词汇和语域上一次大范围的分布迁移；SFT 教的是格式不是先验，而 adapter 在这个规模上会撞天花板 |
| 回放 | 10% 的通用网页数据，全程保持不变 | 回放比例过了大约 5% 之后遗忘急剧下降，而领域收益几乎不受影响；10% 是实践上的平衡点 |
| LR 调度 | 重新升温到适中峰值，cosine 再衰减，梯度裁剪到范数 1.0，尾部做质量退火 | 衰减后的地板会停滞；原始峰值会抹掉基础模型；峰值是权衡遗忘与学习的那一个旋钮 |
| 目标长度 | 64K，不是 128K | p95 是 60K；prefill 的 attention 是平方的，所以用不上的余量在每一条长请求上都要付钱 |
| 重缩放方法 | s = 8 的 YaRN，带 attention 温度修正 | 指令跟随上两个点的红线，让短上下文回退成了主要风险；YaRN 放过了均匀 PI 会挤压的高频频段 |
| 长上下文数据 | 上采样的长临床文档，加上"事实在前、问题在后"的合成插入，按 8K 到 32K 到 64K 分阶段推进 | 要真实的长距离依赖，不是拼起来的短记录；分阶段比一次跑满长度更便宜也更稳 |
| 评估门槛 | 通用套件对着 2 个百分点的红线，NIAH 召回率随深度热力图，RULER 有效长度门槛 | 检索坏掉的时候 perplexity 已经饱和了；只做 NIAH 又是锚在边缘的单跳 |
| 服务 | GQA 基础模型、FlashAttention、paged attention、int8 KV cache | 一个 64K 的产品同时在长 prompt 上受 prefill 限制、在 batch size 上受 KV 限制 |

**token 预算。** DAPT 的混合是 400 亿临床 token 加 10% 回放：400 亿 / 0.9，总共大约
444 亿 token，其中大约 44 亿是回放的通用数据。上下文扩展阶段的成本大约是原预训练
token 量的 0.1%（第 [4](04-context-extension.md) 节里 YaRN 的那个数字）；对着一个
示意性的 15 万亿 token 预训练基础模型，这就是大约 150 亿长上下文 token，并且分阶段
安排，让早期 8K 到 32K 那几步跑在更便宜的短序列上。所以整次适配大约 600 亿 token，
不到那个示意性从零训练成本的千分之五，这就是整章在经济上的论据。

**回放比例。** 10% 意味着每十个 token 里就有一个，在每一步都对通用分布保持一个活的
梯度，于是优化器不会连续见到一长串纯领域 batch，久到足以走出通用极小值。要保持恒定，
不要退火：逐渐衰减的回放份额，恰好会在权重正在固化的时候把漂移重新引进来（第
[8](08-interview-qa.md) 节）。

**64K 下的显存。** 用第 [6](06-serving-and-scaling.md) 节的 KV cache 公式，配一个
示意性的 8B 级别配置（32 层、GQA 下 8 个 KV head、head 维度 128、fp16），单条 64K
token 的序列占 2 x 32 x 8 x 128 x 65536 x 2 字节，大约 8.6 GB 的 KV cache。同一个
模型如果用完整多头注意力（32 个 KV head），每条序列要 34.4 GB，这就是为什么 GQA
必须在预训练时烙进去、而不能事后加装；int8 KV 量化又能把这 8.6 GB 砍一半，paged
attention 则让参差不齐的 batch 长度不至于把它切碎。prefill 是另一笔成本：64K 是训练
长度的 8 倍，所以 prefill 的 attention FLOPs 大约涨 64 倍，长请求在第一个输出 token
之前就已经受 prefill 限制了。

**第一个月会坏在哪。** 早期运维主要被三个失败信号占据，放行之前就把它们接上：通用
benchmark 回退（每个候选 checkpoint 都跑完整套件，对着 2 个百分点的红线；遗忘在领域
这一片里是无声的，一次以 MMLU 为代价的临床能力提升，对产品是净亏），中部深度的召回
空洞（NIAH 热力图在 40% 到 60% 深度处凹下去；用户会反馈出院小结中间的事实被漏掉，
而平均召回的数字看着挺好），以及重新升温期间的 loss 尖峰（峰值太高或者遇到了坏
batch；裁剪到范数 1.0，回退到上一个好的 checkpoint，跳过这个 batch，如果尖峰反复
出现就把峰值调低）。

## 同样的技术在不同约束下

实践中真正要紧的复盘问题不是"哪种重缩放方法最好"，而是"在我的约束下哪种最好"。
下面是同一个适配问题的三个版本。只有中间那一列是上面那套方案；另外两列保持完全
相同的阶段接口，却几乎把每个选择都换掉了。

| | 轻量的内部助手 | 临床 64K（本章） | 128K+ 的合同平台 |
|---|---|---|---|
| 语料 / 目标长度 | 大约 5 亿 token 的内部文档；prompt 不到 8K | 400 亿临床 token；文档 p95 为 60K | 领域迁移幅度不大；整份合同，配置 128K |
| 适配 | 用 SFT 管格式，用 RAG 管事实；顶多加一点 QLoRA 微调 | 带 10% 回放的完整 DAPT | 轻量 DAPT 或者干脆不做；长度这条轴占主导 |
| 上下文扩展 | 不做；训练时的窗口已经覆盖了流量 | s = 8 的 YaRN，分阶段到 64K | 分阶段 YaRN 到 128K；只有出现数百万 token 的目标时才上 LongRoPE 搜索出来的重缩放 |
| 训练 token | 几千到几百万对 SFT 样本 | 总共大约 600 亿（示意） | 由长上下文阶段主导；长数据的整理是卡脖子的约束 |
| 服务 | 标准的短上下文栈 | GQA、FlashAttention、paged attention、int8 KV | 以上全部再加分块 prefill；有效 batch size 由 KV 显存决定 |
| 评估 | 通用套件加一份留出的领域切片 | 2 个百分点的门槛、按深度的 NIAH、RULER | RULER 的有效长度是头条数字；没有它，配置里的 128K 什么都不说明 |
| 什么算过度设计 | 任何 RoPE 重缩放、一条回放流水线、一整套长上下文服务栈 | p95 永远用不到的 128K 余量 | 为了不在合同语料库上搭检索，而把上下文扩到 1M |

由此得出两个教训。第一，左边那一列基本上是在做减法：低于十亿 token 这道门槛，
DAPT 学到的还不如遗忘掉的多（第 [3](03-the-mid-training-phase.md) 节），所以正确
的适配是 SFT 加检索，而长度这条轴随着需求一起消失了。第二，右边那一列展示了两条轴
作为约束时的位置互换：领域那部分工作缩成了微调，数据整理和"有效长度对配置长度"
那个差距成了全部的戏，而真正给产品封顶的是服务账单，也就是每条请求都要付的平方级
prefill 和线性 KV 增长。在这两个规模上，长上下文都不能替代语料库上的检索；它们是
组合关系（第 [6](06-serving-and-scaling.md) 节）。

## 每个约束决定什么

浓缩版的决策指南。从需求里读出左边那一列，右边几列会告诉你，在开始比较任何方法
之前，它先动的是哪根杠杆。

| 你的约束 | 它动的杠杆 | 经验法则 |
|---|---|---|
| 领域内 token 数量 | 适配方法 | 不到大约 10 亿：用 SFT 或 RAG，不用 DAPT。几十上百亿：带回放的完整 DAPT |
| 遗忘预算 | 全量微调还是 adapter；峰值和回放 | 必须在构造上就有界：LoRA / QLoRA。否则：适中的峰值加大约 10% 的回放 |
| 通用能力下限 | 评估门槛 | 训练之前就把回退红线定死；每次改动前后都跑完整套件 |
| 服务长度的 p95 | 目标窗口 | 按测出来的需求扩，不是按营销数字扩；平方级 prefill 会给每一 token 的余量标价 |
| 长度缩放比 s | 重缩放方法 | 极小幅度的扩展：PI 是可接受的基线。大约 8 倍以内且微调很少：NTK-ABF。激进扩展且有短上下文红线：YaRN。2M 以上：LongRoPE |
| 短 prompt 流量占比 | 重缩放的非均匀程度 | 占比高：用 YaRN 放过的高频频段，或者 LongRoPE 的恢复步骤；均匀 PI 会对每一条短请求征税 |
| VRAM 预算 | KV cache 相关的杠杆 | GQA（预训练时烙进去）、KV 量化、paged attention；KV 随长度线性增长，最先崩的是 batch size |
| 语料库还是一份大文档 | 长上下文还是 RAG | 语料库：做检索。一份要整体推理的文档：做扩展。两者是组合关系；用扩展替代检索是错误答案 |
| 证明长度是真的 | 放行门槛 | 按深度的 NIAH 当冒烟测试，RULER 的有效长度当门槛；perplexity 只用来做早停 |

## 最小可运行的上下文扩展

每一篇长上下文文章收到的反馈都一样：读者读到"没见过的旋转角"时点点头，然后仍然
看不见机制。所以下面把本章的核心论断压进一个文件，不用装任何东西。对一个 32 维的
玩具 head，它计算每一对频率在训练过的 8K 窗口内某个位置、以及窗口之外某个位置上的
旋转角，标出训练是否曾把这个角度展示给模型（转完至少一整圈的那一对见过整个圆；更慢
的那一对只见过一小段圆弧），然后分别施加线性位置插值和 YaRN 式的逐频段混合，再检查
一遍。要看的是它的形状；第 [4](04-context-extension.md) 节就是这个文件把常数一般化
之后的样子。

```python
"""Why naive RoPE extrapolation fails and how rescaling fixes it. Stdlib only."""
import math

D = 32                  # per-head dimension (16 frequency pairs)
BASE = 10000.0          # RoPE base
L_ORIG = 8192           # trained context window
L_NEW = 65536           # target window
S = L_NEW / L_ORIG      # length scale s = 8
ALPHA, BETA = 1.0, 32.0 # YaRN ramp edges, in rotations per trained window

def theta(i):                        # original frequency of dimension pair i
    return BASE ** (-2 * i / D)

def rotations(t):                    # full rotations this pair completes in training
    return L_ORIG * t / (2 * math.pi)

def gamma(t):                        # YaRN blend: 1 = keep, 0 = interpolate
    r = rotations(t)
    if r >= BETA:
        return 1.0
    if r <= ALPHA:
        return 0.0
    return (r - ALPHA) / (BETA - ALPHA)

def yarn_theta(t):                   # per-pair blend of keep vs divide-by-s
    g = gamma(t)
    return g * t + (1 - g) * t / S

def seen(t_orig, angle):
    """Was this angle covered in training, under the ORIGINAL frequencies?
    A pair completing >= 1 rotation saw the whole circle (angles wrap mod 2*pi);
    a slower pair only ever saw the arc [0, L_ORIG * theta)."""
    if rotations(t_orig) >= 1.0:
        return True
    return angle <= L_ORIG * t_orig + 1e-9

def report(m):
    where = "inside" if m < L_ORIG else "beyond"
    print(f"\nposition m = {m} ({where} the trained window), angles in radians")
    print(f"{'pair':>4} {'rot/train':>10} {'naive':>10} {'PI':>10} {'YaRN':>10}   naive      PI      YaRN")
    for i in range(D // 2):
        t = theta(i)
        angles = (m * t, m * t / S, m * yarn_theta(t))
        marks = ["ok" if seen(t, a) else "UNSEEN" for a in angles]
        print(f"{i:>4} {rotations(t):>10.3f} {angles[0]:>10.2f} {angles[1]:>10.2f} "
              f"{angles[2]:>10.2f}   " + "  ".join(f"{mk:>6}" for mk in marks))

report(4096)        # inside the trained window: every method is in seen territory
report(L_NEW - 1)   # beyond it: naive pushes slow pairs into unseen angles

t0 = theta(0)
print("\nadjacent-token angle step of the fastest pair (local-ordering resolution):")
print(f"  original {t0:.3f}   PI {t0 / S:.3f} (crowded {S:.0f}x)   "
      f"YaRN {yarn_theta(t0):.3f} (preserved)")
print(f"YaRN attention-temperature factor: 0.1*ln(s) + 1 = {0.1 * math.log(S) + 1:.3f}")
```

跑一下它，两份位置报告用大约六十行演示了本章的核心论断。在训练窗口之内的位置 4096
上，每一对维度在每种方法下都标成 ok。在位置 65535 上，朴素外推把最慢的三对标成了
UNSEEN：它们在整个训练窗口内只转完 0.73、0.41 和 0.23 圈，所以训练只把一小段圆弧
展示给过它们（大约 4.6、2.6 和 1.5 弧度），而朴素外推给出的大约 37、21 和 12 弧度的
角度，落进了模型从未关注过的地带。这几对恰恰就是低频、承载全局位置的频段，那里
attention 分数变成任意值，正是把 `max_position_embeddings` 调大只会输出垃圾的原因。
PI 和 YaRN 都把每一对拉回了见过的范围，但结尾几行显示了代价上的差别：PI 把最快那
一对维度上相邻 token 的角度步长从 1.000 弧度压到 0.125 弧度，把相邻位置挤在了一起，
模糊了局部顺序；而 YaRN 的混合把它保持在 1.000，并在 s = 8 时加上 1.208 的温度因子。
把玩具常数换成真实的 head 维度，再加上在长文档上的继续训练，你就把这一章重新搭出来了。
