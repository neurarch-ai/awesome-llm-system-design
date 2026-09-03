# 10. 把它们拼起来：完整的方案

第 1 到 6 节讲了每个杠杆各自的数学和失效模式，第 7 节展示了真实团队先动了哪个杠杆。它们都没给出的，是一套每个决定都拍板了的完整系统。这一节做三件事：给一套有主张的默认技术栈，让选择困难症不至于卡住第一版；把本章的场景从头到尾走一遍，每个选择都定下来并算清成本；再展示同样这些决定在约束变了之后会怎么翻转。最后收在一个最小可运行的路由加级联上，一个文件，什么都不用装。

## 默认技术栈：从这里开始，有理由再偏离

本章的每个杠杆都有两到四种说得过去的变体，一个第一次搭的人可能花掉一周比较各种路由器，却一分钱都还没省下。别这么干。下面这套栈是第一版生产系统的合理默认值；每一行都写明了什么时候该偏离，以及哪一节解释了原因。工具年年在换，但每个杠杆的接口（测量、把闸、缓存、裁剪、路由、right-size）不变，所以按接口逐个杠杆去挑，任何具体的库都当成可替换的。

| 杠杆 | 默认做法 | 什么时候偏离 | 为什么（对应小节） |
|---|---|---|---|
| 质量测量 | 带分桶分数的标注评测集，在调任何东西之前先建好 | 永远不偏离。下面每一个阈值都要靠它 | [1](01-clarifying-requirements.md) |
| 网关 | 在所有供应商前面架一个代理（LiteLLM 那类）：预算、日志、fallback | 代理本身永远别省；单团队的玩具项目可以先不做预算 | [6](06-serving-and-scaling.md) |
| 语义缓存 | Embedding 加阈值，tau 在标注好的"应该命中 / 不该命中"样本对上调，按租户分范围 | 流量几乎全是各不相同的自由文本：先量自然命中率，可能过不了盈亏平衡点 | [4](04-caching-and-compression.md) |
| 前缀缓存 | 用供应商的 prompt caching，稳定内容放前面，易变内容放最后 | Prompt 之间没有共享的长固定头部 | [4](04-caching-and-compression.md) |
| 上下文裁剪 | 给检索到的 chunk 重排，留 top-3 | 检索本来就已经很收敛，或者任务需要每一个 chunk | [4](04-caching-and-compression.md) |
| Prompt 压缩 | 上线时先不做；只有裁剪之后输入仍然占大头，才加 LLMLingua | 又长又啰嗦的文本在裁剪之后还在，而且输入 token 仍然主导账单 | [4](04-caching-and-compression.md) |
| 路由器 | 稳定模式先用正则启发式层，再接一个小的微调分类器 | 你手上有偏好数据：RouteLLM 那种偏好路由器能跨模型对泛化 | [3](03-routing-and-cascades.md) |
| 级联 | SLO 很紧时别放在主路径上；用在延迟有余量且存在可验证检查的地方 | 任务可验证（代码、SQL、引用），而且 SLO 容得下两次调用 | [3](03-routing-and-cascades.md) |
| Right-sizing | 专用 embedding 模型、小 cross-encoder 重排器、1B 以内的分类器 | 某个子任务 QPS 高又稳定：可以考虑蒸馏 | [5](05-right-sizing.md) |
| 离线流量 | 只要没有用户在等，就走供应商的 batch API | 只有 QPS 过了盈亏平衡点 Q* 才自建 | [5](05-right-sizing.md)、[6](06-serving-and-scaling.md) |

第一行是新手最容易跳过、之后最后悔的：没有分桶质量数字，这套栈里的每个阈值都只是感觉，而一块绿的成本看板能无限期地藏住难长尾的回退。本章的场景里评测集是现成的；如果你手上没有，那第一个要交付的就是它。

## 完整的方案

回到[第 1 节](01-clarifying-requirements.md)的场景：一个交互式、意图混杂的 RAG 聊天产品，响应要在两秒以内，有一份约 500 条标注样本的 LLM-as-a-judge 评测集，账单被输入 token 主导（因为每条 prompt 都带 20 个检索到的 chunk），简单查询按条数占绝大多数，同时有一条带来收入、不能退步的难长尾，而且只有 API 计费这一种选项。下面是整套系统，每个选择都拍了板，并附上它胜出的理由。

| 决定 | 选择 | 为什么它胜出 |
|---|---|---|
| 测量 | 把 500 条裁判集按路由分桶切开，难长尾过采样 | 难长尾不能退步；只有分桶分数看得见它 |
| 网关 | 所有调用走同一个代理：按团队切预算、fallback、成本日志 | 没有它，开支要到账单出来才看得见，每个杠杆都只是建议 |
| 上下文裁剪 | Cross-encoder 给 20 个检索 chunk 重排，留 top-3 | 输入 token 就是明写的成本大头；靠后的 17 个 chunk 是噪声 |
| 压缩 | 上线时不用 LLMLingua | 裁剪已经把输入这块收益拿走了；prompt 现在很短，小语言模型那一趟纯属额外开销 |
| 语义缓存 | Embedding 加阈值，tau 在标注样本对上调，按租户切 key，易变答案设 TTL | 意图混杂的聊天会用各种改写重复问同样的 FAQ；盈亏平衡命中率只有大约 2% |
| 前缀缓存 | 系统 prompt 和工具 schema 放前面，每请求各异的内容放最后 | 每条请求都共享同一个稳定头部；顺序对了命中率才活得下来 |
| 路由器 | 打招呼和模板查找走正则层，再接一个微调过的小分类器 | 简单类别稳定而且有明显模式；剩下的可以用裁判流水线来打标 |
| 级联 | 不放在主路径上 | 两秒的 SLO 挤不出"便宜调用再加打分器"的余量；离线场景再议 |
| 模型档位 | 简单分桶用微调过的小模型，难长尾用前沿模型 | 那两个点的容忍度只适用于简单流量；长尾留在前沿模型上 |
| Right-sizing | 缓存用专用 embedding 模型，裁剪用小 cross-encoder，分类器 1B 以内 | 这里任何一处用前沿模型调用，都会把自己省的钱吃掉 |
| 量化 / batching | 不在考虑范围内 | 只有 API 计费；这些杠杆长在供应商那边，不在我们这边 |

**基线开支。** 今天的 prompt 大约是 300 token 的系统 prompt，加上 20 个 chunk 乘 400 token，再加查询和历史，接近 8,500 个输入 token，答案约 250 token。按示意性的前沿模型价格（每百万输入和输出 token 分别 $2.50 和 $10.00）算，大约是 $0.021 + $0.0025，就算 $0.024 一次查询。按示意性的每天 150,000 次查询算，账单大约是每天 $3,600，每月大概 $107,000。其中输入 token 占 89%，这印证了[第 2 节](02-frame-the-system.md)的 profiling 结论：先裁剪，再路由。

**裁剪。** 20 个 chunk 里只留 top-3，把 prompt 压到约 1,700 个输入 token（系统 prompt、3 个 400 token 的 chunk、查询和历史），输入减少 80%，和[第 4 节](04-caching-and-compression.md)里"20 去 17"的算术是一致的。在任何路由之前，单次查询的前沿模型成本就从 $0.024 降到约 $0.0068，砍掉 3.5 倍，而且重排器本来就在给 chunk 打分，留下来的文本一个字没动。这也是压缩被跳过的原因：浪费在 chunk 之间，不在 chunk 内部。

**缓存。** 盈亏平衡点在 2% 附近（[第 4 节](04-caching-and-compression.md)），FAQ 很多的聊天流量上一个示意性的 15% 语义命中率，净收益是很宽裕的。期望成本变成 0.85 乘（未命中成本）再加上 embedding 那点零头。

**路由。** 示意性地假设：正则层加分类器把 60% 的缓存未命中送给小模型。按示意性的小模型价格（每百万 $0.25 和 $1.25）算，一次小模型查询大约 $0.0007，对比前沿模型的 $0.0068。于是一次未命中的平均成本是 0.6 乘 $0.0007 加 0.4 乘 $0.0068，约 $0.0031，代入[第 2 节](02-frame-the-system.md)的期望成本公式得到

$$\mathbb{E}[C] \approx 0.15 \cdot c_{\text{hit}} + 0.85 \cdot \$0.0031 \approx \$0.0027$$

**最终的单次查询成本。** 约 $0.0027，对比 $0.024 的基线：砍掉 9 倍，大约 89%，也就是每天约 $400 对比 $3,600。这一路都是示意性数字，但和[第 7 节](07-how-teams-do-it-in-production.md)里那些生产系统报出来的区间是对得上的（Anyscale 70%，RouteLLM 约 85%）。顺序很重要：裁剪把两个档位的 token 账单都压小了，缓存直接消掉了一部分调用，路由再把剩下的分开，这正是[第 2 节](02-frame-the-system.md)里从左到右的杠杆顺序。

**上线第一个月会坏在哪儿。** 早期运维主要被三种失效模式主导，所以它们的信号要在上线前就接好：难长尾切片上的分桶质量（路由器在新流量上漂了，就会把新变难的查询甩给小模型，而总体看板一片绿，正是[第 8 节](08-interview-qa.md)里的那个陷阱）、选定 tau 下的缓存命中答案质量（错邻居答案的比例在涨，说明阈值太松了，而裸命中率是看不出来的），以及网关 fallback 率（持续 fallback 说明主供应商出了问题，流量可能正落在一个质量和安全行为都不一样的模型上）。

## 同样的技术，换一组约束

实际工作里真正要评审的问题不是"哪个路由器最好"，而是"在我的约束下哪个路由器最好"。下面是同一套栈搭了三遍。只有中间那一列是上面那套方案，另外两列保持完全一样的杠杆接口，几乎每个实现选择都换掉了。

| | 低流量的内部助手 | 意图混杂的 RAG 聊天（本章） | 每晚跑的批量分类 |
|---|---|---|---|
| 流量 / 成本大头 | 每天约 1k 次查询；绝对金额上总开支很小 | 每天 15 万次查询（示意）；输入 token 占大头 | 每晚数百万条；请求量占大头，没人在等 |
| 延迟预算 | 几秒钟没问题 | 两秒以内 | 没有；只看吞吐和成本 |
| 缓存 | 只用精确缓存，或者干脆不用；改写量太少，tau 调不动 | 语义缓存和前缀缓存串联，tau 在标注样本对上调 | 对重复输入做精确缓存；调用之前先把这批数据去重 |
| 裁剪 / 压缩 | 调一下检索的默认值，别的不做 | 重排到 top-3；压缩留作后手 | 激进：每条用固定的短模板，不留自由文本的水分 |
| 路由 | 没有：所有请求一个中档模型 | 正则加分类器路由器，简单分桶给小模型 | 带可验证或打分检查的级联；延迟有余量，等于白送 |
| 模型档位 | 单个中档 API 模型 | 微调过的小模型加前沿模型接长尾 | 稳定任务上用蒸馏出来的学生模型；前沿模型只当教师和审计员 |
| Batch / 自建 | 不做 | 不做；按场景设定只有 API | 走 batch API，价格约一半；QPS 过了盈亏平衡点 Q* 再自建 |
| 评测 | 小的 golden set，换模型时重跑 | 裁判集按路由分桶切开，难长尾过采样 | 对每晚的输出抽样过裁判；再做一次学生对教师的漂移检查 |
| 什么算过度设计 | 分类器路由器、语义缓存、蒸馏：每一个的维护成本都超过它省下的钱 | 把级联放在主路径上，上线就用 LLMLingua | 低延迟服务栈、流式输出、交互路由逻辑 |

从中掉出来两条教训。第一，左边那一列基本上都是删减：每天 1k 次查询的量级下，每个杠杆的维护成本都超过它的节省，正确的成本优化就是选一个合理的模型，加上网关的日志，好让这个前提不再成立时你能察觉。公式仍然有用，只不过是用作不去搭的理由：[第 3 节](03-routing-and-cascades.md)的路由器节省公式，在路由器的维护开销超过它抓住的价差时就变成负的了。第二，右边那一列展示了延迟和成本互换了"谁是硬约束"的位置：没有 SLO 之后，级联（[第 3 节](03-routing-and-cascades.md)）从例外变成了默认，蒸馏（[第 5 节](05-right-sizing.md)）在这个量级上很快就能回本，而 batch API 那个一半的价格，就是 batching 这个机制在账单上的体现。

## 每一条约束各自决定什么

压缩版的决策指南。从需求里读出左边那一列，右边几列会告诉你它先动哪个杠杆，你甚至还不需要开始比较工具。

| 你的约束 | 它动的杠杆 | 经验法则 |
|---|---|---|
| 主要的成本大头 | 先动哪个杠杆 | 输入偏重：先裁剪再压缩。输出偏重：换更小的模型，答案写短。请求量：缓存加路由 |
| 延迟 SLO | 路由器还是级联 | SLO 紧：盲路由器，亚毫秒级分类器。有余量：用能给真实答案打分的级联 |
| 查询重复度 | 缓存层 | 盈亏平衡命中率在 2% 附近；改写多的流量撑得起语义缓存，各不相同的自由文本可能过不了这条线 |
| 共享的 prompt 结构 | 前缀缓存 | 每条请求都有很长的固定头部：稳定内容放前面，易变的放最后，否则命中率会悄无声息地死掉 |
| 难长尾的容忍度 | 路由的目标函数 | 零容忍：把分桶质量下限写成硬约束，评测集里对难长尾过采样 |
| 质量测量的成熟度 | 所有东西 | 没有评测集，任何地方的阈值都站不住脚；先建评测集，再动第一个杠杆 |
| API 还是自建 | 量化、连续批处理 | 在盈亏平衡 QPS Q* 以下 API 更划算，量化不是你的杠杆；过了这个点，FP8 那一档的收益才成立 |
| 离线流量的占比 | Batch API | 只要没有用户在等，就该离开同步端点；市面上的价格大约是一半 |
| 能养几个模型的运维预算 | Right-sizing 做到多深 | 每多一个模型就多一个可能悄悄退步的质量面；能监控到哪，就 right-size 到哪 |

## 最小可运行的路由加级联

评审每个路由框架时的感受都一样：读者拼好了网关、路由器和两家供应商，还是看不见那个取舍。所以这里把本章的核心机制放进一个文件，零安装。每个生产组件都被换成了接口相同的最小替身：两个模型档位变成成本差 10 倍的偏心抛硬币，可靠性打分器变成一次带噪声的置信度采样，流量变成一条固定随机种子、7:3 的简单 / 困难数据流。扫一遍升级阈值 tau，就能把[第 3 节](03-routing-and-cascades.md)那条级联的成本-质量工作点，和"永远用便宜的"、"永远用强的"两条基线一起打印出来。

```python
"""Router/cascade cost-vs-quality simulator, runnable with no installs."""
import random

random.seed(7)

C_CHEAP, C_STRONG = 1.0, 10.0          # relative per-call cost of each tier
N = 20000                              # queries in the simulated stream

def make_query():
    """70% easy, 30% hard; production: real mixed-intent traffic."""
    return "easy" if random.random() < 0.70 else "hard"

def cheap_model(kind):
    """Cheap tier: strong on easy, weak on hard. Returns (correct, confidence).
    Confidence overlaps across correct/wrong: a deliberately imperfect scorer,
    standing in for logprobs or a trained reliability model."""
    correct = random.random() < (0.95 if kind == "easy" else 0.40)
    conf = random.uniform(0.5, 1.0) if correct else random.uniform(0.0, 0.7)
    return correct, conf

def strong_model(kind):
    """Frontier tier: expensive, near-uniformly good."""
    return random.random() < (0.97 if kind == "easy" else 0.92)

def run(policy, tau=None):
    """policy: 'cheap' | 'strong' | 'cascade'. Returns (avg cost, accuracy, escalation rate)."""
    cost = right = escalated = 0
    for _ in range(N):
        kind = make_query()
        if policy == "cheap":
            ok, _ = cheap_model(kind)
            cost += C_CHEAP
        elif policy == "strong":
            ok = strong_model(kind)
            cost += C_STRONG
        else:                                   # cascade: cheap first, escalate on low confidence
            ok, conf = cheap_model(kind)
            cost += C_CHEAP
            if conf < tau:                      # scorer says "not confident": pay for the strong call
                ok = strong_model(kind)
                cost += C_STRONG
                escalated += 1
        right += ok
    return cost / N, right / N, escalated / N

print(f"{'policy':>22} {'avg cost':>9} {'accuracy':>9} {'escalate':>9}")
c, a, _ = run("cheap")
print(f"{'always-cheap':>22} {c:9.2f} {a:9.3f} {'-':>9}")
c_strong, a_strong, _ = run("strong")
print(f"{'always-strong':>22} {c_strong:9.2f} {a_strong:9.3f} {'-':>9}")
for tau in (0.2, 0.4, 0.5, 0.6, 0.8):
    c, a, e = run("cascade", tau)
    marker = "  <- knee" if abs(tau - 0.6) < 1e-9 else ""
    print(f"{f'cascade tau={tau:.1f}':>22} {c:9.2f} {a:9.3f} {e:9.2f}{marker}")

print("\nthe knee: near always-strong accuracy at a fraction of its cost;")
print("sweeping tau traces the whole cost-quality frontier from cheap to strong.")
```

跑一下，这一遍扫描就会把本章的核心论断打印成一张表。"永远用便宜的"落在成本 1.00、准确率 0.787（它在难长尾上翻车）；"永远用强的"是成本 10.00、准确率 0.956。级联在 tau = 0.6 时达到 0.952，离"永远用强的"不到半个点，成本却只有 4.42，不到一半的价钱，升级掉了 34% 的流量。而在 tau = 0.8 时级联做到成本 7.85、准确率 0.967，在两条轴上都严格压过"永远用强的"，因为打分器补上了盲策略没有的信息：自信而且正确的便宜答案被留下了，强模型的预算集中到了真正需要它的查询上。每一块玩具零件都对应一个生产组件：成本常数是每次调用的 API 价格，`make_query` 是你真实的混杂意图流量，那次故意让正确和错误重叠的置信度采样，就是[第 3 节](03-routing-and-cascades.md)里那条没校准好的信号梯队（最下面是对数概率，往上是训练出来的可靠性打分器，最上面是可验证的检查），tau 扫描是那次必须随流量漂移重跑的留出集校准，升级率那一列则是[第 6 节](06-serving-and-scaling.md)让你设报警的监控指标。把抛硬币换成两个真模型，把置信度采样换成一个训练好的打分器，再把整个循环放到网关后面，你就把这一章重新搭出来了。
