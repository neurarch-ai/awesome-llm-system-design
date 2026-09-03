# 10. 把它们拼起来：完整的方案

第 1 到第 6 节把生命周期的每个阶段连同它的选项和取舍讲了一遍；第 7 节展示了
真实团队在哪里分岔。但没有一节展示过一个把每个决策都定死的完整方案。这一节
做三件事：给出一条有主见的默认路径，让选择困难症不至于卡住第一版计划；把本章
那个场景从头到尾走一遍，每个选择都拍板并算清成本；再展示同样这些决策在约束
改变时会怎么翻转。最后收在一个最小的可运行生命周期规划器上：一个文件，不用
装任何东西。

## 默认技术栈：从这里开始，有理由再偏离

本章每个阶段都有三到六个说得过去的选项，第一次做的人很可能在"预训练还是适配"
上争论一个月，一个 token 都还没训。跳过那一步。下面这条路径是第一版生产方案
的合理默认值；每一行都写清楚什么时候该偏离，以及哪一节解释了原因。基座模型和
训练框架每年都在换，但每个阶段的接口（整理、定规模、训练、对齐、服务、评估）
不会变，所以按阶段做决定，把任何具体的 checkpoint 或库都当成可替换的。

| 阶段 | 默认做法 | 什么时候偏离 | 为什么（章节） |
|---|---|---|---|
| 自建还是买 | 在开放基座（Llama 3 / Qwen3 / OLMo 这一档）上做中期训练 | 既没有数据驻留要求也没有领域缺口：直接用 API，完全跳过训练；所有开放基座都不具备的能力：从零预训练 | [1](01-clarifying-requirements.md)、[3](03-pretraining-and-scaling.md) |
| 数据预算 | 任何训练之前先去重、过滤、去污染；做领域中期训练时混入通用回放数据 | 去污染这一项永远不能省；没有它的 benchmark 数字毫无意义 | [2](02-the-five-stages.md) |
| 模型定规模 | 先按服务目标定规模；只有训练算力才是那份卡死的成本时，才按每参数 20 个 token 来 | 你要服务数十亿 token：把一个更小的模型过度训练，越过它自己的最优点 | [3](03-pretraining-and-scaling.md)、[5](05-inference-economics.md) |
| 架构 | 一个 GQA + RoPE 的 decoder 基座 | QPS 极高：MQA（Character.AI）；追求规模下每 FLOP 的质量：MoE | [3](03-pretraining-and-scaling.md) |
| 后训练 | 在整理好的数据对上做 SFT，再在离线偏好对上做 DPO | 奖励可检查（数学、代码）：GRPO；需要可复用的奖励模型：PPO；标注成本占大头：RLAIF | [4](04-post-training.md) |
| 服务 | vLLM（分页 KV、连续批处理、前缀缓存），INT8 权重 | INT8 还塞不下：在评估把关下上 INT4；架构本身就太大：蒸馏 | [5](05-inference-economics.md)、[6](06-serving-and-scaling.md) |
| 知识 | 事实靠 RAG，行为靠微调；两者叠加 | 事实是静态的、也永远不需要引用出处（少见） | [6](06-serving-and-scaling.md) |
| 评估 | 每个阶段用对应的指标，每一次提升都附上去污染的说明 | 永远不能省。提对策之前先说清指标 | [2](02-the-five-stages.md)、[8](08-interview-qa.md) |

最后一行悄悄决定了其他所有行：没有分阶段的指标（领域 benchmark 加上通用评估
套件、偏好胜率、tokens/秒和每百万 token 成本），每一个训练和压缩决策都只是
凭感觉，你根本判断不出一次改动到底有没有用。

## 完整方案

回到[第 1 节](01-clarifying-requirements.md)那个场景：敏感法律文书不能出内网，
允许用开放基座，500 亿 token 的清洗过的判例法和合同，每季度更新一次，500 并发
用户下 p95 首 token 延迟低于两秒，质量标准是引用准确度和面向执业律师的指令
遵循。下面是整条生命周期，每个选择都已拍板，并附上它胜出的理由。

| 决策 | 选择 | 为什么它胜出 |
|---|---|---|
| 自建还是买 | 在开放基座上做中期训练加后训练；不做预训练 | 数据驻留要求逼着你拥有权重；从零预训练要花几亿美元，只为重新学会基座已经会的东西 |
| 基座模型 | 自带 GQA 和 RoPE 的 8B 档开放基座 | 服务约束卡住了规模上限（[第 6 节](06-serving-and-scaling.md)把交互式法律助手定在 7-13B）；GQA 必须在训练时烤进去，所以它是选基座的标准，不是事后的热修 |
| 中期训练数据 | 500 亿法律 token 加上一份通用回放混合数据 | 没有回放数据的继续预训练会导致灾难性遗忘；通用评估套件就是那个退化警报 |
| 后训练 | 在法律指令对上做 SFT（引用格式、拒答风格），再做 DPO | Llama 3 那套配方：稳定、离线、不需要奖励模型和 PPO 循环；几万条高质量数据对胜过纯堆量 |
| 事实还是行为 | 在判例法索引上做 RAG；微调只管风格和格式 | 法律会变，而且答案必须给出处；权重给不出引用，而且在两次季度刷新之间就过期了 |
| 服务 | INT8，vLLM 配分页 KV 和连续批处理，系统 prompt 上开前缀缓存 | decode 受显存带宽限制；INT8 在评估把关下几乎无损地把每 token 读的字节数减半 |
| 流水线 | 特征 / 训练 / 推理保持分离，由索引和模型注册表连起来 | 语料刷新、重训、服务部署这三件事不能是同一个事件 |
| 评估 | 领域 benchmark + 完整通用套件 + 引用准确度，全部做过去污染 | 遗忘和评估泄漏正是这套方案的两种无声失败方式 |

**中期训练算力。** 用[第 3 节](03-pretraining-and-scaling.md)的估算
$C \approx 6ND$：80 亿参数，500 亿领域 token 再加大约 20% 的通用回放混合数据
（示意值）约合 620 亿 token，所以
$C \approx 6 \times 8\times10^9 \times 6.2\times10^{10} \approx 3\times10^{21}$
FLOPs。而基座自己的预训练（15T token，Llama 3 8B）花掉大约
$7.2\times10^{23}$ FLOPs，差不多是 240 倍。这个比值就是[第 1
节](01-clarifying-requirements.md)的全部论点：通用语言能力是白送的，团队只为
领域上的那点差量付钱。

**服务侧的算术。** INT8 下权重是 8 GB，按[第 5 节](05-inference-economics.md)
的 decode 上界，在 2 TB/s 的 HBM 上 batch 为 1 时每个 token 是
$t_{\text{decode}} \approx 8\times10^9 \times 1 / 2\times10^{12} = 4$ 毫秒，
远在交互预算之内。真正决定要几张卡的是 KV cache：按 Llama-3-8B 的几何结构
（32 层、8 个 KV 头、head dim 128、bf16 缓存），每个 token 要
$2 \times 32 \times 8 \times 128 \times 2 \approx 131$ KB，所以一个 8K token
的会话大约占 1 GB。一张 80 GB 的 GPU 扣掉 8 GB 权重，能装下大约 70 个满长度的
并发会话，于是 500 并发用户在 PagedAttention 把碎片浪费收回来之前，大概需要
8 张 GPU（示意值）。显存问题从来不是模型大小，是缓存。

**为什么这里不做过度训练，以及什么时候会翻转。** 这个团队是在一个已经预训练
好的基座上再训 620 亿 token，所以 Chinchilla 那套算术不是他们的决策，是 Meta
的。但[第 3 节](03-pretraining-and-scaling.md)的逻辑仍然约束着基座的选择：一个
按每参数大约 1800 个 token 训出来的 8B 基座之所以存在，正是因为有人要大规模
服务，愿意多付训练 FLOPs 换一个永久便宜的 decode。这一节末尾的规划器会给这个
交叉点算出一个数。

**第一个月里会出什么问题。** 早期运维里有三种失败方式最常见，所以上线前就要把
它们的信号接好：通用评估退化（领域 benchmark 在涨，中期训练后的模型却在
MMLU 那一档套件上漂移；对策是回放混合数据，加上通用套件上的硬性门槛），引用
失败（编造或过期的判例引用，说明事实是从权重里漏出来的，而不是来自 RAG 索引；
跟踪引用核验通过率，以及索引相对季度语料投放的新鲜度），以及峰值时的 KV cache
压力（并发会话拖长时 p95 首 token 延迟飙升或者直接 OOM；要盯每张 GPU 的缓存
占用，而不是平均 QPS，因为律师会在上下文里塞很长的文书）。

## 同样的技术，换一组约束

实践中真正要紧的面试问题，不是抽象地讨论"我们该不该预训练"，而是"在我这组约束
下，同样这五个阶段会长成什么样"。下面是同一条生命周期规划三遍。只有中间那一列
是上面那套方案；另外两列沿用完全一样的阶段词汇，却几乎把每个决策都翻了过来。

| | API 优先的产品团队 | 法律领域助手（本章） | 20k QPS 的消费级对话 |
|---|---|---|---|
| 什么缺口值得拥有权重 | 没有：没有驻留约束，领域也被前沿 API 覆盖 | 数据驻留要求，加上领域词汇和引用格式 | 单位经济：巨大量下的每 token 毛利 |
| 从哪个阶段切入 | 只有阶段 5：在 API 上做 prompt 工程、RAG、function calling | 在开放基座上做中期训练 + 后训练 | 完整生命周期：预训练一个自己端到端拥有的小模型 |
| 定规模 | 不是他们的决策；按任务挑一档模型 | 8B 档，由服务目标封顶 | 小，而且有意过度训练：推理主导全生命周期成本 |
| 训练数据 | 没有；预算花在评估集和 prompt 上 | 500 亿领域 token + 回放混合数据 | 数万亿 token，远远超过每参数 20 个 |
| 后训练 | 没有；API 厂商做过了 | 为引用格式和拒答做 SFT + DPO | 大量偏好调优；人设活在权重里 |
| 服务 | 厂商的事；他们只盯延迟和每次调用成本 | INT8、GQA、vLLM，约 8 张 GPU（示意值） | MQA，权重和 KV cache 都上 INT8，跨轮前缀缓存（Character.AI） |
| 知识新鲜度 | RAG；索引是他们唯一拥有的产物 | 在判例法索引上做 RAG，每季度刷新 | 基本不需要：人设是稳定的，权重存得住 |
| 什么算是过度工程 | 任何训练；自己买 GPU | 从零预训练；前沿规模的基座；没有把关就上 INT4 | 每次查询都做 RAG；用 70B 模型；全精度服务 |

由此得出两点。第一，左边这一列是工业界最常见的正确答案，却是面试里最少出现的
答案：当没有任何约束逼着你拥有权重时，生命周期里训练那半边整个消失，团队的
杠杆转移到评估、prompt 和检索上。把这话说出来是信号，不是示弱。第二，右边这
一列展示了定规模的目标如何随着量级增长而翻转：法律团队按延迟目标定规模，消费
级团队按每 token 成本目标定规模，这就是为什么后者要过度训练一个小模型，还要
用 MQA 把 KV cache 剥薄。公式相同，最优点相反。

## 每个约束决定什么

压缩过的决策指南。从需求里读出左边那一列，右边几列告诉你它在你比较任何
checkpoint 之前，先动了哪根杠杆。

| 你的约束 | 它动的杠杆 | 经验法则 |
|---|---|---|
| 数据驻留、规模上来后的成本、或者一处能力缺口 | 自建还是买 | 没有缺口：用 API。缺口是开放基座覆盖得了的：在它上面做中期训练。所有基座都没有的能力：这时才预训练 |
| 全生命周期的推理量 | 定规模的目标 | 量小：算力最优（每参数约 20 个 token）或者直接用 API。要生成数十亿 token：把一个更小的模型过度训练，越过它的最优点 |
| 领域数据的形态 | 中期训练还是 SFT | 数十亿原始领域 token：中期训练（知识）。数千条整理过的数据对：SFT（行为）。诊断故障时也按这个分 |
| 知识的变化速度 | RAG 还是权重 | 事实变化快过你重训的节奏，权重在结构上就是过期的；改成检索加引用 |
| 首 token 延迟和并发 | 模型规模、精度、KV 变体 | 按服务目标定规模：先上 INT8，INT4 只在评估把关下用；GQA/MQA 是选基座时的决策，不是服务侧的热修 |
| 标注预算 | 偏好方法 | 有离线偏好对：DPO。奖励可检查：GRPO。需要可复用的奖励模型：PPO。标注成本占大头：对着一份宪法做 RLAIF |
| 审计和质量标准 | 评估门槛 | 相信任何提升之前先去污染；每一步压缩都要评估把关；把攻击成功率和误拒率作为发布门槛来跟踪 |

## 最小的可运行生命周期规划器

对每一篇 scaling law 博客的评价都一样：读者对着曲线点头，然后还是答不上来
"那我们的模型到底该多大"。所以这里把本章的核心算术做成一个文件，零安装。它
接受一个 FLOP 预算，用 $C = 6ND$ 和 $D = 20N$ 打印出 Chinchilla 最优的
$(N, D)$ 划分，然后在不同的全生命周期推理量下，把这个最优模型和一个越过自身
最优点做过度训练的半尺寸模型做总拥有成本对比，并打印出过度训练开始胜出的
交叉点。每一个常数都是旋钮；答案的形状才是要点。

```python
"""Lifecycle planner: Chinchilla split, then the train-vs-serve TCO crossover.

Every rule is the chapter's own: C = 6*N*D (section 3), D = 20*N at the
compute-optimal point (section 3), and decode cost = bytes moved / bandwidth
(section 5). Dollar and utilization figures are illustrative; the crossover
logic is the lesson.
"""

GPU_HOURLY = 2.00      # $/GPU-hour, illustrative rental price
TRAIN_FLOPS = 4.0e14   # sustained training FLOP/s per GPU (~40% MFU), illustrative
HBM_BW = 2.0e12        # HBM bandwidth in bytes/s (A100 class, section 5)
BATCH = 32             # continuous batching amortizes the weight read (section 5)
P_BYTES = 1            # INT8 weights (section 5)


def chinchilla_split(C):
    """C = 6*N*D with D = 20*N gives C = 120*N^2, so N = sqrt(C/120)."""
    N = (C / 120) ** 0.5
    return N, 20 * N


def train_cost(N, D):
    """6*N*D FLOPs at the sustained per-GPU rate, priced per GPU-hour."""
    return 6 * N * D / TRAIN_FLOPS / 3600 * GPU_HOURLY


def serve_cost_per_token(N):
    """Decode is memory-bandwidth bound: every token reads all N*P_BYTES
    weight bytes; continuous batching splits that read across BATCH streams."""
    seconds = N * P_BYTES / HBM_BW / BATCH
    return seconds / 3600 * GPU_HOURLY


def plan(C, small_frac=0.5, overtrain=10):
    """Compare the compute-optimal model against a smaller model overtrained
    past its own optimum (quality treated as comparable for planning; the
    chapter's Llama 3 8B anchor sits at ~90x its optimal token count)."""
    N_opt, D_opt = chinchilla_split(C)
    N_small = small_frac * N_opt
    D_small = overtrain * 20 * N_small
    opt = (N_opt, D_opt, train_cost(N_opt, D_opt), serve_cost_per_token(N_opt))
    small = (N_small, D_small, train_cost(N_small, D_small),
             serve_cost_per_token(N_small))
    return opt, small


def report(C, volumes):
    (N1, D1, t1, s1), (N2, D2, t2, s2) = plan(C)
    print(f"FLOP budget C = {C:.1e}")
    print(f"  Chinchilla-optimal : N = {N1/1e9:5.1f}B params, "
          f"D = {D1/1e9:6.0f}B tokens ({D1/N1:.0f} tok/param)")
    print(f"  Overtrained small  : N = {N2/1e9:5.1f}B params, "
          f"D = {D2/1e9:6.0f}B tokens ({D2/N2:.0f} tok/param)")
    print(f"  Train cost         : optimal ${t1:,.0f}  vs  small ${t2:,.0f}")
    print(f"  Serve cost / 1M tok: optimal {s1*1e6:.3f}  vs  small {s2*1e6:.3f} (USD)")
    print(f"  Decode ms/token b=1: optimal {N1*P_BYTES/HBM_BW*1e3:.1f}  "
          f"vs  small {N2*P_BYTES/HBM_BW*1e3:.1f}")
    print()
    print(f"  {'lifetime tokens':>16} | {'optimal TCO':>12} | "
          f"{'small TCO':>12} | winner")
    for v in volumes:
        a, b = t1 + s1 * v, t2 + s2 * v
        w = "optimal" if a <= b else "small (overtrained)"
        print(f"  {v:16.0e} | {a:12,.0f} | {b:12,.0f} | {w}")
    cross = (t2 - t1) / (s1 - s2)
    print(f"\n  Crossover: overtraining wins past {cross:.1e} lifetime "
          f"generated tokens\n  ({cross * s1:,.0f} dollars of serving at the "
          f"optimal model's rate).")


if __name__ == "__main__":
    report(C=5.9e21, volumes=[1e9, 1e10, 1e11, 1e12, 1e13])
```

跑一下，本章那两条定规模的论证就变成了具体数字。在第 3 节那个
$5.9\times10^{21}$ FLOPs 的预算下，Chinchilla 的划分是 7B 模型配 140B token，
正好是那个例题；半尺寸的替代方案是 3.5B 参数配 701B token（每参数 200 个），
训练成本大约贵 2.5 倍（$20,486 对 $8,194，价格为示意值），但把 decode 延迟
（batch 为 1 时每 token 1.8 毫秒对 3.5 毫秒）和每百万 token 的服务成本
（$0.030 对 $0.061）双双砍半。接着那张 TCO 表就展示了翻转：在 1B 和 10B 的
全生命周期 token 下，算力最优的模型胜出，而到 1e12 token 时，过度训练那个模型
的总成本是 $50,920，对面是 $69,062，打印出来的交叉点在
$4\times10^{11}$ 全生命周期生成 token 附近。这就是[第 3
节](03-pretraining-and-scaling.md)那个 Llama 3 论证被还原成算术的样子：说清楚
你在优化哪份成本，把全生命周期的 token 数算出来，定规模这个决策自己就成立了。
把这些常数换成你自己的 GPU 价格、你自己的带宽和你自己的流量预测，你就为自己的
系统重建了本章的经济学。
