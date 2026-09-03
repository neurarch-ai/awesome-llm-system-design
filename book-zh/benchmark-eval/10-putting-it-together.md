# 10. 把它们拼起来：完整的方案

前面每一节给的都是选项。这一节要拍板：定下一套技术栈，把这次运行的成本算清楚，再在三组不同的约束下重新推导一遍，最后收在一份只用 Python 3 就能跑起来的参考实现上。

场景就是[第 1 节](01-clarifying-requirements.md)那个：从三个候选（两个 API 模型，一个我们自己托管的开放权重模型）里给一个产品家族挑基座模型，同时用同一套协议跟踪我们自己的微调版本，要求是 2 分的差距必须能判定。

## 默认技术栈

| 决策项 | 拍板的选择 | 一句话理由 |
|---|---|---|
| 组合 | 3 个还有提升空间的公开套件、1 个 live 套件、2 个 agent 套件、1 个私有内部题集 | 能力覆盖，加一次污染核查，加一个正式的决策依据 |
| 打分模式 | 全部用生成加答案匹配；对数似然只用于基座模型的训练遥测 | 和模型的实际用法一致，而且在 API 候选上跑得通 |
| 开放式评分 | 每题的 rubric 标准，模型评分器，对着专家标签认证过 | 整体打分复现不了；标准是可调试的 |
| 由 judge 打出来的数字 | 每个候选用 300 个专家标签做 PPI 校正 | 无论评分器质量如何都无偏，而标注成本是能估算的 |
| 解码策略 | 每个 benchmark 一套策略，对所有候选一致，另在一个子集上核对各家推荐的设置 | 可比性优先，同时确认这套策略没有惩罚到谁 |
| 测试时计算 | 有推理能力的候选跑两个算力档 | 质量是一条随开销变化的曲线，不是一个标量 |
| Seed | 300 题以下的套件跑 5 个，agent 套件跑 3 个，1,000 题以上的跑 1 个 | 方差主导的地方补方差，题量主导的地方补题量 |
| 统计 | 逐题配对比较、差值上的 bootstrap 区间、在预注册的比较上做 BH 校正 | 我们关心的量是差距，不是分数 |
| 污染证据 | 在每个候选 cutoff 之后的 live 套件窗口，加上内部题集 | 谁都能跑，不像去污染那种宣称 |
| 产物 | 一张报告卡，含分数、区间、成本、协议哈希，以及每组比较的结论 | 一个不带协议的数字是不可复现的 |

## 这次运行，把成本算清楚

每个候选的题量和样本预算：

| 套件 | 题量 | Seed | 运行次数 | 类型 |
|---|---|---|---|---|
| GPQA Diamond | 198 | 5 | 990 | 短答案推理 |
| MMLU-Pro 子集 | 1,000 | 1 | 1,000 | 广度知识 |
| LiveCodeBench（cutoff 之后的窗口） | 300 | 3 | 900 | 代码，兼作污染核查 |
| 内部题集 | 800 | 3 | 2,400 | rubric 评分的开放式题 |
| SWE-bench Verified | 500 | 1 | 500 | agent，仓库环境 |
| tau2-bench | 300 | 3 | 900 | agent，工具加用户模拟 |

这里的题量是本次运行评估的切片，不是每个套件已发布的完整规模：MMLU-Pro 那行是采样出来的子集，LiveCodeBench 那行是在所有候选 cutoff 之后的一个发布日期窗口，两行 agent 套件是我们选来跑的任务集。GPQA Diamond 和 SWE-bench Verified 是整套跑的。

成本主要由两件大家会低估的事情主导：agent 的 episode（每一轮工具调用都要重发上下文），以及算力档曲线（它把推理套件乘了一遍）。按示意价格每百万输入 token \$3、每百万输出 token \$15 计算：

```text
non-agentic   5,290 runs x (1.5k in + 3k out)   ~=  7.9M in + 15.9M out  ~= $262
SWE-bench       500 episodes x (120k in + 15k out) ~= 60M in +  7.5M out ~= $292
tau2-bench      900 episodes x (40k in + 6k out)   ~= 36M in +  5.4M out ~= $189
                                                       per candidate     ~= $743
3 candidates                                                             ~= $2,230
high-effort curve on 2 candidates (reasoning suites, 3x output)          ~=   $400
rubric grading 7,200 grader calls on a cheap model                       ~=    $30
                                                       total compute     ~= $2,700
```

另一条预算线是人工标注：每个候选 300 条专家判断用于 PPI 校正，总共大约 900 条，每条几分钟，也就是一到两个专家周，这才是决定迭代节奏的真正约束。墙钟时间上，30 路并行时非 agent 套件几个小时，agent 套件要占掉大半天，所以完整网格是一个过夜任务，冒烟子集是一杯咖啡的工夫。

两千美元的算力对两个专家周的标注，这个比例正是 PPI 重要的原因：便宜的资源是模型调用，稀缺的是判断力，所以设计上就该花前者去把后者拉长。

## 报告卡

真正落到文档里的东西，逐 benchmark 列出，聚合指数单独放：

| 候选 | 内部题集（PPI） | 95% CI | GPQA-D | LiveCodeBench | tau2 pass^1 | tau2 pass^3 | 每题输出 token | 每千题 \$ |
|---|---|---|---|---|---|---|---|---|
| A（中算力档） | 0.678 | +/- 0.031 | 0.61 | 0.44 | 0.68 | 0.31 | 3,100 | \$48 |
| A（高算力档） | 0.702 | +/- 0.030 | 0.69 | 0.52 | 0.71 | 0.36 | 9,400 | \$142 |
| B（默认） | 0.671 | +/- 0.032 | 0.64 | 0.49 | 0.66 | 0.29 | 1,400 | \$23 |
| C（开放权重，自托管） | 0.639 | +/- 0.033 | 0.55 | 0.41 | 0.58 | 0.20 | 1,900 | \$9 |

数字是示意的，重点在形态。当成分数读，A 的高算力档赢了。当成报告卡读：A 高算力档的成本是 B 的六倍，换来的是内部题集上一个配对区间极可能包含零的差距；C 距离 B 的质量差距只有三分之一档，成本却低得多，一旦把服务量算进价格，它可能反而胜出；而每个候选在 agent 套件上的 pass^3 大约只有 pass^1 的一半，这个数字是产品团队在承诺任何自主流程之前必须先看到的。

配套的结论是：**B 作为默认**，**A 高算力档作为难题的升级档**（也就是[成本优化一章](../cost-optimization/)里的级联，现在有数字撑腰了），**量涨上来之后重新评估 C**，以及**把 A 对 B 在内部题集上的差距报成"区分不出来"**，并附上能定论所需的题量。

## 同一系统在三组约束下的形态

**自己训模型的前沿实验室。**瓶颈从成本挪到了节奏：你需要的是每个 checkpoint 一个信号，不是每季度一个。拆成一个便宜的代理套件（几百题，用对数似然打分，每几千步跑一次）和只在里程碑上跑的完整网格。去污染在这里是真活儿，不是一句宣称，因为语料是你自己的：对 benchmark 题目及其改写做 n-gram 和近重复去除，同时说清一点，从外部教师模型蒸馏可能重新引入你看不见的泄漏。再加一个带日志查询预算的密封切片，因为在几百个 checkpoint 的规模上，主要风险是选择性泄漏，不是训练集重叠。确定性服务在这里比在任何其他场景都更要紧，因为一条会随 batch 组成而摆动的训练遥测曲线根本没法读。

**用 API 模型的种子期创业公司。**大套件降到一个 seed，只在题量小的地方保留多 seed，公开组合砍到两个套件加一个 live 套件。省下来的全部投到内部题集上，因为它是唯一能决定你产品的东西，再投 200 个人工标签做 PPI 校正。自托管的对比先跳过，等量上来再说。融资材料里诚实的姿态，是在数字旁边放一个协议哈希和一个区间，这很便宜，而且立刻就把你和一张排行榜截图区分开了。

**受监管或安全攸关的领域。**rubric 由领域专家撰写，像代码一样做版本管理；评分器按每个 rubric 版本对着专家标签认证，并按计划重新认证。安全套件作为一道独立的阻塞门禁运行，有自己的阈值，并配一个良性题集，好让过度拒答被测量而不是被奖励。每次运行都留档以备审计，并保留完整来源（协议哈希、容器 digest、原始输出），自动门禁之后还有一道人工签字，因为自动化流水线减少的是需要专家过目的量，不是把它消掉。

## 最小的可运行实验

一个文件，只用标准库。它回答一份 benchmark 报告必须回答的四个问题，并复现[第 6 节](06-statistics-and-leaderboards.md)最核心的那条教训：500 题的 benchmark 上 2 分的差距，算不上一个结果。

```python
"""Benchmark-eval statistics on one page. Python 3, standard library only."""

import random
from math import comb, sqrt

random.seed(10)

# Per-item correctness for two candidates on the SAME 500 items (paired by index).
# Items carry a shared difficulty, so the two models agree on most of them; only a
# thin band plus per-run noise separates them. That correlation is exactly what the
# paired analysis exploits and the unpaired one throws away.
N, SKILL_A, SKILL_B, NOISE = 500, 0.72, 0.70, 0.05
difficulty = [random.random() for _ in range(N)]          # shared across candidates


def run(skill):
    return [int((d < skill) != (random.random() < NOISE)) for d in difficulty]


a, b = run(SKILL_A), run(SKILL_B)


def wilson(k, n, z=1.96):
    """95% interval for a proportion. Correct at small n, unlike the normal approx."""
    p = k / n
    d = 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return center - half, center + half


def mcnemar(x, y):
    """Paired comparison: only items where the two disagree carry information."""
    b_wins = sum(1 for u, v in zip(x, y) if u == 1 and v == 0)   # x right, y wrong
    c_wins = sum(1 for u, v in zip(x, y) if u == 0 and v == 1)   # y right, x wrong
    delta = (b_wins - c_wins) / len(x)
    se = sqrt(b_wins + c_wins) / len(x)
    z = (b_wins - c_wins) / sqrt(b_wins + c_wins) if b_wins + c_wins else 0.0
    return delta, se, z, b_wins, c_wins


def paired_bootstrap(x, y, reps=5000):
    """Resample items (not scores) to get an interval on the difference."""
    n = len(x)
    diffs = []
    for _ in range(reps):
        idx = [random.randrange(n) for _ in range(n)]
        diffs.append(sum(x[i] - y[i] for i in idx) / n)
    diffs.sort()
    return diffs[int(0.025 * reps)], diffs[int(0.975 * reps)]


def items_needed(discordance, delta, z_sum=2.80):
    """Items required to detect `delta` at 5% significance and 80% power, paired."""
    return discordance * (z_sum / delta) ** 2


def pass_hat_k(n, c, k):
    """Reliability: P(all k independent attempts succeed), unbiased from n trials."""
    return 0.0 if c < k else comb(c, k) / comb(n, k)


def ppi(judge_all, judge_labeled, human_labeled):
    """Judge mean, rectified by the judge's measured bias on the human-labeled slice."""
    bias = sum(h - j for h, j in zip(human_labeled, judge_labeled)) / len(human_labeled)
    return sum(judge_all) / len(judge_all) + bias


lo_a, hi_a = wilson(sum(a), N)
lo_b, hi_b = wilson(sum(b), N)
delta, se, z, bw, cw = mcnemar(a, b)
blo, bhi = paired_bootstrap(a, b)

print(f"A = {sum(a)/N:.3f}  95% CI [{lo_a:.3f}, {hi_a:.3f}]   (unpaired)")
print(f"B = {sum(b)/N:.3f}  95% CI [{lo_b:.3f}, {hi_b:.3f}]   (unpaired)")
print(f"paired: delta={delta:+.3f}  se={se:.3f}  mcnemar z={z:.2f}  "
      f"(A-only {bw}, B-only {cw})")
print(f"paired bootstrap 95% CI on the difference: [{blo:+.3f}, {bhi:+.3f}]")
print("verdict:", "distinguishable" if blo > 0 or bhi < 0 else "NOT distinguishable")
d = (bw + cw) / N
print(f"discordance={d:.3f} -> items needed for a 2-point call: "
      f"{items_needed(d, 0.02):,.0f}")

print()
for p in (0.9, 0.68):
    n_trials, c_ok = 100, int(round(100 * p))
    print(f"per-attempt {p:.0%}: pass^3={pass_hat_k(n_trials, c_ok, 3):.2f}  "
          f"pass^8={pass_hat_k(n_trials, c_ok, 8):.2f}")

print()
judged = [1 if random.random() < 0.74 else 0 for _ in range(5000)]   # judge, all items
sub = list(range(300))                                               # human-labeled slice
judge_sub = [judged[i] for i in sub]
# humans are stricter than the judge on ~8% of the items it passed
human_sub = [0 if (j == 1 and random.random() < 0.08) else j for j in judge_sub]
print(f"judge-only estimate : {sum(judged)/len(judged):.3f}   (precise, biased)")
print(f"humans-only (n=300) : {sum(human_sub)/len(human_sub):.3f}   (unbiased, wide)")
print(f"PPI-corrected       : {ppi(judged, judge_sub, human_sub):.3f}   "
      f"(unbiased, narrow)")
```

输出：

```text
A = 0.712  95% CI [0.671, 0.750]   (unpaired)
B = 0.690  95% CI [0.648, 0.729]   (unpaired)
paired: delta=+0.022  se=0.015  mcnemar z=1.46  (A-only 34, B-only 23)
paired bootstrap 95% CI on the difference: [-0.006, +0.052]
verdict: NOT distinguishable
discordance=0.114 -> items needed for a 2-point call: 2,234

per-attempt 90%: pass^3=0.73  pass^8=0.42
per-attempt 68%: pass^3=0.31  pass^8=0.04

judge-only estimate : 0.742   (precise, biased)
humans-only (n=300) : 0.683   (unbiased, wide)
PPI-corrected       : 0.678   (unbiased, narrow)
```

这二十行输出里有三件事值得带走。**A 领先 2.2 分，而这不算一个结果**：配对 bootstrap 区间跨过了零，要判定这么大的差距需要大约 2,200 题。**配对拿走了可得精度里的大部分**：不配对的区间各自约 4 分宽，配对之后标准误是 1.5 分。**judge 的数字精确而错误**：0.742 对上校正后的 0.678，6 分的偏差被 300 个诚实标签既揭示出来又消掉了。这里每一条都是你能在面试里说出口的句子，而每一条也都属于那种"只有当你能指出数字从哪儿来时才可信"的说法。
