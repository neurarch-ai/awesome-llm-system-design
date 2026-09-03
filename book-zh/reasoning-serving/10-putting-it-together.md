# 10. 把它们拼起来：完整的方案

[第 1 节](01-clarifying-requirements.md)里的场景：一个混合负载（可验证的代码和数据
任务，加上开放式的解释），一个 p95 承诺，一个我们自己运维的集群，还有一个把一切都
变慢变贵了的推理模型。

## 默认技术栈

| 决策 | 定下来的选择 | 一句话理由 |
|---|---|---|
| 第一件要干的事 | 按请求记录结果（token、延迟、验证器判定、是否解决） | 没有"解决任务数"这个分母，任何策略都没法比较 |
| 预算 | 每请求一个硬性封顶，取在实测的准确率与预算曲线的拐点上 | 封顶就是一份延迟保证，而拐点就是再加 token 也买不到答案的地方 |
| 边界行为 | 撞顶时强制作答，或者明确拒答 | 静默截断会返回畸形输出，评分时算成一个错误答案 |
| 排队 | 按预算等级分开队列；思考队列内部用长度类别预测器排优先级 | 短请求不能等在长请求后面，而 FIFO 在这里对任何目标都不是最优 |
| 目标利用率 | 思考集群跑在离饱和很远的地方 | 排队延迟带着 $\rho/(1-\rho)$，而思考的服务时间方差很大 |
| 可验证的任务 | 级联：便宜的尝试、执行器检查、失败后升级、升级有配额封顶 | 在成本和解决率上都赢过"一律思考"；配额挡住升级风暴 |
| 不可验证的任务 | 拐点处的固定预算，强制作答，不采样 | 没有选择器，多采的样本用不上 |
| 抽取与格式化 | 不思考的小模型加受约束解码 | 深思买不到任何东西，只增加漂移 |
| 过载 | 准入控制，降级到便宜那条路 | 一个快的次优答案，好过一个用户已经走了才出来的正确答案 |
| 对外主指标 | 每个解决任务的成本，加上 p95 或 p99，成对报 | 单看任何一个，都能被另一个轻易地钻空子 |

## 同一份流量上的三种策略

下面的 capstone 模拟 4,000 个请求以固定速率进入一个 24 槽位的集群，输出长度按每条
路径各自的对数正态分布抽取，并给三种策略打分。三种策略的到达序列完全一致，所以这
是一次配对比较。

| 策略 | p50 | p99 | 平均 token | 每千次请求 \$ | 解决率 | 每千个解决任务 \$ |
|---|---|---|---|---|---|---|
| A：一律思考 | 55.7 s | 345.2 s | 4,412 | \$26.47 | 78.6% | \$33.68 |
| B：effort 路由（只有最难的 20% 走长路） | 6.7 s | 186.5 s | 1,157 | \$6.94 | 58.7% | \$11.82 |
| C：带验证器的级联 | 10.4 s | 250.3 s | 2,120 | \$12.72 | 85.3%（最高） | \$14.92 |

四点解读，其中第三点是能在面试里拿分的那一点。

**A 在除了一个维度以外的每个维度上都是最差的产品。** 它解决了 78.6%，而它是达到这
个数最慢也最贵的方式。它也正是团队"换成推理模型"时默认会上线的那个东西。

**B 的每个解决任务成本最低，质量最差。** 这就是"每请求成本降下来了"那种回答里的陷
阱：把难请求路由到便宜的路上，是靠多失败来降成本的，而只有解决率能把它揭出来。那
19 个百分点的质量值不值每千个解决任务多花的 \$3，是个产品问题，不是系统问题，但没
有结果埋点，这个问题你连问都问不出来。

**C 解决得比"一律思考"还多，成本只有一半。** 级联在便宜路漏掉的那些请求上拿到了两
次尝试，所以它的解决率（85.3%）超过了"一律思考"这个基线（78.6%），而每个解决任务
的成本从 \$33.68 降到了 \$14.92。这是本章里最有用的一个结果：**升级不只是一项成本
优化，它还是一个质量机制**，前提是验证器可信。

**尾巴永远不会完全消失。** 就算是最便宜的那个策略，p99 也接近它 p50 的 30 倍，因为
思考那条路是长尾的，而且和别人共用一个集群。预算封顶、队列分区和利用率目标，就是
为这件事存在的，这也是为什么 p99 要挨着成本数字并排走，而不是被塞在它下面。

## 同一套系统在三组约束下

**严格的交互式 SLO（p95 在几秒以内）。** 思考那条路根本不能出现在关键路径上。同步
地服务便宜那条路，把验证器串在里面跑，然后把升级做成异步的：先返回便宜的答案并附
上一个"核对中"的状态，等升级的结果落地再更新，或者发通知。工程重心从调度挪到了产
品界面上，而当承诺的 p95 低于平均生成时间时，这通常就是诚实的答案。

**成本主导的批量负载（没有用户在等）。** 延迟几乎是免费的，那就花它：大预算，并行
采样加执行器做选择，激进地攒批把槽位填满。这种场景下值得把利用率推高，因为排队的
尾巴不构成产品问题；也值得把钱投在验证器而不是模型上，因为交付质量是被选择这一步
封顶的。

**哪儿都没有验证器（开放式生成）。** 级联和 best-of-n 都用不了，杆子就只剩下这么几
根：一个测量过的固定预算配上强制作答的边界，用训好的分类器而不是接受测试来做
effort 路由，以及一个经过认证的评分标准裁判（如果你愿意去建的话，那就把这种情况又
变回上一种）。面试里诚实的说法是：给这套系统封顶的是缺失的验证器，不是缺失的
GPU。

## 最小的可运行实验

```python
"""Test-time compute planner on one page. Python 3, standard library only.
Illustrative numbers, not a benchmark."""

import heapq
import random
from statistics import mean

SEED = 5
N_REQ = 4000            # requests in the run
SERVERS = 24            # concurrent decode slots
RATE = 0.25             # arrivals per second (offered load, identical across policies)
TOK_S = 60.0            # decoded tokens per second per slot
PRICE = 6.0 / 1e6       # dollars per output token

SHORT = dict(mu=350, sigma=0.45, solve=0.55)
LONG = dict(mu=3200, sigma=0.80, solve=0.78)
VERIFY_TOKENS = 120     # a cheap checker run on the short answer
VERIFY_RECALL = 0.85    # fraction of wrong short answers the verifier catches


def draw_tokens(rng, path):
    """Output length is a random variable, and its tail is the whole problem."""
    return max(20.0, rng.lognormvariate(0, path['sigma']) * path['mu'])


def simulate(rng, per_request):
    """FIFO queue, SERVERS slots. per_request(rng) -> (tokens, solved)."""
    free = [0.0] * SERVERS
    heapq.heapify(free)
    t, lat, toks, solved = 0.0, [], [], 0
    for _ in range(N_REQ):
        t += rng.expovariate(RATE)
        n_tok, ok = per_request(rng)
        start = max(t, heapq.heappop(free))          # wait for the first free slot
        service = n_tok / TOK_S
        heapq.heappush(free, start + service)
        lat.append(start - t + service)              # queueing delay plus service
        toks.append(n_tok)
        solved += ok
    return lat, toks, solved


def pct(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def always_long(rng):
    return draw_tokens(rng, LONG), rng.random() < LONG['solve']


def routed(rng):
    path = LONG if rng.random() < 0.20 else SHORT    # a classifier picks the hard fifth
    return draw_tokens(rng, path), rng.random() < path['solve']


def cascade(rng):
    n = draw_tokens(rng, SHORT) + VERIFY_TOKENS      # cheap attempt plus the checker
    ok = rng.random() < SHORT['solve']
    if not ok and rng.random() < VERIFY_RECALL:      # verifier caught it -> escalate
        n += draw_tokens(rng, LONG)
        ok = rng.random() < LONG['solve']            # a second, independent attempt
    return n, ok


print(f"{'policy':>16} {'p50 s':>7} {'p99 s':>7} {'mean tok':>9} "
      f"{'$/1k req':>9} {'solved':>7} {'$/1k solved':>12}")
for name, fn in [("A always think", always_long), ("B effort routing", routed),
                 ("C cascade", cascade)]:
    rng = random.Random(SEED)                        # same arrivals for every policy
    lat, toks, solved = simulate(rng, fn)
    cost_1k = mean(toks) * PRICE * 1000
    rate = solved / N_REQ
    print(f"{name:>16} {pct(lat, 0.50):7.1f} {pct(lat, 0.99):7.1f} {mean(toks):9.0f} "
          f"{cost_1k:9.2f} {rate:7.1%} {cost_1k / rate:12.2f}")

rng = random.Random(SEED)
long_s = [draw_tokens(rng, LONG) / TOK_S for _ in range(20000)]
short_s = [draw_tokens(rng, SHORT) / TOK_S for _ in range(20000)]


def cv2(xs):
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs) / (m * m)


print()
print(f"short path : mean service {mean(short_s):5.1f}s  CV^2 {cv2(short_s):.2f}")
print(f"long path  : mean service {mean(long_s):5.1f}s  CV^2 {cv2(long_s):.2f}")
print("queue wait grows with E[S^2] (Pollaczek-Khinchine), so a heavier-tailed")
print("thinking distribution inflates p99 faster than it inflates the mean.")
```

输出：

```text
          policy   p50 s   p99 s  mean tok  $/1k req  solved  $/1k solved
  A always think    55.7   345.2      4412     26.47   78.6%        33.68
B effort routing     6.7   186.5      1157      6.94   58.7%        11.82
       C cascade    10.4   250.3      2120     12.72   85.3%        14.92

short path : mean service   6.4s  CV^2 0.23
long path  : mean service  73.5s  CV^2 0.89
queue wait grows with E[S^2] (Pollaczek-Khinchine), so a heavier-tailed
thinking distribution inflates p99 faster than it inflates the mean.
```

最后两行就是这一章的缩影。长路不只是平均慢 11 倍，它在相对意义上还多变四倍，而排队
延迟就是随这份方差增长的。[第 3 节](03-budgets-and-latency.md)里的每一项控制手段，
存在的意义都是去攻击这两个数字之一；[第 4 节](04-allocation-and-routing.md)里的每一
种策略，存在的意义都是让大多数请求根本别落进那个带着这两个数字的分布里。
