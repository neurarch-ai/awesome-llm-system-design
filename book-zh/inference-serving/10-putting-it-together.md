# 10. 把它们拼起来：完整的方案

第 1 到第 6 节把每个环节连同它的选项和取舍都讲了一遍，第 7 节展示了真实团队在哪里分道扬镳。它们都没给出的，是一个每个决策都已经拍板的完整系统。这个 capstone 做三件事：给出一套有主张的默认技术栈，让选择困难不至于卡住第一版实现；把本章的场景从头到尾走一遍，每个选择都定下来并算出成本；再展示约束一变，这些决策会怎么翻过来。最后收在一个最小可运行的批处理调度器上，单文件，零依赖。

## 默认技术栈：从这里开始，有理由再偏离

本章每个环节都有两到五个说得过去的选项，一个第一次动手的人可能花一整周比较各种引擎，却一个 token 都还没 serving 出去。跳过那一步。下面这套栈，是第一版生产实现的合理默认值；每一行都写清了什么时候该偏离，以及理由在哪一节。引擎每年都在换，但每个环节的接口（调度、prefill、缓存、decode、切分、量化、扩缩容、准入）不会变，所以按接口逐个环节做选择，把任何具体引擎都当成可替换的。

| 环节 | 默认 | 什么时候偏离 | 理由（章节） |
|---|---|---|---|
| serving 引擎 | vLLM 这一类：连续批处理加 PagedAttention，默认开启 | 永远不要退回静态批处理；GPU 空转这个问题是普遍存在的 | [3](03-batching.md) |
| 批处理策略 | 迭代级、按 token 预算打包、准入时为后续 token 留出 KV 余量 | 输出长度整齐且都很短：准入余量可以调小 | [3](03-batching.md) |
| prefill 调度 | 单一资源池上的分块 prefill | 集群规模下 prefill 和 decode 的 SLO 确实冲突，且有 NVLink 级别的互连：做分离部署 | [3](03-batching.md) |
| 并行 | 在单节点内用能装下模型的最小 TP 度数；要吞吐就整份复制 | 模型超出一个节点：跨节点上 PP；MoE 专家超出单 GPU：上 EP | [5](05-parallelism-and-quantization.md) |
| 权重精度 | H100 及更新硬件上用 FP8，前面挂一道质量评测闸门 | H100 之前的硬件：用 INT8；问题是装不下：用 4 位，质量风险更高 | [5](05-parallelism-and-quantization.md) |
| KV cache | 分页；当填满 HBM 的是并发而不是权重时，用 INT8 KV | KV 的质量评测不过关；MQA、共享这类 attention 结构改动要靠训练，不是 serving 时的一个开关 | [5](05-parallelism-and-quantization.md)、[2](02-the-throughput-problem.md) |
| 投机解码 | 在按负载测出接受率之前，关着 | 输出复述输入：用 n-gram 起草，不需要额外托管第二个模型 | [4](04-speculative-decoding.md) |
| 自动扩缩容 | 队列深度领先信号、热备缓冲、快照恢复式冷启动 | 流量确实平稳：固定规模集群，没有扩缩容参数要调 | [6](06-autoscaling-and-cost.md) |
| 准入 | SLO 闸门、按序列预留 KV、饱和时返回带 retry-after 的 429 | 永远不要偏离。过载时来者不拒，结果是所有人一起超时 | [6](06-autoscaling-and-cost.md) |

最后一行是新手最容易跳过、之后最后悔的一行：没有感知 SLO 的准入，第一次真实尖峰就会演变成重试风暴，所有在途请求一起错过目标。上线前把 429 这条路接好，是一个下午的工作量，第一个早高峰就回本了。

## 完整的方案

回到[第 1 节](01-clarifying-requirements.md)的那个场景：一个 70B 稠密模型，平均 500 QPS、尖峰 3 倍，p99 TTFT 低于 500 ms，p99 token 间延迟低于 50 ms，混合负载是 8k token 的 RAG prompt 加长输出的 agent 调用，硬件是 H100，有付费和免费两层，目标是把每百万输出 token 的成本压到最低。下面是整个系统，每个选择都已拍板，并附上它胜出的理由。

| 决策 | 选择 | 为什么是它 |
|---|---|---|
| 引擎 | 连续批处理加 PagedAttention | 每步让完成的序列退出，GPU 始终满载；分页把 KV 碎片压到百分之几以内 |
| 并行 | 单个 NVLink 节点内 TP=8；节点整份复制，挂在负载均衡器后面 | 140 GB 的 BF16 权重装不进一张 80 GB 的 H100；节点内 TP 还能压低每 token 延迟，而 50 ms 的 TPOT SLO 正需要这个 |
| 权重精度 | FP8，过评测闸门 | H100 原生支持；在受带宽限制的阶段，把每个 decode step 读的权重字节砍一半 |
| KV 精度 | INT8 KV cache | 8k token 的 RAG prompt 让 KV 而不是权重成为并发上限；KV 字节减半，余量翻倍 |
| prefill 调度 | 分块 prefill，单一资源池 | 否则 8k 的 prefill 会卡住每一条在途 decode，击穿 TPOT SLO；这个规模上还用不着分离部署 |
| 投机解码 | 上线时关闭；之后按各自负载再评估 | 混合流量在准入时无法预判；在打满的 batch 下，投机所需要的那份富余算力并不存在 |
| 准入 | SLO 闸门、按序列预留 KV、两条优先级队列 | 过载时必须保住付费层；免费层先被降载，返回 429 并给出重试提示 |
| 自动扩缩容 | 队列深度与等待时间领先信号、热备缓冲、快照恢复 | 冷启动以分钟计，尖峰以秒计；滞后信号总是在 SLO 已经被打破之后才反应 |
| 上线闸门 | 质量、成本、安全三个轴一起看 | 一个会让质量回退的 FP8 或 INT8 KV 收益，或者一个会 OOM 打死在途请求的配置，不能只凭一个成本数字就上线 |

**GPU 数量与显存规划。** 模型本身逼出了第一个决策：BF16 的 70B 参数约 140 GB 权重，而每张 H100 只有 80 GB HBM（[第 1 节](01-clarifying-requirements.md)），所以切分先于一切。一个 TP=8 的节点有 640 GB HBM；FP8 权重占其中约 70 GB，每张 GPU 大约 9 GB，把 HBM 的绝大部分留给了 KV cache。按本章的 KV 算术（[第 2 节](02-the-throughput-problem.md)），BF16 下一个 token 约 320 KB，INT8 KV 下约 160 KB，所以一条平均 4,000 上下文 token 的混合流量序列（示意值）大约占 640 MB；64 条活跃序列约用 41 GB，在这个节点里绰绰有余，还留得下 8k prompt 那些离群值的空间。

**吞吐与集群规模。** 按平均每个请求 300 个输出 token（示意值），500 QPS 就是全集群每秒 15 万个输出 token。单节点的 roofline 上界很宽松：每步约 70 GB 的 FP8 权重加 41 GB 的 KV，除以约 27 TB/s 的聚合 HBM 带宽，接近每步 4 ms，但真实引擎会比 roofline 慢好几倍，所以按实际做到 25 ms 来算（示意值）。这意味着 TPOT 是 25 ms，落在 50 ms 的 SLO 之内，节点吞吐是每 25 ms 一步、64 条序列，也就是每节点约 2,560 tokens/s、每张 GPU 约 320 tokens/s。稳态集群规模是 150,000 / 2,560，约 60 个节点（480 张 H100）。3 倍尖峰大约需要 180 个；这个差额是自动扩缩容要解决的问题，见下文。

**延迟。** TTFT 预算：最坏情况下 8k token 的 prefill 在该节点上接近 300 ms（示意值），而自动扩缩容是在 500 ms 预算里队列平均等待越过 200 ms 时触发（[第 6 节](06-autoscaling-and-cost.md)），所以只有队列信号足够早地触发，这两部分加起来才装得进 SLO；领先信号在这里是承重结构，不是锦上添花。分块 prefill 把那次 prefill 摊到若干次迭代里，用一点 TTFT 换其他每个用户平滑的 token 流。

**每百万输出 token 的成本。** 按[第 6 节](06-autoscaling-and-cost.md)，成本 =（GPU 每小时价格 x 10^6）/（tokens/s/GPU x 3600）。每 H100 小时 \$3、每张 GPU 320 tokens/s，算出来是每百万输出 token 约 \$2.60；换个算法，60 个节点、每节点每小时 \$24，一天约 \$35,000（示意值），对应约 130 亿个输出 token。跟那一节自己那个 80 tokens/s/GPU 的算例对比：每百万 \$10.40。差距全在分母上，这也正是连续批处理、FP8 和 INT8 KV 属于商业决策、而不是调参细节的原因。剩下的每一个杠杆（在低 batch 那一层上开投机、更好的打包方式），判断标准都是它能不能在不掉出质量闸门的前提下推动那个分母。

**第一个月会坏在哪。** 早期运维主要被三种故障模式支配，所以上线前先把它们的信号接好：抢占与换出计数（数字上涨说明准入正在超出 KV 预算，池子正滑向[第 3 节](03-batching.md)里那道抖动悬崖），队列平均等待时间对照 200 ms 的触发阈值（TTFT 的领先信号；如果它在 p99 TTFT 动了之后才告警，说明自动扩缩容接的是滞后指标），以及分层的 429 降载率（尖峰期降的是免费层，说明设计在正常工作；降到付费层，说明热备缓冲开小了，或者冷启动变慢了）。

## 同样的技术，换一组约束

实践中真正值得复盘的问题不是"哪个引擎最好"，而是"在我的约束下哪套栈是对的"。下面是同一个系统建三遍。只有中间那一列是上面那套方案；另外两列保持完全相同的环节接口，几乎换掉了每一个具体实现选择。

| | 单节点聊天产品 | 70B 双层 API（本章） | 通宵批量生成 |
|---|---|---|---|
| 模型 / 流量 | 8B 模型，约 5 QPS 的交互式聊天 | 70B 稠密，500 QPS，尖峰 3 倍 | 70B 稠密；每晚数百万请求，无交互性 |
| 延迟预算 | 亚秒级 TTFT 就体感不错；没有硬性 SLO | p99 TTFT < 500 ms，p99 TPOT < 50 ms | 没有；只看吞吐和成本 |
| 并行 | 单张 H100 装得下；两个副本只为可用性，不为扩展 | 节点内 TP=8，约 60 个复制节点 | 节点内 TP=8；预算或竞价市场允许多少节点就上多少 |
| 量化 | FP8 权重；这种并发下 BF16 KV 就够 | FP8 权重加 INT8 KV | 评测闸门过得了就用 4 位权重；字节更少也意味着竞价实例上冷启动更快 |
| 批处理 | 连续（永远如此）；prompt 都很短，分块 prefill 几乎无关紧要 | 连续、按 token 预算打包、分块 prefill | 连续，一直打包到算力 roofline；任何单条序列的延迟都无所谓 |
| 投机解码 | 输出复述输入时（代码、模板）用 n-gram 起草；先测接受率 | 关闭；打满的 batch 没有富余算力做验证 | 不用；batch 已经算力饱和，正是投机吃亏的那个区间 |
| 自动扩缩容 | 固定两个副本；扩缩容的配置量比集群本身还大 | 领先信号扩容、热备缓冲、分层降载 | 缩容到零加竞价容量；没人在等的时候，冷启动是免费的 |
| 什么算过度设计 | TP、分离部署、优先级队列、常热集群 | 分离部署（单一资源池已经能同时满足两个 SLO） | 热备缓冲、分块 prefill、优先级分层、流式输出 |

从中掉出来两条教训。第一，单节点那一列基本是做减法：模型装得进一张 GPU、流量又轻的时候，并行、准入分层和自动扩缩容全都消失了，整套方案离默认配置只差一个引擎开关。第二，批量那一列展示了延迟与吞吐的取舍彻底倒向一边：没有 SLO，一切保护延迟的手段（分块 prefill、热备缓冲、投机）都变成纯开销，正确做法是在最便宜的可中断硬件上，用 KV 预算撑得住的最大 batch。

## 每个约束各自决定什么

压缩版的决策指南。从需求里读出左列，右边两列告诉你在比较任何引擎之前，它先动的是哪个杠杆。

| 你的约束 | 它动的杠杆 | 经验法则 |
|---|---|---|
| 模型大小对比单张 GPU 的 HBM | 并行 | 装得下：复制来提吞吐。装不下：节点内上 TP。超出一个节点：跨节点上 PP，接受气泡 |
| TPOT 预算 | batch 上限与精度 | 一直打包到带宽饱和，但绝不越过 KV 预算；FP8 或 INT8 把每步字节减半，直接换来 TPOT |
| 混合负载下的 TTFT 预算 | prefill 调度 | 先上分块 prefill；只有当两个 SLO 在集群规模上确实冲突、且有高速互连时才做分离部署 |
| 并发与长上下文 | KV cache | 永远分页；当填满 HBM 的是 cache 而不是权重时，量化 KV |
| 输出复述输入 | 投机解码 | 用不需要第二个模型的 n-gram 草稿；按各自负载测接受率，接受率低时加速比会掉到 1 以下 |
| 尖峰速度对比冷启动 | 扩容信号与缓冲 | 按队列深度或等待时间扩容，绝不按延迟；热备缓冲按尖峰幅度乘以冷启动时长来定 |
| 优先级分层 | 准入 | 先把付费层的容量切片预留出来；用 429 加 retry-after 降掉免费层，绝不靠悄悄排队 |
| 每百万输出 token 的成本 | tokens/s/GPU，也就是分母 | 任何能把吞吐翻倍的杠杆都能把成本砍半；但它们没有一个能绕过质量评测闸门上线 |

## 最小可运行的调度器

对每一份 serving 引擎教程的评价都一样：读者配了一堆开关，却始终看不见调度器。所以这里把批处理这个决策放进一个文件，零安装，用一个离散事件模拟，把同一条带种子的请求流分别喂给静态批处理和连续批处理。每个生产组件都换成了接口相同的最小替身：一个 decode step 变成一个 tick，KV cache 的并发预算变成八个槽位，而[第 3 节](03-batching.md)里的那个反派、输出长度的方差，则是一个随机整数。要学的是这个形状；本章的每一节都是在升级这份文件里的某一块。

```python
"""Static vs continuous batching on one seeded request stream, no installs."""
import random

random.seed(7)
SLOTS = 8      # concurrent sequences the KV cache can hold
N = 200        # requests in the stream

# One shared workload: arrival step and output length per request.
arrive, out_len, t = [], [], 0.0
for _ in range(N):
    t += random.expovariate(0.08)               # bursty arrivals, ~0.08 req/step
    arrive.append(int(t))
    out_len.append(random.randint(8, 160))      # output-length variance is the villain

def simulate(policy):
    """One GPU; each tick is one decode step; every active slot emits one token."""
    remaining = out_len[:]
    finish, batch = {}, set()
    idle_slot_steps, busy_steps, step, next_req = 0, 0, 0, 0
    while len(finish) < N:
        # admission: continuous refills every step; static only when the batch retires
        if policy == "continuous" or not batch:
            while len(batch) < SLOTS and next_req < N and arrive[next_req] <= step:
                batch.add(next_req)
                next_req += 1
        active = [i for i in batch if remaining[i] > 0]
        if active:
            busy_steps += 1
            idle_slot_steps += SLOTS - len(active)   # held-but-finished slots waste here
            for i in active:
                remaining[i] -= 1
                if remaining[i] == 0:
                    finish[i] = step + 1
                    if policy == "continuous":
                        batch.discard(i)             # slot freed immediately
        if policy == "static" and batch and all(remaining[i] == 0 for i in batch):
            batch.clear()                            # whole batch retires together
        step += 1
    lat = sorted(finish[i] - arrive[i] for i in range(N))
    return {"tokens/step": sum(out_len) / max(finish.values()),
            "p50 latency": lat[N // 2],
            "p99 latency": lat[min(N - 1, int(N * 0.99))],
            "idle fraction": idle_slot_steps / (busy_steps * SLOTS)}

for policy in ("static", "continuous"):
    r = simulate(policy)
    print(f"{policy:>10}: " + "  ".join(f"{k}={v:.2f}" if isinstance(v, float)
                                        else f"{k}={v}" for k, v in r.items()))
```

跑一下，这条带种子的请求流给出的结果是：静态批处理每步 4.58 个 token，空闲槽位占比 0.43，p50 延迟 720 步、p99 1024；连续批处理每步 6.13 个 token，空闲占比 0.23，p50 87、p99 199。到达速率是特意选的，让需求正好落在两种容量之间：连续批处理跟得上，延迟接近纯服务时间；而静态批处理那些占着却闲着的槽位（占它容量的 43%，全耗在等每批里最长的那个成员上）把它推进了永久饱和，于是它的延迟主要是排队，并且随着请求流不断增长。这就是本章的核心论断，五十行不到：同一张 GPU、同一批请求，光是调度器就决定了系统是不是饱和的。每一个玩具部件都对应着一个生产部件：那个 tick 是一个 decode step，它的成本之所以是平的，是因为权重读取占主导、并且被摊到了所有槽位上；SLOTS 是 PagedAttention 管理的 KV cache 并发预算；准入那个循环就是迭代级调度；空闲占比就是[第 3 节](03-batching.md)里那 8 倍调度收益所挽回的 GPU 浪费。它有意省掉的是本章其余的部分：prefill 成本与分块、KV 增长与抢占、量化，以及当连续批处理这条线也跟不上时负责加节点的自动扩缩容。
