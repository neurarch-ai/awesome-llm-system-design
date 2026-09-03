# 10. 把它们拼起来：完整的方案

第 1 到 6 节把每个环节的选项和取舍都讲了一遍，第 7 节展示了真实团队在哪儿分道扬镳。
但它们谁都没给出一套所有决定都已经拍板的完整系统。这一节的 capstone 做三件事：
给一套有明确主张的默认技术栈，让选择困难症不至于卡住第一次搭建；
把本章的场景从头到尾走一遍，每个选择都定下来并算清容量；
再展示同样这些决定在约束变了以后怎么翻过来。最后收在一个最小的、能跑的 token 流上，一个文件，不用装任何东西。

## 默认技术栈：从这里开始，有理由再偏离

本章每个环节都有两到五个说得过去的选项，第一次做的人可能一个 token 都还没流出去，
就花了一周在比较传输方案。跳过这一步。下面这套栈是第一次做生产系统时的合理默认；
每一行都写明了什么时候该偏离，以及哪一节解释了为什么。框架每年都在换，
但每个环节的接口（传输、session 状态、上下文策略、背压、取消、恢复、过载）不换，
所以按接口逐环节地选，把任何具体工具都当成可替换的。

| 环节 | 默认 | 什么时候偏离 | 为什么（第几节） |
|---|---|---|---|
| 传输 | 基于普通 HTTP 的 SSE，一个 token 一个事件，每个事件带 `id:` | 客户端需要在流中途发信号（打断、多路复用）：WebSocket。语音音频：基于 UDP 的 WebRTC | [2](02-the-streaming-model.md) |
| Session 状态 | 服务端存储：Redis 走热路径，Postgres 做持久化后端 | 简单机器人，没有长会话也不需要可恢复：对话记录放客户端 | [3](03-session-and-memory.md) |
| 上下文策略 | 到阈值就把最老的轮次摘要掉 | 早期上下文确实不相关：最近 k 轮的滑动窗口 | [3](03-session-and-memory.md) |
| Prefill 成本 | 前缀缓存加上按 session id 的粘性路由，当成尽力而为 | 只有一个副本：粘性是免费且自动的 | [3](03-session-and-memory.md)、[6](06-serving-and-scaling.md) |
| 背压 | 每条流一个有上限的缓冲；文本就阻塞 decode 循环 | 消费者慢到没救：直接中止这条流。音频：丢帧 | [4](04-backpressure-and-concurrency.md) |
| 取消 | 把取消和掉线一路传到推理引擎；在 SSE 通道上打心跳 | 永远不偏离。无主的槽位在任何规模下都是容量泄漏 | [4](04-backpressure-and-concurrency.md) |
| 恢复 | 只在完成时写；重试带幂等 key；靠 `Last-Event-ID` 加一个短的环形缓冲续传 | 回复要跑几十秒：在句子边界打检查点 | [5](05-reliability.md) |
| 过载 | 越过利用率阈值就用 HTTP 429 加 `Retry-After` 丢负载；降级到更小的模型 | 永远不要无声排队。无上限的队列会放大重试 | [4](04-backpressure-and-concurrency.md)、[6](06-serving-and-scaling.md) |

"取消"那一行是新手最容易跳过、后来最后悔的一行：一条没人在读的流看着像什么都没发生，
但它在整个 decode 期间钉着一个推理槽位。上线前就把掉线即取消接好，这是容量管理，不是"讲究卫生"。

## 完整的方案

回到[第 1 节](01-clarifying-requirements.md)那个场景：一个状态在服务端的多轮文本对话产品，
峰值几万条并发流，p95 首 token 延迟低于一秒，session 要能扛住关浏览器和换设备，
还要有一个真的管用的停止按钮。对话通常五到十五轮，重度用户能到三四十轮。
下面是整个系统，每个选择都定下来了，并写明它凭什么胜出。

| 决定 | 选择 | 凭什么胜出 |
|---|---|---|
| 传输 | SSE，一个 token 一个事件，每个事件带 `id:` 字段 | token 投递是单向的；普通 HTTP 能穿过所有代理；按 id 续传是标准自带的 |
| Session 状态 | 放服务端：Redis 按 session id 存，后面接 Postgres | 多设备可恢复这条就排除了历史放客户端；持久副本扛得住 Redis 驱逐 |
| 路由 | 对 session id 做一致性哈希，从网关到推理副本 | 前缀缓存只有在后续轮次找得到自己的热 KV cache 时才划算；扩容时哈希只重新平衡一部分，不是全部 |
| 上下文策略 | 稳定的开头交给前缀缓存；到阈值就把最老的轮次摘要掉 | 框住每轮的 prefill，也让四十轮的会话在不撞上下文上限的前提下活下来 |
| 背压 | 每条流一个有上限的缓冲，阻塞 decode 循环，慢到没救的消费者直接中止 | 丢 token 会让文本变成乱码；一个小缓冲能很快暴露已死的消费者 |
| 取消 | 停止按钮和掉线都会把 abort 传到推理引擎；SSE 心跳 | 释放出来的槽位才是容量单位；心跳能抓到 socket 从不上报的半开 TCP |
| 恢复 | 只在完成时写，客户端请求 id 用于去重，一个短的 token 环形缓冲用于续传 | 回复很短，所以丢一次生成重试的代价很低；环形缓冲让网络抖动变得不可见 |
| 过载 | 越过阈值就用 429 加 `Retry-After` 丢负载；尖峰持续时降级到更小的模型 | 看得见的降级好过无声的卡死；参数减半，decode 速度大致翻倍 |

**每个副本能扛多少并发流。** 容量单位是并发的 decode 数，不是 QPS（见[第 6 节](06-serving-and-scaling.md)）。
举例说明：一个 GPU 副本在连续批处理下能稳定跑到大约 3000 tok/s 的总吞吐，
服务的每条流各自 decode 速度约 30 tok/s，那它大概能扛 100 条并发流。
峰值 2 万条并发流，就需要 200 个副本这个量级。
降级模型是撬动这个数字最便宜的杠杆：参数减半，decode 速度大致翻倍，
所以同一批机器在尖峰时能扛大约两倍的流，代价是看得见的质量下滑，而不是一块空白屏幕。

**每个 session 的上下文增长。** 按本章举例用的平均每轮 200 个 token 算（见[第 3 节](03-session-and-memory.md)），
一个十五轮的会话在第十五轮要重读大约 3000 个 token 的对话记录，四十轮的重度会话大约 8000 个。
没有前缀缓存的话，四十轮累计的 prefill 是个三角数，大致是 200 x 40 x 41 / 2，
一段对话就是大约 16.4 万个 token。有了前缀缓存加粘性路由，每轮的边际成本就压缩到只剩新消息那部分，
再加上到阈值就摘要，把对话记录的长度重置一次，长会话的尾巴就不再继续变长。
这一对决定基本上就是你那张基础设施账单的大头。

**TTFT 预算。** 一秒的 p95 目标可以拆成排队加 prefill 加传输（见[第 2 节](02-the-streaming-model.md)）。
举例说明的热路径：网络和网关约 50ms，Redis 读对话记录是个位数毫秒，
在前缀缓存是热的情况下对新消息做 prefill 约 100 到 200ms，饱和点以下准入等待接近于零；
加起来几百毫秒，很宽裕。这份预算还得能吸收冷路径：一个重新哈希过或者刚重连的 session
要对它那 3000 个 token 左右的记录做完整 prefill，再多几百毫秒，仍然在一秒之内。
过了饱和点这些就都不重要了：那时 TTFT 由队列长度决定，剩下的杠杆只有丢负载、降级和加副本。

**第一个月会坏在哪儿。** 早期运维里有三种故障模式占大头，所以上线前就要把它们的信号接好：
前缀缓存命中率（掉一下、一两轮内恢复的是重新平衡；一直低着就是路由 bug，
每一轮都把同一个 session 送到不同的副本，见[第 8 节](08-interview-qa.md)），
无主的流（槽位利用率高而用户 QPS 低，说明掉线没有把 abort 传下去，见[第 6 节](06-serving-and-scaling.md)），
以及重试放大（一次 TTFT 尖峰触发客户端重试，尖峰自己把自己加深；
盯着队列深度和丢负载阈值的关系，并确认 429 带了 `Retry-After`、客户端做了带抖动的退避）。

## 同样的技术在不同约束下

实践中真正值得复盘的问题不是"SSE 还是 WebSocket"，而是"在我的约束下，SSE 还是 WebSocket"。
下面是同一个服务层搭了三遍。只有"消费级对话"这一列是上面那套方案；
另外两列保持完全相同的环节接口，但几乎换掉了每一个实现选择。

| | 内部工具机器人 | 消费级对话（本章） | 语音助手 |
|---|---|---|---|
| 流量 | 几百条并发流 | 几万条并发流 | 几千通并发通话，每通都是一段连续的音频会话 |
| 传输 | SSE；别的都不值那个钱 | 带 `id:` 事件和续传缓冲的 SSE | 基于 UDP 的 WebRTC；TCP 队头阻塞会把音频卡住（[7](07-how-teams-do-it-in-production.md)） |
| Session 状态 | 一个 Redis，不做粘性；短记录上的冷 prefill 很便宜 | Redis + Postgres，一致性哈希的粘性路由 | 每通通话的状态放在媒体服务器上；轮次检测本身就是 session 的一部分 |
| 上下文策略 | 最近 k 轮的滑动窗口 | 前缀缓存 + 到阈值就摘要 | 短的滚动上下文；真正卡住的是 L_STT + L_turn + L_LLM + L_TTS 这个和 |
| 背压 / 过载 | FIFO 队列，遇到慢消费者就阻塞，不分档 | 有上限的缓冲，429 丢负载，优先级队列，模型降级 | 丢掉迟到的音频帧；流畅比完整重要；抢跑式轮次结束判定把死时间重叠掉 |
| 可靠性 | 只在完成时写，加一个幂等 key | 靠 `Last-Event-ID` 从环形缓冲续传，按请求 id 去重 | 不做重放：过期的音频毫无价值；重连就是这一轮重来 |
| 什么算过度设计 | 一致性哈希、优先级分档、模型降级、续传缓冲 | 对短回复做句子边界检查点 | 按 id 重放 token、阻塞式背压、任何 TCP 传输 |

从中能得出两个结论。第一，内部机器人那一列基本上全是删减：
几百条流的规模下，一两个副本就能吃下峰值，短记录上的冷 prefill 只要毫秒，
所有亲和性和分档机制都是死重量；唯一活下来的是掉线即取消，
因为无主的槽位在任何规模下都在漏容量。
第二，语音那一列展示了传输和背压策略是一起翻转的：
一旦帧必须准时到达而不是按序到达，UDP 就取代了 TCP，丢弃就取代了阻塞，
延迟这场仗也从 prefill 转移到了[第 7 节](07-how-teams-do-it-in-production.md)里那个四项相加的流水线。

## 每个约束各自决定了什么

压缩版的决策指南。从需求里读出左边那一列，右边几列告诉你它撬动的是哪个杠杆，
这一步在你开始比较任何工具之前就该做完。

| 你的约束 | 它撬动的杠杆 | 经验法则 |
|---|---|---|
| TTFT 目标 | 前缀缓存 + 粘性路由，prefill 的大小 | p95 要压到 1s 以内：多轮场景必须走热缓存路径；冷路径也要留预算，每次重新平衡都会触发它 |
| 峰值并发流数 | 副本数、批处理、降级档位 | 每副本的流数 ≈ 总 tok/s 除以单流 tok/s；尖峰时把模型减半，天花板大致翻倍 |
| 会话长度分布 | 上下文策略 | 短记忆产品：滑动窗口。用户会回来接着聊的长会话：到阈值就摘要，永远不要无界增长 |
| 流中途的客户端信令 | 传输 | 只有服务端到客户端：SSE。打断、实时中断、多路复用：WebSocket。音频：基于 UDP 的 WebRTC |
| 客户端网络不稳 | 事件 id + 重放缓冲 | 每个 SSE 事件都打上 `id:`，每条流留一个短的环形缓冲；没有它的重连是一个无声的正确性 bug |
| 多设备可恢复 | 状态放在哪儿 | session id 绑到账号，对话记录放持久存储；缓存只改变延迟，从不改变正确性 |
| 用户分档 | 队列策略 | 所有人同一个 SLA：FIFO。有付费档：在准入处做优先级队列，不要在 decode 中途抢占 |
| 过载的形态 | 丢负载阈值 + 降级 | 短促尖峰：用 `Retry-After` 丢负载，客户端带抖动退避。持续过载：先换降级模型，再加副本 |
| 慢消费者或者死消费者 | 缓冲上限 + 中止策略 | 文本：小缓冲，阻塞，没救的直接中止。音频：丢帧。永远不要不设上限地缓冲 |

## 最小的可运行流式输出

本章的核心论断说起来容易，也容易让人不信：在一个快的解码器和一个慢的客户端之间放一个无上限的缓冲，
就是一处内存泄漏，而设上限的修法是拿这块内存去换槽位时间。
所以下面把整个取舍放进一个文件，不用装任何东西。每一个生产组件都换成了接口相同的最小替身：
推理引擎变成一个每 20 ms 吐一个 token 的时钟，慢客户端变成一个每 100 ms 左右（带抖动）取走一个的时钟，
网关的单流缓冲变成一个计数器。一次运行用无上限的队列；另一次把它限制在 32 个 token，
满了就阻塞 decode 循环，也就是[第 4 节](04-backpressure-and-concurrency.md)里那套文本对话的策略。

```python
"""One token stream through a gateway buffer: unbounded queue vs bounded backpressure."""
import random

DECODE_MS = 20        # decoder emits a token every 20 ms (50 tok/s)
CONSUME_MS = 100      # a slow client drains a token every ~100 ms (10 tok/s)
TOTAL_TOKENS = 300    # one assistant reply

def stream(bound=None, seed=7):
    """Simulate one generation end to end.
    bound=None: unbounded gateway queue (production: an unlimited socket buffer).
    bound=B:    the decode loop blocks while the per-stream buffer holds B tokens."""
    rng = random.Random(seed)
    buffered = produced = delivered = peak = 0
    now = 0.0                       # simulated clock, ms
    next_produce = DECODE_MS
    next_consume = CONSUME_MS * rng.uniform(0.5, 1.5)
    slot_freed = None               # when the last token decodes, the slot returns
    while delivered < TOTAL_TOKENS:
        can_produce = produced < TOTAL_TOKENS and (bound is None or buffered < bound)
        if can_produce and next_produce <= next_consume:
            now = next_produce
            produced += 1
            buffered += 1
            peak = max(peak, buffered)
            next_produce = now + DECODE_MS
            if produced == TOTAL_TOKENS:
                slot_freed = now
        else:
            now = max(next_consume, now)
            if buffered:
                buffered -= 1
                delivered += 1
            next_consume = now + CONSUME_MS * rng.uniform(0.5, 1.5)
            next_produce = max(next_produce, now)   # a blocked decoder resumes on drain
    return peak, slot_freed / 1000, now / 1000

for label, bound in [("unbounded queue", None), ("bounded, B = 32", 32)]:
    peak, slot_s, done_s = stream(bound)
    print(f"{label}: peak buffer {peak:3d} tokens | "
          f"slot freed at {slot_s:5.1f}s | last token at {done_s:5.1f}s")
```

跑一下，这两行输出就是第 4 节的全部内容，用数字写出来。
无上限的那次运行峰值缓冲到了 237 个 token，几乎整条 300 token 的回复都堆在网关内存里，
因为解码器 6.0 秒就跑完了，而客户端要取到 29.6 秒；
把这个缓冲乘上几万条并发流，网关的内存账单就是所有慢客户端的回复语料同时压在那儿。
设上限的那次运行把缓冲精确地卡在 32，但被阻塞的 decode 循环把推理槽位一直握到 26.2 秒，
而不是 6.0 秒，这正是[第 4 节](04-backpressure-and-concurrency.md)讲的那个取舍，
也是"消费者没救就中止"这条策略之所以存在的原因。
两次运行的送达时间完全一样，都是 29.6 秒，因为定下端到端节奏的是客户端，不是缓冲策略。
把生产者时钟换成真正的推理引擎，把消费者时钟换成一条 SSE 连接，
把计数器换成一个带写超时的单流缓冲，把 `slot_freed` 这个时间戳换成引擎的 abort 并释放调用，
你就把本章的容量模型重搭了一遍。
