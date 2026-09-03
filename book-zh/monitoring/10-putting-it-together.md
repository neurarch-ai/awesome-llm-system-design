# 10. 把它们拼起来：完整的方案

第 1 到 6 节把每一层连同它的选项和取舍都讲了一遍，第 7 节展示了真实团队在哪里分道扬镳。它们都没给出的，是一套每个决定都已经拍板的完整系统。这一节收官，做三件事：给一套有立场的默认技术栈，免得选择困难症卡住第一次搭建；把本章的场景从头到尾走一遍，每个选择都落定并算清成本；再展示同样这些决定在约束变了之后会怎么翻盘。最后以一个最小可运行的监控器收尾，一个文件，不用装任何东西。

## 默认技术栈：从这里开始，有理由再偏离

本章的每一层都有好几个站得住脚的选项，一个第一次搭的人可能花掉一整周比较可观测性厂商，却一条 trace 都还没记下来。别这样。下面这套栈是第一次上生产的合理默认值；每一行都写明什么时候该偏离，以及哪一节解释了原因。工具年年在变，但每一层的接口（trace、指标、代理分数、漂移检查、告警）不变，所以按接口选层，任何具体平台都当作可替换的。

| 层 | 默认选择 | 什么时候偏离 | 为什么（章节） |
|---|---|---|---|
| Tracing | OTel 风格，一步一个 span；原样记录输入、检索到的上下文、输出；模型 id 和 prompt 版本挂在生成 span 上 | 单轮端点，没有检索也没有工具：一个生成 span 装下全部 | [2](02-what-to-observe.md) |
| 全量流量上的指标 | p50/p95/p99 和 TTFT（不是均值）、每请求的成本和 token、按类别分的错误率，都从 span 属性推导 | 没有例外。trace 一旦存在，这些几乎不要钱 | [2](02-what-to-observe.md)、[6](06-serving-and-scaling.md) |
| 没有标签的在线评估 | 抽样跑 LLM judge（faithfulness + relevance），加上对着日志上下文的 grounding 检查，加上隐式用户信号；在拿它们中任何一个告警之前，先用人工标签校出 kappa | 答案本来就不该有文档依据：去掉 grounding 检查，保留 judge 和行为信号 | [3](03-online-eval-without-labels.md) |
| 漂移检测 | 输入 embedding 相对参考窗口的余弦距离（便宜的 encoder，全量流量）作为先行指标；输出侧的代理指标做确认 | 流量太薄，窗口统计不稳：改成靠定时的冻结评估集回放 | [4](04-detecting-drift-and-regressions.md) |
| 回退关卡 | 每次模型或 prompt 变更都回放冻结评估集；全量发布前先跑百分之五到十的金丝雀，覆盖 24 小时流量周期 | 错误不可逆（会执行动作的 agent）：只跑影子，直到 diff 证据足以支撑把它暴露给人 | [4](04-detecting-drift-and-regressions.md) |
| 告警策略 | 对窗口内的比率变化量取 z >= 3，绝不对单个事件告警；分级：护栏飙升叫人、无依据率 z 飙升叫人、judge 衰减开工单、输入漂移进看板 | 周级季节性很强：改用对齐"周内小时"的基线，让告警打在残差上而不是周期性上 | [5](05-alerting.md)、[8](08-interview-qa.md) |
| 采样 | 分层，不是均匀：对丢弃、大改、低检索分数、护栏擦边超额抽样，另外留一片均匀抽的基线 | 高风险变更窗口：临时提高采样率，确认没问题再降回去 | [5](05-alerting.md)、[6](06-serving-and-scaling.md) |

评估那一行里的校准条款，是新手最爱跳过、事后最后悔的一条：一个没校准过的 judge 只是一个自信的猜测，而拿一个自信的猜测去叫人，正是一条监控通道在头一个月就被静音的原因。几百个人工标签，在 judge 和用户第一次意见相左的时候就把成本赚回来了。

## 完整的方案

回到[第 1 节](01-clarifying-requirements.md)的场景：一个企业客服 RAG copilot，生产流量上没有标签，prompt 一周要改好几次，模型每季度换一次，观测预算是服务成本的百分之十五，人工坐席会在五到三十分钟的延迟之后接受或丢弃每个答案，而回退必须在数小时内浮出水面。下面是整个系统，每个选择都已拍板，并附上它胜出的理由。

| 决定 | 选择 | 为什么是它 |
|---|---|---|
| Tracing | 一步一个 span，检索到的上下文原样记录，conversation id 贯穿多轮 | 没有记下上下文，事后就做不了 grounding 检查；多轮会话必须能被重建出来 |
| 质量代理指标 | judge 跑百分之十的抽样，grounding 检查跑同一片，接受/丢弃率跑全量流量 | 坐席的接受/丢弃是一个密度高、诚实、还免费的标签代理；预算把 judge 卡在百分之十五，所以要留余量 |
| judge 设计 | 两阶段（先自由推理，再用小模型重排格式），模型和 prompt 版本钉死，任何告警之前先在几百个人工标签上量出 kappa | 在推理中途强行要求结构化输出会掉准确率；一个没钉版本的 judge 会凭空造出假回退 |
| 漂移监控 | 便宜的 encoder 给每个查询做 embedding；相对参考窗口算余弦距离，每次有意的变更之后重置参考窗口 | 花生成成本的零头就能拿到一个先行指标；参考窗口一旦陈旧，每次有意的改动都会触发它 |
| 回退关卡 | 冻结评估回放接在每一次 prompt 修改上；每季度的模型替换先做影子 diff，再跑百分之五到十的金丝雀 24 小时 | 在这个场景里，prompt 修改才是高频风险；评估集要从被标记的生产 trace 里刷新，才不会过期 |
| 告警 | 护栏飙升立刻叫人；无依据率 z >= 3 一小时内叫人；judge 滚动平均衰减开工单；输入漂移进每周复盘 | 告警速度要和一个坏答案的代价匹配；输入漂移单独永远不叫人，因为它只预示麻烦，并不确认麻烦 |
| 采样 | 分层权重：丢弃、大改、低检索分数、护栏擦边，另加一片均匀抽的 | 均匀抽样会把 judge 和人力预算烧在简单请求上，同时错过那个正在造成损害的罕见失败 |
| 保留策略 | 被标记的 trace 存全量，干净的过一小段窗口就截断，PII 在进长期存储之前脱敏 | 不然 trace 存储会变成整个基础设施里最大的一池敏感数据 |

**采样率和 judge 成本。** 拿一个示意数字：每天十万请求。一次 judge 调用的开销大致和生成调用本身相当（[第 3 节](03-online-eval-without-labels.md)），所以全判一遍等于服务账单翻倍。按百分之十抽样，judge 大约增加服务成本的百分之十；grounding 检查搭同一片抽样的便车，漂移 encoder 和从 span 推导的指标跑全量流量，再加一两个百分点。合计大约百分之十二，落在百分之十五的预算内，还留出了在模型替换前后临时提高采样率的余量。

**检测延迟。** 用[第 6 节](06-serving-and-scaling.md)的公式 $t_{\text{detect}} \approx k / (s \cdot \lambda \cdot r_{\text{fail}})$，代入 $s = 0.10$、$\lambda$ = 每天十万请求、失败率百分之二、置信度所需 $k = 50$ 条被标记的 trace，得到 50 / (0.10 x 100,000 x 0.02) = 0.25 天，大约六小时。这满足"以小时计，不是以天计"的要求，同时也把取舍摆得很实在：采样砍半到百分之五，观测账单砍半，检测时间推到半天。

**告警阈值。** 百分之十的抽样下，judge 每天给大约一万条 trace 打分，所以一个 500 条 trace 的告警窗口大约 72 分钟就能填满。基线无依据率取百分之五时，[第 5 节](05-alerting.md)那个 z-score 会在窗口比率涨到大约百分之八时达到 3：一次三个百分点的比率偏移会在发生后约一小时内叫人，而日常噪声不会。窗口大小和采样率是一起调的，因为 $n_t$ 就坐在同一个公式的分母上。

**存储。** 一条带原始 prompt、检索上下文和输出的 LLM trace 大约 20 KB（示意值）。每天十万请求就是每天 2 GB，原始 trace 每月大约 60 GB。分级保留只给被标记的那约百分之十存全量，其余的过一小段窗口就截断，稳态存储能砍下好几倍，同时也把需要脱敏的面积缩小了。

**第一个月会出什么事。** 早期运营里有三种失败模式占主导，所以它们的信号要在上线前就接好：延迟稳定之下的无声质量回退（周五发了一次 prompt 修改，延迟和错误看板一路绿灯，无依据率整个周末一直在爬，因为冻结回放只接在模型替换上，没接在 prompt 修改上）；judge 漂移（faithfulness 分数在涨，而对着新鲜人工标签的 kappa 在跌，说明是仪器在捧场，不是产品在变好；重新校准要按排期做，不要等到起疑才做）；以及告警疲劳（固定基线的 z-score 在有季节性的流量上每周一早上准时叫人，直到这条通道被静音；按[第 8 节](08-interview-qa.md)说的，趁信任还没耗光就换成对齐"周内小时"的基线）。

## 同样的技术，换一组约束

实践中真正值得复盘的问题，不是"哪个可观测性平台最好"，而是"在我的约束下，什么样的监控设计才是对的"。下面把同一套分层栈搭了三遍。只有中间那一列是上面那套方案，另外两列保持完全相同的层接口，但几乎每一个实现选择都换掉了。

| | 内部文档机器人 | 企业客服 copilot（本章） | 受监管的消费级助手 |
|---|---|---|---|
| 流量 / 风险 | 每天约 2k 请求；答错浪费几分钟 | 每天约 10 万请求（示意值）；一个自信的错答案会伤害客户信任 | 每天数百万请求；一个有害答案就是监管风险 |
| 观测预算 | judge 的绝对成本可以忽略：想判百分之百的流量也行 | 服务成本的百分之十五；judge 抽样约百分之十，做分层 | 预算更大，但百分之百判一遍照样负担不起；便宜的粗筛模型给所有请求打分，完整 judge 只接可疑的尾部 |
| 质量信号 | 点赞点踩，加上每周翻一遍被判过分的 trace | judge + grounding + 坐席接受/丢弃，用 trace id join 起来 | judge + grounding + 从放行流量里抽样的强制安全性重扫 |
| 漂移与关卡 | 发布时回放冻结评估集；不做漂移监控，流量太薄，窗口统计不稳 | 每次 prompt 修改都回放；模型替换先影子再百分之五到十金丝雀 | 永远影子优先；金丝雀跑很久；自动关卡之上再加一道人工签字 |
| 告警 | 不设 pager。每周看一次看板 | 分级：叫人 / 工单 / 看板，比率变化量取 z >= 3 | 护栏和安全重扫的飙升立刻叫人，全天候 |
| 人工审核 | 开发自己端着咖啡翻被标记的 trace | 分层队列，每月校准一次 judge | 常设审核团队；标签同时喂给校准和合规审计 |
| 什么算过度设计 | 金丝雀基础设施、分层抽样、on-call 分级 | 百分之百判一遍、常设审核团队 | 安全这一侧没有过度设计；在这里均匀抽样才叫失职 |

有两条教训自然浮出来。第一，文档机器人那一列基本上全是做减法：每天 2k 请求，哪怕全覆盖，judge 的账单也就是零花钱；流量太薄，窗口化的漂移统计根本稳不下来；而 pager 因为方差响的次数会多过因为真相响的次数。发布时回放一遍冻结评估集，加上每周翻一翻，这就是全部系统了，而这是对的，不是偷懒。第二，受监管那一列说明观测预算和告警分级会朝着跟成本相反的方向走：当一次漏检就是一起监管事件时，采样是被分层得更狠，而不是变得更便宜；放行流量上的安全性重扫不再是可选项（[第 8 节](08-interview-qa.md)）；而昂贵的 judge 是靠一个便宜的粗筛模型来保护的，不是靠降采样率。

## 每一条约束各自决定什么

压缩版的决策指南。从需求里读出左边那一列，右边告诉你它先动的是哪根杠杆，然后你再去比工具。

| 你的约束 | 它动的杠杆 | 经验法则 |
|---|---|---|
| 观测预算 | judge 采样率 $s$ | judge 成本随 $s$ 线性增长；$s$ 砍半，账单砍半，检测时间翻倍。省下来的钱花在分层上，不是花在覆盖率上 |
| 检测延迟目标 | $s$ 和窗口大小，一起动 | $t \approx k / (s \lambda r_{\text{fail}})$；高风险变更前后临时把 $s$ 调高，之后再降 |
| 一个错答案的代价 | 告警分级 | 确认的输出回退叫人；judge 的缓慢衰减开工单；输入漂移单独永远不叫人 |
| 变更频率 | 回放触发方式 | 一周改好几次 prompt，就意味着回放要接在修改这个事件上，而不是挂个夜间定时任务 |
| ground truth 的滞后 | 校准节奏 | 分钟级滞后的行为标签：用 trace id join，按天看趋势。人工标签：按排期重新校准 judge，并且把 kappa 摆在分数旁边一起报 |
| 流量规模 | 窗口大小 $n_t$ | 流量薄还用小窗口，会被方差绊倒；$n_t$ 要取到让 z 检验看得见你在乎的最小比率偏移 |
| 关于"答案有依据"的承诺 | 那一个 span 字段 | 在检索那个 span 上原样记下检索到的上下文；下游任何检查事后都补不回来 |
| 有季节性的流量 | 基线的选法 | 用对齐"周内小时"的基线；剔除已知周期性之后对残差告警，别对周期性本身告警 |
| 隐私与合规 | 保留与脱敏 | 只有被标记的 trace 存全量；进长期存储前把 PII 脱敏；原始 trace 的访问权限要和看板分开管 |

## 最小可运行的监控器

对每一篇可观测性教程的评价都一样：读者把厂商 SDK 和一个看板拼起来了，还是没看见机制本身。所以这里把本章的核心检测闭环放进一个文件，零安装。抽样的 judge 变成一条带种子的日质量分数流，回退变成一个已知的注入日，而[第 5 节](05-alerting.md)里那两种告警哲学并排跑：一个静态阈值（"分数看着不行就告警"）对上一个滚动窗口漂移检测器（当前窗口均值 vs 冻结的参考，以标准误为单位）。形状才是这里要学的东西；本章的每一节，都是在升级这个文件里的某一个函数。

```python
"""Static threshold vs rolling-window drift detection, runnable with no installs."""
import random, statistics

random.seed(13)

# --- simulate a daily quality proxy (sampled judge faithfulness score) -------

BASELINE_MEAN, NOISE_STD = 0.90, 0.02
REGRESSION_DAY, DAILY_DECAY = 31, 0.004          # slow regression: a bad prompt edit

def score(day):
    """Mean judge score for one day; production: aggregated from sampled traces."""
    drift = max(0, day - REGRESSION_DAY + 1) * DAILY_DECAY
    return BASELINE_MEAN - drift + random.gauss(0, NOISE_STD)

scores = [score(d) for d in range(1, 61)]        # days 1..60

# --- detector 1: static threshold (what most teams wire first) ---------------

STATIC_THRESHOLD = 0.82                          # "alert if the score looks bad"

def static_alert(scores):
    for day, s in enumerate(scores, 1):
        if s < STATIC_THRESHOLD:
            return day
    return None

# --- detector 2: rolling-window drift (baseline window vs current window) ----

BASE_WINDOW, CUR_WINDOW, Z_PAGE = 14, 7, 3.0     # page at z >= 3, as in alerting

def drift_alert(scores):
    base = scores[:BASE_WINDOW]                  # frozen reference window
    mu, sd = statistics.mean(base), statistics.stdev(base)
    for day in range(BASE_WINDOW + CUR_WINDOW, len(scores) + 1):
        cur = statistics.mean(scores[day - CUR_WINDOW:day])
        z = (mu - cur) / (sd / CUR_WINDOW ** 0.5)   # std-error units of the mean
        if z >= Z_PAGE:
            return day
    return None

# --- report ------------------------------------------------------------------

s_day, d_day = static_alert(scores), drift_alert(scores)
print(f"regression injected at day {REGRESSION_DAY} "
      f"(-{DAILY_DECAY:.3f}/day, noise std {NOISE_STD})")
print(f"static threshold < {STATIC_THRESHOLD}: "
      + (f"fires day {s_day}, lag {s_day - REGRESSION_DAY} days" if s_day else "never fires"))
print(f"rolling drift z >= {Z_PAGE:.0f}:        "
      + (f"fires day {d_day}, lag {d_day - REGRESSION_DAY} days" if d_day else "never fires"))

# rerun with a shallower regression: static goes blind, drift still catches it
DAILY_DECAY = 0.002
random.seed(13)
scores = [score(d) for d in range(1, 61)]
s_day, d_day = static_alert(scores), drift_alert(scores)
print(f"shallower regression (-{DAILY_DECAY:.3f}/day): "
      f"static {'day ' + str(s_day) if s_day else 'never fires'}, "
      f"drift fires day {d_day}, lag {d_day - REGRESSION_DAY} days")
```

跑一下，输出会在大约六十行里演示完本章关于告警的核心论断。回退在第 31 天注入，分数每天衰减 0.004，噪声标准差 0.02。静态阈值直到第 51 天才响，滞后 20 天，因为一次缓慢的衰减必须从健康的基线一路走到那条线被画的地方。滚动漂移检测器在第 38 天就响了，滞后 7 天，因为它拿当前窗口跟冻结的参考比，对偏移本身告警，而不是对一个绝对水位告警。第二次运行把话说得更狠：把衰减减半到每天 0.002，静态阈值在 60 天的观察期里再也没响过，而漂移检测器照样在第 42 天抓到了它。这就是[第 5 节](05-alerting.md)对比率和变化量告警、而不对水位告警的原因，也是[第 4 节](04-detecting-drift-and-regressions.md)坚持每次有意的变更之后都要重置参考窗口的原因：这个检测器的全部功力，都来自基线是对的。把模拟分数换成你 trace 流上被判过分的 trace 的聚合值，把参考窗口换成对齐"周内小时"的基线，把 print 换成一个 pager，你就把本章重搭了一遍。
