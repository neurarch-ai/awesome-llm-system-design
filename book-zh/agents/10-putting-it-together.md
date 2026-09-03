# 10. 把它们拼起来：完整的方案

第 1 到 6 节把循环的每一层连同它的选项和取舍讲了一遍，第 7 节展示了真实团队在哪些
地方产生分歧。它们都没有给出的，是一套每个决定都已经拍板的完整系统。这一节收官，
做三件事：给出一套有主张的默认技术栈，让你不至于在选型上纠结到动不了工；把本章的
场景从头到尾走一遍，每个选择都定下来并且算清成本；再展示同样这些决定在约束变了
之后会怎么翻转。最后收在一个最小的可运行 agent 循环上，一个文件，什么都不用装。

## 默认技术栈：从这里开始，有理由再偏离

本章的每一层都有两到五个说得过去的选项，一个第一次搭 agent 的人，可以在还没发出
一次工具调用之前就花掉一周去比较框架。别这么干。下面这套栈是第一个生产 agent 的
合理默认值；每一行都写清了什么时候该偏离，以及哪一节解释了原因。框架年年在换，
但每一层的接口（规划、提议、门禁、执行、观察、设限、评估）不会变，所以按接口
逐层选型，把任何具体的库都当成可替换件。

| 层 | 默认选择 | 什么时候偏离 | 为什么（对应小节） |
|---|---|---|---|
| 拓扑 | 单个工具设计得好的 agent | 子任务确实能拆开、每个都需要自己的上下文、而且墙钟延迟是瓶颈：编排器加 subagent，代价约 15 倍 token | [3](03-planning-and-tools.md)、[7](07-how-teams-do-it-in-production.md) |
| 规划 | 先规划后执行，出现矛盾就重新规划 | 第一次工具调用之前根本无从知道路径：反应式（ReAct） | [3](03-planning-and-tools.md) |
| 工具 | 窄口径、带类型、用枚举、参数做防呆设计 | 工具密集的多步任务，JSON 来回占了大头：在沙箱里执行代码 | [3](03-planning-and-tools.md)、[7](07-how-teams-do-it-in-production.md) |
| 安全 | 代码里的确定性调用前门禁：schema、策略、授权 | 永不偏离。任何写工具都必须过这道门禁 | [2](02-frame-the-system.md)、[5](05-reliability-and-cost.md) |
| 记忆 | 短期记录，加上给策略和历史做检索（RAG） | 记录接近上限或者单步成本往上爬：压缩；token 很重的产物：隔离 | [4](04-memory-and-state.md) |
| 上限 | 硬性步数上限和每任务 token 预算，由编排器强制执行 | 永远不要拿掉，只按任务类别调 $N$ 和预算 | [5](05-reliability-and-cost.md) |
| 模型 | 分层：路由步骤用便宜模型，只有推理才用贵的 | 每一步都是真正的策略推理（很少见）：单一强模型 | [5](05-reliability-and-cost.md) |
| 服务 | 复杂度路由：简单任务走同步路径，其余进持久化队列 | 所有任务都是长时间运行、非交互的：只做异步 | [6](06-serving-and-scaling.md) |
| 评估 | 离线用带标注的工单集测端到端任务成功率；线上看升级率、二次联系率和冲销率 | 永不偏离。调任何东西之前先把标注集建起来 | [8](08-interview-qa.md) |

最后一行是新手最容易跳过、也最容易后悔的：没有带标注的工单集，每一次 prompt 和
工具 schema 的改动都只是凭感觉，而且单步正确率看着挺好的时候，端到端的处理结果
可能是错的。一个下午的标注工作，在你第一次替换某个组件时就回本了。

## 完整的方案

回到[第 1 节](01-clarifying-requirements.md)那个场景：一个客服 agent，每天处理
5 万张工单，读工具免费，写工具卡在 schema 和策略检查后面，超过 \$50 的退款转人工
审批队列，简单工单 10 秒内解决，每张工单最多 \$0.10，每一步都要记日志。下面是
整个系统，每个选择都已敲定，并附上它胜出的理由。

| 决定 | 选择 | 为什么它胜出 |
|---|---|---|
| 拓扑 | 单 agent，每张工单一份上下文 | 一张工单很少能拆成几条能并发的独立工作流；扇出那大约 15 倍的 token 在这里换不来任何东西 |
| 规划 | 先规划后执行：查账户、查订单、判资格、执行、回复 | 工单的形态事先就知道；计划长度有界，成本可预测，而 ReAct 可能会跑偏 |
| 工具 schema | 窄口径、带类型、用枚举、金额以分为单位 | 门禁可以确定性地校验调用，而不用去猜一坨任意的 JSON |
| 调用前门禁 | 每个写操作都在代码侧过 schema、策略和授权检查 | 策略写在代码里是保证；写在 prompt 里只是一句 prompt 注入可以推翻的建议 |
| 审批路由 | 超过 \$50 的退款进人工队列，绝不自动执行 | 明确说了不可接受的失败就是未授权退款；门禁让它在结构上不可能发生 |
| 记忆 | 记录做短期；策略和历史按步骤检索；冗长的返回体丢进草稿区 | prefill 每一步都要重读整个上下文，所以留在里面的每样东西都要反复付钱 |
| 上限 | 步数上限 $N = 10$，每张工单 token 预算 1 万，放在编排器里 | 经济账要求成本有界；写在 prompt 一侧的"10 步之后停下"是没有强制力的 |
| 模型分层 | 分派和套模板的回复用便宜模型，策略推理用贵模型 | 循环里大多数步骤是路由不是推理；分层让成本上限站得住 |
| 服务 | 复杂度路由；同步路径 10 秒以内，写操作和升级进持久化队列 | 一张慢工单绝不能把快路径拖住；队列就是让两条路径各自独立扩容的那条接缝 |
| 重试 | 执行器里做指数退避加固定上限，写操作带幂等键 | 模型处理重试并不一致；一笔被重试的退款必须是空操作 |
| 审计与评估 | 只追加地记录推理、提议、门禁裁决和结果；离线用标注集，线上看升级率和冲销率 | 可审计是明确提出的需求，而这份日志同时就是评估和调试的底座 |

**步数和 token 预算。** 计划出来的形态大约 5 步：查账户（1）、查订单（1）、
判资格（1 到 2）、执行（1）、回复（1），所以 $N = 10$ 的上限给重试留出了余地
（[第 5 节](05-reliability-and-cost.md)）。token 预算是从成本上限推出来的：每张
工单 \$0.10，混合单价大约每百万 token \$10，预算就是每张工单 1 万 token 左右。
任何一条边界会被突破的工单，都在完成之前升级出去；升级是设计好的结果，不是失败。

**单张工单的成本。** 每一步的成本是整份记录的 prefill 加上生成：
$C_n = p \cdot T_{n-1} + g \cdot o_n$（[第 5 节](05-reliability-and-cost.md)）。
一张简单工单起步是 400 token 左右的 system prompt 加工单正文，每步增长 100 到
180 token，所以这五次 prefill 分别读到大约 400、550、730、820、890，总共约 3400
token。按示意单价（便宜档每百万输入 token \$1、输出 token \$3）算下来远不到一分钱，
下面那个玩具程序会把这个数字跑出来：大约 \$0.004。所以 \$0.10 这个上限不是给
中位数工单准备的，它是留给长尾的余量，重试、贵的推理模型和长记录都住在那里。
机群层面的核算：5 万张工单按 1 万 token 的预算，每天最多 5 亿 token
（[第 6 节](06-serving-and-scaling.md)），这个数字要在上线之前拿去跟模型提供方谈，
不是上线之后。

**延迟。** 10 秒的同步预算，要分给大约 5 次串行的模型调用外加它们的工具来回。
按示意值算，便宜档每次模型调用 1 到 1.5 秒，每次工具调用 100 到 300 毫秒，串行
循环落在 7 到 9 秒，在预算之内但没有余量。余量要靠并行那些相互独立的前沿节点找
回来：账户查询和订单状态互不依赖，所以一起发出去大约能省掉一整个来回
（[第 6 节](06-serving-and-scaling.md)）。凡是涉及写操作、或者大概率要升级的，
一开始就路由到异步，这样快路径永远不背那条慢尾巴。

**并发。** 每天 5 万张工单，大约每分钟 35 张，也就是每秒 0.6 张。每张工单最多有
$N = 10$ 步在途，模型层要能持续扛住大约 $0.6 \times N$ 个并发请求外加突发，而且
每个工具后端（CRM、OMS、政策检索）的容量都要按 agent 的调用速率来配，不能只按
用户的直接流量（[第 6 节](06-serving-and-scaling.md)）。

**第一个月会坏在哪里。** 早期运营里有三种失败模式占大头，所以上线之前就要把它们
的信号接出来：步数上限导致的升级率（中等难度的工单还没解决就耗光 $N = 10$，说明
计划或者工具 schema 让模型干得太费劲）、退款冲销率（那些本不该发生的冲销，是门禁
的策略表漏了某个边界情况，而且这也是合规部门第一个会来要的指标），以及工具后端上
相关联的重试风暴（下游一次超时，会让几乎所有在途工单在同一瞬间发起重试，所以
告警要按后端的并发量设，不是按单工单的速率，[第 6 节](06-serving-and-scaling.md)）。

## 同样的技术，换一组约束

实践里真正重要的复盘问题不是"哪种规划风格最好"，而是"在我的约束下哪种规划风格
最好"。下面是同一个循环搭了三遍。只有客服那一列是上面这套方案，另外两列保持完全
相同的分层接口，但几乎每一个实现选择都换掉了。

| | on-call 问答 copilot | 客服 agent（本章） | 后台编码 agent |
|---|---|---|---|
| 任务形态 | 在现有文档上一次性问答，没有写操作 | 每天 5 万张工单，读免费，写有门禁，每张 \$0.10 | 长程的仓库任务；花几小时没关系，正确性压倒一切 |
| 延迟预算 | 秒级，交互式 | 简单工单同步 10 秒以内，其余走异步 | 没有；完全异步，结果以一个 PR 落地 |
| 拓扑与规划 | 不是真正的循环：检索然后生成，实际上就一步 | 单 agent，在已知的 5 步形态上先规划后执行 | 单 agent，反应式的"先动手再验证"循环，带自测 |
| 工具接口 | 在向量库上做检索；只读 | 门禁后面的窄口径带类型 JSON 工具 | 在每会话隔离的沙箱 VM 里执行代码 |
| 安全 | 事实依据和引用检查；没有不可逆的东西需要设卡 | 调用前代码门禁，加超过 \$50 的人工审批队列 | 隔离本身就是门禁：VM 只能碰自己那份副本；测试和评审把住合并 |
| 上下文策略 | 选择：按问题检索相关片段 | 按步骤检索策略，接近上限时压缩，返回体放草稿区 | 隔离：每个会话拥有自己的 VM 和状态；结果以 diff 而不是记录的形式送出 |
| 上限 | 每个问题一个 token 上限；不需要步数上限 | 步数上限 10，token 预算 1 万，由编排器强制执行 | 步数上限和墙钟超时都给得宽；单任务成本高，但任务数量少 |
| 评估 | 抽样看回答有没有帮助；问题拦截率 | 带标注的工单集；升级率、二次联系率、冲销率 | 测试套件就是评估：每个任务都有可验证的通过 / 失败 |
| 什么算过度设计 | 用上 agent 循环本身；任何多 agent 的东西 | 多 agent 扇出、代码执行类工具、每工单一个 VM | 为压到 10 秒以内做的工作、流式输出、为省钱做的模型分层 |

由此得出两条结论。第一，copilot 那一列基本上都是做减法：没有任何工具会写状态时，
门禁、审批队列和步数上限统统消失，而[第 2 节](02-frame-the-system.md)那道
"工作流还是 agent"的判定说得很清楚，只要任务形态从不变化，固定的"检索然后生成"
流水线就赢过循环。第二，编码 agent 那一列展示了安全预算和延迟预算怎么互换位置：
没有用户在等，验证就可以做得更深（跑整个测试套件，而不是一次亚秒级的策略检查），
隔离取代了逐次调用的设卡，因为一个最多只能弄坏自己沙箱的 agent，需要的单步管控
更少（[第 7 节](07-how-teams-do-it-in-production.md)）。

## 每个约束各自决定了什么

压缩版的决策指南。从需求里读出左边这一列，右边就告诉你在比较任何框架之前，它先
拨动的是哪根杆。

| 你的约束 | 它拨动的杆 | 经验法则 |
|---|---|---|
| 一次写操作出错的代价 | 门禁的严格程度和审批阈值 | 涉及钱或不可逆状态：一律上确定性代码门禁；超过某个风险阈值，转人工队列 |
| 单任务的成本上限 | 步数上限、token 预算、模型分层 | 预算除以混合 token 单价就是 token 预算；步数上限按计划形态加上重试余量来定 |
| 延迟预算 | 同步 / 异步的划分，以及可并行的前沿 | 10 秒以内：前面放路由，把独立的工具调用并行掉；没有预算：批量异步，把省下的花在验证上 |
| 任务长度（步数） | 上下文策略 | 10 步以内：直接追加加前缀缓存；再长：到 token 阈值就压缩；token 很重的产物：隔离到子上下文里 |
| 有没有可验证的成功信号 | 重试架构 | 有测试或账本可查：Reflexion 式的重试划算；没有信号：重试只是成倍烧钱，质量不涨 |
| 子任务能拆开，同时有延迟压力 | 拓扑 | 两条都成立：编排器加并行 subagent，代价约 15 倍 token；有一条不成立：老实单线程 |
| 任务形态是否已知 | 规划风格 | 每次都是同样的步骤：用固定工作流，不用 agent；形态已知但有变化：先规划后执行；路径无从知道：ReAct |
| 不可信输入、私密数据和出站通道同时存在 | 工具的作用域 | 这就是致命三件套：在架构上拆掉其中一条腿；读过不可信内容的循环，不能同时握着出站通道 |
| 工具数量在增长 | 按步骤选工具 | 超过几十个工具之后，按步骤检索出一个相关子集，而不是每一轮把它们全都暴露出去 |

## 最小的可运行 agent 循环

所有框架教程读完的观感都一样：读者拼装了五个抽象，还是没看见那个循环。所以这里
把整个循环放进一个文件，零安装。每个生产组件都换成了接口相同的最小替身：模型变成
一次带随机种子、成功概率为 $q$ 的抛硬币，工具变成一张 token 数表，门禁是真代码，
成本计量表按[第 5 节](05-reliability-and-cost.md)的定价方式，对不断变长的记录
如实收取 prefill 的钱。要学的是这个形状；本章的每一节，都是在升级这个文件里的
某一个函数。

```python
"""A support-agent loop in one file: gate, step cap, token budget, cost meter."""
import random

STEP_CAP, TOKEN_BUDGET = 10, 10_000        # hard limits, enforced in code
PRICE_IN, PRICE_OUT = 1e-6, 3e-6           # $/token; illustrative blended rates
REFUND_LIMIT = 50                          # policy lives here, not in a prompt

TOOLS = {                                  # tool name -> tokens its result appends
    "lookup_account": 120, "lookup_order": 150, "check_eligibility": 60,
    "issue_refund": 40, "send_reply": 80,
}
PLAN = list(TOOLS)                         # plan-then-execute: known ticket shape

def gate(tool, args):
    """Deterministic pre-call check. The model proposes; code disposes."""
    if tool == "issue_refund":
        if not isinstance(args.get("amount"), (int, float)) or args["amount"] <= 0:
            return "reject: schema"
        if args["amount"] > REFUND_LIMIT:
            return "escalate: human approval queue"
    return "allow"

def run_ticket(rng, q=0.95, refund=32, retry=True, verbose=False):
    """One agent loop. Returns (resolved, steps, cost). q = per-step success."""
    transcript, cost, steps = 400, 0.0, 0  # 400 = system prompt + ticket tokens
    for tool in PLAN:
        while True:
            if steps >= STEP_CAP or transcript > TOKEN_BUDGET:
                return False, steps, cost  # hard limit hit: escalate to human
            steps += 1
            cost += PRICE_IN * transcript + PRICE_OUT * 30  # prefill re-reads all
            args = {"amount": refund} if tool == "issue_refund" else {}
            verdict = gate(tool, args)
            if verdict != "allow":
                if verbose:
                    print(f"  step {steps}: {tool} -> {verdict}")
                return False, steps, cost  # routed to a human, never executed
            ok = rng.random() < q          # per-step success draw
            transcript += TOOLS[tool] + 30  # observation + action text appended
            if verbose:
                print(f"  step {steps}: {tool:18s} {'ok' if ok else 'FAIL'}"
                      f"  transcript={transcript:5d}  cost=${cost:.5f}")
            if ok:
                break                      # next planned step
            if not retry:
                return False, steps, cost  # one bad step sinks the whole task
    return True, steps, cost

rng = random.Random(7)
print("one ticket, verbose (q=0.95, refund=$32):")
done, steps, cost = run_ticket(rng, verbose=True)
print(f"resolved={done}  steps={steps}  cost=${cost:.5f}\n")

print("policy gate demo (a hijacked model proposes refund=$500):")
done, steps, cost = run_ticket(rng, refund=500, verbose=True)
print(f"resolved={done}: escalated; the gate never executed the write\n")

print("error compounding over 2000 tickets per setting (5 planned steps):")
print("   q     q^5 predicted   no-retry observed   retry-under-cap   mean cost")
for q in (0.99, 0.95, 0.90, 0.70):
    plain = [run_ticket(random.Random(i), q=q, retry=False) for i in range(2000)]
    retried = [run_ticket(random.Random(i), q=q) for i in range(2000)]
    obs = sum(r[0] for r in plain) / 2000
    ret = sum(r[0] for r in retried) / 2000
    mc = sum(r[2] for r in retried) / 2000
    print(f"  {q:.2f}      {q**5:.3f}            {obs:.3f}             "
          f"{ret:.3f}         ${mc:.5f}")
```

跑一下，六十行左右的代码里会发生三件事。那张 verbose 的工单用 5 步、\$0.00384
解决掉，而且逐步的成本行随着记录变长在往上走，这就是[第 5
节](05-reliability-and-cost.md)那个 prefill 项被显式画了出来。被劫持的那张工单
提议退 \$500，门禁在第 4 步返回 "escalate: human approval queue"；不管模型想
干什么，那个写操作从没执行过。接着这张蒙特卡洛表展示了错误叠加：不做重试时，
实测成功率几乎正好贴着 $q^5$（$q = 0.95$ 时实测 0.789，理论 0.774），而在步数
上限之内由执行器做重试，把 $q = 0.90$ 拉回 1.000，连 $q = 0.70$ 都拉到 0.949，
代价是平均成本从 \$0.00390 涨到 \$0.00653，因为重试要多烧 prefill。重试是拿钱把
可靠性买回来，而上限框住了这笔采购的额度。把抛硬币换成真正的模型调用，把 token
表换成真工具，门禁和上限原地不动，你就把这一章重建出来了。
