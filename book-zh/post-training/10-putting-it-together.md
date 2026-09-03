# 10. 把它们拼起来：完整的方案

第 1 到 6 节把每个阶段的选项和取舍讲了一遍，第 7 节展示了真实团队在哪里分道扬镳。
它们都没给出的，是一条每个决定都拍板了的完整流水线。本节做三件事：
给一套有主张的默认技术栈，让选择困难症不至于卡住第一次搭建；
把本章的场景从头到尾走一遍，每个选择都定下来并且算清账；
再展示同样这些决定在约束变了之后会怎么翻转。最后收在一个最小可运行的偏好调优器上：一个文件，不用装任何东西。

## 默认技术栈：从这里开始，有理由再偏离

本章每个阶段都有三到六个说得过去的选项，而一个第一次做的人，
可能在还没测过 prompt 基线之前就花掉一个月去立一条 RLHF 流水线。别这样。
下面这套栈是第一次上生产时一个理智的默认值；每一行都说明什么时候该偏离，以及哪一节解释了为什么。
框架每年都在变，但每个阶段的接口（诊断、整理、训练、门禁、服务、刷新）不变，
所以按接口来选阶段，把任何具体的 trainer 都当成可替换件。

| 阶段 | 默认选择 | 什么时候偏离 | 为什么（章节） |
|---|---|---|---|
| 诊断 | 带 few-shot 样例的调优 prompt，作为量过的基线 | 永远不偏离。没有它你无法知道训练到底有没有帮上忙 | [第 2 节](02-decide-prompt-rag-or-train.md) |
| 知识还是行为 | 事实走检索；只有行为、格式和技能才进权重 | 事实确实是冻结的而且很少：few-shot 也许装得下 | [第 2 节](02-decide-prompt-rag-or-train.md) |
| 数据 | 几千条整理过的数据对走完漏斗：规则过滤、去重、去污染，然后是质量门禁 | 真实日志太少：先合成一批引导起来，但要过同样的门禁 | [第 3 节](03-data-curation.md) |
| 模板 | 一个 prompt 模板，钉死，训练和服务逐字节一致 | 永远不偏离。五种写法会训出五种互相打架的行为 | [第 3 节](03-data-curation.md) |
| 方法 | SFT 加 LoRA adapter，attention 和 FFN 投影上取 r = 16 | 行为偏移很大，或者高 rank 下仍然漂出分布：改全量微调 | [第 4 节](04-methods.md) |
| 显存 | QLoRA：4-bit 冻结基座，bf16 adapter，单张 GPU | 有闲置集群而且数据集很大：用 16-bit 全量权重 | [第 4 节](04-methods.md) |
| 对齐 | 不做。光靠 SFT 就能把大多数行为类任务送上线 | 失败模式是"说得通但更差"的答案：上 DPO，beta 取 0.03 到 0.1 | [第 4 节](04-methods.md) |
| 评估门禁 | 留出的去污染集合，相对当前线上的回归检查，冒烟-核心-完整三层 | 永远不偏离。门禁要在第一次训练之前就建好 | [第 5 节](05-evaluation-and-gates.md) |
| 服务 | 多 LoRA：一个热基座，可热切换的 adapter，回滚就是改路由 | 你做了全量微调：每个模型都得占自己的服务槽位 | [第 6 节](06-serving-adapters.md) |
| 刷新 | 飞轮：挖生产失败案例，标注难的那些，重训，过门禁 | 永远不偏离，但要保留一块人工标注的核心数据防止坍缩 | [第 6 节](06-serving-adapters.md) |

门禁那一行是新手最容易跳过、事后最后悔的：
没有一个在训练之前就建好的、留出的去污染评估集，每条损失曲线都只是感觉，
而"损失很低、评估在跌"是经典陷阱。花一个下午把门禁搭起来，
在第一次"更新"本来要悄悄上线成"更差"的时候就回本了。

## 完整的方案

回到[第 1 节](01-clarifying-requirements.md)的场景：一个通用基座模型，输出格式不稳定，
语气也抓不住品牌调性；大约四千条人工标注的 (prompt, ideal response) 对；自托管的开源权重；
领域稳定；延迟要在两秒以内；质量门禁得我们自己设计。
下面是整条流水线，每个选择都拍板了，并附上它胜出的理由。

| 决策 | 选择 | 为什么是它 |
|---|---|---|
| 阶梯停在哪 | 先做 prompt 基线，然后上 SFT；跳过 RAG 和偏好调优 | 这是行为问题不是知识问题；领域稳定，而且落败答案明显很差，所以 SFT 的正面样本就够了 |
| 基座模型 | 7B 级别的开源 instruct 模型，自托管（示意） | 这个场景里服务是我们自己控的；7B 装得下格式和语气，而且第 4 节的 adapter 数学就是按这个规模写的 |
| 数据 | 4000 条数据对走漏斗；约 3200 条训练，600 条留出，200 条冒烟（划分为示意值） | 便宜的门禁放前面，去污染放在质量门禁之前；留出集是整道门禁的地基 |
| 模板 | 一个 chat 模板，钉死，训练和服务完全一致 | 模型学模板和学内容一样卖力；这里的偏差是无声的杀手 |
| 方法 | 用 QLoRA 做 SFT：4-bit 冻结基座，attention 和 FFN 投影上 LoRA r = 16 | 中等程度的行为偏移住在一个低秩更新里；单张 GPU 就够；冻结基座让后面所有事情成为可能 |
| 训练计划 | 1 到 3 个 epoch，LR 在 2e-5 到 1e-4 区间，在验证集最低点早停 | 小数据集过拟合很快；决定什么时候停的是验证曲线，不是训练损失 |
| 评估门禁 | 格式精确校验，相对当前线上的 LLM 裁判胜率（带置信区间），回归测试套件，1% 线上切片 | 格式是结构化的（用 exact match）；语气是比较出来的（用胜率）；回归检查抓的是无声的次要损伤 |
| 服务 | 多 LoRA：热基座加这个 adapter；晋升和回滚都是改路由 | 基座不用重新部署；开一条 A/B 切片就是加第二条路由 |
| 版本命名 | 每个产物都带上 `brand-voice-2026-07-3200-<hash>` | 没有数据快照的名字，你没法复现、没法调试，也没法证明评估集是不相交的 |

**数据集。**4000 条标注数据对按[第 3 节](03-data-curation.md)定下的固定顺序进入漏斗：
规则过滤、去重、去污染，然后是质量门禁。示意的留存率：
约 3800 条挺过便宜的那几道门禁，约 3400 条通过质量筛，
切出 600 条留出集和 200 条冒烟集（按时间分开并验证过不相交）之后，还剩约 3200 条训练样本。
这个不相交检查每次训练前都要重跑，不是查一次就完事：
在这么小的集合上，一条泄漏样本就能把指标抬起来，
而且如果不查，质量门禁反而会优先留下泄漏的样本，因为它们得分高。

**adapter 的大小和显存。**本章的公式是每个被适配矩阵的可训练参数量
$= r(d+k)$。一个 4096 x 4096 的 attention 投影有 1680 万个冻结权重，
它的 rank-16 adapter 训练 16 x (4096 + 4096)，大约 13.1 万个参数，不到这个矩阵的百分之一。
把一个 7B 模型的 attention 和 FFN 投影加起来，总共大约是权重的 0.08%（[第 4 节](04-methods.md)），
数量级上是 600 万个可训练参数，bf16 下约 12 MB 的产物。
QLoRA 的预算是：4-bit 的冻结基座大约每参数 0.5 字节，7B 接近 3.5 GB，
bf16 的 adapter 加上它的优化器状态是几十 MB，
所以配上梯度 checkpointing，这次训练能塞进一张普通 GPU（余量为示意值）。
同一个模型做全量微调则需要 16-bit 权重外加每个参数两份 Adam 动量，
那完全是另一个硬件档次，而这次行为偏移根本不需要。

**训练成本。**约 3200 条样本、每条几百 token，一个 epoch 是 200 万 token 量级，
两到三个 epoch 算 400 万到 600 万 token：单卡上个位数的 GPU 小时，
按市面价格是几十美元（示意值）。真正值得记住的是这个比例：
训练本身是整条流水线里最便宜的一笔开销。
那 4000 条人工标注比所有 GPU 小时加起来还贵，
这正是[第 3 节](03-data-curation.md)说微调是一个穿着算力外衣的数据问题的原因，
也是为什么那个从生产环境里收割标注的飞轮才是会复利的资产。

**门禁。**按[第 5 节](05-evaluation-and-gates.md)分层。
200 条的冒烟集每个 checkpoint 之后都跑（秒级）。
600 条的核心门禁在晋升之前跑：格式合法性用 exact match，因为结构化输出没有部分分，
再加上在同一个集合上相对当前线上模型的整套回归测试。
语气是比较出来的，所以胜率要成对地跑：裁判需要的是 prompt 而不是标准答案，
所以它采样 1000 条生产 prompt（新模型 vs 当前线上，顺序随机，
并塞入完全相同的一对回复作为注意力检查），
门槛是 55% 的胜率且置信下界越过 0.50，这也是为什么要 1000 条而不是 100 条。
裁判要在抽样子集上用人工标注做校准，因为裁判会高估长度和格式，
而这恰恰就是这个模型被训练的那两个轴。
只有离线测试全过之后，才开 1% 的线上切片，因为离线指标会高估上线准备度。

**第一个月会坏在哪。**早期运维里有三种失败模式最常见，所以上线前就要把它们的信号接好：
通用能力回归（回归测试里次要任务的分数在往下漂，而品牌语气那个指标稳住不动；
这是多训了一个 epoch 带来的灾难性遗忘，解法是减少 epoch 或者混入一部分通用数据）；
格式过拟合（模型把训过的那套 schema 硬套到本来想要自由文本的请求上，
而裁判抓不住它，因为裁判本来就奖励格式；把它暴露出来的是注意力检查和人工校准样本）；
以及训练和服务之间的模板偏差（服务侧改了一下 prompt，悄悄和钉死的训练模板分了岔，
模型开始无视 system prompt；解法是在 CI 里对模板做一次逐字节一致性检查，而不是靠人盯着）。

## 同样的技术，换一组约束

实践中真正重要的复盘问题不是"哪个方法最好"，而是"在我的约束下哪个方法最好"。
下面是同一条流水线搭三遍。只有中间那一列是上面那套方案，另外两列保持完全相同的阶段接口，
但几乎每一个实现选择都换掉了。

| | 用闭源 API 的初创机器人 | 品牌语气专家（本章） | 有可验证奖励的编程模型 |
|---|---|---|---|
| 差距 | 格式和语气，程度轻 | 规模化的格式和语气 | 基座缺失的一项推理能力 |
| 数据 | 约 300 条样本，没有标注预算 | 4000 条人工标注对，走过漏斗整理 | 大规模合成语料，加单元测试作为打分信号 |
| 权重权限 | 没有：只能用厂商 API | 完全：自托管开源权重 | 完全，另加一个训练集群 |
| 阶梯停在哪 | 第 1 级：调优 prompt、few-shot、输出 schema | 第 3 级：prompt 基线之后差距还在，上 SFT | 第 3 级然后第 4 级：SFT 热身，然后 RL |
| 适配方式 | 不适配（prompt 真的失败了才考虑厂商的微调 API） | 用 QLoRA 做 LoRA r = 16，单张 GPU | 全量微调：偏移很大，LoRA 会漂出分布 |
| 对齐 | 不做 | 不做：SFT 已经补平差距 | 针对测试通过率做 GRPO：可验证，所以不需要偏好标注 |
| 评估门禁 | 手工维护的 50 条冒烟集 | 分层门禁，带置信区间的裁判胜率，线上切片 | Exact match 和单元测试通过率；有测试的地方不用裁判 |
| 服务 | 厂商的事 | 多 LoRA，adapter 路由，毫秒级回滚 | 独占部署；全量微调放弃了 adapter 切换 |
| 什么算过度设计 | 跑任何训练都算；给一个 schema 就能修好的问题上 DPO | RLHF 那条五组件流水线；没有可验证奖励却上 GRPO | 给测试能打分的东西找人标偏好；编译器就是裁判的地方还用 LLM 裁判 |

从中掉出来两条教训。第一，初创那一列基本上全是删减：
只有 300 条样本、又没有权重权限，[第 2 节](02-decide-prompt-rag-or-train.md)那个阶梯停在第 1 级，
诚实的答案是，在量过的基线证明差距存在之前，
一个调优 prompt 加一个输出 schema 就是整条流水线。
第二，编程那一列展示了训练信号的位置互换：
当奖励能对每个样本验证时，针对检查器做 RL 胜过收集偏好标注
（[第 7 节](07-how-teams-do-it-in-production.md)里 DeepSeek R1 的教训），
行为偏移大到必须全量微调，而评估门禁反而更简单而不是更复杂，
因为 exact match 取代了那个需要校准的裁判。

## 每种约束决定什么

压缩过的决策指南。从你的需求里读出左边那一列，右边两列告诉你它在你比较任何框架之前先动了哪根杠杆。

| 你的约束 | 它动的杠杆 | 经验法则 |
|---|---|---|
| 差距在哪 | 阶梯的级数 | 知识：走检索，永远不要进权重。行为、格式、技能：先 prompt，然后 SFT |
| 标注样本的数量 | 方法可行性 | 几百条：prompt 和 few-shot。几千条且干净：SFT。有比较对：DPO 才成为可能 |
| 权重权限 | 适配的上限 | 闭源 API：prompt、schema 和厂商的微调接口就是全部菜单 |
| 行为偏移的幅度 | LoRA 还是全量微调 | 小幅调整：LoRA r = 8 到 64。高 rank 下仍漂出分布：上全量微调，而不是继续加 rank |
| GPU 预算 | 冻结基座的精度 | 单卡：QLoRA（4-bit 基座，bf16 adapter）。量化没弄坏的东西，加 rank 也修不好 |
| "不该说什么"重不重要 | SFT 还是偏好调优 | 落败样本明显很差：只在胜出样本上做 SFT 就够。落败样本说得通但更差：上 DPO，beta 0.03 到 0.1 |
| 奖励可不可验证 | DPO/RLHF 还是 GRPO | 每个样本都能检查（测试、数学、检索排名）：上 GRPO，不需要奖励模型。开放式：用偏好对 |
| 租户或领域的数量 | 服务形态 | 变体很多：多 LoRA，一个热基座加 N 个 adapter；全量微调放弃了这条路 |
| 领域的变动频率 | 什么该进权重 | 会变的事实：走检索，随时更新。稳定的行为：进权重，很少重训 |
| 晋升的风险 | 门禁的深度 | 每个 checkpoint 跑冒烟集；任何用户看到之前，跑完整套测试加回归检查加线上切片 |

## 最小可运行的偏好调优器

所有对齐教程的读后感都一样：读者把 trainer、accelerator 配置和参考模型都拼起来了，
却始终没看见损失长什么样。所以这里把 SFT 和 DPO 并排放进一个文件，不用装任何东西。
每个生产组件都被换成了接口相同的最小物件：
LLM 变成每个候选回复一个 logit，偏好数据集变成三条 (prompt, chosen, rejected) 三元组，
冻结的参考模型就是初始策略，trainer 是手写的梯度下降。
每个 prompt 还额外带一条旁观回复，任何偏好对都没提到过它，
因为两种方法的差别，正藏在这条回复身上发生的事情里。

```python
"""SFT vs DPO on a toy policy: one logit per candidate response, stdlib only."""
import math

# Three prompts; each has (chosen, rejected, bystander) candidate responses.
# The rejected answer is the tempting-but-wrong one; the bystander is a
# harmless alternative that no preference pair ever mentions.
PAIRS = [
    ("refund past the window", "Politely decline; offer store credit.",
                               "Sure, full refund, no questions!",
                               "Please contact support."),
    ("angry customer",         "Acknowledge, apologize once, give the next step.",
                               "You are totally right, we are terrible!",
                               "Noted."),
    ("feature request",        "Thank them, log it, promise nothing.",
                               "Absolutely, shipping it next week!",
                               "We will see."),
]
C, R, O = 0, 1, 2               # chosen / rejected / bystander
INIT = [0.0, 1.0, 1.0]          # base model: both wrong answers more likely

def logprobs(z):
    lse = math.log(sum(math.exp(v) for v in z))
    return [v - lse for v in z]

def softmax(z):
    e = [math.exp(v) for v in z]
    return [v / sum(e) for v in e]

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def train(method, steps=300, lr=0.5, beta=0.5):
    policy = [list(INIT) for _ in PAIRS]      # one logit per candidate response
    ref = [logprobs(INIT) for _ in PAIRS]     # frozen reference = the SFT checkpoint
    for step in range(steps + 1):
        margins = []
        for z, rf in zip(policy, ref):
            lp = logprobs(z)
            margins.append(lp[C] - lp[R])
            if step == steps:
                continue
            if method == "dpo":
                # loss = -log sigmoid(beta * (policy logratio - reference logratio))
                m = beta * ((lp[C] - rf[C]) - (lp[R] - rf[R]))
                g = (1.0 - sigmoid(m)) * beta   # gradient magnitude on the pair
                z[C] += lr * g                  # chosen logit up ...
                z[R] -= lr * g                  # ... rejected logit down: the margin
            else:                               # sft: cross-entropy on chosen only
                p = softmax(z)
                for j in range(3):
                    z[j] -= lr * (p[j] - (1.0 if j == C else 0.0))
        if step % 100 == 0:
            print(f"  step {step:3d}  mean margin log P(chosen) - log P(rejected) = "
                  f"{sum(margins) / len(margins):+.3f}")
    return policy

for method in ("sft", "dpo"):
    print(f"{method.upper()} training:")
    policy = train(method)
    p, lp = softmax(policy[0]), logprobs(policy[0])
    print(f"  prompt 1 after training: P(chosen)={p[C]:.3f}  "
          f"P(rejected)={p[R]:.3f}  P(bystander)={p[O]:.3f}")
    print(f"  rejected vs bystander: log P(rejected) - log P(bystander) = "
          f"{lp[R] - lp[O]:+.3f}\n")
```

跑一下，打印出来的东西就是本章的论点，六十行讲完。
两种方法都把偏好间隔从 -1.0（基座模型更偏爱那个诱人的错误答案）推到了明显为正：
300 步之后 SFT 的平均间隔到 +6.09，DPO 到 +7.56，两者都把 P(chosen) 留在 0.95 以上。
真正拉开差别的是每一段的最后一行。
在 SFT 下，rejected 相对 bystander 的对数比最终停在正好 +0.000：
只在胜出样本上做交叉熵，从来没见过负面样本，
所以那条谄媚的回复相对于无害的"请联系客服"，概率跟初始化时一模一样。
而在 DPO 下同一个比值落在 -4.279，因为对比损失专门把 rejected 的 logit 压了下去，
这正是[第 4 节](04-methods.md)所说 SFT 表达不了的"不该说什么"。
代码里 DPO 的那步更新也是 beta 所在的位置：梯度按 $(1 - \sigma(m))\beta$ 缩放，
所以随着策略相对冻结参考的间隔变大，更新量衰减到零，缰绳也随之收紧。
把 logit 表换成一个 transformer，把三条三元组换成一份整理好的偏好集，
把手写的那一步换成 DPOTrainer，你就把本章的第 4 级重建出来了。
