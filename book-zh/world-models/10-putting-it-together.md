# 10. 把它们拼起来：完整的方案

第 1 到 6 节讲的是每个阶段各自的选项和取舍；第 7 节展示了前沿团队在哪里分道扬镳。
这一节把每个选择都敲定一次：一套有主张的默认技术栈，免得选范式选到瘫痪连第一版都搭不出来；
把本章的场景从头到尾落实一遍，连数据量、规划开销和评估的账都算清楚；
同一套技术在另外两组约束下重搭一遍；最后是一个最小的可运行规划器，一个文件，不用装任何东西。

## 默认技术栈：从这里开始，有理由再偏离

四种范式加三层数据，第一次动手的人还没跑过一次 rollout，面前就已经摆着十几种看着都合理的组合。
别做调研了。下面这套栈对一个面向控制的世界模型来说是个靠谱的默认值；
每一行都标了什么时候该偏离，以及哪一节解释了原因。范式会一直在你脚下变，
但接口不会：数据金字塔进去，动力学模型出来，规划器包在外面，两条轴的评估罩在上面。

| 阶段 | 默认选择 | 什么时候偏离 | 原因（章节） |
|---|---|---|---|
| 范式 | 面向控制的隐空间动力学或 JEPA 预测式模型 | 产品是合成数据或视频而不是控制：改用生成式视频 | [2](02-frame-as-ml-task.md) |
| 预训练数据 | 管够的被动视频，自监督 | 观测本身就是低维状态：跳过视频，直接学紧凑的动力学 | [3](03-data.md) |
| 适配数据 | 一小份带动作标注的机器人数据，遥操作加上失败的自主 rollout | 演示数据只剩下平滑近最优的轨迹：刻意掺入恢复行为的数据 | [3](03-data.md) |
| 仿真 | 中间层，提供廉价的带动作标注 rollout 和持续评估，配域随机化 | 仿真器的怪癖渗进了模型：提高真实感预算，或者随机化得更狠 | [3](03-data.md)、[5](05-evaluation.md) |
| 规划器 | MPC 里套一个批量 CEM：想象、打分、执行一个动作、重规划 | 一个反应式的学出来的策略就够用了：把规划器蒸馏掉 | [4](04-model-development.md)、[6](06-serving-and-scaling.md) |
| Horizon | 短，每步都重规划 | 任务本身就要求长的开环规划：明确为误差累积留出预算 | [6](06-serving-and-scaling.md) |
| 评估 | 永远两条轴：感知保真度和决策效用，仿真持续跑，真机按里程碑跑 | 决策效用那条轴永远不能砍。视频质量不等于控制质量 | [5](05-evaluation.md) |
| 发布门槛 | 真机成功率过线，并报出一个不大的 sim-to-real 差距 | 没有例外。单一个仿真分数不构成门槛 | [5](05-evaluation.md) |

最后两行是新手翻船的地方：一段漂亮的预测视频让人觉得有进展，
但本章的北极星是任务成功率，而唯一能卡住发布的那个数字，
是在模型从没见过的硬件上量出来的。

## 完整的方案

回到[第 1 节](01-clarifying-requirements.md)那个场景：输入是第一人称 RGB 加本体感知，
输出是末端执行器的增量，任务是机器人没有专门训练过的操作任务，
在仿真里开发，验收标准定在一个新实验室里的真机成功率上。
下面是整个系统，每个选择都已敲定，并附上它胜出的理由。

| 决策 | 选择 | 为什么它赢了 |
|---|---|---|
| 范式 | JEPA 预测式动力学，加一个动作条件化的头 | 目标是控制；预测隐变量省掉了规划器根本用不到的像素渲染 |
| 预训练 | 在被动视频层上自监督 | 物理可以靠看学会；一百万小时的视频是存在的，一百万小时的遥操作永远不会有 |
| 适配 | 在稀缺的机器人层上做动作条件化 | 只有这份数据能教会模型自己的动作会造成什么；但它小到没法从零训练 |
| 仿真层 | 域随机化的仿真器，用于 rollout 和持续评估 | 近乎零成本的无限动作标注；随机化换来的是最坏情况下的迁移能力 |
| 规划器 | 机器人 GPU 上的 MPC 循环里跑批量 CEM | rollout 之间互相独立，所以一次批量前向就能塞进控制预算 |
| Horizon | 短，每步重规划 | 预测误差会累积，长的开环规划既更慢又更错 |
| 评估 | 每次提交都在仿真里测 rollout 漂移和动作忠实度；真机成功率按里程碑测 | 两条轴是各自独立地垮掉的；只有都测才抓得住一个好看但不可控的模型 |
| 门槛 | 真机成功率过线，并报出 sim-to-real 差距 | 零样本迁移到一个新实验室，本来就是写明了的验收测试 |

**数据预算。**[第 3 节](03-data.md)的金字塔定下了各层的量：预训练用百万小时量级的被动视频，
中间是无限量的仿真器 rollout，适配用几十小时的带动作标注的机器人数据（V-JEPA 2 的比例）。
这套配方是被逼出来的，不是挑出来的：没有哪个实验室能在硬件上采出顶层那个量，
而底层又小到教不会物理，所以先预训练再适配是穿过这座金字塔的唯一路径。

**规划预算。**[第 6 节](06-serving-and-scaling.md)的开销模型：每个控制步的模型调用次数按
n 乘 horizon 乘迭代次数增长，所以 200 个采样、horizon 为 5、3 轮 CEM 迭代，
每个动作大约是 3,000 次世界模型评估。10 Hz 的控制回路只给这些留了 100 毫秒，
这正是为什么机器人本体上的模型推演的是一个小隐向量（对 200 条 rollout 做一次批量前向），
而它那个生成式视频的兄弟留在数据中心里。

**评估预算。** 仿真评估是持续跑的，在 GPU 并行仿真器上基本等于免费（一次几千个环境）；
真机试验才是稀缺资源，所以它花在里程碑和发布门槛上，不是每次提交都跑。
两个数字之间的差距（仿真成功率减真机成功率）本身也是一个被跟踪的指标：
差距在扩大，说明模型学的是仿真器，不是世界。

**第一个月里会坏的东西。** 在第一个硬件里程碑之前要接好三个信号：
仿真里的 rollout 漂移（horizon 上每一步的预测误差；它一涨，
那些在想象里打分很高的规划就开始在执行时失败）、sim-to-real 差距的走势
（差距变宽能在里程碑真的挂掉之前就报出仿真过拟合），
以及机器人上的规划器延迟（p99 超出控制预算，会先表现为动作发抖或慢半拍，
远早于有人去做性能剖析）。

## 同一套技术在不同约束下的样子

中间那一列就是上面那套方案。另外两列保持同样的接口（金字塔、动力学模型、规划器、两条轴的评估），
但几乎换掉了每一个实现选择。

| | 只跑仿真的研究 agent | 从实验室到真机的操作任务（本章） | 车队级别的车辆世界模型 |
|---|---|---|---|
| 目标 | 在仿真里刷 benchmark | 在一个新的真实实验室里零样本完成操作任务 | 预测道路场景，供策略训练和评估用 |
| 范式 | 隐空间动力学（Dreamer 一类），在回路里训练 | JEPA 预测式加动作头 | 生成式视频（GAIA / Cosmos 一类） |
| 数据 | 只有仿真器；金字塔塌缩成只剩中间层 | 完整金字塔：视频预训练，仿真中间层，机器人适配 | 大规模车队摄像头日志，加上仿真补罕见场景 |
| 规划器 | 在想象里跑 CEM 或者学出来的策略；延迟无所谓 | 机器人上 100 毫秒预算内的批量 CEM | 在线不用；模型作为数据与评估引擎离线跑 |
| 真机评估 | 没有 | 发布门槛 | 影子模式与驾驶栈对比，示意性质 |
| sim-to-real 差距 | 未定义，也不报 | 被跟踪的门槛指标 | 它本身就是产品：补上车队从没记录过的场景覆盖 |
| 什么算过度设计 | 域随机化、真机里程碑 | 在热路径上做照片级真实的生成渲染 | 把规划器放进车上的控制回路 |

从中掉出来两条教训。第一，只跑仿真那一列把最难的部分（随机化、差距跟踪、硬件里程碑）删掉了，
这恰恰是为什么一个只在仿真里得到的结果不能当作结论迁移出去：被删掉的那些部分，正是具身性所在。
第二，车辆那一列展示了[第 6 节](06-serving-and-scaling.md)的离线场景独立成立的样子：
一个慢到没法拿来规划的世界模型，依然是一支车队能买到的最便宜的罕见场景数据来源和策略评估手段。

## 每个约束各自决定什么

| 你的约束 | 它拨动的杠杆 | 经验法则 |
|---|---|---|
| 是预测还是行动 | 范式和指标 | 控制：成功率是指标，视频保真度只是代理；内容：生成式视频加保真度指标 |
| 观测空间 | 模型家族 | 像素：tokenizer 或隐空间编码器；低维状态：紧凑动力学模型，完全跳过视频 |
| 带动作标注的数据量 | 训练配方 | 稀缺（它永远稀缺）：在被动视频上预训练，用手头那一点点做适配 |
| 每步的延迟预算 | 规划器形状 | 几十毫秒：隐空间 rollout、批量 CEM、短 horizon；没有预算：离线慢慢规划或生成 |
| 任务要求的 horizon | 重规划节奏 | 误差按步累积：把 horizon 保持得短并重规划，别去信一个长的开环规划 |
| 评估面 | 门槛 | 真实硬件在回路里：以真机成功率为门槛并报出差距；只跑仿真：就明说，别对迁移下任何结论 |
| 具身形态的多样性 | 数据覆盖 | 换了机械臂或换了运动学：报清楚数据覆盖了哪些具身形态；别假定能迁移 |
| 仿真真实感的预算 | 随机化 | 比照片级真实便宜：域随机化拿平均真实感换最坏情况下的迁移 |

## 最小的可运行规划器

机器人本体上那个循环的每个部件都有一个最小版本：世界模型缩成一行动力学函数，
里面故意写错一个阻力系数（这就是每个学出来的模型都有的那份单步模型误差），
CEM 规划器缩成几行的采样、排序、重新拟合，机器人缩成一个一维质点。
形状本身就是这一节要讲的东西：在模型里想象，在世界里行动，
然后看看模型误差会把一个你信得太久的规划变成什么样。

```python
"""Open-loop planning vs MPC replanning inside an imperfect world model."""
import random

DT, GOAL = 0.2, 1.0

def true_step(pos, vel, force):
    """The real environment: a 1D point mass with drag the model gets wrong."""
    vel += (force - 0.5 * vel) * DT
    return pos + vel * DT, vel

def model_step(pos, vel, force, eps):
    """The learned dynamics: drag misestimated by eps, the per-step model error."""
    vel += (force - (0.5 + eps) * vel) * DT
    return pos + vel * DT, vel

def imagine(pos, vel, seq, eps):
    for a in seq:
        pos, vel = model_step(pos, vel, a, eps)
    return abs(GOAL - pos) + 0.1 * abs(vel)          # cost of the imagined endpoint

def cem_plan(pos, vel, horizon, eps, rng, n=60, iters=3, elite=10):
    """Cross-entropy method: sample n sequences, refit on the elite, repeat.
    Cost per control step ~ n x horizon x iters world-model calls (section 6)."""
    mean, std = [0.0] * horizon, [1.5] * horizon
    for _ in range(iters):
        seqs = [[rng.gauss(mean[t], std[t]) for t in range(horizon)] for _ in range(n)]
        seqs.sort(key=lambda s: imagine(pos, vel, s, eps))
        top = seqs[:elite]
        mean = [sum(s[t] for s in top) / elite for t in range(horizon)]
        std = [max(0.1, (sum((s[t] - mean[t]) ** 2 for s in top) / elite) ** 0.5)
               for t in range(horizon)]
    return mean

def run(horizon, eps, replan, seed=0):
    rng = random.Random(seed)
    pos, vel = 0.0, 0.0
    seq = cem_plan(pos, vel, horizon, eps, rng)
    for t in range(horizon):
        if replan and t > 0:             # MPC: fresh real observation, replan the tail
            seq = cem_plan(pos, vel, horizon - t, eps, rng)
            action = seq[0]
        else:                            # open-loop: trust the imagined trajectory
            action = seq[t]
        pos, vel = true_step(pos, vel, action)
    return abs(GOAL - pos)

print(f"final |distance to goal| after executing a horizon-length plan (goal {GOAL}):")
print(f"{'model err':>10} {'horizon':>8} {'open-loop':>10} {'MPC replan':>11}")
for eps in (0.0, 0.4, 0.8):
    for horizon in (4, 8, 16):
        print(f"{eps:>10.1f} {horizon:>8} {run(horizon, eps, False):>10.3f} "
              f"{run(horizon, eps, True):>11.3f}")
```

跑一遍，这张表用九行讲完了本章的故事。模型完美时（误差 0.0），
开环规划在每个 horizon 上都落在距目标 0.05 以内：想象和现实一致时，信任规划没问题。
一旦有模型误差，开环的偏差就随 horizon 增长（误差 0.8 时：horizon 4 是 0.098，
8 是 0.453，16 是 0.823），这就是把 rollout 误差累积摆到了明面上：
每一个想象步都以上一个想象步为条件，于是阻力估错的那点偏差不断攒起来。
MPC 重规划在每一个误差水平、每一个 horizon 上都把偏差压在接近零，
因为每一步都从一个新的真实观测出发，误差最多只有一步的时间可以累积。
这就是[第 6 节](06-serving-and-scaling.md)那条规则（短 horizon、频繁重规划）
变成了一个你可以自己重跑的实验。把 `true_step` 换成一台机器人，
`model_step` 换成一个学出来的隐空间动力学模型，`cem_plan` 换成它的批量 GPU 版本，
你就把本章的控制回路重新搭出来了。
