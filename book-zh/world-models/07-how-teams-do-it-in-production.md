# 7. 真实团队在生产环境里怎么做

2024 到 2025 年间，世界模型从研究演示变成了一个有名字的产业方向：世界动作模型。
下面是几家主要力量分道扬镳的地方，附一手资料。

## Meta：V-JEPA 2（在表示空间里做预测）

Meta 的 V-JEPA 2 是一个自监督的视频世界模型，预测发生在 embedding 空间而不是像素上。
它在大约一百万小时的互联网视频上预训练，然后用少得多的无标注机器人录像（几十小时量级）做适配，
之后它的动作条件化版本就能在一个新实验室里零样本地用模型预测控制规划真实机器人的动作。
Meta 还发布了一批物理推理 benchmark（IntPhys 2、Minimal Video Pairs、CausalVQA），
衡量的是理解而不只是生成。这是本章所讲的"在视频上预训练、在机器人上适配"这套配方
最清晰的一个公开实例。
来源：[V-JEPA 2 (arXiv:2506.09985)](https://arxiv.org/abs/2506.09985)。

## NVIDIA：Cosmos 与 GR00T（既生成也行动）

NVIDIA 把这个领域框定为**世界动作模型**：先有一个预训练的生成式世界模型（想象），
再微调成一个动作模型（行动）。Cosmos 世界基础模型生成有物理依据的视频，服务于两个生产任务，
合成训练数据和仿真里的机器人策略评估，同时充当下游 WAM 的骨干；Isaac GR00T 则是
人形机器人上的视觉语言动作模型。这就是第 6 节说的离线引擎视角：世界模型作为批量数据生成器
和评估引擎的价值，往往比作为机器人本体上的规划器更大。
[Cosmos-Drive-Dreams (arXiv:2506.09042)](https://arxiv.org/abs/2506.09042) 是第一个任务的
具体样本：用世界基础模型规模化地生成驾驶数据，也是目前关于"世界模型作为数据引擎"到底产出
什么最清楚的一份公开记录。
来源：[NVIDIA Cosmos](https://www.nvidia.com/en-us/ai/cosmos/)、
[The Rise of World-Action Models](https://developer.nvidia.com/blog/pretrained-to-imagine-fine-tuned-to-act-the-rise-of-world-action-models/)、
[Cosmos-Drive-Dreams (arXiv:2506.09042)](https://arxiv.org/abs/2506.09042)。

## DeepMind：Genie（动作可控的生成世界）

DeepMind 的 Genie 系列生成的是*能玩*的环境：Genie 从无标注视频里学会了动作可控的世界生成
（arXiv:2402.15391），Genie 2 则能从一张图生成多样的、可玩的 3D 世界。重点在交互性，
一个 agent 可以在里面动作的生成世界，这让它更像一片训练和评估的场地，而不是机器人本体上的控制器。
来源：[Genie (arXiv:2402.15391)](https://arxiv.org/abs/2402.15391)。

## Wayve：GAIA（一个特定领域的驾驶世界模型）

Wayve 的 GAIA-1 是面向自动驾驶的生成式世界模型，以视频、文本和动作为条件预测未来的驾驶场景。
它是*领域专用*世界模型最清晰的例子：比通用视频模型窄，但针对某一种具身形态真正在意的状态和动作
调过。
[GAIA-2 (arXiv:2503.20523)](https://arxiv.org/abs/2503.20523) 是它的后继，把这一点讲得更透：
它是可控的、多视角的，而这正是一个驾驶栈真正需要的东西，因为一个你没法指定、也没法从每个
摄像头看到的场景，不构成一个测试用例。
来源：[GAIA-1 (arXiv:2309.17080)](https://arxiv.org/abs/2309.17080)、
[GAIA-2 (arXiv:2503.20523)](https://arxiv.org/abs/2503.20523)。

## Physical Intelligence 与 VLA 这条线

视觉语言动作模型走的是端到端路线：把观测加语言目标直接映射成动作。OpenVLA 是开源的参照系
（arXiv:2406.09246），Physical Intelligence 的 pi-0 则是一个被广泛引用的机器人基础策略。
这些东西今天就能输出动作；世界动作模型这套说法要加的，是在上面再叠一个显式的预测模型，
好让策略能规划，而不只是反应。
来源：[OpenVLA (arXiv:2406.09246)](https://arxiv.org/abs/2406.09246)、
[Physical Intelligence](https://www.physicalintelligence.company/)。

## 2026 年的转向：实时与可交互

本章的服务那一节把世界模型当作离线引擎来处理，在一次 rollout 每步要几秒的时候，这是对的
默认值。2026 年有一条工作线正是在攻这个约束，它会改变上面哪些设计能在线上跑：

- **流式可交互生成。** [Matrix-Game 3.5
  (arXiv:2608.29910)](https://arxiv.org/abs/2608.29910) 面向实时流式可交互世界模型，用
  patch memory 保住上下文，而不必在每一帧上重新付这笔钱。对一个服务侧的回答来说，值得注意的
  是那个记忆结构，不是生成器：这跟 KV cache 在文本上解决的是同一个问题，只是换了地方出现。
- **全栈开源实现。** [minWM (arXiv:2605.30263)](https://arxiv.org/abs/2605.30263) 和
  [Nano World Models (arXiv:2605.23993)](https://arxiv.org/abs/2605.23993) 把最小的端到端
  实现公开出来，这才让这个领域的延迟数字变成可以核对的，而不只是可以引用的。
- **把可复现的评估做成平台。** [stable-worldmodel
  (arXiv:2605.21800)](https://arxiv.org/abs/2605.21800) 是一个面向可复现世界模型研究与评估的
  平台，和 [Inspect](https://inspect.aisi.org.uk/) 对 agent 做的是同一件事：把测试环境固定
  下来，让一个数字衡量的是模型而不是搭法。

这些延迟声明要按第 6 节说的办法看待：先问动作空间和预测步长是多少，以及那个数字是单步还是
一整段 rollout。这套文献里的"实时"通常指的是可交互的逐帧生成，不是在硬件上闭环的规划回路。

## 哪些地方一致，哪些地方分歧

- **一致的地方：** 宽泛地预训练（视频），窄范围地适配（机器人）；在仿真里用具身任务评估，
  到里程碑时上真实硬件。
- **状态表示上的分歧：** 用像素（Cosmos、Genie、GAIA）换保真度和合成数据，
  还是用 embedding（V-JEPA 2）或紧凑隐变量（Dreamer 系列）换机器人本体上便宜的规划。
- **角色定位上的分歧：** 把世界模型当成*离线*的数据与评估引擎（NVIDIA），
  还是当成机器人上*在线*的规划器（Meta 动作条件化的 V-JEPA 2、Dreamer 系列）。

这个领域跑得快，想要一份持续更新的地图，社区阅读清单
[Awesome-WAM](https://github.com/OpenMOSS/Awesome-WAM) 跟踪世界动作模型的论文，
以及第 5 节提到的那些具身评估 benchmark（比如 WorldArena）。
