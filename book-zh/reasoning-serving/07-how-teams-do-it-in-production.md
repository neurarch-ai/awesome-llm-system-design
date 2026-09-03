# 7. 真实团队在生产环境里怎么做

每一套服务思考型模型的技术栈，最后都收敛到同一副骨架：一个预算旋钮，一套决定谁拿
多少的策略，再加上某种判断答案对不对的手段。区别在于**预算是在哪一层控制的**（厂
商 API、你自己的调度器，还是 prompt 里），以及**由什么来扮演验证器**，这两个选择基
本上就解释了设计的其余部分。

## 真实设计在哪里分道扬镳

| 做法 | 预算控制点 | 选择或验证 | 延迟姿态 | 什么时候它赢 | 要当心什么 |
|---|---|---|---|---|---|
| 厂商的 effort 参数（托管的推理 API） | 每请求一个 effort 或 thinking token 设置 | 自带的没有，得你自己加 | 排队在厂商那边；你只看得到总时长 | 上手快，不用做服务端工程 | 对尾部几乎没有可见性；预算的语义各家不一样 |
| 自建加调度器 | 硬性 token 封顶，外加按预算等级划分的队列策略 | 你建什么就是什么 | 尾部归你管，也修得动 | SLO 严格、流量混杂、要控成本 | 服务端工程量：抢占、优先级、长度预测 |
| 在 prompt 层面强制预算 | 压住或者注入"思考结束"标记 | 一般是 self-consistency | 跟底座模型一样 | 没有预算参数可用，或者想要更细的控制 | 强行给模型加一个它没被训练过的预算，可能让输出变差（[s1](https://arxiv.org/abs/2501.19393)） |
| best-of-n 配一个执行器 | 固定的 k，并行 | 单元测试、编译器、跑 SQL | 取决于最慢的那个样本 | 代码和查询生成 | 沙箱容量会变成瓶颈；测试太弱会放过错误答案 |
| 级联配一个检查器 | 先走便宜的路，再升级 | 执行器、schema 检查，或者经过认证的裁判 | 双峰：多数请求很快，升级的很慢 | 混合负载，且大多数请求本来就能解 | 流量一变就可能出现升级风暴 |
| 过程监督式的选择 | 固定的 k，并行 | 一个步骤级奖励模型 | 验证本身还要再花 token | 长推导，且错误能定位 | 验证可能比生成还贵 |
| RL 训出来的推理模型，直接裸服务 | 模型默认是什么就是什么 | 没有 | 完全听凭长度分布摆布 | 研究和评估场景 | 这不是一个生产姿态：没有封顶，没有策略，也没有记账 |

## 分界线

两个问题就能给任何一个设计定位。**尾部归谁管？** 如果归厂商，那你是拿控制权换上手
速度，手里只剩 effort 参数和请求级超时两根杆子。如果归你，你就能把队头阻塞真正修
掉，而且你也必须修。**验证器是什么？** 有执行器，级联和 best-of-n 就既便宜又可信；
用学习得来的奖励模型，它们能做，但会被钻空子；完全没有验证器，那串行思考就是你唯
一的杠杆，多买的那些样本根本用不上。

一个完整的回答会把这两点都点名，并且跟负载对上：可验证的任务应该走级联，拿执行器
去卡；不可验证的任务应该跑一个测量过的固定预算，配上强制作答的边界；两边都要按每
个解决任务的成本来报。

## 一手来源

- **DeepSeek** [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)：用可验证奖励训练推理模型的公开记录，也是那些推理期行为的来源。
- **斯坦福及合作者** [s1: Simple test-time scaling](https://arxiv.org/abs/2501.19393)：推理期的预算强制，包括通过压住"思考结束"token 来逼模型多想一会儿。
- **Google DeepMind 与加州大学伯克利分校** [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314)：按题目难度自适应地分配推理算力，效果好过一个固定设置。
- **斯坦福** [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling](https://arxiv.org/abs/2407.21787)：重复采样能让覆盖率陡增，但只有在你有办法挑出对的那个样本时才兑现得了。
- **OpenAI** [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)：在多次推理尝试之间做选择时，过程监督胜过结果监督。
- **OpenAI** [reasoning guide](https://platform.openai.com/docs/guides/reasoning)：effort 参数，以及 reasoning token 是怎么计费和计数的。
- **Anthropic** [extended thinking](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking)：每请求一个显式的 thinking token 预算。
- **Google** [thinking in the Gemini API](https://ai.google.dev/gemini-api/docs/thinking)：thinking 预算，以及在延迟敏感的调用里怎么把它关掉。
- **METR** [Measuring AI Ability to Complete Long Software Tasks](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/)：把能力表述成"模型能做完多长的任务"，这正是预算问题的需求侧。
- **UIUC 及合作者** [Establishing Best Practices for Building Rigorous Agentic Benchmarks](https://arxiv.org/abs/2507.02825)：弱的结果校验是怎么把分数抬高的，这和级联里接受测试太弱是同一种失效模式。

想要那份密集的单文件参考（同样的材料，按面试走查的形式组织）：
[topics/18-reasoning-and-test-time-compute.md](../../topics/18-reasoning-and-test-time-compute.md)。
