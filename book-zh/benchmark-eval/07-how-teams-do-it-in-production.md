# 7. 真实团队在生产环境里怎么做

认真跑 benchmark 的人最后都收敛到同一副骨架：钉住题目、标准化的 harness、原始输出留档、报告里带着协议。区别在于**他们标准化的是什么**（prompt、环境，还是统计），以及**他们把什么当成威胁**（污染、刷榜，还是不可复现）。按这两列去读下面的表，是给你遇到的任何一套评估栈定位的最快办法。

## 真正的设计分歧在哪儿

| 系统 | 标准化了什么 | 打分方式 | 对污染的态度 | 统计 | 什么时候它最合适 | 要留神的地方 |
|---|---|---|---|---|---|---|
| LM Evaluation Harness（EleutherAI） | prompt 渲染、任务定义、带版本的任务配置 | 每个任务各自用对数似然或生成式打分 | 记录风险，策略交给用户定 | 报告每个任务的标准误 | 跨很多模型做可复现的静态套件比较 | 默认配置和厂商的协议不一样，所以数字仍然需要一个协议哈希 |
| HELM（Stanford CRFM） | 多场景、多指标的报告矩阵 | 任务指标之外，还并排给出效率和校准 | 场景集公开，所以污染随时间增长 | 跨场景报告，而不是压成一个指数 | 用一个数字会误导人时的整体性比较 | 跑起来很重；场景覆盖会老化 |
| Inspect（UK AI Security Institute） | agent 循环和工具沙箱 | solver 加 scorer，包含模型评分器 | 为私有评估和安全评估而设计 | 每个样本留日志以备审计 | 需要受控环境的 agent 与安全评估 | 要做环境工程，不只是写 prompt |
| simple-evals（OpenAI） | chat 格式的生成式打分，prompt 和解析器都可读 | 生成加答案匹配 | 偏好用更新的题集，不用饱和的 | 刻意做到最少 | 把协议写得让别人看得懂、复现得了 | 有意做得极简；不是一个完整平台 |
| HealthBench（OpenAI） | 每题一套带权重的专家 rubric 标准 | 模型评分器逐条判定，并与医生做 meta-evaluation | 专门构建的题集，随 rubric 一起发布 | 评分器与专家判断的 meta-eval | 整体打分复现不了的专家级开放式领域 | 成本在构建 rubric；评分器需要重新认证 |
| Anthropic 的评估统计实践 | 标准化的是分析，不是 prompt | 任意 | 与它正交 | 聚簇标准误、配对差值、重采样、功效分析 | 判定一个报告出来的差距是不是真的 | 分析做对之后，头条差距常常变成平局，这不讨喜 |
| LMArena | 流量规模上的人类两两偏好 | 在投票上拟合 Bradley-Terry，另有风格控制版本 | 新 prompt 持续进来 | 名次区间、风格控制 | 开放式对话上的人类总体口味 | 私下的 best-of-N 提交和风格效应，影响大小有争议 |
| Artificial Analysis | 一个复合指数，外加各家服务商的价格和速度 | 聚合很多个公开 eval | 分项饱和了就换 | 在很多 eval 上做复合 | 在一个视图里看跨服务商的服务与质量权衡 | 指数会掩盖到底是哪一项动了 |
| METR | 带人类基线的任务套件，以任务时长为单位度量 | 任务成功率作为人类完成时长的函数 | 私有任务套件 | 用 logistic 拟合 50% 时间视野 | 用一个业务听得懂的单位来表达能力 | 人类基线昂贵；任务数量少 |
| LiveBench | 每月从近期来源刷新题目 | 客观标准答案，闭环里没有 judge | 污染由构造方式限定住 | 明说刷新策略 | 给任何静态套件的结果做一次污染核查 | 题池会变，跨时间比较需要钉住一个窗口 |
| LiveCodeBench | 题目都标了发布日期 | 单元测试，另有自修复和执行变体 | 严格在模型 cutoff 之后评估 | 按窗口比较 | 不带旧题污染疑问的代码能力 | 窗口怎么选会改变数字，要说清楚 |
| SWE-bench 与 SWE-bench Verified | 仓库环境和测试命令 | 在真实仓库状态上跑单元测试 | 旧 issue 是公开的，而且年头久了 | 每个实例留日志 | 贴近真实的软件工程信号 | 弱测试套件会放过错误的补丁；容器漂移 |

## 分界线

上面几乎所有差异都能用两条轴解释。**你标准化了什么**决定了你的数字能和谁比：标准化 prompt，就能和其他用同一 harness 的人比；标准化环境，就能比 agent；标准化分析，就能比"结论"。**你把什么当成威胁**决定了你的设计：如果是污染，就做时间切分；如果是刷榜，就建私有留出集并审计提交流程；如果是不可复现，就做协议哈希和确定性服务。

一份完整的面试回答，会在这两条轴上各选一个点，并从"这个数字要驱动什么决策"出发给出理由。给产品挑一个基座模型，容得下年头久的公开套件加一个内部题集；发一个模型卡上的数字，就容不下。

## 一手来源

- **EleutherAI** [Lessons from the Trenches on Reproducible Evaluation of Language Models](https://arxiv.org/abs/2405.14782) 和 [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness)：为什么 prompt 格式和打分模式会改变结果，以及一套把两者都钉住的带版本任务系统。
- **Stanford CRFM** [HELM](https://crfm.stanford.edu/helm/)：整体性的多场景、多指标评估，而不是一个头条数字。
- **UK AI Security Institute** [Inspect](https://inspect.aisi.org.uk/)：围绕 solver、scorer 和沙箱化工具环境构建的评估框架，服务于 agent 与安全评估。
- **OpenAI** [simple-evals](https://github.com/openai/simple-evals)：一个刻意做小的生成式打分 harness，它的 prompt 和解析器就是给人读的。
- **OpenAI** [HealthBench](https://openai.com/index/healthbench/) 和[论文](https://arxiv.org/abs/2505.08775)：每段对话配医生写的 rubric 标准，由模型评分，并对评分器做专家 meta-evaluation。
- **Anthropic** [Adding Error Bars to Evals](https://arxiv.org/abs/2411.00640)：把聚簇标准误、配对差值、重采样和功效分析用到评估报告上。
- **LMArena** [对 The Leaderboard Illusion 的回应](https://lmarena.ai/blog/our-response/)，以及批评本身 [The Leaderboard Illusion](https://arxiv.org/abs/2504.20879)：私下的 best-of-N 提交会对偏好排名做什么，以及影响到底有多大。
- **METR** [Measuring AI Ability to Complete Long Software Tasks](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) 和[论文](https://arxiv.org/abs/2503.14499)：带人类基线的 50% 时间视野方法论。
- **LiveBench** [livebench.ai](https://livebench.ai/) 和[论文](https://arxiv.org/abs/2406.19314)：每月刷新的题目，客观标准答案，闭环里没有 judge。
- **LiveCodeBench** [livecodebench.github.io](https://livecodebench.github.io/) 和[论文](https://arxiv.org/abs/2403.07974)：带发布日期标签的题目，在模型 cutoff 之后按窗口评估。
- **Princeton NLP** [SWE-bench](https://arxiv.org/abs/2310.06770)：用仓库自己的测试来判定真实 GitHub issue 是否被解决。
- **UIUC 及合作者** [Establishing Best Practices for Building Rigorous Agentic Benchmarks](https://arxiv.org/abs/2507.02825)：任务规范与结果验证的检查清单，并测出了广泛使用的 agent benchmark 里的高估幅度。
- **Thinking Machines Lab** [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)：完全相同的请求为什么结果不同，以及修好它的 batch 不变 kernel。
- **Epoch AI** [benchmarking hub](https://epoch.ai/benchmarks)：独立重跑的 benchmark 结果，并记录协议。

需要密集的单文件参考（同样的材料，面试串讲的形态）：[topics/16-benchmark-evaluation.md](../../topics/16-benchmark-evaluation.md)。
