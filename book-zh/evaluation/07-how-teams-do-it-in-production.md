# 7. 真实团队在生产环境里怎么做

每个认真做 LLM 的团队最后都收敛到同一副骨架：离线套件卡住改动，在线回路检验这道门禁是否诚实，两者之间的差距再反过来校准套件。公司之间真正不一样的只有两个决定：**开放式输出怎么打分**（用哪个 judge、怎么验证、怎么校准），以及**拿什么当在线真值**（A/B、影子模式、金丝雀、人工放行）。架构大家都一样，杠杆在校准和线上的证据上。

## 真实设计在哪里分道扬镳

| 系统 | 离线信号 | judge 校准 | 在线证据 | 门禁方式 | 什么时候占优 | 要小心什么 |
|---|---|---|---|---|---|---|
| DoorDash（聊天机器人） | 模拟的多轮对话 | 用人工评分校准过的 LLM judge | 发布前的质量线 | 模拟飞轮在发布前卡关 | 上线前拿不到多少真实流量的多轮 agent | 模拟器会偏离真实用户行为；模拟跑通了可能高估了成熟度 |
| DoorDash（AutoEval，搜索） | 整页相关性打分 | 微调过的 LLM 评分器，人在回路 | 搜索结果页质量 | 由人在回路的评分卡关 | 逐条标签抓不住整体观感的整页相关性 | 商品目录和 query 分布一变，微调评分器就得重训 |
| GitHub Copilot | 4000 多个离线测试，约 100 个坏仓库 | 对话用 LLM judge，另加人工评审 | 在真实 Hubbers 上跑内部金丝雀 | CI 门禁，每天跟生产版本做回归 | 有可执行通过或失败信号的、大而稳定的代码任务套件 | 几千条用例维护起来很贵；坏仓库这类 fixture 会过时 |
| Spotify | 把离线评估当成实验前的漏斗 | judge 跟 A/B 结果不符时就重新校准 | A/B 的用户结果 | 评估先筛一遍再进 A/B；差距驱动重新校准 | A/B 量很大、离线能先便宜地筛掉候选的场景 | 需要足够的 A/B 吞吐才能校准；离线与在线的差距必须一直盯着 |
| Thomson Reuters | 公开 benchmark + 半自动的任务评估 | 人类 A/B 作为最终裁决 | 人类偏好 A/B | 三级门禁，人工放行 | 出错代价很高的高风险专家领域（法律） | 人工放行又慢又贵，撑不起每天的改动 |
| Uber（uReview） | 给生成的评论打分 | 带置信度分数的 LLM 评分器 | 已发出评论的有用程度 | 置信度阈值决定哪些能发出去 | 可以把低置信输出压掉的大批量生成场景 | 置信度分数需要校准；阈值定歪了要么淹没要么饿死输出 |
| Discord | critic LLM 审 prompt | critic LLM 辅助人工评审 | A/B 发布的结果 | A/B 之前先过一遍 critic | prompt 迭代很快、由 critic 廉价地抓住明显回归的场景 | critic 只是建议，不是硬门禁；差的 prompt 照样能走到 A/B |
| Ramp | 给 agent 的分类结果配 judge | LLM judge 对比人工标签 | 真实流量上的影子模式 | 发布前跑影子加 judge | 那些可以先在真实流量上静默运行、之后再开放的 agent 动作 | 影子模式需要流量和能镜像的基础设施；此时还拿不到用户信号 |
| GitLab Duo | CEF prompt 库，每日回归 | 大规模跑 LLM judge | 跨多轮迭代的模型对比 | 每天自动回归 | 在同一套共享框架下比较很多模型和 prompt | judge 一漂移就同时打到所有团队；框架是一笔前置投入 |
| Booking.com | 黄金数据集 | 用 LLM-as-judge 做监控 | 生产环境的质量监控 | judge 加黄金集盯着漂移 | 稳定任务上的持续生产监控，用来抓漂移 | 黄金集会过时；监控是在东西发出去之后才抓到漂移 |
| Pinterest | 微调过的开源相关性 judge | 跟人工比 73.7% 精确匹配（XLM-RoBERTa） | 搜索排序的 A/B 实验 | 按品类分层 + FDR 控制 | 大批量搜索排序，微调 judge 胜过通用模型 | query 分布和商品目录一变，微调 judge 就得重训 |

## 分界线在哪

分界线很简单：**judge 的校准和线上的证据买的是门禁的可信度；黄金集和任务指标买的是成本效率。** 一个完整的回答要在这两条轴上各选一个点，并且从任务的可核对性（不用 judge 能不能测？）和改动的爆炸半径（真发出去一个回归有多糟？）出发说明理由。

## 这些系统（一手资料）

- **DoorDash** [A Simulation and Evaluation Flywheel to Develop LLM Chatbots at Scale](https://careersatdoordash.com/blog/doordash-simulation-evaluation-flywheel-to-develop-llm-chatbots-at-scale/)：模拟的多轮对话由一个用人工校准过的 LLM judge 在发布前打分。
- **DoorDash** [How DoorDash leverages LLMs to evaluate search result pages](https://careersatdoordash.com/blog/doordash-llms-to-evaluate-search-result-pages/)：AutoEval，用微调过的 LLM 评分器加人在回路来判断整页相关性。
- **GitHub** [How we evaluate AI models and LLMs for GitHub Copilot](https://github.blog/ai-and-ml/generative-ai/how-we-evaluate-models-for-github-copilot/)：4000 多个离线测试加上修坏仓库，对话用 LLM-as-judge，每天跟生产版本跑回归。
- **Spotify** [Better experiments with LLM evals: a funnel, not a fork](https://engineering.atspotify.com/2026/5/better-experiments-with-llm-evals-a-funnel-not-a-fork)：把离线评估按在线 A/B 校准，串成一条顺序漏斗。
- **Spotify** [Profile-aware LLM-as-a-Judge for Podcasts](https://research.atspotify.com/2025/9/profile-aware-llm-as-a-judge-for-podcasts-a-better-middle-ground-between)：用一个 LLM judge 在离线指标和昂贵的 A/B 之间架桥，评播客质量。
- **Thomson Reuters** [Efficiently evaluating LLMs for legal tasks](https://legal.thomsonreuters.com/blog/evaluating-llms-legal-tasks/)：三级门禁，公开 benchmark、半自动任务评估，最后是人类 A/B 放行。
- **Uber** [uReview: scalable, trustworthy GenAI for code review](https://www.uber.com/us/en/blog/ureview/)：一个 LLM 评分器给生成的评论打分，置信度阈值决定哪些能发出去。
- **Discord** [Developing Rapidly with Generative AI](https://discord.com/blog/developing-rapidly-with-generative-ai)：A/B 发布之前，用 critic LLM 做 AI 辅助的 prompt 评估。
- **Ramp** [How Ramp Fixes Merchant Matches with AI](https://builders.ramp.com/post/fixing-merchant-classifications-with-ai)：影子模式加一个 LLM judge，在发布前把 agent 的分类结果跟人工比对。
- **Microsoft** [LLM-Rubric: a multidimensional, calibrated approach to automated evaluation](https://www.microsoft.com/en-us/research/publication/llm-rubric-a-multidimensional-calibrated-approach-to-automated-evaluation-of-natural-language-texts/)：一个经过校准的多维 rubric judge，用来预测对话系统的人类满意度。
- **GitLab** [Developing GitLab Duo: validating and testing AI models at scale](https://about.gitlab.com/blog/developing-gitlab-duo-how-we-validate-and-test-ai-models-at-scale/)：一套中心化的评估框架带 LLM judge，每天跨几十个功能跑回归。
- **Booking.com** [LLM Evaluation: practical tips at Booking.com](https://mlops.community/blog/llm-evaluation-practical-tips-at-bookingcom)：LLM-as-judge 加黄金数据集，用于持续的生产质量监控。
- **Pinterest** [LLM-Powered Relevance Assessment for Pinterest Search](https://medium.com/pinterest-engineering/llm-powered-relevance-assessment-for-pinterest-search-b846489e358d)：微调的 XLM-RoBERTa judge 给搜索相关性打标，用来大规模评估排序 A/B 实验。
- **Honeycomb** [So we shipped an AI product. Did it work?](https://www.honeycomb.io/blog/we-shipped-ai-product)：上线后靠激活率和采用率指标做产品评估，离线门禁很轻。
- **Instacart** [Scaling Catalog Attribute Extraction with Multi-modal LLMs](https://company.instacart.com/tech-innovation/scaling-catalog-attribute-extraction-with-multi-modal-llms)：LLM-as-judge 的自动评估和人工审核员一起监控属性抽取的质量。
- **LinkedIn** [How we engineered LinkedIn's Hiring Assistant](https://www.linkedin.com/blog/engineering/ai/how-we-engineered-linkedins-hiring-assistant)：一套质量框架把产品策略和给连贯性、事实性打分的 LLM judge 配在一起。
- **Wayfair** [How AI understands what you're looking for](https://www.aboutwayfair.com/careers/tech-blog/smarter-shopping-starts-here-how-ai-understands-what-youre-looking-for)：用 LLM-as-judge 的验证任务定期离线评估 AI 生成的用户兴趣。

完整参考资料（分歧的数学、象限图、全部案例）见
[topics/06-evaluation-system.md](../../topics/06-evaluation-system.md)。
