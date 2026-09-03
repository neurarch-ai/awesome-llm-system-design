# 7. 真实团队在生产环境里怎么做

所有上了生产的 LLM 可观测系统，最后都会收敛到同一副骨架：每个请求发一条便宜的 trace，把昂贵的质量检查异步、抽样地从这条流上分出去，把结果汇总成比率，然后在每次模型或 prompt 变更之后盯住变化量告警。公司之间真正不一样的只有三个决定：**trace 的粒度**（一步一个 span 还是一条消息一个）、**质量信号**（信哪个代理指标），以及**自建还是直接用现成的**（定制 judge 流水线 vs 自动埋点 SDK）。

## 真正的设计分歧在哪

| 系统 | trace 粒度 | 质量信号 | 自建还是采用 | 特色选择 |
|---|---|---|---|---|
| Datadog（RAG） | 一步一个 span，细到单条论断 | LLM judge（GPT-4o，两阶段）+ 确定性检查 | 自建：定制 judge + ETL | 把"自相矛盾"和"无依据"分开标记；直接引用出问题的那条论断，方便快速定位 |
| Datadog（judge 评估） | 同上，同一套 span | 对着人工标注的 HaluBench 和 RAGTruth 算 F1 | 自建 | 两阶段 prompt（先自由推理，再用小模型重排格式）做到 0.810 F1；最难的一课是：合成 benchmark 会高估真实世界的表现 |
| Honeycomb | OTel 一步一个 span（埋点只要一行） | 7 天窗口内 80% 成功率的 SLO；点赞点踩 + 错误分类体系 | 采用：OTel 自动埋点 | 把模型原始输出和修正后实际执行的查询分开记，好把模型的错和修补逻辑的错区分开 |
| Uber Genie | 请求 + 对话日志经 Kafka 进 Hive | Slack 上的四档评分（已解决 / 有帮助 / 没帮助 / 不相关）+ Michelangelo ETL LLM judge | 自建：Michelangelo ETL | 额外加了一条文档质量工作流；检索质量的天花板是源文档质量，所以这个反馈闭环改进的是知识库，不只是模型 |
| Grafana Labs | 通过 OpenLIT SDK 走 OTel（自动埋点 50 多种工具） | GenAI Evaluations 看板（幻觉、偏见、毒性标记） | 采用：OpenLIT SDK + OTLP 网关 | 成本看板是一等公民（gen_ai_usage_cost_USD_sum），TTFT 和质量并列；接入只要四行代码 |
| LangChain / LangSmith | 一次工具调用一个 span（参数、结果、错误） | 在线 LLM-judge + 代码检查 + Insights 聚类 | 自建：定制 evaluator | 把每一个发现的失败都固化成一条永久的离线评估，回退就不可能再悄无声息地重演 |
| Twilio Segment | 用 conversation id 串起来的产品分析事件 | 隐式行为信号（组件点击、业务事件）+ 参与度 | 采用：Segment SDK 规范 | 把 copilot 交互翻译成业务事件（买股票、看图表）；隐式信号很丰富，但替代不了 grounding 和 faithfulness 检查 |

分界线很简单：**数据和校准买来的是质量上限；trace 粒度和采样率买来的是成本和检测延迟。** 一份完整的答案会在这两条轴上各选一个点，并且从产品能承受的风险和观测预算出发讲清楚为什么这么选。

## 这些系统（第一方文章）

- **Datadog** [Detect hallucinations in your RAG LLM applications](https://www.datadoghq.com/blog/llm-observability-hallucination-detection/)：面向生产 RAG 应用的 span 级自相矛盾与无依据论断检测，会直接指出出问题的那条论断。*(产品设计)*
- **Datadog** [Detecting hallucinations with LLM-as-a-judge](https://www.datadoghq.com/blog/ai/llm-hallucination-detection/)：他们怎么造出并 benchmark 那个两阶段 GPT-4o judge；在 HaluBench 和 RAGTruth 上拿到 0.810 F1。*(评估标准)*
- **Honeycomb** [Improving LLMs in Production With Observability](https://www.honeycomb.io/blog/improving-llms-production-observability)：用 OTel span 记录 Query Assistant 的输入、输出、错误、延迟、token 和用户反馈；7 天窗口 80% 成功率的 SLO。*(部署)*
- **Uber** [Genie: Uber's Gen AI On-Call Copilot](https://www.uber.com/us/en/blog/genie-ubers-gen-ai-on-call-copilot/)：一个部署在 Slack 上、处理了 70,000 多个问题的 RAG copilot；Kafka 到 Hive 的管道同时喂给 Michelangelo ETL judge 和一条文档质量反馈闭环。*(产品设计)*
- **Grafana Labs** [Monitor LLMs in production with Grafana Cloud, OpenLIT, and OpenTelemetry](https://grafana.com/blog/ai-observability-llms-in-production/)：用 OpenLIT 自动埋点，把 OTLP 路由到托管的 Prometheus 和 Tempo，四行代码就能拿到成本、TTFT 和评估分数的一等看板。*(部署)*
- **LangChain** [The agent improvement loop starts with a trace](https://www.langchain.com/blog/traces-start-agent-improvement-loop)：把生产 trace 当作持续改进闭环的输入；每一个失败都固化成一条永久的离线评估，这样它就不会悄悄回退。*(部署)*
- **Twilio Segment** [Instrumenting User Insights for your AI Copilot](https://www.twilio.com/en-us/blog/insights/ai/instrumenting-user-insights-for-your-ai-copilot/)：一套标准化的"AI Copilot 规范"，用一个稳定的 id 把对话事件缝在一起，并把 UI 交互翻译成产品分析用的业务事件。*(产品设计)*

想看带数学推导和象限图的完整决策矩阵，去看这份密集参考：[topics/12-production-monitoring-and-observability.md](../../topics/12-production-monitoring-and-observability.md)。
