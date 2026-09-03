# 7. 真实团队在生产环境里怎么做

每一个生产系统都用同一副三段式骨架把模型包起来：模型之前一道输入 guard，之后一道输出 guard，再加一个把判定翻译成动作的策略路由。公司和公司之间真正不同的是两个决定：**把安全当成一个分类问题还是一个结构问题**，以及**延迟预算逼着设计在 guard 模型上从哪里砍**。骨架是大家共有的，杠杆在那两个选择里。

## 真正的设计在哪里分岔

| 系统 | guard 模型思路 | 越狱与注入防御 | 人工审核 | 流量与延迟形态 | 什么时候它赢 | 要当心 |
|---|---|---|---|---|---|---|
| Anthropic（Constitutional Classifiers） | 输入和输出分类器，训练数据由 constitution 合成而来 | 分类器是一个独立决策；把越狱 ASR 从 86% 压到 4.4% | 红队演练；不做逐条决策审核 | 生产环境；23.7% 的算力开销 | 想扛住通用越狱，并且有一条可量化的评估底线；策略以 constitution 的形式表达 | 每个请求都要多付成本；constitution 写得太宽会推高 FRR |
| Microsoft（MSRC） | Prompt Shields（训练出来的多语言分类器）加确定性规则 | spotlighting 标出不可信内容；代码侧的动作闸门；封堵外泄向量 | 高后果动作执行前要人工批准 | 企业 agent；把安全当结构问题 | 检索内容上的间接 prompt 注入；爆炸半径在代码里被压得很小 | 设计复杂；权限模型给开发者带来摩擦 |
| Roblox | 按模态各自蒸馏并量化的 transformer 分类器（文本、语音、PII） | 覆盖 28 种语言的快速过滤器；370k RPS 的 PII 检测 | 数千名人工审核员处理细微判断和申诉；标注回流用于重训 | 文本 750k RPS，语音 8.3k RPS，每天 61 亿条消息 | 超大规模实时场景；每种模态按各自的 RPS 画像来定规模 | 需要一支庞大的人工审核队伍；蒸馏模型会漏掉边缘策略场景 |
| Meta（Llama Guard） | 指令微调的 Llama 2-7B 分类器；分类体系写在 prompt 里 | 配套的 Prompt Guard 标注注入和越狱 | 不适用（作为模型发布） | 自托管；每个请求都要付完整的 guard-LLM 延迟 | 开放、即插即用的分类器，分类体系可零样本适配 | 要付整个 7B LLM 的延迟；guard 和 LLM 共享同一类失效模式 |
| Google（ShieldGemma） | 基于 Gemma 的生成式安全分类器 | 不适用（是分类器，benchmark 上高于 Llama Guard） | 不适用（作为模型发布） | 自托管 | benchmark 上捕获率高于 Llama Guard；权重开放 | 和任何 guard-LLM 一样，自托管和每请求延迟的代价照付 |
| OpenAI（cookbook 模式） | 输入用 Moderation API；输出用 LLM 裁判式的 G-Eval 打分器 | guard 通过 asyncio 和生成赛跑；LLM 裁判继承基座模型的失效模式 | 不适用（是模式，不是服务） | 异步赛跑把 guard 延迟藏起来；LLM 裁判按请求计费 | 生成无副作用时，guard 的延迟被藏在生成后面 | 生成有副作用时，异步会在拦截触发前漏出 token；LLM 裁判是可以被说服的 |
| NVIDIA（NeMo Guardrails） | 配置驱动的 rail，把 LlamaGuard 和 AlignScore 事实核查接起来 | rail 每一轮调用一次 guard 模型；LlamaGuard 跑在独立的 vLLM 引擎上 | 不适用（是框架） | RAG 应用；vLLM guard 层独立扩容 | 声明式的 rail 加上依据检查，不用手工接线；独立的 guard 引擎让批处理成为可能 | rail 的好坏取决于背后那份 YAML 配置；输出侧的 rail 串起来会叠加延迟 |
| Cloudflare（Firewall for AI） | Llama Guard 跑在 Workers AI 的 GPU 上；零样本，13 个类别 | 只做输入；prompt 在到达源站模型之前就被拦下 | 不适用（是边缘代理） | 2 秒异步超时；边缘自动扩容 | 与模型无关；应用零改动；对任何后端一视同仁地保护 | 只做输入，所以不安全的生成会放过去；2 秒超时是一个故意选择的超时即放行 |
| Grab | 按 LLM 给出的违规可能性分数做两级路由 | 不适用（是内容审核，不是注入防御） | 不适用 | 成本驱动；清晰的案例由廉价层处理 | 把昂贵调用集中在模糊内容上，从而控制成本 | 分数没校准好，违规内容会被送进廉价层；必须调阈值 |
| Thomson Reuters（CoCounsel） | 锚定在可信来源上；不用文本毒性分类器 | 不适用（依据本身就是防御机制） | 律师审核加每晚 1,500 条测试的 benchmark | 法律领域；论准确性，依据锚定胜过内容审核 | 高风险领域，依据锚定才是对的安全模型 | 需要一个可信语料库和人在回路；比一道分类器闸门慢 |
| Salesforce（Einstein Trust Layer） | 规则加模型的混合方案；七个毒性类别 | 后置 prompt 的指令防御，加上进网关之前的 PII 掩码 | 不适用（是平台管控） | 企业场景；服务商侧零数据留存；完整审计轨迹 | 每一次 LLM 调用外面都套着 PII 掩码和零留存；用户接受、修改、拒绝都进审计日志 | 绑死在平台上；基于规则的打分对新型攻击模式会滞后 |

真正的分界线是：一个系统把安全当成**分类问题**（训练一个模型给文本打分：Anthropic、Roblox、Meta、Google、Cloudflare），还是当成**结构问题**（隔离不可信文本、在代码里给动作加闸门，或者把回答锚定在来源上：Microsoft、Thomson Reuters、Salesforce），然后在赌注超出分类器置信度的地方，叠上人工审核。

## 这些系统

一手工程文章。看它们是为了拿到面试答案里被跳过的那些东西：系统服务的是谁、产品约束是什么、评估底线定在哪、部署形态长什么样。

- **Roblox** [How Roblox Uses AI to Moderate Content on a Massive Scale](https://about.roblox.com/newsroom/2025/07/roblox-ai-moderation-massive-scale)：每秒 75 万请求量级上的多模型文本、语音和 PII 审核，实时拦截。*(部署)*
- **Roblox** [Deploying ML for Voice Safety](https://about.roblox.com/newsroom/2024/07/deploying-ml-for-voice-safety)：用机器标注的数据训练一个快速量化的语音辱骂分类器，跑到每秒 2,000 请求。*(部署)*
- **Anthropic** [Constitutional Classifiers: defending against universal jailbreaks](https://www.anthropic.com/research/constitutional-classifiers)：用合成 constitution 训练的输入输出分类器，把越狱率从 86% 压到 4.4%。*(评估标准)*
- **Microsoft (MSRC)** [How Microsoft defends against indirect prompt injection attacks](https://www.microsoft.com/en-us/msrc/blog/2025/07/how-microsoft-defends-against-indirect-prompt-injection-attacks)：纵深防御，spotlighting、Prompt Shields 检测、基于权限的缓解。*(部署)*
- **NVIDIA** [Content Moderation and Safety Checks with NeMo Guardrails](https://developer.nvidia.com/blog/content-moderation-and-safety-checks-with-nvidia-nemo-guardrails/)：用 NeMo Guardrails 的配置把 LlamaGuard 和事实核查 rail 接进 RAG 应用。*(产品设计)*
- **Meta** [Llama Guard: LLM-based input-output safeguard](https://arxiv.org/abs/2312.06674)：一个指令微调的分类器，按分类体系审核 prompt 和回复。*(产品设计)*
- **Google** [ShieldGemma: generative AI content moderation](https://arxiv.org/abs/2407.21772)：基于 Gemma 的安全分类器，benchmark 成绩高于 Llama Guard。*(评估标准)*
- **Meta** [Llama Prompt Guard 2](https://developer.meta.com/ai/docs/model-cards-and-prompt-formats/prompt-guard/)：一个轻量二分类器，用来标注 prompt 注入和越狱。*(产品设计)*
- **OpenAI** [How to implement LLM guardrails](https://developers.openai.com/cookbook/examples/how_to_use_guardrails)：异步的输入输出护栏模式，以及它们的延迟取舍。*(部署)*
- **Cloudflare** [Block unsafe prompts with Firewall for AI](https://blog.cloudflare.com/block-unsafe-llm-prompts-with-firewall-for-ai/)：一个用 Llama Guard 在边缘拦截有害 prompt 的代理，覆盖 13 个类别。*(部署)*
- **Salesforce** [Inside the Einstein Trust Layer](https://developer.salesforce.com/blogs/2023/10/inside-the-einstein-trust-layer)：围着 LLM 调用做 PII 掩码、毒性打分和 prompt 注入防御。*(部署)*
- **Grab** [How LLMs make content moderation more precise](https://www.grab.com/inside-grab/stories/how-large-language-models-help-us-make-more-precise-content-moderation-decisions/)：两级审核，按 LLM 给出的违规可能性分数路由内容。*(产品设计)*
- **Thomson Reuters** [Inside CoCounsel's guardrails](https://legal.thomsonreuters.com/blog/why-your-legal-ai-needs-more-than-the-open-web-a-look-inside-cocounsels-guardrails/)：把法律 AI 锚定在可信来源上，配律师审核和每晚 1,500 条测试的 benchmark。*(评估标准)*
- **Slack** [Securing the Agentic Enterprise](https://slack.com/blog/transformation/securing-the-agentic-enterprise)：多层 AI 护栏，强制执行用户权限并实时防御 prompt 注入。*(部署)*
- **Databricks** [Implementing LLM Guardrails for Safe GenAI Deployment](https://www.databricks.com/blog/implementing-llm-guardrails-safe-and-responsible-generative-ai-deployment-databricks)：Foundation Model API 上的安全过滤器拦截不安全的输入输出，并记录下来供审计。*(部署)*
- **Wealthsimple** [Our LLM Gateway for secure, reliable generative AI](https://engineering.wealthsimple.com/get-to-know-our-llm-gateway-and-how-it-provides-a-secure-and-reliable-space-to-use-generative-ai)：一个内部网关，脱敏 PII 并追踪外发数据，让员工安全地用生成式 AI。*(部署)*
