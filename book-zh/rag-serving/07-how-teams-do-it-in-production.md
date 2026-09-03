# 7. 真实团队在生产环境里怎么做

所有生产级 RAG 系统最后都会收敛到同一副骨架：把查询 embedding，从索引里取出候选 chunk，
对候选名单做重排，拼出一段紧凑而有依据的上下文，再让 LLM 生成带引用的答案。
离线那一半是分开的：解析、分块、embedding、建索引，外加一条给变更文档重新 embedding 的新鲜度循环。
团队之间真正的差异在于把力气花在哪儿：检索策略（往搜索本身投入多少智能）、
重排下手有多狠，以及评估做得有多认真。

## 真正的设计分歧在哪里

| 系统 | 检索策略 | 重排 | 分块与新鲜度 | 评估 | 什么情况下它最合适 |
|---|---|---|---|---|---|
| Ramp | 在 ClickHouse 里对预先算好的 NAICS 代码 embedding 做稠密检索 | 两段 prompt 的 LLM 挑选（先收窄再重新打分） | 按分类体系预先算好；很少刷新 | accuracy@k；一个模糊的层级指标 | 标签集封闭可枚举，一次性 embedding 就够 |
| Uber | 向量 + BM25 混合；Query Optimizer agent 在检索前先改写 | 后处理 agent 做去重和重新排序 | 用 LLM 做富化（每个 chunk 配摘要、FAQ、关键词）；自研 Google Docs loader | LLM-as-judge（0 到 5 分） | 查询有歧义、需要检索前改写；内部语料很乱 |
| Microsoft GraphRAG | 走知识图谱的社区遍历；图由 LLM 抽取实体建成 | 用社区预生成的摘要当上下文 | LLM 抽取的实体图；建图和刷新都很贵 | 全面性、多样性、SelfCheckGPT 忠实度 | 多跳、需要通读整个语料才能理解的问题，扁平向量检索抓不到 |
| DoorDash | 向量 RAG 加一层护栏 | LLM judge 当护栏 | 支持文档的 chunk；定期刷新 | LLM judge + 护栏分类 | 支持领域边界清晰，护栏能拦住越界回答 |
| Dropbox Dash | 词法 + 语义混合；分块推迟到查询时才做 | 在候选名单上跑更大的 embedding 模型 | 查询时分块；定期同步加 webhook | LLM judge（正确性、完整性）+ 来源的 P/R/F1 | 变化很快、要求接近实时新鲜度的个人语料 |
| Vespa | BM25 + INT8 或二值量化向量的混合 | 无（重心在 embedding 模型上） | 预先建好索引；量化按模型家族分别调 | 质量对延迟的 benchmark（召回、QPS） | 大索引受内存限制，量化就是那个杠杆 |
| NVIDIA | 第一阶段稠密向量；第二阶段 cross-encoder | Cross-encoder 微服务（NeMo Retriever NIM） | 标准的预分块 | 两阶段方案在精度与成本上的工作点 | 对成本敏感的服务场景，狠重排能压低 LLM 的 token 开销 |
| Glean | 词法 + 向量 + 企业知识图谱 + 按用户的 ACL，混合起来 | 感知权限的多信号排序 | 感知权限的爬虫；持续同步 | 感知权限的相关性 | 企业搜索，按用户的 ACL 没有商量余地 |
| Databricks | 托管向量搜索（Delta Sync）；实时服务 | 一层模型选择 | 企业数据同步 | 质量监控管道 | 已经在 Databricks 平台上、想要托管 RAG 加监控的团队 |
| MongoDB | 通过 Atlas 聚合管道做向量搜索（\$vectorSearch） | 默认没有；只做相似度打分 | 递归分块或定长带重叠；用 Voyage AI 的 embedding | 分数透明（把每篇文档的分数展示给用户） | 已经在用 Atlas、想要零运维向量搜索的团队 |
| Grab | 在一份经过审核的查询 API 目录（Data-Arks）上做 RAG，而不是在文档 chunk 上 | 无；检索单元是一条精选好的查询 | 参数化的 SQL / Python API，不是文本 chunk | 返回表格的摘要质量 | 分析师语料，检索单元是一条可复用的查询而不是一个段落 |
| Thomson Reuters | 稠密检索（MiniLM-L6-v2）；在 Milvus 里算余弦相似度 | 无 | 知识库文章 + CRM 分块；非参数化的存储 | 定性样例；出处引用 | 受监管领域（税务），知识更新不需要重新训练 |
| GitHub Copilot | 在代码仓库上做语义 + 代码搜索 | 有 | 仓库与代码建索引；活跃仓库保持新鲜 | 有依据的回答质量 | 在语义特殊的大型私有代码语料上给答案找依据 |
| Google / ETH RAGO | 服务层的调度与放置优化 | 不适用 | 不适用 | 每颗芯片的 QPS、首 token 延迟 | 从一条已有的大规模 RAG 管道里榨出吞吐和延迟 |

核心分界线是这样的：往更丰富的检索上投入的团队（混合、知识图谱，或者 agent 式的查询改写），
解决的是语料噪声大或者多跳的问题；往重排和 grounding 上投入的团队，解决的是精度和可信度的问题；
往服务层投入的团队（RAGO、Databricks），解决的是高 QPS 下的成本和吞吐问题。

## 这些系统

- **Ramp** [From RAG to Richness: How Ramp Revamped Industry Classification](https://builders.ramp.com/post/industry_classification)：embedding 模型选型，以及在 NAICS 代码上做两段 prompt 的检索，embedding 预先算好放在 ClickHouse 里。*（产品设计）*
- **Uber** [Enhanced Agentic-RAG: near-human precision for chatbots](https://www.uber.com/blog/enhanced-agentic-rag/)：检索前的查询 agent、自研 Google Docs loader、用 LLM 富化 chunk，以及 LLM-as-judge 评估。*（部署）*
- **Microsoft Research** [GraphRAG: unlocking LLM discovery on narrative private data](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/)：在多跳的私有数据查询上，知识图谱检索胜过纯向量 RAG。*（产品设计）*
- **DoorDash** [Path to high-quality LLM-based Dasher support automation](https://careersatdoordash.com/blog/large-language-modules-based-dasher-support-automation/)：带 LLM 护栏和 judge 的 RAG 客服机器人；幻觉减少了 90%。*（评估标准）*
- **Dropbox** [Building Dash: how RAG and AI agents meet business needs](https://dropbox.tech/machine-learning/building-dash-rag-multi-step-ai-agents-business-users)：词法加语义的混合检索，在延迟、新鲜度和成本之间取平衡；用来源 F1 做评估。*（部署）*
- **Vespa** [Embedding Tradeoffs, Quantified](https://blog.vespa.ai/embedding-tradeoffs-quantified/)：INT8 与二值量化，外加混合 BM25 在质量与延迟上的取舍。*（评估标准）*
- **Vespa** [Asymmetric Retrieval: spend on docs, embed queries for free](https://blog.vespa.ai/asymmetric-retrieval-spend-on-docs-queries-for-free/)：文档侧用大模型，查询侧用本地小模型，以此压低服务成本。*（部署）*
- **NVIDIA** [How a reranking microservice improves retrieval accuracy and cost](https://developer.nvidia.com/blog/how-using-a-reranking-microservice-can-improve-accuracy-and-costs-of-information-retrieval/)：先 embedding 再重排的两阶段方案；送进 LLM 的 chunk 变少，成本下来了而精度没掉。*（评估标准）*
- **Glean** [Why vector search isn't enough for enterprise RAG](https://www.glean.com/blog/hybrid-vs-rag-vector)：企业级 RAG 需要混合搜索，加上知识图谱和感知权限的排序。*（产品设计）*
- **Databricks** [Creating High Quality RAG Applications with Databricks](https://www.databricks.com/blog/building-high-quality-rag-applications-databricks)：实时服务、模型选型与评估、质量监控。*（部署）*
- **LinkedIn** [Improving Post Search at LinkedIn](https://www.linkedin.com/blog/engineering/search/improving-post-search-at-linkedin)：分层的一轮和二轮排序器，相关性、质量、新鲜度各用一个独立模型。*（部署）*
- **Pinterest** [How we built Text-to-SQL at Pinterest](https://medium.com/pinterest-engineering/how-we-built-text-to-sql-at-pinterest-30bad30dabff)：用 RAG 检索数据表，为 LLM 在数千张仓库表上生成 SQL 提供依据。*（产品设计）*
- **Cloudflare** [Introducing AutoRAG: managed RAG on Cloudflare](https://blog.cloudflare.com/introducing-autorag-on-cloudflare/)：托管管道，异步建索引、Vectorize 存储、查询时检索与生成。*（部署）*
- **Vimeo** [Unlocking knowledge sharing for videos with RAG](https://medium.com/vimeo-engineering-blog/unlocking-knowledge-sharing-for-videos-with-rag-810ab496ae59)：在转写文本分块、多种尺寸的上下文窗口和向量检索之上做视频问答。*（产品设计）*
- **Elastic** [RAG pipelines in production](https://www.elastic.co/search-labs/blog/rag-in-production)：生产规模下的混合检索、重排、监控与 benchmark。*（部署）*
- **Anyscale** [Building RAG-based LLM Applications for Production](https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-1)：用 Ray Serve 把一套 RAG 从头搭起来、评估并大规模上线。*（部署）*
- **GitHub** [What is retrieval-augmented generation?](https://github.blog/ai-and-ml/generative-ai/what-is-retrieval-augmented-generation-and-what-does-it-do-for-generative-ai/)：Copilot Enterprise 怎么靠内部代码搜索和语义检索给答案找依据。*（产品设计）*
- **Google / ETH Zurich** [RAGO: systematic performance optimization for RAG serving](https://arxiv.org/abs/2503.14649)：一套服务框架，把每颗芯片的 QPS 提到 2 倍，首 token 延迟砍掉 55%。*（部署）*
- **MongoDB** [Taking RAG to Production with the MongoDB Documentation AI Chatbot](https://www.mongodb.com/developer/products/atlas/taking-rag-to-production-documentation-ai-chatbot/)：用 Atlas Vector Search 做的文档问答机器人；分块与 embedding 模型的选择；从原型走到生产。*（部署）*
- **Grab** [Leveraging RAG-powered LLMs for Analytical Tasks](https://engineering.grab.com/transforming-the-analytics-landscape-with-RAG-powered-LLM)：Data-Arks 中间件检索经过审核的查询 API，为生成报表的机器人提供依据。*（产品设计）*
- **Mercado Libre** [Beyond the Hype: Real-World Lessons from Working with Large Language Models](https://medium.com/mercadolibre-tech/beyond-the-hype-real-world-lessons-and-insights-from-working-with-large-language-models-6d637e39f8f8)：在技术文档上做 RAG；用 LLM 生成表描述；靠 function calling 做结构化抽取。*（评估标准）*
- **Thomson Reuters** [Better Customer Support Using RAG at Thomson Reuters](https://medium.com/tr-labs-ml-engineering-blog/better-customer-support-using-retrieval-augmented-generation-rag-at-thomson-reuters-4d140a6044c3)：在领域知识上做 RAG，为受监管领域的客服回答提供依据。*（部署）*
- **Anthropic** [Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval)（2024 年 9 月）：2024 年针对"分块丢上下文"这个问题的标准解法。建索引之前，先在每个 chunk 前面拼上一段由 LLM 生成的说明，交代这个 chunk 在原文档里的位置（contextual embedding），再配上 contextual BM25；文章报告检索失败率大幅下降，和重排器结合后降得更多。只要把文档前缀缓存起来，跑一遍并不贵。*（产品设计）*

完整的案例对比（分歧图、数学推导、四象限图）见
[topics/01-rag-serving.md](../../topics/01-rag-serving.md) 里那份密度更高的参考资料。
