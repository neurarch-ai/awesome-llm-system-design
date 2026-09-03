# 7. 真实团队在生产环境里怎么做

所有上了生产的后训练流水线最后都收敛到同一副骨架：从一个开源或者厂商提供的基座出发，
整理一份小而精的数据集，跑监督微调（通常是一个 LoRA 或 QLoRA adapter，偶尔是全量微调），
可选地再加一段偏好优化，然后在留出评估上过门禁，最后才上线。
各家真正的差别只在三个决定上：**适配要多深**（LoRA 还是全量微调）、
**要不要在 SFT 之上再做对齐**（DPO、RLHF，还是什么都不做），
以及**训练信号从哪来**（人工标注、合成数据，还是生产日志）。
大多数团队只上 SFT；偏好调优只出现在某个 SFT 抓不住的质量轴确实要命的地方。

## 真实设计的分歧在哪里

| 系统 | 适配方式 | SFT 之上的对齐 | 数据整理的重点 | 关键结果 | 评估门禁 |
|---|---|---|---|---|---|
| Grammarly CoEdIT | 指令微调（FLAN-T5 770M 到 11B） | 只做 SFT | 密集的编辑任务指令集，同一意思有多种改写说法 | 参数量少 12 到 60 倍，效果超过 GPT-3 | 相对通用 LLM 的人工偏好评测 |
| Anyscale 迭代式 DPO | 全量微调（LoRA r=64 漂出了分布） | SFT 加迭代式 DPO；用 LLM-as-judge 造合成偏好 | 采样出来的摘要 + 70B 裁判；偏好规则对齐到评估轴上 | 在准确率和压缩率上 Pareto 优于 GPT-4o | 问答准确率加压缩比 |
| Shopify Flow | 全量微调 Qwen3-32B（FSDP，2 台 H200） | 只做 SFT | 反向工程出来的合成工作流，用 Python DSL 表达而不是原生 JSON | 比前沿模型便宜 68%；1% 流量切片暴露出 35% 的激活率差距，被飞轮抹平 | 每周一次、用人工标注校准过的 LLM 裁判，加线上激活率 |
| Mercari 属性抽取 | QLoRA 4-bit gemma-2b-it，单张 A100 | 只做 SFT | 前 20 个品类的模板化 (listing, instruction, keys, values) 数据对 | BLEU 高出 GPT-3.5 五分以上；成本约低 14 倍 | 留出商品列表上相对 GPT-3.5 的 BLEU |
| Grab 多语言 OCR | 先用 LoRA 热启动，再全量微调 Qwen2-VL 2B | 只做 SFT | 合成 OCR 数据（Common Crawl）；真实证件靠 Documint 自动标注 | 泰语 +70pp，越南语 +40pp；蒸馏出的 1B 模型追平 2B | 按文字系统分别统计的抽取准确率 |
| LinkedIn EON | 多任务指令微调（Llama 3.1 8B/70B）；prompt 精简 30% | SFT 加 RLHF 和 DPO；为有害场景合成安全输出 | 约 2 亿 token 的多样化指令 + 推理轨迹 + Economic Graph 领域数据 | 比 GPT-4o mini 高 4pp；比 GPT-4 便宜 75 倍 | 候选人与职位的匹配准确率 |
| Cloudflare Workers AI | 共享基座上挂客户自己的 LoRA adapter（Llama 2 / Mistral / Gemma） | 不适用（这是推理服务） | 每个客户的 adapter 最大 100MB，rank 最高 8 | 多租户共享一个热基座，adapter 毫秒级切换 | 多 LoRA 边缘服务的正确性 |
| Spotify AQE | 拒绝采样 SFT 然后 DPO | 拒绝采样 SFT 加在下游检索排名上做的 DPO | 点击数据和标注过的 query-document 对；偏好来自真实搜索排名 | top-1 30.8% vs 28.5%；服务时的 query 扩展快约 70% | Natural Questions 上的 top-1 准确率 |

分界线很简单。**适配深度**由行为相对基座的偏移量决定（小幅度的微调就停在 LoRA/QLoRA；
偏移很大或者有跑出分布的风险，就得上全量微调）。**SFT 之上的对齐**只在某个真实的质量轴
没法靠模仿正面样本表达出来时才出现（安全对齐、比较型偏好、两条都说得通时的语气选择）。
**数据来源**就看这个团队实际能标出什么，凡是人工标注太慢或太贵的地方，就由 LLM-as-judge 来补上。

## 这些系统（一手资料）

- **Grammarly** [CoEdIT: state-of-the-art text editing with fewer parameters](https://www.grammarly.com/blog/engineering/coedit-text-editing/)：密集的任务专用指令微调，用少 12 到 60 倍的参数打赢通用 LLM。
- **Anyscale** [Fine-Tuning LLMs: LoRA or Full-Parameter?](https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter-an-in-depth-analysis-with-llama-2)：按任务类型给出 LoRA 和全量微调的精度取舍。
- **Anyscale** [Direct Preference Optimization with Synthetic Data](https://www.anyscale.com/blog/direct-preference-optimization-with-synthetic-data)：用合成偏好做迭代式 DPO，异步参考模型，以及和裁判对齐的评估。
- **Shopify** [Flow generation through natural language: an agentic modeling approach](https://shopify.engineering/fine-tuning-agent-shopify-flow)：微调出来的 Qwen3-32B agent，配一个每周重训的 LLM 裁判飞轮。
- **Shopify** [Leveraging multimodal LLMs for the global catalogue](https://shopify.engineering/leveraging-multimodal-llms)：为商品目录抽取微调的小型 VLM，每天跑 4000 万次推理。
- **Mercari** [Fine-Tuning an LLM to Extract Dynamically Specified Attributes](https://engineering.mercari.com/en/blog/entry/20240913-fine-tuning-an-llm-to-extract-dynamically-specified-attributes/)：QLoRA 调出来的 2B 模型打赢 GPT-3.5，成本低 14 倍。
- **Grab** [A custom vision LLM to improve document processing](https://engineering.grab.com/custom-vision-llm-at-grab)：先 LoRA 后全量微调 Qwen2-VL，做多语言 OCR。
- **LinkedIn** [How we built domain-adapted foundation GenAI models](https://www.linkedin.com/blog/engineering/generative-ai/how-we-built-domain-adapted-foundation-genai-models-to-power-our-platform)：基于 Llama 的 EON，指令微调加 RLHF/DPO；比 GPT-4 便宜 75 倍。
- **Cloudflare** [Running fine-tuned models on Workers AI with LoRAs](https://blog.cloudflare.com/fine-tuned-inference-with-loras/)：在共享基座上做客户 adapter 的多 LoRA 边缘服务。
- **Spotify** [Optimizing Query Expansions via LLM Preference Alignment](https://research.atspotify.com/2025/7/optimizing-query-expansions-via-llm-preference-alignment)：拒绝采样 SFT 加 DPO 对齐一个做 query 扩展的 LLM；服务时快 70%。
- **Meta** [How to fine-tune: focus on effective datasets](https://ai.meta.com/blog/how-to-fine-tune-llms-peft-dataset-curation/)：SFT 和 PEFT 的数据整理准则；质量重于数量。
- **GitHub** [Building a faster, smarter Copilot with a custom model](https://github.blog/ai-and-ml/github-copilot/the-road-to-better-completions-building-a-faster-smarter-github-copilot-with-a-new-custom-model/)：为代码补全做的中期训练加 SFT（fill-in-middle）加 RL。
- **Uber** [Open Source and In-House: How Uber Optimizes LLM Training](https://www.uber.com/us/en/blog/open-source-and-in-house-how-uber-optimizes-llm-training/)：自研技术栈，覆盖 LoRA、QLoRA、全量微调和继续预训练。
- **Hugging Face** [Preference Tuning LLMs with Direct Preference Optimization Methods](https://huggingface.co/blog/pref-tuning)：DPO、IPO 和 KTO 的实测对比；beta 主导最终结果。
- **Databricks** [A Practical Guide to LLM Fine Tuning](https://www.databricks.com/blog/llm-fine-tuning)：端到端的生命周期，指标、数据质量、LoRA 优先，以及重训节奏。
- **DeepSeek** [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948)（2025 年 1 月）：2025 年重塑后训练格局的那次转向。推理能力是靠大规模 RL 针对**基于规则、可验证的奖励**（答案检查器、单元测试）训出来的，用的是 GRPO，即 PPO 的一个免 critic 变体，几乎不用甚至完全不用 SFT。对一份现代答案来说，结论是：只要奖励可验证（数学、代码），RL 就比再多收集偏好标注更划算。
- **Ai2** [Tulu 3: pushing frontiers in open language model post-training](https://allenai.org/blog/tulu-3-technical)（2024 年 11 月，[论文](https://arxiv.org/abs/2411.15124)）：一份完全开放的四阶段后训练配方，数据整理、SFT、DPO，然后是**基于可验证奖励的 RL（RLVR）**，它沿用 RLHF 的目标函数，只是把奖励模型换成了一个验证函数。想知道怎么把 SFT、DPO 和可验证奖励 RL 装进同一条流水线，参考它。

完整的对比表、数学推导和全部案例研究，见
[topics/05-post-training-pipeline.md](../../topics/05-post-training-pipeline.md) 里那份密度更高的参考资料。
