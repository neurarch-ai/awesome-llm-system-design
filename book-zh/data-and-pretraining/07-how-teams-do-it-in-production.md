# 7. 真实团队在生产环境里怎么做

任何一次认真的预训练，最后都会收敛到同一条标准漏斗：原始网页归档加上精选语料，流经抽取、语言识别、质量过滤、去重、去污染、混合、分词，然后喂给一次按 scaling law 定好尺寸的分布式 next-token 预测训练。真正有差别的是各家把投入砸在哪一层：决定能力上限的抽取与去重配方，决定 token 经济账的分词器，决定模型尺寸的规模扩展决策，还是决定这次训练到底跑不跑得起来的并行方案与容错。

## 真实流水线在哪里分道扬镳

| 系统 | 层次 | 关键杠杆 | 什么时候赢 | 要留神的地方 |
|---|---|---|---|---|
| HuggingFace FineWeb | 数据准备 | 用消融实验挑出来的启发式规则 + 教育性分类器（FineWeb-Edu）；在 96 个 CC dump 内部和跨 dump 做 MinHash 去重，得到 15T token | 想要一份胜过既有公开数据集的开放预训练数据；在难 benchmark 上，token 少而精更占优 | 去污染是关键；去重过头会掉分，所以"按 dump 去重 + 一次实测过的全局去重"胜过一味最大化全局去重 |
| TII RefinedWeb | 数据准备 | 精细的 WARC 重抽取 + URL 黑名单 + MinHash 去重；纯网页，不掺精选语料 | 想证明只要处理得足够好，光靠网页数据就能追平精选语料 | 整个论点都压在抽取质量上；在窄领域里精选语料仍然有价值 |
| Ai2 Dolma / OLMo | 数据准备 + 开放基础模型 | 3T token 的完全开放语料，工具链有完整文档；训练代码、日志、checkpoint 全开放 | 想把数据整理当成一门科学来研究；想要一次真正可复现的预训练 | 完全开放数据是法律和安全上的承诺；你放了什么进去，全部留在明面上 |
| EleutherAI The Pile | 数据准备 | 22 个精选的多样领域，800 GB | 看重领域多样性胜过原始网页规模；早期"混合比体量更重要"的一个论证 | 年代较早；没有做网页规模的过滤；在容量上已被 FineWeb 级别的数据集取代 |
| Google C4 | 数据准备 | 用启发式规则清洗过的 Common Crawl（简单可复现的 Gopher / C4 规则） | 想要一份干净、可复现的网页语料基线 | 只有启发式规则，没有学习式过滤器；上限就是规则本身的质量 |
| Meta CCNet | 数据准备 | 行级去重 + 按语言分别做的 LM perplexity 质量过滤 | 多语言和低资源语言，全局过滤器会把非英文文本淹掉的场景 | 行级重建的复杂度；perplexity 分数需要按语言各自校准 |
| Google SentencePiece | 分词器 | 与语言无关的子词切分（BPE 或 unigram LM）；把空白符也当成 token | 多语言，以及没有空格分隔的语言（中文、日文、泰文） | 词表大小与 fertility 的取舍：词表越大，embedding 和 softmax 越大 |
| Google DeepMind Chinchilla | 规模扩展决策 | 算力最优的 scaling 研究（400 多个模型）；大约每参数 20 个 token | 算力预算固定，目标是在给定 loss 下最小化训练算力 | 训练最优不管推理成本；服务负载占主导时，该过度训练一个更小的模型 |
| Meta Llama 3 | 完整方案 | 精细的数据整理 + dense 预训练（GQA、RoPE、RMSNorm）+ 分阶段上下文扩展 + 弹性训练 | 想要一个 8B 到 405B 的强开放基础模型；对集群规模下的故障和恢复讲得很坦率 | 405B 的预训练是实验室级别的；多数团队应该改造 Llama 3，而不是照抄那次训练 |
| DeepSeek-V3 | 预训练（MoE） | 总参数 671B，每 token 激活 37B；FP8 训练；无辅助 loss 的负载均衡 | 想在受限的算力预算下，用很小的每 token 算力拿到前沿级容量 | 专家并行会带来 all-to-all 流量；每个专家仍然要占着显存 |
| NVIDIA Megatron-LM | 系统 | 张量并行和流水线并行；把权重矩阵切到多张 GPU 上 | 一层或者整个模型大到单张 GPU 放不下 | TP 需要 NVLink 级别的速度；把 TP 放到慢速的节点间网络上会让 MFU 崩掉 |
| Microsoft ZeRO / PyTorch FSDP | 系统 | 把优化器状态、梯度和参数切分到各个数据并行 rank 上，而不是每份都复制 | 优化器占用超出单卡显存，需要把模型塞进去 | 每步多出 all-gather 和 reduce-scatter；要和计算重叠起来才能保住 MFU |

分界线在于这份技术报告讲的是哪一层：数据流水线定能力上限，分词器定 token 经济账，规模扩展决策定模型尺寸，系统层则决定这次训练可不可行、以及能跑到多少 MFU。

## 这些系统（第一方公开资料）

- **HuggingFace** [FineWeb: decanting the web for the finest text data at scale](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1)：从 96 个 CC dump 里做出的 15T token 开放预训练数据集，过滤器由消融实验选定，在 dump 内和跨 dump 做 MinHash 去重，外加 FineWeb-Edu 教育性分类器。*（数据配方）*
- **TII** [The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only](https://arxiv.org/abs/2306.01116)：精细的 WARC 抽取加过滤加去重，说明只要处理得当，光靠网页数据就能追平甚至超过精选语料。*（数据配方）*
- **Ai2** [Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research](https://arxiv.org/abs/2402.00159)：一份完全开放的 3T token 语料和配套的整理工具链，也就是 OLMo 背后的数据。*（数据配方）*
- **Ai2** [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838)：一个连数据、训练代码和日志都端到端放出来的基础模型；可复现预训练的参考样本。*（完整方案）*
- **EleutherAI** [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://arxiv.org/abs/2101.00027)：为多样性挑出的 22 个领域；早期关于"混合与领域覆盖决定质量"的论证。*（数据配方）*
- **Google** [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (C4)](https://arxiv.org/abs/1910.10683)：C4 语料，用简单可复现的启发式过滤器清洗过的 Common Crawl；干净网页语料的经典基线。*（数据配方）*
- **Meta** [CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data](https://arxiv.org/abs/1911.00359)：行级去重加上语言模型 perplexity 质量过滤，按语言分别跑；多语言爬取数据整理的模板。*（数据配方）*
- **Google** [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/abs/1808.06226)：可逆、与语言无关的子词分词器；多语言模型的默认选择。*（分词器）*
- **Google DeepMind** [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556)：400 多个模型表明参数量和 token 数应该同步扩大，大约每参数 20 个 token；等算力下 70B 的 Chinchilla 胜过 280B 的 Gopher。*（规模扩展决策）*
- **Meta** [The Llama 3 Herd of Models](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/)：从精细的数据整理，到放大规模的 dense 预训练、分阶段上下文扩展，再到让集群规模训练活下来的弹性训练与故障恢复系统，一整套端到端的开放配方。*（完整方案）*
- **DeepSeek** [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)：671B 参数的 MoE（每 token 大约激活 37B），用 FP8 训练，配无辅助 loss 的负载均衡；在受限算力预算下做到前沿规模。*（预训练，MoE）*
- **NVIDIA** [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)：张量并行和流水线并行把权重矩阵切到多张 GPU 上，让单卡放不下的层照样能训。*（系统）*
- **Microsoft** [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)：把优化器状态、梯度和参数切分到各数据并行 rank 上，而不是逐份复制；每张 GPU 的显存降到完整占用的一个零头。*（系统）*
- **Meta** [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277)：ZeRO-3 式切分在 PyTorch 里的原生实现；用之前才 all-gather 参数，用完立刻释放。*（系统）*

想看完整对比，包括那张标准的分歧图和详细的选择表，见 [topics/14-data-curation-and-pretraining.md](../../topics/14-data-curation-and-pretraining.md) 里的密集参考版本。
