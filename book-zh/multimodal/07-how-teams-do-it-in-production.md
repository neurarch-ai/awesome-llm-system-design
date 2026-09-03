# 7. 真实团队在生产环境里怎么做

所有大型视觉语言系统最后都收敛到同一个三段式骨架：视觉编码器把图像变成特征网格，
projector 把这些特征映射进解码器的 embedding 空间，一个 LLM 解码器在文本和图像交错的序列上做生成。
系统之间真正不一样的其实只有两个决定：**一张图变成多少个 token**（由连接器和分辨率策略决定），
以及**服务栈怎么把编码器和解码器隔开**。架构是大家共用的，杠杆落在 token 预算和服务拆分上。

## 真实设计的分歧点

| 系统 | 连接器 | 分辨率策略 | 图像 token 预算 | 服务方式 | 为什么长这样 |
|---|---|---|---|---|---|
| LLaVA（微软） | MLP projector | 固定 CLIP ViT-L/14 336px | 约 576（固定） | 单 server，编码器冻结 | 最省的配方；在冻结的 CLIP 之上一次性训练 projector |
| BLIP-2（Salesforce） | Q-Former cross-attention | 固定（CLIP 或 EVA-CLIP） | 32（固定上限） | 单 server | LLM 微调量最小；编码器和 LLM 都冻结，只留一座很小的可训练桥 |
| Flamingo（DeepMind） | Perceiver resampler 加门控 cross-attention | 固定，token 很少 | 固定的少量 | 两个骨干都冻结 | 在冻结的视觉和语言模型之上做交错的 few-shot |
| Idefics2（HuggingFace） | Perceiver pooling 加 MLP | 原生分辨率最高 980px，可选四分切分 | 64 或 320 | 单 server，NaViT 风格 | 开放配方；用更好的骨干和 OCR 数据，同时把 token 数框住 |
| Qwen2-VL（阿里巴巴） | MLP projector 加 M-RoPE | 原生动态分辨率 | 可变，随分辨率增长 | 单 server，批处理适配变长序列 | 输入尺寸差异很大；动态分辨率避免了裁剪带来的失真 |
| Pixtral 12B（Mistral AI） | 自研 ViT 加 MLP projector | 原生分辨率，每张图各自灵活 | 可变，1024px 时最高 4096 | 128K 上下文，支持多图 | 自研编码器让原生长宽比和灵活预算成为可能 |
| NVLM（NVIDIA） | MLP 或 cross-attention，正面对比过 | 分块（tiling），带一维 tile 标签服务 OCR | 高（分块会成倍放大 token 数） | 单 server，tile 标签在预处理阶段加上 | 面向 OCR 和密集文档；tile 标签保住了空间版式 |
| vLLM V1（Red Hat） | 任意（这是运行时，不是模型） | 按模型配置 | 按模型，带缓存 | 编码器缓存加前缀缓存；CPU/GPU 异步拆分 | 重复的图跳过重新编码；纯文本请求完全跳过编码器 |
| ROCm vLLM（AMD） | 任意（这是运行时，不是模型） | 按模型配置 | 按模型 | 编码器数据并行加解码器张量并行；一次 all-gather 取代逐层 all-reduce | 编码器只占参数量的一小部分；对它做 TP 白费同步开销，又省不下计算 |

分界线很简单：**连接器和分辨率策略买的是质量上限，服务拆分和缓存买的是延迟和单位经济性。**
一个完整的回答会在这两条线上各选一个点，并且用任务对细节的要求和流量构成来说明为什么选这里。

## 这些系统（一手资料）

- **Red Hat（vLLM）** [vLLM V1: accelerating multimodal inference](https://developers.redhat.com/articles/2025/02/27/vllm-v1-accelerating-multimodal-inference-large-language-models)：编码器输出缓存、按图的前缀缓存，以及 CPU/GPU 异步输入处理，让多模态服务更快。想知道怎么按图像 hash 缓存编码器 embedding、以及占位 token 在不同图片间共享时怎么避免前缀缓存撞车，看这一篇。
- **AMD（ROCm）** [Accelerating Multimodal Inference in vLLM](https://rocm.blogs.amd.com/software-tools-optimization/vllm-dp-vision/README.html)：对视觉编码器用数据并行（每张 GPU 各存一份完整副本，各处理一批不同的图），把同步从逐层 all-reduce 降到末尾一次 all-gather；在 InternVL3.5-241B 上吞吐最多提升 44%。
- **阿里巴巴（Qwen）** [Qwen2-VL: enhancing vision-language model perception at any resolution](https://arxiv.org/abs/2409.12191)：Naive Dynamic Resolution 用 MLP projector 把不同分辨率的图处理成数量可变的视觉 token；M-RoPE 用一套方案统一了文本、图像和视频的位置编码。
- **Mistral AI** [Pixtral 12B](https://arxiv.org/abs/2410.07073)：从零训练的自研 ViT 直接吃原生分辨率和原生长宽比；在 128K 上下文里每张图的 token 预算可以灵活分配；以 Apache 2.0 发布，并附带 MM-MT-Bench 用于评估。
- **微软（LLaVA）** [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)：最早那版把冻结的 CLIP 视觉编码器桥接到 LLM 的 MLP projector；用 GPT-4 生成的指令数据作为多模态对话的训练信号。
- **Hugging Face** [Introducing Idefics2: a powerful 8B vision-language model](https://huggingface.co/blog/idefics2)：Perceiver pooling 加 MLP 投影；NaViT 原生分辨率编码，最高 980px；按切分模式每张图限定 64 或 320 个 token；放弃了门控 cross-attention，改用更简单的先池化再投影。
- **NVIDIA** [NVLM: open frontier-class multimodal LLMs](https://research.nvidia.com/labs/adlr/NVLM-1/)：把 decoder-only（MLP）和 cross-attention 两种连接器做了正面对比；在分块的高分辨率 OCR 里用一维 tile 标签保留空间版式；NVLM-D-72B 在数学和文档 benchmark 上追平甚至超过 GPT-4o。
- **DeepMind** [Flamingo: a visual language model for few-shot learning](https://arxiv.org/abs/2204.14198)：用 Perceiver resampler 加门控 cross-attention 桥接冻结的视觉和语言骨干；固定的小 token 预算；成本有界的 few-shot VLM 设计就以它为参照。
- **Salesforce** [BLIP-2: bootstrapping language-image pre-training with frozen image encoders and large language models](https://arxiv.org/abs/2301.12597)：Q-Former 用 32 个 query token 在两个冻结的预训练模型之间搭了一座轻量的桥；微调成本极低。
- **Dropbox** [Creating a modern OCR pipeline using CV and deep learning](https://dropbox.tech/machine-learning/creating-a-modern-ocr-pipeline-using-computer-vision-and-deep-learning)：一条已经产品化的两段式 OCR 流水线（MSER 检测器加 CNN-BLSTM-CTC 识别器），配 1000 万条合成训练样本，并用 LXC 隔离来自不可信来源的文档上传。它不是 VLM，但它是生产级 OCR 工程的参照。
- **苹果** [MM1: methods, analysis, and insights from multimodal LLM pre-training](https://arxiv.org/abs/2403.09611)：围绕图像 token 数、连接器设计和数据配比做的消融实验；关于各个设计轴之间怎么互相权衡，这是公开发表里最彻底的一份分析。
- **Meta** [Chameleon: mixed-modal early-fusion foundation models](https://arxiv.org/abs/2405.09818)：一个 transformer 直接跑在离散的图像和文本 token 上；这是早期融合训练怎么训稳的主要公开配方。

完整的横向对比、案例分析和模型拆解，见 [topics/09-multimodal-serving.md](../../topics/09-multimodal-serving.md) 那份密集的参考资料。
