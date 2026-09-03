# 7. 真实团队在生产环境里怎么做

每一套认真的压缩栈最后都收敛到同一副骨架：挑那根能撬动瓶颈资源的杠杆，留一小撮
层在较高精度上，最后用一次和未压缩模型的配对比较来把关。差别在于**它们到底在解
哪个约束**（服务器成本、设备内存，还是硬件要求的某种数值格式），以及**它们负不负
担得起重训**，下面几乎每一处分歧都能被这两个选择解释掉。

## 真实设计在哪里分道扬镳

| 系统 | 主要约束 | 方法 | 重训 | 什么时候它赢 | 注意什么 |
|---|---|---|---|---|---|
| 服务端纯权重量化（GPTQ、AWQ 这一脉） | 装得下和 decode 延迟 | int4 分组权重，fp16 激活 | 不需要 | 小 batch 的交互式服务，也是拿到收益最快的路 | batch 一高收益就缩水；要有融合 kernel 才是真的 |
| 激活和计算量化（SmoothQuant、fp8 配方） | 每美元吞吐 | int8 或 fp8 的权重加激活 | 不需要到轻量 | 高 batch、prefill 重的负载，成本就是 FLOPs | 必须处理离群值；确认运行时没有悄悄升精度 |
| 基于旋转的低 bit（QuaRot、SpinQuant） | 4-bit 激活和 KV，移动端服务 | 把正交旋转融进网络，再量化 | 不需要（SpinQuant 会学旋转） | 要压过普通 PTQ 还守得住的那个点 | 旋转必须在导出时融进去，否则每次前向都要付一遍 |
| 训练期三值（BitNet） | 极致效率 | 1.58-bit 权重，从头就这么训 | 完整预训练 | 专为这个区间设计的新模型 | 不是一个后训练选项；现有模型转不过去 |
| 一次性剪枝（SparseGPT、Wanda） | 内存，或者一个能加速稀疏的目标 | 非结构化或 2:4，不重训或只轻微重训 | 不需要 | 快速摸清模型能容忍多少稀疏 | 非结构化在稠密 kernel 上换不来速度 |
| 剪枝加蒸馏这一类（Minitron） | 从一个父模型拉出一整条尺寸阶梯 | 结构化剪枝，然后从父模型蒸馏 | 几十亿 token | 没有预训练预算却要造小模型 | 砍深度会伤多步行为；修复预算是实打实的 |
| 继续预训练式剪枝（Sheared LLaMA） | 用从头训练一小部分的成本拿到一个小模型 | 定向结构化剪枝到选定架构，再继续预训练 | 需要 | 想要一个特定的小架构，而且有 token 预算 | token 预算仍然不小 |
| 端侧基础模型（Apple Intelligence） | 硬的设备内存上限和固定的加速器格式 | 小的端侧模型、激进的低 bit 权重压缩、任务 adapter，其余交给更大的服务端模型 | 需要，写进模型设计里 | 上限没得商量的消费级设备 | 两个模型加一条 adapter 流水线要维护和评估 |
| 把 fp8 当一等格式贯穿训练和服务（DeepSeek-V3） | 整个生命周期的成本 | 训练就用 fp8，一路带到服务 | 设计之初就在 | 在支持 fp8 的硬件上从零做大模型 | 需要前期把数值这块做扎实；没法事后加装 |
| CPU 和消费级 GPU 量化（llama.cpp、GGUF） | 在普通硬件上先跑起来 | k-quant 权重格式，CPU 和 Metal kernel | 不需要 | 本地和爱好者部署，以及快速评估量化容忍度 | 格式和质量随 quant 类型变化；要 benchmark 你实际发的那一种 |

## 分水岭

两个问题就能把任何一套技术栈放进上面这张表。**你能不能重训？**不能的话，你就在
量化和一次性剪枝这半边，天花板由模型不做修复能容忍多少来决定。能的话，结构化
剪枝加蒸馏占优，因为它产出的是一个真正更小的稠密模型，在任何硬件上都快，而不是
一个套了件小外套的大模型。

**目标硬件加速的是什么？**一种硬件不加速的格式，是省内存的技术，不是提速的技术。
就这一句话，能了结大部分关于哪种方法"最好"的争论；也正是它解释了为什么端侧技术栈
和服务端技术栈长得完全不一样，哪怕两边嘴上说的都是"4-bit"。

## 一手来源

- **LLM.int8()** [8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)：让大模型 int8 推理变得实用的离群通道分解。
- **SmoothQuant** [Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438)：把激活的离群值搬进权重，两边就都能量化了。
- **GPTQ** [Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323)：逐层的二阶取整加误差补偿。
- **AWQ** [Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978)：保护由激活量级识别出来的那些显著权重通道。
- **QuaRot** [Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456) 和 **SpinQuant** [LLM quantization with learned rotations](https://arxiv.org/abs/2405.16406)：通过旋转基底消掉离群值，这正是解锁 4-bit 激活和 KV 的东西。
- **Microsoft Research** [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)：三值权重，直接在这个区间里训练而不是转换过来。
- **KIVI** [A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750)：key 按通道、value 按 token，因为两者的离群结构不一样。
- **SparseGPT** [Massive Language Models Can Be Accurately Pruned in One-Shot](https://arxiv.org/abs/2301.00774) 和 **Wanda** [A Simple and Effective Pruning Approach for Large Language Models](https://arxiv.org/abs/2306.11695)：基于重建的和激活感知的一次性剪枝。
- **Princeton** [Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning](https://arxiv.org/abs/2310.06694)：把大模型剪到目标架构，然后继续预训练。
- **NVIDIA** [Compact Language Models via Pruning and Knowledge Distillation](https://arxiv.org/abs/2407.14679)：先剪后蒸的配方，用一个父模型拉出一条尺寸阶梯。
- **Google DeepMind** [On-Policy Distillation of Language Models](https://arxiv.org/abs/2306.13649)：拿老师给学生自己的输出打分来训学生，修掉分布错配。
- **Apple** [Apple Intelligence Foundation Language Models](https://arxiv.org/abs/2407.21075)：一个做了低 bit 权重压缩加任务 adapter 的端侧模型，旁边配一个服务端模型。
- **DeepSeek** [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)：把 fp8 当作贯穿训练和服务的一等格式。
- **Microsoft Research India** [Accuracy is Not All You Need](https://arxiv.org/abs/2407.09141)：为什么准确率打平的压缩模型行为仍然不一样，以及该改测什么。
- **Red Hat AI 和 vLLM 项目** [llm-compressor](https://github.com/vllm-project/llm-compressor)：面向 vLLM 的权重、激活和 KV cache 量化生产配方。
- **llama.cpp** [GGUF 量化生态](https://github.com/ggml-org/llama.cpp)：CPU、Metal 和消费级硬件上量化推理的参考实现。

想要单文件的密集版参考（同样的材料，面试串讲的形态）：
[topics/17-model-compression.md](../../topics/17-model-compression.md)。
