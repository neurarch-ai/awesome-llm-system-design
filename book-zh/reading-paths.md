# 阅读路线

十九章，下周就要面试，从头读到尾不现实。下面是针对具体面试类型的几种读法，每一种
都说明了这类面试真正考什么、哪些可以跳过。

所有路线都默认你已经先读过[方法论](00-the-method.md)。它很短，而且是其余内容挂靠的
主干。

## 如果只有一周

| 天 | 读什么 | 为什么 |
|---|---|---|
| 1 | [方法论](00-the-method.md)、[LLM 的生命周期](llm-lifecycle/) | 先把词汇和地图装进脑子，后面就不会迷路 |
| 2 | [RAG 服务](rag-serving/) | 被问得最多的端到端题，而且同时涉及检索、服务和评估 |
| 3 | [KV Cache](kv-cache/)、[推理服务](inference-serving/) | 所有服务相关的追问最后都绕回这套成本模型 |
| 4 | [Agent](agents/) | 第二常考的设计题，工具调用和可靠性都在这里 |
| 5 | [评估](evaluation/)、[Benchmark](benchmark-eval/08-interview-qa.md) | 每个人都会被问"你怎么知道它有效" |
| 6 | [该记住的数字](numbers-to-know.md)、[安全](safety/08-interview-qa.md)、[成本优化](cost-optimization/08-interview-qa.md) | 先练记忆，再看两个最常见的追问方向 |
| 7 | [模拟面试](mock-interview.md)，然后重读你最薄弱那一章的问答 | 练的是表达，不只是内容 |

## 按岗位

**LLM 基础设施与服务。** [KV Cache](kv-cache/)，然后[推理服务](inference-serving/)，
然后[模型压缩](model-compression/)，然后[推理模型服务](reasoning-serving/)，然后
[成本优化](cost-optimization/)，然后[流式对话](streaming-chat/)。这类面试考成本模型、
批处理、并行、量化和尾延迟。做好准备被要求当场口算显存和吞吐，所以
[该记住的数字](numbers-to-know.md)要练得最狠。

**模型团队的 applied scientist 或 ML 工程师。** [LLM 的生命周期](llm-lifecycle/)，然后
[数据整理与预训练](data-and-pretraining/)，然后[中期训练](mid-training/)，然后
[后训练](post-training/)，然后 [Benchmark](benchmark-eval/)，然后[评估](evaluation/)。
这类面试考的是能不能把一条训练管线从头到尾想清楚，最重要的是能不能分辨真正的提升和
测量假象。Benchmark 那一章是这里拉开差距的地方。

**AI 工程师或偏产品的 LLM 工作。** [RAG 服务](rag-serving/)，然后 [Agent](agents/)，
然后[评估](evaluation/)，然后[安全](safety/)，然后[成本优化](cost-optimization/)，然后
[监控](monitoring/)。这类面试考的是能不能交付用户真正会碰到的东西：有据可依、工具
可靠性、护栏、评估门禁，以及一套讲得通的成本故事。

**搜索、检索或推荐方向转向 LLM。** [语义搜索](semantic-search/)，然后
[RAG 服务](rag-serving/)，然后[评估](evaluation/)，再读经典 ML 配套书里的
[候选召回](https://github.com/neurarch-ai/awesome-ml-system-design/tree/main/book/candidate-retrieval/)
和[排序](https://github.com/neurarch-ai/awesome-ml-system-design/tree/main/book/ranking/)
两章。这类面试通常两边都考，对两者接缝处的熟练度会加分。

**端侧或效率方向。** [模型压缩](model-compression/)，然后 [KV Cache](kv-cache/)，
然后[推理服务](inference-serving/)，然后[后训练](post-training/)（为了 adapter），
如果产品带视觉再加[多模态](multimodal/)。预计会被问到数值格式、加速器到底加速了什么，
以及怎么证明压缩后的模型还是同一个模型。

**偏研究或前沿实验室。** [LLM 的生命周期](llm-lifecycle/)，然后
[数据整理与预训练](data-and-pretraining/)，然后[中期训练](mid-training/)，然后
[后训练](post-training/)，然后[推理模型服务](reasoning-serving/)，然后
[Benchmark](benchmark-eval/)，最后把[深入专题](../deep-dives.md)从头到尾过一遍。
这类面试探的是对机制的理解深度和对不确定性的容忍度；深入专题那份题库是应付连珠炮
追问的最好准备。

## 按提前告知的题目

| 对方说 | 按这个顺序读 |
|---|---|
| "设计一个 RAG 系统" | [RAG 服务](rag-serving/)、[语义搜索](semantic-search/)、[评估](evaluation/) |
| "设计一个 agent" | [Agent](agents/)、[安全](safety/)、[评估](evaluation/)、[推理模型服务](reasoning-serving/) |
| "把我们的 LLM 做便宜点" | [成本优化](cost-optimization/)、[KV Cache](kv-cache/)、[模型压缩](model-compression/)、[推理模型服务](reasoning-serving/) |
| "服务与基础设施" | [推理服务](inference-serving/)、[KV Cache](kv-cache/)、[流式对话](streaming-chat/)、[模型压缩](model-compression/) |
| "你会怎么评估它" | [评估](evaluation/)、[Benchmark](benchmark-eval/)、[监控](monitoring/) |
| "微调与训练" | [后训练](post-training/)、[中期训练](mid-training/)、[数据与预训练](data-and-pretraining/) |
| "多模态" | [多模态](multimodal/)、[推理服务](inference-serving/)、[评估](evaluation/) |
| 没说，就是一轮泛 LLM 面试 | 上面的一周路线 |

## 时间紧的时候怎么读一章

如果某一章在你的清单上，而你只有二十分钟，就按这个顺序读，时间到了就停：先读
**README**（问题定义和架构图），然后 **08 面试问答**，然后 **09 小结**并试着做自测题，
最后看 10 里的 **capstone** 表格。这条路线给你的是答案和数字。中间几节才是真正的理解
所在，等有时间好好读的时候再读它们。
