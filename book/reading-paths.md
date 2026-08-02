# Reading paths

Nineteen chapters is a lot to read front to back before a loop next week. These are
the orders that make sense for a specific interview, each with what the loop
actually tests and what to skip.

Every path assumes you have read [the method](00-the-method.md) first. It is short
and it is the spine the rest hangs off.

## If you have one week

| Day | Read | Why |
|---|---|---|
| 1 | [The method](00-the-method.md), [LLM lifecycle](llm-lifecycle/) | The vocabulary and the map, so nothing later is disorienting |
| 2 | [RAG serving](rag-serving/) | The most-asked end-to-end question, and it touches retrieval, serving, and eval |
| 3 | [KV cache](kv-cache/), [inference serving](inference-serving/) | The cost model every serving follow-up returns to |
| 4 | [Agents](agents/) | The second most-asked design, and where tool calling and reliability live |
| 5 | [Evaluation](evaluation/), [benchmarking](benchmark-eval/08-interview-qa.md) | Everyone gets asked how they know it works |
| 6 | [Numbers to know](numbers-to-know.md), [safety](safety/08-interview-qa.md), [cost optimization](cost-optimization/08-interview-qa.md) | Recall drills, then the two most common follow-up areas |
| 7 | [Mock interview](mock-interview.md), then re-read your weakest chapter's Q&A | Practice the delivery, not just the content |

## By role

**LLM infrastructure and serving.** [KV cache](kv-cache/) then
[inference serving](inference-serving/) then [model compression](model-compression/)
then [reasoning serving](reasoning-serving/) then
[cost optimization](cost-optimization/) then [streaming chat](streaming-chat/).
The loop tests the cost model, batching, parallelism, quantization, and tail latency.
Expect to be asked to estimate memory and throughput out loud, so drill
[numbers to know](numbers-to-know.md) hardest.

**Applied scientist or ML engineer on a model team.** [LLM lifecycle](llm-lifecycle/)
then [data curation and pretraining](data-and-pretraining/) then
[mid-training](continued-pretraining/) then [post-training](post-training/) then
[benchmarking](benchmark-eval/) then [evaluation](evaluation/). The loop tests
whether you can reason about a training pipeline end to end and, above all, whether
you can tell a real improvement from a measurement artifact. The benchmarking
chapter is the differentiator here.

**AI engineer or product-facing LLM work.** [RAG serving](rag-serving/) then
[agents](agents/) then [evaluation](evaluation/) then [safety](safety/) then
[cost optimization](cost-optimization/) then [monitoring](monitoring/). The loop
tests whether you can ship something users touch: grounding, tool reliability,
guardrails, an eval gate, and a cost story.

**Search, retrieval, or recommendations crossing into LLMs.**
[Semantic search](semantic-search/) then [RAG serving](rag-serving/) then
[evaluation](evaluation/), then the classic-ML companion's
[candidate retrieval](https://github.com/neurarch-ai/awesome-ml-system-design/tree/main/book/candidate-retrieval/)
and [ranking](https://github.com/neurarch-ai/awesome-ml-system-design/tree/main/book/ranking/)
chapters. The loop usually tests both halves and rewards being fluent in the seam
between them.

**On-device or efficiency-focused.** [Model compression](model-compression/) then
[KV cache](kv-cache/) then [inference serving](inference-serving/) then
[post-training](post-training/) (for adapters) then
[multimodal](multimodal/) if the product has vision. Expect questions about numeric
formats, what the accelerator actually accelerates, and how you prove a compressed
model is still the same model.

**Research-adjacent or frontier-lab.** [LLM lifecycle](llm-lifecycle/) then
[data curation and pretraining](data-and-pretraining/) then
[mid-training](continued-pretraining/) then [post-training](post-training/) then
[reasoning serving](reasoning-serving/) then [benchmarking](benchmark-eval/) then
[deep-dives](../deep-dives.md) end to end. The loop probes depth on mechanisms and
tolerance for uncertainty; the deep-dive bank is the best preparation for
rapid-fire follow-ups.

## By question you were told to expect

| They said | Read, in this order |
|---|---|
| "Design a RAG system" | [RAG serving](rag-serving/), [semantic search](semantic-search/), [evaluation](evaluation/) |
| "Design an agent" | [Agents](agents/), [safety](safety/), [evaluation](evaluation/), [reasoning serving](reasoning-serving/) |
| "Make our LLM cheaper" | [Cost optimization](cost-optimization/), [KV cache](kv-cache/), [model compression](model-compression/), [reasoning serving](reasoning-serving/) |
| "Serving and infrastructure" | [Inference serving](inference-serving/), [KV cache](kv-cache/), [streaming chat](streaming-chat/), [model compression](model-compression/) |
| "How would you evaluate this" | [Evaluation](evaluation/), [benchmarking](benchmark-eval/), [monitoring](monitoring/) |
| "Fine-tuning and training" | [Post-training](post-training/), [mid-training](continued-pretraining/), [data and pretraining](data-and-pretraining/) |
| "Multimodal" | [Multimodal](multimodal/), [inference serving](inference-serving/), [evaluation](evaluation/) |
| Unspecified, a general LLM loop | The one-week path above |

## How to read a chapter under time pressure

If a chapter is on your list and you have twenty minutes, read in this order and
stop when the time runs out: the **README** (the framing and the diagram), then
**08 Interview Q&A**, then **09 Summary** and attempt the test-yourself questions,
then the **capstone** table in 10. That path gives you the answers and the numbers.
The middle sections are where the understanding is, and they are what you read when
you have the time to do it properly.
