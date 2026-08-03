# Numbers to know

The quantities you should be able to produce from memory in an interview, with the
formula that generates them so you can re-derive rather than recall. Everything here
is developed properly in the chapter linked beside it; this page is the compression.

Order-of-magnitude figures for illustration, not benchmarks. The habit that matters
is stating the formula, plugging in, and saying what would change the answer.

## Model size and memory

| Quantity | Formula or figure | Chapter |
|---|---|---|
| Weights in memory | $N \cdot b_w$ bytes; fp16 is 2 bytes per parameter, int8 is 1, int4 group-wise is about 0.52 (4 bits plus an fp16 scale per 128) | [compression](model-compression/) |
| 70B model, fp16 | about 141 GB, so it does not fit one 80GB accelerator | [compression](model-compression/10-putting-it-together.md) |
| 70B model, int4 group-wise | about 36 GB, which does | [compression](model-compression/10-putting-it-together.md) |
| KV cache per token | $2 \cdot L \cdot h_{kv} \cdot d_h \cdot b_{kv}$ bytes | [KV cache](kv-cache/) |
| KV cache, 80 layers, 8 KV heads, head dim 128, fp16 | 320 KB per token, so 32K context is about 10.5 GB per sequence | [KV cache](kv-cache/) |
| Training memory, full fine-tune | roughly $2N$ (bf16 weights) plus about $12N$ (Adam state) bytes, before activations | [post-training](post-training/) |
| LoRA or QLoRA trainable share | well under 1 percent of parameters; QLoRA holds the frozen base at 4 bits | [post-training](post-training/) |

## Throughput and latency

| Quantity | Formula or figure | Chapter |
|---|---|---|
| Decode time per token | (weight bytes + KV bytes read) / memory bandwidth: decode is bandwidth bound | [inference serving](inference-serving/) |
| Prefill time | tracks FLOPs, about $2N$ per prompt token: prefill is compute bound | [inference serving](inference-serving/) |
| Effect of halving weight bytes | nearly halves decode time at batch 1; roughly 1.1 to 1.3x at batch 32 to 64 | [compression](model-compression/02-frame-the-compression.md) |
| Queueing delay | grows with $\frac{\rho}{1-\rho}\cdot\frac{E[S]\,(1+C^2)}{2}$: the second moment, not just the mean | [reasoning serving](reasoning-serving/03-budgets-and-latency.md) |
| Reasoning path variance | thinking traces are long-tailed; a $C^2$ of about 0.9 versus 0.2 for a short path | [reasoning serving](reasoning-serving/10-putting-it-together.md) |
| Speculative decoding | attacks bandwidth-bound decode, so it helps thinking-heavy workloads most | [inference serving](inference-serving/04-speculative-decoding.md) |

## Evaluation statistics

| Quantity | Formula or figure | Chapter |
|---|---|---|
| Interval on a benchmark score | $\pm 1.96\sqrt{\hat p(1-\hat p)/n}$ | [benchmarking](benchmark-eval/06-statistics-and-leaderboards.md) |
| 30-item benchmark at $\hat p = 0.5$ | about $\pm 18$ points; one item is 3.3 points | [benchmarking](benchmark-eval/06-statistics-and-leaderboards.md) |
| 198-item benchmark (GPQA Diamond) | about $\pm 7$ points | [benchmarking](benchmark-eval/06-statistics-and-leaderboards.md) |
| 500-item benchmark (SWE-bench Verified) | about $\pm 4.4$ points | [benchmarking](benchmark-eval/06-statistics-and-leaderboards.md) |
| Paired difference standard error | $\sqrt{b+c}/n$ over discordant items; McNemar $z = (b-c)/\sqrt{b+c}$ | [benchmarking](benchmark-eval/06-statistics-and-leaderboards.md) |
| Items needed to call a 2-point gap | $n \approx d \cdot 7.85/\delta^2$, roughly 2,000 items at 10 percent discordance | [benchmarking](benchmark-eval/06-statistics-and-leaderboards.md) |
| A/B win-rate significance | 95 percent interval must exclude 0.5; about 2,400 comparisons for a 54 percent true rate | [evaluation](evaluation/05-online-eval.md) |
| Judge trust bar | Cohen's kappa around 0.6 before gating on a judge | [evaluation](evaluation/04-llm-as-judge.md) |
| PPI correction | $\hat\theta = \text{mean}(\text{judge}) + \text{mean}(\text{human} - \text{judge})$ on the labeled slice | [benchmarking](benchmark-eval/05-scoring-and-autoraters.md) |

## Sampling and agents

| Quantity | Formula or figure | Chapter |
|---|---|---|
| pass@k (coverage) | $1-(1-p)^k$; unbiased estimator $1 - \binom{n-c}{k}/\binom{n}{k}$ | [evaluation](evaluation/03-offline-eval.md) |
| pass^k (reliability) | $p^k$; a 90 percent agent is about 43 percent at $k=8$ | [benchmarking](benchmark-eval/05-scoring-and-autoraters.md) |
| Cascade breakeven | cascade wins when $a \gt (c_{\text{short}} + c_{\text{verify}})/c_{\text{long}}$, often about 15 percent | [reasoning serving](reasoning-serving/04-allocation-and-routing.md) |
| Delivered quality from sampling | coverage times selector accuracy: a 0.98 coverage with a 0.6 selector delivers about 0.6 | [reasoning serving](reasoning-serving/05-verification.md) |

## Data and training

| Quantity | Formula or figure | Chapter |
|---|---|---|
| Compute-optimal tokens | Chinchilla scaling: roughly 20 tokens per parameter | [data and pretraining](data-and-pretraining/) |
| Training FLOPs | about $6ND$ for $N$ parameters and $D$ tokens | [llm lifecycle](llm-lifecycle/) |
| Replay fraction against forgetting | 5 to 10 percent general data in a continued-pretraining mix | [mid-training](mid-training/03-continued-pretraining.md) |
| Continued-pretraining corpus floor | billions of tokens; below that, use SFT or RAG instead | [mid-training](mid-training/) |
| Re-warm peak learning rate | a fraction of the original pretraining peak, then re-decay | [mid-training](mid-training/03-continued-pretraining.md) |

## Retrieval

| Quantity | Formula or figure | Chapter |
|---|---|---|
| Embedding index size | vectors times dimension times bytes per component; quantization is the main lever | [semantic search](semantic-search/) |
| Recall versus latency | ANN parameters trade them continuously; report recall at a fixed latency, not alone | [semantic search](semantic-search/04-vector-index.md) |
| Re-ranking budget | a cross-encoder over the top 50 to 100 candidates, not over the corpus | [rag serving](rag-serving/04-retrieval-and-reranking.md) |
| Chunk size | hundreds of tokens with overlap; the real constraint is what the reranker and the answer need | [rag serving](rag-serving/03-indexing-and-chunking.md) |

## The five sentences worth memorizing

1. **Decode is bandwidth bound, prefill is compute bound.** Which one you are
   optimizing decides whether a lever can possibly work.
2. **The KV cache is the term that grows with context and batch;** weights are fixed.
3. **A benchmark score is an estimate.** On 200 items, a 3-point gap from one run is
   not resolvable, and the paired comparison is what buys the precision.
4. **Queueing delay grows with the variance of service time,** which is why thinking
   models wreck the tail before they wreck the mean.
5. **Cost per solved task, not cost per request.** Every cheap policy is cheap
   because it fails more, and only the denominator shows it.
