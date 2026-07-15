# 2. The five stages

Building with LLMs is five distinct stages. Each stage has a different input,
a different output, a different dominant cost, and a different failure mode.
Confusing them is the fastest way to waste budget and fail the interview.

## Stage by stage

```mermaid
flowchart TD
  WEB["web + proprietary corpus<br/>(raw text, possibly petabytes)"]
  PREP["1. Data preparation<br/>dedup, filter, decontaminate, PII scrub, tokenize<br/>Output: a clean token stream"]
  PT["2. Pretraining<br/>self-supervised next-token prediction over trillions of tokens<br/>Output: a raw base model"]
  MID["3. Mid-training<br/>continued pretraining on domain data or long-context docs<br/>Output: a domain or extended-context base"]
  SFT["4a. SFT<br/>fine-tune on instruction-response pairs<br/>Output: a model that answers instead of completing"]
  PREF["4b. Preference optimization<br/>RLHF / DPO / GRPO<br/>Output: an aligned chat / instruct model"]
  SERVE["5. Deployment and inference<br/>quantize, KV cache, continuous batching, RAG and tools<br/>Output: a production system"]

  WEB --> PREP --> PT --> MID --> SFT --> PREF --> SERVE
  PT --> SFT
```

## The one-line job of each stage

| Stage | Input | Output | Dominant cost | Typical failure |
|---|---|---|---|---|
| 1. Data prep | raw web plus proprietary text | clean tokenized token stream | pipeline engineering, storage | eval leakage from missing decontamination |
| 2. Pretraining | the token stream | raw base model | compute (GPU cluster, weeks) | under- or over-fitting compute budget (violating Chinchilla, the compute-optimal size-vs-tokens rule) |
| 3. Mid-training | existing base plus domain corpus | domain or long-context base | compute (small fraction of pretrain) | catastrophic forgetting if general data is not mixed in |
| 4. Post-training | base plus (instruction, response) pairs and preference data | aligned instruct model | data quality, labeling cost | reward hacking or alignment tax from dropping the KL leash |
| 5. Deployment | aligned model | production serving system | ongoing GPU spend, engineering | KV cache OOM, latency blowup, hallucination without RAG |

## Where most teams actually enter

Almost no product team runs stage 2. That stage is a lab-scale capital
commitment. Most teams enter the diagram at the base model, having downloaded
an open-weights checkpoint (a saved snapshot of the trained model weights; here
Llama 3, DeepSeek-V3, OLMo, Qwen3), and they iterate from there.

The expensive, rare stage (pretraining) is upstream and shared. The stages a
product team actually owns and iterates on (mid-training, post-training,
serving) are downstream and, by comparison, cheap.

```mermaid
flowchart LR
  BASE["open base<br/>(Llama 3, Qwen3, OLMo,<br/>DeepSeek-V3)"]
  MID["mid-training<br/>(domain / long-context)<br/>weeks on a small cluster"]
  POST["post-training<br/>(SFT + DPO)<br/>days"]
  SERVE["serving<br/>(quantize, batch, RAG)<br/>ongoing cost"]

  BASE --> MID --> POST --> SERVE
  BASE --> POST
```

## A note on evaluation

Each stage has a different metric, and using the wrong one is a classic
mistake:

- **Pretraining:** perplexity (how surprised the model is by held-out text;
  lower is better) or bits-per-byte on a held-out set, plus
  zero/few-shot benchmark suites (MMLU, HellaSwag, GSM8K). Perplexity tracks
  the objective but not usefulness; bits-per-byte is tokenizer-invariant so you
  can compare across models. Perplexity is just the exponential of the average
  next-token loss:

```python
import numpy as np
def perplexity(token_nll):
    # token_nll: per-token negative log-likelihoods (natural log) on held-out text
    return np.exp(np.mean(token_nll))
# perplexity(np.array([0.5, 1.0, 2.0, 1.5])) -> 3.4903...  (avg NLL 1.25 -> exp)
```
- **Mid-training:** domain-specific benchmarks (legal, medical, code) plus the
  full general eval suite to catch forgetting.
- **Post-training:** human preference win rate and LLM-as-judge scores
  (Chatbot Arena style), instruction-following and safety suites, task-specific
  evals (code pass@k, math accuracy).
- **Serving:** latency (p50/p95 time-to-first-token and inter-token),
  throughput (tokens/sec/GPU), and cost per million tokens, with a quality
  check that compression did not regress the model.

Name the metric before you propose a fix. Saying "improve perplexity" when the
real goal is preference win rate is a red flag.
