# 5. Long context

Serving 128k tokens is not just a memory problem. The attention mechanism itself
must be able to correctly attend across that range. Most base models are trained on
sequences far shorter than the contexts they are asked to serve. This section
covers how teams extend that range without retraining from scratch.

## Why position embeddings break at long context

Transformer attention is position-aware because keys and queries are rotated by a
position-dependent angle before the dot product (RoPE) or because a position bias
is added to attention logits (ALiBi). Either way, the model learns what "token at
position 500" means, and that learning is anchored to the training distribution.
Ask the model to reason about "token at position 100 000" and it is in out-of-
distribution territory. Perplexity spikes and recall of early context collapses.

## Position interpolation: the simplest fix

**Position interpolation** (PI, Meta, 2023) re-maps a longer context into the
original training positions by scaling all position indices:

$$\text{pos}_{\text{scaled}} = \text{pos} \times \frac{L_{\text{train}}}{L_{\text{target}}}$$

A token at position 65 536 in a 128k context is treated as if it were at position
4 096 in an 8k-trained model. The model never sees an out-of-range position index.
The cost is that nearby tokens are compressed together, reducing effective
resolution at short range, which can hurt tasks that depend on precise local
positioning.

A brief fine-tuning step (1 000 to 10 000 gradient steps on long sequences) is
sufficient to recover quality after interpolation. Without fine-tuning the method
still beats naive extrapolation.

## YaRN: better long-context extension

**YaRN** (Yet another RoPE extensioN, Mistral, 2023) improves on plain position
interpolation by applying **frequency-dependent scaling**: high-frequency RoPE
dimensions (which encode fine-grained local position) are not compressed, while
low-frequency dimensions (which encode global position) are interpolated. A
temperature parameter also rescales attention logits to keep the softmax
distribution from collapsing at very long range.

In practice YaRN extends context 8x to 16x with a finer quality profile than plain
PI: short-range tasks lose less, long-range recall improves more. It is what
Mistral-7B-v0.2 uses to extend from 8k to 32k, and several community model
extensions use it to push Llama models past 128k.

## Sliding-window attention: trading recall for a fixed cache

**Sliding-window attention** limits each token's attention to the nearest $W$
tokens rather than the entire sequence. The KV cache per layer grows with $W$, not
with $S$, so memory is bounded regardless of context length.

The classic tradeoff: tokens that fall outside the window are simply lost. For
many generative tasks this is fine (the next token mostly depends on recent context)
but it is wrong for document retrieval or question answering where the answer may
sit anywhere in the document.

Mistral 7B uses a sliding window of 4 096 tokens on most layers, combined with a
few full-attention layers deep in the stack. This hybrid gives long-range
connectivity at low cost.

**Attention sinks (StreamingLLM, MIT/Meta, 2023)** extend the idea: keep a small
window of recent tokens plus the first few "sink" tokens (positions 0 to 3), which
the model's attention mass disproportionately falls on even when their content is
not directly relevant. With sinks held fixed and a sliding window of recent tokens,
a model can stream to millions of tokens in a fixed KV budget. The catch: the
middle of the context is truly gone; retrieval from it fails.

## Chunked prefill: avoiding OOM during long-prompt prefill

Even if the model can handle 128k tokens architecturally, the GPU must compute the
KV cache for all 128k tokens during prefill. Doing this in one pass can exceed HBM
capacity for large batch sizes. **Chunked prefill** splits the prompt into chunks
(say, 4k tokens each) and processes them sequentially, reusing each chunk's KV
output as the input to the next. Memory usage during prefill stays bounded at the
chunk size, not the full context.

The tradeoff: chunked prefill serializes what could be one large parallel pass,
so first-token latency is longer. Systems that need both high concurrency (which
favors chunking to bound peak memory) and low first-token latency (which favors
one big pass) tune the chunk size for their traffic mix. SGLang and vLLM both
expose this knob.

## Evaluating position extension

Two metrics confirm whether an extension technique delivers real recall, not just
recovered perplexity.

**Perplexity** (PPL) is the exponentiated mean negative log-likelihood per token:

$$\text{PPL} = \exp\!\left(-\frac{1}{N}\sum_{i=1}^{N}\log p(x_i \mid x_{\lt i})\right)$$

Input: a held-out token sequence at the target context length. A PPL spike beyond
the original training length is the first sign that position extension failed. PPL
is a fast continuous signal during training but saturates: a model can recover normal
PPL while still failing to retrieve facts from the middle of a long context. Gate
training stability on PPL; gate production-readiness on retrieval recall.

**NIAH recall** (needle-in-a-haystack) is the fraction of planted facts retrieved
correctly at each (context-length, insertion-depth) cell of a test grid. Input: a
long filler document with one known fact inserted at a fixed position; the model is
asked to reproduce that fact. A model that passes PPL but fails NIAH is the standard
failure mode after shallow position extension: local next-token prediction is fine,
but attention to distant positions is broken. Report recall as a two-dimensional
heatmap over length and depth; the mid-context dip (worst recall near 50 percent
depth) is the most informative signal and the one that averaged scores hide. For
multi-needle, variable-tracing, and aggregation tasks use RULER, which separates
the effective context length from the configured one (covered in the continued-
pretraining chapter).

**When to use which long-context technique.**

| Reach for | When | Skip it when |
|---|---|---|
| Position interpolation with fine-tuning | 2x to 4x context extension on an existing checkpoint, with a short fine-tuning run available | Naive extrapolation, which works barely or not at all |
| YaRN | 4x to 16x extension with a better quality profile than plain PI; available in most open model forks | Heavier fine-tuning recipe; skip when you need a drop-in with no training |
| Sliding-window attention (Mistral, Databricks MixAttention) | Unbounded or very long sequences where losing the middle of the context is acceptable | Question-answering or retrieval tasks that require whole-document recall |
| Attention sinks (StreamingLLM) | Truly streaming, never-ending conversations; memory must not grow | Any task that requires recall of content beyond the most recent window |
| Chunked prefill (SGLang, vLLM) | Very long prompts where one-pass prefill would OOM at your target batch size | Short prompts where chunking adds latency overhead for no benefit |
