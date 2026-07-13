# 8. Interview Q&A

The questions interviewers actually ask about long-context serving, grouped by how
they are used. The "commonly answered wrong" section is where interviews are won
or lost.

## Commonly asked

**Q: Why is decode memory-bandwidth-bound and prefill compute-bound?**

A: In decode, each step reads the entire model (say 14 GB for a 7B model in FP16)
and the entire KV cache just to emit a single token. The ratio of FLOPs done to
bytes read is about 1 FLOPs/byte; a modern H100 needs around 150 FLOPs/byte to
be compute-bound. Decode is always far below that. In prefill, $S$ tokens are
processed at once, so the same weight read is amortized across all $S$ tokens:
arithmetic intensity rises to roughly $S$ FLOPs/byte, crossing the compute-bound
threshold at $S \approx 150$ tokens or higher. The fix you reach for depends on
which phase is the wall, so profile before optimizing.

**Q: What is the KV-cache memory formula, and why does it matter?**

A: $\text{kv-bytes} \approx 2 \cdot L \cdot S \cdot h_{\text{kv}} \cdot d_{\text{head}} \cdot b \cdot B$.
For a single 100k-token session on a 32-layer GQA model (8 KV heads, $d_{\text{head}} = 128$,
FP16, batch 1): $2 \times 32 \times 100000 \times 8 \times 128 \times 2 \approx 13.1$ GB.
The weights of the same model are 14 GB. At 100 concurrent sessions the cache is
1.3 TB; the weights are still 14 GB. The KV cache, not the weights, dominates
long-context memory. All the levers (GQA, MLA, quantization, paging, prefix
caching) attack this formula at some term.

**Q: What is GQA and why is it the default in most modern models?**

A: Grouped-query attention shares one KV head across a group of $g$ query heads,
shrinking $h_{\text{kv}}$ from $h_q$ down to $h_q / g$. With $h_q = 32$ and $g = 4$,
you get $h_{\text{kv}} = 8$ and a 4x smaller cache at near-MHA quality. The conversion
from an existing MHA checkpoint can be done by uptraining for about 5% of pretraining
compute. No serving-stack changes are required. That combination of easy conversion
and large memory saving makes it the default in Llama 3, Mistral, Gemma, and most
post-2023 production models.

**Q: What does PagedAttention solve?**

A: Fragmentation. Without paging, each sequence reserves a contiguous KV buffer at
arrival. As sequences arrive and complete at different times, freed buffers leave
gaps that cannot be filled by different-sized sequences (external fragmentation),
and each buffer may have unused tail space (internal fragmentation). PagedAttention
manages the cache in fixed-size blocks with a per-sequence block table, just like
OS virtual memory. Blocks scatter anywhere in GPU memory; fragmentation approaches
zero. The result is 2x to 4x higher concurrency at matched throughput. It does not
speed up a single request; it packs more concurrent requests into the same GPU.

**Q: Why does prefix caching help first-token latency?**

A: Because it skips prefill. The prefill phase is the compute cost of building the
KV cache for all prompt tokens. When a request shares a prefix (system prompt,
shared document) with a previously cached request, the serving stack reuses those
KV blocks directly: no prefill computation for those tokens. The model jumps
straight to decoding from the first unique token. For a 4k shared system prompt
that hits the cache on every request, first-token latency drops to near the cost
of a single decode step instead of 4000 tokens of prefill.

## Tricky (the follow-ups that separate people)

**Q: GQA vs MLA: when would you pick MLA despite its extra complexity?**

A: When the KV cache is the binding long-context constraint and you control
training. GQA shrinks $h_{\text{kv}}$, which reduces the cache by a fixed ratio
($4x$ to $8x$ typically). MLA replaces the entire K/V storage with a latent of
$d_c$ dimensions, achieving roughly 93% compression versus MHA regardless of head
count. When even GQA leaves you memory-limited at your target context and
concurrency, MLA is the next move. The cost is real: the RoPE split-head design
must be baked in at training time, and the up-projection adds a small compute cost
per decode step. If you serve a fixed off-the-shelf model and cannot retrain, MLA
is not on the table; GQA uptraining or KV quantization is.

**Q: You serve a cluster of GPU nodes, not one. How does prefix caching break?**

A: Single-node prefix caches do not span nodes. If request A builds the cache for
a shared system prompt on node 1 and request B is routed to node 2, B misses the
cache and pays full prefill cost even though the same prefix was just computed.
At cluster scale with round-robin routing, effective hit rate can be much lower
than the single-node rate would suggest. The fix is cache-aware routing: route
requests that share a prefix to the node holding that prefix's blocks. This is
what llm-d's cache-aware scheduling does. It adds routing complexity but recovers
the hit rate benefit of prefix caching at multi-node scale.

**Q: Estimate the KV cache for a Llama 3 8B serving 1000 concurrent sessions at
32k tokens.**

A: Llama 3 8B has $L = 32$ layers, $h_{\text{kv}} = 8$ (GQA), $d_{\text{head}} = 128$, FP16.

$$\text{kv-bytes} = 2 \times 32 \times 32000 \times 8 \times 128 \times 2 = 4.29 \text{ GB per session}$$

At 1000 concurrent sessions: 4.29 TB. A single H100 node (8 GPUs, 80 GB each,
640 GB total) cannot hold this. You need either: (a) aggressive prefix caching
to reduce the effective S per session, (b) KV quantization to cut the per-session
footprint, (c) multiple nodes with cache-aware routing, or (d) session eviction.
Showing you can do this arithmetic on a real model is the strongest possible
answer to a "how would you scale this" follow-up.

**Q: A small draft model can speed up decode via speculative decoding. When does
it not help?**

A: When the target GPU is already saturated (high batch size, memory-bound decode).
Speculative decoding re-uses the target model's parallel forward pass to verify $k$
draft tokens at once, cutting the number of expensive target steps. But this only
helps when those target steps are the bottleneck. At very high batch sizes the GPU
is already close to memory saturation; adding draft-model compute plus longer
verification sequences can increase total memory pressure without clearing the
bottleneck. Speculative decoding wins most cleanly at low-to-moderate batch sizes
on structured or predictable output (code, formulaic responses).

## Commonly answered wrong (the traps)

**Q: Does PagedAttention speed up individual request latency?**

A: No. PagedAttention reduces fragmentation so more sequences fit in GPU memory,
raising aggregate throughput (tokens per second across all requests). The per-step
latency for any individual request is not improved; if anything, the block-table
indirection adds a tiny overhead to the attention kernel. The win is concurrency,
not per-request speed. Report its benefit in fleet-wide tokens per second, not in
single-request decode latency.

**Q: Is the logQ correction (or any calibration offset) applied at serving time
to adjust KV cache attention scores?**

A: No. This conflates two unrelated things. The logQ correction in candidate
retrieval (subtracting the log sampling probability from logits) is a training-time
debiasing applied to the softmax loss, not to serving inference. In KV-cache
serving there is no comparable correction applied at runtime; the KV cache simply
stores the exact keys and values (or their quantized or latent-compressed
equivalents) and attention proceeds normally. If you hear a variant of this
question about "correcting" attention scores at decode time, the answer is the
same: attention in a production serving stack reads the KV entries as stored,
with no score adjustments.

**Q: Can you quantize both the weights and the KV cache independently?**

A: Yes, and they are orthogonal levers with different tradeoffs. Weight
quantization (INT8, INT4 GPTQ, AWQ) shrinks the model footprint and speeds up the
weight-read portion of decode. KV-cache quantization shrinks the cache footprint
and speeds up the cache-read portion. For long-context serving the cache often
becomes larger than the weights, so KV quantization has a larger marginal impact
at those context lengths. You can apply both: many production stacks run INT8
weights with FP8 or INT4 KV. Each requires its own eval because the error modes
are different (weight quant affects all token computations; KV quant degrades
historical attention accuracy for old tokens in the context).

**Q: MLA is just another way to say "fewer KV heads", right?**

A: No. GQA and MQA literally reduce $h_{\text{kv}}$ in the formula: fewer heads
means fewer entries per token, and the mechanism is sharing one KV head across
multiple query heads. MLA takes a fundamentally different approach: it eliminates
the full-rank K and V tensors from the cache entirely, replacing them with a single
low-rank latent vector per token. That latent is then up-projected to per-head keys
and values at attention time. The cache stores a vector of size $d_c$ (say 512)
instead of $2 \times h_{\text{kv}} \times d_{\text{head}}$ (say $2 \times 32 \times 128 = 8192$)
values. Describing it as "fewer heads" mis-states the mechanism and misses the RoPE
complication entirely.
