# 2. Framing the problem

## Where the bytes and the microseconds go

Before choosing a lever, write down what the model spends. Two budgets matter and
they are not the same budget.

**Memory** is dominated by three terms:

$$\text{bytes} \approx \underbrace{N \cdot b_w}_{\text{weights}} + \underbrace{2 \cdot L \cdot h_{kv} \cdot d_{h} \cdot s \cdot B \cdot b_{kv}}_{\text{KV cache}} + \underbrace{\text{activations}}_{\text{transient}}$$

where $N$ is the parameter count, $b_w$ the bytes per weight, $L$ layers, $h_{kv}$
key-value heads, $d_h$ the head dimension, $s$ the sequence length, $B$ the batch,
and $b_{kv}$ the bytes per cached element. The first term is fixed; the second
grows with context and batch and is the one that surprises people.

**Time** splits by phase:

- **Prefill** processes the whole prompt at once, so each loaded weight is reused
  across many tokens. Arithmetic intensity is high, and the phase is
  **compute bound**: time tracks FLOPs, roughly $2 N$ per token of prompt.
- **Decode** produces one token at a time, so every weight is loaded to serve very
  few multiply-accumulates. The phase is **memory-bandwidth bound**: time per token
  is approximately (bytes of weights plus bytes of KV read) divided by achievable
  bandwidth.

That asymmetry is the single most useful thing to know in this chapter, because it
tells you which lever can possibly work:

$$t_{\text{decode}} \approx \frac{N b_w + \text{KV bytes read}}{\text{HBM bandwidth}} \quad\Longrightarrow\quad \text{halving } b_w \text{ nearly halves decode time}$$

and equally, why the same trick does almost nothing for time-to-first-token, and
why the win erodes as batch size grows and the phase drifts back toward compute
bound.

## The four levers

| Lever | What it changes | Buys you | Costs you | Needs retraining? |
|---|---|---|---|---|
| Quantization | Fewer bits per weight, activation, or KV element | Memory and decode bandwidth, immediately | Accuracy on the tail; activation quantization is much harder than weight-only | No for PTQ, yes for QAT |
| Pruning and sparsity | Fewer weights, or a structurally smaller model | Memory always; speed only if the shape is one the hardware accelerates | Quality, especially without a healing run | Unstructured no, structured yes |
| Distillation | A genuinely smaller model trained to mimic a bigger one | Everything at once (memory, compute, latency) | A real training budget, and a ceiling set by the student's capacity | Yes, by definition |
| Architectural choice | GQA/MLA, MoE, smaller base, shorter effective context | The biggest wins, without a compression step at all | A different model, so a different eval | Depends |

The fourth row is the one candidates forget. If the KV cache is the problem,
switching to a model with grouped-query or latent attention beats quantizing the KV
cache of a model with 64 key-value heads. Compression is what you do after the
architecture is fixed.

## What is not compression

Three things get lumped in and should be named separately, because they buy speed
without touching quality and are therefore strictly easier: **continuous batching**
(fill the machine), **speculative decoding** (spend spare compute to skip
bandwidth-bound steps), and **caching** (do not recompute a shared prefix). They
are covered in [inference serving](../inference-serving/) and
[cost optimization](../cost-optimization/). Reach for them before you accept any
quality loss, and note in the interview that you did.

## Compare and contrast: the three compression families

All three "make the model smaller," get reported as a percentage, and are casually
called compression, which hides that they fail differently.

| Dimension | Quantization | Pruning | Distillation |
|---|---|---|---|
| Unit removed | Precision per value | Values, or whole structures | Parameters, by construction |
| Typical post-training cost | Hours (calibration set) | Hours for one-shot; billions of tokens to heal structured pruning | A full training run |
| Speedup without retraining | Yes, for decode, where kernels exist | Only for 2:4 or structured shapes | Not applicable |
| Quality damage pattern | Long tail and outlier-sensitive paths (long context, rare languages, exact formats) | Concentrated where capacity was removed; depth pruning hurts multi-step behavior | Whatever the student cannot represent; smoothest of the three |
| Composability | Composes with everything | Composes, usually applied before quantization | Composes; often the healing step for pruning |
| The trap | "4-bit is 4x faster" | "50 percent sparse is 2x faster" | "The student matches the teacher on the benchmark" |

The differences change the design because the levers are not interchangeable: if
the constraint is a fixed memory ceiling, quantization is the cheapest path; if it
is latency at high batch, you need fewer FLOPs and therefore a structurally smaller
model; and if it is both plus a device that only supports one numeric format, you
are choosing a different model and distilling into it.

## Choosing the first lever from the binding constraint

| Binding constraint | First lever | Why not the others |
|---|---|---|
| Model does not fit in memory | Weight-only quantization (int8, then int4 group-wise) | Pruning saves less per unit of quality risk; distillation costs a training run |
| Decode latency at small batch | Weight-only quantization plus quantized KV | Prefill-side tricks do not touch the bandwidth-bound phase |
| Time-to-first-token on long prompts | Low-precision compute (fp8) and attention kernels, not weight-only tricks | Weight-only quantization leaves the FLOPs unchanged |
| Throughput cost at high batch | Structurally smaller model (prune plus distill) or fp8 compute | Weight-only wins shrink as the phase becomes compute bound |
| Long-context serving cost | KV: architectural first (GQA, MLA), then quantized and paged KV | Weight compression does not touch the term that grows with context |
| On-device memory ceiling and fixed NPU formats | A smaller model distilled for the task, quantized to a format the NPU accelerates, with task adapters | A compressed frontier model rarely fits the ceiling or the format |

## Inputs and outputs

**Input:** a model, a target (device, memory ceiling, latency budget, cost per
million tokens), a calibration set that matches the serving distribution, and an
evaluation portfolio with a per-capability bar.

**Output:** a compressed artifact plus the two things that make it shippable: a
**measured** latency and memory profile on the target hardware, and a paired
per-item quality comparison against the uncompressed baseline that reports flips,
not just the accuracy delta.

The calibration set deserves one sentence of its own, because it is where quiet
failures come from: post-training quantization fits scales to the activations it
sees, so a calibration set drawn from generic web text will misfit a model that
serves code, long documents, or a non-English language. Draw calibration data from
the serving distribution, and include the long-context and tool-calling shapes you
care about.
