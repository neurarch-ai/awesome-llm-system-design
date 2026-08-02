# 17 - Model compression

> **Interviewer:** "This model is too big and too slow for where we need to run it.
> Make it fit, and tell me what it costs us."

There is a wrong answer that sounds right ("quantize to 4-bit, it is basically
free") and a right answer that is three sentences longer: which resource is
actually binding, which lever moves that resource on *your* hardware, and how you
prove the quality you gave up is quality you can afford.

The book edition, with worked figures and a runnable planner, is
[book/model-compression/](../book/model-compression/).

## 1. Clarify and scope

- **What is binding?** A memory ceiling, decode latency, prefill or high-batch
  throughput, and long-context cost are four different problems with four different
  answers.
- **What is the serving shape?** Batch one interactive or large batches. That
  decides whether you are memory-bandwidth bound or compute bound.
- **How long are the contexts?** Past a few thousand tokens the KV cache starts to
  rival the weights, and compressing weights while ignoring KV solves the wrong half.
- **Can we retrain?** Post-training only, a healing budget of billions of tokens, or
  a real training run. This splits the method space in half.
- **Which capabilities cannot move?** "No regression" is not a bar. Compression
  damage is never uniform.
- **What does the target accelerate?** int4 weight-only, fp8 compute, 2:4 sparsity,
  and an on-device NPU's supported formats are four different answers.

## 2. Requirements

**Functional.** Fit the target, hold the named capabilities, and ship an artifact
someone can serve with the runtime you actually deploy.

**Non-functional.** Measured (not predicted) latency and memory on target hardware
at production batch and context; a paired quality comparison against the
uncompressed baseline; a rollback artifact kept warm.

**Two consequences to state early.**

1. **Compression is a hardware question wearing an algorithm's clothes.** A format
   the accelerator does not accelerate is a memory technique, not a speed technique.
2. **Top-line accuracy is the wrong acceptance test.** Compression scatters behavior
   rather than shifting it, so two models can score the same and disagree on a large
   fraction of items. Report a **flip rate**.

## 3. Where the bytes and microseconds go

$$\text{bytes} \approx \underbrace{N b_w}_{\text{weights}} + \underbrace{2 L h_{kv} d_h s B b_{kv}}_{\text{KV cache}} + \text{activations}$$

- **Prefill** processes the prompt at once, reusing each weight across many tokens:
  compute bound, time tracks FLOPs.
- **Decode** produces one token at a time: memory-bandwidth bound, time tracks bytes
  read, so halving weight bytes nearly halves decode latency at small batch.

```mermaid
flowchart TD
  START["too big / slow / expensive"] --> Q{"what is binding?"}
  Q -->|"memory ceiling"| MEM["quantize weights"]
  Q -->|"decode latency"| DEC["quantize weights + KV"]
  Q -->|"prefill / high batch"| PRE["low-precision compute (fp8)<br/>or fewer FLOPs"]
  Q -->|"long context"| KV["GQA or MLA -> paged KV -> quantized KV"]
  MEM --> GATE["acceptance test"]
  DEC --> GATE
  PRE --> GATE
  KV --> GATE
  GATE -->|"one capability regressed"| MIX["raise precision on the<br/>sensitive layers, re-test"]
  MIX --> GATE
  GATE -->|"still short"| SMALL["structured prune<br/>+ distill from the parent"]
  SMALL --> GATE
  GATE -->|"passes"| SHIP["ship as a new candidate model"]
```

## 4. Deep dives

### Quantization

Symmetric uniform quantization is
$s = \max|w| / (2^{b-1}-1)$, $q = \text{round}(w/s)$, and everything interesting is
in three choices: **what** you quantize (weights only, weights plus activations, or
the KV cache), **granularity** (per tensor, per channel, or group-wise, which is what
makes 4-bit viable and why "4-bit" is really about 4.125 bits at group 128), and
**where the error goes**.

The difficulty is outliers: a few activation channels carry values orders of
magnitude larger than the rest and are functionally important, so a scale wide
enough for them starves everything else. Four families of fix:

| Family | Idea | Representative |
|---|---|---|
| Keep outliers high precision | Decompose the matmul | LLM.int8() |
| Move the difficulty to weights | Per-channel scaling migrates outliers | SmoothQuant |
| Protect salient weights | Scale channels chosen by activation magnitude | AWQ |
| Compensate the error | Layer-wise second-order rounding | GPTQ |
| Rotate outliers away | Orthogonal rotation, fused at export | QuaRot, SpinQuant |

The rotation family is what unlocked 4-bit *activations* and KV, not just weights.
For long contexts, quantize the cache asymmetrically (keys per channel, values per
token), and remember the order of operations: architectural KV reduction (GQA, MLA)
first, then paging, then quantization, then eviction or windowing last.

**PTQ, QAT, and QLoRA are three different things.** PTQ produces a serving artifact
in hours from a calibration set. QAT simulates the quantizer during training and is
what you escalate to below 4 bits, usually with a distillation loss. QLoRA quantizes
a base in order to fine-tune cheaply; it is not a serving quantization.

### Pruning

Whether removing weights buys anything depends on the *shape*:

| Shape | Quality at 50 percent | Speed | Memory |
|---|---|---|---|
| Unstructured | Best | None on dense kernels | Only with sparse storage |
| 2:4 semi-structured | Worse than unstructured | Real, on sparse tensor cores | About half |
| Structured (heads, channels, layers) | Needs a healing run | Real everywhere | A genuinely smaller model |

Selection criteria matter: magnitude alone is weak, and the activation-aware score
$S_{ij} = |W_{ij}|\cdot\lVert X_j\rVert_2$ (Wanda) is competitive with
reconstruction-based one-shot pruning (SparseGPT) at almost no cost. For structured
pruning, width degrades gracefully while depth buys more latency per unit removed
and damages multi-step behavior disproportionately. The production recipe when you
have no pretraining budget: **prune the parent to the target shape, then distill the
parent into it** (Sheared LLaMA, Minitron).

### Distillation

Soft targets carry the teacher's relative preferences over wrong answers, which is a
far denser signal per token than hard labels. Offline (teacher-scored corpus),
sequence-level, and on-policy (student generates, teacher scores) differ in whether
the student ever sees its own mistakes; on-policy is strongest and makes the teacher
part of your training infrastructure. Two side effects: a student distilled from
long reasoning traces inherits their length, and it inherits the teacher's
contamination, which no cleaning of your corpus can undo.

### Serving a compressed model

- **The win depends on batch size.** Weight-only quantization buys bandwidth, not
  arithmetic: over 3x at batch one and closer to 1.15x at batch 64 for the same
  change.
- **Kernel support is the real constraint.** An unfused dequantize-then-matmul path
  can be net slower. Assert the numeric path at startup, because several stacks fall
  back to higher precision silently.
- **Mixed precision by layer is the standard fix.** Embeddings, the output
  projection, the first and last blocks, and norms stay higher; per-layer
  sensitivity decides the rest.
- **On-device is a different problem**: a hard memory ceiling shared with the OS, a
  short list of fast formats, and batch one. The shipped pattern is a small distilled
  student, quantized to the NPU's format, with task adapters over one resident base
  and a server model for the rest.

## 5. Bottlenecks and scaling

| Bottleneck | First sign | Fix |
|---|---|---|
| Dequantization overhead | Quantized model slower than expected at low batch | Fused kernel for your shape and runtime |
| Win evaporates at production batch | Great in the harness, flat in production | fp8 compute or a smaller model |
| KV dominates at long context | Memory grows with traffic despite quantized weights | Quantized plus paged KV, or a GQA / MLA model |
| One capability regressed | Average holds; JSON validity or long-context recall drops | Per-layer sensitivity, raise precision there |
| Adapters degrade on the quantized base | Adapter works on fp16, not on int4 | Train or validate adapters against the served artifact |

## 6. Failure modes

- **Calibration set mismatched to serving traffic**: fine on public benchmarks,
  degraded on your workload, and the public numbers exonerate the artifact.
- **Silent precision fallback**: no speedup, no error, a confusing week.
- **Sparsity quoted without a shape**: "50 percent pruned" that nobody can reproduce
  as a speedup.
- **Skipping the healing run** after structured pruning and blaming pruning.
- **Composing levers without re-evaluating**: pruning and quantization damage the
  same outlier-sensitive paths, so two levers that each cost a point can cost four.

## 7. Likely follow-ups

- "int4 is 4x cheaper, right?" Bytes yes, FLOPs no, and the win shrinks with batch.
- "50 percent sparse is 2x faster?" Only if the shape is one the hardware can skip.
- "Accuracy held, so it was free?" Report the flip rate.
- "Can we PTQ to 2 bits since 1.58-bit models exist?" Those are trained that way.
- "Which order: prune, distill, quantize?" Prune, heal or distill, then quantize,
  and evaluate the composition.
- "A compressed 70B or a native 8B?" Run both through the same acceptance test; the
  third option (distill the parent into the small model) usually beats both.

## Seen in production

### The shared pipeline

Pick the lever that moves the binding resource, keep a small set of layers at higher
precision, and gate on a paired comparison against the uncompressed model.

### How they differ

| Approach | Primary constraint | Retraining |
|---|---|---|
| Weight-only int4 (GPTQ, AWQ lineage) | Fit and decode latency | None |
| Activation and compute quantization (SmoothQuant, fp8) | Throughput per dollar | None to light |
| Rotation-based low-bit (QuaRot, SpinQuant) | 4-bit activations and KV | None, or learned rotations |
| Train-time ternary (BitNet) | Extreme efficiency | Full pretraining |
| One-shot pruning (SparseGPT, Wanda) | Memory or a sparsity-accelerated target | None |
| Prune plus distill (Minitron, Sheared LLaMA) | A size ladder from one parent | Billions of tokens |
| On-device stacks (Apple Intelligence) | A hard device ceiling and fixed formats | Yes, in model design |
| fp8 end to end (DeepSeek-V3) | Cost across the lifecycle | Designed in |

### The systems

- **LLM.int8()** [8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339)
- **SmoothQuant** [paper](https://arxiv.org/abs/2211.10438), **GPTQ** [paper](https://arxiv.org/abs/2210.17323), **AWQ** [paper](https://arxiv.org/abs/2306.00978)
- **QuaRot** [paper](https://arxiv.org/abs/2404.00456) and **SpinQuant** [paper](https://arxiv.org/abs/2405.16406)
- **Microsoft Research** [The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764)
- **KIVI** [asymmetric 2-bit KV cache quantization](https://arxiv.org/abs/2402.02750)
- **SparseGPT** [paper](https://arxiv.org/abs/2301.00774) and **Wanda** [paper](https://arxiv.org/abs/2306.11695)
- **Princeton** [Sheared LLaMA](https://arxiv.org/abs/2310.06694), **NVIDIA** [Compact Language Models via Pruning and Knowledge Distillation](https://arxiv.org/abs/2407.14679)
- **Google DeepMind** [On-Policy Distillation of Language Models](https://arxiv.org/abs/2306.13649)
- **Apple** [Apple Intelligence Foundation Language Models](https://arxiv.org/abs/2407.21075)
- **DeepSeek** [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
- **Microsoft Research India** [Accuracy is Not All You Need](https://arxiv.org/abs/2407.09141)
- **Red Hat AI and vLLM** [llm-compressor](https://github.com/vllm-project/llm-compressor), **llama.cpp** [GGUF ecosystem](https://github.com/ggml-org/llama.cpp)

## Trace the architectures

- **A dense model where the arithmetic is concrete (Llama 3 8B):**
  [open it live](https://www.neurarch.com/?import=https://raw.githubusercontent.com/neurarch-ai/awesome-llm-model-zoo/main/architectures/llama3-8b/model.json).
  Read off layer count, KV head count, and head dimension, then plug them into the
  weight and KV formulas above. That is the whole memory plan, and it is the fastest
  way to see why a long context makes the cache rival the weights.

  ![Llama 3 8B](https://raw.githubusercontent.com/neurarch-ai/awesome-llm-model-zoo/main/architectures/llama3-8b/assets/diagram.png)

- **Where architecture beats compression (DeepSeek-V3):**
  [open it live](https://www.neurarch.com/?import=https://raw.githubusercontent.com/neurarch-ai/awesome-llm-model-zoo/main/architectures/deepseek-v3/model.json).
  Latent attention shrinks the KV cache structurally and the MoE layers mean most
  parameters do not run per token, which is the point that compression is what you
  do after the architecture is fixed.

  ![DeepSeek-V3](https://raw.githubusercontent.com/neurarch-ai/awesome-llm-model-zoo/main/architectures/deepseek-v3/assets/diagram.png)

These are validated reference graphs at real dimensions, shape-checked end to end,
not screenshots. All 92 architectures live in the
[Model Zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo)
([gallery](https://neurarch-ai.github.io/awesome-llm-model-zoo)). Built by
[Neurarch](https://www.neurarch.com).

## Related deep-dive drills

Rapid-fire questions that probe the modeling and systems underneath this topic, from [deep-dives.md](../deep-dives.md):

- [Distillation, pruning, and model compression](../deep-dives.md#distillation-pruning-and-model-compression)
- [Inference, quantization, and serving math](../deep-dives.md#inference-quantization-and-serving-math)
- [Scaling: rooflines, parallelism, and the arithmetic of large models](../deep-dives.md#scaling-rooflines-parallelism-and-the-arithmetic-of-large-models)
