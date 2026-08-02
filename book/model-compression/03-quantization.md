# 3. Quantization

## The mechanics in one screen

Quantization maps a float tensor onto a small integer grid plus a scale. Symmetric
uniform quantization to $b$ bits:

$$s = \frac{\max |w|}{2^{b-1} - 1}, \qquad q = \text{round}\!\left(\text{clip}\left(\frac{w}{s},\, -2^{b-1},\, 2^{b-1}-1\right)\right), \qquad \hat w = s \cdot q$$

Everything interesting is in three choices around that formula.

**What you quantize.** Weights only (the default, easiest, and the one that fixes
memory and decode bandwidth), weights and activations (needed to actually use
low-precision compute units, much harder because activations have outliers), or the
KV cache (the term that grows with context).

**Granularity.** One scale per tensor is cheapest and worst; per output channel is
the usual middle; group-wise (a scale per 64 or 128 contiguous weights) is what
makes 4-bit viable at all, at the cost of storing the scales, which is why "4-bit"
is really about 4.25 to 4.5 effective bits.

**Where the error goes.** Rounding error is not the problem. The problem is that a
few coordinates are enormous, so a single scale chosen to cover them crushes the
precision available to everything else.

```python
def quantize_group(w, bits=4, group=128):     # w: list of floats, symmetric group-wise
    out = []
    for i in range(0, len(w), group):
        g = w[i:i + group]
        qmax = 2 ** (bits - 1) - 1
        s = max(abs(x) for x in g) / qmax or 1e-8       # one scale per group
        out += [s * max(-qmax - 1, min(qmax, round(x / s))) for x in g]  # quantize then dequantize
    return out                                  # returns the reconstructed weights
# smaller `group` -> less error, more scale overhead; group=128 is the common compromise
```

## The outlier problem, and the four families of fix

Activation distributions in large transformers contain a small number of channels
whose magnitudes are orders of magnitude larger than the rest, and those channels
are functionally important, so you can neither scale for them nor drop them. Every
serious method is a different answer to that fact.

| Family | Idea | Representative work |
|---|---|---|
| Keep outliers in high precision | Decompose the matmul: outlier channels in fp16, the rest in int8 | [LLM.int8()](https://arxiv.org/abs/2208.07339) |
| Move the difficulty to the weights | Per-channel scaling that migrates activation outliers into the (easier) weight tensor | [SmoothQuant](https://arxiv.org/abs/2211.10438) |
| Protect the weights that matter | Scale salient weight channels, identified by activation magnitude, before rounding | [AWQ](https://arxiv.org/abs/2306.00978) |
| Compensate the error you cause | Layer-wise second-order (Hessian-aware) rounding that pushes each rounding error into the not-yet-quantized weights | [GPTQ](https://arxiv.org/abs/2210.17323) |
| Rotate the outliers away | Apply an orthogonal rotation so no coordinate is an outlier, quantize in the rotated basis | [QuaRot](https://arxiv.org/abs/2404.00456), [SpinQuant](https://arxiv.org/abs/2405.16406) |

The last row is the one that unlocked 4-bit *activations* and KV, not just weights:
because a rotation is invertible and can be folded into adjacent matrices, you get a
mathematically equivalent network whose tensors happen to be outlier-free. If you
name only one recent development in an interview, name this one, and note the
practical caveat: the rotation has to be fused into the model so it costs nothing at
runtime.

## What each format is for

| Format | Typical use | Quality behavior | Speedup mechanism |
|---|---|---|---|
| int8 weight-only | Safe default when memory is binding | Nearly lossless with per-channel scales | Halves weight bytes; decode-bandwidth win |
| int4 group-wise weight-only (GPTQ, AWQ) | Fitting a model that otherwise does not fit | Small average loss, concentrated in the tail; needs a good calibration set | Quarters weight bytes; large decode win at small batch |
| fp8 (e4m3) weights and activations | Modern accelerators with fp8 tensor cores | Better outlier tolerance than int8 at the same width | Real compute speedup, so it helps prefill and high-batch too |
| 4-bit weights plus 4-bit activations (rotation-based) | Aggressive serving on supported kernels | Workable now, still the frontier; verify per capability | Compute and bandwidth |
| nf4 (QLoRA-style) | Training adapters over a frozen quantized base | Designed for a normally-distributed weight prior | Memory during fine-tuning, not a serving format |
| 2-bit and below | Research, or ternary models trained that way | Post-training 2-bit degrades sharply | Only viable if trained for it ([BitNet b1.58](https://arxiv.org/abs/2402.17764)) |

The BitNet line is worth understanding precisely because it is usually cited
wrongly: 1.58-bit models are **trained** in that regime; you cannot post-training
quantize a normal model to ternary and keep it.

## Quantizing the KV cache

For long contexts the KV cache, not the weights, is what you are paying for. Two
properties make it a different problem from weight quantization: the cache is
written incrementally at serving time (so the quantizer must be cheap and online),
and keys and values have different distributions. The practical recipe from the
literature is asymmetric: quantize keys per channel and values per token, which
respects where the outliers actually live ([KIVI](https://arxiv.org/abs/2402.02750)).

Order of operations for long-context serving, from biggest to smallest win:

1. **Architectural KV reduction** (GQA, MQA, MLA) if you get to choose the model.
2. **Paged KV** so the memory you saved is not lost to fragmentation.
3. **KV quantization** to 8-bit, which is close to free, then 4-bit with per-capability testing.
4. **Eviction or windowing** last, because it changes what the model can attend to.

Detail on all four lives in the [KV cache chapter](../kv-cache/03-shrinking-the-cache.md).

## PTQ, QAT, and the thing in between

**Post-training quantization (PTQ)** needs only a calibration set of a few hundred
samples and produces a quantized artifact in hours. It is the default.

**Quantization-aware training (QAT)** simulates the quantizer inside the forward
pass during training, so the weights learn to be robust to it. It recovers most of
the quality gap at aggressive bit widths, and it costs a training run. Combine it
with distillation from the full-precision teacher and it is usually the strongest
option at 4-bit and below.

**QLoRA is neither.** Training a LoRA adapter over a frozen 4-bit base is a
memory-saving *fine-tuning* technique; the base is quantized before training and
stays quantized, and the adapter is trained in higher precision. Saying "we used
QLoRA to quantize the model for serving" is a common confusion worth avoiding.

## When to use which

| Reach for | When | Instead of |
|---|---|---|
| int8 weight-only, per channel | The model nearly fits, and you want the safest win | Jumping to 4-bit and spending the quality budget you did not need |
| int4 group-wise with GPTQ or AWQ | Memory is binding and decode latency matters at small batch | Unstructured pruning, which saves memory without a speedup |
| fp8 weights and activations | The accelerator has fp8 tensor cores and prefill or high-batch throughput is binding | Weight-only quantization, which leaves the FLOPs unchanged |
| Rotation-based methods (QuaRot, SpinQuant) | You need 4-bit activations or 4-bit KV, not just 4-bit weights | Accepting that activation quantization "does not work" |
| KV quantization | Long contexts, large batch, or both | Compressing weights further while KV dominates the footprint |
| QAT plus distillation | Below 4 bits, or a device format with no PTQ recipe that holds quality | PTQ plus hope |
| Mixed precision by layer | One capability regressed after an otherwise good quantization | Reverting the whole quantization |

**Provenance.** The outlier-decomposition line is LLM.int8() (University of Washington and Meta AI, 2022); the migration and salience lines are SmoothQuant (MIT and NVIDIA, 2022) and AWQ (MIT, 2023); the error-compensation line is GPTQ (IST Austria and ETH Zurich, 2022); the rotation line is QuaRot (2024) and SpinQuant (Meta, 2024); ternary training is BitNet b1.58 (Microsoft Research, 2024); KV-specific asymmetry is KIVI (2024).

**Tools.** PTQ recipes ship in llm-compressor and AutoGPTQ / AutoAWQ, with GGUF plus llama.cpp covering the CPU and on-device path; serving runtimes that consume the artifacts include vLLM, SGLang, and TensorRT-LLM (NVIDIA), and bitsandbytes provides the nf4 path used by QLoRA. Format support, not the algorithm, is usually what decides the choice: check which quantized kernels your runtime and accelerator actually have before selecting a method.

**Worked example.** A team serving an interactive assistant on one accelerator starts with int8 weight-only per-channel quantization because memory is binding and int8 is nearly lossless, then finds it still does not fit and moves to int4 group-wise with AWQ, drawing the calibration set from real serving traffic rather than generic web text so the scales match the code and long-document shapes they actually see. Decode latency drops roughly with the weight bytes, as expected for a bandwidth-bound phase, while time-to-first-token barely moves, which is why they do not bother quantizing weights further for prefill and instead enable fp8 compute where the kernels support it. Contexts run to 32K, so they quantize the KV cache to 8-bit on top, keeping paged attention so the saved memory turns into batch size. The acceptance test catches a regression in JSON tool-call validity, which they fix by keeping the final projection and the attention output at higher precision rather than by abandoning int4.

## Implementation pitfalls

| Problem | Symptom | Fix |
|---|---|---|
| Calibration set mismatched to serving traffic | Quality fine on generic prompts, degraded on code, long documents, or a specific language | Draw calibration data from real traffic, including long-context and tool-call shapes |
| Quantizing everything uniformly | One capability collapses while the average looks fine | Keep sensitive layers (first and last blocks, attention output projection, norms and embeddings) at higher precision |
| Activation quantization without an outlier strategy | Loss explodes or output becomes incoherent | Use smoothing, salience-aware scaling, or rotations; do not simply lower the bit width |
| 4-bit with a large group size | Larger-than-expected degradation | Reduce the group (128 to 64), and account for scale overhead in the real bit budget |
| Measuring the win on a benchmark, not the hardware | Predicted 2x, measured 1.1x | Benchmark on target hardware; the kernel, not the bit width, sets the speed |
| Assuming the win holds at high batch | Great at batch 1, unchanged at batch 64 | Re-measure at production batch; the phase moves toward compute bound |
| KV left at fp16 on long contexts | Memory dominated by the cache; quantized weights bought little | Quantize and page the KV; consider a GQA or MLA model instead |
