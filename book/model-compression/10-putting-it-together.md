# 10. Putting it together: the complete build

Every earlier section presented options. This one commits: one stack, the
arithmetic that justifies it, then the same model re-derived under three different
constraint sets, and a runnable planner you can execute with nothing but Python 3.

The scenario is the one from [section 1](01-clarifying-requirements.md): a 70B-class
model must fit one 80GB accelerator without hurting interactive latency, contexts
run to 32K, code generation and tool calling are load-bearing, we have a modest
healing budget but no pretraining budget, and an on-device variant follows later.

## The default stack

| Decision | Committed choice | Why, in one line |
|---|---|---|
| First lever | int8 weight-only, per channel | Nearly lossless, and it is the cheapest way to learn how much headroom the model has |
| Second lever | int4 group-wise (group 128) weights, calibrated on real traffic | Memory is binding, and decode is bandwidth bound so bytes are latency |
| Activations | Left at higher precision initially; fp8 where the kernels exist | Activation quantization needs an outlier strategy and buys prefill, which is not what is binding |
| KV cache | Paged, then 8-bit | At 32K the cache rivals the weights, and 8-bit KV is close to free |
| Precision profile | Embeddings, output projection, first and last blocks, norms kept higher | Small fraction of parameters, large fraction of the damage |
| Structural change | Only if quantization misses the target: width pruning plus distillation from the parent | A healing run is a real budget line; do not spend it unless needed |
| Calibration set | Drawn from serving traffic, including long-context and tool-call shapes | Scales fit what they see; generic web text misfits your workload |
| Acceptance | Paired per item vs the uncompressed parent, per capability, with flip rate, plus measured latency on target hardware at production batch | Averages hide exactly the damage compression causes |
| Rollout | Shadow on live traffic, then canary by slice, with the parent artifact kept warm | The compressed model is a new candidate model, not a config flag |

## The arithmetic that decides it

For a 70B model with 80 layers, 8 key-value heads, and head dimension 128, the two
budgets are:

$$\text{weights} = N \cdot \frac{b_w + 16/g}{8}, \qquad \text{KV per sequence} = 2 \cdot L \cdot h_{kv} \cdot d_h \cdot s \cdot \frac{b_{kv}}{8}$$

The $16/g$ term is the fp16 scale carried per group of $g$ weights, which is why
"4-bit" is really about 4.125 bits at group 128. Running the numbers (the capstone
below prints this table):

| Configuration | Weights | KV at 32K | Total, batch 1 | Fits 80GB | Decode tok/s, b=1 | Decode tok/s, b=32 |
|---|---|---|---|---|---|---|
| fp16 | 141.1 GB | 10.7 GB | 151.8 GB | no | 22 | 218 |
| int8 weights | 71.1 GB | 10.7 GB | 81.8 GB | no | 40 | 255 |
| int4 group-wise | 36.1 GB | 10.7 GB | 46.8 GB | yes | 70 | 278 |
| int4 weights, int8 KV | 36.1 GB | 5.4 GB | 41.5 GB | yes | 80 | 508 |

Three readings, and the third is the one interviews reward. **int8 is not enough**:
at 71 GB of weights plus a 10.7 GB cache it misses an 80 GB card once anything else
is resident, which is what forces int4 rather than a preference for it. **The
speedup depends on where you measure**: fp16 to int4 is over 3x at batch one and
about 1.3x at batch 32, and serving cost lives in the second number. **KV is not a
footnote**: quantizing the cache to 8-bit does more for batch-32 throughput than any
further bit shaving on the weights, because it is the term that scales with
concurrency.

## Cost and schedule

```text
quantization run (int4 + calibration on 512 traffic samples)   hours, one GPU
per-layer sensitivity sweep (80 blocks x small eval)           ~1 day, one GPU
acceptance eval (paired, 6 capability slices, 2 candidates)    ~$300 of inference
shadow run (1% of live traffic, 3 days)                        serving cost only
--------------------------------------------------------------------------------
if the target is missed and structural change is needed:
  width pruning to ~35B + distillation from the parent         2-5B tokens of healing
                                                               days on a small cluster
```

The asymmetry is the point: the quantization path is hours and a few hundred
dollars, the structural path is days and a cluster. Try the cheap path first and let
the acceptance test tell you whether you need the expensive one.

## The same model under three constraint sets

**Cost-down on a server fleet, high batch.** The binding resource is dollars per
million tokens at batch 64 or more, where the machine is compute bound and weight-only
quantization has almost stopped paying. Switch levers: fp8 weights and activations
where the hardware has fp8 tensor cores, so the arithmetic itself gets cheaper, plus
quantized paged KV to raise achievable batch. If that is not enough, the answer is a
structurally smaller model (prune plus distill), because at high batch you need fewer
FLOPs, not fewer bytes. Check the free levers first: continuous batching and prefix
caching often beat everything above and cost no quality.

**Latency-critical interactive, small batch.** The binding resource is per-token
latency at batch one, which is the best case for weight-only quantization. Go int4
group-wise with a fused kernel, quantize KV to 8-bit, and verify the kernel is
actually fused rather than dequantize-then-matmul. Add speculative decoding, which
attacks the same bandwidth-bound phase from the other side and costs no quality.
Sensitivity-driven mixed precision matters more here than elsewhere because you have
the memory headroom to spend on the sensitive layers.

**On-device.** The binding resources are a hard memory ceiling shared with the OS and
an accelerator with a short list of fast formats, at batch one. A compressed 70B does
not fit any ceiling worth discussing, so the design changes shape: distil a small
student (single-digit billions of parameters) from the parent, quantize it to the
format the NPU accelerates rather than the format that scores best on paper, and ship
task-specific adapters over one resident base so several product features share the
memory. Keep the server model for requests that exceed the device, and evaluate the
handoff as its own product decision. This is the published shape of shipped consumer
stacks ([Apple Intelligence Foundation Language Models](https://arxiv.org/abs/2407.21075)).

## The smallest runnable experiment

One file, standard library only. It answers the three questions a compression
proposal has to answer: does it fit and what does a token cost, what did the bits
actually cost in representation error, and is "same accuracy" the same model.

```python
"""Compression planner on one page. Python 3, standard library only.
Illustrative hardware figures, not a benchmark."""

import random
from math import sqrt

random.seed(11)

N = 70e9                    # parameters
LAYERS, KV_HEADS, HEAD_DIM = 80, 8, 128        # grouped-query attention
HBM_BYTES, HBM_BW = 80e9, 3.3e12               # one accelerator: capacity, bytes/s
GROUP = 128                                     # weights per quantization scale


def weight_bytes(bits):
    """Nominal bits plus the fp16 scale carried per group: the real bit budget."""
    return N * (bits + 16 / GROUP) / 8


def kv_bytes(seq, batch=1, bits=16):
    """Two tensors (K and V) per layer, per KV head, per token."""
    return 2 * LAYERS * KV_HEADS * HEAD_DIM * seq * batch * bits / 8


def decode_tokens_per_s(w_bytes, kv_per_seq, batch):
    """Decode is bandwidth bound: read all weights once, plus each sequence's KV."""
    bytes_per_step = w_bytes + batch * kv_per_seq
    return batch * HBM_BW / bytes_per_step


print("fit and cost, 70B model, 32K context")
print(f"{'weights':>18} {'KV/seq':>9} {'total b=1':>10} {'fits 80GB':>10} "
      f"{'tok/s b=1':>10} {'tok/s b=32':>11}")
for label, wbits, kvbits in [("fp16", 16, 16), ("int8", 8, 16),
                             ("int4 group", 4, 16), ("int4 + int8 KV", 4, 8)]:
    w = weight_bytes(wbits)
    kv = kv_bytes(32_768, bits=kvbits)
    total = w + kv
    print(f"{label:>18} {w/1e9:6.1f} GB {kv/1e9:6.1f} GB {total/1e9:7.1f} GB "
          f"{'yes' if total < HBM_BYTES else 'NO':>10} "
          f"{decode_tokens_per_s(w, kv, 1):10.0f} "
          f"{decode_tokens_per_s(w, kv, 32):11.0f}")
print("note: b=1 is the number quantization demos quote (fp16 -> int4 is over 3x here);")
print("      b=32 is the number that sets serving cost (the same change is ~1.3x).")
print("      b=32 also assumes 32 KV caches fit, which at 32K context they do not:")
print("      that is the argument for quantizing KV, not more bits off the weights.")

W = [random.gauss(0, 0.02) for _ in range(8192)]          # a stand-in weight block
for i in random.sample(range(len(W)), 12):                # a few outlier channels
    W[i] *= 25


def rel_err(orig, approx):
    num = sqrt(sum((a - b) ** 2 for a, b in zip(orig, approx)))
    den = sqrt(sum(a * a for a in orig))
    return num / den


def quantize(w, bits, group=GROUP):
    out = []
    for i in range(0, len(w), group):
        g = w[i:i + group]
        qmax = 2 ** (bits - 1) - 1
        s = max(abs(x) for x in g) / qmax or 1e-12       # one scale per group
        out += [s * max(-qmax - 1, min(qmax, round(x / s))) for x in g]
    return out


def prune_unstructured(w, keep=0.5):
    cutoff = sorted((abs(x) for x in w), reverse=True)[int(len(w) * keep) - 1]
    return [x if abs(x) >= cutoff else 0.0 for x in w]


def prune_2_of_4(w):
    out = []
    for i in range(0, len(w), 4):                        # keep the 2 largest of every 4
        g = list(w[i:i + 4])
        keep = sorted(range(len(g)), key=lambda j: -abs(g[j]))[:2]
        out += [g[j] if j in keep else 0.0 for j in range(len(g))]
    return out


print()
print("representation error on the same weight block (lower is better)")
for bits in (8, 4, 3):
    print(f"  int{bits} group-wise ({GROUP}) : {rel_err(W, quantize(W, bits)):.4f}")
print(f"  int4, group=32          : {rel_err(W, quantize(W, 4, 32)):.4f}")
print(f"  50% unstructured prune  : {rel_err(W, prune_unstructured(W)):.4f}"
      "   (no speedup on dense kernels)")
print(f"  2:4 semi-structured     : {rel_err(W, prune_2_of_4(W)):.4f}"
      "   (this one the hardware can skip)")

n, correct = 1000, 720
base = [1] * correct + [0] * (n - correct)
random.shuffle(base)
comp = list(base)
flip_each = 60                       # equal counts both ways: accuracy is preserved
rights = [i for i, v in enumerate(comp) if v == 1]
wrongs = [i for i, v in enumerate(comp) if v == 0]
for i in random.sample(rights, flip_each):
    comp[i] = 0
for i in random.sample(wrongs, flip_each):
    comp[i] = 1
flips = sum(1 for a, b in zip(base, comp) if a != b)
print()
print(f"baseline accuracy   : {sum(base)/n:.3f}")
print(f"compressed accuracy : {sum(comp)/n:.3f}   (delta {abs(sum(comp)-sum(base))/n:.3f})")
print(f"flip rate           : {flips/n:.3f}   <- the number the average hides")
```

Output:

```text
fit and cost, 70B model, 32K context
           weights    KV/seq  total b=1  fits 80GB  tok/s b=1  tok/s b=32
              fp16  141.1 GB   10.7 GB   151.8 GB         NO         22         218
              int8   71.1 GB   10.7 GB    81.8 GB         NO         40         255
        int4 group   36.1 GB   10.7 GB    46.8 GB        yes         70         278
    int4 + int8 KV   36.1 GB    5.4 GB    41.5 GB        yes         80         508
note: b=1 is the number quantization demos quote (fp16 -> int4 is over 3x here);
      b=32 is the number that sets serving cost (the same change is ~1.3x).
      b=32 also assumes 32 KV caches fit, which at 32K context they do not:
      that is the argument for quantizing KV, not more bits off the weights.

representation error on the same weight block (lower is better)
  int8 group-wise (128) : 0.0173
  int4 group-wise (128) : 0.2479
  int3 group-wise (128) : 0.3662
  int4, group=32          : 0.1385
  50% unstructured prune  : 0.2097   (no speedup on dense kernels)
  2:4 semi-structured     : 0.2858   (this one the hardware can skip)

baseline accuracy   : 0.720
compressed accuracy : 0.720   (delta 0.000)
flip rate           : 0.120   <- the number the average hides
```

Four things to take from it. **int8 does not fit and int4 does**, which is what
forces the aggressive setting rather than a preference for it. **The speedup is a
function of batch**, over 3x at batch one and about 1.3x at batch 32, which is why a
demo number and a production number can both be honest and disagree. **A few outlier
weights dominate the error**: int8 is nearly lossless at 0.017 relative error while
int4 at group 128 is 0.248, and simply shrinking the group to 32 halves it, which is
the cheapest knob in the chapter and the reason group size belongs in every
quantization report. **2:4 sparsity costs more error than unstructured at the same
50 percent** (0.286 versus 0.210), and it is the one the hardware can actually skip
work for, which is the tradeoff that decides pruning shape. And the last block is the
acceptance test in miniature: identical accuracy, 12 percent of items flipped.
