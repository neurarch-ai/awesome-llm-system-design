# 10. 把它们拼起来：完整的方案

前面每一节给的都是选项。这一节要做决定：一套技术栈，支撑它的算术账，然后把同一个
模型放在三组不同的约束下重新推一遍，最后是一个能跑的规划器，除了 Python 3 之外什么都不用装。

场景就是[第 1 节](01-clarifying-requirements.md)那个：一个 70B 级的模型必须装进
一张 80GB 的加速器，同时不能伤到交互延迟，上下文最长到 32K，代码生成和工具调用是
承重的，我们有一笔不大的修复预算但没有预训练预算，而端侧那一版会在稍后跟进。

## 默认技术栈

| 决策 | 定下来的选择 | 一句话理由 |
|---|---|---|
| 第一根杠杆 | int8 纯权重，逐通道 | 几乎无损，而且这是摸清模型还有多少余量最便宜的办法 |
| 第二根杠杆 | int4 分组（group 128）权重，用真实流量校准 | 内存是卡住我们的那个资源，而 decode 受带宽限制，字节数直接就是延迟 |
| 激活 | 初期留在较高精度；有 kernel 的地方上 fp8 | 激活量化得先有一套离群值策略，而且它买到的是 prefill 的提速，可卡住我们的并不是 prefill |
| KV cache | 先分页，再 8-bit | 32K 下这个 cache 已经和权重一个量级了，而把 KV 压到 8-bit 几乎不花什么代价 |
| 精度画像 | embedding、输出投影、第一个和最后一个 block、norm 留高精度 | 占参数的一小部分，占损伤的一大部分 |
| 结构改变 | 只在量化没够到目标时才做：宽度剪枝加从父模型蒸馏 | 修复训练是预算表上实打实的一行；不到真有必要就别花它 |
| 校准集 | 从服务流量里取，包含长上下文和工具调用形态 | scale 是照着它看到的东西拟合出来的；通用网页文本对不上你真实的负载 |
| 验收 | 对照未压缩的父模型逐题配对、分能力、带翻转率，加上在目标硬件按生产 batch 实测的延迟 | 平均分盖住的恰恰是压缩造成的那种损伤 |
| 放量 | 线上流量影子跑，然后按切片灰度，父模型产物保持热的 | 压缩后的模型是一个新的候选模型，不是一个配置开关 |

## 决定这件事的算术账

对一个 80 层、8 个 key-value 头、头维度 128 的 70B 模型，两笔预算是：

$$\text{weights} = N \cdot \frac{b_w + 16/g}{8}, \qquad \text{KV per sequence} = 2 \cdot L \cdot h_{kv} \cdot d_h \cdot s \cdot \frac{b_{kv}}{8}$$

$16/g$ 那一项是每 $g$ 个权重带的那个 fp16 scale，这也是为什么 group 128 下的
"4-bit"其实是 4.125 bit。把数字代进去（下面的 capstone 会把这张表打出来）：

| 配置 | 权重 | 32K 下的 KV | 合计，batch 1 | 装得进 80GB | decode tok/s，b=1 | decode tok/s，b=32 |
|---|---|---|---|---|---|---|
| fp16 | 141.1 GB | 10.7 GB | 151.8 GB | 否 | 22 | 218 |
| int8 权重 | 71.1 GB | 10.7 GB | 81.8 GB | 否 | 40 | 255 |
| int4 分组 | 36.1 GB | 10.7 GB | 46.8 GB | 是 | 70 | 278 |
| int4 权重，int8 KV | 36.1 GB | 5.4 GB | 41.5 GB | 是 | 80 | 508 |

三个读数，其中第三个是面试里最加分的。**int8 不够**：71 GB 的权重加上 10.7 GB 的
cache，只要卡上还驻着别的东西，它就装不进一张 80 GB 的卡，是这一点逼出了 int4，而不是
我们偏爱 int4。**提速多少取决于你在哪儿测**：fp16 到 int4 在 batch 1 下是三倍多，在 batch 32
下大约 1.3 倍，而服务成本活在第二个数字里。**KV 不是脚注**：把 cache 量化到 8-bit
对 batch-32 吞吐的贡献，超过在权重上再抠掉任何一个 bit，因为它才是那个会随着并发
一起长起来的项。

## 成本与排期

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

不对称正是重点：量化那条路是几小时加几百美元，结构那条路是几天加一个集群。先走
便宜的那条路，然后让验收测试来告诉你，到底需不需要走贵的那条。

## 同一个模型在三组约束下

**服务器集群降成本，高 batch。**卡住的资源是 batch 64 以上时每百万 token 多少钱，
那里机器受算力限制，纯权重量化已经基本不再回本。换杠杆：硬件有 fp8 tensor core
的话就上 fp8 权重加激活，让算术本身变便宜，再加量化的分页 KV 来抬高能跑到的 batch。
还不够的话，答案是一个结构上更小的模型（剪枝加蒸馏），因为高 batch 下你需要的是
更少的 FLOPs，不是更少的字节。先把免费的杠杆查一遍：连续批处理和前缀缓存经常比
上面这些都强，而且不花质量。

**延迟敏感的交互式，小 batch。**卡住的资源是 batch 1 下的每 token 延迟，而这正是
纯权重量化的最好情况。上 int4 分组配融合 kernel，把 KV 量化到 8-bit，并且确认这个
kernel 真的是融合的而不是先反量化再矩阵乘。再加上投机解码，它从另一侧攻击同一个
受带宽限制的阶段，也不花质量。敏感度驱动的混合精度在这里比在别处更重要，因为你有
内存余量可以花在敏感层上。

**端侧。**卡住的资源是一个和操作系统共享的硬内存上限，加上一块只有一小串快格式的
加速器，batch 恒为 1。一个压缩过的 70B 装不进任何值得讨论的上限，所以设计整个换了
形状：从父模型蒸馏出一个小学生（个位数十亿参数），把它量化到 NPU 能加速的格式而
不是纸面上分数最高的格式，并在一个常驻底座上发任务专属的 adapter，让好几个产品
功能共享这块内存。超出设备能力的请求留给服务端模型，而这次交接本身要当作一个独立的
产品决策去评估。这就是已公开的消费级技术栈上线时的形状
（[Apple Intelligence Foundation Language Models](https://arxiv.org/abs/2407.21075)）。

## 最小可跑实验

一个文件，只用标准库。它回答一份压缩提案必须回答的三个问题：装不装得下、一个 token
值多少钱，那些 bit 在表示误差上到底花了多少，以及"准确率一样"是不是同一个模型。

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

输出：

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

从里面带走四件事。**int8 装不下而 int4 装得下**，是这一点逼出了这个激进的设置，
而不是我们偏爱它。**提速是 batch 的函数**，batch 1 下三倍多，batch 32 下约 1.3 倍，
这就是为什么一个 demo 数字和一个生产数字可以都很诚实却互相矛盾。**少数几个离群
权重主导了误差**：int8 相对误差 0.017，几乎无损，而 group 128 的 int4 是 0.248，
把 group 缩到 32 就直接减半，这是本章最便宜的旋钮，也是为什么每一份量化报告里都
该写上 group size。**同样 50% 下，2:4 稀疏的误差比非结构化更大**（0.286 对 0.210），
而它恰恰是硬件真能为之省下工作量的那一个，正是这个取舍决定了剪枝该取什么形状。
最后那一段则是整套验收测试的缩微版：两个模型的准确率一模一样，却有 12% 的题翻了面。
