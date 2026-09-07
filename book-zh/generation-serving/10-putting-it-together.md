# 10. 把它们拼起来：完整的方案

## 一套默认技术栈：从这里开始，要偏离就说清理由

对一个没有特殊约束的交互式文生图产品，这就是答案，以及每一块为什么在里面。

| 层 | 默认选择 | 为什么 | 什么时候偏离 |
|---|---|---|---|
| 基座模型 | 一个 1024px 的开源 latent 扩散 checkpoint | latent 是 1024px 负担得起的原因 | 反正要选基座时，选 transformer 或 flow matching 的 |
| 采样器 | 高阶 ODE 求解器，20 到 30 步 | 免费、不用重训、确定性 | 永远不要上线训练时的步数 |
| guidance | 打开，能拿到 guidance 蒸馏就用 | 它是让图跟着 prompt 走的东西 | 蒸馏模型常常根本不需要 guidance |
| 草稿路径 | 蒸馏模型，4 到 8 步 | 一格图以交互速度出现 | 产品没有草稿时刻就不做 |
| 定稿路径 | 完整模型 | 用户留下的那张图要多样性和细节 | 风格固定且窄的产品可以不做 |
| 风格 | 一个常驻基座加 LoRA adapter | 五十种风格，显存里一个模型 | 只有当风格是另一个基座时才单独放 |
| 引擎 | 只为你真正服务的两三种形状编译 | 同一份权重上几十个百分点 | 尖峰流量、必须缩到零时用 eager |
| 预览 | 每隔几步做一次低分辨率解码 | 感知延迟比真实延迟更值钱 | 没人在看的批处理链路里不做 |
| 安全 | 前面 prompt 分类器，后面在预览上跑输出分类器 | 在付全分辨率解码之前就拒掉 | 永远不要事后做，也永远不是可选项 |
| 服务 | 按类别分开队列，准入控制走降级 | 长尾住在队列里，不在模型里 | 只有一类请求时才用一个队列 |

## 完整的方案

第 1 节那个场景：1024 乘 1024、四个变体、p95 三秒以内、自己的权重、adapter、链路里有安全。

**从成本模型出发。** 30 步开 guidance 是 60 次前向评估。在编译好的引擎上、这个分辨率下每次评估约 33 毫秒，循环大约 2.0 秒。文本编码、VAE 解码和安全那一遍再加大约 0.27 秒。所以一张图约 2.2 秒，一格四张约 3.1 秒，因为四个变体走同一个循环，只多付一点 batch 惩罚，而不是四倍。

**然后按顺序动杠杆。**

1. **采样器。** 用高阶求解器把 30 步降到 24 步：循环降到约 1.6 秒。免费。
2. **草稿路径。** 一个把 guidance 蒸馏进去的 4 步模型是 4 次评估，约 0.13 秒的去噪。一格图现在受限于固定项而不是循环，这也是产品发生变化的时刻：用户在远不到一秒内看到四个选项。
3. **定稿渲染。** 用户挑一张，完整模型用 24 步开 guidance 渲染它。这个请求是一张图而不是四张，所以循环的钱花在真正要紧的那一张上。
4. **给承载流量的两种形状编译引擎**（草稿 batch 四、定稿 batch 一），后面放一个热的 eager 副本，让空闲后的第一个请求不必付一分钟的引擎加载。
5. **队列。** 草稿和定稿各自拿自己的算力，而不只是各自一个队列。本节末尾那个可运行模型会说明原因：优先级队列没用，因为它抢不了一个已经开跑的渲染。

**决定账单的那笔算术。** 按每 GPU 小时 3.50 美元算，基线那一格图（60 次评估、四个变体、含 batch 惩罚约 3.1 秒）在满利用率下大约是每张 0.08 分钱，在一个尖峰型交互产品真实的 30% 利用率下大约 0.3 分钱。所以题面里的每张八分钱从来不是去噪器造成的：那是闲置的算力，修法是准入控制和缩容，不是更快的模型。**这才是面试官在等的那个答案。** 本节末尾那个可运行模型会把两个数字都打出来。

## 同一套技术在不同约束下

| | 交互式产品（默认） | 批量目录链路 | 端侧 |
|---|---|---|---|
| 目标 | p95 延迟 | 每张图成本 | 能装下 |
| 步数 | 草稿 4，定稿 24 | 24，没有草稿路径 | 1 到 4，蒸馏 |
| guidance | 蒸馏进去 | 打开，成本被摊掉 | 关掉 |
| 分辨率 | 1024px，2048px 单独一档 | 目录需要多少就多少 | 512px，之后再超分 |
| 引擎 | 编译，带热池 | 编译，大 batch | 设备运行时支持什么就用什么 |
| 批处理 | 一格变体 | 显存允许的最大 batch | 永远 batch 一 |
| 预览 | 有，低分辨率 | 没有 | 有，感知质量大半来自它 |
| 队列 | 按类别分开，过载降级 | 一个深队列，不急 | 没有，一次一个请求 |
| 最该怕的失效 | 冷启动和队头阻塞 | 批与批之间 GPU 闲着 | 显存，然后是过热降频 |

## 每个约束决定了什么

**延迟受限（交互式）。** 这个约束把一切推向更少的评估次数，以及把剩下的那些藏起来：蒸馏的草稿路径、把 guidance 蒸馏进去、预览，以及一个宁可降级也不排队的队列。它每张图的成本比批处理链路差，而这是正确的取舍。

**成本受限（批处理）。** 没人在等，杠杆就反过来。用完整模型，跑显存允许的最大 batch，用深队列把 GPU 喂满，预览和草稿路径全部砍掉。要移动的数字是利用率而不是延迟，而这通常就是账单能不能成立的差别。

**显存受限（端侧）。** 上限和操作系统共享，batch 永远是一，所以格式和步数是被替你决定的：低分辨率上的蒸馏模型、不开 guidance、之后再超分。感知质量大部分来自预览行为和超分，不是基座模型。

## 最小的可运行成本模型

只用标准库，确定性。它回答本章反复回到的两个问题：哪个杠杆能移动这个数字，以及为什么 p95 不是均值。

```python
"""A cost model for diffusion serving, plus a queue simulation.

Part 1 prices four configurations from the chapter's formula, so you can see which
lever moves the dominant term. Part 2 runs the same requests under three queue
policies, and shows that two of them are the same policy wearing different names.

Standard library only, no randomness, so two runs give the same numbers.
"""

from dataclasses import dataclass

EVAL_MS_1024 = 33.0          # one denoiser evaluation at 1024px, compiled engine
FIXED_MS = 25.0 + 180.0 + 60.0   # text encode + VAE decode + safety classifier
GPU_USD_PER_HOUR = 3.50
BATCH_PENALTY = 0.15         # each extra variant adds 15% to an evaluation


@dataclass(frozen=True)
class Config:
    name: str
    steps: int
    guidance: bool
    variants: int

    @property
    def evaluations(self) -> int:
        """The unit you actually buy: steps times the guidance factor."""
        return self.steps * (2 if self.guidance else 1)

    @property
    def latency_ms(self) -> float:
        per_eval = EVAL_MS_1024 * (1.0 + BATCH_PENALTY * (self.variants - 1))
        return FIXED_MS + self.evaluations * per_eval

    @property
    def usd_per_image(self) -> float:
        gpu_seconds = self.latency_ms / 1000.0
        return gpu_seconds * GPU_USD_PER_HOUR / 3600.0 / self.variants


CONFIGS = [
    Config("baseline: 30 steps, guidance", 30, True, 4),
    Config("sampler: 24 steps, guidance", 24, True, 4),
    Config("guidance distilled: 24 steps", 24, False, 4),
    Config("step distilled: 4 steps", 4, False, 4),
]


def print_configs() -> None:
    print(f"{'configuration':<34}{'evals':>7}{'latency':>10}{'$/image':>10}{'speedup':>9}")
    base = CONFIGS[0].latency_ms
    for c in CONFIGS:
        print(
            f"{c.name:<34}{c.evaluations:>7}{c.latency_ms / 1000:>9.2f}s"
            f"{c.usd_per_image:>10.4f}{base / c.latency_ms:>8.1f}x"
        )
    print("\nNote the fixed 265 ms floor: once the loop is short, it dominates.\n")


def one_gpu(requests, priority_class=None):
    """One GPU, non-preemptive. Returns (class, wait_ms) per request.

    With priority_class set, a waiting request of that class is picked before any
    other waiting request. Nothing here can interrupt a job already running.
    """
    pending = sorted(requests)
    queue, waits, now, i = [], [], 0.0, 0
    while i < len(pending) or queue:
        while i < len(pending) and pending[i][0] <= now:
            queue.append(pending[i])
            i += 1
        if not queue:
            now = pending[i][0]
            continue
        if priority_class is not None:
            queue.sort(key=lambda r: (r[1] != priority_class, r[0]))
        else:
            queue.sort(key=lambda r: r[0])
        arrival, cls, service = queue.pop(0)
        waits.append((cls, now - arrival))
        now += service
    return waits


def one_gpu_per_class(requests):
    """Two GPUs, one dedicated to each class. The classes no longer share."""
    free = {"draft": 0.0, "final": 0.0}
    waits = []
    for arrival, cls, service in sorted(requests):
        start = max(arrival, free[cls])
        waits.append((cls, start - arrival))
        free[cls] = start + service
    return waits


def percentile(values, p):
    """Nearest-rank percentile, so the result is always an observed value."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, int(round(p / 100.0 * len(ordered))))
    return ordered[rank - 1]


def print_queues() -> None:
    draft = CONFIGS[3].latency_ms                        # the 4-step grid
    final = Config("final render", 24, True, 1).latency_ms
    # 60 requests, one every 800 ms; every sixth is a final render. That is about 86
    # percent utilization of a single GPU: busy, but not overloaded.
    requests = [
        (i * 800.0, "final" if i % 6 == 5 else "draft", final if i % 6 == 5 else draft)
        for i in range(60)
    ]

    print(f"draft service {draft / 1000:.2f}s, final service {final / 1000:.2f}s, "
          f"arrivals every 0.80s\n")
    print(f"{'policy':<32}{'draft p50':>12}{'draft p95':>12}{'final p95':>12}")
    runs = [
        ("1 GPU, FIFO", one_gpu(requests)),
        ("1 GPU, drafts served first", one_gpu(requests, priority_class="draft")),
        ("2 GPUs, one per class", one_gpu_per_class(requests)),
    ]
    for label, waits in runs:
        d = [w for cls, w in waits if cls == "draft"]
        f = [w for cls, w in waits if cls == "final"]
        print(f"{label:<32}{percentile(d, 50) / 1000:>11.2f}s"
              f"{percentile(d, 95) / 1000:>11.2f}s{percentile(f, 95) / 1000:>11.2f}s")
    print("\nRows one and two are identical, and that is the lesson: a priority queue")
    print("cannot preempt a render that is already running, so the draft tail does not")
    print("move. Only separating the classes onto their own capacity removes the")
    print("blocking, which makes this a capacity decision rather than a scheduling one.")


def main() -> None:
    print_configs()
    print_queues()


if __name__ == "__main__":
    main()
```

用任意 Python 3 直接跑，什么都不用装。第一张表就是本章的论点变成数字：采样器免费而且效果小，guidance 蒸馏让循环减半，步数蒸馏把系统推进另一个产品品类，而到那个时候，文本编码、解码和安全那 265 毫秒的固定项就成了下一个该优化的东西。第二张表是 p95 不等于均值的原因，而且里面有个更有用的意外：把草稿排在前面什么都没改变，因为一个已经开跑的 24 步渲染没法被调度策略抢占。草稿的长尾只有在两类请求不再共享一张 GPU 时才消失。队头阻塞，也就是伤害推理模型的那同一个效应，在这里是容量决策而不是调度决策，而一个把优先级队列当作解法的回答，是没想透的。
