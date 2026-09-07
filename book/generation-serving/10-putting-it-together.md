# 10. Putting it together: the complete build

## The default stack: start here, deviate with reason

For an interactive text-to-image product with no unusual constraint, this is the
answer, and the reason each piece is in it.

| Layer | Default choice | Why | When to deviate |
|---|---|---|---|
| Base model | An open latent-diffusion checkpoint at 1024px | The latent is what makes 1024px affordable | A transformer or flow-matching base when you are choosing anyway |
| Sampler | A higher-order ODE solver, 20 to 30 steps | Free, no retraining, deterministic | Never serve the training-time step count |
| Guidance | On, with guidance distillation if available | It is what makes the image follow the prompt | Distilled models often want no guidance at all |
| Draft path | A distilled model, 4 to 8 steps | The grid appears at interactive speed | Skip it if the product has no draft moment |
| Final path | The full model | Diversity and detail on the image the user keeps | Skip it if the house style is fixed and narrow |
| Styles | One resident base plus LoRA adapters | Fifty styles, one model in memory | A separate checkpoint only when the style is a different base |
| Engine | Compiled for the two or three shapes you actually serve | Tens of percent on the same weights | Eager when traffic is spiky and must scale to zero |
| Preview | Low-resolution decode every few steps | Perceived latency beats real latency | Skip in a batch pipeline where nobody is watching |
| Safety | Prompt classifier before, output classifier on the preview | Rejects before paying for the full decode | Never after the fact, and never optional |
| Serving | Separate queues by class, admission control that degrades | The tail lives in the queue, not the model | One queue only when there is one class of request |

## The complete build

The scenario from section 1: 1024 by 1024, four variants, p95 under three seconds,
own weights, adapters, safety in the path.

**Start from the cost model.** 30 steps with guidance is 60 forward evaluations. At
roughly 33 ms per evaluation on a compiled engine at this resolution, the loop is
about 2.0 seconds. Text encode, VAE decode and the safety pass add roughly 0.27
seconds. So one image is about 2.2 seconds and the four-variant grid about 3.1,
because the variants ride the same loop with a modest batch penalty rather than
costing four times as much.

**Then the levers, in order.**

1. **Sampler.** 30 steps down to 24 with a higher-order solver: the loop falls to
   about 1.6 seconds. Free.
2. **Draft path.** A 4-step distilled model with guidance distilled in is 4
   evaluations, about 0.13 seconds of denoising. The grid is now bounded by the fixed
   terms, not the loop, which is the moment the product changes: the user sees four
   options in well under a second.
3. **Final render.** The user picks one; the full model renders it at 24 steps with
   guidance. That request is one image rather than four, so the loop cost is paid on
   the one image that matters.
4. **Compiled engines** for the two shapes that carry traffic (draft batch of four,
   final batch of one), with a warm eager replica behind them so the first request
   after idle does not pay a minute of engine load.
5. **Queues.** Draft and final get their own capacity, not just their own queue.
   The runnable model at the end of this section shows why: a priority queue does not
   help, because it cannot preempt a render that has already started.

**The arithmetic that decides the bill.** At 3.50 USD per GPU hour, the baseline
grid (60 evaluations, four variants, about 3.1 seconds including the batch penalty)
is roughly 0.08 cents per image at full utilization, and about 0.3 cents at the 30
percent utilization a spiky interactive product actually runs at. The eight cents per
image in the problem statement was therefore never the denoiser: it was idle
capacity, and the fix is admission control and scale-down, not a faster model.
**That is the answer the interviewer is listening for.** The runnable model at the
end of this section prints both numbers.

## The same techniques under different constraints

| | Interactive product (default) | Batch catalogue pipeline | On-device or edge |
|---|---|---|---|
| Objective | p95 latency | Cost per image | Fits at all |
| Steps | 4 draft, 24 final | 24, no draft path | 1 to 4, distilled |
| Guidance | Distilled in | On, cost amortized | Off |
| Resolution | 1024px, 2048px as a tier | Whatever the catalogue needs | 512px, upscaled after |
| Engine | Compiled, warm pool | Compiled, large batches | Whatever the device runtime supports |
| Batching | The variant grid | Maximum batch the memory allows | Batch of one, always |
| Preview | Yes, low resolution | No | Yes, it is most of the perceived quality |
| Queue | Separate classes, degrade under load | One deep queue, no urgency | None, one request at a time |
| Failure mode to fear | Cold start and head-of-line blocking | Idle GPUs between batches | Memory, then thermal throttling |

## What each constraint decides

**Latency-bound (interactive).** The constraint pushes everything toward fewer
evaluations and toward hiding the ones that remain: a distilled draft path, guidance
distilled in, previews, and a queue that degrades rather than grows. The cost per
image is worse than the batch pipeline's and that is the correct trade.

**Cost-bound (batch).** With nobody waiting, the levers invert. Use the full model,
run the largest batch memory allows, keep GPUs saturated with a deep queue, and skip
previews and draft paths entirely. Utilization, not latency, is the number to move,
and it is usually the difference between a bill that works and one that does not.

**Memory-bound (on-device).** The ceiling is shared with the operating system and the
batch is always one, so the format and the step count are decided for you: a distilled
model at a low resolution, no guidance, and an upscaler afterwards. The perceived
quality comes mostly from the preview behaviour and the upscale, not from the base
model.

## The smallest runnable cost model

Standard library only, deterministic. It answers the two questions this chapter keeps
returning to: which lever moves the number, and why p95 is not the mean.

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

Run it with any Python 3 and nothing installed. The first table is the chapter's
argument in numbers: the sampler is free and small, guidance distillation halves the
loop, and step distillation moves the system into a different product category, at
which point the fixed 265 ms of text encode, decode and safety becomes the thing to
optimize next. The second table is the reason your p95 is not your mean, and it
contains the more useful surprise: serving drafts first changes nothing, because a
24-step render already running cannot be preempted by a scheduling policy. The draft
tail only disappears when the classes stop sharing a GPU. Head-of-line blocking, the
same effect that hurts reasoning models, is a capacity decision here rather than a
scheduling one, and an answer that offers a priority queue as the fix has not thought
it through.
