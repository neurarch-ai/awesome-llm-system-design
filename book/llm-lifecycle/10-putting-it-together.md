# 10. Putting it together: the complete build

Sections 1 through 6 taught each lifecycle stage with its options and
tradeoffs; section 7 showed where real teams diverge. What none of them show is
a single build with every decision made. This capstone does three things: it
gives you an opinionated default path so option paralysis never blocks a first
plan, it walks the chapter's scenario end to end with every choice committed
and costed, and it shows how the same decisions flip when the constraints
change. It closes with the smallest runnable lifecycle planner, one file, no
installs.

## The default stack: start here, deviate with reason

Every stage in this chapter has three to six credible options, and a first-time
builder can burn a month debating pretrain-vs-adapt before training a single
token. Skip that. The path below is a sane default for a first production
build; each row names when to deviate and which section explains why. Base
models and trainers change yearly, but the interface of each stage (curate,
size, train, align, serve, evaluate) does not, so decide per stage and treat
any specific checkpoint or library as replaceable.

| Stage | Default | Deviate when | Why (section) |
|---|---|---|---|
| Build vs buy | Mid-train an open base (Llama 3 / Qwen3 / OLMo class) | No residency or domain gap: use an API and skip training entirely; capability absent from every open base: from-scratch pretrain | [1](01-clarifying-requirements.md), [3](03-pretraining-and-scaling.md) |
| Data budget | Dedup, filter, decontaminate before any training; mix general replay data into a domain mid-train | Never on decontamination; a benchmark number without it is meaningless | [2](02-the-five-stages.md) |
| Model sizing | Size to the serving target first; 20 tokens per parameter only when training compute is the binding cost | You will serve billions of tokens: overtrain a smaller model past its optimum | [3](03-pretraining-and-scaling.md), [5](05-inference-economics.md) |
| Architecture | A GQA + RoPE decoder base | Extreme QPS: MQA (Character.AI); quality per FLOP at scale: MoE | [3](03-pretraining-and-scaling.md) |
| Post-training | SFT on curated pairs, then DPO on offline preference pairs | Checkable rewards (math, code): GRPO; reusable reward model needed: PPO; labeling cost dominates: RLAIF | [4](04-post-training.md) |
| Serving | vLLM (paged KV, continuous batching, prefix cache), INT8 weights | INT8 does not fit: INT4 behind an eval gate; the architecture itself is too big: distill | [5](05-inference-economics.md), [6](06-serving-and-scaling.md) |
| Knowledge | RAG for facts, fine-tuning for behavior; they compose | Facts are static and never need a citation (rare) | [6](06-serving-and-scaling.md) |
| Evaluation | The right metric per stage, plus a decontamination claim on every gain | Never. Name the metric before proposing the fix | [2](02-the-five-stages.md), [8](08-interview-qa.md) |

The last row is the one that quietly decides all the others: without per-stage
metrics (domain benchmark plus the general suite, preference win rate,
tokens/sec and cost per million), every training and compression decision is a
vibe, and you cannot tell whether a change helped.

## The complete build

Return to the scenario from [section 1](01-clarifying-requirements.md):
sensitive legal documents that cannot leave premises, an open base allowed, 50
billion tokens of cleaned case law and contracts updated quarterly, p95
time-to-first-token under two seconds at 500 concurrent users, and a quality
bar of citation accuracy and instruction following for trained lawyers. Here is
the whole lifecycle with every choice committed and the reason it won.

| Decision | Choice | Why it won |
|---|---|---|
| Build vs buy | Mid-train + post-train an open base; no pretrain | Data residency forces owned weights; a from-scratch pretrain would cost hundreds of millions to relearn what the base already knows |
| Base model | 8B-class open base shipping GQA and RoPE | The serving constraint caps the size ([section 6](06-serving-and-scaling.md) puts an interactive legal assistant at 7-13B); GQA must be baked in at training time, so it is a base-selection criterion, not a hotfix |
| Mid-training data | 50B legal tokens plus a general replay mix | Continued pretraining without replay causes catastrophic forgetting; the general eval suite is the regression alarm |
| Post-training | SFT on legal instruction pairs (citation format, refusal style), then DPO | The Llama 3 recipe: stable, offline, no reward model or PPO loop; tens of thousands of high-quality pairs beat raw volume |
| Facts vs behavior | RAG over the case-law index; fine-tuning only for style and format | Law changes and answers must cite sources; weights cannot cite and go stale between quarterly refreshes |
| Serving | INT8, vLLM with paged KV and continuous batching, prefix cache on the system prompt | Decode is memory-bandwidth bound; INT8 halves the bytes read per token at near-zero quality loss, behind an eval gate |
| Pipelines | Feature / training / inference kept separate, joined by the index and the model registry | A corpus refresh, a retrain, and a serving deploy must not be the same event |
| Evaluation | Domain benchmark + full general suite + citation accuracy, decontaminated | Forgetting and eval leakage are the two silent failure modes of this exact plan |

**Mid-training compute.** With the [section 3](03-pretraining-and-scaling.md)
estimate $C \approx 6ND$: 8B parameters over 50B domain tokens plus roughly a
20 percent general replay mix (Illustrative) is about 62B tokens, so
$C \approx 6 \times 8\times10^9 \times 6.2\times10^{10} \approx 3\times10^{21}$
FLOPs. The base's own pretrain (15T tokens, Llama 3 8B) cost about
$7.2\times10^{23}$ FLOPs, roughly 240 times more. That ratio is the whole
argument of [section 1](01-clarifying-requirements.md): the general language
ability is free, and the team pays only for the domain delta.

**Serving arithmetic.** At INT8 the weights are 8 GB, and the
[section 5](05-inference-economics.md) decode bound gives
$t_{\text{decode}} \approx 8\times10^9 \times 1 / 2\times10^{12} = 4$ ms per
token at batch 1 on 2 TB/s HBM, far inside the interactive budget. The KV
cache is what actually sizes the fleet: at Llama-3-8B geometry (32 layers, 8 KV
heads, head dim 128, bf16 cache) each token costs
$2 \times 32 \times 8 \times 128 \times 2 \approx 131$ KB, so an 8K-token
session holds about 1 GB. An 80 GB GPU minus 8 GB of weights fits roughly 70
full-length concurrent sessions, so 500 concurrent users need on the order of
8 GPUs before PagedAttention reclaims fragmentation waste (Illustrative). The
model size was never the memory problem; the cache was.

**Why not overtrain, and when that flips.** This team trains 62B tokens on an
already-pretrained base, so Chinchilla arithmetic is not their decision; it was
Meta's. But the [section 3](03-pretraining-and-scaling.md) logic still binds
the base choice: an 8B base trained to roughly 1800 tokens per parameter exists
precisely because someone serving at scale paid extra training FLOPs for a
permanently cheaper decode. The planner at the end of this section puts a
number on that crossover.

**What breaks in month one.** Three failure modes dominate early operations, so
wire their signals before launch: general-eval regression (the mid-trained
model drifts on MMLU-class suites while the domain benchmark improves; the fix
is the replay mix and a hard gate on the general suite), citation failures
(fabricated or stale case citations mean facts are leaking from weights instead
of the RAG index; track a citation-verification rate and the index freshness
against the quarterly corpus drops), and KV-cache pressure at peak (p95 TTFT
spikes or OOMs when concurrent sessions run long; watch cache occupancy per
GPU, not average QPS, because lawyers hold long documents in context).

## The same techniques under different constraints

The interview question that matters in practice is not "should we pretrain" in
the abstract but "what does my constraint set make of the same five stages."
Here is the same lifecycle planned three times. Only the middle column is the
build above; the other two keep the identical stage vocabulary and flip nearly
every decision.

| | API-first product team | Legal domain assistant (this chapter) | Consumer chat at 20k QPS |
|---|---|---|---|
| Gap that justifies weights | None: no residency constraint, domain covered by frontier APIs | Data residency plus domain vocabulary and citation style | Unit economics: per-token margin at massive volume |
| Entry stage | Stage 5 only: prompt engineering, RAG, function calling on an API | Mid-training + post-training on an open base | Full lifecycle: pretrain a small model they own end to end |
| Sizing | Not their decision; pick a model tier per task | 8B-class, capped by the serving target | Small and deliberately overtrained: inference dominates lifetime cost |
| Training data | None; budget goes to eval sets and prompts | 50B domain tokens + replay mix | Trillions of tokens, way past 20 per parameter |
| Post-training | None; the API vendor did it | SFT + DPO for citation format and refusal | Heavy preference tuning; persona lives in the weights |
| Serving | Vendor's problem; they watch latency and cost per call | INT8, GQA, vLLM, ~8 GPUs (Illustrative) | MQA, INT8 weights and KV cache, inter-turn prefix cache (Character.AI) |
| Knowledge freshness | RAG; the index is the only artifact they own | RAG over the case-law index, quarterly refresh | Mostly none: persona is stable, weights hold it |
| What would be over-engineering | Any training at all; owning GPUs | From-scratch pretrain; frontier-scale base; INT4 without a gate | Per-query RAG; a 70B model; full-precision serving |

Two lessons fall out. First, the left column is the most common correct answer
in industry and the least common answer in interviews: when no constraint
forces owned weights, the entire training half of the lifecycle disappears, and
the team's leverage moves to evals, prompts, and retrieval. Saying so is
signal, not weakness. Second, the right column shows the sizing objective
flipping as volume grows: the legal team sizes to a latency target, but the
consumer team sizes to a cost-per-token target, which is why it overtrains a
small model and strips the KV cache with MQA. Same formulas, opposite optima.

## What each constraint decides

The compressed decision guide. Read the left column off your requirements; the
right columns say which lever it moves before you compare any checkpoints.

| Your constraint | Lever it moves | Rule of thumb |
|---|---|---|
| Residency, cost at volume, or a capability gap | Build vs buy | No gap: API. Gap an open base covers: mid-train it. Capability absent from every base: only then pretrain |
| Lifetime inference volume | Sizing objective | Low volume: compute-optimal (~20 tok/param) or an API. Billions of generated tokens: overtrain a smaller model past its optimum |
| Domain data shape | Mid-training vs SFT | Billions of raw domain tokens: mid-train (knowledge). Thousands of curated pairs: SFT (behavior). Diagnose failures the same way |
| Knowledge change rate | RAG vs weights | Facts changing faster than your retrain cadence make weights structurally stale; retrieve and cite instead |
| TTFT and concurrency | Model size, precision, KV variant | Size to serve: INT8 first, INT4 only behind an eval gate; GQA/MQA is a base-selection decision, not a serving hotfix |
| Labeling budget | Preference method | Offline pairs exist: DPO. Checkable rewards: GRPO. Reusable reward model: PPO. Label cost dominates: RLAIF against a constitution |
| Audit and quality bar | Eval gates | Decontaminate before believing any gain; eval-gate every compression; track attack success and false-refusal rates as release gates |

## The smallest runnable lifecycle planner

The review of every scaling-law blog post is the same: the reader nods at the
curves and still cannot answer "so how big should our model be." So here is
the chapter's core arithmetic as one file with zero installs. It takes a FLOP
budget, prints the Chinchilla-optimal $(N, D)$ split from
$C = 6ND$ and $D = 20N$, then compares the total cost of ownership of that
optimal model against a half-size model overtrained past its own optimum,
across lifetime inference volumes, and prints the crossover where overtraining
wins. Every constant is a knob; the shape of the answer is the lesson.

```python
"""Lifecycle planner: Chinchilla split, then the train-vs-serve TCO crossover.

Every rule is the chapter's own: C = 6*N*D (section 3), D = 20*N at the
compute-optimal point (section 3), and decode cost = bytes moved / bandwidth
(section 5). Dollar and utilization figures are illustrative; the crossover
logic is the lesson.
"""

GPU_HOURLY = 2.00      # $/GPU-hour, illustrative rental price
TRAIN_FLOPS = 4.0e14   # sustained training FLOP/s per GPU (~40% MFU), illustrative
HBM_BW = 2.0e12        # HBM bandwidth in bytes/s (A100 class, section 5)
BATCH = 32             # continuous batching amortizes the weight read (section 5)
P_BYTES = 1            # INT8 weights (section 5)


def chinchilla_split(C):
    """C = 6*N*D with D = 20*N gives C = 120*N^2, so N = sqrt(C/120)."""
    N = (C / 120) ** 0.5
    return N, 20 * N


def train_cost(N, D):
    """6*N*D FLOPs at the sustained per-GPU rate, priced per GPU-hour."""
    return 6 * N * D / TRAIN_FLOPS / 3600 * GPU_HOURLY


def serve_cost_per_token(N):
    """Decode is memory-bandwidth bound: every token reads all N*P_BYTES
    weight bytes; continuous batching splits that read across BATCH streams."""
    seconds = N * P_BYTES / HBM_BW / BATCH
    return seconds / 3600 * GPU_HOURLY


def plan(C, small_frac=0.5, overtrain=10):
    """Compare the compute-optimal model against a smaller model overtrained
    past its own optimum (quality treated as comparable for planning; the
    chapter's Llama 3 8B anchor sits at ~90x its optimal token count)."""
    N_opt, D_opt = chinchilla_split(C)
    N_small = small_frac * N_opt
    D_small = overtrain * 20 * N_small
    opt = (N_opt, D_opt, train_cost(N_opt, D_opt), serve_cost_per_token(N_opt))
    small = (N_small, D_small, train_cost(N_small, D_small),
             serve_cost_per_token(N_small))
    return opt, small


def report(C, volumes):
    (N1, D1, t1, s1), (N2, D2, t2, s2) = plan(C)
    print(f"FLOP budget C = {C:.1e}")
    print(f"  Chinchilla-optimal : N = {N1/1e9:5.1f}B params, "
          f"D = {D1/1e9:6.0f}B tokens ({D1/N1:.0f} tok/param)")
    print(f"  Overtrained small  : N = {N2/1e9:5.1f}B params, "
          f"D = {D2/1e9:6.0f}B tokens ({D2/N2:.0f} tok/param)")
    print(f"  Train cost         : optimal ${t1:,.0f}  vs  small ${t2:,.0f}")
    print(f"  Serve cost / 1M tok: optimal {s1*1e6:.3f}  vs  small {s2*1e6:.3f} (USD)")
    print(f"  Decode ms/token b=1: optimal {N1*P_BYTES/HBM_BW*1e3:.1f}  "
          f"vs  small {N2*P_BYTES/HBM_BW*1e3:.1f}")
    print()
    print(f"  {'lifetime tokens':>16} | {'optimal TCO':>12} | "
          f"{'small TCO':>12} | winner")
    for v in volumes:
        a, b = t1 + s1 * v, t2 + s2 * v
        w = "optimal" if a <= b else "small (overtrained)"
        print(f"  {v:16.0e} | {a:12,.0f} | {b:12,.0f} | {w}")
    cross = (t2 - t1) / (s1 - s2)
    print(f"\n  Crossover: overtraining wins past {cross:.1e} lifetime "
          f"generated tokens\n  ({cross * s1:,.0f} dollars of serving at the "
          f"optimal model's rate).")


if __name__ == "__main__":
    report(C=5.9e21, volumes=[1e9, 1e10, 1e11, 1e12, 1e13])
```

Run it and the chapter's two sizing arguments come out as numbers. At the
section 3 budget of $5.9\times10^{21}$ FLOPs the Chinchilla split is a 7B
model on 140B tokens, exactly the worked example; the half-size alternative at
3.5B parameters takes 701B tokens (200 per parameter) and costs about 2.5x
more to train ($20,486 vs $8,194, Illustrative prices) but halves both the
decode latency (1.8 vs 3.5 ms per token at batch 1) and the serving cost per
million tokens ($0.030 vs $0.061). The TCO table then shows the flip: at 1B
and 10B lifetime tokens the compute-optimal model wins, and by 1e12 tokens the
overtrained model's total is $50,920 against $69,062, with the printed
crossover near $4\times10^{11}$ lifetime generated tokens. That is the
Llama 3 argument of [section 3](03-pretraining-and-scaling.md) reduced to
arithmetic: state which cost you are optimizing, count lifetime tokens, and
the sizing decision makes itself. Swap the constants for your GPU price, your
bandwidth, and your traffic forecast, and you have rebuilt this chapter's
economics for your own system.
