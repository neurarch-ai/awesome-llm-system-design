# 10. Putting it together: the complete build

Sections 1 through 6 taught each mechanism with its options and tradeoffs;
section 7 showed where real teams diverge. What none of them show is a single
adaptation with every decision made. This capstone does three things: it gives
you an opinionated default recipe so option paralysis never blocks a first run,
it walks the chapter's clinical scenario end to end with every choice committed
and budgeted, and it shows how the same decisions flip when the constraints
change. It closes with the smallest runnable demonstration of the core
mechanism, one file, no installs.

## The default stack: start here, deviate with reason

Every stage of this pipeline has two to five credible options, and a first-time
builder can burn a week comparing rescaling papers before training a single
step. Skip that. The recipe below is a sane default for a first adaptation; each
row names when to deviate and which section explains why. Methods evolve, but
the interface of each stage (choose the adaptation, mix the data, schedule the
learning rate, rescale the positions, stage the lengths, gate the result) does
not, so decide per stage and treat any specific method as replaceable.

| Stage | Default | Deviate when | Why (section) |
|---|---|---|---|
| Adaptation method | Full continued pretraining (DAPT) with replay | Under ~1B in-domain tokens: SFT or RAG instead; strict forgetting budget: LoRA / QLoRA | [3](03-the-mid-training-phase.md) |
| Domain data mix | Domain corpus plus 10 percent general-data replay | Domain sits close to the pretrain distribution: a lower re-warm peak can substitute for extra replay | [3](03-the-mid-training-phase.md) |
| LR schedule | Re-warm from the decayed floor to a modest peak (a fraction of the pretrain peak), cosine re-decay, anneal highest-quality data at the tail | Run stalls: raise the peak modestly; general benchmarks regress: lower it before touching the data | [3](03-the-mid-training-phase.md) |
| RoPE scaling method | YaRN (non-uniform per-band blend plus attention-temperature correction) | Extension at or under ~8x with near-zero fine-tuning budget: NTK-ABF; extreme 2M+ targets: LongRoPE's searched rescale | [4](04-context-extension.md) |
| Context-extension recipe | Upsampled genuinely long documents, staged length increase, rescale before the long-context training run | Never packed unrelated short documents; that teaches the model distant tokens are irrelevant | [4](04-context-extension.md) |
| Serving posture | GQA base, FlashAttention, paged attention; KV quantization at 64K and beyond | Product only ever sees short prompts: skip the long-context serving stack entirely | [6](06-serving-and-scaling.md) |
| Evaluation | Full general suite before and after, NIAH recall-by-depth heatmap, RULER as the promotion gate | Never. Fix the regression bar before the first training step | [5](05-evaluation.md) |

The last row is the one beginners skip and regret: without the before-and-after
general suite, every replay and peak decision is a vibe, and forgetting is
silent inside the domain slice. Fixing the regression bar up front is one
meeting; discovering forgetting after promotion is a rollback.

## The complete build

Return to the scenario from [section 1](01-clarifying-requirements.md): an
8K-window general base, 40 billion tokens of de-identified clinical notes, a
p95 served document length of 60K tokens, a two-percentage-point regression
budget on MMLU, GSM8K, and instruction following, serving on our own
infrastructure, and the adapted base flowing into post-training afterward. Here
is the whole adaptation with every choice committed and the reason it won.

| Decision | Choice | Why it won |
|---|---|---|
| Order of the two axes | DAPT first, then context extension, gated separately | The axes are independent with different failure modes; sequential passes (the Code Llama shape) let each gate catch its own regression |
| Adaptation method | Full DAPT on the 40B clinical tokens | A broad distributional shift in vocabulary and register; SFT teaches format not prior, and an adapter hits a ceiling at this scale |
| Replay | 10 percent general web data, held constant through the run | Forgetting falls steeply once replay passes about 5 percent while domain gain barely slows; 10 percent is the practical balance point |
| LR schedule | Re-warm to a modest peak, cosine re-decay, gradient clipping at norm 1.0, quality-annealing at the tail | The decayed floor stalls; the original peak erases the base; the peak is the single knob trading forgetting against learning |
| Target length | 64K, not 128K | p95 is 60K; prefill attention is quadratic, so unused headroom is paid for on every long request |
| Rescaling method | YaRN at s = 8, with the attention-temperature correction | The two-point instruction-following bar makes short-context regression the binding risk; YaRN spares the high-frequency bands that uniform PI crowds |
| Long-context data | Upsampled long clinical documents plus synthetic early-fact / late-query insertions, staged 8K to 32K to 64K | Real long-range dependencies, not packed short notes; staging is cheaper and more stable than one full-length run |
| Eval gates | General suite against the 2pp bar, NIAH recall-by-depth heatmap, RULER effective-length gate | Perplexity saturates while retrieval is broken; NIAH alone is edge-anchored single-hop |
| Serving | GQA base, FlashAttention, paged attention, int8 KV cache | A 64K product is prefill-bound on long prompts and KV-bound on batch size at the same time |

**Token budget.** The DAPT mix is the 40B clinical tokens plus 10 percent
replay: 40B / 0.9, about 44.4B tokens total, of which roughly 4.4B are replayed
general data. The context-extension phase costs roughly 0.1 percent of the
original pretraining tokens (the YaRN figure from
[section 4](04-context-extension.md)); against an illustrative 15T-token base
pretrain that is about 15B long-context tokens, staged so the early 8K-to-32K
steps run on cheaper short sequences. The whole adaptation is therefore about
60B tokens, under half a percent of the illustrative from-scratch cost, which
is the economic argument for this entire chapter.

**Replay fraction.** Ten percent means one token in ten keeps a live gradient
on the general distribution at every step, so the optimizer never sees a run of
pure-domain batches long enough to walk out of the general minima. Held
constant, not annealed: a decaying replay share reintroduces drift exactly when
the weights are consolidating ([section 8](08-interview-qa.md)).

**Memory at 64K.** Using the KV-cache formula from
[section 6](06-serving-and-scaling.md) with an illustrative 8B-class
configuration (32 layers, 8 KV heads via GQA, head dimension 128, fp16), one
64K-token sequence holds 2 x 32 x 8 x 128 x 65536 x 2 bytes, about 8.6 GB of
KV cache. The same model with full multi-head attention (32 KV heads) would
need 34.4 GB per sequence, which is why GQA must be baked in rather than
retrofitted; int8 KV quantization halves the 8.6 GB again, and paged attention
keeps heterogeneous batch lengths from fragmenting it. Prefill is the other
cost: 64K is 8x the trained length, so prefill attention FLOPs grow roughly
64x, and long requests are prefill-bound before the first output token.

**What breaks in month one.** Three failure signals dominate early operations,
so wire them before promotion: general-benchmark regression (the full suite run
on every candidate checkpoint against the 2pp bar; forgetting is silent in the
domain slice and a clinical gain that costs MMLU is a net product loss),
mid-depth recall holes (the NIAH heatmap dipping at 40 to 60 percent depth;
users will report that facts in the middle of a discharge summary are missed
while the averaged recall number looks fine), and loss spikes during the
re-warm (a too-high peak or a bad batch; clip at norm 1.0, rewind to the last
good checkpoint, skip the batch, and lower the peak if spikes recur).

## The same techniques under different constraints

The review question that matters in practice is not "which rescaling method is
best" but "which is best under my constraints." Here is the same adaptation
problem three times. Only the middle column is the build above; the other two
keep the identical stage interfaces and swap nearly every choice.

| | Lightweight internal assistant | Clinical 64K (this chapter) | Contract platform at 128K+ |
|---|---|---|---|
| Corpus / target length | ~0.5B tokens of internal docs; prompts under 8K | 40B clinical tokens; p95 doc 60K | Modest domain shift; whole contracts, 128K configured |
| Adaptation | SFT for format plus RAG for facts; at most a QLoRA nudge | Full DAPT with 10 percent replay | Light DAPT or none; the length axis dominates |
| Context extension | None; the trained window already covers the traffic | YaRN at s = 8, staged to 64K | Staged YaRN to 128K; LongRoPE's searched rescale only if multi-million-token targets appear |
| Training tokens | Thousands to low millions of SFT pairs | ~60B total (Illustrative) | Dominated by the long-context phase; long-data curation is the binding constraint |
| Serving | Standard short-context stack | GQA, FlashAttention, paged attention, int8 KV | All of that plus chunked prefill; effective batch size is set by KV memory |
| Evaluation | General suite plus a held-out domain slice | 2pp gate, NIAH by depth, RULER | RULER effective length is the headline number; configured 128K means nothing without it |
| What would be over-engineering | Any RoPE rescale, a replay pipeline, a long-context serving stack | 128K headroom the p95 never uses | Extending to 1M to avoid building retrieval over the contract corpus |

Two lessons fall out. First, the left column is mostly deletions: below the
billion-token threshold DAPT overfits and forgets more than it learns
([section 3](03-the-mid-training-phase.md)), so the correct adaptation is SFT
plus retrieval and the entire length axis disappears with the requirement.
Second, the right column shows the axes trading places as the binding
constraint: domain work shrinks to a nudge, data curation and the
effective-versus-configured length gap become the whole game, and the serving
bill, quadratic prefill and linear KV growth paid per request, is what actually
caps the product. Long context does not replace retrieval over the corpus at
either scale; they compose ([section 6](06-serving-and-scaling.md)).

## What each constraint decides

The compressed decision guide. Read the left column off your requirements; the
right columns say which lever it moves before you compare any methods.

| Your constraint | Lever it moves | Rule of thumb |
|---|---|---|
| In-domain token count | Adaptation method | Under ~1B: SFT or RAG, not DAPT. Billions: full DAPT with replay |
| Forgetting budget | Full-tune vs adapter; peak and replay | Must be bounded by construction: LoRA / QLoRA. Otherwise: modest peak plus ~10 percent replay |
| General-capability floor | The eval gate | Fix the regression bar before training; run the full suite before and after every change |
| p95 served length | Target window | Extend to the measured need, not the marketing number; quadratic prefill prices every token of headroom |
| Length scale s | Rescaling method | Tiny extension: PI is an acceptable baseline. Up to ~8x with little fine-tuning: NTK-ABF. Aggressive with a short-context bar: YaRN. 2M+: LongRoPE |
| Short-prompt traffic share | Non-uniformity of the rescale | High share: YaRN's spared high-frequency bands or LongRoPE's recovery step; uniform PI taxes every short request |
| VRAM budget | KV-cache levers | GQA (baked in at pretraining), KV quantization, paged attention; KV grows linearly with length and batch size collapses first |
| Corpus vs one big document | Long context vs RAG | A corpus: retrieve. One document reasoned over whole: extend. They compose; extension replacing retrieval is a wrong answer |
| Proof the length is real | The promotion gate | NIAH by depth as the smoke test, RULER effective length as the gate; perplexity only for early stopping |

## The smallest runnable context extension

The review of every long-context writeup is the same: the reader nods at
"unseen rotation angles" and still cannot see the mechanism. So here is the
chapter's core claim in one file with zero installs. For a toy 32-dimension
head it computes each frequency pair's rotation angle at a position inside the
trained 8K window and at one beyond it, marks whether training ever showed the
model that angle (a pair that completes at least one full rotation saw the
whole circle; a slower pair only saw a short arc), then applies linear position
interpolation and a YaRN-style per-band blend and checks again. The shape is
the lesson; [section 4](04-context-extension.md) is this file with the
constants generalized.

```python
"""Why naive RoPE extrapolation fails and how rescaling fixes it. Stdlib only."""
import math

D = 32                  # per-head dimension (16 frequency pairs)
BASE = 10000.0          # RoPE base
L_ORIG = 8192           # trained context window
L_NEW = 65536           # target window
S = L_NEW / L_ORIG      # length scale s = 8
ALPHA, BETA = 1.0, 32.0 # YaRN ramp edges, in rotations per trained window

def theta(i):                        # original frequency of dimension pair i
    return BASE ** (-2 * i / D)

def rotations(t):                    # full rotations this pair completes in training
    return L_ORIG * t / (2 * math.pi)

def gamma(t):                        # YaRN blend: 1 = keep, 0 = interpolate
    r = rotations(t)
    if r >= BETA:
        return 1.0
    if r <= ALPHA:
        return 0.0
    return (r - ALPHA) / (BETA - ALPHA)

def yarn_theta(t):                   # per-pair blend of keep vs divide-by-s
    g = gamma(t)
    return g * t + (1 - g) * t / S

def seen(t_orig, angle):
    """Was this angle covered in training, under the ORIGINAL frequencies?
    A pair completing >= 1 rotation saw the whole circle (angles wrap mod 2*pi);
    a slower pair only ever saw the arc [0, L_ORIG * theta)."""
    if rotations(t_orig) >= 1.0:
        return True
    return angle <= L_ORIG * t_orig + 1e-9

def report(m):
    where = "inside" if m < L_ORIG else "beyond"
    print(f"\nposition m = {m} ({where} the trained window), angles in radians")
    print(f"{'pair':>4} {'rot/train':>10} {'naive':>10} {'PI':>10} {'YaRN':>10}   naive      PI      YaRN")
    for i in range(D // 2):
        t = theta(i)
        angles = (m * t, m * t / S, m * yarn_theta(t))
        marks = ["ok" if seen(t, a) else "UNSEEN" for a in angles]
        print(f"{i:>4} {rotations(t):>10.3f} {angles[0]:>10.2f} {angles[1]:>10.2f} "
              f"{angles[2]:>10.2f}   " + "  ".join(f"{mk:>6}" for mk in marks))

report(4096)        # inside the trained window: every method is in seen territory
report(L_NEW - 1)   # beyond it: naive pushes slow pairs into unseen angles

t0 = theta(0)
print("\nadjacent-token angle step of the fastest pair (local-ordering resolution):")
print(f"  original {t0:.3f}   PI {t0 / S:.3f} (crowded {S:.0f}x)   "
      f"YaRN {yarn_theta(t0):.3f} (preserved)")
print(f"YaRN attention-temperature factor: 0.1*ln(s) + 1 = {0.1 * math.log(S) + 1:.3f}")
```

Run it and the two position reports demonstrate the chapter's core claims in
about sixty lines. At position 4096, inside the trained window, every pair is
marked ok under every method. At position 65535, naive extrapolation marks the
three slowest pairs UNSEEN: they complete only 0.73, 0.41, and 0.23 rotations
across the whole trained window, so training only ever showed them a short arc
(about 4.6, 2.6, and 1.5 radians), and the naive angles of roughly 37, 21, and
12 radians land in territory the model has never attended over. Those are
exactly the low-frequency, global-position bands, and arbitrary attention
scores there is why raising `max_position_embeddings` produces garbage. Both PI
and YaRN pull every pair back into seen range, but the closing lines show the
price difference: PI shrinks the fastest pair's adjacent-token angle step from
1.000 to 0.125 radians, crowding neighboring positions 8x and blurring local
ordering, while the YaRN blend keeps it at 1.000 and adds its temperature
factor of 1.208 for s = 8. Swap the toy constants for a real head dimension and
add the continued-training run on long documents, and you have rebuilt this
chapter.
