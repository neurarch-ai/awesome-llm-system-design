# 10. Putting it together: the complete build

Sections 1 through 6 taught each stage with its options and tradeoffs; section 7
showed where real teams diverge. What none of them show is a single pipeline with
every decision made. This capstone does three things: it gives you an opinionated
default stack so option paralysis never blocks a first build, it walks the
chapter's scenario end to end with every choice committed and costed, and it
shows how the same decisions flip when the constraints change. It closes with
the smallest runnable preference tuner, one file, no installs.

## The default stack: start here, deviate with reason

Every stage in this chapter has three to six credible options, and a first-time
builder can burn a month standing up an RLHF pipeline before measuring a prompt
baseline. Skip that. The stack below is a sane default for a first production
build; each row names when to deviate and which section explains why. Frameworks
change yearly, but the interface of each stage (diagnose, curate, train, gate,
serve, refresh) does not, so pick per stage by interface and treat any specific
trainer as replaceable.

| Stage | Default | Deviate when | Why (section) |
|---|---|---|---|
| Diagnosis | Tuned prompt with few-shot examples as the measured baseline | Never. You cannot know whether training helped without it | [2](02-decide-prompt-rag-or-train.md) |
| Knowledge vs behavior | Facts go to retrieval; only behavior, format, and skill go into weights | Facts are genuinely frozen and tiny: few-shot may carry them | [2](02-decide-prompt-rag-or-train.md) |
| Data | A few thousand curated pairs through the funnel: rule filters, dedup, decontamination, then the quality gate | Real logs are thin: bootstrap synthetically, through the same gates | [3](03-data-curation.md) |
| Template | One prompt template, pinned, byte-identical in training and serving | Never. Five styles train five competing behaviors | [3](03-data-curation.md) |
| Method | SFT with a LoRA adapter, r = 16 on attention and FFN projections | Large behavior shift or OOD drift at high rank: full fine-tune | [4](04-methods.md) |
| Memory | QLoRA: 4-bit frozen base, bf16 adapter, one GPU | A cluster is idle and the dataset is large: 16-bit full weights | [4](04-methods.md) |
| Alignment | None. SFT alone ships most behavior tasks | The failure mode is plausible-but-worse answers: DPO, beta 0.03 to 0.1 | [4](04-methods.md) |
| Eval gate | Held-out decontaminated set, regression check vs current prod, tiered smoke-core-full | Never. Build the gate before the first training run | [5](05-evaluation-and-gates.md) |
| Serving | Multi-LoRA: one warm base, hot-swappable adapters, rollback as a route change | You full fine-tuned: each model needs its own serving slot | [6](06-serving-adapters.md) |
| Refresh | Flywheel: mine production failures, label the hard ones, retrain, gate | Never, but keep a human-labeled core to prevent collapse | [6](06-serving-adapters.md) |

The gate row is the one beginners skip and regret: without a held-out,
decontaminated eval set built before training, every loss curve is a vibe, and a
low loss with a falling eval is the classic trap. One afternoon of gate-building
pays for itself the first time "newer" would have quietly shipped "worse."

## The complete build

Return to the scenario from [section 1](01-clarifying-requirements.md): a
general-purpose base model whose output format is inconsistent and whose tone
misses the brand voice, about four thousand human-labeled (prompt, ideal
response) pairs, self-hosted open weights, a stable domain, latency under two
seconds, and a quality gate we must design ourselves. Here is the whole pipeline
with every choice committed and the reason it won.

| Decision | Choice | Why it won |
|---|---|---|
| Ladder stop | Prompt baseline first, then SFT; skip RAG and preference tuning | Behavior problem, not knowledge problem; the domain is stable and the losers are obviously bad, so SFT's positive examples suffice |
| Base model | 7B-class open-weights instruct model, self-hosted (Illustrative) | The scenario controls serving; a 7B carries format and tone, and section 4's adapter math is written at this scale |
| Data | 4,000 pairs through the funnel; ~3,200 train, 600 held out, 200 smoke (Illustrative splits) | Cheap gates first, decontamination before the quality gate; the held-out set is the gate's foundation |
| Template | One chat template, pinned, identical in training and serving | The model learns the template as hard as the content; skew here is a silent killer |
| Method | SFT via QLoRA: 4-bit frozen base, LoRA r = 16 on attention and FFN projections | Moderate behavior shift lives in a low-rank update; one GPU suffices; the frozen base enables everything downstream |
| Schedule | 1 to 3 epochs, LR in the 2e-5 to 1e-4 band, early stop at the validation minimum | Small datasets overfit fast; the validation curve, not the train loss, decides when to stop |
| Eval gate | Format exact-check, LLM-judge win rate vs current prod with CI, regression battery, live 1% slice | Format is structured (exact match); tone is comparative (win rate); the regression check catches silent secondary damage |
| Serving | Multi-LoRA: warm base plus the adapter; promotion and rollback are route changes | No redeploy of the base; an A/B slice is just a second route |
| Versioning | `brand-voice-2026-07-3200-<hash>` on every artifact | Without the data snapshot name you cannot reproduce, debug, or prove eval disjointness |

**Dataset.** 4,000 labeled pairs enter the funnel in the fixed order from
[section 3](03-data-curation.md): rule filters, dedup, decontamination, then the
quality gate. Illustrative retention: ~3,800 survive the cheap gates and ~3,400
the quality pass, leaving ~3,200 training examples after the 600-example
held-out set and 200-example smoke set are carved off, time-separated, and
verified disjoint. That disjointness check reruns before every training run, not
once: a single leaked example can inflate a metric on a set this small, and the
quality gate would otherwise keep leaked examples preferentially, because they
score well.

**Adapter size and memory.** The chapter's formula is trainable params
$= r(d+k)$ per adapted matrix. One 4096 x 4096 attention projection holds 16.8M
frozen weights; its rank-16 adapter trains 16 x (4096 + 4096), about 131k, under
one percent of that matrix. Across the attention and FFN projections of a 7B
model this totals roughly 0.08 percent of the weights ([section
4](04-methods.md)), on the order of 6M trainable parameters, a ~12 MB artifact
in bf16. QLoRA's budget: the frozen base at 4 bits is about 0.5 byte per
parameter, near 3.5 GB for 7B, and the bf16 adapter plus its optimizer state is
tens of megabytes, so the run fits a single commodity GPU with gradient
checkpointing (Illustrative headroom). Full fine-tuning of the same model would
need the 16-bit weights plus two Adam moments per parameter, a different
hardware class entirely, for a behavior shift that does not require it.

**Training cost.** ~3,200 examples at a few hundred tokens each is on the order
of 2M tokens per epoch, call it 4M to 6M over two to three epochs: single-digit
GPU-hours on one card, tens of dollars at commodity rates (Illustrative). The
number worth internalizing is the ratio: the training run is the cheapest line
item in the whole pipeline. The 4,000 human labels cost more than every GPU-hour
combined, which is why [section 3](03-data-curation.md) calls fine-tuning a data
problem wearing a compute costume, and why the flywheel that harvests labels
from production is the compounding asset.

**The gate.** Tiered, per [section 5](05-evaluation-and-gates.md). The
200-example smoke set runs after every checkpoint (seconds). The 600-example
core gate runs before promotion: exact-match on format validity, because
structured output gives no partial credit, plus the regression battery against
the current production model on the same set. Tone is comparative, so the win
rate runs pairwise: the judge needs prompts, not gold labels, so it samples
1,000 production prompts (new model vs current prod, order randomized, identical
pairs as attention checks) and the bar is a 55 percent win rate whose lower
confidence bound clears 0.50, which is why 1,000 prompts and not 100. The judge
is calibrated to human labels on a sampled subset, because judges over-rate
length and formatting, the exact axes this model was trained on. Only after the
offline battery passes does a 1 percent live slice open, because offline
metrics overstate readiness.

**What breaks in month one.** Three failure modes dominate early operations, so
wire their signals before launch: regression on general capability (the
secondary-task scores in the regression battery drift down while the brand-voice
metric holds; catastrophic forgetting from one epoch too many, and the fix is
fewer epochs or a fraction of general data mixed in), format overfitting (the
model forces the trained schema onto requests that wanted free-form text, and
the judge will not catch it because judges reward formatting; the attention
checks and human calibration sample are what surface it), and template skew
between train and serve (a serving-side prompt edit quietly diverges from the
pinned training template, and the model starts ignoring the system prompt; the
fix is a byte-identity check on the template in CI, not vigilance).

## The same techniques under different constraints

The review question that matters in practice is not "which method is best" but
"which method is best under my constraints." Here is the same pipeline built
three times. Only the middle column is the build above; the other two keep the
identical stage interfaces and swap nearly every implementation choice.

| | Closed-API startup bot | Brand-voice specialist (this chapter) | Verifiable-reward coding model |
|---|---|---|---|
| Gap | Format and tone, mild | Format and tone at scale | A reasoning skill the base lacks |
| Data | ~300 examples, no labeling budget | 4,000 human-labeled pairs, funnel-curated | Large synthetic corpus plus unit tests as the scoring signal |
| Weights access | None: vendor API only | Full: self-hosted open weights | Full, plus a training cluster |
| Ladder stop | Rung 1: tuned prompt, few-shot, output schema | Rung 3: SFT, after the prompt baseline left a gap | Rungs 3 then 4: SFT warm-up, then RL |
| Adaptation | None (vendor fine-tune API only if the prompt truly fails) | LoRA r = 16 via QLoRA, one GPU | Full fine-tune: the shift is large and LoRA drifts OOD |
| Alignment | None | None: SFT closed the gap | GRPO against test-pass reward: verifiable, so no preference labels needed |
| Eval gate | 50-prompt smoke set maintained by hand | Tiered gate, judge win rate with CI, live slice | Exact-match and unit-test pass rate; no judge where a test exists |
| Serving | The vendor's problem | Multi-LoRA, adapter routing, ms rollback | Dedicated deployment; full fine-tune forfeits adapter swapping |
| What would be over-engineering | Any training run at all; DPO for a problem a schema fixes | RLHF's five-component pipeline; GRPO with no verifiable reward | Human preference labels for what a test scores; LLM-judge where the compiler is the judge |

Two lessons fall out. First, the startup column is mostly deletions: with 300
examples and no weights access, the ladder from [section
2](02-decide-prompt-rag-or-train.md) stops at rung 1, and the honest answer is
that a tuned prompt plus an output schema is the entire pipeline until the
measured baseline proves a gap. Second, the coding column shows the training
signal trading places: when the reward is verifiable per sample, RL against the
checker beats collecting preference labels (the DeepSeek R1 lesson from [section
7](07-how-teams-do-it-in-production.md)), the behavior shift is large enough to
force full fine-tuning, and the eval gate gets simpler, not more elaborate,
because exact-match replaces the calibrated judge.

## What each constraint decides

The compressed decision guide. Read the left column off your requirements; the
right columns say which lever it moves before you compare any frameworks.

| Your constraint | Lever it moves | Rule of thumb |
|---|---|---|
| Where the gap lives | Ladder rung | Knowledge: retrieval, never weights. Behavior, format, skill: prompt first, then SFT |
| Labeled-example count | Method feasibility | Hundreds: prompt and few-shot. Thousands, clean: SFT. Comparison pairs available: DPO becomes possible |
| Weights access | Adaptation ceiling | Closed API: prompts, schemas, and the vendor's fine-tune endpoint are the whole menu |
| Size of the behavior shift | LoRA vs full fine-tune | Nudge: LoRA r = 8 to 64. OOD drift at high rank: full fine-tune, not more rank |
| GPU budget | Precision of the frozen base | One card: QLoRA (4-bit base, bf16 adapter). Raising rank does not fix what quantization did not break |
| What-not-to-say matters | SFT vs preference tuning | Obviously bad losers: SFT on winners suffices. Plausible-but-worse losers: DPO, beta 0.03 to 0.1 |
| Reward verifiability | DPO/RLHF vs GRPO | Checkable per sample (tests, math, retrieval rank): GRPO, no reward model. Open-ended: preference pairs |
| Tenant or domain count | Serving shape | Many variants: multi-LoRA, one warm base plus N adapters; full fine-tuning forfeits this |
| Domain churn | What goes into weights | Facts that move: retrieval, updated instantly. Stable behavior: weights, retrained rarely |
| Promotion risk | Gate depth | Smoke set per checkpoint; full battery plus regression check plus live slice before any user sees it |

## The smallest runnable preference tuner

The review of every alignment tutorial is the same: the reader assembles a
trainer, an accelerator config, and a reference model and still cannot see the
loss. So here is SFT and DPO side by side in one file with zero installs. Every
production component is swapped for the smallest thing with the same interface:
the LLM becomes one logit per candidate response, the preference dataset becomes
three (prompt, chosen, rejected) triples, the frozen reference is the initial
policy, and the trainer is hand-coded gradient descent. Each prompt also carries
a bystander response that no preference pair ever mentions, because the
difference between the two methods lives in what happens to it.

```python
"""SFT vs DPO on a toy policy: one logit per candidate response, stdlib only."""
import math

# Three prompts; each has (chosen, rejected, bystander) candidate responses.
# The rejected answer is the tempting-but-wrong one; the bystander is a
# harmless alternative that no preference pair ever mentions.
PAIRS = [
    ("refund past the window", "Politely decline; offer store credit.",
                               "Sure, full refund, no questions!",
                               "Please contact support."),
    ("angry customer",         "Acknowledge, apologize once, give the next step.",
                               "You are totally right, we are terrible!",
                               "Noted."),
    ("feature request",        "Thank them, log it, promise nothing.",
                               "Absolutely, shipping it next week!",
                               "We will see."),
]
C, R, O = 0, 1, 2               # chosen / rejected / bystander
INIT = [0.0, 1.0, 1.0]          # base model: both wrong answers more likely

def logprobs(z):
    lse = math.log(sum(math.exp(v) for v in z))
    return [v - lse for v in z]

def softmax(z):
    e = [math.exp(v) for v in z]
    return [v / sum(e) for v in e]

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))

def train(method, steps=300, lr=0.5, beta=0.5):
    policy = [list(INIT) for _ in PAIRS]      # one logit per candidate response
    ref = [logprobs(INIT) for _ in PAIRS]     # frozen reference = the SFT checkpoint
    for step in range(steps + 1):
        margins = []
        for z, rf in zip(policy, ref):
            lp = logprobs(z)
            margins.append(lp[C] - lp[R])
            if step == steps:
                continue
            if method == "dpo":
                # loss = -log sigmoid(beta * (policy logratio - reference logratio))
                m = beta * ((lp[C] - rf[C]) - (lp[R] - rf[R]))
                g = (1.0 - sigmoid(m)) * beta   # gradient magnitude on the pair
                z[C] += lr * g                  # chosen logit up ...
                z[R] -= lr * g                  # ... rejected logit down: the margin
            else:                               # sft: cross-entropy on chosen only
                p = softmax(z)
                for j in range(3):
                    z[j] -= lr * (p[j] - (1.0 if j == C else 0.0))
        if step % 100 == 0:
            print(f"  step {step:3d}  mean margin log P(chosen) - log P(rejected) = "
                  f"{sum(margins) / len(margins):+.3f}")
    return policy

for method in ("sft", "dpo"):
    print(f"{method.upper()} training:")
    policy = train(method)
    p, lp = softmax(policy[0]), logprobs(policy[0])
    print(f"  prompt 1 after training: P(chosen)={p[C]:.3f}  "
          f"P(rejected)={p[R]:.3f}  P(bystander)={p[O]:.3f}")
    print(f"  rejected vs bystander: log P(rejected) - log P(bystander) = "
          f"{lp[R] - lp[O]:+.3f}\n")
```

Run it and the printout is the chapter's argument in sixty lines. Both methods
drive the preference margin from -1.0 (the base model prefers the tempting
wrong answer) to strongly positive: SFT reaches a mean margin of +6.09 and DPO
+7.56 after 300 steps, and both leave P(chosen) above 0.95. The separating
number is the last line of each block. Under SFT the rejected-vs-bystander
log-ratio finishes at exactly +0.000: cross-entropy on the winners never saw a
negative example, so the sycophantic answer ends up precisely as likely as the
harmless "please contact support," relative to each other, as it was at
initialization. Under DPO the same ratio finishes at -4.279, because the
contrastive loss pushed the rejected logit down specifically, exactly the
"what not to say" distinction [section 4](04-methods.md) says SFT cannot
express. The DPO update in the code is also where beta lives: the gradient
scales with $(1 - \sigma(m))\beta$, so as the policy's margin over the frozen
reference grows, the update decays toward zero and the leash tightens. Swap
the logit table for a transformer, the three triples for a curated preference
set, and the hand-coded step for DPOTrainer, and you have rebuilt this
chapter's rung 4.
