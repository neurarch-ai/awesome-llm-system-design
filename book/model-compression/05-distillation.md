# 5. Distillation

Quantization and pruning start from a big model and damage it. Distillation starts
from a small model and teaches it, using a big one as the supervision signal. It is
the most expensive lever and the one whose quality profile is smoothest, which is
why it is what you reach for when the target is a genuinely different size class.

## Why soft targets carry more than labels

Training a small model on hard labels gives it one bit of information per token: the
correct next token. Training it against the teacher's full distribution gives it the
teacher's relative preferences over all the wrong answers too, which is a far denser
signal per example. The standard objective is a temperature-scaled KL between
teacher and student distributions:

$$\mathcal{L} = \tau^{2}\,\text{KL}\!\left(p_{T}^{(\tau)} \,\Vert\, p_{S}^{(\tau)}\right) + \lambda \cdot \mathcal{L}_{\text{CE}}(y, p_S)$$

where $p^{(\tau)}$ is the softmax at temperature $\tau$, which flattens both
distributions so the structure among low-probability tokens survives, and the
$\tau^{2}$ factor keeps the gradient magnitude comparable as $\tau$ changes.
Forward KL (teacher first) is mass-covering: the student is pushed to put
probability everywhere the teacher does, which is what you want when the student has
the capacity to cover it and a problem when it does not.

## The three regimes, and the one that matters for LLMs

| Regime | How the student's training data is produced | Main failure mode |
|---|---|---|
| Offline, teacher-scored corpus | Teacher's logits on a fixed corpus | Cheapest, but the student never sees its own mistakes |
| Sequence-level | Train on teacher-generated sequences | Better formatting transfer; still a fixed distribution |
| On-policy | Student generates, teacher scores the student's own outputs | Needs teacher access during training; the strongest |

The on-policy version fixes the distribution mismatch that limits the other two: the
student is trained on the sequences it will actually produce at inference, not on
sequences the teacher would have produced, so the errors that get corrected are the
errors the student actually makes
([On-Policy Distillation of Language Models](https://arxiv.org/abs/2306.13649)).
The cost is that every training step needs teacher scoring, which makes the teacher
part of your training infrastructure rather than a preprocessing step.

## Distillation as the healing step

The highest-value use of distillation in a compression project is not building a
small model from nothing; it is **repairing one you just made**. The pattern:

1. Structurally prune a large parent to the target architecture.
2. Distill the parent into the pruned child, using the parent's distributions rather
   than plain next-token loss.
3. Evaluate the child against the parent per item, not against a benchmark average.

This is materially cheaper than pretraining the small model and lands closer to the
parent's behavior, which is the point: a child that *disagrees differently* from its
parent breaks downstream prompts and evals that were tuned on the parent
([Compact Language Models via Pruning and Knowledge Distillation](https://arxiv.org/abs/2407.14679)).

## When a student beats a small model trained from scratch

Not always, and the interview follow-up is usually exactly this.

**Distillation wins when** the data budget is the constraint rather than the compute
budget (soft targets are a denser signal per token), the teacher is meaningfully
better than anything you could train directly, or behavioral compatibility with the
parent matters (same formats, same refusals, same tool-call style).

**It is a wash or worse when** the student's capacity is far below the teacher's, so
the mass-covering objective spreads the student thin across modes it cannot
represent, or when the teacher is only marginally better than a directly trained
model, in which case you paid for teacher inference and got a comparable result.
The mode-seeking (reverse-KL style) variants exist for the first case.

Two practical notes that come up:

- **Distilling reasoning traces inflates output length.** A student distilled from
  long chain-of-thought traces learns to produce them, which changes your serving
  cost and your latency budget. Measure tokens per response, not only quality.
- **Distillation inherits the teacher's contamination and its biases.** If the
  teacher saw benchmark items, the student's scores are inflated by inheritance,
  which no decontamination of *your* corpus can catch (see
  [benchmarking, section 4](../benchmark-eval/04-contamination-and-validity.md)).

## When to use which

| Reach for | When | Instead of |
|---|---|---|
| Offline teacher-scored distillation | You have a fixed corpus and cannot serve the teacher during training | On-policy, when teacher access is the constraint |
| Sequence-level distillation | You mainly need the teacher's format and style in a smaller model | Logit matching alone, which transfers less of the surface behavior |
| On-policy distillation | You can afford teacher scoring in the loop and want the strongest student | Offline distillation that never corrects the student's own errors |
| Prune then distill | You need a smaller model and have a large parent plus a modest token budget | Training the target size from scratch |
| Quantization-aware training with a distillation loss | You are below 4 bits and PTQ will not hold quality | PTQ plus mixed precision, once that stops working |
| Not distilling | An off-the-shelf small model already meets the bar | Building a bespoke student that ties you to a training pipeline forever |

**Provenance.** Soft-target distillation is Hinton et al. (Google, 2015); the on-policy formulation for language models is generalized knowledge distillation (Google DeepMind, 2023); the prune-plus-distill production recipe is the Minitron line (NVIDIA, 2024).

**Tools.** Distillation runs on the ordinary training stack (PyTorch with DeepSpeed or Megatron-LM); TRL and Axolotl cover the SFT-shaped variants, and on-policy setups pair a serving runtime (vLLM or SGLang) for student sampling with the teacher scoring pass in the training loop. Teacher logit storage is the practical constraint for the offline variant: full vocabulary distributions are large, so top-k truncation of the teacher distribution is standard.

**Worked example.** A team needs an on-device assistant and has a strong 30B-class parent. Training a 3B model from scratch is out of budget, so it prunes the parent structurally to the target shape and distills the parent into it, using top-k teacher distributions on a corpus drawn from real product traffic rather than generic text. Because the student will be judged on the same prompts as the parent, acceptance is paired per item against the parent with flip rate reported, not a benchmark average. It measures response length as well as quality, since the parent's reasoning style transfers and would otherwise blow the on-device latency budget. Only after the student passes does it quantize to the format the device accelerator supports, re-running the same acceptance test on the quantized student rather than assuming the two steps compose.

## Implementation pitfalls

| Problem | Symptom | Fix |
|---|---|---|
| Student capacity far below teacher | Student is fluent and shallow; mass-covering objective spreads it thin | Pick a larger student, or a mode-seeking objective, or narrow the task |
| Offline-only distillation | Good on teacher-generated text, fragile on its own outputs | Move to on-policy so the student's own errors are what get corrected |
| Distilling on generic corpora | Student is worse than the parent exactly on your traffic | Draw the distillation corpus from real product traffic |
| Ignoring output length | Latency and cost regress even though quality held | Track tokens per response as a first-class metric |
| Comparing student to benchmark, not parent | Student "matches on MMLU" but breaks tuned prompts | Paired per-item comparison against the parent, with flip rate |
| Teacher contamination inherited | Student's benchmark scores look too good for its size | Evaluate on post-cutoff or private items; treat inherited contamination as unfixable downstream |
