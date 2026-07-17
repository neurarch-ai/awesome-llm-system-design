# 4. Methods

## The SFT-to-preference pipeline

```mermaid
flowchart TD
  DATA["data curation<br/>clean (prompt, response) pairs"] --> SFT["SFT<br/>next-token prediction on labeled examples"]
  SFT --> DECIDE{"quality axis SFT cannot capture?"}
  DECIDE -->|"no: format, tone, skill"| GATE["eval gate (section 5)"]
  DECIDE -->|"yes: safety, helpfulness, comparative preference"| PREF["preference tuning<br/>DPO or RLHF or GRPO"]
  PREF --> GATE
  GATE -->|"pass"| SERVE["serve adapter"]
  GATE -->|"fail"| DATA
```

**How it works.** Training starts from curated (prompt, response) pairs and runs SFT,
which is plain next-token prediction on those labeled examples and is usually the only
step needed. The branch point is whether the target quality axis is something SFT can
teach directly: format, tone, and skill are learnable from examples, so those cases go
straight to the eval gate. Axes that depend on comparing two candidate responses,
safety, helpfulness, and other comparative preferences, cannot be expressed as a single
gold answer, so they route through a preference-tuning stage (DPO, RLHF, or GRPO) before
reaching the gate. The eval gate (covered in section 5) is the single arbiter: a pass
ships the trained adapter, and a fail loops back to data curation rather than to more
training, because the usual fix for a failed gate is better or cleaner data, not another
epoch on the same set. The loop makes explicit that post-training is iterative and that
data quality, not method choice, is the dominant lever.

## Supervised fine-tuning (SFT)

SFT is plain next-token prediction on `(prompt, ideal response)` pairs. Show the
model the input and the output you want, and minimize the negative log-likelihood
over the response tokens:

$$L_{\text{SFT}} = -\frac{1}{T}\sum_{t=1}^{T} \log p_\theta\!\left(y_t \,\middle|\, x,\, y_{\lt t}\right)$$

where $x$ is the prompt, $y_1, \ldots, y_T$ is the ideal response, and $\theta$
are the model parameters. The loss is computed over response tokens only; the
prompt tokens are masked out.

SFT is the workhorse and usually the only training step you need. It teaches
format, tone, and task-specific behavior directly. The two failure modes to name:

**Catastrophic forgetting.** Over-training on a narrow set degrades general
ability. Keep learning rates modest (often around 2e-5 to 1e-4), epochs few (one
to three), and mix in a small fraction of general data if breadth matters. A LoRA
adapter avoids this more naturally because the base weights stay frozen.

**Eval contamination.** Training on examples that overlap your eval set inflates
the metrics. Decontaminate every time, not once.

## Parameter-efficient fine-tuning: LoRA and QLoRA

Full fine-tuning updates every weight in the model. For a 7B or 70B parameter
model that means optimizer state (typically two copies of the gradients in Adam),
activations, and a fresh full-size checkpoint per task. You rarely need it.

**LoRA (low-rank adaptation)** freezes the base weights and learns a small pair of
low-rank matrices for each target weight matrix:

$$W = W_0 + \frac{\alpha}{r}\, B A,\qquad B \in \mathbb{R}^{d \times r},\ A \in \mathbb{R}^{r \times k},\ r \ll \min(d, k)$$

$W_0$ is frozen; only $B$ and $A$ train. Trainable parameters drop from $dk$ to
$r(d + k)$. At $r = 16$ on the attention and FFN projections of a 7B model, this
is roughly 0.08 percent of the total, and the task quality is nearly
indistinguishable from full fine-tuning on most behavior-and-format tasks.

As a layer it is a frozen `nn.Linear` with a trainable low-rank detour added to
its output, and $B$ is zero-initialized so training starts exactly at the base
model:

```python
class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=16, alpha=16):
        super().__init__()
        self.base = base.requires_grad_(False)          # W0 frozen
        d, k = base.out_features, base.in_features
        self.A = nn.Parameter(torch.randn(r, k) * 0.01) # down-project k -> r
        self.B = nn.Parameter(torch.zeros(d, r))        # up-project r -> d, starts at 0
        self.scale = alpha / r
    def forward(self, x):
        return self.base(x) + self.scale * (x @ self.A.T @ self.B.T)
```

Because $B$ starts at zero the detour contributes nothing on step one, so the
adapter can only improve on the frozen base, never corrupt it at initialization.
At serving time $\frac{\alpha}{r}BA$ can be folded back into $W_0$ so there is zero
inference overhead (covered in section 6).

**QLoRA** quantizes the frozen base to 4-bit to slash its memory footprint, then
trains the LoRA adapter on top in BFloat16. The approximate memory budget is:

$$M \approx \underbrace{4\text{-bit}\cdot N_{\text{base}}}_{\approx 0.5\ \text{byte/param, frozen}} \;+\; \underbrace{16\text{-bit}\cdot 2\,r(d+k)\,L}_{\text{trainable adapter, tiny}}$$

This is what lets you fine-tune a 7B (or larger) model on a single commodity GPU.
Mercari used QLoRA to fine-tune a 2B model on one A100 and beat GPT-3.5 on their
task at roughly 14x lower inference cost.

![LoRA trainable parameters vs full fine-tune](assets/fig-lora-params.png)

*Trainable parameter counts (log scale) for LoRA at three ranks versus full
fine-tune, across model sizes from 1B to 70B. LoRA r=16 trains roughly 0.08% of
the weights; QLoRA fits the entire frozen base plus the adapter on a single GPU.*

When is full fine-tuning justified? A large dataset, a large behavior shift from
the base, or cases where a LoRA adapter at high rank still drifts out of
distribution (Anyscale found exactly this on their DPO task). For the standard
"adapt a base model to our domain" prompt, LoRA or QLoRA is almost always the
right call, and saying so plainly is the senior answer.

## Preference optimization: DPO, RLHF, and GRPO

SFT teaches the model to imitate good answers. It cannot teach the model to
*prefer* one acceptable answer over another, avoid a tempting-but-wrong style, or
pick a safer response when two responses are both plausible. That is what
preference training does, by training on comparisons rather than imitations.

### DPO (direct preference optimization)

DPO skips the separate reward model and the RL loop entirely. It optimizes the
policy directly on `(prompt, chosen response, rejected response)` triples with a
classification-style loss:

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E}_{(x,\, y_w,\, y_l)} \!\left[\log \sigma\!\left(\beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}\right)\right]$$

$y_w$ is the chosen (winning) response; $y_l$ is the rejected (losing) response;
$\pi_{\text{ref}}$ is the frozen reference model (the SFT checkpoint); and $\beta$
is the KL penalty coefficient that controls how far the policy may move from the
reference. A small $\beta$ (Anyscale used 0.03) keeps the policy close and stable.

The reference model is the load-bearing piece. Without it, the policy could
trivially score $y_w$ higher than $y_l$ by collapsing to degenerate text that
gets arbitrarily high log-probability. The $\pi_{\text{ref}}$ anchor prevents that.

The loss itself is a few lines: take the sequence log-probs of the chosen and
rejected responses under the policy and the frozen reference, then push a
log-sigmoid on their difference.

```python
import torch.nn.functional as F
# each arg: summed log-prob of that response under that model, shape (batch,)
def dpo_loss(pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta=0.1):
    pol_logratio = pol_chosen - pol_rejected   # how much the policy prefers chosen
    ref_logratio = ref_chosen - ref_rejected   # the reference's built-in preference
    return -F.logsigmoid(beta * (pol_logratio - ref_logratio)).mean()
```

**The edge case seniors watch for: DPO can lower the chosen probability.** The
loss constrains only the *margin* between the chosen and rejected log-ratios, not
either term on its own, so gradient descent can satisfy it by pushing the rejected
log-probability down faster than the chosen one, dragging the absolute
log-probability of the preferred response down at the same time. This likelihood
displacement (studied by Razin et al., 2024) shows up as a chosen-reward curve
that falls even while the training loss keeps improving, and in the worst case it
moves probability mass onto a third, unintended response rather than onto $y_w$.
It is most acute when the chosen and rejected texts are near-duplicates that share
most of their tokens, because their gradients largely cancel and only the small
difference is left to steer on. The standard remedy is DPO-Positive (Pal et al.,
2024), which adds a term penalizing the chosen log-probability for dropping below
the reference, so the margin is widened by pushing the rejected response down
rather than by sacrificing the response you actually want.

### RLHF (reinforcement learning from human feedback)

RLHF trains a separate reward model $r_\phi$ on human preference comparisons, then
optimizes the policy against that reward using reinforcement learning (commonly
PPO), subject to a KL penalty:

$$\max_{\pi_\theta}\;\mathbb{E}_{x,\, y \sim \pi_\theta}\!\left[r_\phi(x, y)\right] - \beta\;\text{KL}\!\left[\pi_\theta(y \mid x)\;\Vert\;\pi_{\text{ref}}(y \mid x)\right]$$

The KL term plays the same anchoring role as DPO's $\beta$: drop it and the
policy reward-hacks $r_\phi$ into degenerate output. RLHF is more powerful when
you need a reusable reward signal, but it is a complex, multi-model pipeline
(SFT model, reward model, reference model, value network, policy), and it is
harder to stabilize than DPO.

### GRPO (group relative policy optimization)

GRPO, used in DeepSeek R1 and variants, eliminates the value network by computing
advantages within a *group* of responses sampled for the same prompt. For a group
of $G$ outputs $\{o_1, \ldots, o_G\}$ per query $q$, the group-normalized
advantage is:

$$\hat{A}_i = \frac{r_i - \text{mean}(r_{1:G})}{\text{std}(r_{1:G})}$$

and the training objective maximizes the clipped ratio of new to old policy,
weighted by these advantages, minus the same KL penalty on the reference:

$$\mathcal{L}_{\text{GRPO}} = -\mathbb{E}\!\left[\sum_{i=1}^{G} \min\!\left(\frac{\pi_\theta(o_i \mid q)}{\pi_{\text{old}}(o_i \mid q)}\hat{A}_i,\; \text{clip}\!\left(\cdot,\, 1-\epsilon,\, 1+\epsilon\right)\hat{A}_i\right) - \beta\;\text{KL}[\pi_\theta \Vert \pi_{\text{ref}}]\right]$$

The appeal is that you do not need a learned value function: the group mean acts as
a baseline. GRPO works well when you can run the model many times per prompt and
score the outputs with a verifiable reward (math correctness, code test pass, etc.).
It is less well suited to open-ended generation where such a ground-truth signal is
not available.

### The KL leash in all three methods

All three methods share the same underlying tension: you want to move the policy
toward a better behavior, but you cannot let it drift so far that it loses its
baseline ability or degenerates into reward-hacking gibberish. The $\beta$
coefficient (or its equivalent KL coefficient) is the leash.

![KL penalty (beta): reward vs policy drift](assets/fig-kl-penalty.png)

*Illustrative: task reward peaks near beta ~ 0.03 to 0.1, then falls as the leash
tightens the policy back toward the SFT reference. Too small a beta and the policy
reward-hacks into degenerate text; too large and it over-steers into sycophancy or
evasion. The DPO beta plays the identical anchoring role as the RLHF KL
coefficient.*

![Pipeline complexity: SFT vs DPO vs RLHF](assets/fig-pipeline-complexity.png)

*DPO trains two models (policy + frozen reference) and no RL loop. RLHF requires
five components. The simplicity of DPO is why it is the common first choice when
preference tuning is needed.*

### Compare and contrast: DPO vs RLHF

Both are preference alignment: the same human comparison data, the same goal of
shifting the policy toward preferred behavior, and the same KL leash to a frozen
reference. The common confusion is treating DPO as "RLHF, but cheaper." The
mechanics differ in where the reward lives and what data the policy learns from.

| Dimension | DPO | RLHF (PPO) |
|---|---|---|
| Training signal | human preference pairs (chosen, rejected) | the same human preference pairs |
| Anchor against drift | KL to the frozen reference, expressed via beta inside the closed-form loss | KL to the frozen reference, as an explicit penalty term |
| Reward model | implicit: the policy-to-reference log-ratio acts as the reward | explicit $r_\phi$, trained separately before the RL step |
| Data the policy learns from | offline: only the fixed labeled pairs, reweighted | online: fresh samples drawn from the current policy, scored by $r_\phi$ |
| Models held during training | 2 (policy + reference) | 4 to 5 (policy, reference, reward model, value network) |
| Reward reusability | none: the preference signal is baked into the loss | the reward model is a standalone artifact, reusable for data filtering, rejection sampling, and future runs |

The difference changes the design when the policy must move far from where the
labeled pairs sit: RLHF's reward model scores the policy's own fresh samples, so
the training signal follows the policy as it moves, while DPO can only reweight
the fixed pairs; long alignment campaigns and teams that want reusable reward
infrastructure favor RLHF, single-shot preference nudges favor DPO.

## When to use which

| Reach for | When | Instead of |
|---|---|---|
| SFT only | format, tone, or skill gap with clean labeled examples; behavior is stable | preference tuning, which adds cost for a problem SFT already solves |
| LoRA adapter (r=8 to 64) | small-to-moderate behavior shift; many tenants share one base; fast rollback matters | full fine-tuning, which costs more and blocks hot-swappable adapters |
| QLoRA | same as LoRA but the frozen base must fit a single consumer GPU | 16-bit full weights, which will not fit the memory budget |
| Full fine-tune | large dataset, large behavior shift, or LoRA drifts out of distribution | raising LoRA rank arbitrarily, which rarely fixes an OOD result |
| DPO | preference axis SFT cannot capture; want no separate reward model; simple, stable pipeline | full RLHF, when a classification-style loss over (chosen, rejected) is sufficient |
| RLHF | you need a reusable reward signal or finer control via learned reward model | DPO, when you do not need online RL or a separate reward model |
| GRPO | verifiable reward exists (math, code, retrieval rank); no value function available | RLHF when the reward is not cheaply verifiable per sample |
| Small beta (0.03 to 0.1) | first run; stability matters; Anyscale and Spotify both used this range | large beta, which over-steers the policy back to the SFT reference |

**Provenance.** LoRA came from Microsoft (2021) and QLoRA from the University of
Washington (2023). RLHF was popularized by OpenAI's InstructGPT (2022), whose RL
step uses PPO (OpenAI, 2017); DPO came from Stanford (2023) as a reward-model-free
alternative, and GRPO from DeepSeek (2024) as a value-function-free variant for
verifiable rewards.

**Tools for each method.** Hugging Face TRL implements SFT, DPO, and GRPO through its
SFTTrainer, DPOTrainer, and GRPOTrainer, and PEFT supplies the LoRA and QLoRA
adapters (QLoRA pairs PEFT with bitsandbytes 4-bit quantization). Axolotl and Unsloth
wrap the same TRL and PEFT stack behind declarative configs, with Unsloth focused on
single-GPU speed and memory. Full fine-tuning and RLHF at scale lean on DeepSpeed
(Microsoft) ZeRO for optimizer and gradient sharding across GPUs. GRPO with a
verifiable reward is served by the same TRL GRPOTrainer plus a scoring function you
write for the code or math check.

**Worked example.** A domain-LLM team adapting a 7B base to their document style has
a few thousand clean pairs and no spare GPU cluster, so they reach for QLoRA rather
than full fine-tuning, because the frozen 4-bit base plus a rank-16 adapter fits a
single consumer GPU and the behavior shift is moderate. Since the gap is format and
tone with stable examples, they stop at SFT and skip preference tuning, which would
add cost for a problem SFT already solves. Later they find the model sometimes picks
a confidently wrong phrasing over a safer one, a comparative preference SFT cannot
express, so they add DPO with a small beta around 0.05 rather than standing up the
full RLHF pipeline, because a classification-style loss over chosen and rejected
pairs is enough and needs no separate reward model. They would only escalate to GRPO
if the reward were cheaply verifiable per sample, which for open-ended tone it is not.

> **Open the graph.** LoRA adapts a small fraction of these stacks, and "a small
> fraction" is abstract until you see the real dimensions. The attention query,
> key, value, and output projections plus the FFN up and down matrices are where
> the learned low-rank update lives; the rest is frozen. Open
> [Llama-3 8B live](https://www.neurarch.com/?import=https://raw.githubusercontent.com/neurarch-ai/awesome-llm-model-zoo/main/architectures/llama3-8b/model.json)
> and find those weight matrices to see how little of the network an adapter
> actually moves. All reference graphs are in the
> [Model Zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo).

## Implementation and training pitfalls

Most fine-tuning failures are not exotic; they are a handful of recurring problems.
The first diagnostic is always the loss curve, so learn to read it.

![Reading training curves: four diagnostics](assets/fig-training-diagnostics.png)

*Four shapes a training run takes. Healthy: train and validation loss both fall and
stay close. Overfitting: validation loss bottoms out then rises while training loss
keeps falling, so stop at the turning point (early stopping). LR too high: the loss
oscillates or climbs instead of settling, so lower the learning rate or add warmup.
Underfitting: the loss stays high and flat, so the model, data, or learning rate is
too small. Illustrative curves.*

| Problem | Symptom | Fix |
|---|---|---|
| Learning rate too high | loss spikes, oscillates, or diverges (bottom-left above) | lower the LR, add linear warmup, clip gradients at norm 1.0 |
| Overfitting on a small SFT set | train loss keeps dropping, validation loss rises | early-stop at the validation minimum, fewer epochs (1 to 3), add held-out eval |
| Loss looks fine, model got worse | eval regresses despite low loss | check for train/eval contamination and format drift between training and serving |
| Catastrophic forgetting | fine-tuned model loses general ability | mix in a fraction of general data, prefer LoRA over full fine-tuning, lower LR |
| DPO reward hacking / degeneracy | outputs get shorter or repetitive, reward up but quality down | raise beta (keep the policy near the reference), cap length, re-check the preference data |
| DPO/RLHF instability | reward or KL blows up mid-training | smaller LR, larger beta or KL coefficient, verify the reference model is frozen |
| Chat template mismatch | model ignores system prompt or mis-parses turns | use the exact same chat template in training and serving; pin it |
| QLoRA out-of-memory | OOM at load or first step | 4-bit base with NF4, gradient checkpointing, smaller batch with gradient accumulation |

The through-line: change one thing at a time, always watch a held-out eval next to
the loss (a low loss with a falling eval is the classic trap), and keep the training
and serving formats byte-identical.
