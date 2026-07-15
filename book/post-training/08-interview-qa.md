# 8. Interview Q&A

The questions an interviewer actually asks about fine-tuning and post-training,
grouped by how they are used. The commonly-missed ones are where interviews are
won or lost.

## Commonly asked

**Q: Someone says the model is not good enough at our task. What is the first
thing you do?**
A: Ask what "not good enough" means concretely, then try a better prompt before
anything trains. Wrong facts want retrieval, not fine-tuning. Wrong format or tone
want prompt engineering first, then SFT if the gap survives. A genuinely missing
skill that prompt engineering cannot inject is the real fine-tuning case.
Candidates who open with "we should fine-tune" have already lost the plot.

**Q: What is LoRA and why does everyone use it?**
A: LoRA freezes the base weights and learns a pair of low-rank matrices ($B$ and
$A$) that represent the change a task needs. The base stays untouched; you train
and store only the small adapter. The practical wins are: (a) the adapter is
tiny (under 1% of the base at typical ranks), (b) the base can host many adapters
at once, enabling multi-LoRA serving, and (c) rollback is a route change, not a
redeploy. Use QLoRA when the frozen base must fit a single GPU.

**Q: What is the difference between SFT, DPO, and RLHF?**
A: SFT trains the model to imitate positive examples by minimizing the
cross-entropy loss on ideal responses. It teaches behavior, format, and skill
by example. DPO trains on comparison pairs (chosen, rejected) with a
classification-style loss that directly maximizes the likelihood ratio of the
chosen over the rejected, relative to a frozen reference model; no separate reward
model, no RL loop. RLHF trains a separate reward model from human preferences,
then optimizes the policy against that reward with PPO, subject to a KL penalty.
The ordering for when to reach for each: SFT first; DPO if there is a preference
axis SFT cannot capture; RLHF if you need a reusable reward signal or finer
control.

**Q: How do you know whether the new model is actually better?**
A: Pass it through an eval gate: beat the current production model on a held-out,
decontaminated eval set, plus a safety pass, plus (for preference-tuned models)
a pairwise win rate above the threshold. Then gate the launch on a live traffic
slice before scaling. Offline metrics overstate readiness; the live gate is the
real one.

**Q: What is the data flywheel and why does it matter?**
A: Production logs are your richest training signal because they come from the real
input distribution. Mine them for failures (thumbs-down, escalations, corrections),
label the hard ones, fold them into the next dataset version, retrain, gate,
promote. A mediocre first model plus a tight, well-gated loop beats a great first
model with no feedback path, because the loop targets exactly where the current
model is weakest.

## Tricky (the follow-ups that separate people)

**Q: What does the beta term do in DPO, and what goes wrong if it is too small or
too large?**
A: Beta is the KL penalty coefficient. It controls how far the policy is allowed
to move from the frozen reference model. Too small: the policy moves unconstrained
toward maximizing the preference objective and can reward-hack into degenerate
output or sycophancy. Too large: the policy is pinned so close to the reference
that it barely moves; you have essentially re-run SFT with extra bookkeeping.
Anyscale used beta=0.03; the sweet spot is typically 0.03 to 0.1. This is the
same leash that RLHF's KL penalty imposes, just expressed in closed form inside
DPO's loss.

**Q: Anyscale found that LoRA rank 64 underperformed full fine-tuning on their DPO
task. Why would that happen, and when does it happen?**
A: LoRA constrains the weight update to a low-rank subspace. For a small behavior
nudge that lives in a low-rank manifold, this is fine. When the task requires a
large or geometrically complex shift, the constrained subspace pushes token
log-likelihoods out of distribution: the adapter gets the right direction but
cannot reach the right magnitude within its rank budget. Increasing rank helps
marginally but rarely closes the gap; full fine-tune is the correct call when the
shift is large. The tell is when loss converges but downstream accuracy plateaus.

**Q: How does Shopify's LLM-judge flywheel avoid the circularity of training on
what the model itself generated?**
A: Two mechanisms. First, the judge is calibrated against human labels and
live activation rate on a held-out sample periodically, so it tracks real product
quality rather than the model's self-assessed quality. Second, the flywheel
quarantines low-quality production conversations before they enter training:
only conversations the judge rates above a threshold are used, and the quarantine
threshold is set to match the human calibration. A judge that drifts away from
humans is caught in the periodic calibration step and retrained or replaced before
it poisons the training set.

**Q: A model passes the offline eval gate but activation rate in the 1% traffic
slice is 35 points lower than expected. What happened and what do you do?**
A: The offline eval was measuring the wrong distribution. Common causes: (a) the
eval set was cleaned more carefully than production inputs, so production edge cases
are outside what the model saw during training or evaluation; (b) the task
representation in training (e.g., Shopify used Python vs native JSON DSL) diverges
from production format; (c) the eval examples are close to training examples in
distribution (near-contamination that was not caught). Steps: diff the offline eval
set against live samples to find the gap, add representative production examples to
the eval set, fix the format/representation mismatch, and close the flywheel so
the next training run sees the live distribution.

**Q: GRPO drops the value network that PPO relies on. What replaces it, and why is
that safe here?**
A: GRPO (DeepSeek, 2024) samples a group of G completions for the same prompt,
scores each with a verifiable reward, and computes each sample's advantage as its
reward minus the group mean, normalized by the group standard deviation. PPO
(OpenAI, 2017) instead trains a separate critic network to estimate the baseline
value, which doubles model memory and gives the policy a moving target to chase.
GRPO's baseline is just the batch statistic over the group, so it needs no learned
critic and stays unbiased within the group, which is exactly why it fits
verifiable-reward tasks (math, code, retrieval rank) where every sample can be
scored cheaply. The cost it moves elsewhere: you must draw G samples per prompt, so
training-time inference scales with the group size.

## Commonly answered wrong (the traps)

**Q: Should you fine-tune to teach the model new facts?**
A: Almost never. Fine-tuned facts go stale and must be retrained when they change.
The model hallucinates confidently between the facts it was trained on. The right
tool for knowledge that changes is retrieval (RAG): put the facts in context, cite
sources, update instantly. Fine-tuning teaches behavior and skill; retrieval
teaches knowledge. They compose well: tune for style, retrieve for facts, on the
same base.

**Q: Should you run RLHF on a format-and-tone problem to get the highest quality?**
A: No. RLHF is a five-component pipeline (SFT model, reward model, reference model,
value network, PPO loop) with significant operational complexity and a real risk of
over-steering the model into sycophancy or excessive refusal. For format and tone,
good SFT is almost always sufficient and far cheaper to run and debug. Proposing
RLHF for a problem SFT already solves is the most common over-engineering mistake
in this domain.

**Q: Does LoRA need to run on a separate adapter server from the base?**
A: No. Multi-LoRA serving loads the base once and batches requests across different
adapters against the same base (the Punica SGMV kernel). Adapter swap is a
millisecond-scale matrix add/subtract. The base is always warm; the adapters load
on demand from cache or object storage. Running a separate server per adapter
throws away the entire economic case for LoRA, and it is the reflex answer that
reveals a lack of production experience.

**Q: Can you skip the regression check and just measure task accuracy on the new
model?**
A: No. The regression check is what catches silent regressions on secondary tasks.
A new model that improves the target metric by two points while degrading safety,
tone, or a secondary capability by five points should fail the gate. Without a
comparison to the current production model on the same eval set, "newer" quietly
ships "worse" and you find out from user complaints. The regression check is not
optional.

**Q: QLoRA stores the base in 4-bit, so the adapter you train is 4-bit too, right?**
A: No, and the split is the whole point. QLoRA (University of Washington, 2023)
keeps the frozen base weights in 4-bit NF4 and dequantizes them on the fly for each
forward pass, but the LoRA adapter matrices are held and trained in 16-bit (bf16).
Gradients flow only into the 16-bit adapter; the 4-bit base is never updated. You
therefore get the memory savings of a 4-bit base without quantizing the parameters
you are actually learning, so the learned delta keeps full precision. Assuming the
adapter is 4-bit leads people to expect an accuracy hit on the update that does not
occur, and to over-provision rank to compensate for a problem that is not there.
