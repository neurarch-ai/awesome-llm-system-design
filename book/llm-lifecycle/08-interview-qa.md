# 8. Interview Q&A

The questions that actually come up in LLM system design loops, grouped by how
they are used. The commonly-missed ones are where interviews are won or lost.

## Commonly asked

**Q: Walk me through the LLM lifecycle. What are the five stages?**

A: Data preparation (dedup, filter, decontaminate, tokenize), pretraining
(self-supervised next-token prediction over trillions of tokens to produce a
base model), mid-training (continued pretraining on domain data or long-context
documents), post-training (SFT then preference optimization to produce an
aligned instruct model), and deployment and inference (quantize, serve with
paged KV cache and continuous batching, bolt on RAG and tools). Most product
teams enter at the base model, skipping stages 1 and 2. The expensive stages
are upstream and shared; the stages you iterate on are downstream.

---

**Q: Should we pretrain our own model?**

A: Almost never for a product team. From-scratch pretraining is a lab-scale
budget (hundreds of millions of dollars, weeks of GPU cluster time, hundreds of
billions of tokens). The leverage for a product team is in mid-training a strong
open base (Llama 3, Qwen3, OLMo, DeepSeek-V3) on your proprietary domain data,
then post-training it to follow your instructions. Justify owning weights before
proposing to train them from scratch. The usual reasons are: data residency, cost
at volume, latency, a capability the API refuses, or a domain the open bases are
genuinely weak in.

---

**Q: RAG or fine-tuning for our private knowledge base?**

A: RAG for facts that change or must be cited; fine-tuning for behavior, format,
tone, and skills. They compose: fine-tune the style, retrieve the facts. Do not
fine-tune to memorize a changing database; the facts bake in stale and
hallucination follows. If someone says "fine-tune it on our knowledge base," the
first question is whether the knowledge changes and whether it needs a citation.
If yes to either, RAG.

---

**Q: SFT vs RLHF vs DPO vs RL from verifiable rewards: when to use each?**

A: SFT teaches the format and basic instruction following; it is the first step
regardless of what comes next. Preference methods teach which of two correct
answers humans prefer. DPO is the cheap stable default (offline pairs, no reward
model, no PPO loop); use it first. PPO-style RLHF when you need a reusable
reward model or are willing to pay the engineering cost for the best-in-class
alignment ceiling. Verifiable-reward RL (GRPO, DeepSeek-R1) for math, code, or
any domain where a checker gives binary correct/wrong rewards; it is both cheaper
and harder to hack than a learned preference model.

---

**Q: Why is inference the expensive part?**

A: Training is a one-time capital cost. Inference is the recurring operating cost
that runs as long as the product runs. Decoding is memory-bandwidth bound
(each token requires a full read of all model weights plus the KV cache from
HBM), not compute bound. Throughput, not FLOPs, is the limit. The KV cache
grows with batch times sequence length and dominates VRAM. The levers are paged
KV cache (vLLM), grouped-query attention (smaller KV cache at source), continuous
batching (amortize the weight read), prefix caching, speculative decoding, and
quantization.

## Tricky (the follow-ups that separate people)

**Q: DPO has no reward model and no RL loop. Why does it still need a reference
model, and what is beta doing?**

A: DPO's loss contains the ratio $\log(\pi_{\theta} / \pi_{\text{ref}})$. The
reference model is the implicit reward's baseline: the RLHF-optimal policy has a
closed form that depends on both the reward and the reference, and substituting it
into Bradley-Terry produces the DPO loss. Without a reference, the policy has no
anchor and can drift to degenerate solutions. $\beta$ is the same KL temperature
as in RLHF: small $\beta$ lets the policy move far from the reference (more
optimization, more drift risk); large $\beta$ keeps it close. DPO absorbed the KL
leash into the loss; it did not remove it.

---

**Q: Chinchilla says 20 tokens per parameter. Llama 3 8B used roughly 1800.
Who is wrong?**

A: Neither. Chinchilla minimizes training compute for a target loss. Llama 3
optimizes a different objective: minimize the lifetime cost of training plus
inference, where inference at billions of tokens dominates. A smaller model
trained on far more tokens is cheaper to serve forever. Different objective,
different optimum. State which cost you are optimizing.

---

**Q: Your eval jumped 8 points after a data refresh. What is the first thing
you check?**

A: Contamination. A benchmark gain from new data is guilty until proven clean.
Re-run decontamination (n-gram or embedding overlap of eval against train) before
believing or shipping the number. An 8-point gain that traces to eval leakage is
not a gain; it is a reporting failure.

---

**Q: Your GQA model has the same parameter count as the MHA one. Why is it
faster at inference but not at training?**

A: Training is compute-bound and processes a full batch of tokens per step;
fewer KV heads barely reduces the FLOPs. Decoding is memory-bandwidth-bound
and re-reads the KV cache every token; a $4\times$ smaller KV cache (32 query
heads, 8 KV heads) is a direct and permanent decode speedup. The GQA win is a
serving win, not a FLOPs win.

---

**Q: Quantize to INT4 or distill to a smaller model for cost reduction?**

A: Start with quantization; it is a near-zero engineering investment. INT8 is
nearly lossless; INT4 needs an eval gate but halves memory again. Distillation
costs a training run but can beat quantization on latency and quality for a
fixed memory budget, because it is a smaller architecture rather than the same
architecture at lower precision. The two compose: distill then quantize. Begin
with INT8, move to INT4 under an eval gate, and reach for distillation when
the model architecture itself is the bottleneck.

## Commonly answered wrong (the traps)

**Q: "We will just fine-tune it on our knowledge base."**

Wrong. Fine-tuning teaches behavior and style; it bakes facts into weights at a
point in time and cannot cite a source. Correct: RAG for the facts, fine-tune
only the behavior.

---

**Q: "Bigger model means lower perplexity means better product."**

Wrong twice. Perplexity is tokenizer-dependent and measures next-token fit, not
usefulness. Product quality comes from post-training, data quality, and serving,
not raw size. Name the right metric per stage.

---

**Q: "RLHF is just fine-tuning on good answers."**

That is SFT, not RLHF. RLHF optimizes a preference signal (which of two answers
is better) under a KL constraint via a reward model, which is a different
objective that can improve on any single demonstration and can also reward-hack
if the leash is dropped.

---

**Q: "Add safety tuning and we are done on safety."**

Safety is measured, not asserted. Track attack success rate, false-refusal rate,
and jailbreak robustness as release gates. Assume adversarial evasion is
continuous, including prompt injection through tool calls and RAG retrieval.
Layer input/output filters, isolate tool inputs, and red-team continuously.

---

**Q: "Benchmark scores are state of the art."**

Meaningless without a decontamination claim. The first question is whether the
eval leaked into training. Lead with the decontamination check; do not wait to
be asked.
