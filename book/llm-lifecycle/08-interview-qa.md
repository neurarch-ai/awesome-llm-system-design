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

**Why over-training a small model works at all:** loss keeps falling past the
compute-optimal token count, just with diminishing returns, so the extra training
tokens buy real quality. Meanwhile serving cost scales with parameter count on every
generated token for the life of the product, so a one-time overpayment in training
compute is repaid on billions of inference calls.

---

**Q: Your eval jumped 8 points after a data refresh. What is the first thing
you check?**

A: Contamination. A benchmark gain from new data is guilty until proven clean.
Re-run decontamination (n-gram or embedding overlap of eval against train) before
believing or shipping the number. An 8-point gain that traces to eval leakage is
not a gain; it is a reporting failure.

**Why contamination is the prime suspect:** a data refresh usually pulls newer web
scrapes, and the web increasingly contains published benchmark questions and answers
verbatim (papers, blog posts, GitHub dumps of the test sets). A model that has
memorized test items recalls them instead of reasoning, so the score inflates with no
capability change, which is exactly the signature of a sudden jump tied to a data
change rather than a method change.

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

**Why INT8 is safe but INT4 is not:** 8 bits gives 256 quantization levels, enough
that rounding error stays below the noise floor of most layers, while 4 bits gives
only 16 levels, so the error per weight grows sharply and outlier channels (a few
weights with much larger magnitude than their neighbors) can dominate the
reconstruction error of a whole layer. That is why INT4 methods like GPTQ and AWQ
need calibration data and outlier handling, and why you gate INT4 on your own evals
rather than trusting a benchmark table.

---

**Q: PPO keeps a separate value network to estimate the advantage. GRPO drops it.
Where does GRPO get its advantage signal from instead?**

A: From the group, not from a critic. GRPO (DeepSeek, 2024) samples a group of $G$
completions for the same prompt, scores each with the verifier, and defines each
completion's advantage as its reward standardized against the group's own mean and
standard deviation. The mean of the sampled rewards is the baseline that PPO would
otherwise learn a value head to predict, so a per-prompt Monte Carlo baseline
replaces the learned critic entirely. That is why GRPO is called critic-free: it
halves the number of large models resident in the training loop (no value network to
host and update) and sidesteps the value-function fitting that makes PPO finicky. The
tradeoff is that you must sample several completions per prompt, so it leans on
cheap, high-throughput verifiable rewards (unit tests, math graders) rather than an
expensive learned reward model.

---

**Q: Mid-training (continued pretraining) and SFT look similar; both are "keep
training the model on your data." When does the difference actually matter?**

A: The objective and the data shape are different, so they change different things.
Mid-training runs the same self-supervised next-token objective as pretraining, on
raw domain text at billions-of-tokens scale; it shifts what the model knows and which
distributions it models well. SFT runs supervised learning on curated
instruction-response pairs at thousands-to-millions scale; it shifts how the model
behaves: answering instead of completing, following formats, adopting a persona. The
difference matters when you diagnose a failure: a model that writes fluent prose but
does not know your domain's vocabulary and facts needs mid-training (more SFT will
just teach it to confidently phrase what it does not know), while a model that knows
the domain but rambles or ignores instructions needs SFT (more raw domain text will
not teach it to answer). Mechanically, mid-training also risks catastrophic
forgetting at its scale, which is why it mixes in replay data from the original
distribution, a concern SFT rarely hits at its much smaller token counts.

## Commonly answered wrong (the traps)

**Q: "We will just fine-tune it on our knowledge base."**

Wrong. Fine-tuning teaches behavior and style; it bakes facts into weights at a
point in time and cannot cite a source. Correct: RAG for the facts, fine-tune
only the behavior.

**Why baked-in facts turn into hallucinations:** gradient descent stores a fact as a
diffuse pattern across many weights with no record of where it came from, so the
model cannot distinguish a memorized fact from a plausible interpolation. When the
real-world fact changes, the weights still encode the old version and the model
states it with full confidence, because nothing in the architecture marks stored
knowledge as stale.

---

**Q: "Bigger model means lower perplexity means better product."**

Wrong twice. Perplexity is tokenizer-dependent and measures next-token fit, not
usefulness. Product quality comes from post-training, data quality, and serving,
not raw size. Name the right metric per stage.

**Why tokenizer dependence breaks the comparison:** perplexity is exponentiated loss
per token, and different tokenizers cut the same text into different numbers of
tokens of different difficulty, so two models with different vocabularies are being
scored on different denominators. And even a legitimately lower next-token loss only
says the model predicts text well; it says nothing about instruction following,
refusal behavior, or factual grounding, which are what users experience and what
post-training determines.

---

**Q: "RLHF is just fine-tuning on good answers."**

That is SFT, not RLHF. RLHF (popularized by InstructGPT, OpenAI 2022) optimizes a
preference signal (which of two answers is better) under a KL constraint via a reward
model, which is a different objective that can improve on any single demonstration
and can also reward-hack if the leash is dropped. The mechanism-level reason it beats
SFT-on-good-answers: SFT can only push probability toward demonstrated tokens, so it
is capped by the best demonstration in the dataset, whereas a reward model scores
arbitrary samples and lets the policy climb above any single human answer by
comparing its own generations. That same freedom is why the KL leash matters: with it
removed, the policy drifts to whatever degenerate text maximizes the imperfect reward
model, which is reward hacking.

---

**Q: "Add safety tuning and we are done on safety."**

Safety is measured, not asserted. Track attack success rate, false-refusal rate,
and jailbreak robustness as release gates. Assume adversarial evasion is
continuous, including prompt injection through tool calls and RAG retrieval.
Layer input/output filters, isolate tool inputs, and red-team continuously.

**Why one-time safety tuning decays:** tuning shifts the policy on the attack
distribution seen during training, but attackers search off that distribution, and
every new tool or retrieval source adds an input channel the tuning never saw. A
static defense against an adaptive adversary loses by default, which is why safety
has to be operated as a measured, layered process rather than shipped as a
checkpoint property.

---

**Q: "Benchmark scores are state of the art."**

Meaningless without a decontamination claim. The first question is whether the
eval leaked into training. Lead with the decontamination check; do not wait to
be asked.

**Why leakage is the default assumption:** benchmarks are published on the same web
that pretraining corpora scrape, so any benchmark older than the training cutoff has
plausibly been seen. A contaminated score measures recall of the answer key, not the
capability the benchmark was designed to test, and no amount of statistical
significance on the score itself can distinguish the two.
