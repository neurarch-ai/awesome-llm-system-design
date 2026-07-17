# 8. Interview Q and A

The questions a hiring committee actually asks about continued pretraining and
long-context extension, grouped by how they are used. The commonly-missed ones
are where interviews are won or lost.

## Commonly asked

**Q: Why continued pretraining instead of just fine-tuning on domain examples?**

A: SFT on domain examples teaches the model a format and narrows behavior; it does
not shift the base's broad factual prior, vocabulary, or register. A clinical model
that has only seen a few thousand instruction pairs will not deeply know the domain,
it will just respond in a clinical-sounding format. Continued pretraining on
billions of tokens shifts the actual prior: the model learns the vocabulary,
abbreviations, factual density, and style of the domain from the bottom up. The two
methods compose: DAPT the base first, then SFT and align it. SFT alone is the
shortcut that skips the expensive part and gets the cheaper-looking result.

**Q: Why does naive extrapolation fail past the trained context window?**

A: RoPE assigns each dimension pair a per-position rotation angle proportional to
position times the frequency. The model was trained on positions 0 through
$L_{\text{orig}}$, so it has only seen the rotation angles those positions produce.
Positions beyond $L_{\text{orig}}$ produce angles the model has never trained on,
and the attention scores become arbitrary noise. Setting `max_position_embeddings`
to a larger number in the config changes nothing in the weights; the model cannot
use those positions. Extension requires rescaling the frequencies and a short
continued-training run to adapt to the rescaled angles.

**Q: What is the re-warm schedule, and why is the re-warm peak critical?**

A: The base model ended its pretraining with a fully decayed, near-zero learning
rate. Resuming at that floor gives gradients too small to learn a new domain, so
the run stalls. Resuming at the original pretraining peak blows away the converged
weights and causes catastrophic forgetting. The fix is to re-warm from the floor
to a modest peak (a fraction of the original) and then re-decay. The re-warm peak
is the main hyperparameter: higher means more domain learning and more forgetting;
lower means less forgetting and a slower domain shift. This single value is what
you tune when the DAPT run either stalls or regresses on general benchmarks.

**Q: What is RULER and why is NIAH not enough to prove a long-context model?**

A: Needle-in-a-haystack (NIAH) hides one fact in a long filler passage and asks
the model to retrieve it. It is single-hop retrieval of a verbatim fact, often
anchored near an edge of the window where recall is best. A model can pass NIAH
and still be broken at real long-context tasks. RULER extends the suite to multiple
needles, multi-hop variable tracing, aggregation over the span, and long-context
QA. NVIDIA's finding is that most models claiming 32K or more degrade sharply
before their advertised length; RULER is what reveals the effective context, which
is almost always shorter than the configured one. Gate on RULER, not NIAH.

**Q: When does long context beat RAG, and when does RAG beat long context?**

A: Long context for a single large document the model must reason over whole, where
retrieval would split the reasoning across chunk boundaries. RAG for a corpus you
retrieve chunks from, because quadratic prefill, linear KV-cache growth, and
lost-in-the-middle recall make long context unscalable to a collection. They
compose: extend for the single document, retrieve over the corpus. Long context
replacing RAG is a wrong answer.

## Tricky (the follow-ups that separate people)

**Q: You extended to 128K and perplexity on long documents looks fine, but users
say the model cannot find things. What went wrong?**

A: Perplexity is dominated by local next-token prediction; it stays low even when
long-range retrieval is broken. The model can predict the next word well (mostly
from the preceding sentence) while completely ignoring a fact at position 60K.
Perplexity measures fluency at length; it does not measure use of the context.
Gate on RULER-style retrieval and aggregation tasks and on recall-by-depth, not on
perplexity. A low perplexity on long documents is necessary but not sufficient.

**Deeper:** Perplexity averages over every token position, and the large majority of tokens are predictable from a short local window, so one unretrieved fact at 60K barely moves the mean. Recall-by-depth isolates exactly the positions that this average drowns out, which is why it exposes the mid-context dip that a single perplexity number hides.

**Q: Your DAPT lifted the domain benchmark by six points but MMLU dropped by
five. Diagnose and fix.**

A: Catastrophic forgetting. The domain corpus pushed the optimizer into the domain
minima while the gradient signal rewarding the general ones vanished. Three likely
causes: the re-warm peak was too high (too aggressive a perturbation), there was
no or too little general-data replay, or the domain data set was too small and the
model overfit it. Fix by adding a replay fraction of general web data (start at
ten percent), lowering the re-warm peak, and re-running the full general suite as
the gate before promoting the base.

**Deeper:** Replay works because interleaving general tokens keeps a live gradient on the general distribution at every step, so the optimizer never sees a run of pure-domain batches long enough to walk out of the general minimum. A replay fraction fixed once and held constant is usually more stable than annealing it, since a decaying replay share reintroduces late-training drift exactly when the weights are consolidating.

**Q: Why does uniform position interpolation hurt short-context quality after
extending to 128K?**

A: Uniform PI divides every RoPE frequency by the length scale $s$, including the
high-frequency dimensions that encode local ordering of adjacent tokens. Compressing
those crowds nearby positions together: the model loses resolution between
consecutive tokens and gets worse at short prompts where local ordering is most
important. YaRN fixes this by leaving the high-frequency dimensions near-unscaled
and interpolating only the low-frequency (long-range) ones.

**Deeper:** The high-frequency dimension pairs complete a full rotation within a handful of token positions, so they are the ones that encode adjacency. Dividing their frequency by $s$ stretches that short period, and two neighboring tokens now sit at almost the same angle, which is precisely why local ordering blurs. YaRN's ramp keeps those dimensions near their original period and confines the interpolation to the slow, long-wavelength dimensions where the resolution loss does not matter.

**Q: How much compute does context extension actually take compared to pretraining?**

A: Far less. YaRN reaches long context at roughly 0.1 percent of the original
pretraining tokens, and position interpolation recovers quality in about a thousand
fine-tuning steps. The reason is that the frequencies encode relative position
cheaply; the model mostly adapts to the rescaled angles rather than relearning
language from scratch. Staged extension is even cheaper: early stages use shorter
sequences and consolidate cheaply before the final length. The whole length
adaptation is a tiny fraction of the original pretrain budget.

**Deeper:** The reason it is so cheap is that RoPE encodes position relatively and the semantic weights are already trained, so the run only has to teach attention to read the rescaled angles rather than re-learn language. Most of the loss recovery happens in the first few hundred steps; the long tail buys diminishing returns, which is why staged extension can stop each stage early.

**Q: Lowering the re-warm peak and raising the replay fraction look similar,
both reduce forgetting; when does the difference actually matter?**

A: They act through different mechanisms. The re-warm peak controls the size of
every weight update, so lowering it shrinks the perturbation to all
capabilities at once: less forgetting, but also proportionally less domain
learning. Replay changes the direction of the updates, mixing general-data
gradients into each step so the optimizer is pulled toward weights that fit
both distributions, while the domain gradient keeps its full step size. The
difference matters when you need a large domain shift: cutting the peak far
enough to protect MMLU also stalls the domain gain, whereas keeping a moderate
peak and adding ten percent replay lets domain learning proceed at speed while
the replayed gradient holds the general minimum. If instead the domain is
close to the pretraining distribution, the two knobs are nearly
interchangeable and the cheaper one (no extra data pipeline, so lower peak)
wins.

## Commonly answered wrong (the traps)

**Q: To get a clinical model, fine-tune on our clinical documents.**

A: SFT on documents teaches format and narrows behavior. A few thousand clinical
notes cannot shift the model's broad factual prior, clinical vocabulary, or note-
taking register at depth. Continued pretraining on billions of clinical tokens is
what does that. If you have only a few thousand documents, use RAG to retrieve
relevant chunks rather than hoping SFT will shift the base.

**Deeper:** The tell is that SFT loss can look excellent while the model still hallucinates domain facts, because SFT optimizes the response conditioned on the prompt, not the base's underlying factual prior. Only continued pretraining on raw domain text moves that prior, which is why a fluent-sounding but factually shaky clinical model is the signature of SFT-only adaptation.

**Q: To get 128K context, set `max_position_embeddings` to 128000.**

A: That changes a config number, not the model. Without rescaling the RoPE
frequencies the model sees rotation angles it was never trained on and produces
garbage past its original window. Without long-context continued training it never
learns to use the extended window. Extension is a rescaling plus a training run
on genuine long documents, not an edit.

**Deeper:** That field only sizes the position-embedding table or caps the allowed position index. For a RoPE model there is no learned position table to extend at all: the rotation angles are computed on the fly from position times frequency, so raising the config number just lifts the guardrail on a model whose weights have never seen those angles.

**Q: YaRN is just linear interpolation with an extra term.**

A: No. Linear position interpolation (Chen et al. 2023) divides every RoPE
frequency by the same factor. YaRN is deliberately non-uniform: it interpolates
the low-frequency (long-wavelength) dimensions, leaves the high-frequency (short-
wavelength) ones near-unscaled to preserve local ordering, and adds a softmax
temperature correction to counter the attention-entropy shift at long length.
Conflating them is the most common technical error about context extension.

**Deeper:** Concretely, YaRN partitions the RoPE dimensions by wavelength relative to the target length into interpolate, extrapolate, and ramp bands, so the rescale is per-dimension rather than one global factor. The softmax-temperature term is a separate correction that rescales attention logits to hold the attention entropy roughly constant as the sequence grows, addressing a failure mode linear PI does not even touch.

**Q: Our model passes the needle test, so 128K works.**

A: Needle-in-a-haystack is single-hop retrieval of one verbatim fact and is often
anchored near the edges of the window where recall is highest. RULER's multi-hop,
aggregation, and multi-needle tasks routinely fail on models that pass NIAH,
because the effective context is shorter than the configured one. A NIAH pass is a
minimum smoke test, not a proof of 128K.

**Deeper:** Multi-hop fails because it forces the model to retrieve fact A, then use A to locate fact B elsewhere in the same long span, which stresses cross-position binding rather than a single lookup. Edge-anchored, single-needle NIAH never exercises that chained dependency, so it cannot distinguish a model that truly attends across the window from one that only reads its ends.

**Q: General benchmarks did not drop, so there was no catastrophic forgetting.**

A: Only if you ran them. Forgetting is silent inside the domain slice and shows up
outside it. If you ran only the domain benchmark, you cannot make this claim.
Run the full general suite (MMLU, GSM8K, instruction following) before and after,
compare, and then report the result. Asserting no forgetting without a before-and-
after general-eval gate is the most common and quietly fatal mistake in DAPT.

**Deeper:** Forgetting concentrates in capabilities the domain corpus never exercises, such as math reasoning or code, so a domain-only eval is structurally blind to it. This is why the gate has to run the specific general suites that stress those capabilities rather than a single averaged score, which can stay flat while a targeted skill collapses underneath it.
