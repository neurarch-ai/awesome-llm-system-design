# 5. Evaluation

The most common integrity failure in continued pretraining and long-context work
is declaring victory on a weak eval. A model can pass the easy test and be broken
on anything real. This section names each eval, explains what it actually measures,
and explains what it misses.

## Forgetting checks: the general-benchmark gate

Before running any domain eval, run the full general-benchmark suite on the
adapted base and compare to the base before adaptation. The suite should include
at least a general reasoning benchmark (MMLU), a math benchmark (GSM8K), and an
instruction-following task (MT-Bench or similar). The requirement set in section 1
fixed a two-point regression budget; this gate enforces it.

**What it measures.** Whether the optimizer overwrote the broad knowledge and
reasoning the base already had.

**What it misses.** It only catches forgetting you measure. If you skip a
benchmark because you think the domain will not affect it, you will not see the
forgetting there. Run the full suite.

**Common mistake.** Running only the domain benchmark and reporting the gain.
Forgetting is silent inside the domain slice; it shows up only outside it.

## Needle-in-a-haystack (NIAH): the smoke test

Hide a single fact ("the needle") at a random depth in a long filler context
("the haystack") and ask the model to retrieve it. This is the minimum bar for a
long-context claim.

![Needle-in-a-haystack recall by depth and method](assets/fig-niah-recall-by-method.png)

*Recall at each depth in the context window, for four approaches. Naive
extrapolation (red) degrades sharply everywhere. Linear PI (orange) is better but
shows the "lost-in-the-middle" dip, visible in the shaded zone. YaRN (blue) and
staged extension in the Llama 3 style (green) maintain high recall across most
depths, though the mid-context gap never fully disappears. Illustrative.*

**What it measures.** Single-hop retrieval of one verbatim fact, and where in the
window recall breaks.

**What it misses.** Multi-hop reasoning, aggregation across the span, and
multi-needle retrieval. A model can pass NIAH and still miss facts at arbitrary
depth once more than one needle is involved, or once the task requires reasoning
rather than lookup.

**The lost-in-the-middle problem.** Recall is not uniform across the window.
Models attend best to the beginning and end and worst to the middle, so a fact at
50 percent depth is the hardest to retrieve. Report recall as a function of depth,
not a single averaged number. Any long-context claim without a recall-by-depth
plot is hiding the distribution.

## RULER: the real long-context gate

NVIDIA's RULER extends NIAH into categories that actually stress long context:

- **Multiple needles.** Several facts hidden in the haystack; the model must
  retrieve all of them.
- **Multi-hop variable tracing.** Variable chains where the model must follow
  references across the window to find the final value.
- **Aggregation.** Counting, listing, or summarizing entities distributed across
  the span.
- **Long-context question answering.** QA that requires reading and integrating
  multiple passages across the full length.

RULER's finding is blunt: most models that claim 32K or more degrade sharply well
before their advertised length. The effective context is routinely far shorter
than the configured one. Passing NIAH and declaring 128K is the most common weak-
eval mistake in long-context work.

**What it measures.** Whether the model can actually reason across the window, not
just find one thing.

**What it misses.** Open-domain generalization beyond the synthetic tasks. RULER
uses controlled synthetic data, so a model that fits the RULER distribution might
still fail on real long-document tasks. Gate on RULER first; follow up with real
task evals before production launch.

## Perplexity on long documents: a continuous training signal only

Long-context perplexity on held-out long documents is cheap to compute during
training and gives a useful continuous signal for detecting obvious failures. But
it saturates: a model can have fine long-context perplexity and still fail RULER,
because next-token loss is dominated by local prediction (predicting the next word
given the previous sentence is easy and cheap). Perplexity measures fluency at
length; it does not measure use of the context.

Gate training on perplexity for early stopping and stability. Gate promotion to
post-training on RULER-style retrieval and aggregation tasks and a recall-by-depth
measurement.

## When to use which

| Reach for | When | Instead of |
|---|---|---|
| Full general-benchmark suite (MMLU, GSM8K, MT-Bench) | After every DAPT run, before promoting the adapted base | Skipping it and asserting no forgetting; forgetting is silent in the domain slice |
| NIAH with a recall-by-depth plot | As the minimum smoke test for any long-context claim | Averaging recall over depth and hiding the mid-context dip |
| RULER multi-hop and aggregation | Before declaring an effective context length; only RULER separates real from configured | Passing NIAH only, which is single-hop and edge-anchored |
| Long-context perplexity | As a continuous training signal; cheap and useful for early stopping | As the primary gate for promotion; it saturates and does not catch broken retrieval |
| Domain benchmark (held-out domain data) | To measure what DAPT gained | As the only post-DAPT eval; always pair it with the general regression gate |
