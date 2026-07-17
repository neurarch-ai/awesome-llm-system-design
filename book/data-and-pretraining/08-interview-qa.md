# 8. Interview Q&A

The questions an interviewer actually asks about data curation and pretraining,
grouped by how they are used. The tricky ones and the commonly-missed ones are
where interviews are won or lost.

## Commonly asked

**Q: Should we pretrain from scratch at all?**
A: Almost never, outside a research lab. The default answer is to continue-pretrain
an open base (Llama 3, OLMo, Mistral). Pretraining from scratch is justified only
when: the target language has no adequate open base, the modality is new, the
tokenizer must differ from all open ones, or a capability genuinely absent from
every open model is required. State this pushback before designing the pipeline;
it is the first thing a senior candidate says.

**Q: How much of raw Common Crawl survives the pipeline?**
A: A small fraction, often single-digit percentages of the raw bytes, after
extraction, language filtering, quality filtering, and deduplication. This is
correct, not a bug. Raw Common Crawl is mostly boilerplate, spam, and
near-duplicates. Training on the dirty majority gives a worse model than training
on the clean minority. The keep rate being brutal is the point.

**Q: How do you size the model and the token budget?**
A: From the compute budget via the Chinchilla scaling law. Training FLOPs
approximate to $C \approx 6 N D$, and minimizing loss subject to fixed $C$ gives
$D^{\ast} \approx 20 N^{\ast}$. A 7B model at compute-optimal wants roughly 140B
tokens. But if you will serve the model at scale, deployment-optimal is a smaller
model overtrained far past 20 tokens per parameter, so inference stays cheap
forever. State which cost you are minimizing before quoting any ratio.

**Q: How do you prevent evaluating on training data?**
A: Decontamination. Remove training documents that overlap the benchmark eval sets
by n-gram overlap (flag any document sharing a 13-gram or 50-token span with an
eval example) and report the rate you found and removed. Embedding overlap can
catch paraphrased contamination at extra cost. Lead with this unprompted; a sharp
interviewer probes leakage first.

**Q: Why use both exact dedup and MinHash fuzzy dedup?**
A: Exact dedup (hash set or suffix array) catches byte-identical documents cheaply
but misses the common case: two pages that differ by a timestamp, a header, or a
minor edit. MinHash with LSH banding catches near-duplicates without the quadratic
all-pairs cost. Both are needed: exact for the easy case, fuzzy for the pervasive
web case.

## Tricky (the follow-ups that separate people)

**Q: MinHash estimates Jaccard similarity. Why do you also need LSH, and what
do $b$ and $r$ actually control?**
A: MinHash gives a cheap similarity estimate between any two documents, but you
still cannot afford all-pairs comparison at trillions of tokens. LSH banding
($b$ bands of $r$ rows each) turns the MinHash signature into a hash key so
only likely-similar documents ever get compared. The probability of being a
candidate pair is $1 - (1 - J^r)^b$: an S-curve in $J$. Raising $r$ demands
higher similarity before a pair is a candidate; raising $b$ catches more pairs
at a given similarity. You tune them to place the curve's knee at your target
duplicate threshold.

**Deeper:** The S-curve is an OR-of-ANDs: a pair becomes a candidate if it collides in at least one band (OR over the $b$ bands), and a band collides only if all $r$ of its rows match (AND within the band). So $r$ sharpens the threshold and $b$ lowers it, and the knee lands near $J \approx (1/b)^{1/r}$, which is the quantity you actually solve for when placing the curve at your duplicate cutoff.

**Q: You deduplicated harder and the downstream benchmark got worse. How?**
A: Over-dedup strips legitimately repeated text: canonical explanations, license
headers, quoted references, common phrasings that the model benefits from seeing.
It can also reduce the effective size of high-quality domains. Deduplication is
not monotonic in quality. FineWeb found that per-dump dedup plus a measured global
pass beat maximal global dedup. Ablate aggressiveness on downstream evals rather
than maximizing it.

**Deeper:** The mechanism is that repetition is a crude frequency signal the model rightly uses: a canonical explanation appearing across many pages is upweighted precisely because it is worth learning, so blanket dedup flattens that useful signal along with the spam. That is why FineWeb's per-dump-then-measured-global pass beats maximal dedup; you want to remove pathological duplication without erasing the natural frequency structure of good text.

**Q: Your multilingual model is strong in English and weak in Thai despite
plenty of Thai data. What do you check first?**
A: Tokenizer fertility. A vocabulary trained English-heavy fragments Thai into
many more tokens per word. The same content then costs more tokens, more compute,
and more of the context window. Check tokens-per-word per language; if Thai
fertility is high, the tokenizer is the bug, not the data volume, and fixing it
requires a retrain.

**Deeper:** High fertility hurts twice. Beyond the extra token and context cost, each word is split into subword fragments that individually carry little meaning and occur rarely, so their embeddings are undertrained: the model receives a noisier, sparser signal per concept, not merely a longer sequence. That is why more Thai data does not close the gap while the tokenizer stays English-heavy.

**Q: Chinchilla says 20 tokens per parameter, but Llama 3 8B trained on 15T
tokens (about 1800 tokens per parameter). Justify it.**
A: Chinchilla minimizes training compute for a target loss. A model you serve
billions of times has a second, larger cost: inference. Spending extra training
FLOPs to overtrain a smaller model lowers the per-token serving cost forever, so
the deployment-optimal size is smaller and more-trained than the training-optimal
one. Different objective, different optimum.

**Deeper:** Past the Chinchilla (DeepMind, 2022) point, loss keeps falling but with sharply diminishing returns per training FLOP, as the neural scaling laws (OpenAI, 2020) power law predicts, while per-token inference cost falls roughly linearly with parameter count. Overtraining pays that diminishing training return once to buy the linear inference saving on every future token, which is why it wins only when serving volume is large.

**Q: Your loss spiked at step 40K. Walk me through recovery.**
A: Roll back to the last good checkpoint before the spike. Identify and skip (or
reshuffle) the data batches around the spike, since a poisoned or pathological
batch is a common trigger. Optionally lower the learning rate or tighten gradient
clipping through the rough region, then resume. Confirm the data-loader resumes
at the right position so you neither repeat nor skip tokens. This is routine
tooling at scale, not an incident. Bake it into the training harness before the
run starts.

**Deeper:** A rollback-and-skip works because spikes usually come from a specific pathological batch interacting with a large adaptive-optimizer step, so resuming from just before it with that batch removed avoids re-triggering the divergence. Rolling back only the weights while keeping the optimizer state can re-diverge, because the second-moment estimates are already corrupted; the checkpoint has to restore optimizer state too.

**Q: Deduplication and decontamination look similar, both remove overlapping
text; when does the difference actually matter?**

A: Both scan the corpus for repeated spans and drop documents, often with the
same n-gram machinery, which is why pipelines sometimes treat one as covering
the other. But they compare against different references and protect different
things. Dedup compares training documents against each other and protects
model quality: it removes pathological repetition that wastes compute and
drives memorization. Decontamination compares training documents against the
evaluation sets and protects measurement: it removes text whose presence would
inflate benchmark scores without changing the model's real ability. The
difference matters at the design level because their tolerances point in
opposite directions. Dedup must be tuned, since over-removal strips useful
repetition and can hurt downstream quality; decontamination should be
near-exhaustive against a tiny fixed target, since any residual overlap can
only bias scores upward. A pipeline that "already dedups" has done nothing
about contamination: eval sets are not in the training corpus's comparison
set unless you explicitly put them there.

## Commonly answered wrong (the traps)

**Q: "We'll just train on all of Common Crawl."**
A: Raw crawl is mostly boilerplate, spam, and near-duplicates. Training on it
directly gives a worse model than training on the small clean fraction that
survives a proper pipeline. The keep rate is single digits for a reason.
Extraction, filtering, and dedup are the work, not a preprocessing footnote.
RefinedWeb's thesis is that cleaning web data hard enough alone can match curated
corpora: the investment is in the pipeline, not in adding more raw data.

**Deeper:** Near-duplicates do more than waste tokens: they bias the loss toward memorizing the repeated strings, which surfaces as verbatim regurgitation and as inflated benchmark scores wherever a duplicate of an eval example leaked into training. Cleaning is therefore an integrity control, not just a compute saver.

**Q: "Perplexity is the right metric for comparing our model to competitors'."**
A: Only if both models share a tokenizer. A model with a larger vocabulary emits
fewer tokens per sentence and flatters perplexity while being no better. Use
bits-per-byte (BPB), which normalizes by bytes and is tokenizer-invariant, when
comparing models with different vocabularies. Also confirm both eval sets are
decontaminated.

**Deeper:** Perplexity is a per-token average, so a coarser tokenizer that emits fewer tokens per sentence spreads the same total surprisal over fewer steps and reports a lower number with no real modeling gain. Bits-per-byte re-normalizes by raw bytes, a quantity no tokenizer choice can change, which is exactly why it is comparable across vocabularies.

**Q: "Decontamination is a nice-to-have."**
A: It is the integrity of every number you report. Skip it and your benchmarks
are fiction. A headline score without a decontamination claim is untrustworthy,
and a sharp interviewer probes this first. Lead with the decontamination claim,
not as a footnote.

**Deeper:** The failure is silent and one-directional: contamination can only inflate a score, never deflate it, so a clean-looking leaderboard gain is indistinguishable from leakage unless you measured and reported the overlap you removed. Absence of a decontamination claim is therefore itself evidence that the number is untrustworthy.

**Q: "Bigger vocabulary is strictly better because sequences get shorter."**
A: It also grows the embedding matrix and output softmax (parameters and compute
proportional to vocabulary size), undertrains rare tokens, and makes perplexity
incomparable across models. Vocabulary size is a fertility tradeoff tuned to the
language mix, not a free win.

**Deeper:** The output softmax is computed over the whole vocabulary at every position, so a larger vocab raises both the parameter count and the per-step output compute. Combined with rare tokens that each appear too seldom to train well, past some point the shorter-sequence saving no longer offsets the added width, which is why the optimum tracks the language mix rather than growing without bound.

**Q: "The bottleneck in pretraining is FLOPs, so buy faster GPUs."**
A: At scale the bottleneck is interconnect and memory bandwidth: tensor-parallel
all-reduces inside every layer, pipeline bubbles at stage boundaries, ZeRO/FSDP
gather traffic, and MoE all-to-all routing. A well-tuned frontier run achieves
30 to 50 percent MFU; the gap is communication overhead, not raw FLOPs.
Faster compute you cannot feed does nothing. The parallelism plan and the
network decide MFU.

**Deeper:** The tell is that MFU, not peak FLOP/s, is the number that moves: two clusters with identical rated FLOPs can differ substantially in real throughput purely on interconnect, because the tensor-parallel all-reduces sit on the critical path inside every layer and cannot be hidden behind compute. The ZeRO/FSDP gather traffic and MoE all-to-all routing add to that same communication-bound ceiling.

**Q: "Pretraining is just a big `.fit()` call."**
A: It is a distributed-systems problem. You need multi-axis parallelism to fit
and feed a model that does not fit on one GPU, MFU management to not waste the
cluster, and checkpointing plus elastic restart plus loss-spike recovery to
survive weeks on hardware that fails every few hours. The objective is one line;
the systems are the job.

**Deeper:** The next-token cross-entropy loss really is a handful of lines, but fitting a model that overflows one GPU is what forces multi-axis parallelism, and surviving weeks on failing hardware is what forces checkpoint, elastic restart, and loss-spike recovery. The engineering is dominated by the constraints around the loss, not by the loss itself, which is why `.fit()` framing badly understates the work.
