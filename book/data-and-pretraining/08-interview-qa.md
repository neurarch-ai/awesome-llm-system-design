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

**Q: You deduplicated harder and the downstream benchmark got worse. How?**
A: Over-dedup strips legitimately repeated text: canonical explanations, license
headers, quoted references, common phrasings that the model benefits from seeing.
It can also reduce the effective size of high-quality domains. Deduplication is
not monotonic in quality. FineWeb found that per-dump dedup plus a measured global
pass beat maximal global dedup. Ablate aggressiveness on downstream evals rather
than maximizing it.

**Q: Your multilingual model is strong in English and weak in Thai despite
plenty of Thai data. What do you check first?**
A: Tokenizer fertility. A vocabulary trained English-heavy fragments Thai into
many more tokens per word. The same content then costs more tokens, more compute,
and more of the context window. Check tokens-per-word per language; if Thai
fertility is high, the tokenizer is the bug, not the data volume, and fixing it
requires a retrain.

**Q: Chinchilla says 20 tokens per parameter, but Llama 3 8B trained on 15T
tokens (about 1800 tokens per parameter). Justify it.**
A: Chinchilla minimizes training compute for a target loss. A model you serve
billions of times has a second, larger cost: inference. Spending extra training
FLOPs to overtrain a smaller model lowers the per-token serving cost forever, so
the deployment-optimal size is smaller and more-trained than the training-optimal
one. Different objective, different optimum.

**Q: Your loss spiked at step 40K. Walk me through recovery.**
A: Roll back to the last good checkpoint before the spike. Identify and skip (or
reshuffle) the data batches around the spike, since a poisoned or pathological
batch is a common trigger. Optionally lower the learning rate or tighten gradient
clipping through the rough region, then resume. Confirm the data-loader resumes
at the right position so you neither repeat nor skip tokens. This is routine
tooling at scale, not an incident. Bake it into the training harness before the
run starts.

## Commonly answered wrong (the traps)

**Q: "We'll just train on all of Common Crawl."**
A: Raw crawl is mostly boilerplate, spam, and near-duplicates. Training on it
directly gives a worse model than training on the small clean fraction that
survives a proper pipeline. The keep rate is single digits for a reason.
Extraction, filtering, and dedup are the work, not a preprocessing footnote.
RefinedWeb's thesis is that cleaning web data hard enough alone can match curated
corpora: the investment is in the pipeline, not in adding more raw data.

**Q: "Perplexity is the right metric for comparing our model to competitors'."**
A: Only if both models share a tokenizer. A model with a larger vocabulary emits
fewer tokens per sentence and flatters perplexity while being no better. Use
bits-per-byte (BPB), which normalizes by bytes and is tokenizer-invariant, when
comparing models with different vocabularies. Also confirm both eval sets are
decontaminated.

**Q: "Decontamination is a nice-to-have."**
A: It is the integrity of every number you report. Skip it and your benchmarks
are fiction. A headline score without a decontamination claim is untrustworthy,
and a sharp interviewer probes this first. Lead with the decontamination claim,
not as a footnote.

**Q: "Bigger vocabulary is strictly better because sequences get shorter."**
A: It also grows the embedding matrix and output softmax (parameters and compute
proportional to vocabulary size), undertrains rare tokens, and makes perplexity
incomparable across models. Vocabulary size is a fertility tradeoff tuned to the
language mix, not a free win.

**Q: "The bottleneck in pretraining is FLOPs, so buy faster GPUs."**
A: At scale the bottleneck is interconnect and memory bandwidth: tensor-parallel
all-reduces inside every layer, pipeline bubbles at stage boundaries, ZeRO/FSDP
gather traffic, and MoE all-to-all routing. A well-tuned frontier run achieves
30 to 50 percent MFU; the gap is communication overhead, not raw FLOPs.
Faster compute you cannot feed does nothing. The parallelism plan and the
network decide MFU.

**Q: "Pretraining is just a big `.fit()` call."**
A: It is a distributed-systems problem. You need multi-axis parallelism to fit
and feed a model that does not fit on one GPU, MFU management to not waste the
cluster, and checkpointing plus elastic restart plus loss-spike recovery to
survive weeks on hardware that fails every few hours. The objective is one line;
the systems are the job.
