# 3. Data quality

Quality filtering and deduplication are the two highest-leverage steps in the
entire pipeline. This section covers how each works, when to use which, and the
failure modes that hurt quietly.

## Heuristic quality filters

Heuristic filters are cheap, interpretable rules applied per document. The
canonical sources are the Gopher rules and C4's filtering recipe:

- **Length bounds.** Documents shorter than a word-count floor are likely
  truncated or empty; extremely long ones may be data dumps.
- **Mean word length.** Very short mean word length signals symbol spam or
  encoding artifacts; very long signals concatenated tokens.
- **Symbol-to-word ratio.** A high ratio of special characters to words signals
  boilerplate, markup leftovers, or code in a prose corpus.
- **Fraction of lines ending in punctuation.** Prose usually does; bullet lists
  and navigation do not.
- **Fraction of duplicate lines or paragraphs within a document.** Repeated
  blocks signal templated or generated filler.
- **Stop-word presence.** Documents with no common stop words in the target
  language may be garbled or in the wrong language.
- **C4 rules.** Drop pages without terminal punctuation, drop "lorem ipsum," drop
  pages with curly braces (removes code from a prose corpus).

The key insight from FineWeb's ablation study: the right set of heuristic rules
is small. Out of fifty-plus plausible candidates, only a handful actually move
downstream benchmark scores when ablated. Stacking every reasonable-looking rule
does not compound the benefit and can remove valid text. Choose by ablating on
downstream evals, not by intuition.

## Learned quality classifiers

A learned classifier scores a document by its resemblance to a known-good
reference corpus.

**Perplexity-based (CCNet style).** Train a small language model on a trusted
reference (Wikipedia or books) for each target language, then keep web documents
that score low perplexity (a measure of how surprised the model is by the text; low means it looks familiar) under that model. Low perplexity means the text looks
like the reference, which is a cheap proxy for being well-formed and informative.
CCNet uses this per language, which is what makes multilingual quality filtering
tractable.

**FineWeb-Edu style.** A strong LLM labels a sample of documents by educational
quality (score 0 to 5). A fast classifier (a DistilBERT-scale model) is trained
on those labels, then applied at scale to the full corpus. Keeping only the
top-scoring documents (FineWeb-Edu keeps about 1.3T out of 15T tokens) sharply
lifts knowledge and reasoning benchmarks like MMLU and ARC. Fewer, better tokens
beat more tokens.

The tradeoff: classifiers are powerful but bake in the biases of their reference.
Filter too hard toward "educational" and you may lose valid registers of text
(forums, fiction, conversational text) that are valuable for other capabilities.
Always validate the classifier's effect on downstream evals, not just on how
clean the filtered samples look.

## Deduplication: exact and fuzzy

Duplicates hurt in three concrete ways: they waste training compute on repeated
tokens, they drive memorization (a document seen many times is more likely to be
regurgitated verbatim), and they are the main vector for eval leakage.

**Exact deduplication** removes byte-identical or hash-identical documents. It is
cheap (a hash set or a suffix-array pass) and catches the easy cases. It is not
sufficient alone because most web near-duplicates differ by a timestamp, header,
or minor edit.

**Fuzzy (near-duplicate) deduplication** uses MinHash (compact per-document fingerprints) with LSH (locality-sensitive hashing, which buckets similar fingerprints together). The idea in
three steps:

Represent each document as a set of $n$-gram shingles (overlapping slices of $n$ consecutive words or characters). The overlap between two
documents is their Jaccard similarity (the overlap of two sets, from 0 to 1):

$$J(A, B) = \frac{\lvert A \cap B \rvert}{\lvert A \cup B \rvert}$$

```python
def jaccard(a, b):                     # a, b: sets of n-gram shingles
    if not (a or b):                   # two empty docs: define overlap as 0
        return 0.0
    return len(a & b) / len(a | b)     # shared shingles over total distinct shingles
# e.g. jaccard({1, 2, 3}, {2, 3, 4}) -> 0.5
```

A **MinHash** signature approximates $J$ cheaply. Apply $k$ independent hash
functions to the shingle set and keep the minimum under each. The probability
that two documents share a given min-hash entry equals their Jaccard similarity,
so the fraction of matching entries estimates $J$ without comparing the full
shingle sets:

$$\Pr[\min h(A) = \min h(B)] = J(A, B)$$

```python
def minhash_estimate(a, b, seeds):     # a, b: shingle sets; seeds: k independent hash multipliers
    P = 2_147_483_647                  # a large prime modulus for the hash x -> (m * x) % P
    agree = 0
    for m in seeds:                    # one min-hash entry per seed
        if min((m * s) % P for s in a) == min((m * s) % P for s in b):
            agree += 1                 # this min-hash entry matches between the two docs
    return agree / len(seeds)          # matching fraction estimates Jaccard J(A, B)
# identical sets agree on every entry: minhash_estimate({1, 2, 3}, {1, 2, 3}, [7, 11, 13]) -> 1.0
```

**LSH banding** turns the estimate into a scalable candidate search. Split the
$k$-entry signature into $b$ bands of $r$ rows each. Two documents become a
candidate pair if they match on at least one complete band. The probability of
being a candidate is a tunable S-curve in $J$:

$$\Pr[\text{candidate}] = 1 - \left(1 - J^{r}\right)^{b}$$

```python
def lsh_candidate_prob(J, b, r):     # J: Jaccard sim; b bands of r rows each (b*r = signature length k)
    return 1 - (1 - J ** r) ** b     # P(match on >=1 full band); S-curve knee near J = (1/b)**(1/r)
# lsh_candidate_prob(0.8, b=20, r=5) -> ~0.9996  (near-duplicates almost always become candidates)
```

Raising $r$ demands higher similarity before a pair is a candidate; raising $b$
catches more pairs at a given similarity. You place the S-curve's knee at your
target duplicate threshold.

FineWeb applies MinHash dedup both within and across all 96 Common Crawl dumps.
The across-dump pass matters as much as the within-dump pass: the same page
recurs across snapshots at high rate.

**Dedup is not monotonic in quality.** FineWeb found that per-dump dedup plus a
measured global pass beat naive maximal global dedup: over-removing strips
legitimately repeated text (canonical explanations, common phrasings, quoted
sources) and can lower downstream scores. Ablate aggressiveness on evals rather
than maximizing it.

## Decontamination

Decontamination removes training documents that overlap the benchmark eval sets.
It is the integrity step, not a nice-to-have. Any headline benchmark number
without a decontamination claim is suspect.

**n-gram overlap.** Flag and drop training documents that share a long exact
n-gram (typically a 13-gram or 50-token span) with any eval example. Cheap and
standard; catches direct contamination.

**Embedding overlap.** Catch paraphrased contamination that exact n-grams miss,
at higher compute cost. Used in addition to n-gram overlap, not instead of it.

**Report the rate.** The mature move is to publish the contamination rate found
and removed. "Our eval is decontaminated against the training set by n-gram
overlap, and we found and removed X% of the training documents" is a senior
signal. Waiting to be asked is a junior one.

Lead with decontamination unprompted. A sharp interviewer probes this first.

## When to use which

| Reach for | When | Instead of |
|---|---|---|
| Heuristic filters (Gopher / C4 rules) | You want cheap, interpretable rules with a clear audit trail; validate by ablating a small set on downstream evals | Stacking fifty rules without ablation, which adds noise without compounding benefit |
| Learned quality classifier (FineWeb-Edu style) | You have a known-good reference and want fewer, higher-value tokens that lift MMLU and ARC | Heuristics alone, when the reference is rich enough to train a useful classifier |
| Exact hash / suffix-array dedup | Always, as the cheap first pass that kills byte-identical documents before MinHash | Skipping it, since exact dedup is near-free and handles obvious cases |
| MinHash plus LSH fuzzy dedup | The corpus spans trillions of tokens with cross-dump near-duplicates you cannot compare all-pairs | Maximal global dedup, which over-removes valid common text; tune $b$ and $r$ and ablate |
| Decontamination by n-gram overlap | Always, before the first training token, with the overlap rate reported | Skipping it; embedding overlap is an additional layer that catches paraphrase at extra cost |
| Per-language pipeline (CCNet style) | The target is multilingual and you need per-language filters, thresholds, and quality references | A single global pipeline where high-resource languages set thresholds that starve others |

**Provenance.** These pipelines carry their origins inline: heuristic rules from Gopher and C4, the learned educational classifier from FineWeb-Edu, and per-language filtering from CCNet; the classifier and perplexity components lean on fastText (Meta) and KenLM as noted below. All are data-pipeline techniques rather than model-architecture methods, so no foundational-model attribution applies.

**Tools.** For heuristic filters and the full pipeline, datatrove (Hugging Face) ships the FineWeb Gopher and C4 rule implementations, and NeMo Curator (NVIDIA) offers the same filters on a distributed backend. Learned classifiers lean on fastText (Meta) for fast quality scoring and KenLM for the CCNet-style perplexity models trained per language on Wikipedia. Fuzzy dedup is served by datasketch for MinHash plus LSH, by text-dedup as a batteries-included wrapper, and again by datatrove and NeMo Curator at corpus scale. Exact dedup needs nothing more than a hash set or a suffix-array pass, and decontamination by n-gram overlap is a utility inside most eval harnesses.

**Worked example.** A pretraining team assembling a multilingual base from Common Crawl starts with exact hash dedup because it is near-free and kills byte-identical snapshots before anything expensive runs. They add MinHash plus LSH for near-duplicates rather than reaching for maximal global dedup, tuning the band and row counts and ablating aggressiveness on downstream evals so they do not strip legitimately repeated canonical text. Because the target is multilingual, they run a per-language CCNet-style pipeline with its own perplexity reference per language instead of a single global threshold that would let English set the bar and starve lower-resource scripts. They keep heuristic Gopher and C4 rules to a small ablated set, layer a learned educational classifier only where the reference corpus is rich enough to justify it, and run n-gram decontamination against every eval set before the first training token, publishing the removed overlap rate.
