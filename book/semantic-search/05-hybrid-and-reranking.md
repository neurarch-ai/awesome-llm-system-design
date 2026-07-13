# 5. Hybrid search and reranking

Dense retrieval is not enough by itself. This section covers why, how to fuse
dense and lexical signals, and when to add a cross-encoder reranker on top.

## Why pure dense retrieval has blind spots

A bi-encoder maps text into a continuous vector space. The dot product between
query and document vectors measures semantic similarity as learned from the
training corpus. This is powerful for paraphrase matching and intent retrieval
but has two failure modes.

**Exact-token misses.** Queries containing rare tokens (product SKUs, error
codes, names, version strings, technical identifiers) may retrieve semantically
related documents that do not contain the exact token. A query for
"OOM-killer exit code 137" might retrieve a paragraph about memory management
that never uses the string "137." A user searching for that code wants the exact
match. Lexical search handles this without a model: BM25 scores a document by
weighted term overlap with the query, so "137" in the document scores high for
the query "137."

**Out-of-domain terms.** Tokens the encoder never saw in training (new product
names, internal jargon, foreign-language rare words) have no meaningful
representation in the embedding space. BM25 or SPLADE treats them as ordinary
tokens and still retrieves exact matches.

Hybrid search, running dense and lexical retrieval in parallel and fusing the
results, reliably beats either channel alone across query mixes that include both
natural-language questions and exact-token lookups.

![Dense vs hybrid recall at k](assets/fig-dense-vs-hybrid.png)

*Hybrid retrieval (dense plus BM25, fused with RRF) lifts recall at every k
compared to dense alone. The gap is largest at small k because exact-term
queries land near the top in the lexical channel. Illustrative; the gap depends
on the fraction of exact-token queries in your mix.*

## Lexical retrieval options

**BM25 (probabilistic term weighting).** The standard inverted-index baseline.
Fast, no model required, exact token matches are perfect. IDF downweights common
terms and TF rewards dense mention. The baseline to beat in any hybrid system;
all major search engines expose it natively.

**SPLADE (sparse learned model).** SPLADE is a neural model that produces
sparse weight vectors over the vocabulary, similar in shape to BM25 but learned
to expand queries and documents with related terms. "memory error" in a query
might expand to also cover "OOM," "segfault," "swap," giving retrieval that
blends the term precision of BM25 with semantic expansion. SPLADE vectors are
stored as inverted lists in an existing search engine (Fare uses it on top of
Elasticsearch). The cost is that SPLADE expands posting lists, increasing index
size, and it requires running the SPLADE model at index time (write path) and at
query time.

## Fusing dense and lexical results

The two channels return separate ranked lists. The simplest and most robust
fusion is **reciprocal rank fusion (RRF)**:

$$\text{RRF-score}(d) = \sum_{r \in \text{channels}} \frac{1}{k_0 + \text{rank}_r(d)}$$

where $k_0$ is a small constant (typically 60). A document that ranks 1st in
both channels gets a score of $2 / (60 + 1)$; one that ranks 10th in one and
50th in the other gets a lower combined score. RRF is robust to score scale
differences between channels (dense cosine similarity and BM25 BM25 scores are
not on the same scale), requires no tuning of mixing weights, and has shown
consistent gains in retrieval benchmarks.

Alternatively, linearly interpolate the normalized scores from each channel with
a tuned mixing weight alpha:

$$s(d) = \alpha \cdot s_{\text{dense}}(d) + (1 - \alpha) \cdot s_{\text{BM25}}(d)$$

This gives more control over the mix but requires calibration and normalization
of scores across channels, and alpha may need tuning per query class.

## Cross-encoder reranking

After the fused retrieval returns a shortlist of, say, 100 candidates, a
cross-encoder can reorder them. A cross-encoder reads the query and document
together in one forward pass and outputs a relevance score. Because it models
interactions between query and document tokens (attention across both), it is
significantly more accurate than a bi-encoder for fine-grained relevance
judgments. The cost is thousands of times higher per pair: a cross-encoder
cannot be used over the full corpus, only over the shortlist.

![Cross-encoder rerank precision lift over bi-encoder](assets/fig-rerank-lift.png)

*A cross-encoder reranker applied to the top-100 shortlist consistently lifts
precision at ranks 1, 3, and 5 and NDCG@10 over the bi-encoder alone.
Illustrative; gains depend on query mix and model family.*

Cross-encoder reranking is optional and should be gated on latency budget.
The decision tree: if the surface shows results directly to a human who cares
about the order of the top 3 results, add a reranker. If the shortlist feeds a
downstream model that will re-score anyway, skip it.

## When to use which

| Reach for | When | Instead of |
|---|---|---|
| Dense (ANN) only | The corpus is clean, on-domain text with no exact-term queries; queries are always natural language | Hybrid when your query mix never includes exact tokens or rare identifiers |
| Hybrid dense + BM25 (RRF) | Queries include a mix of natural language and exact terms (SKUs, codes, names); this is the expected default at senior level | Dense alone when exact-term recall matters |
| Dense + SPLADE | You want semantic query expansion with lexical interpretability and have an existing Elasticsearch or OpenSearch cluster | A second dense model when the problem is term-mismatch, not semantic gap |
| RRF fusion | Mixing channels with incompatible score scales (always true for dense vs BM25) | Linear interpolation that requires careful score normalization and per-class tuning |
| Cross-encoder reranker | Top-k order matters to a human reader; budget allows 10-30ms extra; downstream model does not re-score | Bi-encoder for final ordering when the downstream stage will rank anyway |
| Skip reranking | The shortlist feeds a downstream ranker; total latency budget is tight | Adding a reranker that duplicates work already done downstream |
