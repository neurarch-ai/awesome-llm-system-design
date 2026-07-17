# 8. Interview Q&A

The questions that actually come up in semantic search and embedding service
design interviews, grouped by how interviewers use them. The
commonly-answered-wrong group is where interviews are won or lost.

## Commonly asked

**Q: Why not just compute exact nearest neighbors at query time?**
A: At 100 million vectors and 1 microsecond per distance computation, one query
takes 100 seconds. ANN search trades a small, controllable recall loss for a
speedup of 10 to 100x. The recall loss is recovered by the reranking stage for
the few top results that matter for precision. The choice is not "exact vs
approximate" in principle; it is that exact search is physically too slow at
this scale.

**Q: Why run a lexical channel alongside the dense one?**
A: Dense embeddings miss exact-token queries: product SKUs, error codes,
version strings, names, or any token rare in the training corpus may have a
poor representation in the embedding space. BM25 handles these natively and
costs nothing to run in parallel. Fusing the two channels with reciprocal rank
fusion reliably beats either alone across mixed query types. This is the
expected answer; omitting it signals unfamiliarity with production search.

**Q: How do you evaluate the retrieval stage?**
A: Measure recall@k against a labeled query-item set, at the k you actually
pass downstream (say, 100 or 500), not at k=5 or k=10. Use a time-based split
(hold out queries from the future, evaluate whether today's index retrieves the
ground-truth documents) to avoid leaking future knowledge. Gate a launch on an
online A/B measuring end-to-end task success, because retrieval recall and
downstream quality do not always move together.

**Q: How does a new document become searchable?**
A: The write path publishes the document to a queue; an embedding worker encodes
it (using the same encoder as the rest of the index); the vector is upserted
into the ANN index and the text is added to the lexical index. Both writes must
land within the freshness SLA. HNSW supports incremental inserts without a
rebuild; IVF-PQ appends to the nearest centroid's posting list. The cold-start
case requires no special treatment because the encoder works on content features
alone, with no per-document trained component.

**Q: What breaks when you upgrade the embedding model?**
A: Every vector in the index must be re-embedded. Old and new vectors live in
different spaces; mixing them in one index silently destroys recall. The safe
procedure is: build the new index alongside the old, validate recall on a
labeled set, dual-read for a period, then cut over. This is a full re-index
event, not a rolling deploy.

## Tricky (the follow-ups that separate candidates)

**Q: Recall@k improved offline but end-to-end quality dropped online. What are
the possible causes?**
A: Three main ones. First, the recall gain came from surfacing popular items
more (harder to miss) while coverage of the long tail shrank; the user-visible
feed became less diverse. Second, the retrieval improvement was swamped by a
quality drop in the reranker because the retrieved set distribution changed.
Third, the query split for offline eval was not time-based, so the improvement
was from leakage (memorizing future interactions), not generalization. Diagnose
by measuring coverage and tail-item retrievability alongside recall.
**Why the aggregate metric hides the first cause:** recall@k is an average over
the labeled query set, and labeled queries skew toward head content; a model can
therefore win the average by improving on popular items while losing the tail,
so the number goes up while the experience of exactly the queries not in the
label set goes down.

**Q: A team says "just use a bigger embedding model for better recall." Is that
right?**
A: Rarely the first move. Larger dimension increases index RAM linearly and
search latency near-linearly, so the cost multiplies across 100M vectors. Before
spending on a larger model, check whether the recall gap is from a domain
mismatch the lexical channel could close, or from missing hybrid search. Dropbox
found a large MRR gap between models via MTEB, suggesting model fit matters, but
they also constrained dimension for the 4KB storage cap. Measure the gap with
and without the lexical channel first; if hybrid search closes it, a bigger
encoder is unnecessary.
**Why size does not fix a domain gap:** scaling improves the quality of
representations for content the training distribution covers; a token or concept
absent from that distribution has no grounded representation at any parameter
count, so the failure is about what the model saw, not how big it is, and the
lexical channel fixes identity matching that no amount of capacity learns.

**Q: HNSW is popular; when would you not use it?**
A: Two cases. First, when the corpus does not fit in RAM: HNSW stores the full
graph plus vectors, and at 100M float32 vectors at dim 768 that is roughly
600GB, which is expensive. Switch to IVF-PQ for compression, or DiskANN for SSD
storage. Second, when you need very aggressive filtering: HNSW graph traversal
does not naturally skip filtered nodes efficiently. An IVF or partitioned index
can restrict search to the relevant partition before scanning, which is far
cheaper for highly selective filters.
**Why filtering fights the graph:** HNSW reaches the target region by walking
edges through intermediate nodes, and many of those stepping-stones fail the
filter; skip them and the path to the answer can disconnect, keep them and the
search wastes its budget visiting nodes it can never return. A partitioned index
avoids the dilemma by shrinking the search space before any traversal begins.

**Q: HNSW's ef and IVF's nprobe look similar (both widen the search to buy
recall); when does the difference actually matter?**
A: Both are the recall-versus-latency knob for their index, and on a uniform
corpus tuning either looks the same from the outside. The mechanics differ: ef
widens a best-first beam over the graph, so the extra work is adaptive and stays
concentrated in the immediate neighborhood of the query; nprobe adds whole
precomputed cells, so the extra work comes in coarse, fixed-size increments and
is spent scanning everything in each added cell whether promising or not. The
difference matters when the query lands near a cell boundary (its true neighbors
are split across cells, so recall climbs in steps only as whole cells are added,
while ef's frontier crosses the boundary naturally) and when cluster sizes are
skewed, where one added cell can double the scan cost. It also shapes filtered
search: restricting cells is easy for IVF, while restricting graph traversal is
the case where HNSW struggles.

**Q: What is the difference between MIPS and Euclidean nearest neighbor, and
why does it matter for quantization?**
A: MIPS (maximum inner product search) ranks documents by their dot product with
the query; Euclidean NN ranks by distance. They are equivalent only when vectors
are unit-normalized. For two-tower dot-product retrieval, the relevant metric is
inner product, and standard PQ quantization (which minimizes average
reconstruction error) is tuned to Euclidean distance. ScaNN's key insight is
that for MIPS, error parallel to the query vector (which directly reduces the
inner product) should be penalized more than orthogonal error. Using a
Euclidean-tuned quantizer for inner-product search silently loses recall on the
items with the highest inner products, which are exactly the ones that rank first.

**Q: When would you reach for ColBERT late-interaction over a cross-encoder
reranker?**
A: A cross-encoder (descended from Sentence-BERT, UKP Darmstadt, 2019) concatenates
query and document and runs a full transformer forward pass per pair, so it is
accurate but cannot be precomputed: every candidate costs a fresh pass at query time.
ColBERT (Stanford, 2020) embeds each document token independently and offline, then
at query time scores via MaxSim: for each query token it takes the max dot product
over the document's token embeddings and sums those maxima. Because the document
token embeddings are precomputed and indexed, ColBERT scales to first-stage retrieval
or cheap reranking over large candidate sets, trading some accuracy for the ability
to amortize document encoding. Reach for it when you need better-than-bi-encoder
quality at a candidate volume a cross-encoder cannot afford; keep the cross-encoder
for the small final shortlist where its full cross-attention pays off.

MaxSim in code makes the "sum of per-query-token best matches" concrete:

```python
import numpy as np

def maxsim(query_tok, doc_tok):              # query_tok, doc_tok: arrays of shape (n_tokens, dim)
    sims = query_tok @ doc_tok.T             # every query-token vs every doc-token dot product
    return sims.max(axis=1).sum()            # for each query token take its best doc match, then sum
# maxsim(np.eye(2), np.array([[1.,0.],[0.,1.],[0.5,0.5]])) -> 2.0
```

## Commonly answered wrong (the traps)

**Q: Should you apply a score correction for compression bias at query time?**
A: The wrong answer is "yes, subtract a correction term from the ANN scores at
serving." Compression artifacts (PQ codes, int8 quantization) make first-phase
scores approximate, and the fix is a **full-precision rescore** of the shortlist
by paging the original vectors back and recomputing exact scores. There is no
magic correction term to apply to compressed scores; you need the real vectors.
The rescore is cheap because it runs over only the shortlist (say 200
candidates), not the whole corpus.
**Why no correction term can exist:** quantization error is per-vector and
direction-dependent, so two documents with identical compressed scores can carry
true errors of opposite sign; any scalar or per-query correction shifts both
scores the same way and cannot swap their order. Recovering the true ranking
requires the information the codes discarded, which only the original vectors
still hold.

**Q: Can you add term-level features from BM25 into the embedding model so you
need only one retrieval channel?**
A: Not cleanly. Embeddings capture distributional semantics; BM25 exact-match is
a different signal. Concatenating BM25 features as input tokens does not
replicate the recall behavior of a lexical inverted index for rare tokens. The
production answer is two channels that run in parallel and fuse: dense for
semantics, lexical for exact match. Trying to learn both from one embedding
model produces a model that does neither as well.
**Why one vector cannot carry both signals:** an embedding compresses text into
a fixed number of dimensions, while exact-match recall requires distinguishing
the identity of every rare token, an amount of information that grows with the
vocabulary; the compression must discard precisely the rare-token identity that
lexical matching depends on, whereas an inverted index stores that identity
exactly by construction.

**Q: The corpus is 100M but only 1M documents are in the "active" category.
Should you post-filter the ANN results?**
A: No, or at least not naively. Post-filtering ANN results works if the filter
passes a large fraction of documents (say 50%), because ANN over 100M usually
returns at least k documents that pass. If the filter passes only 1% (1M out of
100M), ANN over the full corpus may return fewer than k matching documents in
its top candidates, and recall craters. The fix is to partition the index by
category and route filtered queries to the 1M-document sub-index directly,
or use an IVF index that restricts which clusters are searched based on the
filter before scanning.
**Why the arithmetic is brutal at low pass rates:** the ANN search ranks by
similarity with no knowledge of the filter, so passers appear in the candidate
list at roughly the filter's base rate; at a 1% pass rate a few hundred
candidates contain only a handful of passers in expectation, and asking the
index for enough candidates to guarantee k of them multiplies latency by the
inverse of the pass rate.

**Q: Embeddings capture meaning, so isn't hybrid search just a band-aid for
weak embedding models?**
A: No. Even a perfect embedding model trained on all human text cannot recall
a query for a product code that was minted after training. Any token the model
has not seen maps to a vector in a neighborhood that reflects its rare occurrence
statistics, not its actual identity. Lexical search is structurally complementary
to semantic search, not a substitute for a weaker model.
**Why the unseen token lands in the wrong place:** the tokenizer falls back to
subword pieces, so a new product code is embedded near other strings that share
its character fragments; that is spelling similarity masquerading as meaning,
and no training objective on past data can assign the code its true identity.

**Q: Adding more shards to the ANN index improves recall, right?**
A: No; it usually costs recall unless you compensate. Sharding splits the corpus
across N partitions and each shard returns its local top-k, which are then merged.
HNSW (Malkov and Yashunin, 2016) recall is per-shard: a query that would have found
its true neighbor by traversing the full graph may now land in a shard whose local
graph does not contain that neighbor, and the merge cannot recover a document no
shard surfaced. Sharding buys throughput and RAM headroom, not recall; to hold recall
you must raise per-shard ef or over-fetch (request more than k per shard) so the merge
has enough candidates to work with. Treating shard count as a recall knob inverts the
actual trade.
