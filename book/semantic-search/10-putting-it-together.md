# 10. Putting it together: the complete build

Sections 1 through 6 taught each stage with its options and tradeoffs; section 7
showed where real teams diverge. What none of them show is a single system with
every decision made. This capstone does three things: it gives you an opinionated
default stack so option paralysis never blocks a first build, it walks the
chapter's scenario end to end with every choice committed and sized, and it
shows how the same decisions flip when the constraints change. It closes with
the smallest runnable ANN index, one file, no installs.

## The default stack: start here, deviate with reason

Every stage in this chapter has three to six credible options, and a first-time
builder can burn a week comparing index libraries before serving a single query.
Skip that. The stack below is a sane default for a first production build; each
row names when to deviate and which section explains why. Libraries change
yearly, but the interface of each stage (embed, index, retrieve, fuse, rerank,
evaluate) does not, so pick per stage by interface and treat any specific
library as replaceable.

| Stage | Default | Deviate when | Why (section) |
|---|---|---|---|
| Embedding model | Small 384-dim bi-encoder (MiniLM-L6 class), one deployment for batch writes, one for latency-bound queries | Recall short of the bar with RAM to spare: BGE-large / E5-large; multilingual corpus: multilingual-e5 | [3](03-the-embedding-service.md) |
| Index | HNSW when the corpus fits in RAM; incremental inserts for free | Billion scale on a RAM budget: IVF-PQ; one commodity box: DiskANN; two-tower dot-product retrieval: ScaNN-style anisotropic PQ | [4](04-vector-index.md) |
| Quantization | 8-bit vectors plus a full-precision rescore of the shortlist | Corpus fits RAM comfortably at full precision: skip; extreme scale: PQ codes (64x at 24 bytes per vector) | [4](04-vector-index.md) |
| Hybrid fusion | Dense + BM25 in parallel, fused with RRF | Queries are pure natural language with no exact tokens (rare): dense-only; existing Elasticsearch and a term-mismatch problem: SPLADE | [5](05-hybrid-and-reranking.md) |
| Reranking | Cross-encoder over the fused top-100 when a human reads the results | Shortlist feeds a downstream ranker that re-scores anyway: skip it | [5](05-hybrid-and-reranking.md) |
| Filters | Pushed inside the index, never applied after | Filter passes under ~1% of documents: a per-partition sub-index routed directly | [6](06-serving-and-scaling.md) |
| Freshness | Durable queue, GPU batch embedding workers, incremental upsert into both indexes | Catalog is stable and daily staleness is acceptable: scheduled full rebuild is simpler | [6](06-serving-and-scaling.md), [3](03-the-embedding-service.md) |
| Evaluation | recall@k at the k passed downstream, measured against a flat-scan ceiling on a time-based split | Never. Build the labeled query set first | [4](04-vector-index.md), [8](08-interview-qa.md) |

The last row is the one beginners skip and regret: without a labeled query set
and a brute-force recall ceiling, every index knob is a vibe, and an `ef` set
too low will silently cap recall no matter how much you spend on the encoder.
One afternoon of labeling pays for itself the first time you tune `nprobe`.

## The complete build

Return to the scenario from [section 1](01-clarifying-requirements.md): 100
million text documents growing 10% per quarter, top-k search in under 50ms at
p99, high recall, attribute filters, a query mix of natural language and exact
codes, and freshness within minutes for inserts and deletes. Here is the whole
system with every choice committed and the reason it won.

| Decision | Choice | Why it won |
|---|---|---|
| Encoder | MiniLM-L6 class, 384-dim, separate write and read deployments | ~5ms CPU query inference fits the embed slot of a 50ms budget; 1024-dim would nearly triple index RAM |
| Query embedding | Cached; encoder inference only on miss | Repeated queries are common; a cache hit turns 4-8ms into under 1ms |
| Index | HNSW, sharded, incremental inserts | Corpus fits in RAM once quantized; the minutes freshness SLA rules out rebuild-only structures |
| Compression | 8-bit vectors (Voyager E4M3 class), full-precision rescore of the shortlist | 4x memory cut with the recall loss recovered by rescoring; never trust compressed scores as final |
| Retrieval | Dense ANN + BM25 in parallel, RRF fusion | The interviewer pinned exact-code queries; dense alone misses them, and RRF needs no score calibration |
| Reranking | Cross-encoder over the fused top-100, gated per surface | Human-facing surfaces need top-3 precision; surfaces feeding a downstream ranker skip it and save 10-30ms |
| Filters | Pushed inside the index; per-partition sub-index for highly selective values | A post-filter that discards 99% of candidates wastes the whole ANN budget |
| Freshness | Queue, GPU batch embedding, upsert into vector and lexical index together | Minutes SLA; the two channels must stay in sync or hybrid fusion degrades |
| Evaluation | recall@k at the downstream k vs a flat ceiling, time-based split, online A/B gate | Section 1 pinned that retrieval misses directly hurt the product, so recall is the first number watched |

**Index memory.** 100M vectors at 384 dimensions in float32 is about 153 GB of
raw vectors ([section 3](03-the-embedding-service.md)). HNSW adds graph edges at
`M * 8` bytes per vector: with M = 32 that is another ~26 GB, landing near the
~178 GB full-precision figure from [section 4](04-vector-index.md). Quantizing
vectors to 8 bits cuts the vector payload to ~38 GB, so the whole index is
roughly 64 GB: four shards of ~16 GB each, replicated for query throughput,
with the full-precision vectors kept off the hot path for shortlist rescoring.
The same corpus at 1024 dimensions would start at 409 GB raw, which is why the
encoder dimension was decided together with the index rather than on
leaderboard rank alone.

**Latency.** The component budget from [section 6](06-serving-and-scaling.md)
lands inside 50ms with headroom on the common path: ~1ms cached query embedding
(4-8ms on a miss), 2-15ms ANN search running in parallel with 2-8ms BM25 so
only the slower channel counts, ~1ms RRF fusion, 10-30ms cross-encoder rerank,
2-5ms network. Mid-range figures sum near 40ms p99 with the reranker on. The
reranker is the item that gets truncated or skipped first when the tail spikes;
everything else is either cached, parallel, or trivially fast.

**Cost.** The serving bill is memory rent, not per-query tokens: a ~64 GB
sharded index means a handful of RAM-heavy replicas plus GPU embedding workers
on the write path, and cost stays flat with query volume until a replica is
added. Illustrative: four shards times two replicas is eight index nodes. The
chapter's calibration point is Vespa's published billion-scale build, 1B int8
vectors at 90% recall@10 under 50ms for about \$6K per month
([section 7](07-how-teams-do-it-in-production.md)); a 100M-document service
sits well under that. The number worth internalizing is the counterfactual:
skipping quantization would nearly triple RAM per shard, and choosing a
1024-dim encoder would multiply the whole fleet, for a recall gain
[section 3](03-the-embedding-service.md) shows flattening.

**What breaks in month one.** Three failure modes dominate early operations, so
wire their signals before launch: tombstone accumulation (HNSW deletes leave
dead nodes wired into the graph, so under churn recall drifts down and latency
creeps up with no error; watch the tombstone fraction and budget the rebuild
from [section 4](04-vector-index.md)), per-filter-class recall (queries with
selective filters hit the low-pass-rate arithmetic from
[section 8](08-interview-qa.md) first, and those users will report "search is
broken" while global recall looks fine), and embedding cache misses (the p99
for novel long-tail queries carries the full 4-8ms encoder cost; a falling
cache hit rate is the early warning that the latency budget is about to leak).

## The same techniques under different constraints

The review question that matters in practice is not "which index is best" but
"which index is best under my constraints." Here is the same pipeline built
three times. Only the middle column is the build above; the other two keep the
identical stage interfaces and swap nearly every implementation choice. Corpus
sizes in the outer columns are Illustrative.

| | Internal wiki search | Catalog search (this chapter) | Marketplace retrieval at 1B items |
|---|---|---|---|
| Corpus / consumer | 500k documents; results read by humans | 100M documents; humans and downstream models, designed for the harder case | 1B+ items; feeds a downstream learned ranker only |
| Latency budget | A few hundred ms is fine | p99 < 50ms for the search call | Retrieval feeds a ranker with its own budget; throughput and RAM dominate |
| Encoder | Small 384-dim bi-encoder; re-embed the whole corpus in one batch job | 384-dim, cached query path, GPU batch write path | Matryoshka: short prefix dims for ANN, full vector for the ranker, one training run |
| Index | Default HNSW, one process, no sharding, full precision | Sharded 8-bit HNSW, incremental upsert | IVF-PQ (or ScaNN-style anisotropic PQ if two-tower MIPS), full-precision rescore |
| Hybrid / rerank | BM25 + RRF still mandatory (ticket IDs, jargon); cross-encoder fits the loose budget | Hybrid RRF; cross-encoder gated per surface | Lexical channel per query mix; no cross-encoder, the downstream ranker re-scores |
| Freshness | Re-embed on deploy; the corpus fits in one batch | Queue + incremental upsert, minutes SLA | Weekly batch build plus daily delta, Instacart / LinkedIn style |
| Eval | 100 labeled queries against a flat scan of the whole corpus | recall@k vs flat ceiling, time split, A/B gate | recall@500 at the k the ranker consumes, plus coverage of tail items |
| What would be over-engineering | Sharding, quantization, a rebuild scheduler | DiskANN, PQ codes | A cross-encoder, per-query-class rerank gating, sub-50ms tuning |

Two lessons fall out. First, the wiki column is mostly deletions: at 500k
vectors a single full-precision HNSW process is exact enough, fits anywhere,
and removes sharding, quantization, and rebuild scheduling as tuning surfaces,
but the lexical channel survives every deletion because exact-token queries are
a property of users, not of scale. Second, the marketplace column shows what
changes when a downstream ranker owns final ordering: the cross-encoder
disappears, recall is measured at the large k the ranker consumes, and the
memory budget (not latency) picks the index family, which is exactly the
regime where PQ compression and Matryoshka dimensions earn their complexity.

## What each constraint decides

The compressed decision guide. Read the left column off your requirements; the
right columns say which lever it moves before you compare any tools.

| Your constraint | Lever it moves | Rule of thumb |
|---|---|---|
| Corpus size vs RAM | Index family | Fits in RAM: HNSW. Billion scale on a budget: IVF-PQ. Billion on one box: DiskANN |
| Latency budget | `ef` / `nprobe`, reranker gate | Raise the breadth knob until recall clears the bar, then spend what is left on reranking; 10-30ms is the cross-encoder's price |
| Exact tokens in the query mix | Hybrid retrieval | SKUs, error codes, names: BM25 alongside dense with RRF is mandatory, not an optimization |
| Result consumer | Reranking | Human reads the top 3: cross-encoder. Downstream ranker re-scores: skip it |
| Freshness SLA | Upsert strategy | Minutes: incremental inserts (HNSW, FreshDiskANN). Daily: scheduled rebuild is simpler and self-heals the graph |
| Filter selectivity | Filter placement | Passes most documents: in-index filtering works. Passes under ~1%: partition and route to a sub-index |
| Similarity objective | Quantizer | Two-tower dot-product retrieval: anisotropic PQ or L2-normalize; a Euclidean-tuned quantizer silently loses MIPS recall |
| Churn rate | Rebuild budget | Heavy deletes: track tombstone fraction and schedule rebuilds; incremental inserts are not free forever |
| Recall bar | Encoder dimension, rescore | Smallest model that clears the bar; always rescore compressed shortlists at full precision |

## The smallest runnable ANN index

The review of every vector-database tutorial is the same: the reader stands up
a service and still cannot see why `nprobe` exists. So here is the chapter's
central tradeoff, recall against work, in one file with zero installs. Every
production component is swapped for the smallest thing with the same interface:
the trained coarse quantizer becomes a few Lloyd iterations of k-means-lite,
the FAISS inverted lists become a dict, real embeddings become seeded Gaussian
clusters, and the flat brute-force scan plays its production role exactly, the
ground-truth recall ceiling that every ANN number is measured against.

```python
"""A toy IVF index in one file: build, probe, and measure recall vs work."""
import random

random.seed(7)
DIM, N_CLUSTERS, N_VECS, N_QUERIES, K = 16, 24, 3000, 60, 10

# --- synthetic corpus: clustered vectors, like real embeddings ---------------

def rand_unit():
    return [random.gauss(0, 1) for _ in range(DIM)]

true_centers = [rand_unit() for _ in range(N_CLUSTERS)]
corpus = []
for _ in range(N_VECS):
    c = random.choice(true_centers)
    corpus.append([x + random.gauss(0, 0.35) for x in c])

def dist2(a, b):
    return sum((x - y) ** 2 for x, y in zip(a, b))

# --- index build: k-means-lite centroids + inverted lists --------------------

NLIST = 32
centroids = random.sample(corpus, NLIST)
for _ in range(8):                                   # a few Lloyd iterations
    sums = [[0.0] * DIM for _ in range(NLIST)]
    counts = [0] * NLIST
    for v in corpus:
        j = min(range(NLIST), key=lambda i: dist2(v, centroids[i]))
        counts[j] += 1
        for d in range(DIM):
            sums[j][d] += v[d]
    centroids = [[s / c for s in sums[i]] if (c := counts[i]) else centroids[i]
                 for i in range(NLIST)]

inverted = {i: [] for i in range(NLIST)}             # centroid id -> vector ids
for vid, v in enumerate(corpus):
    inverted[min(range(NLIST), key=lambda i: dist2(v, centroids[i]))].append(vid)

# --- search: exact scan vs probing the n_probe nearest lists -----------------

def exact_topk(q):
    return sorted(range(N_VECS), key=lambda vid: dist2(q, corpus[vid]))[:K]

def ivf_topk(q, n_probe):
    lists = sorted(range(NLIST), key=lambda i: dist2(q, centroids[i]))[:n_probe]
    cands = [vid for i in lists for vid in inverted[i]]
    return sorted(cands, key=lambda vid: dist2(q, corpus[vid]))[:K], len(cands)

queries = [[x + random.gauss(0, 0.35) for x in random.choice(true_centers)]
           for _ in range(N_QUERIES)]
truth = [set(exact_topk(q)) for q in queries]

print(f"{N_VECS} vectors, {NLIST} lists, recall@{K} over {N_QUERIES} queries")
print("n_probe  recall@10  corpus scanned")
for n_probe in (1, 2, 4, 8, 16):
    hits = scanned = 0
    for q, t in zip(queries, truth):
        found, n_cands = ivf_topk(q, n_probe)
        hits += len(t & set(found))
        scanned += n_cands
    recall = hits / (K * N_QUERIES)
    frac = scanned / (N_VECS * N_QUERIES)
    print(f"{n_probe:>7}  {recall:>9.3f}  {frac:>13.1%}")
```

Run it and the table is the whole argument of [section 4](04-vector-index.md)
in five rows: at `n_probe = 1` the index scans 4.0% of the corpus and already
recovers 0.803 recall@10; at 2 probes it scans 7.3% for 0.985; by 4 probes it
reaches 1.000 recall while touching 14.2% of the vectors. Recall climbs with
`n_probe` while the scanned fraction stays far below the 100% a flat scan pays,
and the gap between those two columns is the entire economic case for ANN
search. It also shows why the knob saturates: past the point where the probed
cells cover the query's true neighborhood, extra probes buy nothing but
latency, which is why [section 6](06-serving-and-scaling.md) treats `nprobe`
and `ef` as budgets to tune against a recall bar, not dials to max out. Swap
the Gaussian clusters for real embeddings, the k-means-lite loop for a trained
coarse quantizer, the dict for FAISS inverted lists with PQ codes, and the
exact scan for an offline evaluation job, and you have rebuilt this chapter's
index layer.
