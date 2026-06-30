# 08 - Semantic search and embedding service

> **Interviewer:** "Design a semantic search service over a large corpus, say
> 100M items, that returns the most relevant results for a query in tens of
> milliseconds. This is the search service itself, the thing other systems call,
> not the full answer pipeline."

This is the retrieval engine that sits under RAG ([topic 01](01-rag-serving.md)),
recommendations, and product search. Topic 01 covers the full answer pipeline;
here the focus is the search service: the embedding model as a service, the vector
index, and how you trade recall against latency at scale. The classic mistake is
to say "embed and do nearest neighbor" and stop. The depth is in the index choice
and in knowing that pure dense search has real blind spots.

## 1. Clarify and scope

- **What is being searched?** Documents, products, images, code? This decides the
  embedding model and whether you need multimodal.
- **Scale and growth?** 100M items now, growing how fast? This sets the index
  type and the re-indexing strategy.
- **Latency target?** Tens of milliseconds for the search itself is typical when
  it feeds a larger pipeline. State it; it constrains the index.
- **Recall bar?** Is this feeding a human (precision at the top matters most) or a
  downstream ranker (recall in the top few hundred matters most)?
- **Freshness?** Must a new item be searchable in seconds, or is hourly fine?

## 2. Requirements

**Functional**
- Map queries and items to vectors with an embedding model
- Find the nearest items to a query vector at scale
- Combine semantic and lexical signals (hybrid search)
- Optionally re-rank the top candidates for precision
- Keep the index current as items change

**Non-functional**
- Search latency in the tens of milliseconds at p99
- High recall@k against a labeled set
- Index memory cost that fits the budget at 100M+ vectors
- Freshness target met without full rebuilds

## 3. High-level data flow

Two paths, as always.

### Offline / write path

```mermaid
flowchart LR
  I["items"] --> E["embedding model<br/>(batched)"]
  E --> V["vectors"]
  V --> X["build / upsert<br/>vector index"]
  I --> L["lexical index<br/>(BM25)"]
```

### Online / read path

```mermaid
flowchart LR
  Q["query"] --> E["embed query"]
  E --> A["ANN search<br/>(dense)"]
  Q --> L["lexical search<br/>(BM25)"]
  A --> M["merge / fuse"]
  L --> M
  M --> R["re-rank<br/>(cross-encoder, optional)"]
  R --> T["top-k"]
```

## 4. Deep dives

### The embedding model as a service

The encoder that maps text (or images) to a vector is its own service with two
very different workloads:

- **Write path:** bulk-embed the whole corpus and re-embed on change. Throughput
  bound, batch aggressively, run on cheaper hardware, can be asynchronous.
- **Read path:** embed one query per request, latency bound. Cache query
  embeddings since repeated and near-repeated queries are common.

**Dimension is a real cost knob.** A larger embedding dimension (say 1024 vs 384)
can lift recall slightly but it scales index memory and search time linearly. At
100M vectors, dimension directly sets your RAM bill, so do not pick the biggest
model by default; pick the smallest that clears the recall bar. Some models
support shortening the vector (dimension truncation) to trade recall for cost
without re-embedding.

### Vector index: exact is too slow, so approximate

Exact nearest neighbor over 100M vectors per query is too slow. Use an
**approximate nearest neighbor (ANN)** index and accept a small recall loss:

- **HNSW (graph-based):** excellent recall and latency, but high memory because
  it stores the graph plus full vectors. Great when memory is not the constraint.
- **IVF-PQ (inverted file + product quantization):** clusters vectors and
  compresses them, so memory drops by a large factor. Some recall loss from
  quantization. At 100M+ this is often the pragmatic choice.
- The knobs (HNSW `ef`, IVF `nprobe`) trade recall against latency at query time,
  so you can tune per query class without rebuilding.

**Shard** for capacity and **replicate** for QPS. Searching shards in parallel and
merging keeps latency flat as the corpus grows.

### Hybrid search: why pure dense is not enough

Dense embeddings capture meaning but miss things lexical search nails:

- **Exact matches:** product SKUs, error codes, names, rare tokens. A vector may
  rank a semantically similar item above the exact one the user typed.
- **Out-of-domain terms:** words the embedding model never learned well.

So run **dense and lexical (BM25) in parallel and fuse the results** (for example
with reciprocal rank fusion). Hybrid search reliably beats either alone and is the
expected senior answer. Mention it unprompted.

### Re-ranking for precision

ANN optimizes cheap recall over the whole corpus. A **cross-encoder re-ranker**
then scores the top candidates (say the top 100) jointly with the query and keeps
the best few. It is expensive per pair but runs on a tiny set, and it sharply
improves the order at the top. Make it optional behind the latency budget: skip it
when feeding a downstream ranker that will re-score anyway.

### Freshness and incremental indexing

A full rebuild of a 100M index is slow and expensive. Support **incremental
upserts** so a changed item re-embeds and updates in place. HNSW handles inserts
well; IVF may need periodic re-clustering as the distribution drifts. State the
freshness target and pick accordingly.

## 5. Bottlenecks and scaling

| Bottleneck | First sign | Fix | Tradeoff |
|---|---|---|---|
| Query embedding latency | p99 creeps up | Cache; smaller encoder; truncate dimension | Slight recall loss |
| Index memory at scale | RAM cost balloons | IVF-PQ compression; smaller dimension | Recall loss |
| Search latency | Search dominates the budget | Shard and parallelize; lower `nprobe`/`ef` | Recall loss |
| Recall too low | Misses exact matches | Add lexical, fuse (hybrid) | More moving parts |
| Re-rank cost | Tail latency spikes | Re-rank fewer candidates; gate by budget | Precision loss |
| Re-index lag | Stale results | Incremental upsert; periodic re-cluster | Write-path complexity |

## 6. Failure modes and eval

- **Recall blind spots:** dense-only missing exact terms. Hybrid search is the
  fix; catch it by evaluating on queries with rare tokens.
- **Embedding drift:** if you change the embedding model, every vector must be
  re-embedded, and you cannot mix old and new vectors in one space. Plan model
  upgrades as a full re-index.
- **Eval:** measure **recall@k** against labeled query-item pairs (the ceiling on
  everything downstream) and latency at p99 under realistic QPS. Gate index or
  model changes behind both.

## 7. Likely follow-ups

- "Recall is low, where do you look?" Embedding model fit and missing lexical
  signal, before touching the index parameters.
- "Memory is too high at 100M vectors." Quantize (IVF-PQ), shrink the dimension,
  shard across machines.
- "You want to search images too." Multimodal embeddings that put text and images
  in one shared space, so a text query retrieves images.
- "How do you upgrade the embedding model with zero downtime?" Build the new index
  alongside the old, dual-read, then cut over.

---

## Seen in production

Real systems that ship the patterns above. Each is a first-party engineering
writeup; read them for what an interview answer skips: who the system serves,
the product design, the eval bar, and the deployment shape.

- **Spotify** [Introducing Voyager: Spotify new nearest-neighbor search library](https://engineering.atspotify.com/2023/10/introducing-voyager-spotifys-new-nearest-neighbor-search-library): HNSW ANN library: recall versus speed versus memory tradeoffs, 8-bit compression. *(deployment)*
- **Vespa** [Billion-scale vector search using hybrid HNSW-IF](https://blog.vespa.ai/vespa-hybrid-billion-scale-vector-search/): In-memory HNSW plus disk-backed inverted files for 90% recall under 50ms, cheaply. *(deployment)*
- **LinkedIn** [Semantic Search for AI Agents at Scale](https://www.linkedin.com/blog/engineering/ai/semantic-search-for-ai-agents-at-scale-retrieval-and-ranking-for-linkedins-hiring-assistant): Two-stage ANN retrieval plus ranker over 1B+ profiles using Matryoshka embeddings. *(deployment)*
- **Pinterest** [Advancements in Embedding-Based Retrieval at Pinterest Homefeed](https://medium.com/pinterest-engineering/advancements-in-embedding-based-retrieval-at-pinterest-homefeed-d7d7971a409e): Two-tower embedding retrieval with multi-embedding ANN and interest filters. *(deployment)*

- **Meta** [Faiss: a library for efficient similarity search](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/): A GPU-accelerated billion-scale similarity search library powering retrieval. *(deployment)*
- **Google Research** [Announcing ScaNN: efficient vector similarity search](https://research.google/blog/announcing-scann-efficient-vector-similarity-search/): Anisotropic quantization wins recall-vs-QPS on ann-benchmarks. *(eval bar)*
- **Microsoft Research** [DiskANN: vector search for all](https://www.microsoft.com/en-us/research/project/project-akupara-approximate-nearest-neighbor-search-for-large-scale-semantic-search/): SSD-backed ANN: billion vectors, 95% recall, about 5ms latency. *(deployment)*
- **Meta** [Embedding-based Retrieval in Facebook Search](https://arxiv.org/abs/2006.11632): A unified embedding framework for personalized social search. *(product design)*
- **Instacart** [How Instacart uses embeddings to improve search relevance](https://company.instacart.com/how-its-made/how-instacart-uses-embeddings-to-improve-search-relevance): A two-tower items model served via FAISS ANN with daily indices. *(deployment)*
- **Etsy** [Unified Embedding Based Personalized Retrieval in Etsy Search](https://arxiv.org/abs/2306.04833): Graph, transformer, and term embeddings with HNSW and 4-bit PQ. *(product design)*
- **Airbnb** [Applying Embedding-Based Retrieval to Airbnb Search](https://arxiv.org/abs/2601.06873): EBR for a two-sided marketplace, A/B-tested booking gains. *(who it serves)*
- **Uber Eats** [Scaling Multilingual Semantic Search in Uber Eats](https://arxiv.org/abs/2603.06586): Multilingual retrieval across stores, dishes, grocery in six markets. *(deployment)*
- **Walmart** [Semantic Retrieval at Walmart](https://arxiv.org/abs/2412.04637): Hybrid inverted-index plus neural retrieval for tail product queries. *(product design)*
- **Dropbox** [Selecting a model for semantic search at Dropbox scale](https://dropbox.tech/machine-learning/selecting-model-semantic-search-dropbox-ai): Benchmarking 11 embedding models on MTEB to pick multilingual-e5-large for retrieval. *(eval bar)*
- **GitHub** [Inside Copilot's new code embedding model](https://github.blog/news-insights/product-news/copilot-new-embedding-model-vs-code/): A custom code embedding model lifting Copilot retrieval quality 37.6% at lower latency. *(eval bar)*
- **Faire** [Beyond BM25 and dense embeddings: smart, interpretable retrieval](https://craft.faire.com/beyond-bm25-and-dense-embeddings-841a7b18ce27): SPLADE sparse neural retrieval giving interpretable semantics over Elasticsearch. *(product design)*
- **Snap** [Embedding-based Retrieval with Two-Tower Models in Spotlight](https://eng.snap.com/embedding-based-retrieval): Two-tower user/video embeddings for real-time short-form recommendation retrieval. *(deployment)*
- **Yelp** [Yelp Content As Embeddings](https://engineeringblog.yelp.com/2023/04/yelp-content-as-embeddings.html): Shared low-dimensional embeddings of reviews, businesses, and photos as a ranking baseline. *(deployment)*
- **Stack Overflow** [Vector databases in generative AI applications](https://stackoverflow.blog/2023/10/09/from-prototype-to-production-vector-databases-in-generative-ai-applications/): A self-hosted Weaviate vector database on Azure for production semantic search. *(deployment)*
- **Mercari** [Domain-Aware Text Embeddings for C2C Marketplaces](https://arxiv.org/abs/2512.21021): Domain-aware text embeddings improving search for Japan's largest C2C marketplace. *(product design)*
- **Amazon** [Semantic Product Search](https://arxiv.org/abs/1907.00937): A KDD 2019 two-tower model with kNN retrieval over precomputed catalog embeddings. *(product design)*

More production case studies: the [Evidently AI ML system design database](https://www.evidentlyai.com/ml-system-design) (800 case studies from 150+
companies) is the broadest curated index; this section pulls the ones that map
directly onto this topic.

---
## Trace the architectures

Semantic search lives or dies on the encoder that produces the vectors, and the
encoder is exactly where interview discussions get vague (a box labeled
"embedder"). Open the real thing and read its dimensions.

- **A text embedding model (MiniLM-L6, encoder-only):**
  [open it live](https://www.neurarch.com/?import=https://raw.githubusercontent.com/neurarch-ai/awesome-llm-model-zoo/main/architectures/all-minilm-l6/model.json)
  to trace how an encoder-only stack pools its hidden states into a single vector
  and where the embedding dimension is set, the number that drives your whole
  index memory budget. This is the kind of small sentence-embedding model a real
  search service runs.

  ![MiniLM-L6](https://raw.githubusercontent.com/neurarch-ai/awesome-llm-model-zoo/main/architectures/all-minilm-l6/assets/diagram.png)

- **Shared text-and-image embedding space (CLIP ViT-B/32):**
  [open it live](https://www.neurarch.com/?import=https://raw.githubusercontent.com/neurarch-ai/awesome-llm-model-zoo/main/architectures/clip-vit-b32/model.json)
  to see how two encoders map different modalities into one vector space for
  multimodal search.

  ![CLIP ViT-B/32](https://raw.githubusercontent.com/neurarch-ai/awesome-llm-model-zoo/main/architectures/clip-vit-b32/assets/diagram.png)

These are validated reference graphs at real dimensions, shape-checked end to end,
not screenshots. All 92 architectures live in the
[Model Zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo)
([gallery](https://neurarch-ai.github.io/awesome-llm-model-zoo)). Built by
[Neurarch](https://www.neurarch.com).
