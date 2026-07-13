# 7. How teams do it in production

Every production semantic search service runs the same spine: encode the corpus
offline, build an ANN index, embed the query at request time, retrieve
approximate neighbors, fuse with a lexical channel, optionally rerank. What
differs across companies is where they spend the memory budget, how aggressively
they compress vectors, and whether they run one retrieval stage or two.

## Where the real designs diverge

| System | ANN index | Compression | Hybrid lexical | Reranking | Key design choice |
|---|---|---|---|---|---|
| Spotify (Voyager) | HNSW (hnswlib) | E4M3 8-bit float (4x vs Annoy) | No | No | Stateless in-memory K8s pods; index file loaded on deploy; Python + Java bindings share distance math |
| Meta (Faiss) | IVF-PQ on GPU | OPQ + IMI + PQ 20-byte codes | Not stated | Refine shortlist at full precision | Billion-scale GPU acceleration; k-selection kernel in registers; ~40% recall@1 at 2ms on Deep1B |
| Google (ScaNN) | Partition + anisotropic PQ | Anisotropic learned quantization | Not stated | Full-precision rescore | Penalizes parallel error for MIPS; 2x QPS at equal recall on glove-100-angular |
| Microsoft (DiskANN) | Vamana graph on SSD | PQ codes in DRAM for traversal | Not stated | Full-precision from SSD | 1B vectors on one commodity machine; 95% recall at ~5ms from SSD |
| Vespa | HNSW-IF hybrid | int8 vectors | Yes (dense + inverted file) | Full-precision rescore (depth 4000) | 1B 100-dim int8 vectors; 90% recall@10 under 50ms at ~$6K/month; CRUD-friendly |
| LinkedIn | IVFPQ (Galene) | Matryoshka 2048-dim for ANN, 4096-dim for ranker | Not stated | DCNv2 learned ranker (L2 stage) | 1B+ profiles; weekly batch + daily CDC delta; one model trains both dimensions |
| Instacart | FAISS ANN | Not stated | No (keyword and category complement EBR) | Downstream ranker | Daily FAISS index rebuild; 95% query embedding cache hit; +4.1% cart-adds in A/B |
| Etsy | HNSW | 4-bit PQ | Yes (term + neural) | Not stated | Compact 4-bit PQ codes; +5.58% purchase rate; hard-negative training |
| Dropbox | Not stated | 8-bit custom scaling, full dimension | No | Not stated | Ran 11 models through MTEB to pick multilingual-e5-large (MRR 0.5044 vs 0.3299 runner-up); 4KB/doc metadata cap forced dimension choice |
| Walmart | Inverted index + neural | Not stated | Yes (inverted + neural) | Not stated | Hybrid system for tail product queries where lexical precision matters; two channels to maintain |
| Faire | SPLADE over Elasticsearch | Sparse neural | Yes (sparse neural is the lexical-style channel) | Not stated | Interpretable sparse-neural retrieval; term expansion on existing ES infra |

The dividing line is clear in the table. Systems that hold the corpus in RAM
choose HNSW for best recall at latency (Spotify, Vespa for its centroid tier,
Etsy). Systems at billion scale on a budget compress into PQ codes and
optionally push to SSD (Meta Faiss, DiskANN). Systems with mixed query types
always run a lexical channel alongside dense (Vespa, Etsy, Walmart). Systems
needing two quality levels from one model use Matryoshka embeddings (LinkedIn).

**The Dropbox case is underrated.** Before touching any index, they ran a
rigorous model-selection study: adapted the MTEB benchmark to their multilingual
domain, tested 11 models, and discovered that the gap between the winner and
runner-up was more than 50% relative MRR. The lesson: if model selection is
underdone, a better index type is unlikely to close the gap.

## The systems (first-party write-ups)

- **Spotify** [Introducing Voyager: Spotify's new nearest-neighbor search library](https://engineering.atspotify.com/2023/10/introducing-voyager-spotifys-new-nearest-neighbor-search-library): HNSW wrapped in hnswlib, E4M3 8-bit quantization, stateless in-memory Kubernetes deployment.

- **Vespa** [Billion-scale vector search using hybrid HNSW-IF](https://blog.vespa.ai/vespa-hybrid-billion-scale-vector-search/): In-memory HNSW over centroids plus disk-backed inverted file for 1B int8 vectors; 90% recall@10 under 50ms; CRUD-ready.

- **Meta** [Faiss: a library for efficient similarity search](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/): GPU-accelerated IVF-PQ for billion-scale similarity search with OPQ, IMI, and PQ composition.

- **Google Research** [Announcing ScaNN: efficient vector similarity search](https://research.google/blog/announcing-scann-efficient-vector-similarity-search/): Anisotropic learned quantization tuned to MIPS; top recall-vs-QPS on ann-benchmarks.

- **Microsoft Research** [DiskANN](https://www.microsoft.com/en-us/research/project/project-akupara-approximate-nearest-neighbor-search-for-large-scale-semantic-search/): SSD-backed Vamana graph; 1B vectors at 95% recall at ~5ms on one machine.

- **LinkedIn** [Semantic Search for AI Agents at Scale](https://www.linkedin.com/blog/engineering/ai/semantic-search-for-ai-agents-at-scale-retrieval-and-ranking-for-linkedins-hiring-assistant): Two-stage IVFPQ retrieval plus DCNv2 ranker over 1B+ profiles using Matryoshka embeddings.

- **Pinterest** [Advancements in Embedding-Based Retrieval at Pinterest Homefeed](https://medium.com/pinterest-engineering/advancements-in-embedding-based-retrieval-at-pinterest-homefeed-d7d7971a409e): Two-tower ANN retrieval with multi-embedding fan-out and interest filters.

- **Instacart** [How Instacart uses embeddings to improve search relevance](https://company.instacart.com/how-its-made/how-instacart-uses-embeddings-to-improve-search-relevance): ITEMS bi-encoder served via FAISS, daily index rebuild, 95% query cache hit rate.

- **Etsy** [Unified Embedding Based Personalized Retrieval in Etsy Search](https://arxiv.org/abs/2306.04833): HNSW with 4-bit PQ, term plus neural hybrid; +5.58% purchase rate.

- **Dropbox** [Selecting a model for semantic search at Dropbox scale](https://dropbox.tech/machine-learning/selecting-model-semantic-search-dropbox-ai): MTEB-adapted model selection across 11 models; multilingual-e5-large won; 8-bit quantization with custom scaling.

- **Walmart** [Semantic Retrieval at Walmart](https://arxiv.org/abs/2412.04637): Hybrid inverted-index plus neural retrieval for tail product queries.

- **Faire** [Beyond BM25 and dense embeddings](https://craft.faire.com/beyond-bm25-and-dense-embeddings-841a7b18ce27): SPLADE sparse-neural retrieval on Elasticsearch for interpretable semantics.

- **GitHub** [Inside Copilot's new code embedding model](https://github.blog/news-insights/product-news/copilot-new-embedding-model-vs-code/): Custom code embedding model lifting Copilot retrieval quality 37.6% at lower latency.

- **Uber Eats** [Scaling Multilingual Semantic Search in Uber Eats](https://arxiv.org/abs/2603.06586): Multilingual retrieval across stores, dishes, grocery in six markets.

- **Amazon** [Semantic Product Search](https://arxiv.org/abs/1907.00937): Two-tower model with kNN retrieval over precomputed catalog embeddings; the KDD 2019 canonical reference for two-tower product search.

For the full comparison and additional case studies, see
[topics/08-semantic-search-and-embeddings.md](../../topics/08-semantic-search-and-embeddings.md).
