# 7. How teams do it in production

Every production RAG system converges on the same skeleton: embed the query,
retrieve candidate chunks from an index, rerank the shortlist, assemble a tight
grounded context, and let the LLM generate a cited answer. The offline half is
separate: parse, chunk, embed, and index the corpus with a freshness loop that
re-embeds changed documents. What varies between teams is where they invest:
retrieval strategy (how much intelligence into the search itself), reranking
hardness, and how seriously they evaluate.

## Where the real designs diverge

| System | Retrieval strategy | Reranking | Chunking and freshness | Evaluation | When it wins |
|---|---|---|---|---|---|
| Ramp | Dense over precomputed NAICS-code embeddings in ClickHouse | Two-prompt LLM selection (narrow then re-score) | Precomputed per taxonomy; infrequent refresh | accuracy@k; fuzzy hierarchical metric | Closed enumerable label set you embed once up front |
| Uber | Hybrid vector + BM25; Query Optimizer agent rewrites before retrieval | Post-processor agent deduplicates and reorders | LLM-enriched (summary, FAQ, keywords per chunk); Google Docs loader | LLM-as-judge (0 to 5 scale) | Ambiguous queries needing pre-retrieval rewriting; messy internal corpora |
| Microsoft GraphRAG | Knowledge-graph community traversal; graph built by LLM entity extraction | Community pre-summaries as context | LLM-extracted entity graph; costly to build and refresh | Comprehensiveness, diversity, SelfCheckGPT faithfulness | Multi-hop, whole-corpus sensemaking that flat vector retrieval misses |
| DoorDash | Vector RAG with a guardrail layer | LLM judge as guardrail | Support-doc chunks; periodic refresh | LLM judge + guardrail classification | Bounded support domain where guardrails can catch out-of-scope answers |
| Dropbox Dash | Hybrid lexical + semantic; chunking deferred to query time | Larger embedding model on the shortlist | Query-time chunking; periodic sync and webhooks | LLM judge (correctness, completeness) + source P/R/F1 | Fast-changing personal corpora requiring near-real-time freshness |
| Vespa | Hybrid BM25 + INT8 or binary quantized vectors | None (embedding model focus) | Pre-indexed; quantization tuned per model family | Quality vs latency benchmarks (recall, QPS) | Large indexes memory-constrained where quantization is the lever |
| NVIDIA | Dense vector first stage; cross-encoder second stage | Cross-encoder microservice (NeMo Retriever NIM) | Standard pre-chunking | Two-stage accuracy vs cost operating points | Cost-sensitive serving where hard reranking shrinks LLM token cost |
| Glean | Hybrid lexical + vector + enterprise knowledge graph + per-user ACL | Multi-signal permission-aware ranking | Permission-aware crawler; continuous sync | Permission-aware relevance | Enterprise search where per-user ACLs are non-negotiable |
| Databricks | Managed vector search (Delta Sync); real-time serving | Model selection layer | Enterprise data sync | Quality monitoring pipeline | Teams on the Databricks platform wanting managed RAG plus monitoring |
| MongoDB | Vector search via Atlas aggregation pipeline ($vectorSearch) | None by default; similarity scoring | Recursive or fixed-size with overlap; Voyage AI embeddings | Score transparency (per-doc scores surfaced to user) | Teams already on Atlas wanting zero-infra vector search |
| Grab | RAG over a vetted query-API catalog (Data-Arks); not over doc chunks | None; retrieval unit is a curated query | Parameterized SQL/Python APIs, not text chunks | Summarization quality on returned tables | Analyst corpora where the retrieval unit is a reusable query, not a passage |
| Thomson Reuters | Dense retrieval (MiniLM-L6-v2); cosine similarity in Milvus | None | KB articles + CRM chunked; non-parametric store | Qualitative examples; provenance citations | Regulated domain (tax) where knowledge updates without retraining |
| GitHub Copilot | Semantic + code search over repos | Yes | Repo and code indexing; freshness on active repos | Grounded-answer quality | Grounding answers in a large private code corpus with specialized semantics |
| Google / ETH RAGO | Serving-layer scheduling and placement optimization | Not applicable | Not applicable | QPS per chip, time-to-first-token | Squeezing throughput and latency from an existing RAG pipeline at scale |

The core dividing line: teams investing in richer retrieval (hybrid, knowledge
graph, or agentic query rewriting) are solving noisy-corpus or multi-hop
problems; teams investing in reranking and grounding are solving precision and
trust problems; teams investing in the serving layer (RAGO, Databricks) are
solving cost and throughput problems at large QPS.

## The systems

- **Ramp** [From RAG to Richness: How Ramp Revamped Industry Classification](https://builders.ramp.com/post/industry_classification): Embedding-model selection and two-prompt retrieval over NAICS codes, precomputed embeddings in ClickHouse. *(product design)*
- **Uber** [Enhanced Agentic-RAG: near-human precision for chatbots](https://www.uber.com/blog/enhanced-agentic-rag/): Pre-retrieval query agents, custom Google Docs loader, LLM chunk enrichment, LLM-as-judge eval. *(deployment)*
- **Microsoft Research** [GraphRAG: unlocking LLM discovery on narrative private data](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/): Knowledge-graph retrieval beats vector-only RAG on multi-hop private-data queries. *(product design)*
- **DoorDash** [Path to high-quality LLM-based Dasher support automation](https://careersatdoordash.com/blog/large-language-modules-based-dasher-support-automation/): RAG support bot with an LLM guardrail and judge; 90% fewer hallucinations. *(eval bar)*
- **Dropbox** [Building Dash: how RAG and AI agents meet business needs](https://dropbox.tech/machine-learning/building-dash-rag-multi-step-ai-agents-business-users): Hybrid lexical plus semantic retrieval balancing latency, freshness, cost; source F1 eval. *(deployment)*
- **Vespa** [Embedding Tradeoffs, Quantified](https://blog.vespa.ai/embedding-tradeoffs-quantified/): INT8 and binary quantization plus hybrid BM25 quality-vs-latency tradeoffs. *(eval bar)*
- **Vespa** [Asymmetric Retrieval: spend on docs, embed queries for free](https://blog.vespa.ai/asymmetric-retrieval-spend-on-docs-queries-for-free/): Large model for documents, tiny local model for queries, to cut serving cost. *(deployment)*
- **NVIDIA** [How a reranking microservice improves retrieval accuracy and cost](https://developer.nvidia.com/blog/how-using-a-reranking-microservice-can-improve-accuracy-and-costs-of-information-retrieval/): Two-stage embed-then-rerank; fewer chunks to the LLM cuts cost while holding accuracy. *(eval bar)*
- **Glean** [Why vector search isn't enough for enterprise RAG](https://www.glean.com/blog/hybrid-vs-rag-vector): Enterprise RAG needs hybrid search plus knowledge graph and permission-aware ranking. *(product design)*
- **Databricks** [Creating High Quality RAG Applications with Databricks](https://www.databricks.com/blog/building-high-quality-rag-applications-databricks): Real-time serving, model selection and eval, quality monitoring. *(deployment)*
- **LinkedIn** [Improving Post Search at LinkedIn](https://www.linkedin.com/blog/engineering/search/improving-post-search-at-linkedin): Layered first and second-pass rankers with separate relevance, quality, freshness models. *(deployment)*
- **Pinterest** [How we built Text-to-SQL at Pinterest](https://medium.com/pinterest-engineering/how-we-built-text-to-sql-at-pinterest-30bad30dabff): RAG table retrieval grounds LLM SQL generation over thousands of warehouse tables. *(product design)*
- **Cloudflare** [Introducing AutoRAG: managed RAG on Cloudflare](https://blog.cloudflare.com/introducing-autorag-on-cloudflare/): Managed pipeline with async indexing, Vectorize storage, query-time retrieval and generation. *(deployment)*
- **Vimeo** [Unlocking knowledge sharing for videos with RAG](https://medium.com/vimeo-engineering-blog/unlocking-knowledge-sharing-for-videos-with-rag-810ab496ae59): Video Q&A over transcript chunking, multi-size context windows, and vector retrieval. *(product design)*
- **Elastic** [RAG pipelines in production](https://www.elastic.co/search-labs/blog/rag-in-production): Hybrid retrieval, reranking, monitoring, and benchmarking at production scale. *(deployment)*
- **Anyscale** [Building RAG-based LLM Applications for Production](https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-1): End-to-end RAG built, evaluated, and served at scale with Ray Serve. *(deployment)*
- **GitHub** [What is retrieval-augmented generation?](https://github.blog/ai-and-ml/generative-ai/what-is-retrieval-augmented-generation-and-what-does-it-do-for-generative-ai/): How Copilot Enterprise grounds answers via internal code search and semantic retrieval. *(product design)*
- **Google / ETH Zurich** [RAGO: systematic performance optimization for RAG serving](https://arxiv.org/abs/2503.14649): A serving framework raising QPS per chip 2x and cutting time-to-first-token 55%. *(deployment)*
- **MongoDB** [Taking RAG to Production with the MongoDB Documentation AI Chatbot](https://www.mongodb.com/developer/products/atlas/taking-rag-to-production-documentation-ai-chatbot/): Atlas Vector Search docs chatbot; chunking and embedding-model choices; prototype to production. *(deployment)*
- **Grab** [Leveraging RAG-powered LLMs for Analytical Tasks](https://engineering.grab.com/transforming-the-analytics-landscape-with-RAG-powered-LLM): Data-Arks middleware retrieves vetted query APIs to ground report-generation bots. *(product design)*
- **Mercado Libre** [Beyond the Hype: Real-World Lessons from Working with Large Language Models](https://medium.com/mercadolibre-tech/beyond-the-hype-real-world-lessons-and-insights-from-working-with-large-language-models-6d637e39f8f8): RAG over technical docs; LLM-generated table descriptions; structured extraction via function calling. *(eval bar)*
- **Thomson Reuters** [Better Customer Support Using RAG at Thomson Reuters](https://medium.com/tr-labs-ml-engineering-blog/better-customer-support-using-retrieval-augmented-generation-rag-at-thomson-reuters-4d140a6044c3): RAG over domain knowledge to ground customer-support answers in a regulated domain. *(deployment)*
- **Anthropic** [Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval) (Sept 2024): the standard 2024 fix for the chunking-loses-context problem. Before indexing, prepend an LLM-generated snippet situating each chunk in its document (contextual embeddings), and pair it with contextual BM25; the write-up reports large reductions in retrieval failures, and more still when combined with a re-ranker. Cheap to run once you cache the document prefix. *(product design)*

For the full case-study comparison (divergence diagram, math, quadrant plot),
see the dense reference in [topics/01-rag-serving.md](../../topics/01-rag-serving.md).
