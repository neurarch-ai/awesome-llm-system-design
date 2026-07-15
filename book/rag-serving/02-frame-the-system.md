# 2. Framing the system

## Input and output

The system takes a **natural-language query plus the querying user's identity**
and returns a **grounded natural-language answer with cited source IDs**, or an
explicit abstention when retrieval is too weak to ground a confident reply.

The user identity travels through the whole pipeline because ACL enforcement is
a retrieval constraint, not a post-processing step.

## Two paths: keep them strictly separate

RAG has an offline (write) path and an online (read) path. Keeping them separate
is the key architectural commitment: the offline path pays the expensive embedding
cost once per document change; the online path pays only a single query embedding
plus a fast index lookup per request.

### Write path (offline)

Documents arrive from the knowledge base, either through a bulk ingest or a
change-driven freshness loop. Each goes through parsing, chunking, embedding,
and upsert into the vector index with its metadata (ACL, source URL, timestamp).

```mermaid
flowchart LR
  D["documents"] --> P["parse / clean"]
  P --> C["chunk"]
  C --> E["embed chunks"]
  E --> V["vector index<br/>(with ACL metadata)"]
  C --> M["metadata store<br/>(source, timestamp, ACL)"]
  M --> V
```

A document update re-chunks and re-embeds only the changed document and upserts
the new chunks into the index (deleting the old ones by document ID). This is
what makes sub-hour freshness achievable without a full rebuild.

### Read path (online)

A query arrives, is embedded, and is compared against the index with an ACL
filter baked in. The top-k chunks are optionally reranked, assembled into a
prompt alongside the original query and source IDs, and sent to the LLM for
generation. The answer streams back with inline citations.

```mermaid
flowchart TD
  Q["query + user identity"] --> QE["embed query"]
  QE --> VS["ANN search<br/>(ACL-filtered inside search)"]
  VS --> TK["top-k chunks"]
  TK --> RR["cross-encoder rerank<br/>(optional, top-m output)"]
  RR --> PA["assemble prompt<br/>(system prompt + chunks + query)"]
  PA --> G["LLM generate"]
  G --> A["answer + cited source IDs"]
  A --> S["stream to user"]
```

## The retrieve-then-generate contract

The two stages have different jobs and different cost profiles.

**Retrieval** is cheap and optimizes for recall. It runs in tens of milliseconds.
Its failure mode is missing a relevant chunk (false negative), because no
downstream stage can recover a chunk that was never retrieved.

**Generation** is expensive and optimizes for answer quality given what retrieval
provided. Its failure mode is hallucinating when the retrieved context is weak
or absent.

The contract between them is simple: retrieval must get the relevant chunk into
the context window; generation must not invent facts outside that context. When
answers are wrong, always ask "was the relevant chunk retrieved?" before "was
the generator weak?"

## Why this framing matters in an interview

The candidate who draws one box labeled "RAG" and moves on has said nothing. The
candidate who separates the write path from the read path, identifies the ACL
constraint as a retrieval problem not a post-processing problem, and names
retrieval recall as the quality ceiling has already answered the hardest part
of the question before touching a single component.

## The RAG paradigm ladder

Naive "embed, retrieve top-k, stuff into the prompt" is the bottom rung. When it
plateaus (wrong chunks retrieved, multi-hop questions, a noisy corpus), teams climb
a ladder of paradigms. Naming the rungs, and knowing which failure each one fixes,
is a strong senior-level signal.

| Paradigm | What it adds over naive RAG | Reach for it when |
|---|---|---|
| Naive RAG | embed, retrieve top-k, generate | a clean corpus and single-fact questions |
| Advanced RAG | pre-retrieval (query rewriting/expansion) plus post-retrieval (rerank, compress) | retrieval recall or precision is the bottleneck |
| Modular RAG | swappable modules (router, memory, fusion) composed per query | different query types need different pipelines |
| Self-RAG | the model emits reflection tokens deciding when to retrieve and whether a passage is relevant and supported | the model should retrieve only when needed and self-check its own grounding |
| Corrective RAG (CRAG) | a lightweight retrieval evaluator grades the hits; a low-confidence grade triggers a fallback search | the corpus is patchy and wrong retrievals must be caught before generation |
| GraphRAG | an LLM-built entity graph plus community summaries used as context | whole-corpus sensemaking and multi-hop questions that flat vector search misses |
| Agentic RAG | retrieval becomes a tool an agent calls in a loop, rewriting the query and retrieving repeatedly | complex multi-step questions that need iterative retrieval |

**Provenance.** The naive, advanced, and modular framing is from the RAG survey
(Gao et al., 2023). Self-RAG's reflection-token approach is from Asai et al.
(University of Washington and Allen AI, 2023); Corrective RAG (CRAG) from Yan et al.
(2024); GraphRAG from Microsoft Research (2024). Agentic RAG is the pattern where an
[agent loop](../agents/) drives retrieval, deployed for example in Uber's support
chatbot (section 7).

Climb only as far as the failure mode demands: each rung up the ladder adds latency,
cost, and moving parts, so a clean single-fact FAQ should stay near the bottom while
a noisy multi-hop knowledge base earns the graph or agentic rungs.
