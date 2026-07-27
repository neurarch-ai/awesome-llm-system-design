# 10. Putting it together: the complete build

Sections 1 through 6 taught each stage with its options and tradeoffs; section 7
showed where real teams diverge. What none of them show is a single system with
every decision made. This capstone does three things: it gives you an opinionated
default stack so option paralysis never blocks a first build, it walks the
chapter's scenario end to end with every choice committed and costed, and it
shows how the same decisions flip when the constraints change. It closes with
the smallest runnable RAG pipeline, one file, no installs.

## The default stack: start here, deviate with reason

Every stage in this chapter has three to six credible options, and a first-time
builder can burn a week comparing libraries before retrieving a single chunk.
Skip that. The stack below is a sane default for a first production build; each
row names when to deviate and which section explains why. Tools change yearly,
but the interface of each stage (parse, chunk, embed, index, retrieve, rerank,
generate, evaluate) does not, so pick per stage by interface and treat any
specific library as replaceable.

| Stage | Default | Deviate when | Why (section) |
|---|---|---|---|
| Parsing | Layout-aware parser (Docling or Unstructured class); tables kept as markdown | Corpus is clean markdown already: skip straight to chunking | [3](03-indexing-and-chunking.md) |
| Chunking | Recursive structural, ~400-token cap, 10-15% overlap | Chunks lose meaning standalone: add contextual prefixes | [3](03-indexing-and-chunking.md) |
| Embedding | One strong small encoder (MiniLM / bge-small class, 384-dim) | Recall plateaus on domain jargon: domain-tuned or larger model | [3](03-indexing-and-chunking.md) |
| Index | Flat scan under ~100k chunks; HNSW above | Vectors outgrow RAM: quantize or IVF-PQ | [3](03-indexing-and-chunking.md) |
| Retrieval | Hybrid dense + BM25, RRF fusion, top-n = 50 | Corpus has no exact-term queries at all (rare): dense-only | [4](04-retrieval-and-reranking.md) |
| Reranking | Cross-encoder to top-m = 8 | First-token budget under ~800ms: shrink n or skip | [4](04-retrieval-and-reranking.md), [6](06-serving-and-scaling.md) |
| Generation | Mid-tier instruct model; cite source IDs; abstain on weak retrieval | Answers need multi-hop reasoning: climb the paradigm ladder | [5](05-generation-and-grounding.md), [2](02-frame-the-system.md) |
| Evaluation | 100-query golden set with labeled relevant chunks, built before tuning anything | Never. Build the golden set first | [3](03-indexing-and-chunking.md) |

The last row is the one beginners skip and regret: without a golden set, every
chunking and embedding decision is a vibe, and you cannot tell whether a change
helped. One afternoon of labeling pays for itself the first time you swap a
component.

## The complete build

Return to the scenario from [section 1](01-clarifying-requirements.md): 50
million internal documents, 10,000 employees, 20 QPS peak, p99 first-token under
1.5 seconds, freshness under one hour, cited answers with abstention, and
per-user ACL. Here is the whole system with every choice committed and the
reason it won.

| Decision | Choice | Why it won |
|---|---|---|
| Parsing | Layout-aware, tables preserved, boilerplate stripped | 50M docs include multi-column PDFs; a mangled parse is unrecoverable downstream |
| Chunking | Recursive structural, 400-token cap, 50-token overlap | Wikis and design docs have headings to exploit; overlap keeps boundary answers whole |
| Embedding | 384-dim encoder, separate write and read deployments | Dimension sets index memory (below); write path batches, read path is latency-bound |
| Index | HNSW, sharded, int8-quantized, ACL metadata on every chunk | Recall and latency at 45M chunks; ACL must filter inside the search |
| Retrieval | Hybrid dense + BM25 with RRF, top-n = 50 | Ticket IDs and product codes are exact-term queries dense retrieval blurs |
| Reranking | Cross-encoder, top-m = 8 | Cuts prefill tokens ~5x versus stuffing top-50; precision up, cost down |
| Generation | Mid-tier model, streaming, citation verification, abstention rule | Grounded-or-silent is the stated quality bar |
| Freshness | Change-driven upsert, tombstone by document ID | The one-hour requirement rules out nightly rebuilds |
| Evaluation | Golden set for recall@k offline; sampled LLM-judge and citation-failure rate online | Retrieval recall is the quality ceiling, so it is the first number watched |

**Index sizing.** 50M documents at ~300 tokens with a 400-token cap and
50-token overlap yield roughly 45M chunks ([section 3](03-indexing-and-chunking.md)).
At 384 dimensions and float32 that is 45M x 384 x 4 bytes, about 69 GB of raw
vectors; int8 quantization brings it near 17 GB plus HNSW graph overhead,
comfortable on two or three replicated shards. The same corpus at 768
dimensions would double every number, which is why embedding dimension was
decided together with the index rather than on leaderboard rank alone.

**Latency.** The component budget from [section 6](06-serving-and-scaling.md)
lands at ~1040ms p99 with reranking, inside the 1.5-second budget with
headroom: ~20ms query embed, ~40ms ACL-filtered ANN, ~80ms cross-encoder over
50 candidates, ~250ms prefill over 8 chunks, ~600ms to first decoded token.

**Cost per query.** The assembled prompt is roughly a 300-token system prompt
plus 8 chunks x 400 tokens plus the query, near 3,600 input tokens, with a
~300-token answer:

$$\text{cost/query} \approx T_{\text{in}} \cdot p_{\text{in}} + T_{\text{out}} \cdot p_{\text{out}} + c_{\text{rerank}} + c_{\text{embed}}$$

At illustrative mid-tier prices ($0.25 and $1.25 per million input and output
tokens), that is about $0.0009 + $0.0004 for the LLM; the query embedding is
noise and the self-hosted cross-encoder amortizes to a small fraction of the
LLM cost. Call it $0.0015 per query, roughly $250 per day at 170k queries.
The number worth internalizing is the counterfactual: without the reranker,
stuffing top-50 chunks makes the prompt ~20,000 tokens and multiplies LLM cost
and prefill latency by 5x. The reranker is not a quality luxury; it is the
component that pays for itself.

**What breaks in month one.** Three failure modes dominate early operations, so
wire their signals before launch: freshness lag (index upsert queue depth
against the one-hour promise), citation-verification failures (the
sub-millisecond check from [section 5](05-generation-and-grounding.md); a rising
rate means the generator is drifting off its context), and per-cohort recall
(users with narrow ACL visibility hit the small-candidate-set problem first,
and they will report "it can't find anything" while global metrics look fine).

## The same techniques under different constraints

The review question that matters in practice is not "which chunker is best" but
"which chunker is best under my constraints." Here is the same pipeline built
three times. Only the enterprise column is the build above; the other two keep
the identical stage interfaces and swap nearly every implementation choice.

| | Startup docs bot | Enterprise KB (this chapter) | Batch compliance answers |
|---|---|---|---|
| Corpus / traffic | 2k docs, ~30k chunks; 0.2 QPS | 50M docs, 45M chunks; 20 QPS | 5M docs; 100k queries nightly, no interactivity |
| Latency budget | Seconds are fine | p99 first token < 1.5s | None; throughput and cost only |
| Index | Flat scan in process (a few ms at 30k chunks); no ANN tuning at all | Sharded int8 HNSW, ACL inside search | IVF-PQ, memory-cheap; rebuilt on batch cadence |
| Retrieval / rerank | Dense-only top-10, no reranker until the golden set shows precision pain | Hybrid RRF top-50, cross-encoder to top-8 | Hybrid, hard rerank to top-4: prefill cost dominates the bill |
| Generation | Strong API model; at 0.2 QPS model price is irrelevant | Mid-tier, streaming, citation verify | Small model, huge batches, prefix caching, spot capacity |
| Freshness | Re-embed everything on deploy; the corpus fits in one batch | Change-driven upsert, one-hour SLA | Nightly rebuild is the feature, not a compromise |
| Eval | 50 golden queries maintained by hand | Golden set + online judge sampling + citation-failure rate | LLM-judge over a sampled slice of each batch |
| What would be over-engineering | ANN index, reranker, caches, agentic anything | GraphRAG for single-fact queries | Streaming, semantic cache, low-latency serving stack |

Two lessons fall out. First, the startup column is mostly deletions: at 30k
chunks a flat scan is exact, fast, and removes an entire tuning surface, and at
0.2 QPS every caching layer is dead weight. If the whole corpus fits in one
context window and queries are rare, [section 2](02-frame-the-system.md)'s
stuffing comparison says you may not need retrieval at all yet. Second, the
batch column shows latency and cost trading places as the binding constraint:
with no first-token budget, the reranker gets harder (m = 4), the model gets
smaller, and everything runs in the largest batch the hardware takes.

## What each constraint decides

The compressed decision guide. Read the left column off your requirements; the
right columns say which lever it moves before you compare any tools.

| Your constraint | Lever it moves | Rule of thumb |
|---|---|---|
| Chunk count | Index family | < ~100k: flat scan. To ~10M: HNSW in RAM. Beyond: shard, quantize, or IVF-PQ |
| First-token budget | Rerank depth n, kept chunks m | Under ~800ms: n down to ~20 or skip the cross-encoder; under ~2s: full n = 50, m = 8 |
| Cost per query | m first, then model tier | Prefill scales with m; halving m roughly halves LLM cost before touching the model |
| Freshness | Upsert strategy, paradigm ceiling | Minutes-to-hours: incremental upsert; also rules out expensive per-chunk LLM enrichment on the hot path |
| Query repetition | Caching layers | Repeat-heavy internal traffic: embedding + prefix caches pay immediately; long-tail public traffic: they do not |
| Exact identifiers in queries | Hybrid retrieval | Ticket IDs, SKUs, jargon: BM25 alongside dense is mandatory, +3-5pp recall for one fusion step |
| Multi-hop or whole-corpus questions | Paradigm rung | Climb the [ladder](02-frame-the-system.md) only when the failure mode demands: rewrite, then CRAG, then graph or agentic |
| Per-user permissions | Index choice | ACL must filter inside the ANN search; pick an index that supports metadata filtering natively |
| Answer-quality floor | Abstention + verification | Citation verify is nearly free; abstain below a retrieval-confidence threshold instead of guessing |

## The smallest runnable RAG

The review of every framework tutorial is the same: the reader assembles five
libraries and still cannot see the pipeline. So here is the entire read-and-write
path in one file with zero installs. Every production component is swapped for
the smallest thing with the same interface: the encoder becomes bag-of-words
cosine, the ANN index becomes a list scan, the cross-encoder becomes exact-term
overlap, and the LLM becomes the assembled prompt itself. The shape is the
lesson; every section of this chapter upgrades one function of this file.

```python
"""The whole read-and-write path in one file, runnable with no installs."""
import math, re
from collections import Counter

# --- write path -------------------------------------------------------------

def chunk(doc_id, text, size=40, overlap=8):
    """Fixed-size sliding window over words; production: structural chunking."""
    words, step, out = text.split(), size - overlap, []
    for i in range(0, len(words), step):
        out.append({"id": f"{doc_id}#{len(out)}", "text": " ".join(words[i:i + size])})
    return out

STOP = {"the", "a", "an", "is", "are", "for", "of", "to", "and", "what", "when", "does"}

def embed(text):
    """Bag-of-words vector; production: a transformer encoder (e.g. MiniLM).
    Stopwords are stripped for the same reason parsers strip boilerplate:
    high-frequency filler dominates the vector and poisons the match."""
    return Counter(t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOP)

def cosine(a, b):
    dot = sum(a[t] * b[t] for t in a if t in b)
    norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return dot / norm if norm else 0.0

INDEX = []                                   # production: HNSW / IVF-PQ with ACL metadata

def upsert(doc_id, text, acl):
    global INDEX
    INDEX = [e for e in INDEX if e["doc"] != doc_id]          # tombstone old chunks
    for c in chunk(doc_id, text):
        INDEX.append({"doc": doc_id, "id": c["id"], "text": c["text"],
                      "vec": embed(c["text"]), "acl": acl})

# --- read path --------------------------------------------------------------

def retrieve(query, user, n=4):
    """ACL filter runs inside the search, not after it."""
    visible = [e for e in INDEX if user in e["acl"]]
    return sorted(visible, key=lambda e: cosine(embed(query), e["vec"]), reverse=True)[:n]

def rerank(query, candidates, m=2):
    """Exact-term overlap; production: a cross-encoder over (query, chunk) pairs."""
    terms = set(re.findall(r"[a-z0-9]+", query.lower()))
    return sorted(candidates, key=lambda e: len(terms & set(e["text"].lower().split())), reverse=True)[:m]

def answer(query, user):
    chunks = rerank(query, retrieve(query, user))
    if not chunks or cosine(embed(query), chunks[0]["vec"]) < 0.05:
        return "I could not find a grounded answer."           # abstain, don't guess
    prompt = "Answer ONLY from these sources and cite their ids:\n"
    prompt += "\n".join(f"[{c['id']}] {c['text']}" for c in chunks)
    prompt += f"\nQuestion: {query}"
    return prompt                # production: LLM call + verify cited ids exist in prompt

# --- demo -------------------------------------------------------------------

upsert("wiki/oncall", "The on-call rotation for the payments team changes every "
       "Monday at 09:00 UTC. Escalations page the secondary after 15 minutes.", acl={"alice", "bob"})
upsert("design/refunds", "Refunds above 500 dollars require manager approval and "
       "are processed by the billing service within two business days.", acl={"alice"})

print(answer("when does the on-call rotation change?", user="bob"))
print("---")
print(answer("what is the refund approval threshold?", user="bob"))   # ACL: bob can't see it
```

Run it and the two queries demonstrate the chapter's two non-negotiables in
about sixty lines: bob's on-call question comes back grounded with a citable
chunk ID, and his refund question returns the abstention message, because the
only document that could answer it is outside his ACL and the filter ran inside
retrieval, so the system never saw, leaked, or hallucinated around it. Swap
`embed` for a real encoder, `INDEX` for an ANN index, `rerank` for a
cross-encoder, and the final prompt into an LLM call, and you have rebuilt this
chapter.
