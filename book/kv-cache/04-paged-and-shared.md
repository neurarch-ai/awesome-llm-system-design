# 4. Paged and shared

Shrinking each KV entry addresses one dimension of the problem: the per-token
cost. The next dimension is fragmentation and reuse: how to pack more sequences
into the same memory, and how to avoid recomputing the cache for identical prefixes
across requests.

## PagedAttention: virtual memory for the KV cache

In a naive serving system, each sequence's KV cache is one contiguous buffer
reserved at request arrival. Two kinds of waste arise. First, the sequence might
not use all the space (internal fragmentation: the tail of the buffer sits empty).
Second, freed buffers leave gaps that cannot accommodate other sizes (external
fragmentation). Under a mixed workload of varying lengths this fragmentation can
waste 20% to 40% of GPU memory.

```python
def internal_fragmentation(num_tokens, block_size):
    # fraction of the last block that sits empty when the cache is stored in fixed blocks
    slots = ((num_tokens + block_size - 1) // block_size) * block_size
    return (slots - num_tokens) / slots
# internal_fragmentation(70, 16) -> 0.125   (5 blocks = 80 slots hold 70 tokens, 12.5% wasted)
```

**PagedAttention** (the idea at the core of vLLM) solves this by managing the KV
cache exactly as an operating system manages virtual memory:

- The cache is divided into fixed-size **blocks** (say, 16 or 32 tokens each).
- A per-sequence **block table** maps logical token positions to physical blocks
  scattered anywhere in GPU memory.
- Sequences share blocks when they share content; a reference-counted copy-on-write
  scheme handles divergence.
- When a sequence ends, its blocks are released one by one back to the pool,
  leaving no unusable gaps.

```mermaid
flowchart LR
  subgraph SEQ["Two concurrent sequences"]
    A1[Seq A tokens 0-15] --> BT_A[Block table A]
    A2[Seq A tokens 16-31] --> BT_A
    B1[Seq B tokens 0-15] --> BT_B[Block table B]
    B2[Seq B tokens 16-23] --> BT_B
  end
  BT_A --> P1[Physical block 3]
  BT_A --> P2[Physical block 7]
  BT_B --> P3[Physical block 1]
  BT_B --> P4[Physical block 5]
  P1 & P2 & P3 & P4 --> POOL[(Shared GPU block pool)]
```

**How the diagram works.** The pipeline borrows the operating system's virtual-memory
trick. Each sequence's tokens are grouped into fixed-size logical blocks (say 16
tokens each). Instead of a contiguous buffer, every sequence gets a **block table**,
an array that maps its logical block index to a **physical block** somewhere in one
shared GPU pool, exactly like an OS page table maps virtual pages to physical frames.
When a sequence needs another block it grabs any free physical block from the pool
(not a neighbor of its previous block), so the pool never fragments: allocation is
per-block and on demand, and freeing a sequence just returns its blocks to the pool.
The attention kernel is the part that has to change: it reads the block table to
gather the scattered K and V blocks during the attention computation rather than
assuming one contiguous run. Two sequences can even point their block tables at the
**same** physical block, which is how a shared prompt prefix (next section) is stored
once and reused.

The result: 2x to 4x higher throughput at matched latency compared to
FasterTransformer-style contiguous allocation, because the same GPU memory fits
more concurrent sequences. PagedAttention does **not** speed up a single
request; it raises aggregate concurrency and token throughput. Report it in tokens
per second across the fleet, not in per-request latency.

## Prefix caching: skip the prefill for repeated prefixes

In a RAG system with a fixed 4k system prompt, every request starts by rebuilding
the same KV cache for those 4000 tokens. That is pure waste. **Prefix caching**
(also called prompt caching) stores the KV block sequence for a completed prefix
and reuses it across requests that share the same prefix. The prefill phase is
skipped entirely for the matching tokens; the model jumps straight to decoding the
unique part of each request.

```mermaid
flowchart LR
  REQ1[Request 1: system prompt + query A] --> PF1[Prefill system prompt]
  PF1 --> CACHE[(Prefix KV cache)]
  PF1 --> DEC1[Decode query A part]

  REQ2[Request 2: same system prompt + query B] --> HIT{Prefix in cache?}
  HIT -- hit --> CACHE
  CACHE --> DEC2[Decode query B part only]
  HIT -- miss --> PF2[Prefill again]
```

**How it works.** The first request prefills its system prompt, and instead of discarding those keys and values after decoding, it deposits them in a prefix KV cache keyed by the token sequence. When a second request arrives sharing the same system prompt, a lookup checks whether that exact prefix is already cached. On a hit, the expensive prefill of the shared tokens is skipped entirely and decode starts straight from the cached state, so only the unique query-B suffix has to be computed. On a miss, the request falls back to prefilling from scratch, which is the same cost it would have paid without the cache. The whole win rests on the shared prefix being byte-for-byte identical, which is why the branch is a hard hit-or-miss decision rather than a partial match.

Gains are workload-dependent. Databricks measured 2.5x higher per-replica
input-token throughput and 3x lower P50 latency at a 30% cache hit rate. Anthropic
reports up to 90% cost reduction and 85% latency reduction when the cached context
is very large.

**Exact-prefix matching** is the critical constraint. A single different token at
any position in the prefix forces a full cache miss from that token onward. The
implication: always put the stable content (system prompt, shared document) at the
very beginning of the prompt, before any per-request variable content. The common
mistake is placing user-specific variables early, which defeats the cache for
every request.

For multi-turn chat, prefix caching can be applied turn by turn: each exchange
extends the prefix and can be cached for the next turn. Character.AI implements
this with a rolling-hash LRU tree keyed by conversation prefix, achieving about
95% hit rates across their fleet.

## RadixAttention: prefix caching for branching trees

When multiple requests share not just one fixed prefix but a **branching tree** of
prefixes (few-shot prompts with many examples, agentic workflows where a base
context fans out into many parallel sub-tasks), a flat prefix cache only matches
a single chain. A radix tree captures the full branching structure.

**RadixAttention** (SGLang) organizes the KV-block pool as a radix tree where each
edge is a sequence of tokens and each node is a cached KV block. Any new request
is routed down the tree, following edges that match its tokens and branching to a
new edge at the first mismatch. Eviction under memory pressure uses LRU, preserving
the most recently accessed paths. Cross-request cache sharing happens automatically
without any user API change.

**When to use which paging or sharing technique.**

| Reach for | When | Skip it when |
|---|---|---|
| PagedAttention (vLLM, TGI, SGLang) | Fragmentation is the binding constraint; requests vary widely in length | Block indirection adds a lookup cost; fuse it into the attention kernel |
| Prefix caching (vLLM, Databricks, Anthropic) | A large fixed system prompt or document repeats across requests | Context is all-unique per request; the cache never hits |
| RadixAttention (SGLang) | Requests share branching prefixes: few-shot, agent trees, parallel chains | Prefix diversity collapses cache benefit; LRU thrashes under highly varied traffic |
| Per-turn chat caching (Character.AI rolling hash) | Long-conversation product with heavy turn reuse; 100+ turn histories | Short or ephemeral conversations where reuse is low |

**Provenance.** PagedAttention originated with vLLM (UC Berkeley, 2023), which
brought OS-style block paging to the KV cache; RadixAttention is SGLang's prefix-tree
generalization of it, and prefix caching plus the Character.AI rolling-hash scheme
are attributed inline above.
