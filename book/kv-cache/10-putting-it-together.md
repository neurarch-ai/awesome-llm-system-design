# 10. Putting it together: the complete build

Sections 1 through 6 taught each lever with its options and tradeoffs; section 7
showed where real teams diverge. What none of them show is a single system with
every decision made. This capstone does three things: it gives you an opinionated
default stack so option paralysis never blocks a first build, it walks the
chapter's scenario end to end with every choice committed and sized, and it
shows how the same decisions flip when the constraints change. It closes with
the smallest runnable KV-cache model, one file, no installs.

## The default stack: start here, deviate with reason

Every lever in this chapter has two to five credible options, and a first-time
builder can burn a week comparing serving engines before decoding a single
token. Skip that. The stack below is a sane default for a first production
build; each row names when to deviate and which section explains why. Engines
change yearly, but the levers themselves (shrink each entry, page the pool,
reuse across requests, extend position, batch continuously) do not, so decide
per lever and treat any specific engine as replaceable.

| Lever | Default | Deviate when | Why (section) |
|---|---|---|---|
| Attention variant | GQA, 4 to 8 query heads per KV head (ships in Llama 3, Mistral, Gemma) | Cache is still the wall at target context and you control training: MLA | [3](03-shrinking-the-cache.md) |
| KV precision | FP16 first; FP8 once your long-context eval passes | Fixed checkpoint and tight memory: INT4 per-token, keys at higher bits than values | [3](03-shrinking-the-cache.md) |
| Memory management | PagedAttention, 16-token blocks, copy-on-write sharing | Single sequence on a single GPU: contiguous is fine and simpler | [4](04-paged-and-shared.md) |
| Prefix reuse | Prefix caching with all stable content placed first in the prompt | Traffic is all-unique per request: the cache never hits | [4](04-paged-and-shared.md) |
| Branching reuse | Flat prefix cache | Agent trees or few-shot fan-out share branching prefixes: RadixAttention | [4](04-paged-and-shared.md) |
| Context extension | YaRN plus a short fine-tune for 4x to 16x past training length | Streaming forever with no mid-context recall needed: sliding window plus sinks | [5](05-long-context.md) |
| Prefill policy | One parallel pass; chunked prefill only when it would OOM at target batch | Short prompts: chunking adds latency for nothing | [5](05-long-context.md) |
| Batching | Continuous batching from day one | Never for a shared service; static batching only for offline single-tenant runs | [6](06-serving-and-scaling.md) |
| Speculative decoding | Off | Low-to-moderate batch on structured output (code, templates): turn it on | [6](06-serving-and-scaling.md) |
| Eviction | None; lossless levers only | Cache truly cannot fit and forgetting old context is acceptable: sinks plus window; must stay exact: query-aware sparsity (Quest) | [3](03-shrinking-the-cache.md) |

The last row is the one to internalize first: eviction is the only lever in the
table that changes what the model can answer. Everything above it is lossless,
so exhaust the lossless rows before touching it, and if you do touch it, gate
the change behind a needle-in-a-haystack eval, not perplexity.

## The complete build

Return to the scenario from [section 1](01-clarifying-requirements.md): a RAG
chatbot at 32k context today targeting 128k, first token under 2 s, inter-token
under 100 ms, 500 to 1000 concurrent sessions per GPU node with 3x spikes, a 4k
system prompt shared by every request, and a model we control. Here is the
whole system with every choice committed and the reason it won.

| Decision | Choice | Why it won |
|---|---|---|
| Attention variant | GQA with 8 KV heads, uptrained from the MHA checkpoint | 4x smaller cache at near-MHA quality for ~5% of pretraining cost; we control the weights, so this is the cheapest large win |
| KV precision | FP8, eval-gated; keys kept at higher precision than values | Halves the cache again; the interviewer allowed quantization if quality holds, and keys are the sensitive tensor |
| Memory management | PagedAttention, 16-token blocks, copy-on-write | Mixed 4k-to-32k lengths fragment contiguous buffers; paging recovers 20% to 40% of wasted HBM |
| Prefix reuse | Prefix caching keyed on the 4k system prompt, stable content first; per-turn caching for session history | The shared prompt is the single largest first-token lever; per-turn extension makes multi-round resume cheap |
| Cluster routing | Cache-aware routing (llm-d style) | Per-node prefix caches fragment at fleet scale; round-robin would silently destroy the hit rate |
| Context extension | YaRN plus a short long-sequence fine-tune for the 128k roadmap | 4x extension (32k trained to 128k served) with a better quality profile than plain interpolation |
| Prefill policy | One-pass for cached-prefix requests; chunked prefill for cold 32k prompts | Bounds peak prefill memory at high concurrency; the cold path is rare once the prefix cache warms |
| Batching | Continuous batching with an admission cap and LRU eviction of idle sessions, in-flight sequences protected | The concurrency target forces it; the cap plus eviction is what absorbs the 3x spikes without preempting live decodes |
| Speculative decoding | Off at launch | The node runs at high batch where draft compute adds pressure without clearing the memory-bandwidth bottleneck |
| Eviction of context | None; lossless levers only | A RAG chatbot must answer questions about any part of the retrieved document; lossy eviction fails exactly that workload |

**Bytes per token.** With $L = 32$, $h_{\text{kv}} = 8$, $d_{\text{head}} = 128$
and FP16, each token costs $2 \times 8 \times 128 \times 2 = 4096$ bytes per
layer, 128 KB per token across the stack ([section 2](02-the-cost-model.md)).
The same model in MHA would pay 512 KB per token; GQA already cut 4x. FP8 halves
the 128 KB to 64 KB. This one number drives every other figure below.

**Cache at target context.** A full 32k session (32 768 tokens) costs
$2 \times 32 \times 32768 \times 8 \times 128 \times 2 \approx 4.29$ GB in FP16,
the same arithmetic as [section 8](08-interview-qa.md)'s Llama 3 8B estimate,
and 2.15 GB in FP8. The 4k shared system prompt is stored once and shared by
copy-on-write blocks, so each session uniquely owns about 28.7k tokens, near
1.9 GB at FP8 (Illustrative).

**Memory budget vs the node.** The naive requirement is the
[section 8](08-interview-qa.md) disaster: 1000 sessions at 4.29 GB is 4.29 TB
against a 640 GB H100 node. The build closes the gap from both sides. FP8 plus
the shared prefix cuts the per-session footprint to ~1.9 GB, and after model
weights (14 GB in FP16) and runtime overhead the node keeps roughly 560 GB for
KV, about 290 fully-grown 32k sessions resident at once (Illustrative). The
remaining factor comes from the workload itself: most of the 500 to 1000
sessions are idle between user turns, so idle sessions are LRU-evicted and
resumed through the per-turn prefix cache instead of a full re-prefill. The
128k roadmap re-runs this arithmetic 4x worse, which is why MLA (a further ~4x
under GQA, [section 3](03-shrinking-the-cache.md)) is queued behind it rather
than dismissed.

**First-token latency.** Prefill is compute-bound and scales with prompt
length ([section 2](02-the-cost-model.md)). On a prefix-cache hit the 4k system
prompt costs nothing and only the per-session suffix is prefilled; on a cold
32k prompt, chunked prefill bounds peak memory at the cost of serializing the
pass, so the chunk size is tuned against the 2 s budget rather than set blindly.

**What breaks in month one.** Three failure modes dominate early operations, so
wire their signals before launch: cluster prefix hit rate (the per-node cache
problem from [section 6](06-serving-and-scaling.md); a hit rate far below the
single-node figure means routing, not caching, is broken), long-context recall
under FP8 (the needle-in-a-haystack grid from [section 5](05-long-context.md),
because perplexity will look fine while mid-depth retrieval quietly degrades),
and preemption of live sessions under the 3x spikes (returning users whose
cache was evicted mid-conversation pay a surprise re-prefill, so track evicted
in-flight sequences and TTFT for resumed sessions separately from cold ones).

## The same techniques under different constraints

The review question that matters in practice is not "which cache trick is best"
but "which cache trick is best under my constraints." Here is the same serving
problem built three times. Only the middle column is the build above; the other
two keep the identical levers and swap nearly every setting.

| | Fixed checkpoint, one 24 GB GPU | RAG chatbot at scale (this chapter) | Consumer chat at extreme cost targets |
|---|---|---|---|
| Model control / workload | Third-party open weights, no retraining; a handful of concurrent users | Own weights; 500 to 1000 sessions per node, 32k to 128k context | Own weights; massive chat traffic, 100+ turn histories, cost is the product constraint |
| Attention variant | Whatever the checkpoint shipped (usually GQA); not a lever you hold | GQA 8 heads uptrained; MLA queued for 128k | MQA plus cross-layer KV sharing trained in (the Character.AI profile, [section 7](07-how-teams-do-it-in-production.md)) |
| KV precision | INT4 per-token with a full-precision recent window; the only architectural lever left | FP8, keys above values, eval-gated | Native int8 from training, so no post-hoc quantization loss |
| Memory management | Paged blocks (free with any modern engine) | Paged, copy-on-write, admission cap plus LRU of idle sessions | Paged plus hybrid local/global sliding-window layers to bound growth |
| Prefix reuse | Only if prompts actually repeat; a personal assistant often has no shared prefix | 4k system prompt cached fleet-wide, cache-aware routing | Rolling-hash LRU tree over conversation turns, ~95% hit rate |
| Context policy | Stay inside the checkpoint's trained window; no fine-tuning budget to extend it | YaRN to 128k with a short fine-tune; whole-document recall required, so no eviction | Sliding window plus sinks; forgetting the deep past of a casual chat is acceptable |
| Batching / decode | Low batch, so speculative decoding is the rare big win here | Continuous batching at high batch; speculation off | Continuous batching at extreme batch; every lever stacked and eval-gated |
| What would be over-engineering | Uptraining, cache-aware routing, distributed anything | MQA (quality risk unneeded), INT2 KV | Whole-document recall machinery; the workload does not need it |

Two lessons fall out. First, the left column is defined by what you cannot
touch: with a fixed checkpoint, the train-time rows (GQA ratio, MLA, native
int8) vanish and the entire game is serving-time levers, which is exactly the
[section 3](03-shrinking-the-cache.md) divide between architecture and
quantization. Second, the right column shows what changes when quality headroom
exists and cost is the wall: MQA and aggressive windowing are wrong for the RAG
chatbot because retrieval demands exact recall, and right for casual chat
because it does not. The lever list never changed; the workload's tolerance for
loss did.

## What each constraint decides

The compressed decision guide. Read the left column off your requirements; the
right columns say which lever it moves before you compare any engines.

| Your constraint | Lever it moves | Rule of thumb |
|---|---|---|
| Model control | Attention variant | Own weights: GQA by default, MLA when the cache is still the wall. Fixed checkpoint: KV quantization is the only architectural lever left |
| Context-length target | Position scheme, prefill policy | 2x to 4x past training: PI plus fine-tune. 4x to 16x: YaRN. Prompts that OOM prefill: chunk, and tune chunk size against the TTFT budget |
| Concurrency target | Batching, memory management | Past a handful of concurrent sequences, continuous batching plus paging are mandatory, and they ship together |
| Inter-token budget | Cache size ($h_{\text{kv}}$, $b$) | Decode is bandwidth-bound; shrink bytes read per step, do not add FLOPs |
| First-token budget | Prefix caching, chunk size | A prefix hit skips prefill entirely; put stable content first, byte-identical, or the cache never fires |
| Shared content across requests | Prompt layout | System prompt and shared documents before any per-user token; one early variable token defeats the cache from that point on |
| Quality floor on retrieval | KV bits, eviction family | Gate on needle-in-a-haystack recall, not perplexity; quantize keys less than values; never lossy eviction for whole-document recall |
| Hard memory ceiling | Eviction family | Forgetting acceptable: sinks plus sliding window. Must stay exact: query-aware sparsity keeps everything and reads less |
| Multi-node fleet | Routing | Per-node prefix caches fragment at scale; cache-aware routing or a distributed cache recovers the hit rate |
| Spiky traffic | Admission and eviction policy | Cap admissions, LRU-evict idle sessions only, protect in-flight decodes; never preempt a live conversation to admit a new one |

## The smallest runnable KV cache

The review of every serving-engine tutorial is the same: the reader installs
CUDA, vLLM, and a model checkpoint and still cannot see the two numbers that
decide everything. So here are both in one file with zero installs. Part 1 is
[section 2](02-the-cost-model.md)'s size formula evaluated for all four
attention variants of [section 3](03-shrinking-the-cache.md). Part 2 is a toy
allocator race from [section 4](04-paged-and-shared.md): the same 60 GB KV
budget (Illustrative) filled by naive contiguous reservation versus fixed-size
paged blocks, on a seeded mix of session lengths. The shape is the lesson;
every section of this chapter moves one constant in this file.

```python
"""KV-cache arithmetic and a toy paged allocator, runnable with no installs."""
import random

GB = 1e9

def kv_bytes_per_token(n_layers, h_kv, d_head, bytes_per_elem):
    """Section 2 formula without S and B: 2 (K and V) x L x h_kv x d_head x b."""
    return 2 * n_layers * h_kv * d_head * bytes_per_elem

def mla_bytes_per_token(n_layers, d_c, bytes_per_elem):
    """MLA caches one latent of size d_c per layer instead of full K and V."""
    return n_layers * d_c * bytes_per_elem

# --- part 1: the size formula for a 7B-class model (L=32, d_head=128, FP16) --

L, D, FP16 = 32, 128, 2
variants = {
    "MHA (h_kv=32)": kv_bytes_per_token(L, 32, D, FP16),
    "GQA (h_kv=8) ": kv_bytes_per_token(L, 8, D, FP16),
    "MQA (h_kv=1) ": kv_bytes_per_token(L, 1, D, FP16),
    "MLA (d_c=512)": mla_bytes_per_token(L, 512, FP16),
}
S = 32768  # the chapter's 32k session
print(f"KV bytes per token, and one {S}-token session:")
for name, per_tok in variants.items():
    ratio = per_tok / variants["MHA (h_kv=32)"]
    print(f"  {name}: {per_tok:>7,} B/token  x {S} tokens = "
          f"{per_tok * S / GB:5.2f} GB  ({ratio:.1%} of MHA)")

# --- part 2: contiguous reservation vs paged blocks under one HBM budget ----

BUDGET = 60 * GB      # KV budget: one 80 GB GPU minus weights and overhead
MAX_CTX = 32768       # every request may grow to the 32k cap
PER_TOK = variants["GQA (h_kv=8) "]
BLOCK = 16            # tokens per block, as in PagedAttention

random.seed(0)
def sample_len():
    # RAG chatbot mix: 4k shared system prompt plus a long tail of history
    return min(MAX_CTX, 4096 + int(random.expovariate(1 / 6000)))
lengths = [sample_len() for _ in range(2000)]

# contiguous: every arrival reserves the full max-context buffer up front
contig_cap = int(BUDGET // (MAX_CTX * PER_TOK))
used = sum(lengths[:contig_cap]) * PER_TOK
reserved = contig_cap * MAX_CTX * PER_TOK
print(f"\ncontiguous: {contig_cap:>3} sequences fit; "
      f"{1 - used / reserved:.0%} of reserved bytes sit unused")

# paged: allocate ceil(len/BLOCK) blocks on demand from one shared pool
pool = int(BUDGET // (BLOCK * PER_TOK))
admitted = wasted_toks = used_toks = 0
for n in lengths:
    blocks = -(-n // BLOCK)          # ceil division
    if blocks > pool:
        break
    pool -= blocks
    admitted += 1
    used_toks += n
    wasted_toks += blocks * BLOCK - n
print(f"paged:      {admitted:>3} sequences fit; "
      f"{wasted_toks / (used_toks + wasted_toks):.2%} of allocated bytes sit unused")
print(f"paged admits {admitted / contig_cap:.1f}x more concurrent sequences "
      f"from the same {BUDGET / GB:.0f} GB")
```

Run it and part 1 prints the chapter's own numbers: 512 KB per token for MHA
against 128 KB for GQA (25.0%), 16 KB for MQA (3.1%), and 32 KB for MLA's
latent (6.2%), so one 32k session costs 17.18 GB, 4.29 GB, 0.54 GB, and
1.07 GB respectively; the GQA figure is exactly the per-session cost used in
the build above. Part 2 then shows why paging ships everywhere: contiguous
reservation fits 13 sequences with 70% of its reserved bytes sitting unused,
while paged blocks fit 40 sequences wasting 0.07%, a 3.1x concurrency gain
inside [section 4](04-paged-and-shared.md)'s 2x-to-4x claim. The toy reserves
the full 32k cap per sequence, which is the naive worst case; real contiguous
allocators that reserve less still lose the 20% to 40% the section quotes. Each
toy piece stands in for one production component: `kv_bytes_per_token` is the
sizing formula every capacity plan starts from, the block pool and ceil-divide
allocation are vLLM's block table without the attention kernel, the seeded
length mix is the variable-length traffic that makes fragmentation bite, and
the one thing the toy deliberately omits, block sharing between sequences, is
prefix caching, which would let all 2000 sequences point at the same 256
blocks of system prompt.
