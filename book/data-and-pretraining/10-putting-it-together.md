# 10. Putting it together: the complete build

Sections 1 through 6 taught each stage with its options and tradeoffs; section 7
showed where real teams diverge. What none of them show is a single system with
every decision made. This capstone does three things: it gives you an opinionated
default stack so option paralysis never blocks a first build, it walks the
chapter's scenario end to end with every choice committed and costed, and it
shows how the same decisions flip when the constraints change. It closes with
the smallest runnable deduplicator, one file, no installs.

## The default stack: start here, deviate with reason

Every stage in this pipeline has three to six credible options, and a first-time
builder can burn a month comparing filtering recipes before producing a single
clean token. Skip that. The stack below is a sane default for a first serious
build; each row names when to deviate and which section explains why. Tools
change yearly, but the interface of each stage (extract, route, filter, dedup,
decontaminate, mix, tokenize, size, train) does not, so pick per stage by
interface and treat any specific library as replaceable.

| Stage | Default | Deviate when | Why (section) |
|---|---|---|---|
| Extraction | Re-extract from WARC with a boilerplate-stripping extractor; do not trust WET | Corpus is already clean text (books, papers): go straight to language ID | [2](02-the-data-pipeline.md) |
| Language ID | fastText-class classifier, per-language streams, drop below-confidence docs | Single-language corpus with known provenance: a spot-check suffices | [2](02-the-data-pipeline.md) |
| Quality filtering | Small ablated set of Gopher/C4 heuristics plus a FineWeb-Edu-style learned classifier | No reference corpus rich enough to train a classifier: heuristics only | [3](03-data-quality.md) |
| Deduplication | Exact hash first, then MinHash + LSH within and across dumps | Corpus is small enough for all-pairs comparison: skip the LSH machinery | [3](03-data-quality.md) |
| Decontamination | n-gram overlap (13-gram) against every eval set, rate found and reported | Never. Run it before the first training token | [3](03-data-quality.md) |
| Mixing | Domain weights with code, math, and papers upsampled; anneal to highest quality late | Single-domain corpus: there is nothing to weight | [2](02-the-data-pipeline.md) |
| Tokenizer | Byte-level BPE, 32K-64K vocabulary, fit once on the final mixture | Multilingual or whitespace-free scripts: SentencePiece, 128K+, fertility per language | [4](04-pretraining-choices.md) |
| Sizing | Spend the budget on paper with C ~ 6ND before touching any hardware | Serving-heavy: overtrain a smaller model far past 20 tokens per parameter | [4](04-pretraining-choices.md) |
| Architecture | Dense pre-norm decoder, GQA + RoPE + RMSNorm + SwiGLU | Capacity needed beyond the per-token FLOP budget: MoE with load balancing | [4](04-pretraining-choices.md) |
| Parallelism | FSDP/ZeRO sharded data parallel, bf16, frequent sharded checkpoints | One layer overflows a GPU: add in-node TP; the stack overflows: add PP | [5](05-systems.md) |

The decontamination row is the one beginners skip and regret: without it, every
benchmark number you report is potentially fiction, and you will not find out
until someone else does. It is the cheapest row in the table and the only one
with no deviation condition.

## The complete build

Return to the scenario from [section 1](01-clarifying-requirements.md): a
general-purpose English-primary base model, pretrained from scratch on roughly
$6 \times 10^{22}$ FLOPs (about 10,000 A100 GPU-days), from web crawl plus
licensed books, code, and papers, evaluated on a decontaminated MMLU / ARC /
HellaSwag / HumanEval suite, intended for heavy serving at billions of tokens
per day. Here is the whole system with every choice committed and the reason it
won.

| Decision | Choice | Why it won |
|---|---|---|
| From-scratch justification | Stated upfront; proceed only because no open base covers the target | Almost no one should pretrain from scratch; saying so unprompted is the first senior signal |
| Extraction | WARC re-extraction with boilerplate stripping and URL blocklists | Garbage extraction inflates duplicate counts and poisons every downstream filter |
| Language ID | fastText-class routing; English-primary stream plus routed multilingual slices | Per-language thresholds keep English from setting the bar that starves other scripts |
| Quality filtering | A handful of ablated Gopher/C4 rules plus an educational classifier | FineWeb's ablation: only a few rules move benchmarks; fewer better tokens beat more tokens |
| Deduplication | Exact hash, then MinHash + LSH within and across dumps, aggressiveness ablated | Near-dups differ by a timestamp, so exact alone misses most; maximal dedup lowers scores |
| Decontamination | 13-gram overlap against all four benchmarks; rate found and published | Any headline number without a decontamination claim is suspect |
| Tokenizer | Byte-level BPE, 64K vocabulary, fit once on the final mixture | English-primary with code and some multilingual: 128K would undertrain rare tokens |
| Sizing | 7B dense trained on ~1.4T tokens (about 200 tokens per parameter) | Heavy serving makes deployment-optimal beat the Chinchilla training-optimal point |
| Architecture | Dense, GQA + RoPE + RMSNorm + SwiGLU | Serving VRAM and predictable per-token cost; RoPE keeps late context extension cheap |
| Schedule | Linear warmup, cosine decay, gradient clipping at norm 1.0, muP-tuned on a proxy | The stability recipe every base run uses; tune the peak LR cheaply, scale once |
| Parallelism | FSDP (ZeRO-3-class) sharded data parallel, bf16 with fp32 master weights | Every 7B layer fits one GPU; the full optimizer footprint does not |
| Fault tolerance | Sharded async checkpoints, loss-spike rollback with batch skip, elastic restart | Weeks on hardware that fails every few hours; recovery is routine, not an incident |
| Evaluation | Bits-per-byte plus decontaminated benchmarks on a time-based split | Perplexity is tokenizer-bound; a random split leaks the future and flatters the model |

**Sizing the run.** The budget is $C \approx 6ND = 6 \times 10^{22}$ FLOPs.
Chinchilla-optimal ($D = 20N$) solves to roughly a 22B model on 450B tokens.
But [section 1](01-clarifying-requirements.md) pinned heavy serving, and
[section 4](04-pretraining-choices.md) says that flips the objective: the same
budget instead buys a 7B model on $6 \times 10^{22} / (6 \times 7 \times 10^9)
\approx 1.4$T tokens, about 200 tokens per parameter, ten times past the
Chinchilla point. Loss per training FLOP is worse; cost per served token,
forever, is roughly 3x better than the 22B alternative.

**The token supply.** Can the pipeline deliver 1.4T clean tokens? The keep rate
through the funnel is single-digit percentages of raw bytes
([section 2](02-the-data-pipeline.md)), and that is the design, not a failure.
Illustrative funnel: 1 PB of raw crawl text, ~20% surviving extraction, ~60% of
that on-target after language ID, ~30% of that past the quality filters, ~50% of
that past dedup, lands near 2% end to end, about 20 TB, roughly 5T tokens at 4
bytes per token. The educational classifier then keeps the best ~1.4T, the
FineWeb-Edu move of trading volume for quality. Precedent says this is
comfortable: FineWeb distilled 96 Common Crawl dumps to 15T tokens. And if the
supply had come up short, Muennighoff's result applies: up to roughly four
epochs of repetition is nearly as good as fresh tokens.

**GPU-time.** 10,000 A100 GPU-days at a bf16 peak of 312 TFLOP/s is
$2.7 \times 10^{23}$ peak FLOPs, so delivering $6 \times 10^{22}$ requires only
~22% MFU, below the 30-50% a well-tuned run achieves
([section 5](05-systems.md)). At 40% MFU the run needs about 5,600 GPU-days:
on 1,024 GPUs (Illustrative cluster size), under six days of training, with the
rest of the budget absorbed by ablations, restarts, and the proxy runs that
tuned the schedule. The budget that is actually tight is not FLOPs; it is the
weeks of pipeline work upstream of the first training token.

**Memory and parallelism.** Mixed-precision Adam costs 16 bytes per parameter
([section 5](05-systems.md)), so the 7B model carries a 112 GB training
footprint, over an 80 GB A100. Plain data parallelism therefore cannot
replicate it, but sharding solves it without heroics: FSDP over just 64 ranks
puts the persistent footprint near $16 \times 7\times10^9 / 64 \approx 1.75$ GB
per GPU, leaving VRAM for activations. No tensor or pipeline parallelism is
needed at 7B; the parallelism plan is the simplest one in the chapter, which is
itself a consequence of the deployment-optimal sizing decision.

**What breaks in month one.** Three failure modes dominate early operations, so
wire their signals before launch: loss spikes (gradient-norm and loss alarms
plus automated rollback-and-skip from [section 5](05-systems.md); the first one
arrives mid-run, not in a postmortem), contamination discovered late (a
benchmark that jumps after a data refresh means the decontamination pass did not
cover the new slice; rerun it and republish the rate rather than defending the
number), and dedup misses (verbatim regurgitation of boilerplate or license
spans in samples means near-duplicates survived; check the cross-dump MinHash
pass and its threshold before blaming the model).

## The same techniques under different constraints

The review question that matters in practice is not "which dedup is best" but
"which dedup is best under my constraints." Here is the same discipline applied
three times. Only the middle column is the build above; the other two keep the
identical stage interfaces and swap nearly every implementation choice.

| | Domain continue-pretrain | General 7B from scratch (this chapter) | Frontier multilingual MoE |
|---|---|---|---|
| Corpus / compute | ~30B domain tokens (Illustrative); tens of GPUs for days | ~1.4T tokens; 10,000 A100 GPU-days ($6 \times 10^{22}$ FLOPs) | ~15T tokens; lab-scale cluster for months |
| From scratch? | No: continue-pretrain Llama 3 or OLMo | Yes: justified by a genuine capability gap | Yes: the frontier is the point |
| Tokenizer | Inherited from the base; changing it would force from-scratch | Byte-level BPE, 64K, fit once on the mixture | SentencePiece, 128K+; fertility tracked per language |
| Quality filtering | Hand curation plus a few heuristics; the corpus is small enough to inspect | Ablated heuristics plus educational classifier | Per-language CCNet-style perplexity references plus learned classifiers |
| Dedup | Exact hash plus one MinHash pass | Exact + MinHash/LSH within and across dumps, ablated | Same, with $b$ and $r$ tuned per language and aggressiveness ablated |
| Decontamination | Still mandatory: n-gram against the evals you will report | 13-gram against the standard suite, rate published | n-gram plus embedding overlap; public scores get read adversarially |
| Architecture / parallelism | The base's architecture as-is; FSDP alone | Dense GQA 7B; FSDP, bf16, no TP or PP | MoE (DeepSeek-V3 class: 671B total, 37B active); TP + PP + EP + ZeRO, FP8 |
| What would be over-engineering | A new tokenizer, cross-dump dedup machinery, any MoE | FP8, expert parallelism, a 128K vocabulary | Maximal global dedup; scaling dense to equal capacity |

Two lessons fall out. First, the left column is where most readers actually
live, and it is mostly deletions: [section 1](01-clarifying-requirements.md)'s
"almost no one should pretrain from scratch" lands here as a concrete build
that inherits the tokenizer, the architecture, and the parallelism plan, and
spends all of its effort on curation. The one row that never shrinks is
decontamination. Second, the right column shows the bottleneck migrating: at
frontier scale data work is table stakes and the binding constraints become
interconnect and precision, which is why FP8 and expert parallelism appear
there and nowhere else.

## What each constraint decides

The compressed decision guide. Read the left column off your requirements; the
right columns say which lever it moves before you compare any tools.

| Your constraint | Lever it moves | Rule of thumb |
|---|---|---|
| Compute budget | N and D jointly | $C \approx 6ND$; at fixed C, training-optimal is $D \approx 20N$ |
| Serving volume | Sizing objective | Billions of tokens per day: overtrain a smaller model far past 20 tokens per parameter |
| Unique-token supply | Repetition vs filtering aggressiveness | Up to ~4 epochs of repetition is nearly as good as fresh tokens; past that, each repeat adds almost nothing |
| Language mix | Vocabulary size and pipeline shape | Multilingual: SentencePiece at 128K+, per-language thresholds; always report fertility per language |
| Corpus scale | Dedup machinery | Small corpus: all-pairs is fine. Trillions of tokens: MinHash + LSH with the knee near $J \approx (1/b)^{1/r}$ |
| Public benchmark reporting | Decontamination depth | n-gram before the first training token, always; add embedding overlap when scores go public |
| Parameters vs GPU memory | Parallelism axes | 16 bytes/param with Adam; shard optimizer state first (ZeRO-1/2), then parameters (ZeRO-3/FSDP); TP only when one layer overflows, and only in-node |
| Run length x failure rate | Checkpoint interval | Size so mean-time-between-failures times fraction-lost is acceptable; shard the writes and make them async |
| Stability vs throughput | Precision | bf16 with fp32 master weights by default; FP8 only when interconnect-bound at frontier scale |

## The smallest runnable dedup

The review of every pipeline writeup is the same: the reader nods at "MinHash
plus LSH" and still cannot see why exact hashing is not enough. So here is the
chapter's highest-leverage stage in one file with zero installs. Every
production component is swapped for the smallest thing with the same interface:
the hash family is a salted stdlib hash, the LSH buckets become a direct
comparison (fine at toy scale), and the corpus is seven documents containing
one exact duplicate, two near-duplicates with small edits, and three unique
documents. The shape is the lesson; [section 3](03-data-quality.md) upgrades
every function of this file.

```python
"""Exact + MinHash near-duplicate dedup in one file, runnable with no installs."""
import hashlib

def shingles(text, n=3):
    """Overlapping word n-grams; production: 5-grams over normalized text."""
    words = text.lower().split()
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}

def h(seed, shingle):
    """One of k independent hash functions, made by salting a stdlib hash."""
    return int.from_bytes(hashlib.md5(f"{seed}:{shingle}".encode()).digest()[:8], "big")

def minhash(shingle_set, k=64):
    """Keep the minimum of each salted hash; production: MinHash + LSH banding."""
    return [min(h(seed, s) for s in shingle_set) for seed in range(k)]

def estimate_jaccard(sig_a, sig_b):
    """Fraction of matching signature entries estimates J(A, B)."""
    return sum(a == b for a, b in zip(sig_a, sig_b)) / len(sig_a)

DOCS = [
    ("web/chinchilla-1", "The Chinchilla result says parameters and tokens should "
     "grow together, at roughly twenty tokens per parameter, when minimizing loss "
     "for a fixed training compute budget, and a seventy billion parameter "
     "Chinchilla beat a far larger Gopher at equal compute."),
    ("web/chinchilla-2", "The Chinchilla result says parameters and tokens should "
     "grow together, at roughly twenty tokens per parameter, when minimizing loss "
     "for a fixed training compute budget, and a seventy billion parameter "
     "Chinchilla beat a far larger Gopher at equal compute."),  # exact duplicate
    ("web/chinchilla-3", "Updated 2024: The Chinchilla result says parameters and "
     "tokens should grow together, at about twenty tokens per parameter, when "
     "minimizing loss for a fixed training compute budget, and a seventy billion "
     "parameter Chinchilla beat a far larger Gopher at equal compute."),  # near-dup
    ("blog/dedup-1", "Near-duplicate documents dominate the web because the same "
     "page recurs across crawl snapshots with only a timestamp or header changed, "
     "so exact hashing alone removes almost none of them."),
    ("blog/dedup-2", "Near-duplicate documents dominate the web because the same "
     "page recurs across the crawl snapshots with only a timestamp or a header "
     "changed, so exact hashing alone removes almost none of them."),  # near-dup
    ("paper/tokenizer", "An English-heavy vocabulary fragments other scripts into "
     "many more tokens per word, so fertility must be reported per language."),
    ("code/readme", "This repository trains a small decoder-only transformer with "
     "warmup, cosine decay, and gradient clipping at norm one."),
]

def dedup(docs, threshold=0.7):
    kept, seen_exact = [], set()
    exact_drops = fuzzy_drops = 0
    for doc_id, text in docs:
        fingerprint = hashlib.sha256(" ".join(text.lower().split()).encode()).hexdigest()
        if fingerprint in seen_exact:                    # stage 1: exact hash
            exact_drops += 1
            print(f"drop {doc_id:<18} exact duplicate (same content hash)")
            continue
        seen_exact.add(fingerprint)
        sig = minhash(shingles(text))                    # stage 2: MinHash
        best_id, best_j = None, 0.0
        for kept_id, kept_sig in kept:                   # toy scale: compare all;
            j = estimate_jaccard(sig, kept_sig)          # production: LSH buckets
            if j > best_j:
                best_id, best_j = kept_id, j
        if best_j >= threshold:
            fuzzy_drops += 1
            print(f"drop {doc_id:<18} near-duplicate of {best_id} "
                  f"(J~{best_j:.2f}, exact hash differs)")
            continue
        kept.append((doc_id, sig))
        print(f"keep {doc_id:<18} best match J~{best_j:.2f}, below threshold")
    print("\nfunnel:")
    print(f"  raw documents       {len(docs)}")
    print(f"  after exact dedup   {len(docs) - exact_drops}  (-{exact_drops} byte-identical)")
    print(f"  after fuzzy dedup   {len(kept)}  (-{fuzzy_drops} near-duplicates exact hashing missed)")
    print(f"  keep rate           {len(kept) / len(docs):.0%}")

dedup(DOCS)
```

Run it and the output is the chapter's dedup argument in miniature. The exact
mirror (`web/chinchilla-2`) dies at stage 1 on a content hash. The two
near-duplicates sail straight past that hash, because a "Updated 2024" prefix
or an inserted article changes every byte-level fingerprint, and are caught
only by the MinHash stage, both at an estimated J of about 0.70 against their
originals, while the reworded-but-related `web/chinchilla-3` versus the unique
documents scores near zero. The funnel prints 7 raw documents, 6 after exact
dedup, 4 after fuzzy dedup, a 57% keep rate, with two of the three drops
invisible to exact hashing. Swap the salted hashes for a real MinHash family,
the all-pairs loop for LSH banding with tuned $b$ and $r$, and run it within
and across crawl dumps, and you have rebuilt the highest-leverage stage of this
chapter.
