# 10. Putting it together: the complete build

Sections 1 through 6 taught each stage with its options and tradeoffs; section 7
showed where real teams diverge. What none of them show is a single system with
every decision made. This capstone does three things: it gives you an opinionated
default stack so option paralysis never blocks a first build, it walks the
chapter's scenario end to end with every choice committed and costed, and it
shows how the same decisions flip when the constraints change. It closes with
the smallest runnable image-token budget calculator, one file, no installs.

## The default stack: start here, deviate with reason

Every stage in this chapter has three to six credible options, and a first-time
builder can burn a week comparing connectors before serving a single image.
Skip that. The stack below is a sane default for a first production build; each
row names when to deviate and which section explains why. Models change yearly,
but the interface of each stage (encode, project, interleave, decode, evaluate)
does not, so pick per stage by interface and treat any specific model as
replaceable.

| Stage | Default | Deviate when | Why (section) |
|---|---|---|---|
| Fusion strategy | Late fusion: pre-trained encoder plus pre-trained LLM, glued by a trained projector | The product must generate images, not just read them: early fusion, and accept the pretraining bill | [4](04-model-choices.md) |
| Vision encoder | Frozen CLIP ViT-L/14 class (or SigLIP where available) | Target domain is far from the encoder's pretraining (documents, medical, satellite): unfreeze or add adapters | [4](04-model-choices.md) |
| Connector / projector | MLP projector, one decoder token per patch | Per-request cost must be strictly bounded regardless of resolution: Perceiver / Q-Former resampler | [3](03-the-projector-and-tokens.md) |
| Resolution and tiling | Fixed 336px, no tiling; downscale at the gateway | Task needs OCR, charts, or dense text: 1024px with tiling and tile tags, per request, not globally | [3](03-the-projector-and-tokens.md), [1](01-clarifying-requirements.md) |
| Image-token budget | Run the token formula before deploying; cap tokens per request | Never skip the formula. "An image is one token" is the mistake this chapter exists to kill | [3](03-the-projector-and-tokens.md), [8](08-interview-qa.md) |
| Serving layout | Two tiers: batchable DP encoder, TP decoder with continuous batching; text-only bypass | Traffic is nearly 100 percent image-bearing and low volume: one server is simpler and fine | [6](06-serving-and-scaling.md) |
| Caching | Encoder embeddings keyed by image content hash; image hash folded into the prefix-cache key | Traffic is long-tail unique images: the cache buys little, spend elsewhere | [6](06-serving-and-scaling.md) |
| Evaluation | VQAv2 soft accuracy plus POPE adversarial F1, with TTFT and cost per request tracked per resolution tier | Never. Accuracy without the serving-cost check ships something correct but unaffordable | [5](05-evaluation.md) |

The last row is the one beginners skip and regret: an offline accuracy gain that
doubles the image-token budget quadruples prefill compute, and nothing in the
benchmark harness will tell you. Track TTFT at serving resolution from day one.

## The complete build

Return to the scenario from [section 1](01-clarifying-requirements.md): a visual
question answering service, one image per request up to 1024x1024, a streamed
text answer, first-token latency under 2 seconds, and a mixed workload where 30
percent of requests carry an image and 70 percent are text-only. General visual
understanding is the task; dense text and charts are a future concern. Here is
the whole system with every choice committed and the reason it won.

| Decision | Choice | Why it won |
|---|---|---|
| Fusion | Late fusion | Read-only VQA; reusing a pre-trained encoder and LLM delivers the capability for a fraction of the training cost |
| Vision encoder | Frozen CLIP ViT-L/14 at 336px | The task is natural-image QA; a strong pre-trained backbone beats training from scratch on this budget |
| Projector | MLP, tapping the penultimate encoder layer | Detail scales with cost at an affordable 576 tokens; the final CLIP layer sheds the local detail the decoder needs |
| Resolution policy | Downscale every upload to 336px at the gateway | General QA does not need native 1024px; the senior move is resolution per task, not max by default |
| Token budget | 576 image tokens per request, hard cap enforced before the encoder | One oversized upload must not OOM the encoder or starve the batch |
| Serving layout | Separate encoder tier (data parallel) and decoder tier (tensor parallel, continuous batching) | The encoder is a stateless batchable pass at a few percent of parameters; TP on it wastes sync without saving compute |
| Routing | Text-only requests bypass the vision tier entirely | 70 percent of traffic pays nothing for image infrastructure; structural win, not an optimization |
| Caching | Encoder embeddings by content hash; hash folded into the prefix-cache key | Repeat images skip encoding; the hash in the KV key stops two images with identical placeholders from colliding |
| Evaluation | VQAv2 soft accuracy, POPE adversarial F1, TTFT and cost per resolution tier | VQA rewards confident answers; POPE catches the confident-but-invented ones; TTFT keeps the 2-second contract honest |

**Image-token count.** At the committed 336px with 14-pixel patches, the patch
grid is 24x24 = 576 tokens ([section 3](03-the-projector-and-tokens.md)). The
alternative nobody should default into: native 1024px at patch 16 is 64x64 =
4096 tokens, 7x more, for a task that does not read fine print. The 576-token
choice is the single decision that makes every downstream number work.

**Prefill cost per image.** An image request carries roughly 576 image tokens
plus a ~64-token question, near 640 tokens of prefill, about 21x the sequence
length of a 30-token text-only request. Had we served native 1024px, the same
request would be 4096-plus tokens, the 130x blowup from
[section 1](01-clarifying-requirements.md), and prefill compute grows with the
square of sequence length, so the latency gap is far worse than the token ratio.

**KV cache per session.** In the reference 32-layer GQA decoder from
[section 3](03-the-projector-and-tokens.md) (8 KV heads, head dimension 128,
fp16), 576 image tokens add roughly 72 MB to the KV cache per request, against
roughly 512 MB at 4096 tokens. That 7x saving is what lets the decoder tier hold
several times more concurrent image sessions per GPU.

**Latency.** Illustrative, consistent with the [section 6](06-serving-and-scaling.md)
breakdown: tens of milliseconds to validate and downscale at the gateway, tens
of milliseconds for the encoder pass (zero on a cache hit), then prefill over
~640 tokens and the first decoded token. At 336px prefill and decode are
roughly balanced and TTFT lands well under one second, leaving half the
2-second budget as headroom for queueing, cold caches, and multi-turn prefixes.

**Cost per request.** Illustrative, at \$0.25 per million input tokens and \$1.25
per million output tokens: ~640 input tokens plus a ~150-token answer is about
\$0.0004 per image request, and the 70 percent text-only majority costs a tenth
of that. The number worth internalizing is the counterfactual: serving native
1024px multiplies the image-side input bill by more than 6x and the prefill
latency by far more, for zero measured quality gain on general VQA. The
resolution policy is not a quality compromise; it is the component that pays
for itself.

**What breaks in month one.** Three failure modes dominate early operations, so
wire their signals before launch: fine-detail hallucination (users upload
receipts and charts despite the general-QA framing, and at 336px the model
cannot read them, so it guesses; a rising flagged-answer rate on text-in-image
uploads is the [POPE-style](05-evaluation.md) signal to watch), token-budget
blowup (the first teammate who raises the resolution cap "just for one
customer" quadruples prefill; alert on TTFT p99 per resolution tier, not
globally), and encoder-decoder coupling (if the two tiers share GPUs or one
queue, large image requests head-of-line block the 70 percent text majority,
and text-only p99 rising with image traffic is the tell that the
[serving split](06-serving-and-scaling.md) has quietly eroded).

## The same techniques under different constraints

The review question that matters in practice is not "which connector is best"
but "which connector is best under my constraints." Here is the same three-stage
pipeline built three times. Only the middle column is the build above; the other
two keep the identical encoder-projector-decoder interfaces and swap nearly
every implementation choice.

| | Consumer photo QA (this chapter) | Invoice and document assistant | Catalog batch captioning |
|---|---|---|---|
| Task / traffic | General VQA; 30% image, 70% text-only; interactive | OCR-grade reading of dense pages; nearly every request carries a document | Millions of product photos captioned nightly; no user waiting |
| Latency budget | TTFT < 2s | Seconds tolerated; correctness of small print is the bar | None; throughput and cost per thousand images only |
| Encoder + resolution | Frozen CLIP class, fixed 336px downscale | Native or tiled 1024px with tile tags; unfreeze or adapt the encoder for documents | Fixed low resolution; captions need gist, not glyphs |
| Connector / token budget | MLP, 576 tokens | MLP with tiling; thousands of tokens per page, accepted | Resampler (Q-Former / Perceiver class), fixed 32-64 tokens |
| Serving | Two tiers, text-only bypass, embedding cache | Single queue is fine; the win is capping tiles per page | Huge batches on spot capacity; embedding cache pays hard because catalog images repeat |
| Eval | VQAv2 + POPE F1, TTFT per tier | DocVQA (ANLS) + ChartQA (relaxed) + TextVQA; exact match punishes OCR slips | Sampled human or LLM-judge caption review; cost per thousand images |
| What would be over-engineering | Tiling, native resolution, image generation | A resampler: the fixed cap deletes exactly the detail the product sells | Two-tier serving, streaming, TTFT dashboards, high resolution |

Two lessons fall out. First, the document column inverts the chapter's headline
economy: where the photo-QA build spends its effort keeping tokens down, the
document build deliberately spends tokens (tiling, tile tags, native
resolution) because sub-word detail is the product, and its eval switches to
metrics that tolerate OCR slips instead of the soft-voting VQA family. Second,
the batch column shows latency and cost trading places as the binding
constraint: with nobody waiting, the resampler's fixed 32-64 tokens make every
image cost the same small amount, the batches grow to whatever the hardware
takes, and the embedding cache does its best work because a catalog re-serves
the same images daily.

## What each constraint decides

The compressed decision guide. Read the left column off your requirements; the
right columns say which lever it moves before you compare any models.

| Your constraint | Lever it moves | Rule of thumb |
|---|---|---|
| Detail the task must recover | Connector and resolution together | Gist: resampler or low-res MLP. Rich understanding: MLP at 336px. OCR and charts: tiling with tile tags at ~1024px |
| First-token budget | Image-token count | Tokens drive prefill quadratically; cut resolution or cap the connector before touching the decoder |
| Mixed image / text traffic | Routing and tiering | Any meaningful text-only share: bypass the vision tier; separate queues so images never block text |
| Image repetition | Encoder embedding cache | Catalogs, multi-turn chat over one image: cache by content hash; long-tail uploads: skip it |
| Multi-turn chat over one image | Prefix-cache key | Fold the image hash into the KV prefix key, or two images with identical placeholders collide |
| Multi-image requests | Per-request token cap | k images stack linearly; cap images per request or compress the extras with a resampler |
| Cost per request | Token budget first, model tier second | Halving image tokens roughly halves the image-side bill before you shop for a cheaper model |
| Encoder share of parameters | Encoder parallelism | At a few percent of params, DP replicas beat TP sharding; one all-gather replaces per-layer all-reduces |
| Aspect-ratio diversity | Resolution policy | Screenshots and pages at wild ratios: dynamic native resolution; uniform photos: a fixed crop is cheaper |
| Must generate images | Fusion strategy | Only early fusion can emit visual tokens; read-only products should not pay its pretraining bill |

## The smallest runnable token budget

The review of every VLM postmortem is the same: someone shipped a resolution or
tiling change without running the token math, and the bill or the TTFT graph
found it first. So here is the entire cost model in one file with zero installs.
Every production component is swapped for the smallest thing with the same
interface: the tiling policy becomes a ceiling division, the projector becomes
an integer pooling factor, and the serving bill becomes a per-page token count
at an illustrative price. The shape is the lesson: resolution drives tokens
quadratically, tiling quantizes the cost into steps, and the connector claws it
back.

```python
"""Image-token budget calculator: resolution -> tiles -> tokens -> prefill bill."""
import math

PATCH = 16            # pixels per patch side (Qwen2-VL / Pixtral class)
TILE = 512            # tiling policy: images above this side are split into tiles
POOL = 4              # projector pooling factor (2x2 patch merge); 1 = plain MLP
TEXT_TOKENS = 60      # text prompt tokens accompanying the images
IMAGES_PER_PAGE = 4   # images in one document page / request
PRICE_IN = 0.25       # illustrative $ per million input tokens

def tiles_for(H, W, tile=TILE):
    """How many tile crops the tiling policy produces; 1 if the image fits."""
    return math.ceil(H / tile) * math.ceil(W / tile)

def tokens_per_tile(H, W, patch=PATCH, tile=TILE):
    """Patch grid of one crop: the full image if it fits, else a full tile."""
    side = min(max(H, W), tile)
    return (side // patch) ** 2

def image_tokens(H, W, patch=PATCH, tile=TILE, pool=1):
    """Raw decoder tokens for one image under the tiling policy, then pooled."""
    raw = tiles_for(H, W, tile) * tokens_per_tile(H, W, patch, tile)
    return raw // pool

def page_bill(H, W, pool):
    """Prefill tokens and cost for a page of IMAGES_PER_PAGE images plus text."""
    toks = IMAGES_PER_PAGE * image_tokens(H, W, pool=pool) + TEXT_TOKENS
    return toks, toks * PRICE_IN / 1e6

def main():
    resolutions = [336, 512, 672, 1024]
    base = image_tokens(resolutions[0], resolutions[0])
    hdr = (f"{'res':>5} {'tiles':>5} {'raw tok':>8} {'pooled':>7} "
           f"{'vs 336':>7} {'page tok':>9} {'page $':>9}")
    print(hdr)
    print("-" * len(hdr))
    for r in resolutions:
        raw = image_tokens(r, r)
        pooled = image_tokens(r, r, pool=POOL)
        toks, cost = page_bill(r, r, pool=1)
        print(f"{r:>5} {tiles_for(r, r):>5} {raw:>8} {pooled:>7} "
              f"{raw / base:>6.1f}x {toks:>9} {cost:>9.4f}")
    print()
    toks_raw, cost_raw = page_bill(1024, 1024, pool=1)
    toks_p, cost_p = page_bill(1024, 1024, pool=POOL)
    print(f"1024px page, plain MLP projector : {toks_raw:>6} prefill tokens  ${cost_raw:.4f}")
    print(f"1024px page, {POOL}x pooled projector : {toks_p:>6} prefill tokens  ${cost_p:.4f}")
    print(f"resolution 336 -> 1024 is {1024/336:.1f}x the side, "
          f"{image_tokens(1024,1024)/base:.1f}x the tokens (quadratic); "
          f"pooling claws back {POOL}x")

if __name__ == "__main__":
    main()
```

Run it and the table makes the chapter's three cost claims concrete in about
sixty lines. A 336px image at patch 16 is 441 tokens while 1024px is 4096, the
same number [section 3](03-the-projector-and-tokens.md) derives for the
Pixtral-class grid: 3x the side, 9.3x the tokens, the quadratic in action. The
tiles column shows the tiling policy quantizing cost into steps: a 672px upload
spills into the same four padded tiles as a full 1024px page and pays the
identical 4096-token bill, which is why resolution caps belong at tile
boundaries. And the pooled column shows the connector clawing it back: a 4x
patch-merge projector turns a four-image 1024px page from 16,444 prefill tokens
(about \$0.0041, illustrative) into 4,156 (about \$0.0010), the same lever a
resampler pulls harder with a fixed cap. Change POOL to 1, 4, and 16 and you
are walking the [connector tradeoff curve](03-the-projector-and-tokens.md) from
MLP toward Q-Former; every section of this chapter is a policy for one constant
in this file.
