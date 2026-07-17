# 8. Interview Q&A

The questions an interviewer actually asks about multimodal serving, grouped by
how they are used. The commonly-missed ones are where interviews are won or lost.

## Commonly asked

**Q: Why is serving a vision-language model more expensive than serving a text-only
LLM?**
A: An image becomes hundreds or thousands of tokens that land in the most expensive
part of the pipeline. A 1024x1024 image with 16-pixel patches produces 4096
tokens. Those tokens enter the LLM prefill pass (compute scales quadratically with
sequence length) and inflate the KV cache at every decoder layer. A text-only
question might be 30 tokens; adding one high-resolution image makes that request
over 130x more expensive at prefill. The serving cost is not about the image; it
is about the image tokens.

**Q: What are the three components of a vision-language model?**
A: Vision encoder (ViT-style: produces one feature vector per image patch),
projector or connector (maps encoder features into the LLM's token embedding
space, setting the image-token budget), and LLM decoder (consumes the interleaved
image-plus-text token sequence and generates autoregressively). The encoder is
batchable and cacheable; the decoder is autoregressive and memory-bound. Keeping
this distinction clear drives the whole serving architecture.

**Q: How do you keep first-token latency under 2 seconds for image requests?**
A: Lower the image-token count (use a lower resolution or a fixed-cap connector
like a resampler), cache encoder output so repeated images skip encoding, and run
the encoder as a separate batchable tier so it does not share GPU resources with
the autoregressive decoder. For a mixed workload, also route text-only requests
past the encoder entirely.

**Q: What happens when a user uploads a very high-resolution image?**
A: Validate and cap dimensions before encoding. A 4000x3000 image must be
downscaled to the system's resolution cap before the encoder sees it, or the
token count OOMs the encoder GPU and slows every other request in the batch. The
policy is a product decision: for general QA, scale to 512px or 768px; for
document OCR, scale to 1024px and accept the higher cost.

**Q: What is the role of the projector, and why does it matter for serving?**
A: The projector maps encoder features into the LLM's embedding space and sets the
image-token count. An MLP projector passes one token per encoder patch (detail
scales with cost); a resampler or Q-Former compresses to a fixed few tokens (cost
is bounded but detail is capped). Picking the projector is picking the
quality-cost operating point.

## Tricky

**Q: Offline VQA accuracy improved with a higher-resolution encoder, but online
TTFT went from 0.8s to 3.2s. What happened, and how do you fix it?**
A: Doubling resolution quadruples the token count (quadratic in image side), which
makes prefill over the longer sequence disproportionately slower. A common mistake
is evaluating accuracy without measuring latency at serving resolution. Fix: use
tiling selectively only for tasks that need fine detail, and expose resolution as a
per-request or per-task knob rather than a global constant. Serve general QA at
336px and OCR requests at 1024px with tiling.

**Why the slowdown is worse than 4x:** token count grows quadratically with image
side, and self-attention cost grows quadratically with token count, so doubling the
resolution multiplies the attention FLOPs by up to 16x. The offline eval never
surfaced this because accuracy was measured per example with no latency budget; the
regression only exists as a function of sequence length at serving time.

**Q: Should you cache raw image bytes, encoder embeddings, or decoder KV entries?**
A: Cache encoder embeddings (not raw bytes, which require re-encoding). KV entries
are too large and session-specific. Encoder output is deterministic given the
image and reusable across any prompt about the same image. Key the cache by
image content hash, not by filename or URL, so identical images with different
names still hit the cache. vLLM V1 also folds the image hash into the prefix
cache key so KV recompute is avoided on multi-turn conversations about one image.

**Why the three cache candidates differ:** each sits at a different point in the
dependency chain. Raw bytes depend on nothing but save nothing (the expensive
encode still runs). Encoder embeddings depend only on the pixels, so they are valid
for any prompt, any user, any session. KV entries additionally depend on every
token before them in the sequence, including positions and the surrounding prompt,
so they are only reusable when the entire prefix matches, which is why they belong
to the prefix cache rather than a standalone image cache.

**Q: Your prefix cache for text prompts is already working. Why does it break on
multimodal input?**
A: Text prefix caching uses token IDs as the cache key. In multimodal models, an
image placeholder is a fixed set of special tokens that look identical regardless
of which image is inserted. Two different images produce the same placeholder
token sequence, so a naive prefix cache silently returns cached KV entries from
the wrong image. The fix is to fold the image content hash into the cache key.

**Q: When would you use a resampler instead of an MLP projector?**
A: When per-request cost and latency must be strictly bounded regardless of input
resolution. A resampler compresses the encoder's variable-length feature grid to
a fixed-size token block (BLIP-2 uses 32, Idefics2 uses 64 or 320). The cost is
that dense content, OCR, and fine geometric detail are partially lost because the
resampler must discard information to hit the fixed budget. Use a resampler when
the task is general-understanding and the budget is tight; use an MLP projector
when the task needs fine detail and you can afford variable token counts.

**Why the fixed budget loses detail:** the resampler works by cross-attending a
small set of learned query tokens over the full patch grid, so everything the
decoder will ever know about the image must pass through that fixed-width
bottleneck. Information theory does the rest: a 4096-patch grid summarized into 64
vectors cannot preserve every glyph and edge, so the queries learn to keep what the
training tasks rewarded, which is global semantics rather than dense text.

**Q: You cache encoder embeddings by content hash, but two users crop or re-save
the same photo slightly and get zero cache hits. Is the cache broken?**
A: No, it is working exactly as designed, and that is the point. A content hash is
intentionally brittle: it hits only on byte-identical (or pixel-identical, if you
hash the decoded pixels) inputs, because the encoder output is only guaranteed
reusable when the input is truly the same. A re-crop, a re-compression, or an EXIF
rewrite produces a different pixel grid and therefore a legitimately different
embedding, so a hit there would return stale features and silently corrupt the
answer. If you want near-duplicate reuse you cannot get it from hashing; you would
need perceptual hashing or an embedding-similarity lookup, which trades exactness for
recall and reintroduces the risk of serving the wrong image's features. The
interview signal is recognizing that the "miss" is the cache refusing to be wrong,
not the cache failing. The right lever for raising hit rate is normalizing inputs
upstream (fixed downscale, strip metadata, canonical re-encode) so genuinely
equivalent uploads converge to the same bytes before hashing.

**Q: How do you scale the vision encoder and the LLM decoder independently?**
A: Run them as separate services. The encoder is a bounded, batchable, stateless
pass that can be scaled horizontally by replicating encoder pods. The decoder is
the autoregressive, memory-bandwidth-bound pass that needs GPU memory for the KV
cache and scales differently. The encoder output (a fixed-size embedding block)
is passed to the decoder tier via a lightweight message. This separation also
allows DP on the encoder and TP on the decoder independently.

**Why the two tiers scale so differently:** the encoder is a stateless feed-forward
pass, so throughput scales almost linearly with replica count and any replica can
serve any request. The decoder holds per-session KV state that must stay resident
on the GPU where the sequence lives, so its capacity is bounded by HBM and requests
are sticky to their replica. Mixing them in one service forces the stateless,
compute-hungry phase and the stateful, memory-hungry phase to share one scaling
policy, which fits neither.

**Q: Tiling and a native higher-resolution encoder look similar; both feed the model
more pixels. When does the difference actually matter?**

A: The mechanics differ in where the extra pixels meet attention. A native
high-resolution encoder runs self-attention over the full patch grid, so every patch
can attend to every other patch: global layout is preserved, but cost grows
quadratically in the grid size and the encoder must be trained at that resolution.
Tiling reuses a fixed-resolution encoder by slicing the image into crops and
encoding each independently: cost grows only linearly with tile count and no
retraining is needed, but a patch in one tile can never attend to a patch in
another, so structures that span tile boundaries (a table row, a diagram arrow) are
fragmented and must be re-stitched by the decoder, usually with tile-position tags
and a low-resolution thumbnail as a global map. The difference matters exactly when
the task needs cross-region geometry: reading a full-page table or chart favors
native resolution (or careful tile tagging), while spotting localized fine detail
like small print tolerates tiling and gets it far cheaper.

## Commonly answered wrong

**Q: Does the image go into the LLM's context as a single special token?**
A: No. The image becomes a block of tokens, one per encoder patch (for an MLP
projector), or a fixed set for a resampler, but never one token. Treating an
image as a single token is the core mistake that leads to wrong cost estimates.
A 336px image is roughly 576 tokens; a 1024px image can be 4096. These are the
numbers to have on hand.

**Why one token cannot work:** the decoder can only use information that is present
in the token sequence, and a single embedding vector has nowhere near the capacity
to represent a scene at answerable detail. One token per patch keeps a spatial
correspondence the attention mechanism can exploit (the model can attend to the
region containing the answer), which is also why cost estimates based on "an image
is one token" are off by three orders of magnitude.

**Q: Is image-token caching done by filename or URL?**
A: By content hash. Filenames and URLs are not stable; the same image can arrive
with different names or from different URLs. Content hashing ensures that identical
bytes always hit the cache, and different images never collide even if their URLs
match. This is subtle but matters in practice.

**Why a collision is worse than a miss:** a miss just costs one redundant encode,
but a collision serves another image's embeddings as if they were this one's, and
the decoder happily answers about the wrong picture with no error signal anywhere
in the pipeline. Cache keys must therefore be derived from the content the cached
value actually depends on, which for encoder output is the pixels, not the name
they arrived under.

**Q: Does adding more GPU memory to the decoder server speed up image-heavy
requests?**
A: Not primarily. The bottleneck for image-heavy requests is prefill compute, which
scales quadratically with sequence length and is compute-bound, not
memory-bandwidth-bound. More GPU memory helps the KV cache (more concurrent
sessions) but does not fix high TTFT caused by long image-token prefill. Lower
the token count or parallelize prefill instead.

**Q: Can you just append image tokens after all the text tokens to keep the LLM
unchanged?**
A: Technically possible, but it hurts quality. Most VLMs interleave image tokens
at the image placeholder position in the instruction template (before or in the
middle of the question), so the model attends to the image in context with the
question. Appending image tokens at the end forces the model to first read the
question without the image, then see the image. Instruction-tuned VLMs expect
the interleaved layout they were fine-tuned on. The mechanism-level reason it hurts:
the decoder is causal, so a token only attends to tokens at or before its own
position. If the image tokens sit after the question, none of the question tokens
can attend to the image at all, and only the generated answer tokens ever see it, so
the model effectively answers a text-only question and then glances at the image
while decoding. Placing the image before the question lets every question token
attend to the visual evidence, which is why the LLaVA (2023) instruction templates
and their successors interleave image tokens at the placeholder rather than
appending them.
