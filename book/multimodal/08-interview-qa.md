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

**Q: Should you cache raw image bytes, encoder embeddings, or decoder KV entries?**
A: Cache encoder embeddings (not raw bytes, which require re-encoding). KV entries
are too large and session-specific. Encoder output is deterministic given the
image and reusable across any prompt about the same image. Key the cache by
image content hash, not by filename or URL, so identical images with different
names still hit the cache. vLLM V1 also folds the image hash into the prefix
cache key so KV recompute is avoided on multi-turn conversations about one image.

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

**Q: How do you scale the vision encoder and the LLM decoder independently?**
A: Run them as separate services. The encoder is a bounded, batchable, stateless
pass that can be scaled horizontally by replicating encoder pods. The decoder is
the autoregressive, memory-bandwidth-bound pass that needs GPU memory for the KV
cache and scales differently. The encoder output (a fixed-size embedding block)
is passed to the decoder tier via a lightweight message. This separation also
allows DP on the encoder and TP on the decoder independently.

## Commonly answered wrong

**Q: Does the image go into the LLM's context as a single special token?**
A: No. The image becomes a block of tokens, one per encoder patch (for an MLP
projector), or a fixed set for a resampler, but never one token. Treating an
image as a single token is the core mistake that leads to wrong cost estimates.
A 336px image is roughly 576 tokens; a 1024px image can be 4096. These are the
numbers to have on hand.

**Q: Is image-token caching done by filename or URL?**
A: By content hash. Filenames and URLs are not stable; the same image can arrive
with different names or from different URLs. Content hashing ensures that identical
bytes always hit the cache, and different images never collide even if their URLs
match. This is subtle but matters in practice.

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
the interleaved layout they were fine-tuned on.
