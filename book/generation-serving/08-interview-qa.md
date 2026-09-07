# 8. Interview Q&A

## Commonly asked

**Q: What actually costs money when you generate an image?**

Forward evaluations of the denoiser. The number is $N_{\text{FE}} = S \cdot c$, with
$S$ the sampler steps and $c = 2$ under classifier-free guidance, because guidance
runs the network once conditioned and once unconditioned. Everything else in the
pipeline (text encode, VAE decode, safety classification) is paid once per request.
Quoting a step count without saying whether guidance is on is quoting half the cost.

**Q: Your product shows four variants. How much more does that cost than one?**

Almost nothing extra. The four variants are the batch dimension of the same denoising
loop, so they share the same $N_{\text{FE}}$ passes over a larger tensor. That is why
products offer a grid at all, and pricing the grid as four requests is wrong by
nearly four times.

**Q: How do you get from 30 steps to 4?**

Two stages. First a better sampler, which is free and gets you from the training
schedule to roughly 20 to 30 evaluations with no change to the weights. Below that you
have to distill: consistency distillation, its latent and LoRA forms, or adversarial
distillation for one to four steps. The second stage always costs diversity.

**Q: Why is 2048px more than four times the cost of 1024px?**

Because the latent grows quadratically and attention grows quadratically in that. A
1024px image at downsample 8 and patch 2 is 4,096 tokens; 2048px is 16,384. Four times
the tokens, sixteen times the attention work.

**Q: Where does the memory go?**

The VAE decode, not the denoising loop. Its activation memory scales with output
pixels, so out-of-memory failures appear only at high resolution and only at the end
of a request. Tiled decode bounds it.

## Tricky (the follow-ups that separate candidates)

**Q: p95 is twelve seconds and your cost model says the floor is three. Where is the
gap?**

Not in the denoiser. Look at queueing first (arrival rate against capacity, and no
admission control), then cold start if the traffic is spiky, then the pipeline around
the loop: an uncached text encode, a full-resolution preview decoded every few steps,
a safety pass serialized after the full decode. The value of building the cost model
first is exactly that it makes this a short list.

**Q: You adopted a compiled engine and now deployments are painful. What happened?**

Compilation freezes the shapes, so every resolution and batch size is its own engine
and every engine is a build artifact to version with the model. And cold start went
from roughly ten seconds to about a minute, which breaks scale-to-zero. The usual
resolution is a compiled pool for steady traffic plus a warm eager replica to absorb
the first request after idle.

**Q: A user says the new model "makes everything look the same". What did you ship?**

A distilled model. Step distillation narrows the output distribution, and it shows up
as reduced variety across seeds for the same prompt rather than as worse individual
images. It also will not appear on demo prompts. The fix is routing: distilled for the
draft grid, full model for the final render.

**Q: Why not just turn guidance off, since it doubles the cost?**

Because guidance is what makes the image follow the prompt; turning it off changes
what the user gets, which is a product change disguised as an optimization. The
equivalent that is not a product change is guidance distillation, which trains the
effect into a single evaluation.

**Q: How do you evaluate a change here?**

Paired human preference on a fixed prompt set, plus an automatic prompt-adherence
proxy (a VQA model asked whether the requested content is present, or an image-text
similarity score) to catch regressions cheaply between human rounds. Keep seeded
determinism so two versions can be diffed on identical inputs.

## Commonly answered wrong (the traps)

**Q: How would you speed this up? Most candidates start with quantization and
batching.**

**The tempting answer:** quantize the weights, batch more requests, add more GPUs.

**Why it is wrong:** those are the levers for a memory-bandwidth-bound autoregressive
decoder. This workload is compute-bound with fixed shapes and no KV cache, and the
dominant term is a multiplier you can attack directly. Halving the evaluations halves
the cost; quantization is a second-order effect next to that.

**What to say instead:** name the cost unit, cut the evaluations (sampler, then
guidance distillation, then step distillation), then make each evaluation cheaper with
a compiled engine, then talk about batching the variant grid.

---

**Q: Is FID a good way to check quality did not regress?**

**The tempting answer:** yes, it is the standard metric for image generation.

**Why it is wrong:** FID measures the distance between two distributions of images
against a reference dataset. It says nothing about whether a specific prompt produced
a good image, it is sensitive to implementation details and sample count, and a
distilled model can hold FID while losing exactly the diversity FID is supposed to
detect.

**What to say instead:** paired human preference on a fixed prompt set as the gate,
with a prompt-adherence proxy for cheap continuous checks, and a diversity measure
across seeds if you have shipped a distilled model.

---

**Q: Video is just images in a loop, right?**

**The tempting answer:** generate each frame and stitch them together.

**Why it is wrong:** independent frames flicker. Textures crawl, identities drift and
lighting jumps, because nothing ties the frames together. Fixing that means attending
across time, which is what makes the cost superlinear in clip length and forces
chunked generation with overlap for anything longer than the native window.

**What to say instead:** temporal attention as the requirement, chunking with
conditioning as the practical approach, a queued job with a progress channel as the
serving shape, and cost per second of output as the metric.
