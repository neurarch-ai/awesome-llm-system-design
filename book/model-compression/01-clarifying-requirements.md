# 1. Clarifying the requirements

Compression questions are won in the first two minutes, because "make it smaller"
hides four different problems with four different answers. Here is the exchange.

**Candidate:** What is actually binding: memory, latency, throughput cost, or a
device constraint?

**Interviewer:** It has to run on one 80GB accelerator instead of two, and
time-to-first-token and per-token latency cannot get worse. Later we want a variant
that runs on-device.

**Candidate:** Those are two different projects. Fitting on one accelerator is a
memory problem, and a weight-only quantization usually solves it. On-device is a
memory ceiling plus a fixed accelerator that only runs certain formats fast, which
often forces a smaller model rather than a compressed big one.

**Interviewer:** Assume both, and tell me which you would do first.

**Candidate:** What is the serving shape? Batch size one interactive, or large
batches? That decides whether we are memory-bandwidth bound or compute bound, and
the two respond to completely different levers.

**Interviewer:** Interactive, small batches, long-ish prompts.

**Candidate:** Then decode is bandwidth bound and prefill is compute bound, so a
weight-only quantization helps per-token latency but not time-to-first-token.
How long are the contexts? Past a few thousand tokens the KV cache starts to rival
the weights, and compressing weights while ignoring KV solves the wrong half.

**Interviewer:** Up to 32K, sometimes more.

**Candidate:** Can we retrain, or is this post-training only? A calibration-set
quantization takes hours; a healing run after structured pruning takes billions of
tokens, and a distillation takes a real training budget.

**Interviewer:** We have some compute but not a pretraining budget.

**Candidate:** What is the quality bar, and on which capabilities? "No regression"
is not a bar. I need to know which capabilities cannot move, because compression
damage is never uniform.

**Interviewer:** Code generation and tool calling are load-bearing. Long-context
retrieval matters for the RAG path.

**Candidate:** Last one: what does the target hardware actually accelerate? Int4
weight-only, fp8 compute, 2:4 sparsity, and the on-device NPU's supported formats
are four different answers, and a lever the kernel does not support is a memory
saving with no speedup.

Let us summarize. **We are asked to fit a model onto one accelerator without
hurting interactive latency, on long-ish contexts, with post-training methods plus
a modest healing budget, holding code generation, tool calling, and long-context
retrieval fixed, and later to produce an on-device variant.**

Two consequences fall out immediately, and stating them early is most of the signal.

**Consequence 1: compression is a hardware question wearing an algorithm's
clothes.** The size of a tensor and the speed of the kernel that consumes it are
different facts. Unstructured 50 percent sparsity saves nothing on a dense matmul
unit. Int4 weights only go fast where an int4 kernel exists, and at large batch the
win shrinks because the bottleneck moves from reading weights to doing arithmetic.
The right first question is not "how much can I compress" but "what does this
accelerator make cheap," and every recommendation should name the format and the
kernel, not just the bit width.

**Consequence 2: top-line accuracy is the wrong acceptance test.** Two models can
score the same on a benchmark and disagree on a large fraction of individual items,
which is exactly what compression does: it does not lower the average so much as
scatter behavior around it, concentrated in the long tail. The acceptance test is
therefore per capability (code, tool-call and JSON validity, long-context
retrieval, multilingual, safety refusals), paired per item against the uncompressed
baseline, and reported with a flip rate, not a single accuracy delta. That framing
comes straight from the compression-evaluation literature
([Accuracy is Not All You Need](https://arxiv.org/abs/2407.09141)), and it is the
part interviewers use to separate people who have shipped a compressed model from
people who have read about one.

A third point is worth saying out loud because it reframes the whole exercise:
**compression competes with not compressing.** A smaller base model, a router that
sends easy requests elsewhere, a shorter prompt, or a better KV strategy can each
buy the same resource with less quality risk. The
[cost-optimization](../cost-optimization/) and [KV cache](../kv-cache/) chapters own
those alternatives, and a strong answer checks them before reaching for a
quantizer.
