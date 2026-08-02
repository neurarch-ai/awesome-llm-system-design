# 7. How teams do it in production

Every serious compression stack converges on the same skeleton: pick the lever that
moves the binding resource, keep a small set of layers at higher precision, and gate
on a paired comparison against the uncompressed model. What differs is **which
constraint they are solving for** (server cost, device memory, or a numeric format
the hardware demands) and **whether they can afford to retrain**, and those two
choices explain almost every divergence below.

## Where the real designs diverge

| System | Primary constraint | Method | Retraining | When it wins | Watch out |
|---|---|---|---|---|---|
| Server weight-only quantization (GPTQ, AWQ lineage) | Fit and decode latency | int4 group-wise weights, fp16 activations | None | Interactive serving at small batch, fastest path to a win | Win shrinks at high batch; needs a fused kernel to be real |
| Activation and compute quantization (SmoothQuant, fp8 recipes) | Throughput per dollar | int8 or fp8 weights and activations | None to light | High-batch and prefill-heavy workloads where FLOPs are the cost | Outlier handling is mandatory; verify the runtime is not silently upcasting |
| Rotation-based low-bit (QuaRot, SpinQuant) | 4-bit activations and KV, mobile serving | Orthogonal rotations fused into the network, then quantize | None (SpinQuant learns rotations) | Pushing below the point where plain PTQ holds | Rotations must be fused at export or you pay them every forward |
| Train-time ternary (BitNet) | Extreme efficiency | 1.58-bit weights, trained that way from scratch | Full pretraining | New models designed for the regime | Not a post-training option; you cannot convert an existing model |
| One-shot pruning (SparseGPT, Wanda) | Memory, or a sparsity-accelerated target | Unstructured or 2:4, no or minimal retraining | None | Fast exploration of how much sparsity a model tolerates | Unstructured buys no speed on dense kernels |
| Prune plus distill families (Minitron) | A whole size ladder from one parent | Structured pruning, then distillation from the parent | Billions of tokens | Building small models without a pretraining budget | Depth cuts damage multi-step behavior; healing budget is real |
| Continued-pretraining pruning (Sheared LLaMA) | A small model at a fraction of from-scratch cost | Targeted structured pruning to a chosen architecture, then continued pretraining | Yes | You want a specific small architecture and have token budget | Token budget still substantial |
| On-device foundation models (Apple Intelligence) | Hard device memory ceiling and fixed accelerator formats | Small on-device model, aggressive low-bit weight compression, task adapters, larger server model for the rest | Yes, in model design | Consumer devices where the ceiling is not negotiable | Two models and an adapter pipeline to maintain and evaluate |
| fp8 as a first-class training and serving format (DeepSeek-V3) | Cost across the whole lifecycle | fp8 in training, carried into serving | Designed in | Greenfield large models on fp8-capable hardware | Requires the numerics work up front; not retrofittable |
| CPU and consumer-GPU quantization (llama.cpp, GGUF) | Running at all on commodity hardware | k-quant weight formats, CPU and Metal kernels | None | Local and hobby deployment, and quick evaluation of quantization tolerance | Format and quality vary per quant type; benchmark the exact one you ship |

## The dividing line

Two questions place any stack in the table. **Can you retrain?** If no, you are in
the quantization and one-shot pruning half, and your ceiling is set by how much the
model tolerates without repair. If yes, structured pruning plus distillation
dominates, because it produces a genuinely smaller dense model that is fast on any
hardware instead of a big model wearing a smaller coat.

**What does the target accelerate?** A format the hardware does not accelerate is a
memory technique, not a speed technique. That single sentence resolves most of the
arguments about which method is "best," and it is why on-device stacks look nothing
like server stacks even when both say "4-bit."

## First-party sources

- **LLM.int8()** [8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339): the outlier-channel decomposition that made int8 inference practical for large models.
- **SmoothQuant** [Accurate and Efficient Post-Training Quantization for Large Language Models](https://arxiv.org/abs/2211.10438): migrating activation outliers into the weights so both sides can be quantized.
- **GPTQ** [Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323): layer-wise second-order rounding with error compensation.
- **AWQ** [Activation-aware Weight Quantization for LLM Compression and Acceleration](https://arxiv.org/abs/2306.00978): protect the salient weight channels identified by activation magnitude.
- **QuaRot** [Outlier-Free 4-Bit Inference in Rotated LLMs](https://arxiv.org/abs/2404.00456) and **SpinQuant** [LLM quantization with learned rotations](https://arxiv.org/abs/2405.16406): remove outliers by rotating the basis, which is what unlocked 4-bit activations and KV.
- **Microsoft Research** [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764): ternary weights, trained in that regime rather than converted.
- **KIVI** [A Tuning-Free Asymmetric 2bit Quantization for KV Cache](https://arxiv.org/abs/2402.02750): per-channel keys, per-token values, because the two have different outlier structure.
- **SparseGPT** [Massive Language Models Can Be Accurately Pruned in One-Shot](https://arxiv.org/abs/2301.00774) and **Wanda** [A Simple and Effective Pruning Approach for Large Language Models](https://arxiv.org/abs/2306.11695): reconstruction-based and activation-aware one-shot pruning.
- **Princeton** [Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning](https://arxiv.org/abs/2310.06694): prune a large model to a target architecture, then continue pretraining.
- **NVIDIA** [Compact Language Models via Pruning and Knowledge Distillation](https://arxiv.org/abs/2407.14679): the prune-then-distill recipe for building a size ladder from one parent.
- **Google DeepMind** [On-Policy Distillation of Language Models](https://arxiv.org/abs/2306.13649): train the student on its own outputs scored by the teacher, fixing the distribution mismatch.
- **Apple** [Apple Intelligence Foundation Language Models](https://arxiv.org/abs/2407.21075): an on-device model with low-bit weight compression plus task adapters, alongside a server model.
- **DeepSeek** [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437): fp8 as a first-class format through training and serving.
- **Microsoft Research India** [Accuracy is Not All You Need](https://arxiv.org/abs/2407.09141): why compressed models that match on accuracy still behave differently, and what to measure instead.
- **Red Hat AI and the vLLM project** [llm-compressor](https://github.com/vllm-project/llm-compressor): production recipes for weight, activation, and KV-cache quantization targeting vLLM.
- **llama.cpp** [the GGUF quantization ecosystem](https://github.com/ggml-org/llama.cpp): the reference for CPU, Metal, and consumer-hardware quantized inference.
