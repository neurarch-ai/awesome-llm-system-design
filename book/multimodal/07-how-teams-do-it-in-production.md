# 7. How teams do it in production

Every large vision-language system converges on the same three-stage skeleton:
a vision encoder turns the image into a feature grid, a projector maps those
features into the decoder's embedding space, and one LLM decoder generates over
an interleaved text-plus-image sequence. What actually differs between systems is
two decisions: **how many tokens the image becomes** (controlled by the connector
and the resolution policy), and **how the serving stack isolates the encoder from
the decoder**. The architecture everyone shares; the leverage is in the token
budget and the serving split.

## Where the real designs diverge

| System | Connector | Resolution policy | Image-token budget | Serving approach | Why this shape |
|---|---|---|---|---|---|
| LLaVA (Microsoft) | MLP projector | Fixed CLIP ViT-L/14 336px | ~576 (fixed) | Single server, frozen encoder | Cheapest recipe; one-time projector training on top of frozen CLIP |
| BLIP-2 (Salesforce) | Q-Former cross-attention | Fixed (CLIP or EVA-CLIP) | 32 (fixed cap) | Single server | Minimal LLM fine-tuning; frozen encoder and LLM, tiny learned bridge |
| Flamingo (DeepMind) | Perceiver resampler plus gated cross-attention | Fixed, few tokens | Fixed few | Frozen backbone pair | Interleaved few-shot over frozen vision and language models |
| Idefics2 (HuggingFace) | Perceiver pooling plus MLP | Native up to 980px, optional 4-way split | 64 or 320 | Single server, NaViT-style | Open recipe; bounded tokens with better backbones and OCR data |
| Qwen2-VL (Alibaba) | MLP projector plus M-RoPE | Native dynamic resolution | Variable, scales with resolution | Single server, batching adapts to variable lengths | Inputs vary widely in size; dynamic resolution avoids crop artifacts |
| Pixtral 12B (Mistral AI) | Custom ViT plus MLP projector | Native resolution, flexible per image | Variable, up to 4096 at 1024px | 128K context, multi-image | Custom encoder enables native aspect ratios and flexible budgets |
| NVLM (NVIDIA) | MLP or cross-attention, compared head-to-head | Tiled with 1-D tile tags for OCR | High (tiling multiplies count) | Single server, tile tags at preprocessing | OCR and dense documents; tile tags preserve spatial layout |
| vLLM V1 (Red Hat) | Any (runtime, not a model) | Per-model config | Per-model, with caching | Encoder cache plus prefix cache; async CPU/GPU split | Repeated images skip re-encoding; text-only skips encoder entirely |
| ROCm vLLM (AMD) | Any (runtime, not a model) | Per-model config | Per-model | DP encoder plus TP decoder; one all-gather instead of per-layer all-reduce | Encoder is a small fraction of params; TP on it wastes sync without saving compute |

The dividing line is simple: **the connector and resolution policy buy the quality
ceiling; the serving split and caching buy the latency and unit economics.** A
complete answer picks a point on both and justifies it from the task's detail
requirements and traffic mix.

## The systems (first-party write-ups)

- **Red Hat (vLLM)** [vLLM V1: accelerating multimodal inference](https://developers.redhat.com/articles/2025/02/27/vllm-v1-accelerating-multimodal-inference-large-language-models): Encoder output caching, per-image prefix caching, and async CPU/GPU input processing for faster multimodal serving. The reference for how to cache encoder embeddings by image hash and avoid collision in a prefix cache when placeholder tokens are shared across different images.
- **AMD (ROCm)** [Accelerating Multimodal Inference in vLLM](https://rocm.blogs.amd.com/software-tools-optimization/vllm-dp-vision/README.html): Data parallelism for vision encoders (each GPU holds a full copy, processes a different image batch) cuts synchronization from per-layer all-reduces to one final all-gather; up to 44 percent throughput gain on InternVL3.5-241B.
- **Alibaba (Qwen)** [Qwen2-VL: enhancing vision-language model perception at any resolution](https://arxiv.org/abs/2409.12191): Naive Dynamic Resolution processes images at varying resolutions into variable visual token counts via MLP projector; M-RoPE unifies position encoding across text, image, and video in one scheme.
- **Mistral AI** [Pixtral 12B](https://arxiv.org/abs/2410.07073): A custom ViT trained from scratch ingests native resolution and aspect ratio; flexible per-image token budget inside a 128K context; ships as Apache 2.0 with MM-MT-Bench for eval.
- **Microsoft (LLaVA)** [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485): The original MLP projector bridging a frozen CLIP vision encoder to an LLM; GPT-4-generated instruction data as the training signal for multimodal chat.
- **Hugging Face** [Introducing Idefics2: a powerful 8B vision-language model](https://huggingface.co/blog/idefics2): Perceiver pooling plus MLP projection; NaViT native-resolution encoding up to 980px; bounded 64 or 320 tokens per image depending on split mode; drops gated cross-attention in favor of simpler pool-then-project.
- **NVIDIA** [NVLM: open frontier-class multimodal LLMs](https://research.nvidia.com/labs/adlr/NVLM-1/): Head-to-head comparison of decoder-only (MLP) vs cross-attention connectors; 1-D tile-tagging for spatial layout in tiled high-resolution OCR; NVLM-D-72B matches or beats GPT-4o on math and document benchmarks.
- **DeepMind** [Flamingo: a visual language model for few-shot learning](https://arxiv.org/abs/2204.14198): Perceiver resampler plus gated cross-attention to bridge frozen vision and language backbones; fixed small token budget; the reference for bounded-cost few-shot VLM design.
- **Salesforce** [BLIP-2: bootstrapping language-image pre-training with frozen image encoders and large language models](https://arxiv.org/abs/2301.12597): Q-Former with 32 query tokens as a lightweight bridge between two frozen pre-trained models; minimal fine-tuning cost.
- **Dropbox** [Creating a modern OCR pipeline using CV and deep learning](https://dropbox.tech/machine-learning/creating-a-modern-ocr-pipeline-using-computer-vision-and-deep-learning): A productionized two-stage OCR pipeline (MSER detector plus CNN-BLSTM-CTC recognizer), with 10M synthetic training samples and LXC isolation for untrusted document uploads. Not a VLM, but the production OCR engineering reference.
- **Apple** [MM1: methods, analysis, and insights from multimodal LLM pre-training](https://arxiv.org/abs/2403.09611): Ablations on image-token count, connector design, and data mix; the most thorough published analysis of how each design axis trades off against the others.
- **Meta** [Chameleon: mixed-modal early-fusion foundation models](https://arxiv.org/abs/2405.09818): A single transformer over discrete image-and-text tokens; the main published recipe for stable early-fusion training.

For the full comparison, case studies, and model teardowns, see the dense reference
in [topics/09-multimodal-serving.md](../../topics/09-multimodal-serving.md).
