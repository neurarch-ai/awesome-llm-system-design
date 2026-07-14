# 7. How teams do it in production

Every production post-training pipeline converges on the same spine: start from
an open or vendor base, curate a small high-quality dataset, run supervised
fine-tuning (usually a LoRA or QLoRA adapter, occasionally a full fine-tune),
optionally add preference optimization, then gate on a held-out eval before
serving. What actually differs between companies is three decisions: **how deep
to adapt** (LoRA vs full fine-tune), **whether to align past SFT** (DPO vs RLHF
vs nothing), and **where the training signal comes from** (human labels vs
synthetic vs production logs). Most ship SFT alone; preference tuning shows up
only where a quality axis SFT could not capture actually mattered.

## Where the real designs diverge

| System | Adaptation | Alignment beyond SFT | Data curation focus | Key result | Eval gate |
|---|---|---|---|---|---|
| Grammarly CoEdIT | Instruction fine-tune (FLAN-T5 770M to 11B) | SFT only | Dense edit-task instruction set with paraphrased phrasings | Beats GPT-3 at 12x to 60x fewer params | Human pref vs generalist LLMs |
| Anyscale iterative DPO | Full fine-tune (LoRA r=64 drifted OOD) | SFT plus iterative DPO; LLM-as-judge synthetic prefs | Sampled summaries + 70B judge; preference rule aligned to eval axis | Pareto-dominates GPT-4o on accuracy + compression | Q-and-A accuracy plus compression ratio |
| Shopify Flow | Full fine-tune Qwen3-32B (FSDP, 2x H200) | SFT only | Reverse-engineered synthetic workflows in Python DSL, not native JSON | 68% cheaper than frontier; 1% traffic slice showed 35% activation gap that flywheel closed | Weekly LLM judge calibrated to human labels plus live activation rate |
| Mercari attribute extraction | QLoRA 4-bit gemma-2b-it, single A100 | SFT only | Templated (listing, instruction, keys, values) pairs for top 20 categories | Beats GPT-3.5 by 5+ BLEU; ~14x cheaper | BLEU vs GPT-3.5 on held-out listings |
| Grab multilingual OCR | LoRA warm-start then full fine-tune Qwen2-VL 2B | SFT only | Synthetic OCR (Common Crawl); real ID docs via Documint auto-labeling | +70pp on Thai, +40pp on Vietnamese; 1B distilled model matches 2B | Extraction accuracy per script |
| LinkedIn EON | Multi-task instruction tuning (Llama 3.1 8B/70B); prompt simplification -30% | SFT plus RLHF and DPO; synthetic safe outputs for harmful scenarios | ~200M tokens diverse instructions + reasoning traces + Economic Graph domain data | Beats GPT-4o mini by 4pp; 75x cheaper than GPT-4 | Candidate-job matching accuracy |
| Cloudflare Workers AI | Customer LoRA adapters on shared base (Llama 2 / Mistral / Gemma) | Not applicable (inference serving) | Per-customer adapters up to 100MB, rank at most 8 | Many tenants, one warm base, ms-scale adapter swap | Multi-LoRA edge serving correctness |
| Spotify AQE | Rejection-sampling SFT then DPO | Rejection-sampling SFT plus DPO on downstream retrieval rank | Click-through and annotated query-document pairs; preference from real search rank | 30.8% vs 28.5% top-1; ~70% faster query-expansion at serve time | Top-1 accuracy on Natural Questions |

The dividing line is simple. **Adaptation depth** is set by how large the behavior
shift is relative to the base (small nudge stays LoRA/QLoRA; large shift or
OOD risk forces full fine-tune). **Alignment past SFT** appears only when a
genuine quality axis cannot be expressed by imitating positive examples (safety
alignment, comparative preference, tone vs two plausible outputs). **Data source**
is whatever the team could actually label, with LLM-as-judge closing the gap
wherever human labeling is too slow or expensive.

## The systems (first-party write-ups)

- **Grammarly** [CoEdIT: state-of-the-art text editing with fewer parameters](https://www.grammarly.com/blog/engineering/coedit-text-editing/): Dense task-specific instruction tuning beats generalist LLMs at 12x to 60x fewer params.
- **Anyscale** [Fine-Tuning LLMs: LoRA or Full-Parameter?](https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter-an-in-depth-analysis-with-llama-2): LoRA vs full fine-tune accuracy tradeoffs per task type.
- **Anyscale** [Direct Preference Optimization with Synthetic Data](https://www.anyscale.com/blog/direct-preference-optimization-with-synthetic-data): Iterative DPO with synthetic prefs, async reference model, and judge-aligned eval.
- **Shopify** [Flow generation through natural language: an agentic modeling approach](https://shopify.engineering/fine-tuning-agent-shopify-flow): Fine-tuned Qwen3-32B agent with a weekly LLM-judge retraining flywheel.
- **Shopify** [Leveraging multimodal LLMs for the global catalogue](https://shopify.engineering/leveraging-multimodal-llms): Small VLM fine-tuned for catalogue extraction at 40M inferences per day.
- **Mercari** [Fine-Tuning an LLM to Extract Dynamically Specified Attributes](https://engineering.mercari.com/en/blog/entry/20240913-fine-tuning-an-llm-to-extract-dynamically-specified-attributes/): QLoRA-tuned 2B model beats GPT-3.5 at 14x lower cost.
- **Grab** [A custom vision LLM to improve document processing](https://engineering.grab.com/custom-vision-llm-at-grab): LoRA then full fine-tune of Qwen2-VL for multilingual OCR.
- **LinkedIn** [How we built domain-adapted foundation GenAI models](https://www.linkedin.com/blog/engineering/generative-ai/how-we-built-domain-adapted-foundation-genai-models-to-power-our-platform): Llama-based EON via instruction tuning plus RLHF/DPO; 75x cheaper than GPT-4.
- **Cloudflare** [Running fine-tuned models on Workers AI with LoRAs](https://blog.cloudflare.com/fine-tuned-inference-with-loras/): Multi-LoRA edge serving of customer adapters on shared base models.
- **Spotify** [Optimizing Query Expansions via LLM Preference Alignment](https://research.atspotify.com/2025/7/optimizing-query-expansions-via-llm-preference-alignment): Rejection-sampling SFT plus DPO aligns a query-expansion LLM; 70% faster at serve time.
- **Meta** [How to fine-tune: focus on effective datasets](https://ai.meta.com/blog/how-to-fine-tune-llms-peft-dataset-curation/): Data-curation rules for SFT and PEFT; quality over quantity.
- **GitHub** [Building a faster, smarter Copilot with a custom model](https://github.blog/ai-and-ml/github-copilot/the-road-to-better-completions-building-a-faster-smarter-github-copilot-with-a-new-custom-model/): Mid-training plus SFT (fill-in-middle) plus RL for code completion.
- **Uber** [Open Source and In-House: How Uber Optimizes LLM Training](https://www.uber.com/us/en/blog/open-source-and-in-house-how-uber-optimizes-llm-training/): In-house stack spanning LoRA, QLoRA, full fine-tuning, and continued pre-training.
- **Hugging Face** [Preference Tuning LLMs with Direct Preference Optimization Methods](https://huggingface.co/blog/pref-tuning): Empirical comparison of DPO, IPO, and KTO; beta drives outcomes.
- **Databricks** [A Practical Guide to LLM Fine Tuning](https://www.databricks.com/blog/llm-fine-tuning): End-to-end lifecycle: metrics, data quality, LoRA-first, and retrain cadence.
- **DeepSeek** [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) (Jan 2025): the shift that reframed post-training in 2025. Reasoning is trained with large-scale RL against **rule-based, verifiable rewards** (answer checkers, unit tests) using GRPO, a critic-free variant of PPO, with little or no SFT. The takeaway for a modern answer: where the reward is verifiable (math, code), RL beats collecting more preference labels.
- **Ai2** [Tulu 3: pushing frontiers in open language model post-training](https://allenai.org/blog/tulu-3-technical) (Nov 2024, [paper](https://arxiv.org/abs/2411.15124)): a fully open post-training recipe of four stages, data curation, SFT, DPO, then **RL from verifiable rewards (RLVR)**, that reuses the RLHF objective but swaps the reward model for a verification function. The reference for how to combine SFT, DPO, and verifiable-reward RL in one pipeline.

For the full comparison table, math, and all case studies see the dense reference
in [topics/05-post-training-pipeline.md](../../topics/05-post-training-pipeline.md).
