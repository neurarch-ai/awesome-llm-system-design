# 7. How teams do it in production

Every production system wraps the model in the same three-stage skeleton: an input
guard before the model, an output guard after, and a policy router that turns
verdicts into actions. What actually differs between companies is two decisions:
**whether safety is treated as a classification problem or a structural one**, and
**where the latency budget forces the design to cut corners on the guard models.**
The skeleton everyone shares; the leverage is in those two choices.

## Where the real designs diverge

| System | Guard model approach | Jailbreak / injection defense | Human review | Volume / latency shape | When it wins | Watch out for |
|---|---|---|---|---|---|---|
| Anthropic (Constitutional Classifiers) | Input and output classifiers trained on synthetic constitution-derived data | Classifiers are a separate decision; cut jailbreak ASR from 86% to 4.4% | Red-team exercise; no per-decision review | Production; 23.7% compute overhead | Universal-jailbreak resistance with a measurable eval bar; policy expressed as a constitution | Added cost on every request; over-broad constitution raises FRR |
| Microsoft (MSRC) | Prompt Shields (trained multilingual classifier) plus deterministic rules | Spotlighting marks untrusted content; code-side action gates; exfil vector blocking | Human approval before high-consequence actions | Enterprise agents; safety as a structural problem | Indirect prompt injection over retrieved content; blast radius kept small in code | Design complexity; permission model adds developer friction |
| Roblox | Distilled and quantized transformer classifiers per modality (text, voice, PII) | Fast filters across 28 languages; PII detection at 370k RPS | Thousands of human reviewers for nuance and appeals; labels feed retraining | 750k RPS text, 8.3k RPS voice, 6.1B messages per day | Massive real-time scale; each modality sized to its own RPS profile | Needs a large human-review workforce; distilled models can miss edge-policy cases |
| Meta (Llama Guard) | Instruction-tuned Llama 2-7B classifier; taxonomy in the prompt | Paired Prompt Guard flags injection and jailbreaks | Not applicable (shipped as a model) | Self-hosted; full guard-LLM latency on every request | Open, drop-in classifier with zero-shot taxonomy adaptation | You pay the full 7B LLM latency; guard shares LLM failure modes |
| Google (ShieldGemma) | Gemma-based generative safety classifiers | Not applicable (classifier, benchmarked above Llama Guard) | Not applicable (shipped as a model) | Self-hosted | Higher catch rate than Llama Guard on benchmarks; open weights | Same self-host and per-request latency as any guard-LLM |
| OpenAI (cookbook pattern) | Moderation API for input; LLM-judge G-Eval scorer for output | Guard races generation via asyncio; LLM-judge inherits base model failure modes | Not applicable (pattern, not a service) | Async race hides guard latency; LLM-judge cost per request | Guard latency hidden behind generation when generation is side-effect-free | Async can leak tokens before block fires if generation has side effects; LLM-judge is persuadable |
| NVIDIA (NeMo Guardrails) | Config-driven rails wiring LlamaGuard plus AlignScore fact-check | Rails invoke a guard model per turn; LlamaGuard on a separate vLLM engine | Not applicable (framework) | RAG apps; vLLM guard tier scales independently | Declarative rails plus grounding check without hand-wiring; separate guard engine enables batching | Rails as good as the YAML config behind them; output rails in series stack latency |
| Cloudflare (Firewall for AI) | Llama Guard on Workers AI GPUs; zero-shot, 13 categories | Input-only; prompt blocked before it reaches the origin model | Not applicable (edge proxy) | 2s async timeout; edge auto-scales | Model-agnostic; zero app changes; protects any backend uniformly | Input-only so unsafe generations pass; 2s timeout is a deliberate fail-open on timeout |
| Grab | Two-tier routing by LLM violation-likelihood score | Not applicable (content moderation, not injection defense) | Not applicable | Cost-driven; cheap tier handles clear cases | Cost control by concentrating expensive calls on ambiguous content | Miscalibrated score sends violating content down the cheap tier; threshold tuning required |
| Thomson Reuters (CoCounsel) | Grounding in trusted sources; no text-toxicity classifier | Not applicable (grounding is the defense mechanism) | Attorney review plus nightly 1,500-test benchmark | Legal domain; grounding beats moderation on accuracy | High-stakes domain where grounding is the right safety model | Needs a trusted corpus and a human-in-the-loop; slower than a classifier gate |
| Salesforce (Einstein Trust Layer) | Hybrid rules plus model; seven toxicity categories | Instruction-defense post-prompting plus PII masking before the gateway | Not applicable (platform controls) | Enterprise; zero data retention at provider; full audit trail | PII masking and zero-retention around every LLM call; audit log of user accept/modify/reject | Platform-bound; rules-based scoring can lag novel attack patterns |

The core dividing line is whether a system treats safety as a **classification
problem** (train a model to score text: Anthropic, Roblox, Meta, Google,
Cloudflare) or a **structural one** (isolate untrusted text and gate actions or
ground answers in code: Microsoft, Thomson Reuters, Salesforce), with human review
layered in wherever the stakes outrun the classifier's confidence.

## The systems

First-party engineering write-ups. Read them for what an interview answer skips:
who the system serves, the product constraints, the eval bar, and the deployment
shape.

- **Roblox** [How Roblox Uses AI to Moderate Content on a Massive Scale](https://about.roblox.com/newsroom/2025/07/roblox-ai-moderation-massive-scale): Multi-model text, voice, and PII moderation at 750k requests per second with real-time prevention. *(deployment)*
- **Roblox** [Deploying ML for Voice Safety](https://about.roblox.com/newsroom/2024/07/deploying-ml-for-voice-safety): Machine-labeled data trains a fast quantized voice-abuse classifier at 2,000 requests per second. *(deployment)*
- **Anthropic** [Constitutional Classifiers: defending against universal jailbreaks](https://www.anthropic.com/research/constitutional-classifiers): Input and output classifiers trained on a synthetic constitution cut jailbreaks from 86% to 4.4%. *(eval bar)*
- **Microsoft (MSRC)** [How Microsoft defends against indirect prompt injection attacks](https://www.microsoft.com/en-us/msrc/blog/2025/07/how-microsoft-defends-against-indirect-prompt-injection-attacks): Defense in depth: spotlighting, Prompt Shields detection, permission-based mitigation. *(deployment)*
- **NVIDIA** [Content Moderation and Safety Checks with NeMo Guardrails](https://developer.nvidia.com/blog/content-moderation-and-safety-checks-with-nvidia-nemo-guardrails/): Wiring LlamaGuard plus fact-check rails into RAG apps via NeMo Guardrails config. *(product design)*
- **Meta** [Llama Guard: LLM-based input-output safeguard](https://arxiv.org/abs/2312.06674): An instruction-tuned classifier moderating prompts and responses by taxonomy. *(product design)*
- **Google** [ShieldGemma: generative AI content moderation](https://arxiv.org/abs/2407.21772): Gemma-based safety classifiers benchmarked above Llama Guard. *(eval bar)*
- **Meta** [Llama Prompt Guard 2](https://developer.meta.com/ai/docs/model-cards-and-prompt-formats/prompt-guard/): A lightweight binary classifier flagging prompt injection and jailbreaks. *(product design)*
- **OpenAI** [How to implement LLM guardrails](https://developers.openai.com/cookbook/examples/how_to_use_guardrails): Async input and output guardrail patterns with latency tradeoffs. *(deployment)*
- **Cloudflare** [Block unsafe prompts with Firewall for AI](https://blog.cloudflare.com/block-unsafe-llm-prompts-with-firewall-for-ai/): An edge proxy using Llama Guard to block harmful prompts across 13 categories. *(deployment)*
- **Salesforce** [Inside the Einstein Trust Layer](https://developer.salesforce.com/blogs/2023/10/inside-the-einstein-trust-layer): PII masking, toxicity scoring, and prompt-injection defense around LLM calls. *(deployment)*
- **Grab** [How LLMs make content moderation more precise](https://www.grab.com/inside-grab/stories/how-large-language-models-help-us-make-more-precise-content-moderation-decisions/): Two-tier moderation routing content by an LLM violation-likelihood score. *(product design)*
- **Thomson Reuters** [Inside CoCounsel's guardrails](https://legal.thomsonreuters.com/blog/why-your-legal-ai-needs-more-than-the-open-web-a-look-inside-cocounsels-guardrails/): Grounding legal AI in trusted sources with attorney review and nightly 1,500-test benchmarks. *(eval bar)*
- **Slack** [Securing the Agentic Enterprise](https://slack.com/blog/transformation/securing-the-agentic-enterprise): Multi-layered AI guardrails enforcing user permissions and real-time prompt-injection defense. *(deployment)*
- **Databricks** [Implementing LLM Guardrails for Safe GenAI Deployment](https://www.databricks.com/blog/implementing-llm-guardrails-safe-and-responsible-generative-ai-deployment-databricks): Safety filters on the Foundation Model API blocking unsafe inputs/outputs, logged for audit. *(deployment)*
- **Wealthsimple** [Our LLM Gateway for secure, reliable generative AI](https://engineering.wealthsimple.com/get-to-know-our-llm-gateway-and-how-it-provides-a-secure-and-reliable-space-to-use-generative-ai): An internal gateway redacting PII and tracking external data for safe employee GenAI use. *(deployment)*
