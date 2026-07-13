# 7. How teams do it in production

Every system here shares the same skeleton: a gateway fronts the providers, a
cache short-circuits repeats, a router or cascade picks a tier, and only the
surviving hard queries reach the frontier model. What actually differs between
teams is which part of the pipeline they optimized first and which lever buys
the most for their specific cost driver.

## Where the real designs diverge

| System | Primary lever | Model tier | Cost driver they attacked | Why this shape |
|---|---|---|---|---|
| Stanford FrugalGPT | Cascade with trained scorer | 3-stage chain (cheap to frontier) | Request cost on mixed-quality queries | A scorer trained on answer reliability is more precise than a blind router |
| LMSYS RouteLLM | Preference-data blind router | Weak (Mixtral class) vs strong (GPT-4 class) | Request cost on Chatbot Arena-style traffic | 55k preference pairs teach "hard vs easy" transferably across model pairs |
| Anyscale | Fine-tuned classifier router | Mixtral-8x7B vs GPT-4 | Request cost; mixed open/closed fleet | 5-way quality score gives stronger gradient than binary hard/easy |
| IBM Research | Multi-model predictive router | 11-model library | Request cost across specialist tasks | Some 13B specialists beat 70B generalists; routing to the right specialist wins |
| Microsoft LLMLingua | Prompt compression | No model change | Input-token cost on long verbose RAG context | Perplexity-based token scoring drops 20x with near-zero quality loss |
| Databricks batch inference | Batch API / offline routing | Any self-hosted model | Bulk jobs at online prices | ai_query SQL interface plus auto-scale runs bulk work at max batch size |
| Baseten FP8 | Quantization | Mistral 7B FP8 on H100 | Self-hosted per-token cost | FP8 halves VRAM, +33% tokens/s, -24% cost per token versus FP16 |
| Cloudflare AI Gateway | Exact-match cache + gateway | Any provider | Request count on repeated identical calls | Hash(body) cache; exact only today, semantic planned; gateway for fallback |
| Uber GenAI Gateway | Gateway with budgets + fallbacks | Multi-vendor fleet | Spend visibility and reliability at scale | Unified proxy replaces per-service provider calls; budget per team |

The dividing line: **routing and cascades choose a cheaper model** (before or
after seeing an answer); **caching and compression make the call itself cheaper**.
Right-sizing and quantization shrink the model-tier cost floor. The gateway makes
all of them enforceable. Pick the lever that matches your cost driver.

## The systems (first-party write-ups)

- **Stanford** [FrugalGPT: Using LLMs While Reducing Cost and Improving Performance](https://arxiv.org/abs/2305.05176): A learned cascade defers to pricier models only when the cheap answer scores unreliable; matches GPT-4 quality at up to 98% lower cost or lifts accuracy 4% at the same spend.
- **LMSYS** [RouteLLM: an open framework for cost-effective LLM routing](https://www.lmsys.org/blog/2024-07-01-routellm/): Preference-data router trained on 55k Chatbot Arena comparisons; sends only 14% of traffic to GPT-4 while retaining 95% of GPT-4 quality on MT Bench; four router flavors, all transfer to unseen model pairs.
- **Anyscale** [Building an LLM Router for High-Quality and Cost-Effective Responses](https://www.anyscale.com/blog/building-an-llm-router-for-high-quality-and-cost-effective-responses): Llama-3 8B fine-tuned as a 5-way complexity classifier; 70% cost cut on MT Bench; GPT-4-as-judge labels 109k training queries.
- **IBM Research** [LLM routing for quality, low-cost responses](https://research.ibm.com/blog/LLM-routers): Predictive router across an 11-model library; benchmark-trained to predict accuracy-to-cost ratio; 11-model ensemble beats every individual model; up to 85% cost cut.
- **Microsoft Research** [LLMLingua: prompt compression for LLM efficiency](https://www.microsoft.com/en-us/research/blog/llmlingua-innovating-llm-efficiency-with-prompt-compression/): Small-LM perplexity scoring with coarse-then-fine compression; up to 20x compression, about 1.5-point quality loss; ships integrated into LlamaIndex for RAG.
- **Databricks** [Simple, Fast, Scalable Batch LLM Inference](https://www.databricks.com/blog/introducing-simple-fast-and-scalable-batch-llm-inference-mosaic-ai-model-serving): ai_query SQL interface for governed batch inference over warehouse tables; auto-scales, no data movement, fault-tolerant retries; the batch-versus-online cost lever at warehouse scale.
- **Baseten** [33% faster LLM inference with FP8 quantization](https://www.baseten.co/blog/33-faster-llm-inference-with-fp8-quantization/): Mistral 7B in FP8 on H100 via TensorRT-LLM; 33% more tokens/s, 24% lower cost per million tokens vs FP16; perplexity matches, manual evals show only minor stylistic variation.
- **Cloudflare** [Caching in AI Gateway](https://developers.cloudflare.com/ai-gateway/features/caching/): Exact-match cache (hash on provider, model, full body); cf-aig-cache-ttl and cf-aig-cache-key headers for per-request TTL and key customization; covers text and image responses; semantic cache planned.
- **Uber** [Uber's GenAI Gateway](https://www.uber.com/blog/genai-gateway/): Unified multi-vendor proxy with team-level budget management, fallbacks across providers, and per-call logging; central cost-control without which optimization is guesswork.

For the full comparison table, all math, and the quadrant plot, see the dense
reference in [topics/11-cost-optimization-and-model-routing.md](../../topics/11-cost-optimization-and-model-routing.md).
