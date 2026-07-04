# Case studies, by industry

The same shipped systems as [CASE-STUDIES.md](CASE-STUDIES.md), pivoted by industry so you can see which the same LLM problem patterns recur in your domain. 205 case studies across 11 industries.

---

## Big Tech and cloud (64)

| Company | Case study | Topic |
| --- | --- | --- |
| Alibaba | [Qwen2-Audio Technical Report](https://arxiv.org/abs/2407.10759) | Multimodal Serving |
| Alibaba | [Qwen2-VL: enhancing vision-language perception at any resolution](https://arxiv.org/abs/2409.12191) | Multimodal Serving |
| Amazon | [Semantic Product Search](https://arxiv.org/abs/1907.00937) | Semantic Search And Embeddings |
| AMD | [Accelerating Multimodal Inference in vLLM](https://rocm.blogs.amd.com/software-tools-optimization/vllm-dp-vision/README.html) | Multimodal Serving |
| Apple | [MM1: methods, analysis, and insights from multimodal LLM pre-training](https://arxiv.org/abs/2403.09611) | Multimodal Serving |
| Cloudflare | [Block unsafe prompts with Firewall for AI](https://blog.cloudflare.com/block-unsafe-llm-prompts-with-firewall-for-ai/) | Safety And Guardrails |
| Cloudflare | [Caching in AI Gateway](https://developers.cloudflare.com/ai-gateway/features/caching/) | Cost Optimization And Model Routing |
| Cloudflare | [Durable Objects for WebSockets and auth in AI Gateway](https://blog.cloudflare.com/do-it-again/) | Realtime Streaming Chat |
| Cloudflare | [Introducing AutoRAG: managed RAG on Cloudflare](https://blog.cloudflare.com/introducing-autorag-on-cloudflare/) | RAG Serving |
| Cloudflare | [Running fine-tuned models on Workers AI with LoRAs](https://blog.cloudflare.com/fine-tuned-inference-with-loras/) | Post Training Pipeline |
| Databricks | [A Practical Guide to LLM Fine Tuning](https://www.databricks.com/blog/llm-fine-tuning) | Post Training Pipeline |
| Databricks | [Accelerating LLM inference with prompt caching](https://www.databricks.com/blog/accelerating-llm-inference-prompt-caching-open-source-models-databricks) | Long Context And KV Cache |
| Databricks | [Creating High Quality RAG Applications with Databricks](https://www.databricks.com/blog/building-high-quality-rag-applications-databricks) | RAG Serving |
| Databricks | [Implementing LLM Guardrails for Safe GenAI Deployment](https://www.databricks.com/blog/implementing-llm-guardrails-safe-and-responsible-generative-ai-deployment-databricks) | Safety And Guardrails |
| Databricks | [Inference-Friendly Models with MixAttention](https://www.databricks.com/blog/mixattention) | Long Context And KV Cache |
| Databricks | [LLM inference performance engineering: best practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) | Inference Serving At Scale |
| Databricks | [Simple, Fast, Scalable Batch LLM Inference](https://www.databricks.com/blog/introducing-simple-fast-and-scalable-batch-llm-inference-mosaic-ai-model-serving) | Cost Optimization And Model Routing |
| Dropbox | [Building Dash: how RAG and AI agents meet business needs](https://dropbox.tech/machine-learning/building-dash-rag-multi-step-ai-agents-business-users) | RAG Serving |
| Dropbox | [Creating a modern OCR pipeline using CV and deep learning](https://dropbox.tech/machine-learning/creating-a-modern-ocr-pipeline-using-computer-vision-and-deep-learning) | Multimodal Serving |
| Dropbox | [Selecting a model for semantic search at Dropbox scale](https://dropbox.tech/machine-learning/selecting-model-semantic-search-dropbox-ai) | Semantic Search And Embeddings |
| Elastic | [RAG pipelines in production](https://www.elastic.co/search-labs/blog/rag-in-production) | RAG Serving |
| Google | [Announcing ScaNN: efficient vector similarity search](https://research.google/blog/announcing-scann-efficient-vector-similarity-search/) | Semantic Search And Embeddings |
| Google | [Fast inference from transformers via speculative decoding](https://arxiv.org/abs/2211.17192) | Inference Serving At Scale |
| Google | [Flamingo: a visual language model for few-shot learning](https://arxiv.org/abs/2204.14198) | Multimodal Serving |
| Google | [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) | Long Context And KV Cache |
| Google | [PaLI-X: on scaling up a multilingual vision-language model](https://arxiv.org/abs/2305.18565) | Multimodal Serving |
| Google | [RAGO: systematic performance optimization for RAG serving](https://arxiv.org/abs/2503.14649) | RAG Serving |
| Google | [ShieldGemma: generative AI content moderation](https://arxiv.org/abs/2407.21772) | Safety And Guardrails |
| Hugging Face | [Introducing Idefics2: a powerful 8B vision-language model](https://huggingface.co/blog/idefics2) | Multimodal Serving |
| Hugging Face | [Introducing smolagents](https://huggingface.co/blog/smolagents) | Agent Orchestration |
| Hugging Face | [Preference Tuning LLMs with Direct Preference Optimization Methods](https://huggingface.co/blog/pref-tuning) | Post Training Pipeline |
| Hugging Face | [Unlocking longer generation with KV cache quantization](https://huggingface.co/blog/kv-cache-quantization) | Long Context And KV Cache |
| IBM | [LLM routing for quality, low-cost responses](https://research.ibm.com/blog/LLM-routers) | Cost Optimization And Model Routing |
| Meta | [Chameleon: mixed-modal early-fusion foundation models](https://arxiv.org/abs/2405.09818) | Multimodal Serving |
| Meta | [Embedding-based Retrieval in Facebook Search](https://arxiv.org/abs/2006.11632) | Semantic Search And Embeddings |
| Meta | [Faiss: a library for efficient similarity search](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/) | Semantic Search And Embeddings |
| Meta | [How to fine-tune: focus on effective datasets](https://ai.meta.com/blog/how-to-fine-tune-llms-peft-dataset-curation/) | Post Training Pipeline |
| Meta | [Llama Guard: LLM-based input-output safeguard](https://arxiv.org/abs/2312.06674) | Safety And Guardrails |
| Meta | [Llama Prompt Guard 2](https://developer.meta.com/ai/docs/model-cards-and-prompt-formats/prompt-guard/) | Safety And Guardrails |
| Microsoft | [DiskANN: vector search for all](https://www.microsoft.com/en-us/research/project/project-akupara-approximate-nearest-neighbor-search-for-large-scale-semantic-search/) | Semantic Search And Embeddings |
| Microsoft | [Florence-2: a unified representation for vision tasks](https://arxiv.org/abs/2311.06242) | Multimodal Serving |
| Microsoft | [GraphRAG: unlocking LLM discovery on narrative private data](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/) | RAG Serving |
| Microsoft | [How Microsoft defends against indirect prompt injection attacks](https://www.microsoft.com/en-us/msrc/blog/2025/07/how-microsoft-defends-against-indirect-prompt-injection-attacks) | Safety And Guardrails |
| Microsoft | [LLM-Rubric: a multidimensional, calibrated approach to automated evaluation](https://www.microsoft.com/en-us/research/publication/llm-rubric-a-multidimensional-calibrated-approach-to-automated-evaluation-of-natural-language-texts/) | Evaluation System |
| Microsoft | [LLMLingua: prompt compression for LLM efficiency](https://www.microsoft.com/en-us/research/blog/llmlingua-innovating-llm-efficiency-with-prompt-compression/) | Cost Optimization And Model Routing |
| Microsoft | [MInference 1.0: accelerating pre-filling via dynamic sparse attention](https://arxiv.org/abs/2407.02490) | Long Context And KV Cache |
| Microsoft | [Sarathi-Serve: taming the throughput-latency tradeoff](https://arxiv.org/abs/2403.02310) | Inference Serving At Scale |
| Microsoft | [Splitwise: efficient generative LLM inference using phase splitting](https://arxiv.org/abs/2311.18677) | Inference Serving At Scale |
| Microsoft | [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) | Multimodal Serving |
| NVIDIA | [5x faster time to first token with TensorRT-LLM KV cache early reuse](https://developer.nvidia.com/blog/5x-faster-time-to-first-token-with-nvidia-tensorrt-llm-kv-cache-early-reuse/) | Long Context And KV Cache |
| NVIDIA | [Accelerating VLM inference with TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/) | Multimodal Serving |
| NVIDIA | [Content Moderation and Safety Checks with NeMo Guardrails](https://developer.nvidia.com/blog/content-moderation-and-safety-checks-with-nvidia-nemo-guardrails/) | Safety And Guardrails |
| NVIDIA | [How a reranking microservice improves retrieval accuracy and cost](https://developer.nvidia.com/blog/how-using-a-reranking-microservice-can-improve-accuracy-and-costs-of-information-retrieval/) | RAG Serving |
| NVIDIA | [NVIDIA Dynamo: a low-latency distributed inference framework](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) | Inference Serving At Scale |
| NVIDIA | [NVLM: open frontier-class multimodal LLMs](https://research.nvidia.com/labs/adlr/NVLM-1/) | Multimodal Serving |
| NVIDIA | [Optimizing inference with NVFP4 KV cache](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/) | Long Context And KV Cache |
| Red Hat | [vLLM V1: accelerating multimodal inference](https://developers.redhat.com/articles/2025/02/27/vllm-v1-accelerating-multimodal-inference-large-language-models) | Multimodal Serving |
| Salesforce | [BLIP-2: bootstrapping with frozen image encoders and LLMs](https://arxiv.org/abs/2301.12597) | Multimodal Serving |
| Salesforce | [Inside Agentforce: the Atlas Reasoning Engine](https://engineering.salesforce.com/inside-the-brain-of-agentforce-revealing-the-atlas-reasoning-engine/) | Agent Orchestration |
| Salesforce | [Inside the Einstein Trust Layer](https://developer.salesforce.com/blogs/2023/10/inside-the-einstein-trust-layer) | Safety And Guardrails |
| Snowflake | [Arctic Inference with Shift Parallelism](https://www.snowflake.com/en/blog/engineering/arctic-inference-shift-parallelism/) | Inference Serving At Scale |
| Vespa | [Asymmetric Retrieval: spend on docs, embed queries for free](https://blog.vespa.ai/asymmetric-retrieval-spend-on-docs-queries-for-free/) | RAG Serving |
| Vespa | [Billion-scale vector search using hybrid HNSW-IF](https://blog.vespa.ai/vespa-hybrid-billion-scale-vector-search/) | Semantic Search And Embeddings |
| Vespa | [Embedding Tradeoffs, Quantified](https://blog.vespa.ai/embedding-tradeoffs-quantified/) | RAG Serving |

## AI labs and foundation models (17)

| Company | Case study | Topic |
| --- | --- | --- |
| Ai2 | [Molmo and PixMo: open weights and open data for VLMs](https://arxiv.org/abs/2409.17146) | Multimodal Serving |
| Anthropic | [Building effective agents](https://www.anthropic.com/research/building-effective-agents) | Agent Orchestration |
| Anthropic | [Code execution with MCP: building more efficient agents](https://www.anthropic.com/engineering/code-execution-with-mcp) | Agent Orchestration |
| Anthropic | [Constitutional Classifiers: defending against universal jailbreaks](https://www.anthropic.com/research/constitutional-classifiers) | Safety And Guardrails |
| Anthropic | [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) | Agent Orchestration |
| Anthropic | [Prompt caching with Claude](https://claude.com/blog/prompt-caching) | Long Context And KV Cache |
| Anthropic | [Writing effective tools for agents, with agents](https://www.anthropic.com/engineering/writing-tools-for-agents) | Agent Orchestration |
| Character.AI | [Optimizing AI Inference at Character.AI](https://blog.character.ai/optimizing-ai-inference-at-character-ai-2/) | Long Context And KV Cache |
| Character.AI | [Optimizing AI Inference at Character.AI](https://blog.character.ai/optimizing-ai-inference-at-character-ai/) | Inference Serving At Scale |
| Cognition | [Don't Build Multi-Agents](https://cognition.com/blog/dont-build-multi-agents) | Agent Orchestration |
| DeepSeek | [DeepSeek-V2: a strong, economical, efficient MoE language model](https://arxiv.org/abs/2405.04434) | Long Context And KV Cache |
| Mistral | [Pixtral 12B](https://arxiv.org/abs/2410.07073) | Multimodal Serving |
| Moonshot AI | [Mooncake: a KVCache-centric disaggregated architecture](https://arxiv.org/abs/2407.00079) | Inference Serving At Scale |
| OpenAI | [A practical guide to building agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf) | Agent Orchestration |
| OpenAI | [How to implement LLM guardrails](https://developers.openai.com/cookbook/examples/how_to_use_guardrails) | Safety And Guardrails |
| OpenAI | [Updates for developers building with voice](https://developers.openai.com/blog/updates-audio-models) | Realtime Streaming Chat |
| OpenGVLab | [InternVL 2.5: model, data, and test-time scaling](https://arxiv.org/abs/2412.05271) | Multimodal Serving |

## AI infra and developer tools (55)

| Company | Case study | Topic |
| --- | --- | --- |
| Anyscale | [Building an LLM Router for High-Quality and Cost-Effective Responses](https://www.anyscale.com/blog/building-an-llm-router-for-high-quality-and-cost-effective-responses) | Cost Optimization And Model Routing |
| Anyscale | [Building RAG-based LLM Applications for Production](https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-1) | RAG Serving |
| Anyscale | [Direct Preference Optimization with Synthetic Data](https://www.anyscale.com/blog/direct-preference-optimization-with-synthetic-data) | Post Training Pipeline |
| Anyscale | [Fine-Tuning LLMs: LoRA or Full-Parameter?](https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter-an-in-depth-analysis-with-llama-2) | Post Training Pipeline |
| Anyscale | [How continuous batching enables 23x throughput in LLM inference](https://www.anyscale.com/blog/continuous-batching-llm-inference) | Inference Serving At Scale |
| AssemblyAI | [Universal-Streaming: ultra-fast speech-to-text for voice agents](https://www.assemblyai.com/blog/introducing-universal-streaming) | Realtime Streaming Chat |
| Baseten | [33% faster LLM inference with FP8 quantization](https://www.baseten.co/blog/33-faster-llm-inference-with-fp8-quantization/) | Cost Optimization And Model Routing |
| Baseten | [How we built BEI: high-throughput embedding, reranker, classifier inference](https://www.baseten.co/blog/how-we-built-bei-high-throughput-embedding-inference/) | Inference Serving At Scale |
| Baseten | [The Baseten inference stack](https://www.baseten.co/resources/guide/the-baseten-inference-stack/) | Inference Serving At Scale |
| Cartesia | [Announcing Sonic: a low-latency voice model](https://cartesia.ai/blog/sonic) | Realtime Streaming Chat |
| Daily | [Benchmarking LLMs for voice agent use cases](https://www.daily.co/blog/benchmarking-llms-for-voice-agent-use-cases/) | Realtime Streaming Chat |
| Daily | [Smart Turn v3, with CPU inference in 12ms](https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/) | Realtime Streaming Chat |
| Datadog | [Detect hallucinations in your RAG LLM applications](https://www.datadoghq.com/blog/llm-observability-hallucination-detection/) | Production Monitoring And Observability |
| Datadog | [Detecting hallucinations with LLM-as-a-judge](https://www.datadoghq.com/blog/ai/llm-hallucination-detection/) | Production Monitoring And Observability |
| Deepgram | [Optimize voice agent latency with eager end of turn](https://developers.deepgram.com/docs/flux/voice-agent-eager-eot) | Realtime Streaming Chat |
| Discord | [Developing Rapidly with Generative AI](https://discord.com/blog/developing-rapidly-with-generative-ai) | Evaluation System |
| Discord | [How Discord Scaled Elixir to 5,000,000 Concurrent Users](https://discord.com/blog/how-discord-scaled-elixir-to-5-000-000-concurrent-users) | Realtime Streaming Chat |
| Discord | [Our Approach to Content Moderation](https://discord.com/safety/our-approach-to-content-moderation) | Safety And Guardrails |
| ElevenLabs | [Enhancing conversational AI latency with efficient TTS](https://elevenlabs.io/blog/enhancing-conversational-ai-latency-with-efficient-tts-pipelines) | Realtime Streaming Chat |
| Fireworks | [FireAttention V2: long contexts practical for online inference](https://fireworks.ai/blog/fireattention-v2-long-context-inference) | Long Context And KV Cache |
| Fireworks | [FireOptimizer: customizing latency and quality](https://fireworks.ai/blog/fireoptimizer) | Inference Serving At Scale |
| GitHub | [Building a faster, smarter Copilot with a custom model](https://github.blog/ai-and-ml/github-copilot/the-road-to-better-completions-building-a-faster-smarter-github-copilot-with-a-new-custom-model/) | Post Training Pipeline |
| GitHub | [Evaluating the Copilot agentic harness across models and tasks](https://github.blog/ai-and-ml/github-copilot/evaluating-performance-and-efficiency-of-the-github-copilot-agentic-harness-across-models-and-tasks/) | Agent Orchestration |
| GitHub | [How we evaluate AI models and LLMs for GitHub Copilot](https://github.blog/ai-and-ml/generative-ai/how-we-evaluate-models-for-github-copilot/) | Evaluation System |
| GitHub | [Inside Copilot's new code embedding model](https://github.blog/news-insights/product-news/copilot-new-embedding-model-vs-code/) | Semantic Search And Embeddings |
| GitHub | [What is retrieval-augmented generation?](https://github.blog/ai-and-ml/generative-ai/what-is-retrieval-augmented-generation-and-what-does-it-do-for-generative-ai/) | RAG Serving |
| GitLab | [Developing GitLab Duo: validating and testing AI models at scale](https://about.gitlab.com/blog/developing-gitlab-duo-how-we-validate-and-test-ai-models-at-scale/) | Evaluation System |
| Glean | [Why vector search isn't enough for enterprise RAG](https://www.glean.com/blog/hybrid-vs-rag-vector) | RAG Serving |
| Grafana | [Monitor LLMs in production with Grafana Cloud, OpenLIT, and OpenTelemetry](https://grafana.com/blog/ai-observability-llms-in-production/) | Production Monitoring And Observability |
| Grammarly | [CoEdIT: state-of-the-art text editing with fewer parameters](https://www.grammarly.com/blog/engineering/coedit-text-editing/) | Post Training Pipeline |
| Honeycomb | [Improving LLMs in Production With Observability](https://www.honeycomb.io/blog/improving-llms-production-observability) | Production Monitoring And Observability |
| Honeycomb | [So we shipped an AI product. Did it work?](https://www.honeycomb.io/blog/we-shipped-ai-product) | Evaluation System |
| KIVI | [A tuning-free asymmetric 2-bit quantization for KV cache](https://arxiv.org/abs/2402.02750) | Long Context And KV Cache |
| Krisp | [A 6M-weight turn-taking model for voice AI agents](https://krisp.ai/blog/turn-taking-for-voice-ai/) | Realtime Streaming Chat |
| LangChain | [Context Engineering for Agents](https://www.langchain.com/blog/context-engineering-for-agents) | Agent Orchestration |
| LangChain | [The agent improvement loop starts with a trace](https://www.langchain.com/blog/traces-start-agent-improvement-loop) | Production Monitoring And Observability |
| LiveKit | [Why WebRTC beats WebSockets for realtime voice AI](https://livekit.com/blog/why-webrtc-beats-websockets-for-voice-ai-agents) | Realtime Streaming Chat |
| LiveKit | [Why you shouldn't build voice agents directly on model APIs](https://livekit.com/blog/real-time-voice-agents-vs-model-apis) | Realtime Streaming Chat |
| llm-d | [KV-Cache wins you can see: prefix caching to distributed scheduling](https://llm-d.ai/blog/kvcache-wins-you-can-see) | Long Context And KV Cache |
| LMSYS | [RouteLLM: an open framework for cost-effective LLM routing](https://www.lmsys.org/blog/2024-07-01-routellm/) | Cost Optimization And Model Routing |
| LMSYS / SGLang | [Fast and expressive LLM inference with RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/) | Long Context And KV Cache |
| Modal | [High-performance LLM inference](https://modal.com/docs/guide/high-performance-llm-inference) | Inference Serving At Scale |
| Replit | [Enabling Agent 3 to self-test at scale with REPL verification](https://replit.com/blog/automated-self-testing) | Agent Orchestration |
| Replit | [Replit Code v1.5 on Hugging Face](https://replit.com/blog/replit-code-v1_5) | Post Training Pipeline |
| Slack | [Real-time Messaging](https://slack.engineering/real-time-messaging/) | Realtime Streaming Chat |
| Slack | [Securing the Agentic Enterprise](https://slack.com/blog/transformation/securing-the-agentic-enterprise) | Safety And Guardrails |
| Sourcegraph | [Agentic Coding: a practical guide for big code](https://sourcegraph.com/blog/agentic-coding) | Agent Orchestration |
| Stack Overflow | [Vector databases in generative AI applications](https://stackoverflow.blog/2023/10/09/from-prototype-to-production-vector-databases-in-generative-ai-applications/) | Semantic Search And Embeddings |
| Together | [ATLAS: runtime-learning speculative decoding](https://www.together.ai/blog/adaptive-learning-speculator-system-atlas) | Inference Serving At Scale |
| Together | [Serving MiniMax-M3: 1M-token context without regrets](https://www.together.ai/blog/serving-minimax-m3-for-efficient-inference-unlocking-1m-token-context-and-multimodality-without-regrets) | Long Context And KV Cache |
| Twilio | [Instrumenting User Insights for your AI Copilot](https://www.twilio.com/en-us/blog/insights/ai/instrumenting-user-insights-for-your-ai-copilot/) | Production Monitoring And Observability |
| Twilio | [Introducing Media Streams](https://www.twilio.com/en-us/blog/media-streams-public-beta) | Realtime Streaming Chat |
| Vapi | [How we built Vapi's voice AI pipeline (part 2)](https://vapi.ai/blog/how-we-built-vapi-s-voice-ai-pipeline-part-2) | Realtime Streaming Chat |
| Vercel | [Chat SDK brings agents to your users](https://vercel.com/blog/chat-sdk-brings-agents-to-your-users) | Realtime Streaming Chat |
| vLLM | [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) | Long Context And KV Cache |

## E-commerce and retail (10)

| Company | Case study | Topic |
| --- | --- | --- |
| Etsy | [Unified Embedding Based Personalized Retrieval in Etsy Search](https://arxiv.org/abs/2306.04833) | Semantic Search And Embeddings |
| Faire | [Beyond BM25 and dense embeddings: smart, interpretable retrieval](https://craft.faire.com/beyond-bm25-and-dense-embeddings-841a7b18ce27) | Semantic Search And Embeddings |
| Instacart | [How Instacart uses embeddings to improve search relevance](https://company.instacart.com/how-its-made/how-instacart-uses-embeddings-to-improve-search-relevance) | Semantic Search And Embeddings |
| Instacart | [Scaling Catalog Attribute Extraction with Multi-modal LLMs](https://company.instacart.com/tech-innovation/scaling-catalog-attribute-extraction-with-multi-modal-llms) | Evaluation System |
| Mercari | [Domain-Aware Text Embeddings for C2C Marketplaces](https://arxiv.org/abs/2512.21021) | Semantic Search And Embeddings |
| Mercari | [Fine-Tuning an LLM to Extract Dynamically Specified Attributes](https://engineering.mercari.com/en/blog/entry/20240913-fine-tuning-an-llm-to-extract-dynamically-specified-attributes/) | Post Training Pipeline |
| Shopify | [Flow generation through natural language: an agentic modeling approach](https://shopify.engineering/fine-tuning-agent-shopify-flow) | Post Training Pipeline |
| Shopify | [Leveraging multimodal LLMs for the global catalogue](https://shopify.engineering/leveraging-multimodal-llms) | Post Training Pipeline |
| Walmart | [Semantic Retrieval at Walmart](https://arxiv.org/abs/2412.04637) | Semantic Search And Embeddings |
| Wayfair | [How AI understands what you're looking for](https://www.aboutwayfair.com/careers/tech-blog/smarter-shopping-starts-here-how-ai-understands-what-youre-looking-for) | Evaluation System |

## Media and streaming (5)

| Company | Case study | Topic |
| --- | --- | --- |
| Spotify | [Better experiments with LLM evals: a funnel, not a fork](https://engineering.atspotify.com/2026/5/better-experiments-with-llm-evals-a-funnel-not-a-fork) | Evaluation System |
| Spotify | [Introducing Voyager: Spotify new nearest-neighbor search library](https://engineering.atspotify.com/2023/10/introducing-voyager-spotifys-new-nearest-neighbor-search-library) | Semantic Search And Embeddings |
| Spotify | [Optimizing Query Expansions via LLM Preference Alignment](https://research.atspotify.com/2025/7/optimizing-query-expansions-via-llm-preference-alignment) | Post Training Pipeline |
| Spotify | [Profile-aware LLM-as-a-Judge for Podcasts](https://research.atspotify.com/2025/9/profile-aware-llm-as-a-judge-for-podcasts-a-better-middle-ground-between) | Evaluation System |
| Vimeo | [Unlocking knowledge sharing for videos with RAG](https://medium.com/vimeo-engineering-blog/unlocking-knowledge-sharing-for-videos-with-rag-810ab496ae59) | RAG Serving |

## Social platforms (16)

| Company | Case study | Topic |
| --- | --- | --- |
| LinkedIn | [Accelerating LLM inference with speculative decoding](https://www.linkedin.com/blog/engineering/ai/accelerating-llm-inference-with-speculative-decoding-lessons-from-linkedins-hiring-assistant) | Inference Serving At Scale |
| LinkedIn | [Defending Against Abuse at LinkedIn's Scale](https://www.linkedin.com/blog/engineering/trust-and-safety/defending-against-abuse-at-linkedins-scale) | Safety And Guardrails |
| LinkedIn | [How we built domain-adapted foundation GenAI models](https://www.linkedin.com/blog/engineering/generative-ai/how-we-built-domain-adapted-foundation-genai-models-to-power-our-platform) | Post Training Pipeline |
| LinkedIn | [How we engineered LinkedIn's Hiring Assistant](https://www.linkedin.com/blog/engineering/ai/how-we-engineered-linkedins-hiring-assistant) | Evaluation System |
| LinkedIn | [Improving Post Search at LinkedIn](https://www.linkedin.com/blog/engineering/search/improving-post-search-at-linkedin) | RAG Serving |
| LinkedIn | [Musings on building a Generative AI product](https://www.linkedin.com/blog/engineering/generative-ai/musings-on-building-a-generative-ai-product) | Realtime Streaming Chat |
| LinkedIn | [Semantic Search for AI Agents at Scale](https://www.linkedin.com/blog/engineering/ai/semantic-search-for-ai-agents-at-scale-retrieval-and-ranking-for-linkedins-hiring-assistant) | Semantic Search And Embeddings |
| LinkedIn | [The LinkedIn GenAI tech stack: extending to build AI agents](https://www.linkedin.com/blog/engineering/generative-ai/the-linkedin-generative-ai-application-tech-stack-extending-to-build-ai-agents) | Agent Orchestration |
| Pinterest | [Advancements in Embedding-Based Retrieval at Pinterest Homefeed](https://medium.com/pinterest-engineering/advancements-in-embedding-based-retrieval-at-pinterest-homefeed-d7d7971a409e) | Semantic Search And Embeddings |
| Pinterest | [How Pinterest built its Trust & Safety team](https://medium.com/pinterest-engineering/how-pinterest-built-its-trust-safety-team-8d6c026dd4b9) | Safety And Guardrails |
| Pinterest | [How we built Text-to-SQL at Pinterest](https://medium.com/pinterest-engineering/how-we-built-text-to-sql-at-pinterest-30bad30dabff) | RAG Serving |
| Pinterest | [LLM-Powered Relevance Assessment for Pinterest Search](https://medium.com/pinterest-engineering/llm-powered-relevance-assessment-for-pinterest-search-b846489e358d) | Evaluation System |
| Roblox | [Deploying ML for Voice Safety](https://about.roblox.com/newsroom/2024/07/deploying-ml-for-voice-safety) | Safety And Guardrails |
| Roblox | [How Roblox Uses AI to Moderate Content on a Massive Scale](https://about.roblox.com/newsroom/2025/07/roblox-ai-moderation-massive-scale) | Safety And Guardrails |
| Roblox | [Running AI Inference at Scale in the Hybrid Cloud](https://about.roblox.com/newsroom/2024/09/running-ai-inference-at-scale-in-the-hybrid-cloud) | Multimodal Serving |
| Snap | [Embedding-based Retrieval with Two-Tower Models in Spotlight](https://eng.snap.com/embedding-based-retrieval) | Semantic Search And Embeddings |

## Fintech and banking (7)

| Company | Case study | Topic |
| --- | --- | --- |
| Block | [Introducing codename goose: an open framework for AI agents](https://block.xyz/inside/block-open-source-introduces-codename-goose) | Agent Orchestration |
| Nubank | [Fine-Tuning Transaction User Models](https://building.nubank.com/fine-tuning-transaction-user-models/) | Post Training Pipeline |
| Ramp | [From RAG to Richness: How Ramp Revamped Industry Classification](https://builders.ramp.com/post/industry_classification) | RAG Serving |
| Ramp | [How Ramp Fixes Merchant Matches with AI](https://builders.ramp.com/post/fixing-merchant-classifications-with-ai) | Evaluation System |
| Ramp | [Why We Built Our Own Background Agent](https://builders.ramp.com/post/why-we-built-our-background-agent) | Agent Orchestration |
| Stripe | [Can AI agents build real Stripe integrations?](https://stripe.com/blog/can-ai-agents-build-real-stripe-integrations) | Agent Orchestration |
| Wealthsimple | [Our LLM Gateway for secure, reliable generative AI](https://engineering.wealthsimple.com/get-to-know-our-llm-gateway-and-how-it-provides-a-secure-and-reliable-space-to-use-generative-ai) | Safety And Guardrails |

## Delivery and mobility (13)

| Company | Case study | Topic |
| --- | --- | --- |
| DoorDash | [A Simulation and Evaluation Flywheel to Develop LLM Chatbots at Scale](https://careersatdoordash.com/blog/doordash-simulation-evaluation-flywheel-to-develop-llm-chatbots-at-scale/) | Evaluation System |
| DoorDash | [How DoorDash leverages LLMs to evaluate search result pages](https://careersatdoordash.com/blog/doordash-llms-to-evaluate-search-result-pages/) | Evaluation System |
| DoorDash | [Path to high-quality LLM-based Dasher support automation](https://careersatdoordash.com/blog/large-language-modules-based-dasher-support-automation/) | RAG Serving |
| Grab | [A custom vision LLM to improve document processing](https://engineering.grab.com/custom-vision-llm-at-grab) | Post Training Pipeline |
| Grab | [How LLMs make content moderation more precise](https://www.grab.com/inside-grab/stories/how-large-language-models-help-us-make-more-precise-content-moderation-decisions/) | Safety And Guardrails |
| Uber | [Enhanced Agentic-RAG: near-human precision for chatbots](https://www.uber.com/blog/enhanced-agentic-rag/) | RAG Serving |
| Uber | [From Predictive to Generative: how Michelangelo accelerates Uber AI](https://www.uber.com/blog/from-predictive-to-generative-ai/) | Evaluation System |
| Uber | [Genie: Uber's Gen AI on-call copilot](https://www.uber.com/en-US/blog/genie-ubers-gen-ai-on-call-copilot/) | Agent Orchestration |
| Uber | [Genie: Uber's Gen AI On-Call Copilot](https://www.uber.com/us/en/blog/genie-ubers-gen-ai-on-call-copilot/) | Production Monitoring And Observability |
| Uber | [Open Source and In-House: How Uber Optimizes LLM Training](https://www.uber.com/us/en/blog/open-source-and-in-house-how-uber-optimizes-llm-training/) | Post Training Pipeline |
| Uber | [Scaling Multilingual Semantic Search in Uber Eats](https://arxiv.org/abs/2603.06586) | Semantic Search And Embeddings |
| Uber | [Uber's GenAI Gateway](https://www.uber.com/blog/genai-gateway/) | Cost Optimization And Model Routing |
| Uber | [uReview: scalable, trustworthy GenAI for code review](https://www.uber.com/us/en/blog/ureview/) | Evaluation System |

## Travel and hospitality (3)

| Company | Case study | Topic |
| --- | --- | --- |
| Airbnb | [Applying Embedding-Based Retrieval to Airbnb Search](https://arxiv.org/abs/2601.06873) | Semantic Search And Embeddings |
| Airbnb | [Automation Platform v2: improving conversational AI](https://medium.com/airbnb-engineering/automation-platform-v2-improving-conversational-ai-at-airbnb-d86c9386e0cb) | Agent Orchestration |
| Booking.com | [LLM Evaluation: practical tips at Booking.com](https://booking.ai/llm-evaluation-practical-tips-at-booking-com-1b038a0d6662) | Evaluation System |

## Professional, legal, and education (5)

| Company | Case study | Topic |
| --- | --- | --- |
| Thomson Reuters | [Efficiently evaluating LLMs for legal tasks](https://legal.thomsonreuters.com/blog/evaluating-llms-legal-tasks/) | Evaluation System |
| Thomson Reuters | [Inside CoCounsel's guardrails](https://legal.thomsonreuters.com/blog/why-your-legal-ai-needs-more-than-the-open-web-a-look-inside-cocounsels-guardrails/) | Safety And Guardrails |
| Thomson Reuters | [Scaling LLM research with Amazon SageMaker HyperPod](https://aws.amazon.com/blogs/machine-learning/scaling-thomson-reuters-language-model-research-with-amazon-sagemaker-hyperpod/) | Post Training Pipeline |
| Yelp | [An AI pipeline for inappropriate-language detection in reviews](https://engineeringblog.yelp.com/2024/03/ai-pipeline-inappropriate-language-detection.html) | Post Training Pipeline |
| Yelp | [Yelp Content As Embeddings](https://engineeringblog.yelp.com/2023/04/yelp-content-as-embeddings.html) | Semantic Search And Embeddings |

## Research and academia (10)

| Company | Case study | Topic |
| --- | --- | --- |
| Colfax / Together | [FlashAttention-3: fast, accurate attention with asynchrony and low precision](https://arxiv.org/abs/2407.08608) | Long Context And KV Cache |
| MIT / Meta | [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) | Long Context And KV Cache |
| Peking University / UCSD | [DistServe: disaggregating prefill and decoding](https://arxiv.org/abs/2401.09670) | Inference Serving At Scale |
| Shinn et al. | [Reflexion: language agents with verbal reinforcement learning](https://arxiv.org/abs/2303.11366) | Agent Orchestration |
| Stanford | [FrugalGPT: Using LLMs While Reducing Cost and Improving Performance](https://arxiv.org/abs/2305.05176) | Cost Optimization And Model Routing |
| UIUC / Cohere | [SnapKV: LLM knows what you are looking for before generation](https://arxiv.org/abs/2404.14469) | Long Context And KV Cache |
| UT Austin / Stanford | [H2O: Heavy-Hitter Oracle for efficient generative inference](https://arxiv.org/abs/2306.14048) | Long Context And KV Cache |
| Wang et al. | [Voyager: an open-ended embodied agent with LLMs](https://arxiv.org/abs/2305.16291) | Agent Orchestration |
| Wu et al. | [AutoGen: next-gen LLM apps via multi-agent conversation](https://arxiv.org/abs/2308.08155) | Agent Orchestration |
| Yao et al. | [ReAct: synergizing reasoning and acting in language models](https://arxiv.org/abs/2210.03629) | Agent Orchestration |
