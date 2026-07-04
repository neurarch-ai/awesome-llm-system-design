# Case studies, by company

The same shipped systems as [CASE-STUDIES.md](CASE-STUDIES.md), pivoted by company so you can see how one org approaches the same LLM problem across problems. 205 case studies, 95 companies. Deep teardowns of many of these live in [CASE-TEARDOWNS.md](CASE-TEARDOWNS.md).

---

### Microsoft (10)

- [GraphRAG: unlocking LLM discovery on narrative private data](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/) *(RAG Serving)*
- [MInference 1.0: accelerating pre-filling via dynamic sparse attention](https://arxiv.org/abs/2407.02490) *(Long Context And KV Cache)*
- [Splitwise: efficient generative LLM inference using phase splitting](https://arxiv.org/abs/2311.18677) *(Inference Serving At Scale)*
- [Sarathi-Serve: taming the throughput-latency tradeoff](https://arxiv.org/abs/2403.02310) *(Inference Serving At Scale)*
- [LLM-Rubric: a multidimensional, calibrated approach to automated evaluation](https://www.microsoft.com/en-us/research/publication/llm-rubric-a-multidimensional-calibrated-approach-to-automated-evaluation-of-natural-language-texts/) *(Evaluation System)*
- [How Microsoft defends against indirect prompt injection attacks](https://www.microsoft.com/en-us/msrc/blog/2025/07/how-microsoft-defends-against-indirect-prompt-injection-attacks) *(Safety And Guardrails)*
- [DiskANN: vector search for all](https://www.microsoft.com/en-us/research/project/project-akupara-approximate-nearest-neighbor-search-for-large-scale-semantic-search/) *(Semantic Search And Embeddings)*
- [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485) *(Multimodal Serving)*
- [Florence-2: a unified representation for vision tasks](https://arxiv.org/abs/2311.06242) *(Multimodal Serving)*
- [LLMLingua: prompt compression for LLM efficiency](https://www.microsoft.com/en-us/research/blog/llmlingua-innovating-llm-efficiency-with-prompt-compression/) *(Cost Optimization And Model Routing)*

### LinkedIn (8)

- [Improving Post Search at LinkedIn](https://www.linkedin.com/blog/engineering/search/improving-post-search-at-linkedin) *(RAG Serving)*
- [The LinkedIn GenAI tech stack: extending to build AI agents](https://www.linkedin.com/blog/engineering/generative-ai/the-linkedin-generative-ai-application-tech-stack-extending-to-build-ai-agents) *(Agent Orchestration)*
- [Accelerating LLM inference with speculative decoding](https://www.linkedin.com/blog/engineering/ai/accelerating-llm-inference-with-speculative-decoding-lessons-from-linkedins-hiring-assistant) *(Inference Serving At Scale)*
- [How we built domain-adapted foundation GenAI models](https://www.linkedin.com/blog/engineering/generative-ai/how-we-built-domain-adapted-foundation-genai-models-to-power-our-platform) *(Post Training Pipeline)*
- [How we engineered LinkedIn's Hiring Assistant](https://www.linkedin.com/blog/engineering/ai/how-we-engineered-linkedins-hiring-assistant) *(Evaluation System)*
- [Defending Against Abuse at LinkedIn's Scale](https://www.linkedin.com/blog/engineering/trust-and-safety/defending-against-abuse-at-linkedins-scale) *(Safety And Guardrails)*
- [Semantic Search for AI Agents at Scale](https://www.linkedin.com/blog/engineering/ai/semantic-search-for-ai-agents-at-scale-retrieval-and-ranking-for-linkedins-hiring-assistant) *(Semantic Search And Embeddings)*
- [Musings on building a Generative AI product](https://www.linkedin.com/blog/engineering/generative-ai/musings-on-building-a-generative-ai-product) *(Realtime Streaming Chat)*

### Uber (8)

- [Enhanced Agentic-RAG: near-human precision for chatbots](https://www.uber.com/blog/enhanced-agentic-rag/) *(RAG Serving)*
- [Genie: Uber's Gen AI on-call copilot](https://www.uber.com/en-US/blog/genie-ubers-gen-ai-on-call-copilot/) *(Agent Orchestration)*
- [Open Source and In-House: How Uber Optimizes LLM Training](https://www.uber.com/us/en/blog/open-source-and-in-house-how-uber-optimizes-llm-training/) *(Post Training Pipeline)*
- [uReview: scalable, trustworthy GenAI for code review](https://www.uber.com/us/en/blog/ureview/) *(Evaluation System)*
- [From Predictive to Generative: how Michelangelo accelerates Uber AI](https://www.uber.com/blog/from-predictive-to-generative-ai/) *(Evaluation System)*
- [Scaling Multilingual Semantic Search in Uber Eats](https://arxiv.org/abs/2603.06586) *(Semantic Search And Embeddings)*
- [Uber's GenAI Gateway](https://www.uber.com/blog/genai-gateway/) *(Cost Optimization And Model Routing)*
- [Genie: Uber's Gen AI On-Call Copilot](https://www.uber.com/us/en/blog/genie-ubers-gen-ai-on-call-copilot/) *(Production Monitoring And Observability)*

### Databricks (7)

- [Creating High Quality RAG Applications with Databricks](https://www.databricks.com/blog/building-high-quality-rag-applications-databricks) *(RAG Serving)*
- [Inference-Friendly Models with MixAttention](https://www.databricks.com/blog/mixattention) *(Long Context And KV Cache)*
- [Accelerating LLM inference with prompt caching](https://www.databricks.com/blog/accelerating-llm-inference-prompt-caching-open-source-models-databricks) *(Long Context And KV Cache)*
- [LLM inference performance engineering: best practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices) *(Inference Serving At Scale)*
- [A Practical Guide to LLM Fine Tuning](https://www.databricks.com/blog/llm-fine-tuning) *(Post Training Pipeline)*
- [Implementing LLM Guardrails for Safe GenAI Deployment](https://www.databricks.com/blog/implementing-llm-guardrails-safe-and-responsible-generative-ai-deployment-databricks) *(Safety And Guardrails)*
- [Simple, Fast, Scalable Batch LLM Inference](https://www.databricks.com/blog/introducing-simple-fast-and-scalable-batch-llm-inference-mosaic-ai-model-serving) *(Cost Optimization And Model Routing)*

### Google (7)

- [RAGO: systematic performance optimization for RAG serving](https://arxiv.org/abs/2503.14649) *(RAG Serving)*
- [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245) *(Long Context And KV Cache)*
- [Fast inference from transformers via speculative decoding](https://arxiv.org/abs/2211.17192) *(Inference Serving At Scale)*
- [ShieldGemma: generative AI content moderation](https://arxiv.org/abs/2407.21772) *(Safety And Guardrails)*
- [Announcing ScaNN: efficient vector similarity search](https://research.google/blog/announcing-scann-efficient-vector-similarity-search/) *(Semantic Search And Embeddings)*
- [Flamingo: a visual language model for few-shot learning](https://arxiv.org/abs/2204.14198) *(Multimodal Serving)*
- [PaLI-X: on scaling up a multilingual vision-language model](https://arxiv.org/abs/2305.18565) *(Multimodal Serving)*

### NVIDIA (7)

- [How a reranking microservice improves retrieval accuracy and cost](https://developer.nvidia.com/blog/how-using-a-reranking-microservice-can-improve-accuracy-and-costs-of-information-retrieval/) *(RAG Serving)*
- [5x faster time to first token with TensorRT-LLM KV cache early reuse](https://developer.nvidia.com/blog/5x-faster-time-to-first-token-with-nvidia-tensorrt-llm-kv-cache-early-reuse/) *(Long Context And KV Cache)*
- [Optimizing inference with NVFP4 KV cache](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/) *(Long Context And KV Cache)*
- [NVIDIA Dynamo: a low-latency distributed inference framework](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/) *(Inference Serving At Scale)*
- [Content Moderation and Safety Checks with NeMo Guardrails](https://developer.nvidia.com/blog/content-moderation-and-safety-checks-with-nvidia-nemo-guardrails/) *(Safety And Guardrails)*
- [NVLM: open frontier-class multimodal LLMs](https://research.nvidia.com/labs/adlr/NVLM-1/) *(Multimodal Serving)*
- [Accelerating VLM inference with TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/) *(Multimodal Serving)*

### Anthropic (6)

- [Prompt caching with Claude](https://claude.com/blog/prompt-caching) *(Long Context And KV Cache)*
- [Building effective agents](https://www.anthropic.com/research/building-effective-agents) *(Agent Orchestration)*
- [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) *(Agent Orchestration)*
- [Writing effective tools for agents, with agents](https://www.anthropic.com/engineering/writing-tools-for-agents) *(Agent Orchestration)*
- [Code execution with MCP: building more efficient agents](https://www.anthropic.com/engineering/code-execution-with-mcp) *(Agent Orchestration)*
- [Constitutional Classifiers: defending against universal jailbreaks](https://www.anthropic.com/research/constitutional-classifiers) *(Safety And Guardrails)*

### Meta (6)

- [How to fine-tune: focus on effective datasets](https://ai.meta.com/blog/how-to-fine-tune-llms-peft-dataset-curation/) *(Post Training Pipeline)*
- [Llama Guard: LLM-based input-output safeguard](https://arxiv.org/abs/2312.06674) *(Safety And Guardrails)*
- [Llama Prompt Guard 2](https://developer.meta.com/ai/docs/model-cards-and-prompt-formats/prompt-guard/) *(Safety And Guardrails)*
- [Faiss: a library for efficient similarity search](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/) *(Semantic Search And Embeddings)*
- [Embedding-based Retrieval in Facebook Search](https://arxiv.org/abs/2006.11632) *(Semantic Search And Embeddings)*
- [Chameleon: mixed-modal early-fusion foundation models](https://arxiv.org/abs/2405.09818) *(Multimodal Serving)*

### Anyscale (5)

- [Building RAG-based LLM Applications for Production](https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-1) *(RAG Serving)*
- [How continuous batching enables 23x throughput in LLM inference](https://www.anyscale.com/blog/continuous-batching-llm-inference) *(Inference Serving At Scale)*
- [Fine-Tuning LLMs: LoRA or Full-Parameter?](https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter-an-in-depth-analysis-with-llama-2) *(Post Training Pipeline)*
- [Direct Preference Optimization with Synthetic Data](https://www.anyscale.com/blog/direct-preference-optimization-with-synthetic-data) *(Post Training Pipeline)*
- [Building an LLM Router for High-Quality and Cost-Effective Responses](https://www.anyscale.com/blog/building-an-llm-router-for-high-quality-and-cost-effective-responses) *(Cost Optimization And Model Routing)*

### Cloudflare (5)

- [Introducing AutoRAG: managed RAG on Cloudflare](https://blog.cloudflare.com/introducing-autorag-on-cloudflare/) *(RAG Serving)*
- [Running fine-tuned models on Workers AI with LoRAs](https://blog.cloudflare.com/fine-tuned-inference-with-loras/) *(Post Training Pipeline)*
- [Block unsafe prompts with Firewall for AI](https://blog.cloudflare.com/block-unsafe-llm-prompts-with-firewall-for-ai/) *(Safety And Guardrails)*
- [Durable Objects for WebSockets and auth in AI Gateway](https://blog.cloudflare.com/do-it-again/) *(Realtime Streaming Chat)*
- [Caching in AI Gateway](https://developers.cloudflare.com/ai-gateway/features/caching/) *(Cost Optimization And Model Routing)*

### GitHub (5)

- [What is retrieval-augmented generation?](https://github.blog/ai-and-ml/generative-ai/what-is-retrieval-augmented-generation-and-what-does-it-do-for-generative-ai/) *(RAG Serving)*
- [Evaluating the Copilot agentic harness across models and tasks](https://github.blog/ai-and-ml/github-copilot/evaluating-performance-and-efficiency-of-the-github-copilot-agentic-harness-across-models-and-tasks/) *(Agent Orchestration)*
- [Building a faster, smarter Copilot with a custom model](https://github.blog/ai-and-ml/github-copilot/the-road-to-better-completions-building-a-faster-smarter-github-copilot-with-a-new-custom-model/) *(Post Training Pipeline)*
- [How we evaluate AI models and LLMs for GitHub Copilot](https://github.blog/ai-and-ml/generative-ai/how-we-evaluate-models-for-github-copilot/) *(Evaluation System)*
- [Inside Copilot's new code embedding model](https://github.blog/news-insights/product-news/copilot-new-embedding-model-vs-code/) *(Semantic Search And Embeddings)*

### Hugging Face (4)

- [Unlocking longer generation with KV cache quantization](https://huggingface.co/blog/kv-cache-quantization) *(Long Context And KV Cache)*
- [Introducing smolagents](https://huggingface.co/blog/smolagents) *(Agent Orchestration)*
- [Preference Tuning LLMs with Direct Preference Optimization Methods](https://huggingface.co/blog/pref-tuning) *(Post Training Pipeline)*
- [Introducing Idefics2: a powerful 8B vision-language model](https://huggingface.co/blog/idefics2) *(Multimodal Serving)*

### Pinterest (4)

- [How we built Text-to-SQL at Pinterest](https://medium.com/pinterest-engineering/how-we-built-text-to-sql-at-pinterest-30bad30dabff) *(RAG Serving)*
- [LLM-Powered Relevance Assessment for Pinterest Search](https://medium.com/pinterest-engineering/llm-powered-relevance-assessment-for-pinterest-search-b846489e358d) *(Evaluation System)*
- [How Pinterest built its Trust & Safety team](https://medium.com/pinterest-engineering/how-pinterest-built-its-trust-safety-team-8d6c026dd4b9) *(Safety And Guardrails)*
- [Advancements in Embedding-Based Retrieval at Pinterest Homefeed](https://medium.com/pinterest-engineering/advancements-in-embedding-based-retrieval-at-pinterest-homefeed-d7d7971a409e) *(Semantic Search And Embeddings)*

### Spotify (4)

- [Optimizing Query Expansions via LLM Preference Alignment](https://research.atspotify.com/2025/7/optimizing-query-expansions-via-llm-preference-alignment) *(Post Training Pipeline)*
- [Better experiments with LLM evals: a funnel, not a fork](https://engineering.atspotify.com/2026/5/better-experiments-with-llm-evals-a-funnel-not-a-fork) *(Evaluation System)*
- [Profile-aware LLM-as-a-Judge for Podcasts](https://research.atspotify.com/2025/9/profile-aware-llm-as-a-judge-for-podcasts-a-better-middle-ground-between) *(Evaluation System)*
- [Introducing Voyager: Spotify new nearest-neighbor search library](https://engineering.atspotify.com/2023/10/introducing-voyager-spotifys-new-nearest-neighbor-search-library) *(Semantic Search And Embeddings)*

### Baseten (3)

- [How we built BEI: high-throughput embedding, reranker, classifier inference](https://www.baseten.co/blog/how-we-built-bei-high-throughput-embedding-inference/) *(Inference Serving At Scale)*
- [The Baseten inference stack](https://www.baseten.co/resources/guide/the-baseten-inference-stack/) *(Inference Serving At Scale)*
- [33% faster LLM inference with FP8 quantization](https://www.baseten.co/blog/33-faster-llm-inference-with-fp8-quantization/) *(Cost Optimization And Model Routing)*

### Discord (3)

- [Developing Rapidly with Generative AI](https://discord.com/blog/developing-rapidly-with-generative-ai) *(Evaluation System)*
- [Our Approach to Content Moderation](https://discord.com/safety/our-approach-to-content-moderation) *(Safety And Guardrails)*
- [How Discord Scaled Elixir to 5,000,000 Concurrent Users](https://discord.com/blog/how-discord-scaled-elixir-to-5-000-000-concurrent-users) *(Realtime Streaming Chat)*

### DoorDash (3)

- [Path to high-quality LLM-based Dasher support automation](https://careersatdoordash.com/blog/large-language-modules-based-dasher-support-automation/) *(RAG Serving)*
- [A Simulation and Evaluation Flywheel to Develop LLM Chatbots at Scale](https://careersatdoordash.com/blog/doordash-simulation-evaluation-flywheel-to-develop-llm-chatbots-at-scale/) *(Evaluation System)*
- [How DoorDash leverages LLMs to evaluate search result pages](https://careersatdoordash.com/blog/doordash-llms-to-evaluate-search-result-pages/) *(Evaluation System)*

### Dropbox (3)

- [Building Dash: how RAG and AI agents meet business needs](https://dropbox.tech/machine-learning/building-dash-rag-multi-step-ai-agents-business-users) *(RAG Serving)*
- [Selecting a model for semantic search at Dropbox scale](https://dropbox.tech/machine-learning/selecting-model-semantic-search-dropbox-ai) *(Semantic Search And Embeddings)*
- [Creating a modern OCR pipeline using CV and deep learning](https://dropbox.tech/machine-learning/creating-a-modern-ocr-pipeline-using-computer-vision-and-deep-learning) *(Multimodal Serving)*

### OpenAI (3)

- [A practical guide to building agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf) *(Agent Orchestration)*
- [How to implement LLM guardrails](https://developers.openai.com/cookbook/examples/how_to_use_guardrails) *(Safety And Guardrails)*
- [Updates for developers building with voice](https://developers.openai.com/blog/updates-audio-models) *(Realtime Streaming Chat)*

### Ramp (3)

- [From RAG to Richness: How Ramp Revamped Industry Classification](https://builders.ramp.com/post/industry_classification) *(RAG Serving)*
- [Why We Built Our Own Background Agent](https://builders.ramp.com/post/why-we-built-our-background-agent) *(Agent Orchestration)*
- [How Ramp Fixes Merchant Matches with AI](https://builders.ramp.com/post/fixing-merchant-classifications-with-ai) *(Evaluation System)*

### Roblox (3)

- [How Roblox Uses AI to Moderate Content on a Massive Scale](https://about.roblox.com/newsroom/2025/07/roblox-ai-moderation-massive-scale) *(Safety And Guardrails)*
- [Deploying ML for Voice Safety](https://about.roblox.com/newsroom/2024/07/deploying-ml-for-voice-safety) *(Safety And Guardrails)*
- [Running AI Inference at Scale in the Hybrid Cloud](https://about.roblox.com/newsroom/2024/09/running-ai-inference-at-scale-in-the-hybrid-cloud) *(Multimodal Serving)*

### Salesforce (3)

- [Inside Agentforce: the Atlas Reasoning Engine](https://engineering.salesforce.com/inside-the-brain-of-agentforce-revealing-the-atlas-reasoning-engine/) *(Agent Orchestration)*
- [Inside the Einstein Trust Layer](https://developer.salesforce.com/blogs/2023/10/inside-the-einstein-trust-layer) *(Safety And Guardrails)*
- [BLIP-2: bootstrapping with frozen image encoders and LLMs](https://arxiv.org/abs/2301.12597) *(Multimodal Serving)*

### Thomson Reuters (3)

- [Scaling LLM research with Amazon SageMaker HyperPod](https://aws.amazon.com/blogs/machine-learning/scaling-thomson-reuters-language-model-research-with-amazon-sagemaker-hyperpod/) *(Post Training Pipeline)*
- [Efficiently evaluating LLMs for legal tasks](https://legal.thomsonreuters.com/blog/evaluating-llms-legal-tasks/) *(Evaluation System)*
- [Inside CoCounsel's guardrails](https://legal.thomsonreuters.com/blog/why-your-legal-ai-needs-more-than-the-open-web-a-look-inside-cocounsels-guardrails/) *(Safety And Guardrails)*

### Vespa (3)

- [Embedding Tradeoffs, Quantified](https://blog.vespa.ai/embedding-tradeoffs-quantified/) *(RAG Serving)*
- [Asymmetric Retrieval: spend on docs, embed queries for free](https://blog.vespa.ai/asymmetric-retrieval-spend-on-docs-queries-for-free/) *(RAG Serving)*
- [Billion-scale vector search using hybrid HNSW-IF](https://blog.vespa.ai/vespa-hybrid-billion-scale-vector-search/) *(Semantic Search And Embeddings)*

### Airbnb (2)

- [Automation Platform v2: improving conversational AI](https://medium.com/airbnb-engineering/automation-platform-v2-improving-conversational-ai-at-airbnb-d86c9386e0cb) *(Agent Orchestration)*
- [Applying Embedding-Based Retrieval to Airbnb Search](https://arxiv.org/abs/2601.06873) *(Semantic Search And Embeddings)*

### Alibaba (2)

- [Qwen2-VL: enhancing vision-language perception at any resolution](https://arxiv.org/abs/2409.12191) *(Multimodal Serving)*
- [Qwen2-Audio Technical Report](https://arxiv.org/abs/2407.10759) *(Multimodal Serving)*

### Character.AI (2)

- [Optimizing AI Inference at Character.AI](https://blog.character.ai/optimizing-ai-inference-at-character-ai-2/) *(Long Context And KV Cache)*
- [Optimizing AI Inference at Character.AI](https://blog.character.ai/optimizing-ai-inference-at-character-ai/) *(Inference Serving At Scale)*

### Daily (2)

- [Benchmarking LLMs for voice agent use cases](https://www.daily.co/blog/benchmarking-llms-for-voice-agent-use-cases/) *(Realtime Streaming Chat)*
- [Smart Turn v3, with CPU inference in 12ms](https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/) *(Realtime Streaming Chat)*

### Datadog (2)

- [Detect hallucinations in your RAG LLM applications](https://www.datadoghq.com/blog/llm-observability-hallucination-detection/) *(Production Monitoring And Observability)*
- [Detecting hallucinations with LLM-as-a-judge](https://www.datadoghq.com/blog/ai/llm-hallucination-detection/) *(Production Monitoring And Observability)*

### Fireworks (2)

- [FireAttention V2: long contexts practical for online inference](https://fireworks.ai/blog/fireattention-v2-long-context-inference) *(Long Context And KV Cache)*
- [FireOptimizer: customizing latency and quality](https://fireworks.ai/blog/fireoptimizer) *(Inference Serving At Scale)*

### Grab (2)

- [A custom vision LLM to improve document processing](https://engineering.grab.com/custom-vision-llm-at-grab) *(Post Training Pipeline)*
- [How LLMs make content moderation more precise](https://www.grab.com/inside-grab/stories/how-large-language-models-help-us-make-more-precise-content-moderation-decisions/) *(Safety And Guardrails)*

### Honeycomb (2)

- [So we shipped an AI product. Did it work?](https://www.honeycomb.io/blog/we-shipped-ai-product) *(Evaluation System)*
- [Improving LLMs in Production With Observability](https://www.honeycomb.io/blog/improving-llms-production-observability) *(Production Monitoring And Observability)*

### Instacart (2)

- [Scaling Catalog Attribute Extraction with Multi-modal LLMs](https://company.instacart.com/tech-innovation/scaling-catalog-attribute-extraction-with-multi-modal-llms) *(Evaluation System)*
- [How Instacart uses embeddings to improve search relevance](https://company.instacart.com/how-its-made/how-instacart-uses-embeddings-to-improve-search-relevance) *(Semantic Search And Embeddings)*

### LangChain (2)

- [Context Engineering for Agents](https://www.langchain.com/blog/context-engineering-for-agents) *(Agent Orchestration)*
- [The agent improvement loop starts with a trace](https://www.langchain.com/blog/traces-start-agent-improvement-loop) *(Production Monitoring And Observability)*

### LiveKit (2)

- [Why WebRTC beats WebSockets for realtime voice AI](https://livekit.com/blog/why-webrtc-beats-websockets-for-voice-ai-agents) *(Realtime Streaming Chat)*
- [Why you shouldn't build voice agents directly on model APIs](https://livekit.com/blog/real-time-voice-agents-vs-model-apis) *(Realtime Streaming Chat)*

### Mercari (2)

- [Fine-Tuning an LLM to Extract Dynamically Specified Attributes](https://engineering.mercari.com/en/blog/entry/20240913-fine-tuning-an-llm-to-extract-dynamically-specified-attributes/) *(Post Training Pipeline)*
- [Domain-Aware Text Embeddings for C2C Marketplaces](https://arxiv.org/abs/2512.21021) *(Semantic Search And Embeddings)*

### Replit (2)

- [Enabling Agent 3 to self-test at scale with REPL verification](https://replit.com/blog/automated-self-testing) *(Agent Orchestration)*
- [Replit Code v1.5 on Hugging Face](https://replit.com/blog/replit-code-v1_5) *(Post Training Pipeline)*

### Shopify (2)

- [Flow generation through natural language: an agentic modeling approach](https://shopify.engineering/fine-tuning-agent-shopify-flow) *(Post Training Pipeline)*
- [Leveraging multimodal LLMs for the global catalogue](https://shopify.engineering/leveraging-multimodal-llms) *(Post Training Pipeline)*

### Slack (2)

- [Securing the Agentic Enterprise](https://slack.com/blog/transformation/securing-the-agentic-enterprise) *(Safety And Guardrails)*
- [Real-time Messaging](https://slack.engineering/real-time-messaging/) *(Realtime Streaming Chat)*

### Together (2)

- [Serving MiniMax-M3: 1M-token context without regrets](https://www.together.ai/blog/serving-minimax-m3-for-efficient-inference-unlocking-1m-token-context-and-multimodality-without-regrets) *(Long Context And KV Cache)*
- [ATLAS: runtime-learning speculative decoding](https://www.together.ai/blog/adaptive-learning-speculator-system-atlas) *(Inference Serving At Scale)*

### Twilio (2)

- [Introducing Media Streams](https://www.twilio.com/en-us/blog/media-streams-public-beta) *(Realtime Streaming Chat)*
- [Instrumenting User Insights for your AI Copilot](https://www.twilio.com/en-us/blog/insights/ai/instrumenting-user-insights-for-your-ai-copilot/) *(Production Monitoring And Observability)*

### Yelp (2)

- [An AI pipeline for inappropriate-language detection in reviews](https://engineeringblog.yelp.com/2024/03/ai-pipeline-inappropriate-language-detection.html) *(Post Training Pipeline)*
- [Yelp Content As Embeddings](https://engineeringblog.yelp.com/2023/04/yelp-content-as-embeddings.html) *(Semantic Search And Embeddings)*

### Ai2 (1)

- [Molmo and PixMo: open weights and open data for VLMs](https://arxiv.org/abs/2409.17146) *(Multimodal Serving)*

### Amazon (1)

- [Semantic Product Search](https://arxiv.org/abs/1907.00937) *(Semantic Search And Embeddings)*

### AMD (1)

- [Accelerating Multimodal Inference in vLLM](https://rocm.blogs.amd.com/software-tools-optimization/vllm-dp-vision/README.html) *(Multimodal Serving)*

### Apple (1)

- [MM1: methods, analysis, and insights from multimodal LLM pre-training](https://arxiv.org/abs/2403.09611) *(Multimodal Serving)*

### AssemblyAI (1)

- [Universal-Streaming: ultra-fast speech-to-text for voice agents](https://www.assemblyai.com/blog/introducing-universal-streaming) *(Realtime Streaming Chat)*

### Block (1)

- [Introducing codename goose: an open framework for AI agents](https://block.xyz/inside/block-open-source-introduces-codename-goose) *(Agent Orchestration)*

### Booking.com (1)

- [LLM Evaluation: practical tips at Booking.com](https://booking.ai/llm-evaluation-practical-tips-at-booking-com-1b038a0d6662) *(Evaluation System)*

### Cartesia (1)

- [Announcing Sonic: a low-latency voice model](https://cartesia.ai/blog/sonic) *(Realtime Streaming Chat)*

### Cognition (1)

- [Don't Build Multi-Agents](https://cognition.com/blog/dont-build-multi-agents) *(Agent Orchestration)*

### Colfax / Together (1)

- [FlashAttention-3: fast, accurate attention with asynchrony and low precision](https://arxiv.org/abs/2407.08608) *(Long Context And KV Cache)*

### Deepgram (1)

- [Optimize voice agent latency with eager end of turn](https://developers.deepgram.com/docs/flux/voice-agent-eager-eot) *(Realtime Streaming Chat)*

### DeepSeek (1)

- [DeepSeek-V2: a strong, economical, efficient MoE language model](https://arxiv.org/abs/2405.04434) *(Long Context And KV Cache)*

### Elastic (1)

- [RAG pipelines in production](https://www.elastic.co/search-labs/blog/rag-in-production) *(RAG Serving)*

### ElevenLabs (1)

- [Enhancing conversational AI latency with efficient TTS](https://elevenlabs.io/blog/enhancing-conversational-ai-latency-with-efficient-tts-pipelines) *(Realtime Streaming Chat)*

### Etsy (1)

- [Unified Embedding Based Personalized Retrieval in Etsy Search](https://arxiv.org/abs/2306.04833) *(Semantic Search And Embeddings)*

### Faire (1)

- [Beyond BM25 and dense embeddings: smart, interpretable retrieval](https://craft.faire.com/beyond-bm25-and-dense-embeddings-841a7b18ce27) *(Semantic Search And Embeddings)*

### GitLab (1)

- [Developing GitLab Duo: validating and testing AI models at scale](https://about.gitlab.com/blog/developing-gitlab-duo-how-we-validate-and-test-ai-models-at-scale/) *(Evaluation System)*

### Glean (1)

- [Why vector search isn't enough for enterprise RAG](https://www.glean.com/blog/hybrid-vs-rag-vector) *(RAG Serving)*

### Grafana (1)

- [Monitor LLMs in production with Grafana Cloud, OpenLIT, and OpenTelemetry](https://grafana.com/blog/ai-observability-llms-in-production/) *(Production Monitoring And Observability)*

### Grammarly (1)

- [CoEdIT: state-of-the-art text editing with fewer parameters](https://www.grammarly.com/blog/engineering/coedit-text-editing/) *(Post Training Pipeline)*

### IBM (1)

- [LLM routing for quality, low-cost responses](https://research.ibm.com/blog/LLM-routers) *(Cost Optimization And Model Routing)*

### KIVI (1)

- [A tuning-free asymmetric 2-bit quantization for KV cache](https://arxiv.org/abs/2402.02750) *(Long Context And KV Cache)*

### Krisp (1)

- [A 6M-weight turn-taking model for voice AI agents](https://krisp.ai/blog/turn-taking-for-voice-ai/) *(Realtime Streaming Chat)*

### llm-d (1)

- [KV-Cache wins you can see: prefix caching to distributed scheduling](https://llm-d.ai/blog/kvcache-wins-you-can-see) *(Long Context And KV Cache)*

### LMSYS (1)

- [RouteLLM: an open framework for cost-effective LLM routing](https://www.lmsys.org/blog/2024-07-01-routellm/) *(Cost Optimization And Model Routing)*

### LMSYS / SGLang (1)

- [Fast and expressive LLM inference with RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/) *(Long Context And KV Cache)*

### Mistral (1)

- [Pixtral 12B](https://arxiv.org/abs/2410.07073) *(Multimodal Serving)*

### MIT / Meta (1)

- [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) *(Long Context And KV Cache)*

### Modal (1)

- [High-performance LLM inference](https://modal.com/docs/guide/high-performance-llm-inference) *(Inference Serving At Scale)*

### Moonshot AI (1)

- [Mooncake: a KVCache-centric disaggregated architecture](https://arxiv.org/abs/2407.00079) *(Inference Serving At Scale)*

### Nubank (1)

- [Fine-Tuning Transaction User Models](https://building.nubank.com/fine-tuning-transaction-user-models/) *(Post Training Pipeline)*

### OpenGVLab (1)

- [InternVL 2.5: model, data, and test-time scaling](https://arxiv.org/abs/2412.05271) *(Multimodal Serving)*

### Peking University / UCSD (1)

- [DistServe: disaggregating prefill and decoding](https://arxiv.org/abs/2401.09670) *(Inference Serving At Scale)*

### Red Hat (1)

- [vLLM V1: accelerating multimodal inference](https://developers.redhat.com/articles/2025/02/27/vllm-v1-accelerating-multimodal-inference-large-language-models) *(Multimodal Serving)*

### Shinn et al. (1)

- [Reflexion: language agents with verbal reinforcement learning](https://arxiv.org/abs/2303.11366) *(Agent Orchestration)*

### Snap (1)

- [Embedding-based Retrieval with Two-Tower Models in Spotlight](https://eng.snap.com/embedding-based-retrieval) *(Semantic Search And Embeddings)*

### Snowflake (1)

- [Arctic Inference with Shift Parallelism](https://www.snowflake.com/en/blog/engineering/arctic-inference-shift-parallelism/) *(Inference Serving At Scale)*

### Sourcegraph (1)

- [Agentic Coding: a practical guide for big code](https://sourcegraph.com/blog/agentic-coding) *(Agent Orchestration)*

### Stack Overflow (1)

- [Vector databases in generative AI applications](https://stackoverflow.blog/2023/10/09/from-prototype-to-production-vector-databases-in-generative-ai-applications/) *(Semantic Search And Embeddings)*

### Stanford (1)

- [FrugalGPT: Using LLMs While Reducing Cost and Improving Performance](https://arxiv.org/abs/2305.05176) *(Cost Optimization And Model Routing)*

### Stripe (1)

- [Can AI agents build real Stripe integrations?](https://stripe.com/blog/can-ai-agents-build-real-stripe-integrations) *(Agent Orchestration)*

### UIUC / Cohere (1)

- [SnapKV: LLM knows what you are looking for before generation](https://arxiv.org/abs/2404.14469) *(Long Context And KV Cache)*

### UT Austin / Stanford (1)

- [H2O: Heavy-Hitter Oracle for efficient generative inference](https://arxiv.org/abs/2306.14048) *(Long Context And KV Cache)*

### Vapi (1)

- [How we built Vapi's voice AI pipeline (part 2)](https://vapi.ai/blog/how-we-built-vapi-s-voice-ai-pipeline-part-2) *(Realtime Streaming Chat)*

### Vercel (1)

- [Chat SDK brings agents to your users](https://vercel.com/blog/chat-sdk-brings-agents-to-your-users) *(Realtime Streaming Chat)*

### Vimeo (1)

- [Unlocking knowledge sharing for videos with RAG](https://medium.com/vimeo-engineering-blog/unlocking-knowledge-sharing-for-videos-with-rag-810ab496ae59) *(RAG Serving)*

### vLLM (1)

- [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) *(Long Context And KV Cache)*

### Walmart (1)

- [Semantic Retrieval at Walmart](https://arxiv.org/abs/2412.04637) *(Semantic Search And Embeddings)*

### Wang et al. (1)

- [Voyager: an open-ended embodied agent with LLMs](https://arxiv.org/abs/2305.16291) *(Agent Orchestration)*

### Wayfair (1)

- [How AI understands what you're looking for](https://www.aboutwayfair.com/careers/tech-blog/smarter-shopping-starts-here-how-ai-understands-what-youre-looking-for) *(Evaluation System)*

### Wealthsimple (1)

- [Our LLM Gateway for secure, reliable generative AI](https://engineering.wealthsimple.com/get-to-know-our-llm-gateway-and-how-it-provides-a-secure-and-reliable-space-to-use-generative-ai) *(Safety And Guardrails)*

### Wu et al. (1)

- [AutoGen: next-gen LLM apps via multi-agent conversation](https://arxiv.org/abs/2308.08155) *(Agent Orchestration)*

### Yao et al. (1)

- [ReAct: synergizing reasoning and acting in language models](https://arxiv.org/abs/2210.03629) *(Agent Orchestration)*
