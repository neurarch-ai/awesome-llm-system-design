# Production case studies, by topic

The same landscape of shipped ML and LLM systems that the broad indexes catalog
(the [Evidently AI ML system design database](https://www.evidentlyai.com/ml-system-design)
is the widest, 800 case studies from 150+ companies), re-organized into this
repo's own pipeline-stage taxonomy and presented our way: every entry is a
first-party engineering writeup with a verified link, tagged by what it actually
shows you (*who it serves* / *product design* / *eval bar* / *deployment*).

This is a roll-up of the **Seen in production** section inside each topic, so you
can browse the whole set by category in one place. 189 systems and growing.

---
### [RAG serving](topics/01-rag-serving.md) · 18 systems

- **Ramp** [From RAG to Richness: How Ramp Revamped Industry Classification](https://builders.ramp.com/post/industry_classification): Embedding-model selection plus two-prompt retrieval over NAICS codes, with precomputed embeddings. *(product design)*
- **Uber** [Enhanced Agentic-RAG: near-human precision for chatbots](https://www.uber.com/blog/enhanced-agentic-rag/): Pre-retrieval query agents, doc loaders, metadata enrichment, and LLM-as-judge eval. *(deployment)*
- **Microsoft Research** [GraphRAG: unlocking LLM discovery on narrative private data](https://www.microsoft.com/en-us/research/blog/graphrag-unlocking-llm-discovery-on-narrative-private-data/): Knowledge-graph retrieval beats vector-only RAG on multi-hop private-data queries. *(product design)*
- **DoorDash** [Path to high-quality LLM-based Dasher support automation](https://careersatdoordash.com/blog/large-language-modules-based-dasher-support-automation/): RAG support bot with an LLM guardrail and judge; 90% fewer hallucinations. *(eval bar)*
- **Dropbox** [Building Dash: how RAG and AI agents meet business needs](https://dropbox.tech/machine-learning/building-dash-rag-multi-step-ai-agents-business-users): Hybrid lexical plus chunking plus rerank retrieval balancing latency, freshness, cost. *(deployment)*
- **Vespa** [Embedding Tradeoffs, Quantified](https://blog.vespa.ai/embedding-tradeoffs-quantified/): INT8 and binary quantization plus hybrid BM25 quality-vs-latency tradeoffs. *(eval bar)*
- **Vespa** [Asymmetric Retrieval: spend on docs, embed queries for free](https://blog.vespa.ai/asymmetric-retrieval-spend-on-docs-queries-for-free/): A big model for docs, a tiny local model for queries, to cut serving cost. *(deployment)*
- **NVIDIA** [How a reranking microservice improves retrieval accuracy and cost](https://developer.nvidia.com/blog/how-using-a-reranking-microservice-can-improve-accuracy-and-costs-of-information-retrieval/): Two-stage embed-then-rerank; fewer chunks to the LLM cuts cost. *(eval bar)*
- **Glean** [Why vector search isn't enough for enterprise RAG](https://www.glean.com/blog/hybrid-vs-rag-vector): Enterprise RAG needs hybrid search plus knowledge graph and permissions. *(product design)*
- **Databricks** [Creating High Quality RAG Applications with Databricks](https://www.databricks.com/blog/building-high-quality-rag-applications-databricks): Real-time serving, model selection and eval, and quality monitoring for RAG. *(deployment)*
- **LinkedIn** [Improving Post Search at LinkedIn](https://www.linkedin.com/blog/engineering/search/improving-post-search-at-linkedin): Layered first and second-pass rankers with separate relevance, quality, freshness models. *(deployment)*
- **Pinterest** [How we built Text-to-SQL at Pinterest](https://medium.com/pinterest-engineering/how-we-built-text-to-sql-at-pinterest-30bad30dabff): RAG table retrieval grounds LLM SQL generation over thousands of warehouse tables. *(product design)*
- **Cloudflare** [Introducing AutoRAG: managed RAG on Cloudflare](https://blog.cloudflare.com/introducing-autorag-on-cloudflare/): A managed pipeline: async indexing, Vectorize storage, query-time retrieval plus generation. *(deployment)*
- **Vimeo** [Unlocking knowledge sharing for videos with RAG](https://medium.com/vimeo-engineering-blog/unlocking-knowledge-sharing-for-videos-with-rag-810ab496ae59): Video Q&A over transcript chunking, multi-size context windows, and vector retrieval. *(product design)*
- **Elastic** [RAG pipelines in production](https://www.elastic.co/search-labs/blog/rag-in-production): Operationalizing RAG with hybrid retrieval, reranking, monitoring, and benchmarking. *(deployment)*
- **Anyscale** [Building RAG-based LLM Applications for Production](https://www.anyscale.com/blog/a-comprehensive-guide-for-building-rag-based-llm-applications-part-1): End-to-end RAG built, evaluated, and served at scale with Ray Serve. *(deployment)*
- **GitHub** [What is retrieval-augmented generation?](https://github.blog/ai-and-ml/generative-ai/what-is-retrieval-augmented-generation-and-what-does-it-do-for-generative-ai/): How Copilot Enterprise grounds answers via internal code search and semantic retrieval. *(product design)*
- **Google / ETH Zurich** [RAGO: systematic performance optimization for RAG serving](https://arxiv.org/abs/2503.14649): A serving framework raising QPS per chip 2x and cutting time-to-first-token 55%. *(deployment)*

---

### [Semantic search & embedding service](topics/08-semantic-search-and-embeddings.md) · 21 systems

- **Spotify** [Introducing Voyager: Spotify new nearest-neighbor search library](https://engineering.atspotify.com/2023/10/introducing-voyager-spotifys-new-nearest-neighbor-search-library): HNSW ANN library: recall versus speed versus memory tradeoffs, 8-bit compression. *(deployment)*
- **Vespa** [Billion-scale vector search using hybrid HNSW-IF](https://blog.vespa.ai/vespa-hybrid-billion-scale-vector-search/): In-memory HNSW plus disk-backed inverted files for 90% recall under 50ms, cheaply. *(deployment)*
- **LinkedIn** [Semantic Search for AI Agents at Scale](https://www.linkedin.com/blog/engineering/ai/semantic-search-for-ai-agents-at-scale-retrieval-and-ranking-for-linkedins-hiring-assistant): Two-stage ANN retrieval plus ranker over 1B+ profiles using Matryoshka embeddings. *(deployment)*
- **Pinterest** [Advancements in Embedding-Based Retrieval at Pinterest Homefeed](https://medium.com/pinterest-engineering/advancements-in-embedding-based-retrieval-at-pinterest-homefeed-d7d7971a409e): Two-tower embedding retrieval with multi-embedding ANN and interest filters. *(deployment)*
- **Meta** [Faiss: a library for efficient similarity search](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/): A GPU-accelerated billion-scale similarity search library powering retrieval. *(deployment)*
- **Google Research** [Announcing ScaNN: efficient vector similarity search](https://research.google/blog/announcing-scann-efficient-vector-similarity-search/): Anisotropic quantization wins recall-vs-QPS on ann-benchmarks. *(eval bar)*
- **Microsoft Research** [DiskANN: vector search for all](https://www.microsoft.com/en-us/research/project/project-akupara-approximate-nearest-neighbor-search-for-large-scale-semantic-search/): SSD-backed ANN: billion vectors, 95% recall, about 5ms latency. *(deployment)*
- **Meta** [Embedding-based Retrieval in Facebook Search](https://arxiv.org/abs/2006.11632): A unified embedding framework for personalized social search. *(product design)*
- **Instacart** [How Instacart uses embeddings to improve search relevance](https://company.instacart.com/how-its-made/how-instacart-uses-embeddings-to-improve-search-relevance): A two-tower items model served via FAISS ANN with daily indices. *(deployment)*
- **Etsy** [Unified Embedding Based Personalized Retrieval in Etsy Search](https://arxiv.org/abs/2306.04833): Graph, transformer, and term embeddings with HNSW and 4-bit PQ. *(product design)*
- **Airbnb** [Applying Embedding-Based Retrieval to Airbnb Search](https://arxiv.org/abs/2601.06873): EBR for a two-sided marketplace, A/B-tested booking gains. *(who it serves)*
- **Uber Eats** [Scaling Multilingual Semantic Search in Uber Eats](https://arxiv.org/abs/2603.06586): Multilingual retrieval across stores, dishes, grocery in six markets. *(deployment)*
- **Walmart** [Semantic Retrieval at Walmart](https://arxiv.org/abs/2412.04637): Hybrid inverted-index plus neural retrieval for tail product queries. *(product design)*
- **Dropbox** [Selecting a model for semantic search at Dropbox scale](https://dropbox.tech/machine-learning/selecting-model-semantic-search-dropbox-ai): Benchmarking 11 embedding models on MTEB to pick multilingual-e5-large for retrieval. *(eval bar)*
- **GitHub** [Inside Copilot's new code embedding model](https://github.blog/news-insights/product-news/copilot-new-embedding-model-vs-code/): A custom code embedding model lifting Copilot retrieval quality 37.6% at lower latency. *(eval bar)*
- **Faire** [Beyond BM25 and dense embeddings: smart, interpretable retrieval](https://craft.faire.com/beyond-bm25-and-dense-embeddings-841a7b18ce27): SPLADE sparse neural retrieval giving interpretable semantics over Elasticsearch. *(product design)*
- **Snap** [Embedding-based Retrieval with Two-Tower Models in Spotlight](https://eng.snap.com/embedding-based-retrieval): Two-tower user/video embeddings for real-time short-form recommendation retrieval. *(deployment)*
- **Yelp** [Yelp Content As Embeddings](https://engineeringblog.yelp.com/2023/04/yelp-content-as-embeddings.html): Shared low-dimensional embeddings of reviews, businesses, and photos as a ranking baseline. *(deployment)*
- **Stack Overflow** [Vector databases in generative AI applications](https://stackoverflow.blog/2023/10/09/from-prototype-to-production-vector-databases-in-generative-ai-applications/): A self-hosted Weaviate vector database on Azure for production semantic search. *(deployment)*
- **Mercari** [Domain-Aware Text Embeddings for C2C Marketplaces](https://arxiv.org/abs/2512.21021): Domain-aware text embeddings improving search for Japan's largest C2C marketplace. *(product design)*
- **Amazon** [Semantic Product Search](https://arxiv.org/abs/1907.00937): A KDD 2019 two-tower model with kNN retrieval over precomputed catalog embeddings. *(product design)*

---

### [Long-context inference & the KV cache](topics/02-long-context-and-kv-cache.md) · 20 systems

- **vLLM (UC Berkeley)** [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180): OS-style KV-cache paging cuts fragmentation, boosting throughput 2x to 4x. *(deployment)*
- **Character.AI** [Optimizing AI Inference at Character.AI](https://blog.character.ai/optimizing-ai-inference-at-character-ai-2/): MQA, hybrid local/global attention, and cross-layer KV-sharing cut cost 33x. *(deployment)*
- **DeepSeek** [DeepSeek-V2: a strong, economical, efficient MoE language model](https://arxiv.org/abs/2405.04434): Multi-head Latent Attention compresses the KV cache into a latent vector, shrinking it 93%. *(product design)*
- **Google Research** [GQA: Training Generalized Multi-Query Transformer Models](https://arxiv.org/abs/2305.13245): Grouped-query attention trades KV heads for speed at near-MHA quality. *(product design)*
- **NVIDIA** [5x faster time to first token with TensorRT-LLM KV cache early reuse](https://developer.nvidia.com/blog/5x-faster-time-to-first-token-with-nvidia-tensorrt-llm-kv-cache-early-reuse/): Early KV reuse, flexible block sizing, and smart eviction cut TTFT. *(deployment)*
- **NVIDIA** [Optimizing inference with NVFP4 KV cache](https://developer.nvidia.com/blog/optimizing-inference-for-long-context-and-large-batch-sizes-with-nvfp4-kv-cache/): 4-bit KV cache halves memory vs FP8, doubling context with under 1% loss. *(deployment)*
- **Databricks** [Inference-Friendly Models with MixAttention](https://www.databricks.com/blog/mixattention): Cross-layer KV sharing plus sliding-window attention shrinks the cache. *(product design)*
- **Databricks** [Accelerating LLM inference with prompt caching](https://www.databricks.com/blog/accelerating-llm-inference-prompt-caching-open-source-models-databricks): Automatic prefix KV reuse: 2.5x throughput, 3x lower P50 latency. *(deployment)*
- **llm-d** [KV-Cache wins you can see: prefix caching to distributed scheduling](https://llm-d.ai/blog/kvcache-wins-you-can-see): Single-instance prefix caching breaks in clusters; cache-aware scheduling fixes it. *(deployment)*
- **LMSYS / SGLang** [Fast and expressive LLM inference with RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/): A radix-tree KV cache enables automatic cross-request prefix reuse. *(deployment)*
- **Together AI** [Serving MiniMax-M3: 1M-token context without regrets](https://www.together.ai/blog/serving-minimax-m3-for-efficient-inference-unlocking-1m-token-context-and-multimodality-without-regrets): Paged sparse attention and KV-block-major kernels make 1M-token serving practical. *(deployment)*
- **Hugging Face** [Unlocking longer generation with KV cache quantization](https://huggingface.co/blog/kv-cache-quantization): Per-token int4 KV quantization yields about 2.5x memory savings. *(product design)*
- **KIVI** [A tuning-free asymmetric 2-bit quantization for KV cache](https://arxiv.org/abs/2402.02750): Per-channel keys and per-token values enable 2-bit KV compression. *(product design)*
- **MIT / Meta** [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453): The attention-sink insight lets fixed-window LLMs stream to millions of tokens. *(product design)*
- **Anthropic** [Prompt caching with Claude](https://claude.com/blog/prompt-caching): Caches reused context across API calls, cutting cost up to 90% and latency 85%. *(product design)*
- **Colfax / Together** [FlashAttention-3: fast, accurate attention with asynchrony and low precision](https://arxiv.org/abs/2407.08608): Hopper-optimized attention via warp-specialization and FP8, 1.5-2x faster. *(deployment)*
- **UT Austin / Stanford** [H2O: Heavy-Hitter Oracle for efficient generative inference](https://arxiv.org/abs/2306.14048): KV-cache eviction keeping recent plus heavy-hitter tokens, up to 29x throughput. *(deployment)*
- **UIUC / Cohere** [SnapKV: LLM knows what you are looking for before generation](https://arxiv.org/abs/2404.14469): Fine-tuning-free KV compression via observation-window clustering, 3.6x faster decode. *(product design)*
- **Fireworks AI** [FireAttention V2: long contexts practical for online inference](https://fireworks.ai/blog/fireattention-v2-long-context-inference): Production long-context kernels with FP8 prefill and multi-host deployment. *(deployment)*
- **Microsoft** [MInference 1.0: accelerating pre-filling via dynamic sparse attention](https://arxiv.org/abs/2407.02490): Dynamic sparse attention cuts long-context prefill latency up to 10x. *(deployment)*

---

### [LLM inference serving at scale](topics/04-inference-serving-at-scale.md) · 16 systems

- **Anyscale** [How continuous batching enables 23x throughput in LLM inference](https://www.anyscale.com/blog/continuous-batching-llm-inference): Iteration-level scheduling plus PagedAttention beat static batching up to 23x. *(deployment)*
- **Character.AI** [Optimizing AI Inference at Character.AI](https://blog.character.ai/optimizing-ai-inference-at-character-ai/): MQA, cross-layer KV sharing, and int8 quant cut serving cost 13.5x. *(deployment)*
- **LinkedIn** [Accelerating LLM inference with speculative decoding](https://www.linkedin.com/blog/engineering/ai/accelerating-llm-inference-with-speculative-decoding-lessons-from-linkedins-hiring-assistant): N-gram speculative decoding gave 4x throughput and 66% lower P90 latency. *(eval bar)*
- **Baseten** [How we built BEI: high-throughput embedding, reranker, classifier inference](https://www.baseten.co/blog/how-we-built-bei-high-throughput-embedding-inference/): Batching, backpressure, FP8, and TensorRT-LLM for 2x higher-throughput serving. *(deployment)*
- **NVIDIA** [NVIDIA Dynamo: a low-latency distributed inference framework](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/): Disaggregated serving with prefill and decode separation and routing. *(deployment)*
- **Together AI** [ATLAS: runtime-learning speculative decoding](https://www.together.ai/blog/adaptive-learning-speculator-system-atlas): Speculative decoding that adapts to live traffic for large speedups. *(product design)*
- **Fireworks AI** [FireOptimizer: customizing latency and quality](https://fireworks.ai/blog/fireoptimizer): Adaptive speculative decoding and per-workload config tuning. *(product design)*
- **Modal** [High-performance LLM inference](https://modal.com/docs/guide/high-performance-llm-inference): Engine choice, quantization, CUDA graphs, and snapshots for throughput. *(deployment)*
- **Databricks** [LLM inference performance engineering: best practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices): Prefill and decode, batching, hardware selection, and latency metrics. *(eval bar)*
- **Google** [Fast inference from transformers via speculative decoding](https://arxiv.org/abs/2211.17192): Draft-then-verify decoding: 2-3x speedup with identical outputs. *(product design)*
- **Baseten** [The Baseten inference stack](https://www.baseten.co/resources/guide/the-baseten-inference-stack/): Multi-cloud autoscaling, routing, custom kernels, and speculation. *(deployment)*
- **Moonshot AI** [Mooncake: a KVCache-centric disaggregated architecture](https://arxiv.org/abs/2407.00079): Kimi's prefill/decode-disaggregated serving with a pooled CPU/DRAM/SSD KV cache. *(deployment)*
- **Microsoft** [Splitwise: efficient generative LLM inference using phase splitting](https://arxiv.org/abs/2311.18677): Splits prefill and decode onto separate machines for cost and throughput. *(deployment)*
- **Peking University / UCSD** [DistServe: disaggregating prefill and decoding](https://arxiv.org/abs/2401.09670): Disaggregates prefill and decode across GPUs to optimize goodput under SLOs. *(deployment)*
- **Microsoft Research** [Sarathi-Serve: taming the throughput-latency tradeoff](https://arxiv.org/abs/2403.02310): Chunked-prefills and stall-free scheduling balance throughput against latency. *(deployment)*
- **Snowflake** [Arctic Inference with Shift Parallelism](https://www.snowflake.com/en/blog/engineering/arctic-inference-shift-parallelism/): A vLLM plugin with dynamic shift parallelism adapting to real traffic. *(deployment)*

---

### [Realtime streaming chat](topics/10-realtime-streaming-chat.md) · 17 systems

- **LinkedIn** [Musings on building a Generative AI product](https://www.linkedin.com/blog/engineering/generative-ai/musings-on-building-a-generative-ai-product): End-to-end token streaming and progressive parsing to cut perceived latency. *(deployment)*
- **Cloudflare** [Durable Objects for WebSockets and auth in AI Gateway](https://blog.cloudflare.com/do-it-again/): Scaling persistent WebSocket connections for concurrent AI inference streams. *(deployment)*
- **Vercel** [Chat SDK brings agents to your users](https://vercel.com/blog/chat-sdk-brings-agents-to-your-users): Streaming responses cross-platform via native streaming versus a throttled fallback. *(product design)*
- **OpenAI** [Updates for developers building with voice](https://developers.openai.com/blog/updates-audio-models): New audio model snapshots for STT, TTS, and realtime speech-to-speech. *(product design)*
- **LiveKit** [Why WebRTC beats WebSockets for realtime voice AI](https://livekit.com/blog/why-webrtc-beats-websockets-for-voice-ai-agents): WebRTC handles packet loss, jitter, and congestion better than TCP. *(deployment)*
- **LiveKit** [Why you shouldn't build voice agents directly on model APIs](https://livekit.com/blog/real-time-voice-agents-vs-model-apis): Model APIs lack transport, echo cancellation, and turn detection. *(deployment)*
- **Deepgram** [Optimize voice agent latency with eager end of turn](https://developers.deepgram.com/docs/flux/voice-agent-eager-eot): Start the LLM on medium-confidence transcripts to overlap with speech. *(deployment)*
- **AssemblyAI** [Universal-Streaming: ultra-fast speech-to-text for voice agents](https://www.assemblyai.com/blog/introducing-universal-streaming): Immutable streaming transcripts in about 300ms with intelligent endpointing. *(eval bar)*
- **ElevenLabs** [Enhancing conversational AI latency with efficient TTS](https://elevenlabs.io/blog/enhancing-conversational-ai-latency-with-efficient-tts-pipelines): Reducing streaming TTS time-to-first-byte for responsive conversation. *(deployment)*
- **Daily** [Benchmarking LLMs for voice agent use cases](https://www.daily.co/blog/benchmarking-llms-for-voice-agent-use-cases/): An open benchmark for latency, tool calling, and instruction adherence in voice. *(eval bar)*
- **Cartesia** [Announcing Sonic: a low-latency voice model](https://cartesia.ai/blog/sonic): A state-space TTS hitting 135ms model latency for streaming voice agents. *(product design)*
- **Krisp** [A 6M-weight turn-taking model for voice AI agents](https://krisp.ai/blog/turn-taking-for-voice-ai/): A tiny CPU turn-detection model deciding when agents speak, listen, or wait. *(product design)*
- **Twilio** [Introducing Media Streams](https://www.twilio.com/en-us/blog/media-streams-public-beta): Forks raw call audio over WebSockets for real-time bidirectional voice apps. *(deployment)*
- **Vapi** [How we built Vapi's voice AI pipeline (part 2)](https://vapi.ai/blog/how-we-built-vapi-s-voice-ai-pipeline-part-2): VAD, endpointing, streaming STT, and inference coordination for low-latency voice. *(deployment)*
- **Daily (Pipecat)** [Smart Turn v3, with CPU inference in 12ms](https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/): An open-source semantic-VAD turn-detection model, 8MB, 23 languages, CPU-friendly. *(product design)*
- **Slack** [Real-time Messaging](https://slack.engineering/real-time-messaging/): A stateful WebSocket gateway and channel servers deliver messages globally in 500ms. *(deployment)*
- **Discord** [How Discord Scaled Elixir to 5,000,000 Concurrent Users](https://discord.com/blog/how-discord-scaled-elixir-to-5-000-000-concurrent-users): Elixir GenServer sessions and Manifold fan-out for millions of concurrent WebSockets. *(deployment)*

---

### [Agent orchestration](topics/03-agent-orchestration.md) · 22 systems

- **Anthropic** [Building effective agents](https://www.anthropic.com/research/building-effective-agents): When to use workflows versus agents, and five composable orchestration patterns. *(product design)*
- **Anthropic** [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system): Orchestrator-worker pattern with parallel subagents; +90.2% over a single agent. *(deployment)*
- **Cognition** [Don't Build Multi-Agents](https://cognition.com/blog/dont-build-multi-agents): The counter-case: single-threaded agents win, and why parallel subagents are fragile. *(product design)*
- **Ramp** [Why We Built Our Own Background Agent](https://builders.ramp.com/post/why-we-built-our-background-agent): Closed-loop coding agent on sandboxed Modal VMs with verification. *(deployment)*
- **LangChain** [Context Engineering for Agents](https://www.langchain.com/blog/context-engineering-for-agents): Write, select, compress, and isolate context to control token cost and latency. *(product design)*
- **OpenAI** [A practical guide to building agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf): Orchestration patterns, guardrails, and single vs multi-agent from deployments. *(product design)*
- **Anthropic** [Writing effective tools for agents, with agents](https://www.anthropic.com/engineering/writing-tools-for-agents): Designing and evaluating tool definitions to raise agent task success. *(product design)*
- **Anthropic** [Code execution with MCP: building more efficient agents](https://www.anthropic.com/engineering/code-execution-with-mcp): Code execution over MCP cuts tokens and latency at scale. *(product design)*
- **Uber** [Genie: Uber's Gen AI on-call copilot](https://www.uber.com/en-US/blog/genie-ubers-gen-ai-on-call-copilot/): A production RAG on-call copilot serving 45k engineer questions monthly. *(deployment)*
- **Block** [Introducing codename goose: an open framework for AI agents](https://block.xyz/inside/block-open-source-introduces-codename-goose): An open extensible agent running local multi-step tasks via MCP. *(product design)*
- **Sourcegraph** [Agentic Coding: a practical guide for big code](https://sourcegraph.com/blog/agentic-coding): Running agent loops with tools across large enterprise codebases. *(who it serves)*
- **Replit** [Enabling Agent 3 to self-test at scale with REPL verification](https://replit.com/blog/automated-self-testing): REPL plus browser verification lets the agent self-test autonomously. *(eval bar)*
- **GitHub** [Evaluating the Copilot agentic harness across models and tasks](https://github.blog/ai-and-ml/github-copilot/evaluating-performance-and-efficiency-of-the-github-copilot-agentic-harness-across-models-and-tasks/): Benchmarking a multi-model agent harness on resolution and token cost. *(eval bar)*
- **Salesforce** [Inside Agentforce: the Atlas Reasoning Engine](https://engineering.salesforce.com/inside-the-brain-of-agentforce-revealing-the-atlas-reasoning-engine/): A model-agnostic reasoning and planning engine driving enterprise agent actions. *(deployment)*
- **Airbnb** [Automation Platform v2: improving conversational AI](https://medium.com/airbnb-engineering/automation-platform-v2-improving-conversational-ai-at-airbnb-d86c9386e0cb): An LLM reasoning engine with chain-of-thought tool orchestration, context, and guardrails. *(deployment)*
- **LinkedIn** [The LinkedIn GenAI tech stack: extending to build AI agents](https://www.linkedin.com/blog/engineering/generative-ai/the-linkedin-generative-ai-application-tech-stack-extending-to-build-ai-agents): Multi-agent orchestration over messaging infra: agent registry, lifecycle, observability. *(deployment)*
- **Hugging Face** [Introducing smolagents](https://huggingface.co/blog/smolagents): The case for code-writing agents over JSON tool calls for multi-step tool use. *(product design)*
- **Stripe** [Can AI agents build real Stripe integrations?](https://stripe.com/blog/can-ai-agents-build-real-stripe-integrations): A benchmark of 11 challenges scoring agents on integration, testing, and error recovery. *(eval bar)*
- **Yao et al.** [ReAct: synergizing reasoning and acting in language models](https://arxiv.org/abs/2210.03629): The foundational pattern interleaving reasoning traces with tool actions. *(product design)*
- **Shinn et al.** [Reflexion: language agents with verbal reinforcement learning](https://arxiv.org/abs/2303.11366): Agents self-reflect on feedback to improve future actions without weight updates. *(eval bar)*
- **Wang et al.** [Voyager: an open-ended embodied agent with LLMs](https://arxiv.org/abs/2305.16291): A lifelong Minecraft agent with an auto curriculum, skill library, and self-verification. *(product design)*
- **Wu et al.** [AutoGen: next-gen LLM apps via multi-agent conversation](https://arxiv.org/abs/2308.08155): A framework for multi-agent systems via customizable conversable agents. *(deployment)*

---

### [Multimodal serving](topics/09-multimodal-serving.md) · 19 systems

- **Red Hat (vLLM)** [vLLM V1: accelerating multimodal inference](https://developers.redhat.com/articles/2025/02/27/vllm-v1-accelerating-multimodal-inference-large-language-models): Encoder caching, per-image prefix caching, and async CPU/GPU for faster multimodal serving. *(deployment)*
- **AMD (ROCm)** [Accelerating Multimodal Inference in vLLM](https://rocm.blogs.amd.com/software-tools-optimization/vllm-dp-vision/README.html): Batch-level data parallelism for vision encoders cuts sync overhead. *(deployment)*
- **Alibaba (Qwen)** [Qwen2-VL: enhancing vision-language perception at any resolution](https://arxiv.org/abs/2409.12191): Dynamic resolution turns any image into variable visual tokens, with an MLP projector and M-RoPE. *(product design)*
- **Mistral AI** [Pixtral 12B](https://arxiv.org/abs/2410.07073): A custom ViT trained from scratch ingests native resolution with a flexible image token budget. *(product design)*
- **Microsoft (LLaVA)** [Visual Instruction Tuning](https://arxiv.org/abs/2304.08485): The MLP projector connecting a frozen CLIP vision encoder to the LLM embedding space. *(product design)*
- **Dropbox** [Creating a modern OCR pipeline using CV and deep learning](https://dropbox.tech/machine-learning/creating-a-modern-ocr-pipeline-using-computer-vision-and-deep-learning): Productionizing a deep-learning OCR and document-scan pipeline. *(deployment)*
- **Hugging Face** [Introducing Idefics2: a powerful 8B vision-language model](https://huggingface.co/blog/idefics2): Vision encoder plus projector plus perceiver resampler design choices. *(product design)*
- **NVIDIA** [NVLM: open frontier-class multimodal LLMs](https://research.nvidia.com/labs/adlr/NVLM-1/): Decoder-only vs cross-attention connector tradeoffs, plus tile-tagging for OCR. *(deployment)*
- **DeepMind** [Flamingo: a visual language model for few-shot learning](https://arxiv.org/abs/2204.14198): Bridging frozen vision and language models with interleaved input. *(product design)*
- **Ai2** [Molmo and PixMo: open weights and open data for VLMs](https://arxiv.org/abs/2409.17146): An open VLM family with curated data rivaling proprietary models. *(eval bar)*
- **OpenGVLab** [InternVL 2.5: model, data, and test-time scaling](https://arxiv.org/abs/2412.05271): Scaling an open multimodal model across model, data, and inference-time. *(eval bar)*
- **Alibaba Qwen** [Qwen2-Audio Technical Report](https://arxiv.org/abs/2407.10759): An audio-language model with voice-chat and audio-analysis modes. *(who it serves)*
- **NVIDIA** [Accelerating VLM inference with TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/): A C++ runtime for low-latency on-device VLM inference on embedded. *(deployment)*
- **Apple** [MM1: methods, analysis, and insights from multimodal LLM pre-training](https://arxiv.org/abs/2403.09611): Ablations on image-token count, connector design, and data mix for building VLMs. *(product design)*
- **Meta** [Chameleon: mixed-modal early-fusion foundation models](https://arxiv.org/abs/2405.09818): A single transformer over interleaved image/text tokens; a stable early-fusion recipe. *(product design)*
- **Roblox** [Running AI Inference at Scale in the Hybrid Cloud](https://about.roblox.com/newsroom/2024/09/running-ai-inference-at-scale-in-the-hybrid-cloud): vLLM, Ray, and a custom feature store serving 250 ML pipelines across hybrid cloud. *(deployment)*
- **Google** [PaLI-X: on scaling up a multilingual vision-language model](https://arxiv.org/abs/2305.18565): Scaling VLM components and task mix advances 25+ vision-language benchmarks. *(eval bar)*
- **Microsoft** [Florence-2: a unified representation for vision tasks](https://arxiv.org/abs/2311.06242): A prompt-based seq2seq VLM unifying caption, detection, grounding, and segmentation. *(product design)*
- **Salesforce** [BLIP-2: bootstrapping with frozen image encoders and LLMs](https://arxiv.org/abs/2301.12597): A lightweight Q-Former projector bridges a frozen vision encoder to a frozen LLM. *(product design)*

---

### [Fine-tuning & post-training](topics/05-post-training-pipeline.md) · 19 systems

- **Grammarly** [CoEdIT: state-of-the-art text editing with fewer parameters](https://www.grammarly.com/blog/engineering/coedit-text-editing/): Dense task-specific instruction tuning beats generalist LLMs at 12x to 60x fewer params. *(product design)*
- **Anyscale** [Fine-Tuning LLMs: LoRA or Full-Parameter?](https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter-an-in-depth-analysis-with-llama-2): LoRA versus full fine-tune accuracy tradeoffs, broken down per task type. *(eval bar)*
- **Anyscale** [Direct Preference Optimization with Synthetic Data](https://www.anyscale.com/blog/direct-preference-optimization-with-synthetic-data): Iterative DPO: synthetic prefs, async reference model, judge-aligned eval. *(deployment)*
- **Hugging Face** [Preference Tuning LLMs with Direct Preference Optimization Methods](https://huggingface.co/blog/pref-tuning): Empirical DPO versus IPO versus KTO; the beta parameter drives outcomes. *(eval bar)*
- **Databricks** [A Practical Guide to LLM Fine Tuning](https://www.databricks.com/blog/llm-fine-tuning): End-to-end lifecycle: metrics, data quality, LoRA-first, and retrain cadence. *(deployment)*
- **Shopify** [Flow generation through natural language: an agentic modeling approach](https://shopify.engineering/fine-tuning-agent-shopify-flow): A fine-tuned Qwen3-32B agent with a weekly LLM-judge retraining flywheel. *(product design)*
- **Shopify** [Leveraging multimodal LLMs for the global catalogue](https://shopify.engineering/leveraging-multimodal-llms): Fine-tunes small VLMs for catalogue extraction at 40M inferences per day. *(deployment)*
- **Meta** [How to fine-tune: focus on effective datasets](https://ai.meta.com/blog/how-to-fine-tune-llms-peft-dataset-curation/): Data-curation rules for SFT and PEFT; quality over quantity. *(product design)*
- **GitHub** [Building a faster, smarter Copilot with a custom model](https://github.blog/ai-and-ml/github-copilot/the-road-to-better-completions-building-a-faster-smarter-github-copilot-with-a-new-custom-model/): A mid-training plus SFT (fill-in-middle) plus RL pipeline. *(deployment)*
- **Replit** [Replit Code v1.5 on Hugging Face](https://replit.com/blog/replit-code-v1_5): Trained and fine-tuned a code model on Replit user code. *(product design)*
- **Grab** [A custom vision LLM to improve document processing](https://engineering.grab.com/custom-vision-llm-at-grab): LoRA then full fine-tune of Qwen2-VL for OCR and key-info extraction. *(who it serves)*
- **Nubank** [Fine-Tuning Transaction User Models](https://building.nubank.com/fine-tuning-transaction-user-models/): SFT of transaction foundation models with joint fusion. *(product design)*
- **LinkedIn** [How we built domain-adapted foundation GenAI models](https://www.linkedin.com/blog/engineering/generative-ai/how-we-built-domain-adapted-foundation-genai-models-to-power-our-platform): Llama-based EON models via instruction tuning plus RLHF/DPO, 75x cheaper than GPT-4. *(deployment)*
- **Cloudflare** [Running fine-tuned models on Workers AI with LoRAs](https://blog.cloudflare.com/fine-tuned-inference-with-loras/): Serving customer LoRA adapters on shared base models across edge inference. *(deployment)*
- **Uber** [Open Source and In-House: How Uber Optimizes LLM Training](https://www.uber.com/us/en/blog/open-source-and-in-house-how-uber-optimizes-llm-training/): An in-house stack using LoRA/QLoRA, full fine-tuning, and continued pre-training. *(deployment)*
- **Thomson Reuters** [Scaling LLM research with Amazon SageMaker HyperPod](https://aws.amazon.com/blogs/machine-learning/scaling-thomson-reuters-language-model-research-with-amazon-sagemaker-hyperpod/): Domain-adapted legal LLMs (7B-30B) trained to beat general models on legal tasks. *(who it serves)*
- **Mercari** [Fine-Tuning an LLM to Extract Dynamically Specified Attributes](https://engineering.mercari.com/en/blog/entry/20240913-fine-tuning-an-llm-to-extract-dynamically-specified-attributes/): A QLoRA-tuned 2B model beats GPT-3.5 on attribute extraction at 14x lower cost. *(eval bar)*
- **Yelp** [An AI pipeline for inappropriate-language detection in reviews](https://engineeringblog.yelp.com/2024/03/ai-pipeline-inappropriate-language-detection.html): A fine-tuned LLM classifier flags inappropriate reviews; blocked 23,600+ in 2023. *(product design)*
- **Spotify** [Optimizing Query Expansions via LLM Preference Alignment](https://research.atspotify.com/2025/7/optimizing-query-expansions-via-llm-preference-alignment): Rejection-sampling SFT plus DPO aligns a query-expansion LLM, 70% faster. *(product design)*

---

### [LLM evaluation](topics/06-evaluation-system.md) · 18 systems

- **DoorDash** [A Simulation and Evaluation Flywheel to Develop LLM Chatbots at Scale](https://careersatdoordash.com/blog/doordash-simulation-evaluation-flywheel-to-develop-llm-chatbots-at-scale/): Simulated multi-turn conversations graded by an LLM judge calibrated to humans before release. *(eval bar)*
- **DoorDash** [How DoorDash leverages LLMs to evaluate search result pages](https://careersatdoordash.com/blog/doordash-llms-to-evaluate-search-result-pages/): AutoEval: fine-tuned LLM raters with a human in the loop for whole-page relevance. *(eval bar)*
- **Thomson Reuters** [Efficiently evaluating LLMs for legal tasks](https://legal.thomsonreuters.com/blog/evaluating-llms-legal-tasks/): Three-stage gate: public benchmarks, semi-automated task eval, then human A/B. *(eval bar)*
- **Uber** [uReview: scalable, trustworthy GenAI for code review](https://www.uber.com/us/en/blog/ureview/): An LLM grader scores generated comments; confidence thresholds gate what gets posted. *(deployment)*
- **Uber** [From Predictive to Generative: how Michelangelo accelerates Uber AI](https://www.uber.com/blog/from-predictive-to-generative-ai/): Michelangelo's eval framework compares models, prompts, and fine-tunes across iterations. *(deployment)*
- **Discord** [Developing Rapidly with Generative AI](https://discord.com/blog/developing-rapidly-with-generative-ai): A critic-LLM AI-assisted eval of prompts before A/B rollout. *(eval bar)*
- **Honeycomb** [So we shipped an AI product. Did it work?](https://www.honeycomb.io/blog/we-shipped-ai-product): Post-launch product eval via activation and adoption metrics. *(eval bar)*
- **GitHub** [How we evaluate AI models and LLMs for GitHub Copilot](https://github.blog/ai-and-ml/generative-ai/how-we-evaluate-models-for-github-copilot/): 4000+ offline tests plus manual and safety gates before deploy. *(eval bar)*
- **Spotify** [Better experiments with LLM evals: a funnel, not a fork](https://engineering.atspotify.com/2026/5/better-experiments-with-llm-evals-a-funnel-not-a-fork): Offline evals calibrated against online A/B as a funnel. *(deployment)*
- **Spotify** [Profile-aware LLM-as-a-Judge for Podcasts](https://research.atspotify.com/2025/9/profile-aware-llm-as-a-judge-for-podcasts-a-better-middle-ground-between): An LLM judge bridging offline metrics and costly A/B tests. *(eval bar)*
- **Booking.com** [LLM Evaluation: practical tips at Booking.com](https://booking.ai/llm-evaluation-practical-tips-at-booking-com-1b038a0d6662): LLM-as-judge plus golden datasets for production monitoring. *(eval bar)*
- **Pinterest** [LLM-Powered Relevance Assessment for Pinterest Search](https://medium.com/pinterest-engineering/llm-powered-relevance-assessment-for-pinterest-search-b846489e358d): Fine-tuned LLM judges label search relevance to evaluate ranking A/B experiments. *(deployment)*
- **Instacart** [Scaling Catalog Attribute Extraction with Multi-modal LLMs](https://company.instacart.com/tech-innovation/scaling-catalog-attribute-extraction-with-multi-modal-llms): LLM-as-judge auto-eval monitors attribute-extraction quality alongside human auditors. *(eval bar)*
- **Ramp** [How Ramp Fixes Merchant Matches with AI](https://builders.ramp.com/post/fixing-merchant-classifications-with-ai): Shadow mode plus an LLM judge evaluate agent classifications against humans pre-rollout. *(deployment)*
- **Microsoft** [LLM-Rubric: a multidimensional, calibrated approach to automated evaluation](https://www.microsoft.com/en-us/research/publication/llm-rubric-a-multidimensional-calibrated-approach-to-automated-evaluation-of-natural-language-texts/): A calibrated multi-dimension rubric judge predicts human satisfaction for a dialogue system. *(eval bar)*
- **LinkedIn** [How we engineered LinkedIn's Hiring Assistant](https://www.linkedin.com/blog/engineering/ai/how-we-engineered-linkedins-hiring-assistant): A quality framework pairs product policy with LLM judges scoring coherence and factuality. *(product design)*
- **GitLab** [Developing GitLab Duo: validating and testing AI models at scale](https://about.gitlab.com/blog/developing-gitlab-duo-how-we-validate-and-test-ai-models-at-scale/): A central eval framework with an LLM judge runs daily regression at scale. *(deployment)*
- **Wayfair** [How AI understands what you're looking for](https://www.aboutwayfair.com/careers/tech-blog/smarter-shopping-starts-here-how-ai-understands-what-youre-looking-for): LLM-as-judge validation tasks periodically evaluate AI-generated customer interests offline. *(eval bar)*

---

### [Safety, moderation & guardrails](topics/07-safety-and-guardrails.md) · 19 systems

- **Roblox** [How Roblox Uses AI to Moderate Content on a Massive Scale](https://about.roblox.com/newsroom/2025/07/roblox-ai-moderation-massive-scale): Multi-model text, voice, and PII moderation at 750k requests per second with real-time prevention. *(deployment)*
- **Anthropic** [Constitutional Classifiers: defending against universal jailbreaks](https://www.anthropic.com/research/constitutional-classifiers): Input/output classifiers trained on a synthetic constitution cut jailbreaks from 86% to 4.4%. *(eval bar)*
- **Microsoft (MSRC)** [How Microsoft defends against indirect prompt injection attacks](https://www.microsoft.com/en-us/msrc/blog/2025/07/how-microsoft-defends-against-indirect-prompt-injection-attacks): Defense in depth: spotlighting, Prompt Shields detection, permission-based mitigation. *(deployment)*
- **NVIDIA** [Content Moderation and Safety Checks with NeMo Guardrails](https://developer.nvidia.com/blog/content-moderation-and-safety-checks-with-nvidia-nemo-guardrails/): Wiring LlamaGuard plus fact-check rails into RAG apps via NeMo Guardrails config. *(product design)*
- **Roblox** [Deploying ML for Voice Safety](https://about.roblox.com/newsroom/2024/07/deploying-ml-for-voice-safety): Machine-labeled data trains a fast quantized voice-abuse classifier at 2000 requests per second. *(deployment)*
- **Meta** [Llama Guard: LLM-based input-output safeguard](https://arxiv.org/abs/2312.06674): An instruction-tuned classifier moderating prompts and responses by taxonomy. *(product design)*
- **Google** [ShieldGemma: generative AI content moderation](https://arxiv.org/abs/2407.21772): Gemma-based safety classifiers benchmarked above Llama Guard. *(eval bar)*
- **Meta** [Llama Prompt Guard 2](https://developer.meta.com/ai/docs/model-cards-and-prompt-formats/prompt-guard/): A lightweight binary classifier flagging prompt injection and jailbreaks. *(product design)*
- **OpenAI** [How to implement LLM guardrails](https://developers.openai.com/cookbook/examples/how_to_use_guardrails): Async input and output guardrail patterns with latency tradeoffs. *(deployment)*
- **Cloudflare** [Block unsafe prompts with Firewall for AI](https://blog.cloudflare.com/block-unsafe-llm-prompts-with-firewall-for-ai/): An edge proxy using Llama Guard to block harmful prompts across 13 categories. *(deployment)*
- **Salesforce** [Inside the Einstein Trust Layer](https://developer.salesforce.com/blogs/2023/10/inside-the-einstein-trust-layer): PII masking, toxicity scoring, and prompt-injection defense around LLM calls. *(deployment)*
- **Grab** [How LLMs make content moderation more precise](https://www.grab.com/inside-grab/stories/how-large-language-models-help-us-make-more-precise-content-moderation-decisions/): Two-tier moderation routing content by an LLM violation-likelihood score. *(product design)*
- **Slack** [Securing the Agentic Enterprise](https://slack.com/blog/transformation/securing-the-agentic-enterprise): Multi-layered AI guardrails enforcing user permissions and real-time prompt-injection defense. *(deployment)*
- **Databricks** [Implementing LLM Guardrails for Safe GenAI Deployment](https://www.databricks.com/blog/implementing-llm-guardrails-safe-and-responsible-generative-ai-deployment-databricks): Safety filters on the Foundation Model API blocking unsafe inputs/outputs, logged for audit. *(deployment)*
- **Thomson Reuters** [Inside CoCounsel's guardrails](https://legal.thomsonreuters.com/blog/why-your-legal-ai-needs-more-than-the-open-web-a-look-inside-cocounsels-guardrails/): Grounding legal AI in trusted sources with attorney review and nightly 1,500-test benchmarks. *(eval bar)*
- **Wealthsimple** [Our LLM Gateway for secure, reliable generative AI](https://engineering.wealthsimple.com/get-to-know-our-llm-gateway-and-how-it-provides-a-secure-and-reliable-space-to-use-generative-ai): An internal gateway redacting PII and tracking external data for safe employee GenAI use. *(deployment)*
- **LinkedIn** [Defending Against Abuse at LinkedIn's Scale](https://www.linkedin.com/blog/engineering/trust-and-safety/defending-against-abuse-at-linkedins-scale): Real-time abuse defense at 4M+ transactions/sec using ML models and statistical rules. *(deployment)*
- **Pinterest** [How Pinterest built its Trust & Safety team](https://medium.com/pinterest-engineering/how-pinterest-built-its-trust-safety-team-8d6c026dd4b9): Building moderation infrastructure, policies, real-time signals, and ML before scaling. *(product design)*
- **Discord** [Our Approach to Content Moderation](https://discord.com/safety/our-approach-to-content-moderation): Layered human plus ML moderation including AutoMod filters and CSAM detection. *(product design)*
