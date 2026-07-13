# 7. How teams do it in production

Every serious pretraining effort converges on the same canonical funnel: raw web
archives plus curated corpora flow through extraction, language ID, quality
filtering, deduplication, decontamination, mixing, and tokenization, then feed
a distributed next-token-prediction run sized by scaling laws. What actually
differs is where each team invests: the extraction and dedup recipe that sets
the capability ceiling, the tokenizer that sets the token economy, the scaling
decision that decides size, or the parallelism and fault tolerance that decide
whether the run is feasible at all.

## Where real pipelines diverge

| System | Layer | Key lever | When it wins | Watch out |
|---|---|---|---|---|
| HuggingFace FineWeb | data prep | ablation-chosen heuristics + educational classifier (FineWeb-Edu); MinHash dedup within and across 96 CC dumps to 15T tokens | open pretrain data that beats prior public sets; fewer better tokens win on hard benchmarks | decontamination critical; over-dedup can lower scores, so per-dump plus measured global pass beats maximal global |
| TII RefinedWeb | data prep | careful WARC re-extraction + URL blocklists + MinHash dedup; web-only, no curated corpora | proving web data alone, with strong enough processing, can match curated corpora | the whole thesis rests on extraction quality; curated corpora still add value in narrow domains |
| Ai2 Dolma / OLMo | data prep + open base | 3T-token fully open corpus with documented toolkit; open training code, logs, checkpoints | studying curation as a science; a truly reproducible pretrain | fully open data is a legal and safety commitment; the bar for what you include is on the record |
| EleutherAI The Pile | data prep | 22 curated diverse domains, 800 GB | domain diversity over raw web scale; an early argument for mixture over volume | older; not web-scale-filtered; superseded by FineWeb-scale datasets for capacity |
| Google C4 | data prep | heuristic-cleaned Common Crawl (simple reproducible Gopher / C4 rules) | a clean, reproducible baseline web corpus | heuristics only; no learned filter; the ceiling is the quality of the rules |
| Meta CCNet | data prep | line-level dedup + per-language LM-perplexity quality filter | multilingual and low-resource languages where global filters drown non-English text | line-level reconstruction complexity; perplexity score needs calibration per language |
| Google SentencePiece | tokenizer | language-independent subword (BPE or unigram-LM); whitespace as a token | multilingual and non-whitespace-delimited languages (Chinese, Japanese, Thai) | vocab-size fertility tradeoff: bigger vocab means bigger embedding and softmax |
| Google DeepMind Chinchilla | scaling decision | compute-optimal scaling study (400+ models); about 20 tokens per parameter | fixed compute budget and you are minimizing training compute for a target loss | training-optimal ignores inference cost; overtrain a smaller model when serving dominates |
| Meta Llama 3 | full build | careful curation + dense pretrain (GQA, RoPE, RMSNorm) + staged context extension + elastic training | a strong open base at 8B to 405B; candid about failures and recovery at cluster scale | 405B pretrain is lab-scale; most teams should adapt Llama 3 rather than copy the run |
| DeepSeek-V3 | pretraining (MoE) | 671B total, 37B active per token; FP8 training; aux-loss-free load balancing | frontier-quality capacity at a small per-token compute budget | expert parallelism adds all-to-all traffic; every expert still sits in VRAM |
| NVIDIA Megatron-LM | systems | tensor and pipeline parallelism; split weight matrices across GPUs | a layer or model too large for one GPU | TP needs NVLink speeds; putting TP across the slow inter-node network collapses MFU |
| Microsoft ZeRO / PyTorch FSDP | systems | partition optimizer states, gradients, and parameters across data-parallel ranks instead of replicating them | fitting a model whose optimizer footprint exceeds one GPU's memory | extra all-gather and reduce-scatter every step; overlap with compute to protect MFU |

The dividing line is which layer the writeup is about: data pipeline sets the
capability ceiling, tokenizer sets the token economy, scaling decision sets the
size, and the systems layer decides whether the run is feasible at what MFU.

## The systems (first-party write-ups)

- **HuggingFace** [FineWeb: decanting the web for the finest text data at scale](https://huggingface.co/spaces/HuggingFaceFW/blogpost-fineweb-v1): a 15T-token open pretraining set from 96 CC dumps, ablation-chosen filters, MinHash dedup within and across dumps, and the FineWeb-Edu educational classifier. *(data recipe)*
- **TII** [The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only](https://arxiv.org/abs/2306.01116): careful WARC extraction plus filtering plus dedup shows properly processed web data alone can match or beat curated corpora. *(data recipe)*
- **Ai2** [Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research](https://arxiv.org/abs/2402.00159): a fully open 3T-token corpus and curation toolkit, the data behind OLMo. *(data recipe)*
- **Ai2** [OLMo: Accelerating the Science of Language Models](https://arxiv.org/abs/2402.00838): a base model released with its data, training code, and logs end to end; the reference for a reproducible pretrain. *(full build)*
- **EleutherAI** [The Pile: An 800GB Dataset of Diverse Text for Language Modeling](https://arxiv.org/abs/2101.00027): 22 curated domains for diversity; an early argument that mixture and domain coverage drive quality. *(data recipe)*
- **Google** [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer (C4)](https://arxiv.org/abs/1910.10683): the C4 corpus, Common Crawl cleaned with simple reproducible heuristic filters; a canonical clean-web baseline. *(data recipe)*
- **Meta** [CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data](https://arxiv.org/abs/1911.00359): line-level deduplication plus a language-model perplexity quality filter, run per language; the template for multilingual crawl curation. *(data recipe)*
- **Google** [SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing](https://arxiv.org/abs/1808.06226): reversible, language-independent subword tokenizer; the default for multilingual models. *(tokenizer)*
- **Google DeepMind** [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556): 400+ models show parameters and tokens should scale together, about 20 tokens per parameter; a 70B Chinchilla beats 280B Gopher at equal compute. *(scaling decision)*
- **Meta** [The Llama 3 Herd of Models](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/): end-to-end open recipe from careful data curation to a scaled dense pretrain, staged context extension, and the elastic-training and failure-recovery systems that keep a cluster-scale run alive. *(full build)*
- **DeepSeek** [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437): 671B-parameter MoE (about 37B active per token) trained with FP8 and auxiliary-loss-free load balancing; frontier scale on a constrained compute budget. *(pretraining, MoE)*
- **NVIDIA** [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053): tensor and pipeline parallelism splits weight matrices across GPUs so a layer too large for one device still trains. *(systems)*
- **Microsoft** [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054): partition optimizer states, gradients, and parameters across data-parallel ranks instead of replicating them; per-GPU memory falls toward a fraction of the full footprint. *(systems)*
- **Meta** [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277): the native-PyTorch realization of ZeRO-3-style sharding; all-gather parameters just before use, free them right after. *(systems)*

For the full comparison including the canonical divergence diagram and the
detailed choices table, see the dense reference in
[topics/14-data-curation-and-pretraining.md](../../topics/14-data-curation-and-pretraining.md).
