# 7. How teams do it in production

Every major long-context adaptation converges on the same skeleton: take a
pretrained base, re-enter the next-token objective under a lowered re-warmed
learning rate, rescale the RoPE positional frequencies, continue-train on
upsampled long documents in stages, and gate on a regression check plus a
RULER-style eval before post-training. What actually differs between teams is two
decisions: **how non-uniform their frequency rescaling is**, and **how they prove
the extended length is real**. The skeleton everyone shares; the leverage is in
the rescaling recipe and the eval discipline.

## Where the real designs diverge

| System | Axis | Extension mechanism | Reach | Key lever | Watch out |
|---|---|---|---|---|---|
| Meta Llama 3 | length | RoPE rescale + staged continued train | 8K to 128K in 6 stages | staged extension late in pretraining | 405B pretrain is lab-scale |
| Meta Code Llama | domain + length | NTK-ABF (RoPE base 10000 to 1000000) | 16K trained, extrapolates to 100K | continued pretrain a code domain from a general base | code narrows general ability; needs replay for general language |
| 01.AI Yi | length | continued pretrain on long data, RoPE rescaled | up to 200K | data quality over architecture novelty | long-data curation is the binding constraint |
| Nous YaRN | length | non-uniform freq scale + softmax temperature | 64K to 128K+ | roughly 0.1 percent of pretrain tokens | ramp bands and temperature coefficient must be tuned |
| Microsoft LongRoPE | length | evolutionary per-dimension search + progressive extension | beyond 2M tokens | searched non-uniform rescale + short-context recovery | search cost; a short-context recovery step is required |
| Alibaba Qwen2.5 | domain + length | progressive length + YaRN-style + Dual Chunk Attention | 128K open (1M turbo) | staged non-uniform scaling plus attention chunking | effective length below configured; verify with RULER |
| Mila continual pretraining | domain | re-warm + re-decay + replay (methodology, not a product) | matches from-scratch retrain | small replay fraction sharply cuts forgetting | modest re-warm peak is the key hyperparameter |
| Ai2 DAPT ("Don't Stop Pretraining") | domain | domain-adaptive and task-adaptive continued pretraining | task-specific NLP | canonical evidence that two-phase in-domain pretraining pays off | contamination risk; decontaminate against your evals |

The dividing line: **the rescaling recipe and the data quality buy the effective
context ceiling; the eval discipline is what tells you whether you actually reached
it.** Configured length and effective length diverge in almost every release; the
gap is the thing to measure.

## The systems (first-party write-ups)

- **Meta** [The Llama 3 Herd of Models](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/): extends the context window from 8K to 128K in six incremental stages late in pretraining, using RoPE rescaling plus grouped-query attention. Staged extension is cheaper (short sequences early) and more stable than one long-length run. *(length)*

- **Meta** [Code Llama: Open Foundation Models for Code](https://arxiv.org/abs/2308.12950): continued pretraining of Llama 2 on a code corpus with an Adjusted Base Frequency (RoPE base raised from 10000 to 1000000), trained on 16K sequences and extrapolating to inputs up to 100K tokens. The canonical production example that raising the RoPE base is a real non-uniform rescale, not a heuristic. *(domain + length)*

- **01.AI** [Yi: Open Foundation Models by 01.AI](https://arxiv.org/abs/2403.04652): 6B and 34B bilingual bases extended to 200K via continued pretraining on long data. The report credits data quality over architecture, a useful counterweight to the frequency-scaling literature: once RoPE rescaling is adequate, long-data curation is the binding constraint. *(length)*

- **Nous Research** [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071): non-uniform RoPE frequency scaling plus a softmax attention-temperature correction extends context at roughly 0.1 percent of original pretraining tokens, far cheaper than uniform interpolation. Became the default aggressive-extension recipe. *(length)*

- **Microsoft** [LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753): evolutionary search over per-dimension RoPE rescale factors plus progressive extension and a short-context recovery step reaches 2M+ tokens. Shows that the optimal rescaling is non-uniform and input-length dependent; a fixed closed-form cannot find the best factors that far out. *(length)*

- **Meta** [Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595): linear position interpolation uniformly compresses position indices into the trained range, extending LLaMA to 32K with about a thousand fine-tuning steps. Naive extrapolation fails catastrophically; this paper showed the minimal working alternative. *(length)*

- **Ai2** [Don't Stop Pretraining: Adapt Language Models to Domains and Tasks](https://arxiv.org/abs/2004.10964): domain-adaptive pretraining plus task-adaptive pretraining lift downstream tasks across four domains. The canonical evidence that a second in-domain pretraining phase pays off. *(domain)*

- **Mila** [Simple and Scalable Strategies to Continually Pre-train Large Language Models](https://arxiv.org/abs/2403.08763): learning-rate re-warming plus re-decaying plus a small replay fraction lets continued pretraining match a full from-scratch retrain at a fraction of the compute. Quantifies the forgetting-versus-learning tradeoff as a function of re-warm peak and replay ratio. *(domain)*

- **Alibaba** [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115): pretrained on 18 trillion tokens with long-context adaptation via progressive length increase, YaRN-style non-uniform scaling, and Dual Chunk Attention, reaching 128K on open models (up to 1M on the turbo variant). Candid that configured length and effective length differ. *(domain + length)*

- **NVIDIA** [RULER: What's the Real Context Size of Your Long-Context Language Models?](https://arxiv.org/abs/2404.06654): a synthetic benchmark with multi-hop variable tracing, aggregation, and multi-needle tasks shows most models that claim 32K+ degrade sharply well before their advertised length. The standard reference for measuring effective context. *(evaluation)*

- **Meta** [Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation (ALiBi)](https://arxiv.org/abs/2108.12409): a linear distance penalty on attention scores lets a model trained at short context extrapolate to longer inputs at test time without retraining. An alternative to RoPE rescaling, but requires adoption at pretraining time. *(length, architectural alternative)*
