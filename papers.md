# Papers, by topic

The repo's [production case studies](CASE-STUDIES.md) answer "how did a real team
ship this." This page answers the other half: **which papers does an interviewer
assume you have read** before they ask a follow-up.

Every topic caps at eight. That cap is the point. A list of two thousand papers is
an index of a field, not a study plan, and nobody reads one before a loop. These are
the ones whose result gets referenced by name in an interview: "that is the GQA
tradeoff", "that is the Chinchilla point", "that is what SelfCheckGPT does."

**How to read this page.** For each entry, the sentence after the year is the thing
to be able to say about it. If you can produce that sentence and the number attached
to it, you have gotten what an interview will ask for. Read the abstract and the one
figure that carries the result; read the whole paper only for the topics you are
being hired to own.

A paper appears under two topics when both interviews genuinely assume it. That
repetition is deliberate: this is a study list per topic, not a bibliography.

Not on this page: the data. Corpora, instruction sets, preference data and benchmarks
are in [datasets.md](datasets.md), organized the same way and under the same cap.

Also not on this page: engineering blog posts and first-party writeups. Those are the
**Seen in production** section of every topic, rolled up in
[CASE-STUDIES.md](CASE-STUDIES.md) (with per-system
[teardowns](CASE-TEARDOWNS.md)), and for several topics they carry more of the real
answer than the papers do.

---

## Start here

The five an interviewer will not stop to explain, in any LLM loop, for any role.

- **[Attention Is All You Need](https://arxiv.org/abs/1706.03762)** (2017). The
  transformer. You need it for the shapes: what Q, K and V are, why the KV cache is
  a thing at all, and where the quadratic term in prefill comes from.
- **[Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165)**
  (2020). GPT-3. In-context learning as the reason prompting is a design surface
  rather than a hack, and the first serious "just scale it" result.
- **[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)**
  (2020). Loss is a power law in compute, data and parameters. This is the paper
  that makes "how big should the model be" a calculation instead of an opinion.
- **[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)**
  (2022). Chinchilla. Corrects the previous one: most models of that era were far
  too big for their token budget. The 20-tokens-per-parameter rule of thumb comes
  from here, and interviewers still use it as a sanity check.
- **[Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)**
  (2022). InstructGPT. Why a base model is not a product, and the SFT then reward
  model then PPO shape that every post-training pipeline is still a variant of.

One more if you have time: **[The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)**
(2024), which is the most complete public description of an end-to-end modern build,
data through post-training through serving, and useful as a reference answer for
almost any "how would you actually do this" question.

---

## Building and adapting the model

### [The LLM training lifecycle](topics/13-llm-lifecycle.md) · [book chapter](book/llm-lifecycle/)

- **[The Llama 3 Herd of Models](https://arxiv.org/abs/2407.21783)** (2024). The
  whole lifecycle in one paper, with the numbers: data mix, compute, parallelism,
  post-training stages, and the serving decisions at the end.
- **[DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)** (2024). The
  cost-conscious counterpart: MoE plus MLA plus FP8 training, and an explicit
  training-cost figure, which is why it comes up whenever "could a small team do
  this" is the question.
- **[Tulu 3](https://arxiv.org/abs/2411.15124)** (2024). Fully open post-training
  recipe, including the eval suite and the decontamination, so it is the citable
  answer for what a modern SFT plus preference-tuning stack contains.
- **[DeepSeek-R1](https://arxiv.org/abs/2501.12948)** (2025). Reasoning ability
  from RL with verifiable rewards, and the reason "post-training" now includes a
  stage that did not exist in the InstructGPT picture.
- **[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)**
  (2020) and **[Chinchilla](https://arxiv.org/abs/2203.15556)** (2022). The budget
  arithmetic that decides what stage you are even allowed to be in.

### [Data curation and pretraining](topics/14-data-curation-and-pretraining.md) · [book chapter](book/data-and-pretraining/)

- **[Exploring the Limits of Transfer Learning (C4)](https://arxiv.org/abs/1910.10683)**
  (2019). The original filtered web corpus, and the source of the heuristic filters
  everyone still starts from.
- **[Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)**
  (2021). Near-duplicate removal improves the model and cuts memorization. This is
  the paper to name when asked why dedup is a pipeline stage rather than a nicety.
- **[The RefinedWeb Dataset for Falcon LLM](https://arxiv.org/abs/2306.01116)**
  (2023). Web data only, properly filtered, beats curated corpora. It settles the
  "do we need books and code" question with an experiment.
- **[The FineWeb Datasets](https://arxiv.org/abs/2406.17557)** (2024). The current
  reference for how filtering decisions are actually ablated, one at a time, with
  the eval that justified each.
- **[DataComp-LM](https://arxiv.org/abs/2406.11794)** (2024). Fixes the model and
  varies the data, which is the experiment design an interviewer wants when you
  claim a data change helped.
- **[Megatron-LM](https://arxiv.org/abs/1909.08053)** (2019) and
  **[ZeRO](https://arxiv.org/abs/1910.02054)** (2019). Tensor parallelism and the
  optimizer-state sharding that make the memory arithmetic work. Between them they
  cover most of what "how do you shard a 70B train" is asking.
- **[Textbooks Are All You Need](https://arxiv.org/abs/2306.11644)** (2023). Data
  quality substituting for scale, and the case for synthetic data with its caveats
  attached.
- **[Tensor Programs V (muP)](https://arxiv.org/abs/2203.03466)** (2022). Tune
  hyperparameters on a small model and transfer them. The answer to "you cannot
  afford to sweep learning rates at that scale."

### [Mid-training: continued pretraining and long context](topics/15-continued-pretraining-and-long-context.md) · [book chapter](book/mid-training/)

- **[RoFormer (RoPE)](https://arxiv.org/abs/2104.09864)** (2021). Rotary position
  embeddings. Every context-extension trick below is a modification of this, so it
  is the prerequisite, not optional background.
- **[Extending Context Window via Position Interpolation](https://arxiv.org/abs/2306.15595)**
  (2023). The first cheap extension: squeeze positions into the trained range. Know
  why it works and what it costs at short context.
- **[YaRN](https://arxiv.org/abs/2309.00071)** (2023). The interpolation method
  actually used in production, per-frequency rather than uniform.
- **[LongRoPE](https://arxiv.org/abs/2402.13753)** (2024). Search over the rescaling
  factors, to 2M tokens, plus the recovery training that keeps short-context quality.
- **[Don't Stop Pretraining](https://arxiv.org/abs/2004.10964)** (2020). Domain and
  task adaptive pretraining. Still the cleanest evidence for "continue pretraining
  before you fine-tune."
- **[Simple and Scalable Strategies to Continually Pre-train LLMs](https://arxiv.org/abs/2403.08763)**
  (2024). Learning-rate re-warming plus a replay fraction, which is the concrete
  answer to catastrophic forgetting.
- **[DoReMi](https://arxiv.org/abs/2305.10429)** (2023). Domain weights optimized
  rather than guessed, the mechanism behind mixture reweighting.
- **[RULER](https://arxiv.org/abs/2404.06654)** (2024). Claimed context length
  versus usable context length. Bring this one when someone quotes a 1M window.

### [Fine-tuning and post-training](topics/05-post-training-pipeline.md) · [book chapter](book/post-training/)

- **[LoRA](https://arxiv.org/abs/2106.09685)** (2021). Low-rank adapters. The
  parameter and memory arithmetic here is the most commonly asked back-of-envelope
  in the whole post-training topic.
- **[QLoRA](https://arxiv.org/abs/2305.14314)** (2023). 4-bit base plus adapters,
  which is what makes "fine-tune a 65B on one GPU" a real answer rather than a wish.
- **[Direct Preference Optimization](https://arxiv.org/abs/2305.18290)** (2023).
  Preference tuning without a separate reward model or RL loop. Be able to say what
  it gives up relative to PPO, not just that it is simpler.
- **[Constitutional AI](https://arxiv.org/abs/2212.08073)** (2022). Preference labels
  from a written policy instead of from people, and the origin of RLAIF.
- **[Tulu 3](https://arxiv.org/abs/2411.15124)** (2024). The full open recipe, with
  the eval gates and decontamination that a real pipeline needs and most answers skip.
- **[Llama 2](https://arxiv.org/abs/2307.09288)** (2023). Still the most detailed
  public account of iterative RLHF with two reward models, useful for the "how many
  rounds, and what breaks" follow-up.

---

## Inference and serving

### [Long-context inference and the KV cache](topics/02-long-context-and-kv-cache.md) · [book chapter](book/kv-cache/)

- **[FlashAttention](https://arxiv.org/abs/2205.14135)** (2022). Attention is
  memory-bound, not compute-bound. This reframing is the foundation of the whole
  cost model, and the reason tiling beats a smarter algorithm.
- **[GQA](https://arxiv.org/abs/2305.13245)** (2023). Grouped-query attention. The
  single most asked "shrink the cache" lever, and the quality-versus-memory tradeoff
  you should be able to quantify.
- **[Efficient Memory Management with PagedAttention](https://arxiv.org/abs/2309.06180)**
  (2023). vLLM. Fragmentation as the real reason batch sizes were small, solved with
  a page table. Expect to be asked to explain it as if it were virtual memory.
- **[DeepSeek-V2](https://arxiv.org/abs/2405.04434)** (2024). Multi-head latent
  attention: compress the cache instead of sharing heads. The current alternative to
  GQA, so knowing both is what separates a current answer from a 2023 one.
- **[Efficient Streaming LMs with Attention Sinks](https://arxiv.org/abs/2309.17453)**
  (2023). Why dropping the first tokens destroys quality, and the fix. The best
  single paper on what a sliding window actually breaks.
- **[H2O: Heavy-Hitter Oracle](https://arxiv.org/abs/2306.14048)** (2023). Evicting
  KV entries by attention mass, the reference point for every cache-eviction policy.
- **[SnapKV](https://arxiv.org/abs/2404.14469)** (2024). Compress the prompt's cache
  at prefill using the attention the prompt itself produces.
- **[KIVI](https://arxiv.org/abs/2402.02750)** (2024). 2-bit KV quantization, tuning
  free, and the per-channel versus per-token asymmetry that makes it work.

### [LLM inference serving at scale](topics/04-inference-serving-at-scale.md) · [book chapter](book/inference-serving/)

- **[Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)**
  (2022). A draft model proposes, the target verifies, and the output distribution
  is unchanged. That last clause is the part interviews test.
- **[Medusa](https://arxiv.org/abs/2401.10774)** (2024). Extra decoding heads instead
  of a second model, which removes the "where do I get a draft model" problem.
- **[EAGLE](https://arxiv.org/abs/2401.15077)** (2024). Speculate in feature space,
  currently the strongest acceptance rates. Know why feature-level beats token-level.
- **[Sarathi-Serve](https://arxiv.org/abs/2403.02310)** (2024). Chunked prefill:
  the throughput-versus-latency knob, and how prefill stalls decode in one batch.
- **[Splitwise](https://arxiv.org/abs/2311.18677)** (2023) and
  **[DistServe](https://arxiv.org/abs/2401.09670)** (2024). Prefill and decode have
  different bottlenecks, so run them on different machines. The clearest statement of
  disaggregation, which is where serving architecture went.
- **[Mooncake](https://arxiv.org/abs/2407.00079)** (2024). A KV-cache-centric
  cluster: the cache, not the GPU, as the thing you schedule around.
- **[SGLang](https://arxiv.org/abs/2312.07104)** (2023). RadixAttention, prefix
  sharing across requests. The right answer to "many requests share a system prompt."

### [Cost optimization and model routing](topics/11-cost-optimization-and-model-routing.md) · [book chapter](book/cost-optimization/)

- **[FrugalGPT](https://arxiv.org/abs/2305.05176)** (2023). Cascades and routing,
  with the cost-versus-quality frontier drawn explicitly. The reference for "use the
  cheap model first" as an engineered policy rather than a vibe.
- **[RouteLLM](https://arxiv.org/abs/2406.18665)** (2024). Train the router on
  preference data, and evaluate it as a router. Note what it needs: labelled
  preferences you probably do not have on day one.
- **[Mixtral of Experts](https://arxiv.org/abs/2401.04088)** (2024). Sparse MoE:
  active parameters versus total parameters, which is the distinction that makes an
  MoE cost estimate right or wrong.
- **[Chatbot Arena](https://arxiv.org/abs/2403.04132)** (2024). Where the quality
  number a router optimizes against comes from, and how noisy it is.

Most of the evidence for this topic is operational rather than academic. The
routing, caching and right-sizing writeups in
[CASE-STUDIES.md](CASE-STUDIES.md) carry more of the real answer than the papers.

### [Model compression](topics/17-model-compression.md) · [book chapter](book/model-compression/)

- **[LLM.int8()](https://arxiv.org/abs/2208.07339)** (2022). Outlier features are
  why naive quantization collapses. This is the paper that defines the problem the
  rest of the list is solving.
- **[GPTQ](https://arxiv.org/abs/2210.17323)** (2022). One-shot weight quantization
  to 3 and 4 bits with second-order information.
- **[AWQ](https://arxiv.org/abs/2306.00978)** (2023). Protect the salient weight
  channels, identified from activations. The most-deployed weight-only method.
- **[SmoothQuant](https://arxiv.org/abs/2211.10438)** (2022). Migrate activation
  outliers into the weights so both sides can be 8-bit, which is what makes W8A8
  serving practical.
- **[SparseGPT](https://arxiv.org/abs/2301.00774)** (2023) and
  **[Wanda](https://arxiv.org/abs/2306.11695)** (2023). One-shot pruning, and then
  the same result with a far simpler criterion. Read both: the second is the reason
  to be suspicious of complexity in this area.
- **[Sheared LLaMA](https://arxiv.org/abs/2310.06694)** (2023) and
  **[Minitron](https://arxiv.org/abs/2407.14679)** (2024). Structured pruning plus
  continued training, which is the only pruning shape hardware actually rewards.
- **[The Era of 1-bit LLMs](https://arxiv.org/abs/2402.17764)** (2024). BitNet
  b1.58. Know it as the frontier, and know it needs training from scratch.
- **[Accuracy is Not All You Need](https://arxiv.org/abs/2407.09141)** (2024). Two
  models with equal benchmark accuracy can disagree on half their answers. The
  paper to cite when someone accepts a compressed model on one aggregate number.

### [Reasoning and test-time compute](topics/18-reasoning-and-test-time-compute.md) · [book chapter](book/reasoning-serving/)

- **[Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903)** (2022). Where
  test-time compute as a lever starts.
- **[Self-Consistency](https://arxiv.org/abs/2203.11171)** (2022). Sample many, take
  the majority. The cheapest version of "spend more at inference", and the baseline
  any fancier method has to beat.
- **[Tree of Thoughts](https://arxiv.org/abs/2305.10601)** (2023). Search over
  reasoning states. Mostly useful now as the cost cautionary tale.
- **[Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)** (2023). Process
  supervision beats outcome supervision. The origin of the verifier in modern
  reasoning stacks.
- **[Large Language Monkeys](https://arxiv.org/abs/2407.21787)** (2024). Coverage
  scales with samples, but only if you can pick the right one. The pass@k versus
  pass^k distinction lives here.
- **[Scaling LLM Test-Time Compute Optimally](https://arxiv.org/abs/2408.03314)**
  (2024). When extra inference compute beats a bigger model, and when it does not.
- **[s1: Simple test-time scaling](https://arxiv.org/abs/2501.19393)** (2025).
  Budget forcing with 1k examples. The "you can do this cheaply" data point.
- **[DeepSeek-R1](https://arxiv.org/abs/2501.12948)** (2025). RL with verifiable
  rewards, and what a long thinking trace does to your serving cost model.

### [Realtime streaming chat](topics/10-realtime-streaming-chat.md) · [book chapter](book/streaming-chat/)

- **[Efficient Streaming LMs with Attention Sinks](https://arxiv.org/abs/2309.17453)**
  (2023). What breaks in an endless conversation, and the minimal fix.
- **[Whisper](https://arxiv.org/abs/2212.04356)** (2022). The ASR half of a voice
  pipeline, and where its latency actually goes.
- **[Moshi](https://arxiv.org/abs/2410.00037)** (2024). Full-duplex speech-text, the
  paper to name when asked how to remove turn-taking latency instead of shaving it.

This topic is mostly a transport and backpressure problem, so the
[production writeups](CASE-STUDIES.md) matter more here than the literature does.

---

## Retrieval and knowledge

### [RAG serving](topics/01-rag-serving.md) · [book chapter](book/rag-serving/)

- **[Retrieval-Augmented Generation for Knowledge-Intensive NLP](https://arxiv.org/abs/2005.11401)**
  (2020). The original. Worth reading for how much of the current stack was already
  there, and which part (the trained retriever) mostly was not adopted.
- **[Dense Passage Retrieval](https://arxiv.org/abs/2004.04906)** (2020). Dual
  encoders beating BM25, and the in-batch negatives trick that made it trainable.
- **[Lost in the Middle](https://arxiv.org/abs/2307.03172)** (2023). Position in the
  context window changes whether the model uses a passage. The single best argument
  for re-ranking and for a small k.
- **[HyDE](https://arxiv.org/abs/2212.10496)** (2022). Generate a hypothetical
  answer, embed that. The cheapest fix for query-document vocabulary mismatch.
- **[Self-RAG](https://arxiv.org/abs/2310.11511)** (2023). Retrieve on demand and
  critique what came back, rather than retrieving unconditionally.
- **[Ragas](https://arxiv.org/abs/2309.15217)** (2023). Reference-free RAG metrics
  (faithfulness, answer and context relevance). Know the definitions; you will be
  asked how to evaluate this without labels.
- **[GraphRAG](https://arxiv.org/abs/2404.16130)** (2024). For global questions no
  single chunk answers. Also read it for the cost, which is the usual reason not to.
- **[RAGO](https://arxiv.org/abs/2503.14649)** (2025). RAG serving as a scheduling
  and placement problem, which is the systems half the other papers skip.

### [Semantic search and embedding services](topics/08-semantic-search-and-embeddings.md) · [book chapter](book/semantic-search/)

- **[HNSW](https://arxiv.org/abs/1603.09320)** (2016). The index almost everything
  uses. Recall versus latency versus memory is tuned with the parameters in this
  paper, so know what ef and M do.
- **[ColBERT](https://arxiv.org/abs/2004.12832)** (2020). Late interaction: better
  quality than a single vector, at a storage cost you should be able to estimate.
- **[MTEB](https://arxiv.org/abs/2210.07316)** (2022). How embedding models are
  compared, and why a leaderboard rank does not transfer to your corpus.
- **[E5: Text Embeddings by Weakly-Supervised Contrastive Pre-training](https://arxiv.org/abs/2212.03533)**
  (2022). The recipe behind the open embedding models you would actually deploy.
- **[Embedding-based Retrieval in Facebook Search](https://arxiv.org/abs/2006.11632)**
  (2020). The first end-to-end production account: serving, hybrid retrieval, and
  the deployment problems that only appear at scale.
- **[Applying Embedding-Based Retrieval to Airbnb Search](https://arxiv.org/abs/2601.06873)**
  (2026) and **[Scaling Multilingual Semantic Search in Uber Eats](https://arxiv.org/abs/2603.06586)**
  (2026). Two recent first-party systems, useful because they report what they
  traded away, which papers usually do not.

---

## Building applications

### [Agent orchestration](topics/03-agent-orchestration.md) · [book chapter](book/agents/)

- **[ReAct](https://arxiv.org/abs/2210.03629)** (2022). Interleave reasoning and
  acting. The loop every agent framework is still running.
- **[Toolformer](https://arxiv.org/abs/2302.04761)** (2023). Tool use learned rather
  than prompted, and a useful contrast with today's fine-tuned tool-calling APIs.
- **[Reflexion](https://arxiv.org/abs/2303.11366)** (2023). Verbal self-critique in
  an episodic memory. Know it, and know the cost it adds per task.
- **[AutoGen](https://arxiv.org/abs/2308.08155)** (2023). Multi-agent conversation
  as a programming model. The reference for "should this be one agent or several",
  where the defensible answer is usually one.
- **[Voyager](https://arxiv.org/abs/2305.16291)** (2023). A skill library that grows,
  which is the cleanest example of agent memory that is not just a vector store.
- **[SWE-bench](https://arxiv.org/abs/2310.06770)** (2023). The benchmark that made
  agent claims falsifiable. Expect the "why is this hard to score" follow-up.
- **[tau-bench](https://arxiv.org/abs/2406.12045)** (2024). Tool-agent-user
  interaction with a simulated user, and pass^k as the reliability metric. This is
  the number to quote when asked whether an agent is production-ready.
- **[Measuring AI Ability to Complete Long Software Tasks](https://arxiv.org/abs/2503.14499)**
  (2025). Task length as the axis of difficulty, which reframes "what can an agent
  do" as "for how long can it stay on track."

### [Multimodal serving](topics/09-multimodal-serving.md) · [book chapter](book/multimodal/)

- **[CLIP](https://arxiv.org/abs/2103.00020)** (2021). Contrastive image-text
  pretraining. The encoder half of most VLMs, and the source of the shared space.
- **[ViT](https://arxiv.org/abs/2010.11929)** (2020). Images as patch tokens. Read it
  for the token-count arithmetic: patch size decides your serving cost.
- **[Flamingo](https://arxiv.org/abs/2204.14198)** (2022). Cross-attention into a
  frozen LLM, the first of the two architectural families.
- **[BLIP-2](https://arxiv.org/abs/2301.12597)** (2023). A small trained bridge
  between a frozen encoder and a frozen LLM: the cheapest way to build one.
- **[LLaVA](https://arxiv.org/abs/2304.08485)** (2023). A linear projector and
  instruction tuning. The second family, and the default starting point.
- **[MM1](https://arxiv.org/abs/2403.09611)** (2024). Ablations over the design
  choices, so it is the paper to cite for why a choice is made rather than just what.
- **[Qwen2-VL](https://arxiv.org/abs/2409.12191)** (2024). Native dynamic
  resolution: variable image tokens, and the batching problem that creates.
- **[Chameleon](https://arxiv.org/abs/2405.09818)** (2024). Early fusion, one token
  stream. The architecture to name when asked what comes after the projector.

---

## Quality and safety

### [LLM evaluation system](topics/06-evaluation-system.md) · [book chapter](book/evaluation/)

- **[Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685)**
  (2023). Establishes both that judges agree with humans at roughly human-human
  rates, and the biases (position, verbosity, self-preference) you must control for.
- **[Judging the Judges](https://arxiv.org/abs/2406.12624)** (2024). The failure
  modes measured. Read it before proposing an LLM judge as a regression gate.
- **[JudgeBench](https://arxiv.org/abs/2410.12784)** (2024). Judges evaluated on
  objectively checkable pairs, which is how you would validate your own.
- **[Stratified Prediction-Powered Inference](https://arxiv.org/abs/2406.04291)**
  (2024). Combine a few human labels with many model labels and keep a valid
  confidence interval. The rigorous answer to "we cannot label everything."
- **[How to Correctly Report LLM-as-a-Judge Evaluations](https://arxiv.org/abs/2511.21140)**
  (2025). What the error bar on a judged eval actually has to include.
- **[HealthBench](https://arxiv.org/abs/2505.08775)** (2025). Physician-written
  rubrics at scale: the best public model of a domain eval built with experts.

### [Benchmarking a model](topics/16-benchmark-evaluation.md) · [book chapter](book/benchmark-eval/)

- **[MMLU](https://arxiv.org/abs/2009.03300)** (2020) and
  **[HELM](https://arxiv.org/abs/2211.09110)** (2022). The benchmark everyone quotes,
  and the framework that argued a single number was never enough.
- **[Chatbot Arena](https://arxiv.org/abs/2403.04132)** (2024). Pairwise human
  preference with Elo, and its sampling and identifiability problems.
- **[The Leaderboard Illusion](https://arxiv.org/abs/2504.20879)** (2025). How
  private variants and selective reporting distort a public board. The strongest
  available answer to "why not just trust the leaderboard."
- **[Adding Error Bars to Evals](https://arxiv.org/abs/2411.00640)** (2024). The
  paper to have read before reporting any eval number at all.
- **[A Careful Examination of LLM Performance on Grade School Arithmetic](https://arxiv.org/abs/2405.00332)**
  (2024). GSM1k: a fresh copy of GSM8k exposes contamination as a measurable gap.
- **[LiveBench](https://arxiv.org/abs/2406.19314)** (2024) and
  **[LiveCodeBench](https://arxiv.org/abs/2403.07974)** (2024). Contamination
  handled by construction, with continuously refreshed questions.
- **[Establishing Best Practices for Building Rigorous Agentic Benchmarks](https://arxiv.org/abs/2507.02825)**
  (2025). Task validity and outcome validity, and how many agent benchmarks fail them.

### [Safety, moderation, and guardrails](topics/07-safety-and-guardrails.md) · [book chapter](book/safety/)

- **[Constitutional AI](https://arxiv.org/abs/2212.08073)** (2022). Safety from a
  written policy the model applies to itself, which is also how a policy becomes
  auditable.
- **[Llama Guard](https://arxiv.org/abs/2312.06674)** (2023). A dedicated
  input-output classifier with a taxonomy, the standard shape of a guardrail model.
- **[ShieldGemma](https://arxiv.org/abs/2407.21772)** (2024). The same idea at
  several sizes, so it is the paper for the latency-versus-accuracy tradeoff on the
  filter itself.
- **[Universal and Transferable Adversarial Attacks](https://arxiv.org/abs/2307.15043)**
  (2023). GCG. Automated, transferable jailbreak suffixes. This is why "we tested
  some prompts" is not a defense.
- **[Ignore Previous Prompt](https://arxiv.org/abs/2211.09527)** (2022). Prompt
  injection stated plainly, and still the clearest framing of why data in the
  context window is untrusted input.
- **[NeMo Guardrails](https://arxiv.org/abs/2310.10501)** (2023). Programmable
  rails as a system component, useful for what a policy-routing layer looks like.
- **[TruthfulQA](https://arxiv.org/abs/2109.07958)** (2021). Models imitate human
  falsehoods, and scale alone makes it worse. Good grounding for the difference
  between safety and correctness.

### [Production monitoring and observability](topics/12-production-monitoring-and-observability.md) · [book chapter](book/monitoring/)

- **[SelfCheckGPT](https://arxiv.org/abs/2303.08896)** (2023). Sample several
  responses and measure their consistency. The reference method for flagging
  hallucinations online with no labels and no ground truth.
- **[How is ChatGPT's behavior changing over time?](https://arxiv.org/abs/2307.09009)**
  (2023). A model you do not control drifts under you. The paper to cite for why a
  regression suite runs on a schedule and not only on your own releases.
- **[Stratified Prediction-Powered Inference](https://arxiv.org/abs/2406.04291)**
  (2024). Online quality estimates from a small human-labelled slice, with a valid
  interval attached.

This topic is thinner in the literature than the others on purpose: tracing,
sampling and alerting are engineering, and the
[production writeups](CASE-STUDIES.md) are the better source.

---

## Frontier

### [World models and embodied agents](book/world-models/)

- **[World Models](https://arxiv.org/abs/1803.10122)** (2018). Learn a compressed
  model of the environment, then train the policy inside it. The framing everything
  here inherits.
- **[MuZero](https://arxiv.org/abs/1911.08265)** (2019). Planning with a learned
  model that never has to reconstruct observations, only what matters for the value.
- **[DreamerV3](https://arxiv.org/abs/2301.04104)** (2023). One configuration across
  many domains, which is what made learned-model RL look like an engineering choice.
- **[GAIA-1](https://arxiv.org/abs/2309.17080)** (2023). A generative world model for
  driving, and the clearest example of the video-prediction-as-simulator argument.
- **[Genie](https://arxiv.org/abs/2402.15391)** (2024). Interactive environments
  learned from video with no action labels.
- **[OpenVLA](https://arxiv.org/abs/2406.09246)** (2024). An open vision-language
  action model, the concrete artifact behind "a VLM that outputs robot actions."
- **[V-JEPA 2](https://arxiv.org/abs/2506.09985)** (2025). Prediction in
  representation space rather than pixel space, plus zero-shot robot planning.
- **[WorldArena](https://arxiv.org/abs/2602.08971)** (2026). Evaluating perception
  against functional utility, which is the measurement problem this whole area has.

---

## Contributing to this page

Add a paper only with the sentence that says what it changes for the reader, and
only where a topic has fewer than eight. If a topic is full, the honest move is to
argue for a swap in the pull request rather than to make the list longer. See
[CONTRIBUTING.md](CONTRIBUTING.md).
