# Datasets, by what you would use them for

"Where would the data come from?" is asked in almost every LLM system design loop,
and it is the question candidates answer worst. The usual answer names a model and
waves at the data. This page is the other half: the corpora, instruction sets,
preference data and benchmarks people actually build on, grouped by the stage of the
pipeline that consumes them.

Same cap and same rule as [papers.md](papers.md): at most eight per section, and each
entry says what it is for and what is wrong with it. A dataset with no stated limit
is a dataset nobody has used in anger.

**Four questions to ask about any dataset, in this order.** They come up as
follow-ups more often than the names do.

1. **What is the licence, and does it cover the model you train on it?** Many
   research sets are non-commercial or carry a "distilled from a proprietary model"
   restriction. Check the card, not the paper.
2. **What is the provenance?** Scraped, synthetic, expert-written, or logged from
   users. Each has a different failure mode, and only one of them has consent
   questions attached.
3. **Is it in your evaluation?** A training set that overlaps your benchmark makes
   every number afterwards meaningless. See the contamination section of
   [Benchmarking a model](topics/16-benchmark-evaluation.md).
4. **Has it been deduplicated, and against what?** Near-duplicate removal changes
   both quality and memorization, and it is the one preprocessing step whose absence
   is measurable later.

---

## Pretraining corpora

- **[The Pile](https://arxiv.org/abs/2101.00027)** (2021). 22 mixed sources
  assembled deliberately rather than scraped uniformly. Read it for the argument
  that a mixture is a design decision. Parts of it carry known licence problems, so
  treat it as a reference design more than a drop-in corpus.
- **[C4](https://arxiv.org/abs/1910.10683)** (2019). The original filtered Common
  Crawl, and the source of the heuristic filters everyone still starts from. Its
  filters are also famously blunt, removing more than intended.
- **[RefinedWeb](https://arxiv.org/abs/2306.01116)** (2023). Evidence that
  properly filtered web data alone beats curated corpora. Only a subset is public.
- **[FineWeb](https://arxiv.org/abs/2406.17557)** (2024). The current reference for
  how filtering decisions get ablated one at a time, with the eval that justified
  each. FineWeb-Edu is the classifier-filtered subset most small models now use.
- **[Dolma](https://arxiv.org/abs/2402.00159)** (2024). Trillions of tokens with the
  toolkit and the decisions published alongside, which is what makes it reproducible
  rather than merely open.
- **[DataComp-LM](https://arxiv.org/abs/2406.11794)** (2024). A benchmark where the
  model is fixed and the data varies. Use it as the experiment design when you claim
  a data change helped.
- **[Deduplicating Training Data Makes Language Models Better](https://arxiv.org/abs/2107.06499)**
  (2021). Not a corpus but the preprocessing result that governs all of them.

## Instruction and chat data

- **[OpenAssistant Conversations](https://arxiv.org/abs/2304.07327)** (2023).
  Human-written, multi-turn, multilingual, with quality ratings, collected in the
  open. The reference for what volunteer-collected instruction data looks like.
- **[LMSYS-Chat-1M](https://arxiv.org/abs/2309.11998)** (2023). A million real
  conversations from a public arena. Its value is distributional: it shows what
  people actually ask, which curated sets do not.
- **[WildChat](https://arxiv.org/abs/2405.01470)** (2024). The same idea with
  consented logging and toxicity annotation, and the closest public stand-in for
  production traffic.
- **[Tulu 3](https://arxiv.org/abs/2411.15124)** (2024). Not just a dataset but a
  full published mixture with the decontamination and eval gates attached. Start
  here when someone asks what an SFT mixture contains.
- **[Textbooks Are All You Need](https://arxiv.org/abs/2306.11644)** (2023). The
  synthetic-data case, and the paper to cite for both its promise and the caveat
  that its quality claims are hard to reproduce independently.

Two practical notes that come up as follow-ups. Instruction sets distilled from a
proprietary model usually carry terms that forbid using them to train a competing
model, and that restriction survives redistribution. And chat logs are personal
data: consent, retention and deletion are part of the design, not a legal footnote.

## Preference and alignment data

- **[Anthropic HH-RLHF](https://arxiv.org/abs/2204.05862)** (2022). The original
  large helpfulness and harmlessness comparison set, and still the clearest
  illustration of how a preference dataset encodes a policy.
- **[UltraFeedback](https://arxiv.org/abs/2310.01377)** (2023). Large-scale AI
  feedback with per-aspect ratings. It is what most open DPO recipes actually train
  on, so know that the preferences are model-generated.
- **[Constitutional AI](https://arxiv.org/abs/2212.08073)** (2022). The method that
  turns a written policy into preference labels, which is the alternative to buying
  them.
- **[Let's Verify Step by Step](https://arxiv.org/abs/2305.20050)** (2023). Released
  step-level correctness labels for math reasoning, the data behind process
  supervision.

The interview question underneath this section is usually "who wrote the
preferences, and what did they optimize for". Vendor-collected, volunteer, or
model-generated are three different answers with three different biases.

## Retrieval and embeddings

- **[MS MARCO](https://microsoft.github.io/msmarco/)**. Real Bing queries with
  passage relevance labels. The default training set for dense retrievers, and the
  reason most public models are tuned to short web-search-shaped queries.
- **[BEIR](https://arxiv.org/abs/2104.08663)** (2021). Zero-shot retrieval across
  many domains. The result to remember is that a model tuned on MS MARCO often does
  not transfer, which is why you evaluate retrieval on your own corpus.
- **[MTEB](https://arxiv.org/abs/2210.07316)** (2022). The embedding leaderboard.
  Useful for shortlisting, and heavily overfit as a target, so re-rank the shortlist
  on your data.
- **[Natural Questions](https://ai.google.com/research/NaturalQuestions)** and
  **[HotpotQA](https://hotpotqa.github.io/)**. Single-hop and multi-hop QA over
  Wikipedia, the two shapes a RAG evaluation set usually imitates.

For a production RAG system none of these is your eval. They are how you calibrate
the components before you have your own labelled queries, and a set of fifty
questions written by your own domain experts beats all of them for the decision you
are actually making.

## Evaluation: knowledge and reasoning

- **[MMLU](https://arxiv.org/abs/2009.03300)** (2020). The most quoted knowledge
  benchmark, saturated at the top and carrying a measurable item-error rate.
- **[GPQA](https://arxiv.org/abs/2311.12022)** (2023). Graduate-level questions
  designed to resist search. The Diamond subset is small, so its error bars are
  wide: about 7 points at 95 percent.
- **[MATH](https://arxiv.org/abs/2103.03874)** (2021) and
  **[GSM8K](https://arxiv.org/abs/2110.14168)** (2021). Competition and grade-school
  math. Both are heavily contaminated by now, which is what GSM1k was built to show.
- **[HELM](https://arxiv.org/abs/2211.09110)** (2022). Not a dataset but the
  multi-metric scaffolding that stops any of the above from being read as one number.
- **[LiveBench](https://arxiv.org/abs/2406.19314)** (2024). Refreshed items, so
  contamination is handled by construction rather than detection.

## Evaluation: code and agents

- **[HumanEval](https://arxiv.org/abs/2107.03374)** (2021) and
  **[MBPP](https://arxiv.org/abs/2108.07732)** (2021). Small function-level code
  benchmarks. Both are saturated and contaminated; quote them only for continuity
  with older results.
- **[SWE-bench](https://arxiv.org/abs/2310.06770)** (2023). Real repository issues
  scored by the project's own tests. The Verified subset is the one to quote, and the
  scaffold has to be reported with the number.
- **[LiveCodeBench](https://arxiv.org/abs/2403.07974)** (2024). Release-window
  scoring, so a model is judged only on problems published after its cutoff.
- **[tau-bench](https://arxiv.org/abs/2406.12045)** (2024). Tool-agent-user
  interaction with a simulated user, and the source of pass^k as a reliability metric.
- **[Establishing Best Practices for Building Rigorous Agentic Benchmarks](https://arxiv.org/abs/2507.02825)**
  (2025). Read before trusting any agent number, including your own: it documents how
  often the checker accepts a wrong answer.

## Multimodal: training pairs and evaluation

- **[LAION-5B](https://arxiv.org/abs/2210.08402)** (2022). Billions of image-text
  pairs from Common Crawl. The dataset most open image models were built on, and the
  one whose content and licensing problems are best documented. It is distributed as
  URLs, so it decays as the web does.
- **[DataComp](https://arxiv.org/abs/2304.14108)** (2023). Fixes the training recipe
  and competes on the data filtering, which is how multimodal data curation became a
  measurable discipline rather than a preference.
- **[OBELICS](https://arxiv.org/abs/2306.16527)** (2023). Interleaved image-text
  documents rather than isolated pairs, which is the data shape a modern VLM wants.
- **[MMMU](https://arxiv.org/abs/2311.16502)** (2023). Expert-level multimodal
  questions across disciplines, currently the reference capability benchmark for VLMs.
- **[Video-MME](https://arxiv.org/abs/2405.21075)** (2024). Video understanding
  across short, medium and long durations, with subtitles and audio available, which
  is what makes it usable for measuring a frame-budget decision.
- **[LongVideoBench](https://arxiv.org/abs/2407.15754)** (2024). Referring reasoning
  over long interleaved video and language, built so that a model cannot answer from
  a single sampled frame.

## Image and video generation

- **[LAION-Aesthetics](https://laion.ai/blog/laion-aesthetics/)**. The aesthetic-score
  filtered subsets that most open text-to-image models trained on. Read it as the
  clearest example of a filter encoding a taste.
- **[Emu](https://arxiv.org/abs/2309.15807)** (2023). Quality-tuning on a couple of
  thousand hand-picked images changes output quality more than another scrape, which
  is the counter-argument to scale in this area.
- **[Stable Video Diffusion](https://arxiv.org/abs/2311.15127)** (2023). The paper
  documents the video-data curation pipeline (cut detection, captioning, motion and
  aesthetic filtering) in more detail than the model architecture.

Video training data is where licensing questions are sharpest right now, and the
honest interview answer names that constraint before it names a source.

---

## Contributing

Same rules as [papers.md](papers.md): eight per section, one line on what it is for
and one on its limit, and a swap argued in the pull request when a section is full.
Do not add a dataset without checking its card for the licence, and do not restate a
size you have not seen stated on the card or in the paper. See
[CONTRIBUTING.md](CONTRIBUTING.md) and [template/](template/).
