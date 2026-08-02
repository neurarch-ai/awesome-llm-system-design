# 7. How teams do it in production

Everyone who runs benchmarks seriously converges on the same skeleton: pinned items,
a standardized harness, raw outputs kept, and a report that carries the protocol.
What differs is **what they standardize** (the prompt, the environment, or the
statistics) and **what they treat as the threat** (contamination, gaming, or
irreproducibility). Reading the table by those two columns is the fastest way to
place any eval stack you meet.

## Where the real designs diverge

| System | What it standardizes | Scoring approach | Contamination stance | Statistics | When it wins | Watch out |
|---|---|---|---|---|---|---|
| LM Evaluation Harness (EleutherAI) | Prompt rendering, task definitions, versioned task configs | Log-likelihood and generative, per task | Documents the risk; leaves policy to the user | Per-task stderr reported | Reproducible static-suite comparisons across many models | Defaults differ from vendor protocols, so numbers still need a protocol hash |
| HELM (Stanford CRFM) | Multi-scenario, multi-metric reporting matrix | Task metrics plus efficiency and calibration alongside accuracy | Public scenario set, so contamination grows with age | Reports across scenarios rather than one index | Holistic comparison where one number would mislead | Heavy to run; scenario coverage ages |
| Inspect (UK AI Security Institute) | The agent loop and the tool sandbox | Solvers plus scorers, including model-graded scorers | Designed for private and safety evals | Per-sample logs for audit | Agentic and safety evaluations that need a controlled environment | Requires environment engineering, not just prompts |
| simple-evals (OpenAI) | Chat-formatted generative scoring with readable prompts and parsers | Generative plus answer matching | Prefers newer sets over saturated ones | Minimal by design | Making a protocol legible so others can replicate it | Deliberately minimal; not a full platform |
| HealthBench (OpenAI) | Per-item expert rubric criteria with weights | Model grader scoring criterion by criterion, meta-evaluated against physicians | Purpose-built set, released with the rubric | Meta-eval of grader against expert judgment | Expert open-ended domains where holistic scores do not reproduce | Rubric construction is the cost; graders need re-certification |
| Anthropic eval-statistics practice | The analysis, not the prompt | Any | Orthogonal | Clustered SE, paired differences, resampling, power | Deciding whether a reported gap is real | Correct analysis often turns headline gaps into ties, which is unpopular |
| LMArena | Human pairwise preference at traffic scale | Bradley-Terry over votes, plus style-controlled variants | Fresh prompts arrive continuously | Ranking intervals, style control | Aggregate human taste on open-ended chat | Best-of-N private submissions and style effects, disputed in size |
| Artificial Analysis | A composite index plus price and speed per provider | Aggregates many public evals | Refreshes components as they saturate | Composite over many evals | Cross-provider serving and quality tradeoffs in one view | An index hides which component moved |
| METR | Task suites with human baselines, measured in task length | Task success as a function of human completion time | Private task suite | Logistic fit for the 50 percent time horizon | Expressing capability in units a business understands | Expensive human baselining; small task counts |
| LiveBench | Monthly refreshed items from recent sources | Objective ground truth, no judge in the loop | Contamination bounded by construction | Refresh policy stated | A contamination check on any static-suite result | The pool changes, so cross-time comparisons need a pinned window |
| LiveCodeBench | Problems tagged with release dates | Unit tests, plus self-repair and execution variants | Evaluate strictly after a model's cutoff | Windowed comparisons | Code capability without the old-problem contamination question | Window choice changes the number; state it |
| SWE-bench and SWE-bench Verified | The repository environment and the test command | Unit-test pass on a real repo state | Old issues are public and aged | Per-instance logs | Realistic software-engineering signal | Weak test suites can accept wrong patches; container drift |

## The dividing line

Two axes explain nearly every difference above. **What you standardize** determines
what your numbers are comparable to: standardize the prompt and you can compare
against other harness users; standardize the environment and you can compare agents;
standardize the analysis and you can compare claims. **What you treat as the threat**
determines your design: if it is contamination you build time-gating, if it is gaming
you build private held-out sets and audit the submission process, and if it is
irreproducibility you build protocol hashes and deterministic serving.

A complete interview answer picks a point on both axes and justifies it from the
decision the number drives. Picking a base model for a product tolerates aged public
suites plus one internal set. Publishing a model-card number does not.

## First-party sources

- **EleutherAI** [Lessons from the Trenches on Reproducible Evaluation of Language Models](https://arxiv.org/abs/2405.14782) and the [LM Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness): why prompt format and scoring mode change results, and a versioned task system that pins both.
- **Stanford CRFM** [HELM](https://crfm.stanford.edu/helm/): holistic multi-scenario, multi-metric evaluation instead of a single headline number.
- **UK AI Security Institute** [Inspect](https://inspect.aisi.org.uk/): an evaluation framework built around solvers, scorers, and sandboxed tool environments for agentic and safety evals.
- **OpenAI** [simple-evals](https://github.com/openai/simple-evals): a deliberately small generative-scoring harness whose prompts and parsers are meant to be read.
- **OpenAI** [HealthBench](https://openai.com/index/healthbench/) and the [paper](https://arxiv.org/abs/2505.08775): per-conversation physician-written rubric criteria, model-graded, with expert meta-evaluation of the grader.
- **Anthropic** [Adding Error Bars to Evals](https://arxiv.org/abs/2411.00640): clustered standard errors, paired differences, resampling, and power analysis applied to eval reporting.
- **LMArena** [response to The Leaderboard Illusion](https://lmarena.ai/blog/our-response/), alongside the critique itself, [The Leaderboard Illusion](https://arxiv.org/abs/2504.20879): what private best-of-N submission does to a preference ranking, and how much it matters.
- **METR** [Measuring AI Ability to Complete Long Software Tasks](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) and the [paper](https://arxiv.org/abs/2503.14499): the 50 percent time-horizon methodology with human baselines.
- **LiveBench** [livebench.ai](https://livebench.ai/) and the [paper](https://arxiv.org/abs/2406.19314): monthly-refreshed items with objective ground truth and no judge in the loop.
- **LiveCodeBench** [livecodebench.github.io](https://livecodebench.github.io/) and the [paper](https://arxiv.org/abs/2403.07974): release-date-tagged problems evaluated in windows after a model's cutoff.
- **Princeton NLP** [SWE-bench](https://arxiv.org/abs/2310.06770): real GitHub issues resolved against the repository's own tests.
- **UIUC and collaborators** [Establishing Best Practices for Building Rigorous Agentic Benchmarks](https://arxiv.org/abs/2507.02825): the checklist for task specification and outcome validation, with measured overestimation in widely used agentic benchmarks.
- **Thinking Machines Lab** [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/): why identical requests differ, and batch-invariant kernels that fix it.
- **Epoch AI** [benchmarking hub](https://epoch.ai/benchmarks): independently re-run benchmark results with the protocol documented.

For the dense single-file reference (same material, interview-walkthrough shape):
[topics/16-benchmark-evaluation.md](../../topics/16-benchmark-evaluation.md).
