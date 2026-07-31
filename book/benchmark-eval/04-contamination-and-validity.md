# 4. Contamination and item validity

Two things can make a benchmark number meaningless while the harness runs
flawlessly: the model has already seen the answers, or the items were never right in
the first place. Both are measurement problems, not model problems, and both are
routinely hand-waved in interviews with "we deduplicated the training data."

## The five kinds of leakage

Contamination is usually discussed as if there were one kind. There are five, and
they need different defenses.

| Kind | How it happens | Why deduplication misses it |
|---|---|---|
| Verbatim item leakage | The benchmark file itself is in the crawl | Only caught if you can search the training corpus, which only the trainer can do |
| Near-duplicate leakage | Paraphrases, translations, reformatted mirrors, forum posts with the answers | Exact-match dedup passes them straight through |
| Format leakage | The model was trained on the benchmark's answer style rather than its items | Nothing is duplicated at all; the model learned the test's shape |
| Distillation leakage | Training on outputs of a model that was itself contaminated, or on synthetic data generated from benchmark items | The contaminated text never appears in your corpus |
| Selection leakage | Nothing leaked into training; you chose checkpoints, hyperparameters, or prompts by looking at the test set | It is overfitting through the experimenter, and it accumulates silently |

Selection leakage is the one senior candidates name and juniors miss. Running a
benchmark 200 times while sweeping a training recipe turns the test set into a
validation set. The fix is procedural: keep a sealed slice with a **query budget**,
record every time it is looked at, and treat a checkpoint chosen on a sealed-slice
comparison as having consumed one of those looks.

## Detection: what you can actually run

| Method | What it needs | What it tells you | Limits |
|---|---|---|---|
| N-gram or substring overlap between items and training corpus | Access to the training corpus | Direct evidence of verbatim and lightly-edited leakage | Only available to whoever trained the model; sensitive to n, casing, whitespace normalization |
| Embedding near-duplicate search | Training corpus plus an embedder | Catches paraphrase and translation | Threshold is a judgment call; expensive at web scale |
| Canary strings | The benchmark shipped a canary GUID | If the model can reproduce the canary, the file was in training | Only proves presence of the file, absence proves nothing |
| Membership inference on token probabilities (Min-K% Prob, Min-K%++) | Token log-probabilities from the model | Statistical evidence that a specific text was in pretraining | Needs log-probs; weak on heavily-trained or paraphrased text ([Detecting Pretraining Data from Large Language Models](https://arxiv.org/abs/2310.16789), [Min-K%++](https://arxiv.org/abs/2404.02936)) |
| Time-split comparison | Item release dates plus the model's training cutoff | The cleanest black-box signal: score on post-cutoff items versus pre-cutoff items of matched difficulty | Requires a benchmark that timestamps items, and difficulty must actually be matched |
| Functional twins | A regenerated benchmark built to the same spec with new items | A large drop on the twin is direct evidence of overfitting to the original | Expensive; the twin must be difficulty-matched, which is the hard part |
| Perturbation and permutation tests | Just API access | Sensitivity to option reordering or surface rewording suggests memorized surface form | Confounded with general brittleness, so it is suggestive, not conclusive |

The functional-twin result is the one worth memorizing as a concrete anchor:
rebuilding a grade-school math benchmark from scratch to the same specification and
re-running the same models revealed systematic gaps for some model families and none
for others, which is exactly the signature contamination produces
([A Careful Examination of Large Language Model Performance on Grade School
Arithmetic](https://arxiv.org/abs/2405.00332)). A survey of the move from static to
dynamic evaluation collects the rest of the toolkit
([Recent Advances in LLM Benchmarks against Data Contamination](https://arxiv.org/abs/2502.17521)).

```python
def min_k_percent(token_logprobs, k=0.2):     # k = fraction of least-likely tokens
    n = max(1, int(len(token_logprobs) * k))  # how many tokens to keep
    worst = sorted(token_logprobs)[:n]        # the least likely tokens in the text
    return sum(worst) / n                     # higher (closer to 0) => more likely memorized
# Compare the score for benchmark items against a held-out reference distribution of
# same-domain text the model provably could not have seen; a shifted distribution is
# the signal, a single number means nothing.
```

The comparison-against-a-reference-distribution point is the part people get wrong.
A Min-K% value is not interpretable on its own; it is only interpretable against the
distribution of the same statistic on text of the same genre that postdates the
training cutoff.

## Prevention: designs that make leakage bounded by construction

- **Time-gated live benchmarks.** Draw items from sources that postdate every
  model's cutoff and refresh continuously.
  [LiveBench](https://arxiv.org/abs/2406.19314) builds questions from recent
  competitions, papers, and news and rotates a fraction of the pool on a schedule;
  [LiveCodeBench](https://arxiv.org/abs/2403.07974) tags every problem with a
  release date so you can evaluate on a window strictly after a model's cutoff.
  The tradeoff is that the item pool changes underneath you, so cross-time
  comparisons need the window pinned.
- **Private internal sets.** Built from your own workload, never published, never
  posted to a third-party API without a data-retention review. This is the only set
  where you can *prove* the no-leak story rather than argue it.
- **Held-out by construction.** Procedurally generated items with a verifier
  (templated symbolic tasks, generated code with property tests) give unlimited
  fresh items, at the cost of a narrower construct.
- **Canary and licence hygiene.** Ship canary GUIDs and a no-train licence with any
  benchmark you publish, knowing both are norms rather than enforcement.
- **Report the dates.** Every benchmark result should carry the model's training
  cutoff and the item pool's date range. A reader can then judge contamination risk
  without trusting your decontamination claim.

## The other half: item validity

Even a leak-free benchmark lies if the items are broken.

- **Label errors** cap the achievable score and add noise where models are
  separated. Re-annotation of MMLU found a single-digit-percent error rate spread
  across parsing problems, multiple correct options, and unanswerable questions
  ([Are We Done with MMLU?](https://arxiv.org/abs/2406.04127)); adversarially
  filtered commonsense benchmarks show similar validity problems
  ([What the HellaSwag?](https://arxiv.org/abs/2504.07825)). When two candidates sit
  within the label-error band, the benchmark cannot rank them, full stop.
- **Weak outcome validation** is the agentic analogue: the environment declares
  success on evidence too thin to support it. The rigorous-agentic-benchmarks audit
  found task setups whose test suites accept incorrect solutions and success criteria
  that can count doing nothing as passing, and it packages the fixes as a checklist
  covering task specification, outcome validation, and reporting
  ([Establishing Best Practices for Building Rigorous Agentic Benchmarks](https://arxiv.org/abs/2507.02825)).
- **Format artifacts.** If items can be answered above chance without the question,
  the benchmark is partly measuring option-pattern discrimination
  ([Answer Matching Outperforms Multiple Choice](https://arxiv.org/abs/2507.02856)).
  The cheap audit: run the benchmark with the questions removed and report the
  no-question baseline. Anything meaningfully above chance is a red flag you should
  publish alongside the score.

## Diagnosing a suspicious result

| Symptom | First hypothesis | Test that discriminates |
|---|---|---|
| A model is far above its peers on one old benchmark only | Contamination on that benchmark | Score on post-cutoff items of the same construct, or on a functional twin |
| Score collapses when options are reordered or the question is reworded | Memorized surface form, or brittleness | Perturbation set plus a no-question baseline |
| Your fine-tune gained 5 points on the target benchmark and 0 elsewhere | Format overfitting or selection leakage | Free-form version of the same construct; check how many times the set was queried during development |
| Everyone scores above 90 and the ordering flips between runs | Saturation plus label noise | Compute the interval; retire the benchmark for ranking purposes |
| An agent passes tasks it visibly did not complete | Weak outcome validation | Manually audit 20 passing trajectories; that audit is not optional before publishing an agentic number |

The through-line for the whole section: **a suspiciously good number is
contamination, a broken item, or a protocol bug until proven otherwise.** Interviews
reward the candidate who says that first and then names which of the three they
would test.
