# 6. Statistics and leaderboards

Benchmarks are experiments, and until recently almost nobody analyzed them like
experiments. The reference argument for treating them properly is
[Adding Error Bars to Evals](https://arxiv.org/abs/2411.00640) (Anthropic, 2024),
which imports standard experimental analysis into eval reporting and points out that
a number of claimed advances sit inside the margin of error once the analysis is
done correctly.

## A score is an estimate, so it has a width

For an accuracy $\hat p$ over $n$ independent items, the standard error and the
usual interval are:

$$\text{SE}(\hat p) = \sqrt{\frac{\hat p (1 - \hat p)}{n}} \qquad \text{CI}_{95} = \hat p \pm 1.96\,\text{SE}$$

Put the item counts of real benchmarks into that and the implications are blunt.

| Benchmark size | Example | 95% half-width at $\hat p = 0.5$ | What it means |
|---|---|---|---|
| 30 items | Competition math sets | about 18 points | A single run cannot distinguish anything; one item is over 3 points |
| 198 items | GPQA Diamond | about 7 points | Sub-5-point gaps between models are not resolvable from one run |
| 500 items | SWE-bench Verified | about 4.4 points | The usual headline gaps are borderline |
| 2,000 items | Large aggregate suites | about 2.2 points | 2-point claims become discussable, not yet decided |

This table alone answers the most common follow-up in the interview: *"the new model
scores 3 points higher, is it better?"* On a 200-item benchmark, from one run, the
honest answer is "not distinguishable yet, and here is what it would take."

For small $n$, do not lean on the normal approximation. A position paper argues
explicitly against the CLT below a few hundred datapoints
([Don't Use the CLT in LLM Evals With Fewer Than a Few Hundred Datapoints](https://arxiv.org/abs/2503.01747));
use a Wilson interval, an exact binomial interval, or a bootstrap instead.

```python
def wilson(p_hat, n, z=1.96):                 # better than normal approx at small n / extreme p
    d = 1 + z * z / n
    center = (p_hat + z * z / (2 * n)) / d
    half = z * ((p_hat * (1 - p_hat) / n + z * z / (4 * n * n)) ** 0.5) / d
    return center - half, center + half
# wilson(0.5, 198) -> about (0.431, 0.569); wilson(0.9, 30) -> about (0.741, 0.965)
```

## Compare paired, always

The naive comparison treats two models' scores as two independent proportions and
adds their errors. That throws away the fact that both models answered *the same
items*, which is the single cheapest variance reduction available.

Compare per item. Let $b$ be the count of items model A got right and B got wrong,
and $c$ the reverse (the concordant items cancel). Then:

$$\hat\Delta = \frac{b - c}{n}, \qquad \text{SE}(\hat\Delta) \approx \frac{\sqrt{b + c}}{n}, \qquad z = \frac{b - c}{\sqrt{b + c}}$$

The last expression is McNemar's test. A worked case on a 500-item suite: A wins 25
items, B wins 15, so the headline gap is 2 points. Unpaired, each score carries about
a 4.4-point interval and the comparison looks hopeless. Paired,
$\text{SE} = \sqrt{40}/500 \approx 1.3$ points and $z = 10/\sqrt{40} \approx 1.6$:
still not significant, but now you know exactly how far off you are and that the
answer is more items, not more argument.

Sizing follows directly. To detect a difference $\delta$ at 5 percent significance
and 80 percent power with a discordance rate $d = (b+c)/n$:

$$n \gtrsim d \left(\frac{z_{0.975} + z_{0.80}}{\delta}\right)^{2} \approx d \cdot \frac{7.85}{\delta^{2}}$$

At $d = 0.1$ and $\delta = 0.02$, that is roughly 2,000 items. That number is the
real answer to "we need to call a 2-point difference": either find a benchmark with
thousands of items, or pool several benchmarks and accept that you are now measuring
a composite construct.

## The three other variance sources people forget

**Clustering.** When items are drawn in related groups (many questions on one
passage, many tasks in one repository, many turns in one conversation), they are not
independent and the plain SE is too small. Compute clustered standard errors at the
group level; the effective sample size is closer to the number of groups than the
number of items.

**Sampling variance.** With temperature above zero, each item's score is itself a
random variable. Averaging $m$ samples per item cuts that component by $m$, which is
often cheaper than adding items when the item pool is fixed. Report the mean over
samples, never the best sample: taking the max over runs is best-of-N selection
applied to yourself.

**Seed and run variance.** On small reasoning benchmarks this dominates everything
else. Re-running the same model with a different seed can swing double digits, and
comparisons built on single runs have produced modest real effects reported as large
ones ([A Sober Look at Progress in Language Model Reasoning](https://arxiv.org/abs/2504.07086)).
Report mean and standard deviation over a fixed number of seeds, and state the seed
count next to the score.

## Many models times many benchmarks is a multiple-comparison problem

A 12-model by 15-benchmark grid is 180 numbers and up to 990 pairwise comparisons per
benchmark. At 5 percent significance you will find "significant" differences that are
not there. Two defenses:

- **Control the false discovery rate** (Benjamini-Hochberg) across the family of
  comparisons you actually make, and say which family you controlled over.
- **Pre-register the comparison.** Declare before the run which benchmarks decide the
  question and which are descriptive. A comparison chosen after seeing the grid is a
  hypothesis generated by the data.

The same bias operates at leaderboard scale on the submission side.
[The Leaderboard Illusion](https://arxiv.org/abs/2504.20879) documents private
best-of-N testing before public release, where a provider evaluates many variants and
publishes only the best, which violates the assumptions of the ranking model and
inflates the published score; the arena operators dispute the magnitude
([LMArena's response](https://lmarena.ai/blog/our-response/)). Whatever the size of
the effect, the lesson for your own reporting is the same: **the number of variants
you tried before publishing one is part of the result.**

## Aggregating into an index without lying

Composite indices are more robust than any single benchmark, because gaming one eval
barely moves the aggregate. They are also easy to build badly.

- **Normalize before averaging.** Benchmarks have different floors (chance level is
  25 percent on a 4-option set, 0 percent on free-form) and different ceilings.
  Average raw percentages and you weight by scale, not importance.
- **State the weights.** An unweighted mean is a weighting, chosen by which
  benchmarks happened to be included.
- **Bootstrap the aggregate.** Resample items within each benchmark, recompute the
  index, and report the interval on the index and, more importantly, on the *rank*.
  Rank stability is what readers actually consume, and it is usually much weaker than
  the point estimates suggest.
- **Drop saturated components.** A benchmark where every candidate is above 90
  contributes noise and a false sense of coverage.

For preference-based ranking, the arena family fits a Bradley-Terry model over
pairwise votes, where the probability that model $i$ beats model $j$ is
$\sigma(\beta_i - \beta_j)$ with $\beta$ the fitted strengths. Two practical notes:
the fitted strengths come with confidence intervals, and overlapping intervals mean
a tie no matter what the sorted table looks like; and preference votes partly track
style, which is why arenas now publish style-controlled variants that regress out
response length and formatting. A model that wins on style control and loses without
it (or the reverse) is telling you something a single rank cannot.

## Report cost with quality or the comparison is not a comparison

Test-time compute makes quality a curve. A candidate that spends five times the
output tokens should not be compared against one that does not, at least not without
saying so. The reportable unit is a point on a frontier:

| Candidate | Score | 95% CI | Output tokens per item | Cost per 1k items | Latency p50 |
|---|---|---|---|---|---|
| Model A, low effort | 61.2 | +/- 3.4 | 480 | \$4.10 | 3.1 s |
| Model A, high effort | 68.9 | +/- 3.2 | 4,900 | \$38.60 | 27 s |
| Model B, default | 66.4 | +/- 3.3 | 1,100 | \$11.20 | 7.4 s |

Read as a table of scores, A-high wins. Read as a frontier, B is the better default
and A-high is a fallback for hard items, which is the decision the product actually
needs. METR's time-horizon methodology takes the same idea further by expressing
capability in units of human task length rather than percentage points, fitting the
task duration at which a model succeeds half the time
([Measuring AI Ability to Complete Long Software Tasks](https://arxiv.org/abs/2503.14499)).

## The decision rule

| Situation | Verdict to report |
|---|---|
| Paired CI on the difference excludes 0 on the pre-registered benchmarks | Better, with the interval |
| Paired CI includes 0 | Not distinguishable at this sample size; state the size needed |
| Wins on some benchmarks, loses on others, all inside intervals | Tie; decide on cost, latency, or the internal benchmark |
| Wins the aggregate but loses the internal benchmark | Prefer the internal benchmark and say why: it matches the construct and cannot be contaminated |
| Wins only after being given more test-time compute | Report as a cost-matched loss and an effort-matched win |

Being comfortable saying "not distinguishable" is the most senior-sounding thing in
this chapter, and it is almost always the statistically correct verdict for the small
public benchmarks people quote most often.
