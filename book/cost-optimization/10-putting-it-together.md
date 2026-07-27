# 10. Putting it together: the complete build

Sections 1 through 6 taught each lever with its math and failure modes; section
7 showed which lever real teams pulled first. What none of them show is a single
system with every decision made. This capstone does three things: it gives you
an opinionated default stack so option paralysis never blocks a first build, it
walks the chapter's scenario end to end with every choice committed and costed,
and it shows how the same decisions flip when the constraints change. It closes
with the smallest runnable router-cascade, one file, no installs.

## The default stack: start here, deviate with reason

Every lever in this chapter has two to four credible variants, and a first-time
builder can burn a week comparing routers before saving a single dollar. Skip
that. The stack below is a sane default for a first production build; each row
names when to deviate and which section explains why. Tools change yearly, but
the interface of each lever (measure, gate, cache, trim, route, right-size)
does not, so pick per lever by interface and treat any specific library as
replaceable.

| Lever | Default | Deviate when | Why (section) |
|---|---|---|---|
| Quality measurement | Labeled eval set with per-bucket scores, built before tuning anything | Never. Every threshold below needs it | [1](01-clarifying-requirements.md) |
| Gateway | One proxy (LiteLLM class) in front of every provider: budgets, logging, fallback | Never for the proxy itself; single-team toy projects may defer budgets | [6](06-serving-and-scaling.md) |
| Semantic cache | Embed + threshold, tau tuned on labeled should-hit / should-not pairs, scoped per tenant | Traffic is nearly all unique free text: measure organic hit rate first, may not clear break-even | [4](04-caching-and-compression.md) |
| Prefix cache | Provider prompt caching with stable content first, volatile content last | Prompts share no long fixed header | [4](04-caching-and-compression.md) |
| Context trimming | Rerank retrieved chunks, keep top-3 | Retrieval is already tight, or the task needs every chunk | [4](04-caching-and-compression.md) |
| Prompt compression | Skip at launch; add LLMLingua only if input still dominates after trimming | Long verbose text survives trimming and input tokens still dominate the bill | [4](04-caching-and-compression.md) |
| Router | Heuristic regex layer for stable patterns, then a small fine-tuned classifier | You have preference data: a RouteLLM-style preference router generalizes across model pairs | [3](03-routing-and-cascades.md) |
| Cascade | Off the hot path under a tight SLO; use where latency slack and a verifiable check exist | Task is verifiable (code, SQL, citations) and the SLO allows two calls | [3](03-routing-and-cascades.md) |
| Right-sizing | Dedicated embedding model, small cross-encoder reranker, classifier at 1B or less | A subtask is high-QPS and stable: consider distillation | [5](05-right-sizing.md) |
| Offline traffic | Provider batch API for anything with no user waiting | Self-host only above the QPS break-even Q* | [5](05-right-sizing.md), [6](06-serving-and-scaling.md) |

The first row is the one beginners skip and regret: without per-bucket quality
numbers, every threshold in this stack is a vibe, and a green cost dashboard
can hide a hard-tail regression indefinitely. The chapter's scenario already
has the eval set; if yours does not, building it is the first deliverable.

## The complete build

Return to the scenario from [section 1](01-clarifying-requirements.md): an
interactive mixed-intent RAG chat product, responses under two seconds, an
LLM-as-a-judge eval set of about 500 labeled examples, a bill dominated by
input tokens because every prompt carries 20 retrieved chunks, easy queries
dominating by count with a hard revenue-driving tail that cannot regress, and
API pricing only. Here is the whole system with every choice committed and the
reason it won.

| Decision | Choice | Why it won |
|---|---|---|
| Measurement | Split the 500-example judge set into routing buckets, hard tail oversampled | The hard tail cannot regress; only per-bucket scores can see it |
| Gateway | Single proxy for all calls: per-team budgets, fallback, cost logging | Without it spend is invisible until the invoice and every lever is advisory |
| Context trimming | Cross-encoder reranks the 20 retrieved chunks, keeps top-3 | Input tokens are the stated cost driver; the bottom 17 chunks are noise |
| Compression | LLMLingua skipped at launch | Trimming captured the input win; the small-LM pass would be overhead on the now-short prompts |
| Semantic cache | Embed + threshold, tau tuned on labeled pairs, per-tenant key, TTL on volatile answers | Mixed-intent chat repeats FAQs in paraphrase; break-even hit rate is only about 2% |
| Prefix cache | System prompt and tool schemas first, per-request content last | Stable header is shared by every request; ordering keeps the hit rate alive |
| Router | Regex layer for greetings and template lookups, then a fine-tuned small classifier | Easy classes are stable and pattern-shaped; a judge pipeline can label the rest |
| Cascade | Not on the hot path | The two-second SLO has no slack for cheap-call-then-scorer; revisit offline |
| Model tiers | Fine-tuned small model for the easy bucket, frontier model for the hard tail | The 2-point tolerance applies only to easy traffic; the tail stays on the frontier model |
| Right-sizing | Dedicated embedding model for the cache, small cross-encoder for trimming, classifier at 1B or less | A frontier call for any of these would eat its own savings |
| Quantization / batching | Off the table | API pricing only; these levers exist on the provider's side, not ours |

**Baseline spend.** The prompt today is roughly a 300-token system prompt plus
20 chunks x 400 tokens plus query and history, near 8,500 input tokens, with a
~250-token answer. At illustrative frontier prices ($2.50 and $10.00 per
million input and output tokens) that is about $0.021 + $0.0025, call it
$0.024 per query. At an illustrative 150,000 queries per day the bill is about
$3,600 per day, roughly $107,000 per month. Input tokens are 89% of it, which
confirms the profiling in [section 2](02-frame-the-system.md): trim before
routing.

**Trimming.** Keeping the top-3 of 20 chunks cuts the prompt to about 1,700
input tokens (system prompt, 3 x 400-token chunks, query and history), an 80%
input reduction in line with [section 4](04-caching-and-compression.md)'s
17-of-20 arithmetic. Frontier cost per query falls from $0.024 to about
$0.0068, a 3.5x cut, before any routing, and the reranker was already scoring
chunks so the surviving text is untouched. This is why compression was skipped:
the waste was between chunks, not inside them.

**Caching.** With the cache break-even near 2% ([section 4](04-caching-and-compression.md)),
an illustrative 15% semantic hit rate on FAQ-heavy chat traffic is comfortably
net-positive. Expected cost becomes 0.85 x (miss cost) plus embedding noise.

**Routing.** Illustrative: the regex layer plus classifier sends 60% of cache
misses to the small model. At illustrative small-model prices ($0.25 and $1.25
per million), a small-model query costs about $0.0007 against the frontier's
$0.0068. The average miss then costs 0.6 x $0.0007 + 0.4 x $0.0068, about
$0.0031, and [section 2](02-frame-the-system.md)'s expected-cost formula gives

$$\mathbb{E}[C] \approx 0.15 \cdot c_{\text{hit}} + 0.85 \cdot \$0.0031 \approx \$0.0027$$

**Final cost per query.** About $0.0027 against a $0.024 baseline: a 9x cut,
roughly 89%, or about $400 per day against $3,600. Illustrative throughout,
but consistent with the headline range the production systems in
[section 7](07-how-teams-do-it-in-production.md) report (Anyscale 70%,
RouteLLM about 85%). The order mattered: trimming shrank the token bill for
both tiers, caching removed calls entirely, and routing then split what was
left, exactly the left-to-right lever order from
[section 2](02-frame-the-system.md).

**What breaks in month one.** Three failure modes dominate early operations, so
wire their signals before launch: per-bucket quality on the hard-tail slice (a
router drifting on new traffic dumps newly-hard queries on the small model
while the aggregate dashboard stays green, the exact trap in
[section 8](08-interview-qa.md)), cache-hit quality at the chosen tau (a rising
rate of wrong-neighbor answers means the threshold is too loose, and raw hit
rate will not show it), and gateway fallback rate (sustained fallback means the
primary provider is in trouble and traffic may be landing on a model with
different quality and safety behavior).

## The same techniques under different constraints

The review question that matters in practice is not "which router is best" but
"which router is best under my constraints." Here is the same stack built three
times. Only the middle column is the build above; the other two keep the
identical lever interfaces and swap nearly every implementation choice.

| | Low-traffic internal assistant | Mixed-intent RAG chat (this chapter) | Nightly bulk classification |
|---|---|---|---|
| Traffic / cost driver | ~1k queries/day; total spend small in absolute terms | 150k queries/day (illustrative); input tokens dominate | Millions of items nightly; request count dominates, nobody waiting |
| Latency budget | Seconds are fine | Under two seconds | None; throughput and cost only |
| Cache | Exact cache only, if any; paraphrase volume too low to tune tau | Semantic + prefix cache in series, tau on labeled pairs | Exact cache on repeated inputs; dedupe the batch before calling at all |
| Trimming / compression | Trim retrieval defaults, nothing more | Rerank to top-3; compression held in reserve | Aggressive: fixed short template per item, no free-text padding |
| Routing | None: one mid-tier model for everything | Regex + classifier router, easy bucket to small model | Cascade with a verifiable or scored check; latency slack makes it free |
| Model tier | Single mid-tier API model | Fine-tuned small model + frontier tail | Distilled student for the stable task; frontier only as teacher and auditor |
| Batch / self-host | No | No; API only per the scenario | Batch API at about half price; self-host above the QPS break-even Q* |
| Eval | Small golden set, re-run on model swaps | Judge set split per routing bucket, hard tail oversampled | Sampled judge over each night's output; student-vs-teacher drift check |
| What would be over-engineering | Classifier router, semantic cache, distillation: each costs more to maintain than it saves | Cascade on the hot path, LLMLingua at launch | Low-latency serving stack, streaming, interactive routing logic |

Two lessons fall out. First, the left column is mostly deletions: at 1k queries
per day every lever's maintenance cost exceeds its savings, and the correct
cost optimization is a single sensible model choice plus the gateway's logging
so you notice when that stops being true. The formulas still earn their keep,
but as reasons not to build: the router-savings equation in
[section 3](03-routing-and-cascades.md) goes negative once router upkeep
outweighs the captured gap. Second, the right column shows latency and cost
trading places as the binding constraint: with no SLO, the cascade
([section 3](03-routing-and-cascades.md)) becomes the default rather than the
exception, distillation ([section 5](05-right-sizing.md)) pays back fast at
volume, and the batch API's half price is the mechanism of batching showing up
in the bill.

## What each constraint decides

The compressed decision guide. Read the left column off your requirements; the
right columns say which lever it moves before you compare any tools.

| Your constraint | Lever it moves | Rule of thumb |
|---|---|---|
| Dominant cost driver | Which lever fires first | Input-heavy: trim, then compress. Output-heavy: smaller model, shorter answers. Request count: cache and route |
| Latency SLO | Router vs cascade | Tight SLO: blind router, sub-ms classifier. Slack available: cascade that scores a real answer |
| Query repetition | Caching layers | Break-even hit rate is near 2%; paraphrase-rich traffic justifies semantic, unique free text may not clear it |
| Shared prompt structure | Prefix cache | Long fixed header on every request: stable content first, volatile last, or the hit rate silently dies |
| Hard-tail tolerance | Routing objective | Zero tolerance: quality floor per bucket as a hard constraint, hard tail oversampled in the eval set |
| Quality measurement maturity | Everything | No eval set means no defensible threshold anywhere; build the eval set before the first lever |
| API vs self-host | Quantization, continuous batching | Below break-even QPS Q* the API wins and quantization is not your lever; above it, FP8 class gains apply |
| Offline share of traffic | Batch API | Anything with no user waiting leaves the sync endpoint; about half price is the going rate |
| Ops budget for more models | Right-sizing depth | Each extra model is a quality surface that can silently regress; right-size only as far as you can monitor |

## The smallest runnable router-cascade

The review of every routing framework is the same: the reader assembles a
gateway, a router, and two providers and still cannot see the tradeoff. So here
is the chapter's core mechanism in one file with zero installs. Every
production component is swapped for the smallest thing with the same interface:
the two model tiers become biased coin flips with a 10x cost gap, the
reliability scorer becomes a noisy confidence draw, and the traffic becomes a
seeded 70/30 easy/hard stream. Sweeping the escalation threshold tau prints
the cost-quality operating points of [section 3](03-routing-and-cascades.md)'s
cascade against the always-cheap and always-strong baselines.

```python
"""Router/cascade cost-vs-quality simulator, runnable with no installs."""
import random

random.seed(7)

C_CHEAP, C_STRONG = 1.0, 10.0          # relative per-call cost of each tier
N = 20000                              # queries in the simulated stream

def make_query():
    """70% easy, 30% hard; production: real mixed-intent traffic."""
    return "easy" if random.random() < 0.70 else "hard"

def cheap_model(kind):
    """Cheap tier: strong on easy, weak on hard. Returns (correct, confidence).
    Confidence overlaps across correct/wrong: a deliberately imperfect scorer,
    standing in for logprobs or a trained reliability model."""
    correct = random.random() < (0.95 if kind == "easy" else 0.40)
    conf = random.uniform(0.5, 1.0) if correct else random.uniform(0.0, 0.7)
    return correct, conf

def strong_model(kind):
    """Frontier tier: expensive, near-uniformly good."""
    return random.random() < (0.97 if kind == "easy" else 0.92)

def run(policy, tau=None):
    """policy: 'cheap' | 'strong' | 'cascade'. Returns (avg cost, accuracy, escalation rate)."""
    cost = right = escalated = 0
    for _ in range(N):
        kind = make_query()
        if policy == "cheap":
            ok, _ = cheap_model(kind)
            cost += C_CHEAP
        elif policy == "strong":
            ok = strong_model(kind)
            cost += C_STRONG
        else:                                   # cascade: cheap first, escalate on low confidence
            ok, conf = cheap_model(kind)
            cost += C_CHEAP
            if conf < tau:                      # scorer says "not confident": pay for the strong call
                ok = strong_model(kind)
                cost += C_STRONG
                escalated += 1
        right += ok
    return cost / N, right / N, escalated / N

print(f"{'policy':>22} {'avg cost':>9} {'accuracy':>9} {'escalate':>9}")
c, a, _ = run("cheap")
print(f"{'always-cheap':>22} {c:9.2f} {a:9.3f} {'-':>9}")
c_strong, a_strong, _ = run("strong")
print(f"{'always-strong':>22} {c_strong:9.2f} {a_strong:9.3f} {'-':>9}")
for tau in (0.2, 0.4, 0.5, 0.6, 0.8):
    c, a, e = run("cascade", tau)
    marker = "  <- knee" if abs(tau - 0.6) < 1e-9 else ""
    print(f"{f'cascade tau={tau:.1f}':>22} {c:9.2f} {a:9.3f} {e:9.2f}{marker}")

print("\nthe knee: near always-strong accuracy at a fraction of its cost;")
print("sweeping tau traces the whole cost-quality frontier from cheap to strong.")
```

Run it and the sweep prints the chapter's central claim as a table.
Always-cheap lands at cost 1.00 and accuracy 0.787 (it fails the hard tail);
always-strong at cost 10.00 and accuracy 0.956. The cascade at tau = 0.6
reaches 0.952, within half a point of always-strong, at cost 4.42, less than
half the price, escalating 34% of traffic. And at tau = 0.8 the cascade hits
0.967 at cost 7.85, strictly dominating always-strong on both axes, because
the scorer adds information a blind policy lacks: confident-correct cheap
answers are kept and the strong model's budget concentrates on the queries
that need it. Each toy piece stands in for a production component: the cost
constants are per-call API prices, `make_query` is your real mixed-intent
traffic, the confidence draw with its deliberate correct/wrong overlap is the
miscalibrated signal ladder of [section 3](03-routing-and-cascades.md)
(logprobs at the bottom, a trained reliability scorer above, a verifiable
check at the top), the tau sweep is the held-out calibration that must be
re-run as traffic drifts, and the escalation-rate column is the monitoring
metric [section 6](06-serving-and-scaling.md) tells you to alert on. Swap the
coin flips for two real models, the confidence draw for a trained scorer, and
put the whole loop behind a gateway, and you have rebuilt this chapter.
