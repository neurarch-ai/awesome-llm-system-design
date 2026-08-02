# 10. Putting it together: the complete build

The scenario from [section 1](01-clarifying-requirements.md): a mixed workload
(verifiable code and data tasks plus open-ended explanation), a p95 promise, a fleet
we operate, and a reasoning model that has made everything slower and more
expensive.

## The default stack

| Decision | Committed choice | Why, in one line |
|---|---|---|
| First work item | Per-request outcome logging (tokens, latency, verifier verdict, solved) | No policy can be compared without the solved-task denominator |
| Budget | Hard cap per request, set at the measured knee of the accuracy-versus-budget curve | A cap is a latency guarantee, and the knee is where more tokens stop buying answers |
| Boundary behavior | Forced answer at the cap, or explicit decline | Silent truncation returns malformed output and scores as a wrong answer |
| Queueing | Separate queues by budget class; length-class predictor for priority inside the thinking queue | Short requests must not wait behind long ones, and FIFO is optimal for nothing here |
| Utilization target | Well below saturation on the thinking fleet | Queueing delay carries $\rho/(1-\rho)$ and thinking has high service-time variance |
| Verifiable tasks | Cascade: cheap attempt, executor check, escalate on failure, escalation quota-capped | Beats always-thinking on both cost and solve rate; the quota prevents escalation storms |
| Unverifiable tasks | Fixed budget at the knee, forced answer, no sampling | Extra samples are unusable without a selector |
| Extraction and formatting | Non-thinking small model plus constrained decoding | Deliberation buys nothing and adds drift |
| Overload | Admission control that downgrades to the cheap path | A fast worse answer beats a slow correct one after the user left |
| Headline metrics | Cost per solved task, and p95 or p99, reported as a pair | Each alone is trivially gamed by the other |

## The three policies on the same traffic

The capstone below simulates 4,000 requests arriving at a fixed rate into a 24-slot
fleet, with output length drawn from a lognormal per path, and scores three
policies. The arrivals are identical across policies, so the comparison is paired.

| Policy | p50 | p99 | mean tokens | \$ per 1k requests | solved | \$ per 1k solved |
|---|---|---|---|---|---|---|
| A: always think | 55.7 s | 345.2 s | 4,412 | \$26.47 | 78.6% | \$33.68 |
| B: effort routing (hard 20% only) | 6.7 s | 186.5 s | 1,157 | \$6.94 | 58.7% | \$11.82 |
| C: cascade with a verifier | 10.4 s | 250.3 s | 2,120 | \$12.72 | 85.3% (highest) | \$14.92 |

Four readings, and the third is the one that wins interviews.

**A is the worst product on every axis except one.** It solves 78.6 percent, and it
is the slowest and most expensive way to get there. It is also the default that
teams ship when they "switch to a reasoning model."

**B is the cheapest per solved task and the worst on quality.** This is the trap in
the "cost per request went down" answer: routing hard requests to a cheap path
reduces cost by failing more, and only the solve rate exposes it. Whether the
19 percentage points of quality are worth \$3 per thousand solved is a product
question, not a systems one, but you cannot even ask it without outcome logging.

**C solves more than always-thinking, at half the cost.** The cascade gets two
attempts on the requests the cheap path missed, so its solve rate (85.3 percent)
exceeds the always-think baseline (78.6 percent), while cost per solved task falls
from \$33.68 to \$14.92. This is the single most useful result in the chapter:
**escalation is not just a cost optimization, it is a quality mechanism**, provided
the verifier is trustworthy.

**The tail never fully goes away.** Even the cheapest policy has a p99 nearly 30
times its p50, because the thinking path is long-tailed and shares a fleet. That is
what the budget cap, the queue partition, and the utilization target are for, and it
is why p99 travels next to the cost number rather than underneath it.

## The same system under three constraint sets

**Strict interactive SLO (p95 under a few seconds).** The thinking path cannot be
on the critical path at all. Serve the cheap path synchronously, run the verifier
inline, and make escalation asynchronous: return the cheap answer with a "checking"
state and update it when the escalation lands, or notify. The engineering moves from
scheduling to product surface, which is usually the honest answer when the promised
p95 is below the mean generation time.

**Cost-dominated batch workload (no user waiting).** Latency is nearly free, so
spend it: large budgets, parallel sampling with an executor selecting, and
aggressive batching to keep slots full. Here it is worth pushing utilization high,
because the queueing tail is not a product problem, and worth investing in the
verifier rather than the model, since delivered quality is capped by selection.

**No verifier anywhere (open-ended generation).** Cascades and best-of-n are both
unavailable, so the levers shrink to: a measured fixed budget with a forced-answer
boundary, effort routing by a trained classifier rather than by an accept test, and
a certified rubric judge if you are willing to build one (which converts this case
back into the previous one). The honest framing in an interview is that the missing
verifier, not the missing GPUs, is what caps this system.

## The smallest runnable experiment

```python
"""Test-time compute planner on one page. Python 3, standard library only.
Illustrative numbers, not a benchmark."""

import heapq
import random
from statistics import mean

SEED = 5
N_REQ = 4000            # requests in the run
SERVERS = 24            # concurrent decode slots
RATE = 0.25             # arrivals per second (offered load, identical across policies)
TOK_S = 60.0            # decoded tokens per second per slot
PRICE = 6.0 / 1e6       # dollars per output token

SHORT = dict(mu=350, sigma=0.45, solve=0.55)
LONG = dict(mu=3200, sigma=0.80, solve=0.78)
VERIFY_TOKENS = 120     # a cheap checker run on the short answer
VERIFY_RECALL = 0.85    # fraction of wrong short answers the verifier catches


def draw_tokens(rng, path):
    """Output length is a random variable, and its tail is the whole problem."""
    return max(20.0, rng.lognormvariate(0, path['sigma']) * path['mu'])


def simulate(rng, per_request):
    """FIFO queue, SERVERS slots. per_request(rng) -> (tokens, solved)."""
    free = [0.0] * SERVERS
    heapq.heapify(free)
    t, lat, toks, solved = 0.0, [], [], 0
    for _ in range(N_REQ):
        t += rng.expovariate(RATE)
        n_tok, ok = per_request(rng)
        start = max(t, heapq.heappop(free))          # wait for the first free slot
        service = n_tok / TOK_S
        heapq.heappush(free, start + service)
        lat.append(start - t + service)              # queueing delay plus service
        toks.append(n_tok)
        solved += ok
    return lat, toks, solved


def pct(xs, q):
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def always_long(rng):
    return draw_tokens(rng, LONG), rng.random() < LONG['solve']


def routed(rng):
    path = LONG if rng.random() < 0.20 else SHORT    # a classifier picks the hard fifth
    return draw_tokens(rng, path), rng.random() < path['solve']


def cascade(rng):
    n = draw_tokens(rng, SHORT) + VERIFY_TOKENS      # cheap attempt plus the checker
    ok = rng.random() < SHORT['solve']
    if not ok and rng.random() < VERIFY_RECALL:      # verifier caught it -> escalate
        n += draw_tokens(rng, LONG)
        ok = rng.random() < LONG['solve']            # a second, independent attempt
    return n, ok


print(f"{'policy':>16} {'p50 s':>7} {'p99 s':>7} {'mean tok':>9} "
      f"{'$/1k req':>9} {'solved':>7} {'$/1k solved':>12}")
for name, fn in [("A always think", always_long), ("B effort routing", routed),
                 ("C cascade", cascade)]:
    rng = random.Random(SEED)                        # same arrivals for every policy
    lat, toks, solved = simulate(rng, fn)
    cost_1k = mean(toks) * PRICE * 1000
    rate = solved / N_REQ
    print(f"{name:>16} {pct(lat, 0.50):7.1f} {pct(lat, 0.99):7.1f} {mean(toks):9.0f} "
          f"{cost_1k:9.2f} {rate:7.1%} {cost_1k / rate:12.2f}")

rng = random.Random(SEED)
long_s = [draw_tokens(rng, LONG) / TOK_S for _ in range(20000)]
short_s = [draw_tokens(rng, SHORT) / TOK_S for _ in range(20000)]


def cv2(xs):
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs) / (m * m)


print()
print(f"short path : mean service {mean(short_s):5.1f}s  CV^2 {cv2(short_s):.2f}")
print(f"long path  : mean service {mean(long_s):5.1f}s  CV^2 {cv2(long_s):.2f}")
print("queue wait grows with E[S^2] (Pollaczek-Khinchine), so a heavier-tailed")
print("thinking distribution inflates p99 faster than it inflates the mean.")
```

Output:

```text
          policy   p50 s   p99 s  mean tok  $/1k req  solved  $/1k solved
  A always think    55.7   345.2      4412     26.47   78.6%        33.68
B effort routing     6.7   186.5      1157      6.94   58.7%        11.82
       C cascade    10.4   250.3      2120     12.72   85.3%        14.92

short path : mean service   6.4s  CV^2 0.23
long path  : mean service  73.5s  CV^2 0.89
queue wait grows with E[S^2] (Pollaczek-Khinchine), so a heavier-tailed
thinking distribution inflates p99 faster than it inflates the mean.
```

The last two lines are the chapter in miniature. The long path does not just take
11 times longer on average, it is four times more variable in relative terms, and
queueing delay grows with that variance. Every control in
[section 3](03-budgets-and-latency.md) exists to attack one of those two numbers,
and every policy in [section 4](04-allocation-and-routing.md) exists to keep most
requests out of the distribution that has them.
