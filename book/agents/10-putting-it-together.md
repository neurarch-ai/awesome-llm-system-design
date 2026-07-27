# 10. Putting it together: the complete build

Sections 1 through 6 taught each layer of the loop with its options and
tradeoffs; section 7 showed where real teams diverge. What none of them show is
a single system with every decision made. This capstone does three things: it
gives you an opinionated default stack so option paralysis never blocks a first
build, it walks the chapter's scenario end to end with every choice committed
and costed, and it shows how the same decisions flip when the constraints
change. It closes with the smallest runnable agent loop, one file, no installs.

## The default stack: start here, deviate with reason

Every layer in this chapter has two to five credible options, and a first-time
builder can burn a week comparing frameworks before executing a single tool
call. Skip that. The stack below is a sane default for a first production
agent; each row names when to deviate and which section explains why.
Frameworks change yearly, but the interface of each layer (plan, propose, gate,
execute, observe, bound, evaluate) does not, so pick per layer by interface and
treat any specific library as replaceable.

| Layer | Default | Deviate when | Why (section) |
|---|---|---|---|
| Topology | Single well-tooled agent | Subtasks are genuinely separable, each needs its own context, and wall-clock latency is the bottleneck: orchestrator plus subagents at roughly 15x tokens | [3](03-planning-and-tools.md), [7](07-how-teams-do-it-in-production.md) |
| Planning | Plan-then-execute; re-plan on contradiction | The path is unknowable before the first tool call: reactive (ReAct) | [3](03-planning-and-tools.md) |
| Tools | Narrow, typed schemas with enums and poka-yoke arguments | Tool-heavy multi-step work where JSON round-trips dominate: code execution in a sandbox | [3](03-planning-and-tools.md), [7](07-how-teams-do-it-in-production.md) |
| Safety | Deterministic pre-call gate in code: schema, policy, authorization | Never. The gate is non-negotiable for any write tool | [2](02-frame-the-system.md), [5](05-reliability-and-cost.md) |
| Memory | Short-term transcript plus retrieval (RAG) for policy and history | Transcript nears the limit or per-step cost climbs: compress; token-heavy artifacts: isolate | [4](04-memory-and-state.md) |
| Limits | Hard step cap and per-task token budget, enforced by the orchestrator | Never remove them; only tune $N$ and the budget per task class | [5](05-reliability-and-cost.md) |
| Models | Tiering: cheap model for routing steps, expensive only for reasoning | Every step is genuine policy reasoning (rare): single strong model | [5](05-reliability-and-cost.md) |
| Serving | Complexity router: sync path for simple tasks, durable queue for the rest | All tasks are long-running and non-interactive: async only | [6](06-serving-and-scaling.md) |
| Evaluation | Labeled ticket set for end-to-end task success offline; escalation, re-contact, and reversal rates online | Never. Build the labeled set before tuning anything | [8](08-interview-qa.md) |

The last row is the one beginners skip and regret: without a labeled ticket
set, every prompt and tool-schema change is a vibe, and per-step correctness
can look fine while end-to-end resolution is wrong. One afternoon of labeling
pays for itself the first time you swap a component.

## The complete build

Return to the scenario from [section 1](01-clarifying-requirements.md): a
customer-support agent handling 50,000 tickets per day, free read tools, write
tools gated behind schema and policy checks, refunds above \$50 routed to a
human approval queue, simple tickets resolved within 10 seconds, at most \$0.10
per ticket, every step logged. Here is the whole system with every choice
committed and the reason it won.

| Decision | Choice | Why it won |
|---|---|---|
| Topology | Single agent, one context per ticket | One ticket rarely splits into separable concurrent work streams; fan-out's roughly 15x token multiple buys nothing here |
| Planning | Plan-then-execute: lookup, order check, eligibility, act, reply | The ticket shape is known in advance; bounded plan length makes cost predictable where ReAct can wander |
| Tool schemas | Narrow, typed, enums, amounts in cents | The gate can validate calls deterministically instead of guessing over arbitrary JSON |
| Pre-call gate | Code-side schema, policy, and authorization check on every write | Policy in code is a guarantee; policy in a prompt is a suggestion prompt injection can override |
| Approval routing | Refunds above \$50 go to a human queue, never auto-execute | The stated unacceptable failure is an unauthorized refund; the gate makes it structurally impossible |
| Memory | Transcript short-term; policy and history via retrieval per step; verbose payloads to a scratchpad | Prefill re-reads the whole context every step, so everything kept in it is paid for repeatedly |
| Limits | Step cap $N = 10$, token budget 10,000 per ticket, in the orchestrator | The economics require bounded cost; a prompt-side "stop after 10" is unenforced |
| Model tiering | Cheap model for dispatch and templated replies, expensive model for policy reasoning | Most loop steps are routing, not reasoning; tiering keeps the ceiling honest |
| Serving | Complexity router; sync path under 10 s, writes and escalations on a durable queue | A slow ticket must never hold the fast path hostage; the queue is the seam that scales each path independently |
| Retries | Exponential backoff with a fixed cap in the executor, idempotency keys on writes | The model handles retries inconsistently; a retried refund must be a no-op |
| Audit and eval | Append-only log of reasoning, proposal, gate verdict, result; labeled set offline, escalation and reversal rates online | Auditability is a stated requirement, and the log doubles as the eval and debugging substrate |

**Step and token budget.** The planned shape is about 5 steps: lookup (1),
order check (1), eligibility (1-2), action (1), reply (1), so a cap of
$N = 10$ leaves room for retries ([section 5](05-reliability-and-cost.md)).
The token budget falls out of the cost ceiling: at \$0.10 per ticket and a
blended rate of roughly \$10 per million tokens, the budget is about 10,000
tokens per ticket. A ticket that would exceed either bound escalates before it
completes; escalation is a designed outcome, not a failure.

**Cost per ticket.** Per step, cost is prefill over the whole transcript plus
generation: $C_n = p \cdot T_{n-1} + g \cdot o_n$
([section 5](05-reliability-and-cost.md)). A simple ticket starts near 400
tokens of system prompt plus ticket and grows by 100-180 tokens per step, so
the five prefills read roughly 400 + 550 + 730 + 820 + 890, about 3,400 tokens
in total. At illustrative rates (\$1 and \$3 per million input and output
tokens for the cheap tier) that is well under a cent, and the toy program below
reproduces the number: about \$0.004. The \$0.10 ceiling is therefore not for
the median ticket; it is headroom for the tail, where retries, the expensive
reasoning model, and long transcripts live. The fleet-level check: 50,000
tickets at the 10,000-token budget is at most 500 million tokens per day
([section 6](06-serving-and-scaling.md)), which is the number to take to the
model provider before launch, not after.

**Latency.** The 10-second sync budget divides across about 5 sequential model
calls plus their tool round-trips. At an illustrative 1-1.5 s per model call
on the cheap tier and 100-300 ms per tool call, a serial loop lands near 7-9 s,
inside budget but without slack. The recovered headroom comes from
parallelizing the independent frontier: account lookup and order status do not
depend on each other, so issuing them together saves roughly one full
round-trip ([section 6](06-serving-and-scaling.md)). Anything involving a
write or a likely escalation is routed async up front, so the fast path never
carries the slow tail.

**Concurrency.** 50,000 tickets per day is about 35 per minute, roughly 0.6
per second. With up to $N = 10$ steps in flight per ticket, the model tier
must sustain about $0.6 \times N$ concurrent requests plus bursts, and every
tool back-end (CRM, OMS, policy search) must be provisioned for the agent's
call rate, not just direct user traffic
([section 6](06-serving-and-scaling.md)).

**What breaks in month one.** Three failure modes dominate early operations,
so wire their signals before launch: step-cap escalation rate (moderate
tickets exhausting $N = 10$ before resolution means the plan or the tool
schemas are making the model work too hard), refund reversal rate (reversals
that should not have happened are the gate's policy table missing an edge
case, and they are the one metric compliance will ask for first), and
correlated retry storms on tool back-ends (one downstream timeout makes every
in-flight ticket retry at nearly the same instant, so alert on per-back-end
concurrency, not per-ticket rates,
[section 6](06-serving-and-scaling.md)).

## The same techniques under different constraints

The review question that matters in practice is not "which planning style is
best" but "which planning style is best under my constraints." Here is the
same loop built three times. Only the support column is the build above; the
other two keep the identical layer interfaces and swap nearly every
implementation choice.

| | On-call Q&A copilot | Support agent (this chapter) | Background coding agent |
|---|---|---|---|
| Task shape | One-shot question over existing docs; no writes | 50k tickets/day, reads free, writes gated, \$0.10/ticket | Long-horizon repo tasks; hours are fine, correctness is everything |
| Latency budget | Seconds; interactive | Under 10 s sync for simple tickets; async for the rest | None; fully async, results land as a PR |
| Topology and planning | Not a true loop: retrieve then generate, effectively one step | Single agent, plan-then-execute over a known 5-step shape | Single agent, reactive act-then-verify loop with self-testing |
| Tool interface | Retrieval over a vector store; read-only | Narrow typed JSON tools behind the gate | Code execution in an isolated per-session sandbox VM |
| Safety | Grounding and citation checks; nothing irreversible to gate | Pre-call code gate plus human approval queue above \$50 | Isolation is the gate: the VM can only touch its own copy; tests and review gate the merge |
| Context strategy | Select: retrieve relevant chunks per question | Retrieve policy per step, compress near the limit, scratchpad for payloads | Isolate: each session owns a VM and its state; results leave as diffs, not transcript |
| Limits | Per-question token cap; no step cap needed | Step cap 10, token budget 10,000, enforced in the orchestrator | Generous step cap and wall-clock timeout; cost per task high but tasks are few |
| Eval | Answer helpfulness sampling; deflection rate | Labeled ticket set; escalation, re-contact, reversal rates | The test suite is the eval: verifiable pass/fail per task |
| What would be over-engineering | An agent loop at all; multi-agent anything | Multi-agent fan-out, code-execution tools, per-ticket VMs | Sub-10-second latency work, streaming, model tiering for cost |

Two lessons fall out. First, the copilot column is mostly deletions: when no
tool writes state, the gate, the approval queue, and the step cap all
disappear, and [section 2](02-frame-the-system.md)'s workflow-vs-agent test
says a fixed retrieve-then-generate pipeline beats a loop whenever the task
shape never varies. Second, the coding-agent column shows the safety budget
and the latency budget trading places: with no user waiting, verification gets
deeper (run the whole test suite, not a sub-second policy check) and isolation
replaces per-call gating, because an agent that can only damage its own
sandbox needs fewer per-step controls
([section 7](07-how-teams-do-it-in-production.md)).

## What each constraint decides

The compressed decision guide. Read the left column off your requirements; the
right columns say which lever it moves before you compare any frameworks.

| Your constraint | Lever it moves | Rule of thumb |
|---|---|---|
| Failure cost of a write | Gate strictness and approval threshold | Money or irreversible state: deterministic code gate always; above a risk threshold, route to a human queue |
| Cost ceiling per task | Step cap, token budget, model tiering | Budget divided by blended token price gives the token budget; cap steps at the planned shape plus retry headroom |
| Latency budget | Sync/async split and the parallel frontier | Under ~10 s: router up front, parallelize independent tool calls; no budget: batch async and spend the savings on verification |
| Task horizon (steps) | Context strategy | Under ~10 steps: append and prefix-cache; beyond: compress at a token threshold; token-heavy artifacts: isolate in a sub-context |
| Verifiable success signal | Retry architecture | Tests or a ledger to check: Reflexion-style retries pay; no signal: retries multiply cost with no quality gain |
| Separable subtasks plus latency pressure | Topology | Both true: orchestrator with parallel subagents at roughly 15x tokens; either false: stay single-threaded |
| Known task shape | Planning style | Same steps every time: a fixed workflow, not an agent; known shape with variation: plan-then-execute; unknowable path: ReAct |
| Untrusted input, private data, and an egress channel together | Tool scoping | The lethal trifecta: remove one leg in architecture; a loop that read untrusted content must not also hold an outbound channel |
| Growing tool count | Per-step tool selection | Beyond a few dozen tools, retrieve a relevant subset per step instead of exposing all of them every turn |

## The smallest runnable agent loop

The review of every framework tutorial is the same: the reader assembles five
abstractions and still cannot see the loop. So here is the entire loop in one
file with zero installs. Every production component is swapped for the
smallest thing with the same interface: the model becomes a seeded coin flip
with success probability $q$, the tools become a token-count table, the gate is
real code, and the cost meter charges prefill over the growing transcript
exactly as [section 5](05-reliability-and-cost.md) prices it. The shape is the
lesson; every section of this chapter upgrades one function of this file.

```python
"""A support-agent loop in one file: gate, step cap, token budget, cost meter."""
import random

STEP_CAP, TOKEN_BUDGET = 10, 10_000        # hard limits, enforced in code
PRICE_IN, PRICE_OUT = 1e-6, 3e-6           # $/token; illustrative blended rates
REFUND_LIMIT = 50                          # policy lives here, not in a prompt

TOOLS = {                                  # tool name -> tokens its result appends
    "lookup_account": 120, "lookup_order": 150, "check_eligibility": 60,
    "issue_refund": 40, "send_reply": 80,
}
PLAN = list(TOOLS)                         # plan-then-execute: known ticket shape

def gate(tool, args):
    """Deterministic pre-call check. The model proposes; code disposes."""
    if tool == "issue_refund":
        if not isinstance(args.get("amount"), (int, float)) or args["amount"] <= 0:
            return "reject: schema"
        if args["amount"] > REFUND_LIMIT:
            return "escalate: human approval queue"
    return "allow"

def run_ticket(rng, q=0.95, refund=32, retry=True, verbose=False):
    """One agent loop. Returns (resolved, steps, cost). q = per-step success."""
    transcript, cost, steps = 400, 0.0, 0  # 400 = system prompt + ticket tokens
    for tool in PLAN:
        while True:
            if steps >= STEP_CAP or transcript > TOKEN_BUDGET:
                return False, steps, cost  # hard limit hit: escalate to human
            steps += 1
            cost += PRICE_IN * transcript + PRICE_OUT * 30  # prefill re-reads all
            args = {"amount": refund} if tool == "issue_refund" else {}
            verdict = gate(tool, args)
            if verdict != "allow":
                if verbose:
                    print(f"  step {steps}: {tool} -> {verdict}")
                return False, steps, cost  # routed to a human, never executed
            ok = rng.random() < q          # per-step success draw
            transcript += TOOLS[tool] + 30  # observation + action text appended
            if verbose:
                print(f"  step {steps}: {tool:18s} {'ok' if ok else 'FAIL'}"
                      f"  transcript={transcript:5d}  cost=${cost:.5f}")
            if ok:
                break                      # next planned step
            if not retry:
                return False, steps, cost  # one bad step sinks the whole task
    return True, steps, cost

rng = random.Random(7)
print("one ticket, verbose (q=0.95, refund=$32):")
done, steps, cost = run_ticket(rng, verbose=True)
print(f"resolved={done}  steps={steps}  cost=${cost:.5f}\n")

print("policy gate demo (a hijacked model proposes refund=$500):")
done, steps, cost = run_ticket(rng, refund=500, verbose=True)
print(f"resolved={done}: escalated; the gate never executed the write\n")

print("error compounding over 2000 tickets per setting (5 planned steps):")
print("   q     q^5 predicted   no-retry observed   retry-under-cap   mean cost")
for q in (0.99, 0.95, 0.90, 0.70):
    plain = [run_ticket(random.Random(i), q=q, retry=False) for i in range(2000)]
    retried = [run_ticket(random.Random(i), q=q) for i in range(2000)]
    obs = sum(r[0] for r in plain) / 2000
    ret = sum(r[0] for r in retried) / 2000
    mc = sum(r[2] for r in retried) / 2000
    print(f"  {q:.2f}      {q**5:.3f}            {obs:.3f}             "
          f"{ret:.3f}         ${mc:.5f}")
```

Run it and three things happen in about sixty lines. The verbose ticket
resolves in 5 steps for \$0.00384, and the per-step cost lines rise as the
transcript grows, which is [section 5](05-reliability-and-cost.md)'s prefill
term made visible. The hijacked ticket proposes a \$500 refund and the gate
returns "escalate: human approval queue" at step 4; the write never executes,
no matter what the model wanted. The Monte Carlo table then shows error
compounding: without retries, observed success tracks $q^5$ almost exactly
(0.789 observed vs 0.774 predicted at $q = 0.95$), while executor retries
under the step cap lift $q = 0.90$ back to 1.000 and even $q = 0.70$ to 0.949,
at a mean cost that climbs from \$0.00390 to \$0.00653 as retries burn extra
prefill. Retries buy back reliability with money, and the cap bounds the
purchase. Swap the coin flip for a model call, the token table for real tools,
keep the gate and the limits exactly where they are, and you have rebuilt this
chapter.
