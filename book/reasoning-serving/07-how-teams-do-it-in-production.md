# 7. How teams do it in production

Every stack that serves a thinking model converges on the same skeleton: a budget
knob, a policy that decides who gets how much, and some way to tell whether the
answer was right. What differs is **where the budget is controlled** (provider API,
your scheduler, or the prompt) and **what plays the role of verifier**, and those
two choices explain the rest of the design.

## Where the real designs diverge

| Approach | Budget control | Selection or verification | Latency posture | When it wins | Watch out |
|---|---|---|---|---|---|
| Provider effort parameter (hosted reasoning APIs) | An effort or thinking-token setting per request | None built in; you add it | Provider-side queueing; you see only the total | Fast to adopt, no serving work | Little visibility into the tail; budget semantics differ per vendor |
| Self-hosted with a scheduler | Hard token caps plus queue policy per budget class | Whatever you build | You own the tail and can fix it | Strict SLOs, mixed traffic, cost control | Serving engineering: preemption, priority, length prediction |
| Prompt-level budget forcing | Suppress or inject the end-of-thinking marker | Usually self-consistency | Same as the underlying model | No budget parameter available, or you want finer control | Forcing a budget the model was not trained for can degrade output ([s1](https://arxiv.org/abs/2501.19393)) |
| Best-of-n with an executor | Fixed k, parallel | Unit tests, compiler, SQL execution | Latency of the slowest sample | Code and query generation | Sandbox capacity becomes the bottleneck; weak tests accept wrong answers |
| Cascade with a checker | Cheap path, then escalate | Executor, schema check, or certified judge | Bimodal: fast for most, slow for escalations | Mixed workloads with a solvable majority | Escalation storms when traffic shifts |
| Process-supervised selection | Fixed k, parallel | A step-level reward model | Verification adds its own tokens | Long derivations where errors are localizable | Verification can cost more than generation |
| RL-trained reasoner, served plainly | Whatever the model does by default | None | Fully at the mercy of the length distribution | Research and evaluation settings | Not a production posture: no cap, no policy, no accounting |

## The dividing line

Two questions place any design. **Who owns the tail?** If the provider does, you
trade control for speed of adoption and your only levers are the effort parameter
and request-level timeouts. If you do, you can fix head-of-line blocking properly,
and you have to. **What is the verifier?** An executor makes cascades and best-of-n
cheap and trustworthy; a learned reward model makes them possible but gameable; no
verifier at all means sequential thinking is your only lever and the extra samples
you buy are unusable.

A complete answer names both and connects them to the workload: verifiable tasks
should be cascading against an executor, unverifiable ones should be running a
measured fixed budget with a forced-answer boundary, and both should be reported as
cost per solved task.

## First-party sources

- **DeepSeek** [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948): the open account of training a reasoning model with verifiable rewards, and where the inference-time behavior comes from.
- **Stanford and collaborators** [s1: Simple test-time scaling](https://arxiv.org/abs/2501.19393): budget forcing at inference time, including forcing more thinking by suppressing the end-of-thinking token.
- **Google DeepMind and UC Berkeley** [Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters](https://arxiv.org/abs/2408.03314): allocating inference compute adaptively by problem difficulty beats a fixed setting.
- **Stanford** [Large Language Monkeys: Scaling Inference Compute with Repeated Sampling](https://arxiv.org/abs/2407.21787): coverage rises sharply with repeated sampling, and is only realizable with a way to select the right sample.
- **OpenAI** [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050): process supervision beats outcome supervision for selecting among reasoning attempts.
- **OpenAI** [reasoning guide](https://platform.openai.com/docs/guides/reasoning): the effort parameter and how reasoning tokens are billed and counted.
- **Anthropic** [extended thinking](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking): an explicit thinking-token budget per request.
- **Google** [thinking in the Gemini API](https://ai.google.dev/gemini-api/docs/thinking): thinking budgets and how to turn them off for latency-sensitive calls.
- **METR** [Measuring AI Ability to Complete Long Software Tasks](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/): capability expressed as the length of task a model completes, which is the demand side of the budget question.
- **UIUC and collaborators** [Establishing Best Practices for Building Rigorous Agentic Benchmarks](https://arxiv.org/abs/2507.02825): how weak outcome validation inflates results, which is the same failure mode as a weak accept test in a cascade.

For the dense single-file reference (same material, interview-walkthrough shape):
[topics/18-reasoning-and-test-time-compute.md](../../topics/18-reasoning-and-test-time-compute.md).
