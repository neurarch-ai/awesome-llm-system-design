# LLM Inference Serving at Scale

An interviewer rarely says "implement continuous batching." They say **"we are
serving an LLM API at high QPS and GPU costs are climbing. Walk me through the
serving stack."** That question is about throughput engineering, not model design.
The whole game is packing as many tokens as possible into each GPU step without
letting tail latency slip past the SLO. Those two goals pull against each other in
ways that determine every architectural choice in this chapter.

This chapter builds the serving system end to end and shows how Anyscale, Character.AI,
LinkedIn, NVIDIA, Together AI, Fireworks, Moonshot, and others actually ship it.

## Sections

1. [Clarifying the requirements](01-clarifying-requirements.md) - the dialogue that scopes the problem and the two consequences that follow.
2. [The throughput problem](02-the-throughput-problem.md) - prefill vs. decode, the memory wall, TTFT vs. inter-token latency.
3. [Batching](03-batching.md) - continuous batching, chunked prefill, and disaggregated serving.
4. [Speculative decoding](04-speculative-decoding.md) - draft-and-verify, the speedup formula, and when it helps.
5. [Parallelism and quantization](05-parallelism-and-quantization.md) - tensor, pipeline, and expert parallelism; weight and KV precision.
6. [Autoscaling and cost](06-autoscaling-and-cost.md) - leading-signal autoscaling, cold starts, and cost per million tokens.
7. [How teams do it in production](07-how-teams-do-it-in-production.md) - named companies, where they diverge, and first-party write-ups.
8. [Interview Q and A](08-interview-qa.md) - commonly asked, tricky, and commonly answered wrong.
9. [Summary](09-summary.md) - one-page recap, the system on one page, and self-test questions.
10. [Putting it together: the complete build](10-putting-it-together.md) - a default stack, the scenario built end to end with sizing and cost math, the same system under three different constraint sets, and the smallest runnable scheduler.

## The whole system on one page

```mermaid
flowchart LR
  REQ["incoming request"] --> GATE["SLO gate<br/>(admit or queue)"]
  GATE --> SCHED["continuous-batching<br/>scheduler"]
  SCHED --> PRE["prefill<br/>(compute-bound)"]
  PRE -->|"writes KV"| KV["paged KV cache"]
  KV -->|"read per step"| DEC["decode<br/>(bandwidth-bound)"]
  DEC -->|"appends KV"| KV
  DRAFT["draft model<br/>(optional)"] --> DEC
  DEC --> OUT["streamed tokens"]
  AUTO["autoscaler<br/>(SLO-driven)"] -.-> SCHED
```

Read the sections in order the first time; they build on each other. Each opens
with the question an interviewer actually asks, then answers it.

## Companion chapters

[Model compression](../model-compression/) goes deeper on the artifact this chapter
serves: quantization formats and the outlier problem, pruning shapes the hardware
can actually skip work for, distillation, and the acceptance test a compressed model
has to pass before it replaces the original.

The classic-ML companion book covers the same ground from the other side:
[realtime-serving](https://github.com/neurarch-ai/awesome-ml-system-design/tree/main/book/realtime-serving/) is the classic-ML serving problem, where requests are cheap and uniform so queueing, not the KV cache, sets the p99.
