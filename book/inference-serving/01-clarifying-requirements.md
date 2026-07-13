# 1. Clarifying the requirements

Before designing anything, pin down what the system must do. Here is a typical
exchange between a candidate and an interviewer. Notice that every question either
removes work, changes the design, or changes where the bottleneck sits.

**Candidate:** What is the latency target, and is it on first token or on each
subsequent token?
**Interviewer:** Both matter. We want p99 time-to-first-token (TTFT) under 500 ms
and p99 inter-token latency (also called TPOT, time per output token) under 50 ms.
They are separate SLOs.

**Candidate:** What is the expected QPS, and does traffic spike?
**Interviewer:** Average around 500 requests per second, with spikes up to 3x that.
Traffic is bursty around certain hours.

**Candidate:** What does a typical request look like? Long prompts or long outputs?
**Interviewer:** Mixed. Some are RAG calls with 8k-token prompts and short answers.
Others are code or agent calls with short prompts and long outputs. Assume you
cannot predict which at admission time.

**Candidate:** Does the model fit on one GPU?
**Interviewer:** The model is a 70-billion-parameter dense transformer. It does not
fit on a single A100. Assume H100s are available.

**Candidate:** Are there multiple models or fine-tunes to serve, or one model?
**Interviewer:** One model for now. If you want to add a smaller draft model for
speculative decoding, you can.

**Candidate:** Do we need to support multiple tenants with different priority
classes?
**Interviewer:** Yes, a paid tier and a free tier. The paid tier must be protected
under overload.

**Candidate:** What is the cost target?
**Interviewer:** Minimize cost per million output tokens. GPU time is the main
expense.

Let us summarize the problem statement. **We are asked to design a serving system
for a 70B-parameter LLM at around 500 QPS (with 3x spikes) that meets a 500 ms
p99 TTFT and a 50 ms p99 inter-token latency SLO, across a mixed prefill-heavy
and decode-heavy workload, on H100 GPUs, with two priority tiers, while minimizing
cost per million output tokens.**

Two consequences fall out immediately, and stating them early is most of the signal
in this question:

**TTFT and TPOT are different knobs.** A long prefill occupying the GPU for a
full step blocks all in-flight decode requests for that step, spiking TPOT. A
packed batch with many decode steps in flight delays new requests waiting to start
their prefill, spiking TTFT. Optimizing one without hurting the other requires
either chunking prefill or disaggregating the two phases onto separate pools.
Which choice you make is the first design decision.

**The model does not fit on one GPU.** A 70B dense model in BF16 is about 140 GB
of weights alone. A single H100 has 80 GB of HBM. Before you can think about
batching or scheduling, you must shard the model across at least two GPUs. Tensor
parallelism (splitting each layer across GPUs) or pipeline parallelism (splitting
layers across stages) is not optional; it is a prerequisite. This also means you
are designing for a multi-GPU engine from the start, not a simple single-device
stack.
