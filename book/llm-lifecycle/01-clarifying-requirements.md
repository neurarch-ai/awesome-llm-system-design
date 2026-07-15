# 1. Clarifying requirements

Before sizing anything, pin down which stage you are actually being asked about.
The most common mistake in an LLM system design is treating every question as a
pretraining question (the from-scratch, trillions-of-tokens training run that
produces a raw base model). It almost never is.

Here is a typical exchange. Notice that every question either removes a whole
stage from scope or fundamentally changes the design.

---

**Candidate:** What capability does the company not get from an existing API
today? What is the gap that justifies owning weights?

**Interviewer:** Data residency, mostly. The company processes sensitive legal
documents and cannot send them off-premises. They also want domain-specific
terminology and citation style that the frontier models get wrong.

---

**Candidate:** Do you have any existing open weights to start from, or does
this need to be a from-scratch pretrain?

**Interviewer:** An open base is fine. Budget is real but not lab-scale; no
hundreds of millions of dollars.

---

**Candidate:** What does the domain corpus look like? How many tokens, how
clean, and how fast does it change?

**Interviewer:** About 50 billion tokens of case law and contracts, already
cleaned. Updated quarterly.

---

**Candidate:** What is the serving target? Latency, throughput, cost per
query?

**Interviewer:** Interactive: p95 time-to-first-token under two seconds, maybe
500 concurrent users at peak. Cost matters; it is not a batch job.

---

**Candidate:** What is the safety and eval bar? Who gets hurt by a bad output?

**Interviewer:** The users are trained lawyers reviewing suggestions. A wrong
citation is bad but audited. We need citation accuracy and instruction
following, not consumer-grade safety hardening.

---

Let us summarize the problem. We need to take a capable open-weights base
model, adapt it to legal domain text and vocabulary (mid-training, meaning continued
pretraining on domain data, on 50B domain tokens), turn it into an
instruction-following model that cites sources correctly (post-training, the
SFT-plus-preference-tuning stage that makes a base model follow instructions),
and serve it under interactive latency at modest concurrency with cost
discipline.

**Two consequences fall out immediately:**

- **This is almost certainly a mid-training plus post-training problem, not a
  pretrain.** A from-scratch pretrain of a model large enough to matter costs
  hundreds of millions of dollars and weeks of compute. The leverage for this
  team is in adapting an open base (Llama 3, Qwen3, OLMo) on the 50B domain
  tokens, then running SFT (supervised fine-tuning on instruction-response pairs)
and preference optimization. The base model's general
  language ability is free; you are only paying for the delta.

- **The serving constraint rules out the largest models at full precision.** Two
  seconds time-to-first-token at p95 with 500 concurrent users means the model
  must be small enough to serve under budget, which requires quantization
  (storing weights in lower-precision integers to shrink memory, INT8
  at minimum), and the KV cache (the saved keys and values for past tokens)
  size per user sets how many concurrent requests
  fit on each GPU. The serving design is not a detail; it determines which model
  size is even viable.

## A quick rule of thumb for the four cases

| Interviewer says | What stage is actually being asked | Typical answer |
|---|---|---|
| "Build us an LLM for our domain" | mid-training on an open base | continued pretraining plus post-training on a Llama / Qwen3 / OLMo base |
| "Make it follow our style guide and refuse harmful requests" | post-training | SFT plus DPO on the base's outputs; no pretraining |
| "We need a 200K context for our legal documents" | mid-training (context extension) | RoPE rescaling plus continued train on long docs from an existing base |
| "We need a new foundation model for a new language" | pretraining | from-scratch pretrain on a large multilingual corpus; only justified at lab scale |

State which stage the problem belongs to before you size anything. Jumping
straight to "we will pretrain" when mid-training would do the job is the
fastest way to fail this interview.
