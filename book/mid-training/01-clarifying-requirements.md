# 1. Clarifying the requirements

Before designing anything, pin down what the system must actually do. Here is a
typical exchange between a candidate and an interviewer. Notice that every
question either removes work or changes the design.

---

**Candidate:** Is the goal to adapt the model to a specialized domain, to extend
its context window (the span of text, measured in tokens, that the model can read
at once), or both?

**Interviewer:** Both, ultimately. We have a clinical-notes corpus and documents
that run to 200 pages. The base model was trained on the general web at 8K tokens
(a token is a chunk of text, roughly a word or word-piece).

---

**Candidate:** How many in-domain tokens do we have?

**Interviewer:** About 40 billion tokens of de-identified clinical notes.

*Why this matters: below a few billion tokens the domain set is too small for
continued pretraining to shift the prior; you would reach for retrieval or a small
SFT run instead. 40B is enough to move the base substantially.*

---

**Candidate:** What is the actual p95 document length we need to serve?

**Interviewer:** About 60K tokens. Some discharge summaries run longer, but p95
is around 60K.

*Why this matters: if the real ask is 60K, extending to 128K and paying quadratic
attention cost for the extra headroom wastes serving budget. Better to extend to
64K or 128K and confirm the gap with a measurement, not an assumption.*

---

**Candidate:** Is there a general-capability floor we must not drop below?

**Interviewer:** Yes. MMLU, GSM8K, and instruction-following scores must not
regress by more than two percentage points from the base.

*Why this matters: this is the catastrophic-forgetting contract. It fixes the
regression bar before we start, so we can detect forgetting rather than celebrate
only the domain gain.*

---

**Candidate:** Where does this adapted base sit in the pipeline?

**Interviewer:** It feeds into supervised fine-tuning and preference optimization.
Downstream post-training will align it as a chat model.

*Why this matters: adaptation happens on the base before alignment. Extending
context or shifting the domain after alignment risks disturbing aligned behavior,
so almost all recipes do it here.*

---

**Candidate:** Do we need the long-context model available via the proxy, or can
users supply their own keys and serve it themselves?

**Interviewer:** It will run on our infrastructure. Serving budget is real.

---

## Summary

We are asked to adapt an 8K-window general base to clinical notes and to serve
documents up to roughly 60K tokens, while holding MMLU, GSM8K, and
instruction-following within two percentage points of the base. The adapted base
must be serveable on our infrastructure and must flow into post-training afterward.

Two consequences fall out of this immediately, and stating them early is most of
the signal in this question:

- **Domain adaptation and context extension are independent problems with
  different failure modes.** Domain adaptation risks catastrophic forgetting (the
  model gains clinical prior and loses general reasoning). Context extension risks
  short-context regression (the model reads 60K but gets worse at 2K prompts) and
  serving cost blowup (quadratic attention prefill, linear KV-cache growth). Naming
  the two axes and their distinct failure modes before reaching for a tool is the
  first signal a strong candidate sends.

- **We are not pretraining from scratch.** That single fact unlocks the whole
  design: continued pretraining re-enters the same next-token objective the base
  already converged on, under a carefully lowered learning-rate schedule, with a
  changed data mix. For context extension it adds a rescaling of the positional
  encoding before a short training run on long documents. The art is entirely in
  the schedule, the mix, and the rescaling, not in a new loss function.
