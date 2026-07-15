# 1. Clarifying the requirements

Before designing anything, pin down what the system must do. Here is a typical
exchange between a candidate and an interviewer. Notice that every question either
removes work or fundamentally changes the design.

**Candidate:** Are we genuinely pretraining from scratch, or continue-pretraining
an existing open base?
**Interviewer:** From scratch. Assume we have a new target domain and the open
bases do not cover it adequately.

**Candidate:** Good. I will say upfront that pretraining from scratch is almost
never the right call outside a research lab: a new language, a new modality, a
new tokenizer (the component that splits raw text into the integer tokens a model reads), or a genuine capability gap in every open base are the only
situations that justify it. I'll proceed assuming one of those applies, but I'll
note the alternative at the end. What is the compute budget?

**Interviewer:** Roughly 10,000 GPU-days of A100s. Call it about
$6 \times 10^{22}$ FLOPs (floating-point operations, the raw count of arithmetic used to measure compute).

**Candidate:** That pins the model size and token count before I pick any
architecture. What data do we have rights to, and what languages must the model
cover?

**Interviewer:** We have access to web crawl (Common Crawl, a free periodic snapshot of the public web), a licensed book
corpus, code repositories, and academic papers. Target is primarily English with
some multilingual capability.

**Candidate:** What is the target use case: a general-purpose base, a
domain-specialized base, or a multilingual base? That choice drives the data
mixture, the tokenizer vocabulary size, and the evaluation suite.

**Interviewer:** General purpose, English-primary, with the ability to handle
some code and reasoning.

**Candidate:** How will we know it worked, and how do we protect against fooling
ourselves on the eval? Specifically: which benchmarks, and how will we
decontaminate training data against them (remove any training text that overlaps the benchmark questions)?

**Interviewer:** Standard suite: MMLU, ARC, HellaSwag, HumanEval. You choose
the decontamination approach.

**Candidate:** Last question: will the model be served heavily at inference time,
or is this primarily a research artifact? That changes whether I optimize for
training compute or lifetime inference cost.

**Interviewer:** Heavy serving. Assume billions of tokens served per day.

---

Let us summarize the problem statement. **We are building a general-purpose
English-primary base model from scratch on roughly $6 \times 10^{22}$ FLOPs,
from web crawl plus curated corpora, evaluated on a decontaminated standard
suite, intended for heavy production serving.**

Two consequences fall out immediately, and stating them early is most of the
signal in this question:

**Data is the capability ceiling.** Model quality is bounded by data quality
long before it is bounded by architecture or optimizer. A petabyte of raw
Common Crawl is mostly boilerplate, spam, and near-duplicates; training on it
directly gives a worse model than training on the small, clean fraction that
survives a proper pipeline. The keep rate after extraction, language filtering,
quality filtering, and deduplication (removing repeated and near-repeated documents) is commonly single-digit percentages of
the raw bytes. That is not a failure; it is the design.

**Almost no one should pretrain from scratch.** The right answer to most
pretraining questions is "continue-pretrain an open base." Pretraining from
scratch is justified when: the target language has no adequate open base, the
modality is new (images, molecules, code from a private corpus), the tokenizer
must differ, or a capability genuinely absent from every open model is required.
Stating this unprompted, and then explaining why you are proceeding anyway, is a
senior signal. The rest of this chapter assumes the interviewer's framing holds.
