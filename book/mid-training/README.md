# Mid-Training: Continued Pretraining and Long-Context Adaptation

> **Style note.** This chapter teaches first. It borrows the thinking of a
> structured system-design interview (clarifying requirements, framing the
> two axes, deep-diving each mechanism, evaluating, serving, and finishing with
> interview Q and A) without copying any particular format. On top of that it
> keeps what this repo adds: real production case studies with first-party links,
> a "when to use which" table per method group, worked figures (mermaid and
> matplotlib), and an interview Q and A. Split into one file per section so no
> single file gets long.

An interviewer rarely says "implement YaRN." They say **"you have a strong open
base model. Your product needs it to know a specialized domain and to read
documents far longer than the 8K window it was pretrained at. Walk me through how
you adapt the base without wrecking what it already knows."** That is continued
pretraining and long-context extension: two independent adaptations that sit in
the cheap, high-leverage gap between a base model and an aligned chat model. This
chapter builds both end to end, and shows how Meta, Nous Research, Microsoft, 01.AI,
Alibaba, and the Mila group actually do it.

**This stage is what labs now call mid-training.** The phase between pretraining
and post-training has its own budget, its own data, and its own evals: mixture
reweighting, the anneal (decay) phase, capability injection, long-context
extension, and preparing the base for post-training. Section 3 opens with that
lab-side view and then walks the practitioner-side recipe (adapting a released open
base) that most product teams actually run. Where the model's data pipeline itself
is the subject, see [data curation and pretraining](../data-and-pretraining/); where
the question is the whole five-stage map, see [the LLM lifecycle](../llm-lifecycle/).

## Sections

1. [Clarifying the requirements](01-clarifying-requirements.md) -- the dialogue that scopes the problem.
2. [Two axes](02-two-axes.md) -- the adaptation axis (domain) and the length axis; input and output.
3. [The mid-training phase](03-the-mid-training-phase.md) -- what mid-training means now, data mixture and microanneals, the anneal phase and stable-phase branching, capability injection and RL readiness, then DAPT, replay against forgetting, and the LR schedule.
4. [Context extension](04-context-extension.md) -- PI, NTK-ABF, YaRN, LongRoPE, ALiBi; KaTeX for the math.
5. [Evaluation](05-evaluation.md) -- needle-in-a-haystack, RULER, forgetting checks; what each measures.
6. [Serving and scaling](06-serving-and-scaling.md) -- memory cost at length, bottlenecks table.
7. [How teams do it in production](07-how-teams-do-it-in-production.md) -- Meta, Nous, Microsoft, 01.AI, Alibaba, Mila.
8. [Interview Q and A](08-interview-qa.md) -- commonly asked, tricky, and commonly answered wrong.
9. [Summary](09-summary.md) -- one-page recap, mermaid, test-yourself, further reading.
10. [Putting it together: the complete build](10-putting-it-together.md) -- a default stack, the scenario built end to end with token-budget and cost math, the same recipe under three different constraint sets, and the smallest runnable context extension.

## The whole pipeline on one page

```mermaid
flowchart TD
  BASE["pretrained base (4K to 8K window)"] --> MIX["domain corpus + general-data replay"]
  MIX --> DAPT["domain-adaptive pretraining<br/>(re-warm LR, modest peak, re-decay)"]
  DAPT --> ADB["domain base"]
  ADB --> RS["rescale RoPE frequencies<br/>(PI / NTK-ABF / YaRN / LongRoPE)"]
  RS --> LONG["long-context corpus<br/>(upsampled long docs + synthetic)"]
  LONG --> LCT["long-context continued training<br/>(staged length increase)"]
  LCT --> EXB["domain and long-context base"]
  EXB --> POST["post-training (SFT + preference opt)"]
  DAPT -. "general-benchmark regression gate" .-> ADB
  LCT -. "needle + RULER long-context gate" .-> EXB
```

Read the sections in order the first time; they build on each other. Each section
opens with the question an interviewer actually asks, then answers it.
