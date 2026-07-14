# 2. Decide: prompt, RAG, or train

Say this ordering out loud in the interview, because the ordering itself is the
signal. Every rung is cheaper and faster than the one below it. Stop at the first
rung that clears the quality bar.

## The decision ladder

**Rung 1: Prompt engineering.** Rewrite the system prompt, add few-shot examples,
introduce output schemas, decompose complex requests into steps. This is free,
instant to iterate, and it is your baseline. You cannot know whether fine-tuning
helped if you never tuned the prompt first.

When it works: format gaps, tone gaps, instruction-following failures, and tasks
where the model has the underlying capability but is not being directed to use it.

When it is not enough: a skill the base model genuinely does not have; a very long
few-shot prompt you want to compress for cost; a behavior that must be deeply
consistent across millions of calls.

**Rung 2: Retrieval-augmented generation (RAG).** Put the facts in context instead
of in the weights. RAG updates the instant a document changes, cites sources, does
not require a training run, and never goes stale the way baked-in weights do. The
decision rule: **RAG teaches the model what it does not know; fine-tuning teaches
it how to behave.**

When it works: the gap is knowledge (your documentation, your product catalog, your
support tickets), and that knowledge changes over time.

When it is not enough: the problem is not knowledge but behavior, tone, or format.
A model that retrieves perfectly but still writes in the wrong style needs
fine-tuning, not more retrieval. The two compose well: tune for behavior, retrieve
for knowledge, on the same base.

**Rung 3: Supervised fine-tuning (SFT).** Show the model the inputs and outputs
you want, and train it to produce those. The right tool for consistent output
format, a domain-specific tone, a reasoning pattern, or a structured-extraction
skill the base model fumbles. Also how you compress a long few-shot prompt into the
weights so every request pays a short-prompt cost.

When it works: you have a few thousand clean labeled examples, the behavior is
stable (not churning weekly), and prompt engineering left a meaningful gap.

When it is not enough: you need to pick one correct answer from two plausible ones,
or align the model on a safety axis, or suppress a tempting-but-wrong style that
SFT alone cannot reliably rule out.

**Rung 4: Preference optimization (DPO or RLHF).** Train on comparisons: "this
response is better than that one for this prompt." Teaches the model to *prefer*
one acceptable answer over another, which SFT (which only imitates positive
examples) cannot express. Details in the methods chapter.

When it works: after SFT, a quality axis you care about (helpfulness, safety,
picking the better of two valid answers) is still not met; you have comparison
data or can generate it.

When it is not enough: SFT already solved the problem. Proposing RLHF for a
format-and-tone problem is over-engineering and a common interview trap.

## The trap to name out loud

Fine-tuning to teach facts is expensive, goes stale, hallucinates confidently
between the facts you trained on, and must be redone whenever the facts move.
If the gap is knowledge, reach for retrieval. If the gap is behavior, reach for
fine-tuning. Getting this wrong in either direction is a common and costly mistake,
and naming it explicitly is one of the clearest signals in this question.

## When to use which

| Reach for | When | Instead of |
|---|---|---|
| Prompt engineering | format, tone, or instruction-following is the gap; zero cost, instant iteration | anything below on this list, until you have a baseline |
| RAG | the gap is knowledge that changes over time and must cite sources | fine-tuning stale facts into weights that need retraining to update |
| SFT | behavior or skill gap; you have thousands of clean labeled examples | RAG for a problem that is about style, not knowledge |
| Preference tuning (DPO or RLHF) | a quality axis SFT cannot capture: safety, helpfulness, picking the better answer | SFT, when better examples already close the gap |
| None of the above yet | the problem description is too vague to diagnose | committing to a training run before the baseline is measured |

**Tools for each rung.** Prompt engineering needs nothing beyond the model API and
an eval harness to check the baseline. RAG is served by retrieval and orchestration
stacks such as LlamaIndex, LangChain, and Haystack over a vector store. SFT and
preference tuning both run on Hugging Face TRL (its SFTTrainer and DPOTrainer),
usually with PEFT for LoRA or QLoRA adapters, and higher-level wrappers like Axolotl
and Unsloth make the training config declarative. DeepSpeed (Microsoft) backs the
distributed training when a run outgrows a single GPU.

**Worked example.** A support-automation team finds the assistant answers in the
wrong tone and does not know the current return policy. They start at rung 1 and
rewrite the system prompt, which fixes the tone but not the policy gap, because the
policy changes every quarter and must cite the source document. That knowledge gap
is exactly what RAG is for, so they index the policy pages rather than baking dates
into weights that would go stale by the next revision. When the tone still drifts
across millions of calls, they add a small SFT pass on curated transcripts to make
the voice consistent, composing tuned behavior with retrieved knowledge on the same
base, and they hold off on preference tuning because SFT already closed the gap.
