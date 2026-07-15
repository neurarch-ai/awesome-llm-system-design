# 4. Output guardrails

Output guardrails run on the LLM's generation before it reaches the user. They
catch a different class of problem from input guards: a perfectly benign prompt can
still yield an unsafe completion, a hallucinated claim, or a regurgitation of
another user's private data. Input guards are necessary but not sufficient.

## Toxicity and policy classifiers

A toxicity classifier (toxicity means harmful language such as hate speech,
threats, or self-harm content) scores the completion for harmful content: violence,
self-harm, sexual content, hate speech. A policy classifier scores for
domain-specific violations: off-topic replies in a customer-service product,
medical advice in an application that is not licensed to give it, financial
recommendations with disclaimed caveats.

Both are typically implemented as guard models: fine-tuned transformers trained on
labeled examples of the categories the product enforces. The key design point is
that they run on the output text as a separate decision, independent of the main
model. An attacker who successfully jailbreaks the base model's refusal training
still has to get past the output classifier, which did not participate in the
conversation and cannot be argued with.

Salesforce's Einstein Trust Layer uses a hybrid of deterministic rules plus a model
for seven toxicity categories: pure-model scoring misses obvious violations, and
pure rules miss nuance. The hybrid is more robust than either alone.

## Groundedness classifiers (for RAG)

In a RAG product, an additional class of unsafe output is an ungrounded claim: the
model stated something that is not supported by the retrieved sources. In a
high-stakes domain (legal, medical, financial), an ungrounded hallucination is a
safety issue, not just a quality issue.

A groundedness classifier compares the generated text against the retrieved chunks
and returns a support score. A minimal version scores what fraction of the answer's
words actually appear in the sources; a low score flags a claim the sources do not back:

```python
def support_score(answer, sources):   # answer: generated text; sources: list of retrieved chunk texts
    a = set(answer.lower().split())
    src = {w for chunk in sources for w in chunk.lower().split()}
    # fraction of the answer's words that actually appear in the retrieved sources
    return len(a & src) / len(a)
# support_score("the filing reports a profit", ["the filing reports a loss"]) -> 0.8
```

NVIDIA's NeMo Guardrails uses AlignScore for this.
Thomson Reuters' CoCounsel grounds legal answers in a trusted corpus and runs
1,500 automated tests per night to verify the grounding holds.

The key insight is that groundedness and toxicity are orthogonal: a perfectly
polite, non-toxic output can still be an unsafe hallucination in a regulated
domain.

## The operating point: recall, precision, and the KL-anchored objective

Setting a classifier threshold is a business decision, not a default. The tradeoff
is explicit: a lower threshold blocks more harm but blocks more legitimate requests
too.

Define the catch rate (true positive rate) as:

$$\text{Recall} = \frac{TP}{TP + FN}$$

```python
def recall(tp, fn):                  # tp: attacks caught; fn: attacks that slipped through
    # catch rate: of all real attacks (tp + fn), what fraction the guard flagged
    return tp / (tp + fn)
# recall(90, 10) -> 0.9
```

And the false-refusal rate (false positive rate: the share of legitimate requests
wrongly blocked) as:

$$\text{FRR} = \frac{FP}{FP + TN}$$

```python
def frr(fp, tn):                     # fp: benign requests wrongly blocked; tn: benign correctly allowed
    # false-refusal rate: of all benign requests (fp + tn), what fraction got blocked
    return fp / (fp + tn)
# frr(3, 997) -> 0.003
```

The operating point is the threshold that sets both. Reporting only the catch rate
is misleading. Anthropic's Constitutional Classifiers dropped the attack success
rate from 86% to 4.4% on an adversarial eval set, and also held the benign
production refusal rate increase to 0.38%, which is the number that proves low
over-blocking.

A useful way to state the operating point constraint is: fix the false-refusal rate
budget and read off the catch rate you can achieve, or vice versa:

$$\text{Recall} \Bigl|_{\text{FRR} \leq \delta} = \max \Bigl\lbrace \frac{TP}{TP + FN} : \frac{FP}{FP + TN} \leq \delta \Bigr\rbrace$$

```python
import numpy as np
def recall_at_frr(scores_pos, scores_neg, delta):   # guard scores for attacks (pos) and benign (neg)
    best = 0.0
    # sweep every candidate threshold; keep the best attack-recall whose benign block rate stays <= delta
    for t in np.unique(np.concatenate([scores_pos, scores_neg])):
        frr = np.mean(scores_neg >= t)              # benign wrongly blocked at this threshold
        if frr <= delta:
            best = max(best, np.mean(scores_pos >= t))  # recall on attacks at this threshold
    return best
# recall_at_frr(np.array([.9,.8,.4]), np.array([.3,.2,.7]), 0.0) -> 0.6666666666666666
```

During training, the KL-anchored objective keeps refusal training from wrecking
benign behavior:

$$\max_{\pi} \; \mathbb{E}_{x \sim D}\bigl[R_{\text{safe}}(x, \pi)\bigr] - \beta \cdot \text{KL}\bigl(\pi \;\|\; \pi_{\text{ref}}\bigr)$$

The KL term is the concrete penalty on drift; between two next-token distributions it is:

```python
import numpy as np
def kl_divergence(p, q):             # p: trained-policy probs; q: reference-model probs, same tokens
    p, q = np.asarray(p), np.asarray(q)
    # sum p * log(p / q): how far the trained policy p has drifted from the reference q
    return float(np.sum(p * np.log(p / q)))
# kl_divergence([0.5, 0.5], [0.25, 0.75]) -> 0.14384103622589042
```

The refusal reward pushes the policy to decline harmful prompts. The KL term to
the reference model penalizes drift from benign behavior. A large $\beta$ keeps
the model helpful to legitimate users; a small $\beta$ increases catch rate but
raises the benign refusal rate.

![False-refusal vs harm tradeoff](assets/fig-harm-vs-overrefusal.png)

*Moving the operating point rightward (lower threshold) reduces false refusals but
allows more harm through. The curve is the product's safety-helpfulness frontier;
the operating point is a business decision, not a technical default. Illustrative.*

## When to use which output guard

| Reach for | When | Instead of |
|---|---|---|
| Small fine-tuned toxicity classifier | High-QPS output moderation; policy is stable; need low latency on every completion | A guard-LLM on the output path, which adds 80-150ms in series |
| Guard-LLM output classifier | Taxonomy flexibility matters; moderate QPS; you want category-level verdicts for routing | A binary toxic/not-toxic signal that cannot distinguish what kind of violation to route to |
| Grounding classifier (AlignScore or similar) | RAG product in a high-stakes domain (legal, medical, financial) | Assuming the LLM only uses retrieved content; hallucination and toxicity are orthogonal |
| Hybrid rules plus model (Salesforce) | Enterprise platform where deterministic rules must never miss obvious cases | Pure-model scoring that can miss rule-level violations and is harder to audit |
| Streaming token-level classifier | You want to cut off mid-generation and avoid returning partial unsafe output (Anthropic) | Waiting for the full completion before checking, which already surfaced unsafe tokens |
| G-Eval LLM-judge scorer (OpenAI cookbook) | Qualitative domain-specific criteria that are hard to encode in a trained classifier; low QPS | High-QPS paths; an LLM judge inherits the base model's persuadability and costs a full generation |
| Human review escalation | High-stakes ambiguity in regulated domains; appeals (Thomson Reuters, Roblox) | Automated hard-block on irreversible decisions where a false positive causes real harm |

**Tools.** Guard-LLM output classifiers include Llama Guard (Meta) and ShieldGemma (Google); NeMo Guardrails (NVIDIA) orchestrates output rails and uses AlignScore for groundedness, and Guardrails AI provides output validators plus a hybrid rules-and-model layer. Small fine-tuned toxicity classifiers such as Detoxify run on Hugging Face Transformers for the low-latency path, and streaming token-level checks are wired into the serving loop so generation can be cut off mid-stream. LLM-judge scorers along the G-Eval pattern are built on whichever frontier model you already call, and human-review escalation is a workflow and queue you build around the automated verdicts.

**Provenance.** The guard-LLM output row uses Llama Guard (Meta), orchestrated by NeMo Guardrails (NVIDIA). The streaming token-level cut-off row is attributed to Anthropic, as already noted in the table.

**Worked example.** A document-AI team ships a RAG assistant that answers questions over regulated financial filings, where a polite but ungrounded claim is itself a safety failure. On every completion they run a small fine-tuned toxicity classifier for low latency, but because toxicity and groundedness are orthogonal they add a grounding classifier that compares the answer against the retrieved chunks rather than assuming the model only used its sources. Since obvious rule-level violations must never slip through for audit reasons, they pair the models with deterministic rules in a hybrid rather than pure-model scoring. They set the classifier threshold by fixing a false-refusal budget and reading off the catch rate they can hit, and route the small slice of high-stakes ambiguous outputs to human review instead of hard-blocking, since a false positive on an irreversible decision would cause real harm.

## Implementation and training pitfalls

Output guards fail in two opposite directions at once: they over-block benign
users and they under-block determined attackers. Both are operating-point and
implementation problems, not reasons to abandon the guard. The recurring
failures:

| Problem | Symptom | Fix |
|---|---|---|
| Guardrail false positives (over-refusal) | Benign requests get refused, user complaints climb, engagement drops | Set the threshold against an explicit false-refusal budget and read off the catch rate; KL-anchor refusal training to the reference model so helpfulness does not drift |
| Jailbreak leakage past the base model | The model's own refusal is argued away and an unsafe completion is returned | Score the output with an independent classifier that never joined the conversation, so a jailbroken base model still has to clear it |
| Groundedness false flag on paraphrase | Correct answers that paraphrase the sources score low on lexical overlap and get blocked | Use an entailment or semantic support scorer rather than bare word-overlap; reserve overlap for a coarse pre-filter |
| Partial unsafe output already streamed | Unsafe tokens reach the user before the full-completion check runs | Run a token-level streaming classifier that can cut off generation mid-stream instead of checking only the finished text |
| Private-data regurgitation | The completion echoes another user's PII even from a benign prompt | Add an output-side PII or DLP scan independent of the input, since input guards cannot see what the model generated |
| Threshold drift over time | Catch rate degrades quietly as the traffic and attack mix shift | Monitor recall and false-refusal rate on a held-out labeled set and recalibrate the operating point on a schedule |
| Non-English bypass | Attacks in other languages sail past an English-trained guard | Use a multilingual guard model or per-language thresholds; do not assume the base model's language coverage matches the guard's |
| Guard-LLM latency stacking | Every completion pays a full extra generation in series | Keep a small fine-tuned classifier on the hot path and reserve the guard-LLM for low-QPS or ambiguous routing only |

The guard is a business decision expressed as a threshold: pick the operating
point from the false-refusal budget the product can tolerate, then watch it drift
and recalibrate, rather than shipping a default and trusting it.
