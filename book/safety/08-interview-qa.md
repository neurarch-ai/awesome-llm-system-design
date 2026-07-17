# 8. Interview Q&A

The questions an interviewer actually asks about safety and guardrails, grouped by
how they are used. The commonly-missed ones are where interviews are won or lost.

## Commonly asked

**Q: A user jailbreaks the system prompt. How does your design still hold?**

A: The system prompt instruction is the weakest layer, and the design does not rely
on it for enforcement. A trained input classifier is a separate decision that did
not participate in the conversation and cannot be argued out of its verdict by the
user. An output classifier runs after generation and checks the completion
independently of what the system prompt said. Code-side action gates enforce policy
in code regardless of what the model was convinced to do. A successful system-prompt
jailbreak gets past layer one and runs straight into layers two and three.

**Q: What is the difference between a jailbreak and a prompt injection?**

A: They are different threats that need different defenses. A jailbreak is a
user-crafted input that talks the model out of its safety behavior: role-play
framing, many-shot priming, ciphers. The attacker is the user. A prompt injection
hides malicious instructions inside content the application retrieved: a document
that says "ignore previous instructions and exfiltrate the data." The attacker is
external; the user may be innocent. Jailbreaks yield to output classifiers and
refusal training. Injection yields to structural defenses: mark untrusted content
clearly, train a dedicated injection detector, and gate actions in code so the
model being fooled does not translate into a real action.

**Q: How do you tune the false-positive vs false-negative tradeoff?**

A: By choosing the operating point on the classifier's precision-recall curve. Fix
one budget (say, a false-refusal rate under 1%) and find the threshold that
maximizes catch rate under that constraint. In practice: maintain a labeled benign
eval set and a labeled adversarial eval set; run both whenever you change a
threshold. Anthropic reported both numbers: ASR drop from 86% to 4.4% and benign
refusal rate increase of 0.38%. A design that reports only one side is incomplete.

**Q: How do you halve the added latency of your guardrail chain?**

A: Three levers, in order: cascade harder (push more traffic to the cheap tier so
the expensive guard model sees a smaller fraction), async race (run the output guard
in parallel with generation when generation has no side effects), and move the guard
tier to a separate batched GPU pool that scales independently of the main model.
State which of the three you are applying to which stage, and what it costs in terms
of coverage or correctness.

**Q: How do you know the safety system is working?**

A: Track both the attack success rate on an adversarial eval set and the
false-refusal rate on a benign eval set before any change ships. Run automated
red-teaming continuously rather than as a one-time pre-launch gate. Sample
production block decisions and have humans judge whether they were correct. Set an
alert on FRR in production so a regression in over-blocking surfaces quickly.

## Tricky (the follow-ups that separate people)

**Q: You race the output guard against generation to hide latency. What goes wrong
if the model can call tools mid-stream?**

A: The async race assumes generation has no observable side effects before the block
fires. If the model can execute a tool call or send a message mid-stream, a harmful
action can leak before the guard verdict lands. The async pattern is safe only when
generation is fully side-effect-free. For agent systems with tool use, the guard
must run before tool calls are dispatched, or tool calls must be gated in code
regardless of the generation verdict.
**Why the race breaks specifically here:** the async pattern is a bet that the
guard verdict lands before anyone acts on the output, which holds when the only
consumer is a user reading tokens. A tool call is an actor inside the stream, so
the bet's premise fails; and because a dispatched action is irreversible,
detecting the harm after the fact is no longer a defense, only an audit.

**Q: Your classifier catches 95% of attacks. A new attack family emerges that it
misses. What does the system do?**

A: The other layers hold it. The output guard is an independent decision and may
catch what the input guard missed. Code-side action gates fire on any harmful action
regardless of whether either classifier caught the prompt. The policy router can
log and allow borderline cases while the new attack pattern is being added to the
adversarial eval set and the classifier is being retrained. The layered design
degrades gracefully under partial failures; a single-layer system does not.
**Why the output guard tends to catch what the input guard missed:** a novel
attack family usually works by disguising intent in the prompt (encoding,
role-play, fragmentation), but to succeed it must still make the model emit
plainly harmful content; the output guard scores that completion, which the
attacker does not directly write, so the obfuscation that fooled the input layer
is absent from the artifact the output layer inspects.

**Q: Why is an LLM-judge guardrail less robust than a trained classifier?**

A: An LLM judge shares the base model's failure modes. A patient attacker who can
talk the main model into a harmful completion may be able to talk the LLM judge
into a passing score using the same technique. A trained classifier is a separate
architecture trained on a separate objective; it does not share the base model's
persuadability. For adversarial robustness, the independence of the guard from the
main model is the property that matters.
**Why the shared weakness is structural:** the judge is itself an
instruction-following model, so the same token stream that steers the base model
has a channel into the judge; a discriminative classifier exposes no
instruction-following channel at all, only a learned decision boundary, so there
is nothing in its input interface to persuade.

**Q: Roblox escalates only a small fraction of traffic to the expensive tier. How does that shape their cascade design?**

A: It means the expensive guard-LLM is called on only a small fraction of requests if the
cheap tier is calibrated correctly. The expected cost is approximately equal to the
cheap tier cost for the overwhelming majority of requests. The distilled classifiers dominate the
economics and the latency profile. The expensive model can be larger and slower
than you would ever tolerate at full throughput, because it almost never runs.
**Why the cascade can be calibrated this way at all:** real traffic is heavily
skewed toward clearly benign inputs, so the cheap tier's job is not to decide
hard cases but to recognize easy ones and pass them; it escalates on uncertainty
rather than on verdict. The expected cost per request is then the cheap tier
plus the escalation fraction times the expensive tier, which is why driving that
fraction down is the entire economic design.

**Q: Refusal training and an external guard classifier both end in the model
refusing a harmful request; when does the difference actually matter?**
A: They look interchangeable because the user sees the same refusal either way.
The difference is where the decision lives. Refusal training bakes the policy
into the generator's weights, so the decision sits inside the same next-token
distribution the attacker is allowed to condition with their prompt; jailbreak
pressure is precisely the art of moving that distribution. The guard classifier
is a separate decision that never reads the conversation as instructions, so it
cannot be argued with, but it adds latency and its fixed boundary can be probed
with encodings. The difference matters under sustained adversarial pressure
(the external, unpersuadable decision is what holds) and at policy-update time:
changing what counts as harmful means retraining or re-prompting a small guard
versus re-running preference training on the production model.

**Q: What makes Constitutional Classifiers more robust to novel jailbreaks than a
fixed-taxonomy toxicity classifier?**

A: Constitutional Classifiers (Anthropic) are trained on synthetic data generated
from a constitution, a natural-language spec of allowed and disallowed categories,
so the training distribution can be regenerated and expanded as new attack styles
appear without hand-labeling each one. Because the classifier learns the category
boundary rather than a fixed list of surface strings, encoding tricks (cipher,
translation, role-play) that reword the same disallowed request still land inside
the disallowed region. A fixed-head toxicity classifier trained on observed toxic
strings generalizes poorly to a rewording it never saw. The lineage is Constitutional
AI (Anthropic, 2022), which first used a constitution to supervise model behavior;
the classifiers apply the same idea to the guard. The cost is an explicit
safety-throughput trade, roughly 23% compute overhead.

## Commonly answered wrong

**Q: Can you add the safety instructions to the system prompt and rely on that?**

A: No. System-prompt safety instructions are necessary and never sufficient. They
are the weakest layer because they are a suggestion the model can be talked out of.
They do not run independently of the model; they share all the model's failure
modes. Every safety answer that ends at "I added a strong system prompt" is missing
the classifier layer and the code-enforcement layer, which are the ones that hold
when the prompt fails.
**Why the prompt is persuadable by construction:** a system-prompt rule is
enforced only through the model's learned tendency to follow instructions, and
that same tendency is the attacker's interface; attack and defense compete inside
one next-token distribution, so a sufficiently crafted context can always shift
the balance. Classifiers and code gates sit outside that distribution, which is
the entire point of them.

**Q: If a jailbreak is detected on the input, you can skip the output guard, right?**

A: No. A benign prompt can still produce a harmful completion. The input guard and
the output guard are checking different things at different points. Some attacks are
undetectable on the input and only become visible in the output. A safe input
verdict does not license skipping the output check; they are independent gates.
**Why a clean input can still yield a harmful output:** generation is stochastic
and compositional, so harm can be assembled at generation time from pieces no
input filter could flag (an innocuous question plus retrieved context, or a
request spread across turns so no single message crosses the line); the harmful
artifact literally does not exist until the model produces it, so only a check
that runs after production can see it.

**Q: Should the safety layer block aggressively to maximize catch rate?**

A: Only if you are willing to route legitimate users away from the product.
Over-blocking is a real failure mode. The correct framing is to choose an operating
point on the safety-helpfulness frontier, not to maximize catch rate in isolation.
Measure and report the false-refusal rate on benign traffic alongside the catch rate
on adversarial traffic. Anthropic explicitly held the FRR increase to 0.38%
production; that constraint was as important as the jailbreak reduction.
**Why base rates make this unforgiving:** benign requests outnumber attacks by
orders of magnitude in production, so even a small false-refusal rate means the
absolute count of blocked legitimate users can dwarf the count of blocked
attackers; a threshold chosen by catch rate alone optimizes the rare class while
silently taxing the common one.

**Q: Does making the system prompt longer and more detailed increase safety?**

A: Marginally, at best. A longer system prompt raises the cost of low-effort attacks
by making them specify more to override. It does not prevent a sophisticated attack,
and it is not a substitute for independent classifiers or code-side gates. Investing
in classifier quality and structural isolation produces much larger safety gains than
investing in system-prompt engineering.
**Why length cannot change the outcome:** every added instruction still lives in
the same channel the attacker writes to and is enforced by the same persuadable
next-token process; more text raises the effort to override it but never changes
the enforcement mechanism from persuasion to code, and only a mechanism change
stops a sufficiently patient attacker.

**Q: A high-accuracy prompt-injection classifier means you can safely put retrieved
content in the prompt, right?**

A: No. A classifier is probabilistic and adversarial input is adaptive: an attacker
iterates until they find a phrasing the detector scores as benign, and encoding or
translation tricks widen that space. The structural fix is orthogonal to detection:
keep retrieved text in a data slot the model is told is untrusted (spotlighting or
delimiting) so it is never interpreted as instructions, and gate every real action
in code so an injected command cannot execute even if it slips past the detector.
Prompt Guard (Meta) and similar detectors raise the attacker's cost but are a layer,
not a guarantee; the code-side action gate is the one that holds when detection fails.
