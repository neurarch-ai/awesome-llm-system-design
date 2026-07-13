# 1. Clarifying the requirements

Before designing anything, pin down what the system must defend against and who it
serves. Here is a typical exchange. Notice that every question either removes work
or changes the design entirely.

**Candidate:** What are we protecting against? Harmful generations, policy
violations, PII leakage, or manipulation like prompt injection and jailbreaks?

**Interviewer:** All of the above, but weight them by risk. Harmful generations and
jailbreak resistance are the primary concern. PII leakage and prompt injection are
close seconds, especially since this product uses RAG.

**Candidate:** What is the trust boundary? Does untrusted content arrive only from
users, or also from retrieved documents and tool output?

**Interviewer:** Both. Retrieved documents and tool results arrive alongside user
messages, and we do not control their content.

**Candidate:** Who is the end user, and what are the stakes? Consumer at scale,
internal tooling, or a regulated domain like health or finance?

**Interviewer:** Consumer at scale. Think millions of requests per day, not
thousands. No specific regulation, but brand risk is high.

**Candidate:** What is our latency budget for the safety layer? A real-time chat
product can tolerate far less overhead than an async pipeline.

**Interviewer:** We need to stay under 100ms added latency at p50, and absolute
blocking on clearly harmful inputs. Throughput is tens of thousands of requests per
second at peak.

**Candidate:** What is the acceptable false-positive rate? Every blocked legitimate
request is a user who routes around the product.

**Interviewer:** We care a lot about this. We would rather miss a borderline case
than block five innocent users. Track both the catch rate and the benign-refusal
rate.

Let us summarize the problem statement. **We are designing a layered safety system
for a consumer LLM product with a RAG component.** The threat surface includes
user jailbreaks, indirect prompt injection over retrieved content, harmful
generations, and PII leakage. The system must stay under 100ms added latency at
p50, handle tens of thousands of requests per second, minimize false positives, and
log every block decision for audit and tuning.

Two consequences fall out of this immediately:

- **The trust boundary extends beyond the user.** A product that ingests retrieved
  documents or tool output faces a much larger attack surface than a pure chat
  interface. Injection that arrives via a document is harder to catch than a user
  who types "ignore previous instructions." This changes the design: structural
  isolation and code-side action gates become load-bearing, not optional.

- **No single guard is trusted.** At tens of thousands of requests per second with
  a 100ms budget, a single large guard model on the critical path is already too
  slow and too brittle. Defense must be layered: cheap deterministic checks first,
  then a fast classifier, and reserve a larger model for the fraction of inputs that
  survive the cheap tier. The cascade is not a performance trick; it is how you
  build a system that fails closed without eating the latency budget.
