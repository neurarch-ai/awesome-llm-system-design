# 1. Clarifying the Requirements

Before designing anything, pin down what the system must do. Every question below
either removes work or changes the design. Notice that none of them are about
tooling; they are all about the contract the monitoring system has to satisfy.

**Candidate:** What kind of LLM application are we monitoring? A RAG system, an
agent with tool calls, or a single-shot completion endpoint?

**Interviewer:** A RAG copilot. Each request retrieves a set of documents, builds
a prompt, and generates an answer that should be grounded in those documents.

**Candidate:** What is the cost of a wrong answer? Is a bad response a
three-in-the-morning page, or something we review in a weekly dashboard?

**Interviewer:** This is an enterprise support copilot. A confidently wrong answer
damages customer trust. We want to catch a quality regression within hours, not
days.

**Candidate:** How often do the model and prompt change in practice?

**Interviewer:** The prompt is edited several times a week. The model may be
swapped quarterly, but we want the ability to do it more often.

**Candidate:** What is our budget for observation? Running an LLM judge on every
request roughly doubles our inference spend.

**Interviewer:** We can spend up to fifteen percent of our serving cost on
monitoring. We cannot judge every call.

**Candidate:** Does any delayed ground truth arrive? For instance, does the
support agent accept or reject the copilot answer, or does a ticket get reopened?

**Interviewer:** Yes, the human agent either accepts and sends the answer or
discards it. That signal arrives with a five-to-thirty-minute lag.

---

Let us summarize the problem statement. **We are asked to design the observability
layer for a production RAG copilot** where every request retrieves documents and
generates a grounded answer. Answers must be traceable, quality must be estimated
continuously without pre-labeled production data, and a hallucination spike or a
regression after a prompt edit must surface within hours. Our observation budget is
around fifteen percent of serving cost, so we must sample.

Two consequences fall out of this immediately, and stating them early is most of
the signal in this question:

- **Production has no ground truth, so quality is always estimated, never
  measured.** Accuracy the way we computed it on the labeled eval set does not
  exist online. What we have instead are proxy signals: an LLM judge scoring
  faithfulness on a sample, a grounding check comparing each claim to the
  retrieved context we logged, and implicit user actions (accept, discard, retry)
  as a free behavioral signal. Every number on the dashboard is a proxy until it
  is calibrated against a human label.
- **The expensive checks cannot run synchronously on every request.** A
  fifteen-percent cost budget with an LLM judge that costs as much as the
  generation itself means judging at most fifteen percent of traffic. Traces must
  be emitted cheaply on the hot path, and every heavy check, the judge, the
  grounding scorer, the safety re-scan, the human review queue, runs
  asynchronously off that trace stream. This shapes every architectural decision
  in the chapters that follow.
