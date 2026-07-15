# 1. Clarifying the requirements

Before designing anything, pin down what the system must do. Here is a typical
exchange between a candidate and an interviewer. Notice that every question
either removes work or fundamentally changes the design.

---

**Candidate:** What level of autonomy do we need? Should the agent act on its
own, or does a human approve risky steps like refunds or account changes?

**Interviewer:** Fully automatic for low-risk actions like lookups and FAQ
replies. Risky writes, say a refund above \$50, should route to a human
approval queue.

---

**Candidate:** What tools does the agent have access to? Are they read-only
lookups, or do some change state in real systems?

**Interviewer:** Both. Read tools: fetch account details, look up order status,
search the policy knowledge base. Write tools: issue a refund, update order
status, send a reply email. Assume these are real, live calls.

---

**Candidate:** What is the latency tolerance? Is this a synchronous live-chat
experience, or can it run in the background?

**Interviewer:** Mixed. Simple tickets (lookup and reply) should resolve within
10 seconds. Escalations to human review can be asynchronous.

---

**Candidate:** What is the expected volume, and is there a cost ceiling per
ticket?

**Interviewer:** About 50,000 tickets per day. Assume a budget of roughly
\$0.10 per ticket all-in across all model calls.

---

**Candidate:** What does failure look like, and what is its cost? A wrong FAQ
answer is a minor annoyance; a wrong refund is a real financial loss.

**Interviewer:** Correct. Incorrect refunds are expensive. An unanswered ticket
that escalates correctly is fine. The unacceptable failure mode is the agent
issuing an unauthorized refund or taking an irreversible action with no record.

---

**Candidate:** Do we need auditability? Can we trace why the agent took a
specific action?

**Interviewer:** Yes. Every action must be logged with the reasoning that led
to it, for compliance and for debugging.

---

Let us summarize what we have. **We are designing a customer-support agent that
reads tickets, calls read tools freely, gates write tools behind schema and
policy checks, escalates high-risk actions to humans, resolves simple tickets
within 10 seconds, costs at most \$0.10 per ticket, and logs every step.**

Three consequences fall out immediately, and naming them early is most of the
signal in this question:

- **The tool surface splits cleanly into reads and writes.** Read tools are
  safe to call freely. Write tools touch real state (money, orders), so every
  proposed write must clear a deterministic code gate before executing. A prompt
  that says "only refund under \$50" is a suggestion; a code check is a
  guarantee. This single distinction prevents most agent disasters.

- **Cost per ticket is steps times growing transcript, not a flat per-call
  price.** At \$0.10 per ticket and 50k tickets per day, that is \$5,000 per day.
  A runaway 30-step loop on a single ticket can eat a disproportionate share.
  A hard step cap and token budget (a token is a chunk of text, roughly a few
  characters, that models read and are billed by) are not optional.

- **Latency and reliability pull in opposite directions.** More verification
  steps raise quality but add latency. The 10-second budget for simple tickets
  means verification must be lightweight (a code gate, not a second model call)
  on the fast path, with deeper checks only for the async escalation path.
