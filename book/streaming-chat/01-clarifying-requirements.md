# 1. Clarifying the requirements

Before sketching any architecture, you need to know which constraints are real.
Here is a typical opening exchange. Every question either removes a whole design
branch or reveals a key constraint.

---

**Candidate:** Are we building a pure text chat interface, or is voice or
multimodal on the scope?

**Interviewer:** Text only for now. The product is a chat UI where the assistant
replies in real time as tokens arrive.

---

**Candidate:** What is the latency target? I want to know whether we are
optimizing the time before the first character appears (time-to-first-token) or
the time for the full reply.

**Interviewer:** Time-to-first-token. Users notice if the screen stays blank.
Aim for under one second at p95.

---

**Candidate:** How long are conversations expected to be? A few turns or
hundreds of turns? This drives the context-growth problem.

**Interviewer:** Mostly short, five to fifteen turns. But some power users run
long sessions, so we should not break at thirty or forty turns either.

---

**Candidate:** Does the server hold conversation state, or does the client send
the full transcript on every request?

**Interviewer:** Server-side state. The client sends only the new message.

---

**Candidate:** Multi-device and session resumability? If a user closes the tab
and reopens, do we reconstruct the session?

**Interviewer:** Yes. The session should survive a browser close and reopen on
the same or a different device, as long as they are signed in.

---

**Candidate:** What is the peak concurrent-stream count we are designing for?
Concurrent streams, not just requests per second, because each stream holds an
inference slot for its entire duration.

**Interviewer:** On the order of tens of thousands of concurrent streams at
peak.

---

**Candidate:** Should a user be able to cancel a generation mid-stream?

**Interviewer:** Yes, the stop button must work.

---

## What the dialogue tells us

Let us gather the answers into a tight problem statement, because stating the
consequences of each answer early is most of the signal in this question.

**Problem statement.** Design the serving and application layer for a
server-stateful, multi-turn text chat product. Clients connect and receive
token streams in real time; the server holds conversation context; users can
cancel mid-generation; sessions survive reconnects. The system must handle tens
of thousands of concurrent streams at peak and deliver p95 time-to-first-token
under one second.

Three consequences follow immediately from these answers, and they set up every
section that follows:

**Consequence 1: TTFT is the user-facing metric.** The gap before the first
token appears is what users feel as "fast" or "slow." Total generation time
matters for cost, but TTFT is what drives the streaming transport and the prefill
optimization work.

**Consequence 2: Concurrent streams are the capacity unit, not QPS.** A request
that takes 10 seconds holds an inference slot for 10 seconds. The binding
constraint on GPU capacity is how many simultaneous decodes are in flight, not
how many requests started in the last second. Every design decision about
backpressure, cancellation, and scheduling flows from this.

**Consequence 3: Server-side state creates a sticky-routing requirement.** Once
the server holds the transcript and the KV cache for a session, follow-up turns
must land on the same (or a nearby) replica to reuse that cache. Without
stickiness, every turn re-reads the full transcript from scratch, and the
cost-and-latency benefit of the server-side state disappears.
