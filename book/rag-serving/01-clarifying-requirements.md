# 1. Clarifying the requirements

Before designing anything, pin down what the system must do. Here is a typical
exchange between a candidate and an interviewer. Every question either removes
work or changes the design.

**Candidate:** Who uses this, and at what volume? Is this customer-facing or internal?
**Interviewer:** Internal tool. About 10,000 employees. Peak load is around 20 queries per second.

**Candidate:** What is the corpus? How large, and what kinds of documents?
**Interviewer:** Internal knowledge base: wikis, tickets, design docs. Roughly 50 million documents.
Length varies wildly, from one-line tickets to 80-page design documents.

**Candidate:** What is the latency budget?
**Interviewer:** Interactive chat. Users expect the first token within about 1.5 seconds at p99.
A full answer in a few seconds is fine.

**Candidate:** How fresh must answers be?
**Interviewer:** A new document should be answerable within an hour of creation. A daily batch
would be too slow.

**Candidate:** What is the quality bar? Are citations required?
**Interviewer:** Every answer must cite its source documents. Abstaining when the system cannot
find relevant content is better than generating a confident but wrong answer.

**Candidate:** Are there access-control requirements? Can any employee ask about any document?
**Interviewer:** No. Employees can only see documents they have permission to access. If someone
cannot read a design doc, they must not receive an answer sourced from it.

**Candidate:** Are there cost constraints? Is this charged per query or covered by a flat
infrastructure budget?
**Interviewer:** Internal use is unlimited. Cost per query needs to stay reasonable, but the
primary concern is quality and latency.

---

Let us summarize the problem statement. **We are asked to design a question-answering system
over 50 million internal documents, serving 10,000 employees at a peak of 20 QPS, with p99
first-token latency under 1.5 seconds, freshness under one hour, grounded cited answers,
and per-user access control.**

Three consequences fall out of this immediately, and stating them early is most of the
signal in this question:

- **Access control must be enforced inside retrieval, not after.** If you post-filter the
  top-k results by permission, you both leak (a user can infer a document exists from an
  abstention) and starve recall (when the user's visible set is small, a post-filter can
  empty the candidate list). The ACL must travel with every chunk through the index so the
  search itself returns only visible documents.

- **Exact nearest-neighbor search over 50 million chunks per query is too slow.** At
  1.5-second first-token latency, the vector search must complete in tens of milliseconds.
  That rules out brute-force comparison and forces an approximate nearest-neighbor index.
  The index choice, and how ACL filters integrate with it, is a core design decision.

- **Retrieval quality sets the ceiling on answer quality.** If the relevant chunk is never
  in the top-k, no generator downstream can recover it. The question "is recall low?" is
  always asked before "is the generator weak?"
