# 1. Clarifying the requirements

<!--
  Four to eight exchanges. The rule that makes this section worth reading: every
  question either removes work or fundamentally changes the design. A question whose
  answer changes nothing is filler, and a reviewer will ask you to cut it.

  The answers you write here are the chapter's fixed numbers. Every later section
  computes against them, so pick values you are willing to do arithmetic with.
-->

Before designing anything, pin down what the system must do. Here is a typical
exchange between a candidate and an interviewer. Notice that every question either
removes work or fundamentally changes the design.

---

**Candidate:** The first question, the one whose answer decides the architecture
rather than a parameter?

**Interviewer:** The answer, specific enough to design against, with a number in it.

---

**Candidate:** The scale question. How much, how fast, how many?

**Interviewer:** The numbers the rest of the chapter will use.

---

**Candidate:** The constraint question. Budget, latency, hardware, or a policy
boundary?

**Interviewer:** The constraint, stated as a hard limit.

---

## What we are building

One paragraph restating the problem with every answer folded in, so a reader who
skipped the dialogue can still follow the chapter.

**Requirements**

- Functional: ...
- Non-functional, with numbers: ...
- Explicitly out of scope: ...
