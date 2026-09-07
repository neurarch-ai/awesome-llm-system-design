# 1. Clarifying the requirements

Before designing anything, pin down what the system must do. Here is a typical
exchange between a candidate and an interviewer. Notice that every question either
removes work or fundamentally changes the design.

---

**Candidate:** What are we generating, and at what resolution? A 512px thumbnail and
a 1024px hero image differ by four times in latent area, and video is another order
of magnitude.

**Interviewer:** Text-to-image at 1024 by 1024. Assume we add a short video path
later, five seconds at 24 frames per second.

---

**Candidate:** Is this interactive or batch? A user watching a spinner and a nightly
catalogue job are different systems: one is a latency problem with a step budget, the
other is a throughput problem with a queue.

**Interviewer:** Interactive. Users click generate and wait. We want p95 under three
seconds, p99 under six.

---

**Candidate:** How many images per request? Most products show a grid so the user can
pick one.

**Interviewer:** Four variants per request. They are shown together.

---

**Candidate:** That is important, so let me say it back: four variants is one batch
through one denoising loop, not four requests. Do we own the model, or are we calling
a provider?

**Interviewer:** We serve open weights ourselves, with our own fine-tune and a set of
style adapters that users can select.

---

**Candidate:** What is the safety and provenance requirement?

**Interviewer:** Prompt filtering before generation, a classifier on every output
before it is returned, and provenance metadata attached to every asset. All of that
counts against the latency budget.

---

**Candidate:** Last one, and it is the one that decides whether we can optimize at
all: what counts as "not visibly worse"?

**Interviewer:** Good question. Assume a paired human preference test on a fixed
prompt set of a few hundred prompts, and a change ships if it does not lose.

---

## What we are building

An interactive text-to-image service at 1024 by 1024, four variants per request,
p95 under three seconds, serving our own fine-tuned open weights with selectable
style adapters, with prompt filtering, output classification and provenance in the
request path, and a queued video path added later. Changes are gated on a paired
human preference test rather than on a distribution metric.

**Requirements**

- Functional: text-to-image, four variants, style adapters per request, image-to-image
  and inpainting on the same path, a queued video job.
- Non-functional, with numbers: p95 under 3 seconds and p99 under 6 for the grid,
  under 2 cents per generated image at target utilization, safety classification on
  every output, provenance metadata on every asset.
- Explicitly out of scope: training or fine-tuning the base model, and moving the
  quality bar. Quality here is a gate on changes, not a target.

**The one thing to carry into the next section.** Nothing in this list is priced in
tokens. The unit of cost is a forward evaluation of the denoiser, and the first thing
to put on the board is how many of those the current system runs.
