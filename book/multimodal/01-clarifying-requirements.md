# 1. Clarifying the requirements

Before designing anything, pin down what the system must do. Here is a typical
exchange between a candidate and an interviewer. Notice that every question either
removes work or changes the design, especially around the image-token budget.

**Candidate:** What modalities (the input types, such as image, video, audio, or
text) are in scope? Images only, or also video, audio, or documents?
**Interviewer:** Start with images. We can discuss video if we have time.

**Candidate:** What is the typical image resolution, and how many images per
request?
**Interviewer:** Assume one image per request for now, up to 1024 by 1024 pixels.
Multi-image is a stretch goal.

**Candidate:** What task are we solving? Visual question answering, captioning,
document OCR, or something else?
**Interviewer:** Primarily visual question answering: the user uploads a photo and
asks a natural-language question about it.

**Candidate:** Is this an interactive product (the user waits for the first token)
or batch enrichment (no latency contract)?
**Interviewer:** Interactive. We want first-token latency under 2 seconds.

**Candidate:** How much image detail does the model need to recover? A "what is in
this picture" task needs far less resolution than reading a receipt or a chart.
**Interviewer:** General understanding for now. Dense text and charts are a future
concern.

**Candidate:** What is the expected request volume, and is the mix mostly
image-bearing or also text-only?
**Interviewer:** Assume a mixed workload. Maybe 30 percent of requests carry an
image; the rest are text-only.

Let us summarize the problem statement. **We are asked to design a service that
accepts an image and a text prompt and returns a grounded natural-language answer
using a vision-language model.** The input is one image (up to 1024x1024) plus a
text question; the output is a streamed text response. First-token latency target
is under 2 seconds. The workload is 30 percent image requests, 70 percent
text-only.

Three consequences fall out of this immediately, and stating them early is most of
the signal in this question.

**The image-token budget is the whole cost story.** A 1024x1024 image with a
16-pixel patch size produces up to 4096 tokens. Those tokens land in the LLM's
prefill pass, which dominates first-token latency, and in the KV cache at every
layer. A text-only question might be 30 tokens; that same request with a
high-resolution image is 130x more expensive at prefill. Everything downstream
follows from this math.

**Resolution is a quality-cost knob, not a default.** For general visual QA, a
lower-resolution view recovers enough detail while cutting the token count
dramatically. The senior move is to pick resolution per task, not to max it always.

**The image-bearing and text-only workloads are different enough to warrant
separate infrastructure paths.** Routing text-only requests past the vision encoder
avoids paying for image infrastructure on 70 percent of traffic. This is a
straightforward win that a surprising number of designs miss.
