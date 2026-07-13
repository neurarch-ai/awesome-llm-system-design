# 5. Evaluation

Vision-language models have no single metric that covers all tasks; the right
eval depends on what the model must do. Naming the right benchmark for the right
task is a concrete signal in an interview.

## Task-specific offline metrics

**Visual question answering (VQA accuracy).** For open-ended QA over natural
images, VQAv2 and TextVQA are the standard benchmarks. VQAv2 tests general
understanding; TextVQA tests reading text inside images. Report exact-match
accuracy against a held-out set.

$$\text{VQA accuracy} = \frac{1}{|Q|} \sum_{q \in Q} \mathbf{1}\!\left[\hat{a}_q \in \text{accepted}(q)\right]$$

where $\hat{a}_q$ is the model's answer and $\text{accepted}(q)$ is the set of
valid human answers for question $q$.

**Grounding and localization.** RefCOCO and Visual Genome test whether the model
can correctly identify or locate a referred region. A model that describes "the
dog on the left" while drawing a bounding box around the dog on the right scores
well on VQA but fails grounding.

**OCR and document understanding.** DocVQA and ChartQA test reading dense text and
structured visuals. These tasks require fine resolution; a model that scores 80%
on VQAv2 with 336px input may score 40% on DocVQA because the text is too small.

**Hallucination rate.** POPE (Polling-based Object Probing Evaluation) probes
whether the model claims to see objects that are not in the image. A model that
confidently describes hallucinated content is worse than one that says "I'm not
sure," even if it scores higher on VQA.

![VQA accuracy vs image-token budget across connector types](assets/fig-accuracy-vs-tokens.png)

*Illustrative VQA scores alongside representative image-token counts for different
connector designs. Fixed-cap connectors (BLIP-2, Flamingo) are cheapest but score
lower on detail-heavy tasks. MLP projectors with dynamic resolution reach higher
scores at proportionally higher token cost. Scores are illustrative composites from
published benchmarks.*

## Latency and cost metrics

**Time to first token (TTFT).** For interactive use, TTFT is the user-facing
latency that matters most. It measures from request receipt to the first generated
token and is dominated by prefill over the full image-plus-text sequence. Track
TTFT separately at each resolution tier.

**Cost per request.** Because image-token blowup is easy to ship and hard to
notice in offline evals, track compute cost per request explicitly. A model that
scores 3 points higher on VQAv2 but costs 4x more to serve is not always a good
tradeoff.

**Throughput under load.** Image requests are heterogeneous in size; a dynamic-
resolution model makes every request a different length. Measure tokens per second
at the p50 and p99, not just on a single short request.

## Online metrics

**User engagement.** In a product that shows VLM answers, do users rate answers as
helpful? Do they ask follow-up questions, or do they drop? Offline accuracy on
benchmarks does not perfectly predict user satisfaction.

**Refusal and hallucination rate in production.** Log cases where the model refuses
to answer or users flag an answer as wrong. Hallucination shows up here long before
it shows up in offline POPE scores.

## When to use which metric

| Reach for | When | Instead of |
|---|---|---|
| VQAv2 accuracy | General natural-image question answering | TextVQA or DocVQA when the task is not text-reading |
| TextVQA or DocVQA | The task requires reading text in images (receipts, charts, signs) | VQAv2 alone, which does not test OCR |
| POPE hallucination rate | You need to quantify how often the model invents content | VQA accuracy alone, which rewards hallucinated-but-confident answers |
| RefCOCO grounding | The model must locate or refer to specific objects or regions | Caption quality, which does not test spatial reference |
| TTFT at target resolution | Interactive serving with a latency contract | Offline accuracy only, which ignores serving cost |
| Cost per request alongside accuracy | Any deployment decision | Accuracy alone, since image-token blowup makes it easy to ship something correct but unaffordable |

The guardrail to state out loud: an offline VQA gain that doubles the image-token
budget should survive a cost analysis and an online TTFT check before it ships.
Accuracy and cost must both pass; optimizing one while ignoring the other is the
classic mistake.
