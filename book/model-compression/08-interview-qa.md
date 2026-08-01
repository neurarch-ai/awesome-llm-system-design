# 8. Interview Q&A

The questions actually asked about compression, grouped by how they are used. The
traps at the bottom all share one shape: a number that is true about the *tensor*
and false about the *system*.

## Commonly asked

**Q: Our model is too big and too slow. Walk me through how you would approach it.**

A: Find the binding resource first, because the four levers move different
resources. Memory ceiling means weight quantization, starting at int8 per-channel
and moving to int4 group-wise if needed. Small-batch decode latency means the same
lever, since decode is bandwidth bound and time per token tracks bytes read.
Prefill or high-batch throughput means FLOPs are the cost, so weight-only tricks do
almost nothing and I need a compute-side format (fp8) or a structurally smaller
model. Long-context serving cost means the KV cache dominates, so the order is
architectural (GQA or MLA), then paged, then quantized KV. Before any of it I check
the free levers, continuous batching, prefix caching, and speculative decoding,
which buy speed without spending quality. Then the acceptance test: measured latency
and memory on the target hardware, and a paired per-item quality comparison against
the uncompressed model, per capability.

**Q: Weight-only versus weight-and-activation quantization: when do you need each?**

A: Weight-only shrinks the model and speeds up the bandwidth-bound decode phase, and
it is nearly lossless at int8 and workable at int4 with group-wise scales. It does
not change the arithmetic: the kernel dequantizes and multiplies in higher precision,
so FLOPs are unchanged and prefill is unaffected. To make the compute itself cheaper
you have to quantize activations too, which is much harder because activation
distributions have extreme outlier channels. That is why fp8 is attractive on
hardware that supports it: the floating exponent tolerates outliers far better than
int8 at the same width, and it gives a real speedup in both phases.

**Q: Why is activation quantization so much harder, and what do you do about it?**

A: A handful of activation channels carry values orders of magnitude larger than the
rest, and they matter functionally, so a scale wide enough to represent them leaves
almost no resolution for everything else. Four families of fix: keep the outlier
channels in high precision and decompose the matmul (LLM.int8), migrate the outliers
into the weight tensor with per-channel scaling (SmoothQuant), protect the salient
weight channels identified by activation magnitude before rounding (AWQ), or rotate
the basis with an orthogonal transform so no coordinate is an outlier and quantize
in the rotated space (QuaRot, SpinQuant). The rotation family is what made 4-bit
activations and 4-bit KV practical, and the operational requirement is that the
rotation is fused into adjacent matrices at export so it costs nothing at runtime.

**Q: Unstructured, 2:4, or structured pruning. How do you choose?**

A: By what the hardware can skip work for. Unstructured sparsity gives the best
quality at a given sparsity level and no speedup on dense matmul units, so it is a
memory technique unless you have sparse storage and kernels. The 2:4 pattern exists
because sparse tensor cores can actually skip it, and it costs more quality than
unstructured at the same nominal sparsity precisely because the constraint is rigid.
Structured pruning removes heads, channels, or layers and produces a smaller dense
model that is fast everywhere, at the cost of needing a healing run measured in
billions of tokens. If I can retrain, structured plus distillation from the parent
is the strongest option; if I cannot, quantization usually beats one-shot pruning on
quality per unit of resource saved.

**Q: How do you evaluate a compressed model?**

A: Not with a benchmark average. Compression scatters behavior rather than shifting
it, so two models can post the same accuracy and disagree on a large fraction of
individual items. The test has four parts: paired per-item comparison against the
uncompressed baseline with a reported flip rate; per-capability slices, since damage
concentrates (code, tool-call and JSON validity, long-context retrieval,
multilingual, refusal behavior are the usual casualties); measured latency, memory,
and throughput on the target hardware at production batch and context length, not in
a batch-one harness; and a shadow run on live traffic diffing outputs against the
current model, which catches format regressions offline sets miss.

**Q: What do you compress for a long-context workload?**

A: The KV cache, because it is the term that grows with context and batch while the
weights are fixed. The order is architectural first (a model with grouped-query or
latent attention beats compressing a many-KV-head model after the fact), then paged
allocation so the saved memory becomes batch size instead of fragmentation, then
quantization of the cache itself, keys per channel and values per token because
their outlier structure differs, then eviction or windowing last since it changes
what the model can attend to. Compressing weights while the KV dominates is solving
the wrong half.

**Q: How does the answer change for on-device?**

A: Three constraints replace the server ones: a hard memory ceiling shared with the
OS, an accelerator that only goes fast on a short list of formats, and a batch size
of one. Batch one is the best case for weight-only quantization, so the per-token
win is maximal. But the hard ceiling usually means a heavily compressed large model
still does not fit, so the shipped pattern is a small model distilled for the target
size, quantized to a format the NPU accelerates, with task-specific adapters swapped
over one resident base, plus a server model for requests that exceed it.

## Tricky (the follow-ups that separate people)

**Q: You want to prune, distill, and quantize. What order?**

A: Prune, heal or distill, then quantize, and re-evaluate the composed artifact
rather than the steps. The reasoning is that each step should be applied to a model
that is already in its final structural form: quantization scales fit the weights
they see, so quantizing before pruning fits scales to weights you are about to
delete, and a healing run after quantization would fight the quantizer unless it is
quantization-aware. The other ordering rule is that the healing step wants
full-precision gradients, so it belongs before the final quantization. If you need
both aggressive bit width and structural change, the composed path is prune, distill
with the parent as teacher, then quantization-aware training for the last step.

**Deeper:** The reason "evaluate the composition" is not pedantry: each step passes
its own gate by a small margin, and the errors are not independent, since both
pruning and quantization concentrate damage on the same outlier-sensitive paths.
Two levers that each cost one point can cost four together.

**Q: You quantized to int4, benchmark averages held, and users report broken JSON in
tool calls. What happened and what do you do?**

A: This is the canonical compression failure: the damage is real and concentrated in
exactly the capability that an average hides. Structured output depends on the model
placing high probability on a small number of exact tokens, and low-bit quantization
flattens those margins, so the format breaks before the semantics do. Diagnosis: run
a paired per-item comparison on a tool-call set, report validity rate rather than a
quality score, and compute per-layer sensitivity. The fix is almost never to revert
the quantization; it is to raise precision on the layers that matter, typically the
output projection, the embeddings, and the first and last blocks, and re-test. A
constrained decoder or grammar is a complementary mitigation but should not be used
to paper over a regression you have not measured.

**Q: Where do you draw the calibration set, and how would a bad one show up?**

A: From the serving distribution, including the shapes you care about: long contexts,
code, tool-call formats, and every language you serve. Post-training quantization
fits scales to the activations it observes, so a calibration set of generic web text
produces scales that misfit your actual traffic. It shows up as a model that looks
fine on public benchmarks and degrades specifically on your workload, which is the
hardest failure to attribute because the public numbers exonerate the artifact. A few
hundred well-chosen samples are enough; representativeness matters far more than
volume.

**Q: Someone proposes QLoRA to quantize the model for serving. Is that right?**

A: No, and the confusion is worth untangling. QLoRA quantizes the base to 4-bit in
order to *fine-tune* cheaply, training a higher-precision adapter over a frozen
quantized base. It is a training-memory technique. Post-training quantization (GPTQ,
AWQ, fp8 recipes) is what produces a serving artifact, and quantization-aware
training is what you do when PTQ will not hold quality at the bit width you need,
usually combined with a distillation loss from the full-precision teacher. The
related practical trap is that adapters trained against a full-precision base do not
always transfer cleanly to the quantized artifact, so adapters should be trained or
at least validated against the model you will actually serve.

**Q: How do you decide which layers keep higher precision?**

A: Measure, do not guess. Per-layer sensitivity analysis: quantize one layer (or
block) at a time, hold the rest at full precision, and record the delta on a small
but capability-diverse eval set. The result is a ranking, and you spend your memory
budget from the top of it. Priors worth starting from, because they hold across
models: embeddings and the output projection are a small fraction of parameters and
a large fraction of the damage, the first and last transformer blocks are usually
more sensitive than the middle, and norms and biases are tiny so they always stay
high. The output is a mixed-precision profile whose effective bit width sits above
the nominal one, which is the normal shape of a shipped artifact.

**Q: A compressed large model, or a smaller model from the same family. How do you
choose?**

A: Compare them as two candidates on the same acceptance test rather than by
parameter count. The smaller model was trained to be that size, so its capacity is
allocated coherently and it has no outlier-sensitivity damage; the compressed large
model retains more of the parent's knowledge and, importantly, the parent's
behavioral quirks, which matters when prompts and evals were tuned on the parent.
Rules of thumb: at moderate compression the compressed parent usually wins on
quality; at aggressive compression, or when the target format is fixed by the
hardware, the natively small model wins; and if you need behavioral compatibility
with the parent, distilling the parent into the small model gets you both.

## Commonly answered wrong (the traps)

**Q: Going from fp16 to int4 will cut our serving cost roughly 4x. True?**

A: No. It cuts weight *bytes* about 4x, which nearly translates to per-token decode
latency at small batch because that phase is bandwidth bound. It does not cut FLOPs
at all, so prefill is unaffected, and as batch size rises the decode phase amortizes
weight reads across many rows and drifts toward compute bound, where the win shrinks
toward nothing. Production serving cost is dominated by throughput at the batch size
you actually run, so the honest answer is to measure at that batch. If cost per
million tokens at high batch is the target, the lever is a compute-side format like
fp8 or a genuinely smaller model.

**Deeper:** There is also a floor you can hit in the wrong direction: an unfused
dequantize-then-matmul path can make the quantized model *slower* than the original,
because you added work per weight without removing any arithmetic. Whether a fused
kernel exists for your shape, runtime, and hardware is a more important question than
which quantization algorithm you picked.

**Q: We pruned 50 percent of the weights, so inference should be about 2x faster.**

A: Only if the sparsity has a shape the hardware can skip. Unstructured zeros are
multiplied like any other number on a dense matmul unit, so a 50 percent unstructured
model runs at exactly the original speed and only saves memory if you also store it
sparsely. The 2:4 semi-structured pattern is what sparse tensor cores accelerate, and
structured pruning gives a real speedup everywhere because the matrices are literally
smaller. Whenever someone quotes a sparsity percentage, the first question is which
of the three it is, and the second is what the measured latency was.

**Q: The compressed model matches the original on our benchmarks, so the compression
was free.**

A: Matching averages is not matching behavior. Compressed models routinely preserve
top-line accuracy while disagreeing with the baseline on a substantial fraction of
individual items: the wins and losses cancel in the mean. That matters because your
prompts, few-shot examples, guardrails, and downstream evals were tuned against the
original's behavior, so flips break things even when the average holds. Report the
paired flip rate and the per-capability slices, and treat divergence from the parent
as a cost even where accuracy is unchanged.

**Deeper:** The measurement to add is cheap: on a fixed set, record the baseline and
compressed outputs per item and compute the disagreement rate, plus a distributional
distance on the output probabilities where you have them. A compression that holds
accuracy with a low flip rate is genuinely safe; one that holds accuracy with a high
flip rate is a different model wearing the same score.

**Q: 1.58-bit models exist, so we can quantize ours to 2 bits.**

A: Those models are *trained* in that regime; the weights learn to be ternary during
pretraining. Post-training quantization to 2 bits or below degrades sharply, because
the network never learned to be robust to that quantizer. The path to very low bit
widths on an existing model is quantization-aware training with a distillation loss,
which costs a training run, and even then 4-bit weights with group-wise scales plus a
mixed-precision profile is where most production stacks stop.

**Q: Distillation always beats training the small model directly.**

A: Not always. Distillation wins when the data budget rather than compute is the
constraint, since soft targets carry far more information per token than hard labels,
when the teacher is meaningfully better than anything you could train directly, or
when behavioral compatibility with the parent matters. It is a wash or worse when the
student's capacity is far below the teacher's, because the mass-covering objective
spreads a small student thin across modes it cannot represent, and it costs teacher
inference throughout training. Two side effects to state: a student distilled from
long reasoning traces inherits their length and therefore your latency and cost, and
a student inherits the teacher's contamination, which no cleaning of your own corpus
can undo.

**Q: For long context, quantize the KV cache first.**

A: It is rarely first. The largest and most durable KV reduction is architectural,
grouped-query or latent attention cuts the cache by a fixed multiplicative factor
with modest, recoverable quality cost, so if the model choice is open that comes
first. Then paged allocation, which is free and converts the saved memory into usable
batch size rather than fragmentation. Then quantization of the cache, which is close
to free at 8-bit and needs per-capability testing at 4-bit. Eviction and windowing go
last because they change what the model can attend to, which is a capability change
rather than a representation change.

**Q: We measured a 2x speedup in the eval harness, so we are done.**

A: Harnesses usually run batch one with short prompts, which is the single most
favorable configuration for weight-only quantization and the least representative of
production. Re-measure at the batch size, context length, and concurrency you
actually serve, on the target hardware, with the runtime and kernel you will deploy.
Also assert the numeric path at startup: several stacks silently fall back to a
higher precision when a kernel is missing, which shows up as no speedup, no error,
and a confusing week.
