# 6. Evaluation and scaling

Evaluation of a base model is judged differently from evaluation of a chat model,
and using the wrong metric is a classic mistake. There are two layers: the
training signal (loss), and the downstream capability (benchmarks). Both require
care.

## The training signal: perplexity and bits-per-byte

The primary training metric is the held-out loss, reported as **perplexity**:

$$\text{PPL} = \exp\!\left(\mathcal{L}\right)$$

where $\mathcal{L}$ is the mean negative log-likelihood per token on the
held-out set. Lower is better.

Mechanically, $\mathcal{L}$ is just the cross-entropy between the model's
next-token distribution and the actual next token, averaged over positions. The
one detail that trips people up is the **shift by one**: the logits at position
$t$ are scored against the token at position $t{+}1$.

```python
import torch.nn.functional as F
# logits: (batch, seq, vocab); targets: (batch, seq) of token ids
def loss_and_perplexity(logits, targets):
    logits = logits[:, :-1, :]          # drop the last position (no next token)
    targets = targets[:, 1:]            # the next token is the label (shift by one)
    ce = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
    return ce, ce.exp()                 # perplexity = exp(mean cross-entropy)
```

Perplexity is the exponential of that mean cross-entropy, so a perplexity of 10
means the model is on average as uncertain as if it were choosing uniformly among
10 tokens.

**Perplexity is only comparable across models that share a tokenizer.** A model
with a larger vocabulary emits fewer tokens per sentence (each token covers more
text), which mechanically lowers perplexity while the model may be no better on
actual tasks. This is the tokenizer fertility problem from the other direction.

**Bits-per-byte (BPB)** removes this artifact by normalizing by the number of
bytes in the text:

$$\text{BPB} = \frac{\mathcal{L}}{\ln 2} \cdot \frac{n_{\text{tokens}}}{n_{\text{bytes}}}$$

BPB is the tokenizer-invariant metric. Use it whenever you compare models with
different vocabularies. Serious pretraining papers report both.

## Benchmark evaluation

After (or alongside) loss-based evaluation, run the model on capability
benchmarks: MMLU (knowledge breadth), ARC-Challenge (reasoning), HellaSwag
(common sense), HumanEval (code generation), GSM8K (math reasoning).

**Use a time-based split, not a random split.** Hold out documents from a future
time window and evaluate whether today's model handles them. A random split
leaks the future and flatters the model.

**Decontaminate before reporting any benchmark number.** Any score without a
decontamination claim is suspect. The mature move is to report the contamination
rate you found and removed. A score that looks good only because the eval leaked
into training is a liability, not an asset.

## What evaluation does not tell you

A base model evaluation measures next-token prediction quality. It does not
directly measure instruction following, safety, alignment, or real-world
usefulness. Those require post-training evaluation (RLHF, DPO, red-teaming).
Pretraining evaluation sets the capability floor; post-training shapes how that
capability is expressed.

## Bottlenecks

| Bottleneck | First sign | Fix | Tradeoff |
|---|---|---|---|
| Dirty extraction | Duplicate counts explode; heuristic filters misfire | Re-extract from WARC; boilerplate removal; URL blocklists | Heavier pipeline than using WET plaintext directly |
| Near-duplicates | Memorization issues; eval leakage; wasted tokens | MinHash / LSH fuzzy dedup within and across dumps | Over-dedup strips valid common text; ablate aggressiveness |
| Low-quality web text | Benchmarks plateau despite more tokens | Heuristic filters plus learned quality classifier | Classifier bakes in reference bias; validate on downstream evals |
| Eval contamination | Benchmark spikes after a data refresh | n-gram and embedding decontamination; report the rate found | Some real gains removed; integrity over score |
| Tokenizer fertility | A language costs $3 \times$ the tokens per word | Fit vocab on the true mixture; size vocab per language fertility | Bigger embedding and softmax parameters |
| Memory wall | OOM before the model fits one GPU | Tensor parallel, ZeRO / FSDP sharding, activation checkpointing | Extra communication; lower MFU |
| Low MFU | GPUs waiting; actual throughput far below rated | TP in-node, many micro-batches for PP, overlap communications | Complex parallelism plan to tune and maintain |
| Hardware failures | Run dies every few hours | Frequent sharded checkpoints; elastic restart | Checkpoint I/O cost and storage |
| Loss spikes | Loss diverges mid-run at step N | Roll back to last good checkpoint; skip or reshuffle batches; lower LR | Lost steps; automated or manual intervention |
| Compute misallocation | Budget runs out before compute-optimal token count | Chinchilla sizing before committing the run; overtrain smaller if serving-heavy | Give up training-optimal for inference savings |
