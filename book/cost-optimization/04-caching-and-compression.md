# 4. Caching and compression

The cheapest LLM call is the one you never make. Caching eliminates the call;
compression shrinks it. Both act upstream of the model and compose with routing.

## Semantic caching

An **exact cache** keys on the normalized prompt string (plus model and params)
and returns a stored response on a hit. Risk is zero (identical input, identical
answer) but hit rate is near zero on free-text input: users rarely type the same
sentence twice.

A **semantic cache** embeds the incoming request with a small embedding model and
retrieves the stored response closest in embedding space if that cosine similarity
clears a threshold $\tau$:

$$\text{serve cached response} \iff \max_k \cos(e_q, e_k) \ge \tau, \quad \tau \in (0,1)$$

This catches paraphrases ("what is your return policy" vs "how do I return
something") where real hit rate lives. The threshold $\tau$ is the whole game.
Too loose and you serve the answer to a different question (confidently, cheaply
wrong). Too tight and the hit rate collapses to exact match.

### The cache math

Let $h$ be the hit rate, $c_{\text{hit}}$ the cost of a cache lookup (small
embedding plus index query), $c_{\text{embed}}$ the embedding cost on a miss,
and $c_{\text{model}}$ the full model call cost. Expected cost per request:

$$\mathbb{E}[C_{\text{cache}}] = h \cdot c_{\text{hit}} + (1-h)(c_{\text{embed}} + c_{\text{model}})$$

Caching pays when savings exceed the embedding overhead, i.e., when hit rate
clears the break-even:

$$h^* = \frac{c_{\text{embed}}}{c_{\text{model}} - c_{\text{hit}}}$$

If $c_{\text{model}} = 1$, $c_{\text{hit}} = 0.005$, and $c_{\text{embed}} =
0.02$, then $h^* \approx 2\%$. Caching pays even at modest hit rates; the
question is whether the semantic threshold can deliver them without degrading
quality.

![Cache hit rate vs net cost savings, with break-even marked](assets/fig-cache-hit-savings.png)

*Net cost savings as a function of hit rate. The break-even hit rate $h^*$ is
low (around 2% for typical embedding-vs-model cost ratios), so caching pays at
modest hit rates. The profit zone grows roughly linearly; the constraint is
threshold quality, not break-even math. Illustrative.*

### What not to cache

- **Personalized or tenant-scoped answers.** A shared cache that keys on query
  text alone can serve user A's response to user B. Scope the cache key or skip
  caching for any response that contains private context.
- **Volatile facts.** "What is today's stock price?" should have TTL 0. Cache
  stable content (definitions, policies); TTL volatile content aggressively.
- **Long-tail free-text at a loose threshold.** If $\tau$ is low enough to catch
  paraphrases, it is also low enough to return a near-neighbor's answer to a
  genuinely different question. Tune $\tau$ on labeled should-hit / should-not
  pairs, not by raw hit rate.

## Prompt compression

You pay per token. Tokens the model did not need are money burned. Two moves,
each appropriate in different regimes.

### Context trimming

The blunt, safe move: send fewer retrieved chunks. Most RAG pipelines over-retrieve
(top-20 by default) and the bottom 17 chunks add noise, not signal. A good
reranker (cross-encoder, colBERT) scores the 20 retrieved chunks and keeps only
the top 3 most relevant. This is often free quality-wise and directly cheaper,
because the answer already lived in the top chunks and the rest were padding. Try
trimming before any compression algorithm: it is zero-risk and the savings can be
large (17/20 chunks = 85% context reduction if you were over-retrieving that
badly).

### LLMLingua-style token compression

The sharper tool: a small LM scores each token by its perplexity (how surprised
the small LM was to see it) and drops low-information tokens, yielding a shorter
prompt the large model still understands. Microsoft Research's LLMLingua uses a
coarse pass (whole sentences) followed by a fine pass (individual tokens) with a
distribution-alignment step to match the target LLM's language patterns. On RAG
benchmarks it reaches up to 20x compression with about 1.5 points of quality
loss.

![Prompt compression: quality vs tokens saved at increasing compression ratio](assets/fig-compression-quality.png)

*Quality versus tokens saved as the compression ratio rises. Trimming (rho
around 2-3) is safe; moderate LLMLingua compression (rho around 5) captures most
of the savings; aggressive compression (rho \gt 10) risks dropping the
load-bearing token. Illustrative.*

The net-win condition: compression only pays when input tokens dominate the bill
and the context is long and redundant enough that the small-LM pass costs less
than the tokens it removes:

$$\text{net win iff} \quad c_{\text{big}} \cdot (n_{\text{orig}} - n_{\text{comp}}) \gt c_{\text{small}} \cdot n_{\text{orig}}$$

which simplifies to: the per-token saving from removing tokens must exceed the
per-token cost of the compression pass, weighted by how many tokens survive. On
short prompts or output-dominated workloads, the small-LM pass is pure overhead.

### What never to compress

Back off compression on any task where a single dropped token changes the answer:
exact extraction, legal or compliance text, code, citations. The compression is
lossy and aggressive ratios can remove the exact detail the answer hinged on.
Gate the compression ratio behind the same quality eval you use for every other
lever.

## When to use which

| Reach for | When | Instead of |
|---|---|---|
| Exact cache (hash on body) | Identical requests recur (fixed prompts, shared system messages); zero tolerance for near-neighbor errors | Semantic cache, when you need its broader coverage and can afford threshold tuning |
| Semantic cache (embed + threshold) | Free-text repeats or paraphrases over stable content (definitions, policies) | Exact cache alone, which almost never fires on varied natural language |
| Context trimming (fewer chunks) | RAG pipeline retrieves too many chunks and the bottom ones are noise | LLMLingua, which is overkill if simple top-k reranking already solves it |
| LLMLingua compression | Input tokens dominate and context is long, verbose, and redundant (not short or output-heavy) | Trimming alone, when verbose redundant text within each chunk is the problem |
| Skip caching | Personalized, scoped, or volatile answers | Sharing a cache across users/tenants for scoped content (data leak) |

**Tools.** GPTCache is the reference open-source semantic-cache layer and supports both exact key hashing and embedding-plus-threshold lookup with pluggable vector stores; a plain Redis or in-process dictionary keyed on the normalized prompt covers the exact-cache case. Embeddings for the semantic path come from sentence-transformers or a hosted embedding API, backed by a vector index such as FAISS (Meta) or a managed store. For compression, LLMLingua and LLMLingua-2 (Microsoft Research) implement the perplexity-based coarse-to-fine token dropping, and context trimming is just a reranking step using a cross-encoder or ColBERT reranker from Hugging Face Transformers over the retrieved chunks.

**Worked example.** A document-AI team answers questions over company policy PDFs, where the same handful of policies get asked about in many different phrasings. An exact cache almost never fires because users paraphrase, so they add a semantic cache with an embedding model and a tuned similarity threshold to catch "what is the return window" against "how long do I have to send it back," scoping the cache key per tenant so one customer's answer never leaks to another. Their RAG stage over-retrieves twenty chunks, so before touching any compression algorithm they trim to the top few with a reranker, which is zero-risk and cheaper because the answer already lived in the top chunks. Only for the rare long, verbose contract passages that survive trimming do they reach for LLMLingua compression, and they disable it entirely on exact-clause extraction where a single dropped token would change the answer.
