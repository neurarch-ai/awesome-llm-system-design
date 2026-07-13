# 1. Clarifying the requirements

Before drawing boxes, pin down what the system must do. Every question below
either removes a whole design branch or forces a constraint that dominates the
rest of the chapter.

---

**Candidate:** What is being searched? Documents, products, code, images, or a
mix?
**Interviewer:** Text documents. Assume a general corpus, something like an
internal knowledge base or a product catalog. You choose the specifics.

**Candidate:** What is the scale? How many documents now, and at what growth
rate?
**Interviewer:** 100 million documents today, growing maybe 10% per quarter.

**Candidate:** What is the latency target for the search itself, not the full
answer pipeline?
**Interviewer:** Tens of milliseconds at p99 for a search call. Say 50ms.

**Candidate:** Is this feeding a human reading the results, or a downstream
ranker?
**Interviewer:** Both. Some surfaces show results directly to users; others feed
a downstream model. Design for the harder case.

**Candidate:** What recall bar do we need? If we miss a relevant document in
retrieval, can a later stage recover?
**Interviewer:** High recall is required. Miss rate in retrieval directly hurts
the final product.

**Candidate:** How fresh must new documents be? Searchable in seconds, minutes,
or is daily fine?
**Interviewer:** New documents should be searchable within minutes. Deletions
too.

**Candidate:** Are there structured filters alongside the semantic query? For
example, date range, category, author?
**Interviewer:** Yes, assume the system must support attribute filters combined
with semantic search.

**Candidate:** Is the query always natural language, or do we need to handle
exact-token queries like product SKUs, error codes, or names?
**Interviewer:** Both. Users type natural-language questions and exact codes.
That matters for design.

---

Let us summarize the problem. **We are building a semantic search service over
100 million text documents that returns the top-k most relevant results for a
query in under 50ms at p99, with high recall, attribute filters, support for
both natural-language and exact-term queries, and document freshness within
minutes.**

Three consequences fall out immediately, and naming them early is the signal
the interviewer is looking for.

**First, pure dense retrieval will miss exact-term queries.** An embedding model
maps text into a continuous vector space where "SKU-4821" and "product 4821"
may be far apart if those tokens were rare in training data. Lexical search
(BM25 or SPLADE) is required alongside the dense channel. The design must run
both and fuse them.

**Second, exact nearest-neighbor search over 100 million vectors per query is
too slow.** Linear scan at even 1 microsecond per vector is 100 seconds. The
design needs an approximate nearest neighbor index, and the index choice is the
central design decision: it is a three-way tradeoff between recall, latency, and
memory.

**Third, attribute filters must be push-inside the index, not applied as a
post-filter.** If 99% of documents fail a filter, a post-filter over retrieved
candidates wastes the whole query budget on items that will be discarded. The
index structure must support pre-filtering or native filter-aware search.
