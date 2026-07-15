# 2. Frame the system

## Where the tokens and dollars go

An LLM API bill has three terms. Which one dominates determines which lever you
pull first.

| Cost driver | What it means | Who it hits hardest | First lever |
|---|---|---|---|
| Input tokens | Tokens in the prompt: system message, retrieved context, conversation history | RAG systems (long retrieved chunks), multi-turn chat (growing history), agents (long tool outputs) | Trim context, compress prompt, cache whole responses |
| Output tokens | Tokens generated in the response | Long-form generation, agents writing code or plans | Smaller model for short answers, streaming truncation, structured output |
| Request count | Number of API calls, regardless of length | High-QPS classification, embeddings re-generated per request | Route to a smaller model, cache exact or semantic hits, batch |

Read the bill. Optimizing input when output dominates, or output when the bill
is driven by request volume, is free work that saves nothing.

## The system in one picture

```mermaid
flowchart LR
  subgraph Inputs["what flows in"]
    SYS["system prompt<br/>(static, shared)"]
    CTX["retrieved context<br/>(20 chunks today)"]
    HIST["conversation history"]
    Q["user query"]
  end
  subgraph Levers["cost levers (applied left to right)"]
    CACHE{"cache?<br/>exact or semantic"}
    TRIM["trim context<br/>+ rerank to top-3"]
    COMP["compress<br/>low-info tokens"]
    ROUTE{"route<br/>or cascade"}
  end
  subgraph Models["model tier"]
    SMALL["small cheap model<br/>classify / lookup / short answer"]
    BIG["frontier model<br/>reasoning / code / hard generation"]
  end
  SYS --> CACHE
  CTX --> TRIM
  HIST --> TRIM
  Q --> CACHE
  CACHE -->|"miss"| TRIM
  TRIM --> COMP
  COMP --> ROUTE
  ROUTE -->|"easy"| SMALL
  ROUTE -->|"hard"| BIG
```

**How it works.** Four things flow in from the left: a static shared system
prompt, retrieved context, conversation history, and the user query. They meet a
stack of cost levers applied left to right, each one shrinking the token bill
before the next runs. The cache is checked first, keyed on the stable prompt and
the query; a hit short-circuits everything and returns a stored response, while a
miss falls through to trimming the retrieved context down to a reranked top-3.
The trimmed prompt is then compressed to drop low-information tokens, and only
then does a router or cascade decide the destination tier. Easy traffic lands on
the small cheap model and hard traffic on the frontier model, so by the time any
token crosses the API boundary the volume has already been cut at every prior
stage.

The interesting decisions are all **upstream of the model call**: whether to
serve a cached response, how many tokens reach the model, and which model they
reach. By the time a token crosses the API boundary you have already paid for
it.

## A practical cost model

Let the per-request expected cost be:

$$\mathbb{E}[C] = h \cdot c_{\text{hit}} + (1-h)\bigl(c_{\text{embed}} + f_{\text{small}} \cdot c_{\text{small}} + f_{\text{big}} \cdot c_{\text{big}}\bigr)$$

where $h$ is the cache hit rate, $c_{\text{hit}}$ is the (tiny) cost of a cache
lookup, $c_{\text{embed}}$ is the cost of computing the query embedding (a numeric
vector standing in for the query's meaning) on a miss, and $f_{\text{small}} +
f_{\text{big}} = 1 - h$ is how the remaining traffic splits across the two tiers.

```python
def expected_cost(h, c_hit, c_embed, f_small, c_small, f_big, c_big):
    # h: cache hit rate; a hit pays only the tiny lookup cost c_hit
    # on a miss (prob 1-h) we embed the query, then split across the two tiers
    miss = c_embed + f_small * c_small + f_big * c_big
    return h * c_hit + (1 - h) * miss
# f_small + f_big must equal 1 - h; e.g. expected_cost(0.5, 0.25, 0.5, 0.25, 2.0, 0.25, 8.0) -> 1.625
```

The formula shows the order to attack: raise $h$ (caching) before tuning
$f_{\text{small}}$ (routing), and reduce $c_{\text{small}}$ or
$c_{\text{big}}$ (compression, right-sizing) only after the routing fractions
are good.

## What this chapter builds next

The next four sections are one lever each, in the order you would typically
apply them:

1. **Routing and cascades** (section 03): send easy queries to the cheap tier
   before the expensive one ever fires.
2. **Caching and compression** (section 04): eliminate or shorten the prompt
   before any model sees it.
3. **Right-sizing** (section 05): ensure the "cheap tier" is as cheap as it can
   be for the task, through model selection, quantization, or distillation.
4. **Serving and gateway** (section 06): make all of the above enforceable,
   observable, and resilient in production.

They compose. A request first hits the cache, then is compressed, then routed;
the fractions shrink at each step so the frontier model only touches the hard
tail.
