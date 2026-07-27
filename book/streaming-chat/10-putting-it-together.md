# 10. Putting it together: the complete build

Sections 1 through 6 taught each stage with its options and tradeoffs; section 7
showed where real teams diverge. What none of them show is a single system with
every decision made. This capstone does three things: it gives you an opinionated
default stack so option paralysis never blocks a first build, it walks the
chapter's scenario end to end with every choice committed and sized, and it
shows how the same decisions flip when the constraints change. It closes with
the smallest runnable token stream, one file, no installs.

## The default stack: start here, deviate with reason

Every stage in this chapter has two to five credible options, and a first-time
builder can burn a week comparing transports before streaming a single token.
Skip that. The stack below is a sane default for a first production build; each
row names when to deviate and which section explains why. Frameworks change
yearly, but the interface of each stage (transport, session state, context
policy, backpressure, cancellation, recovery, overload) does not, so pick per
stage by interface and treat any specific tool as replaceable.

| Stage | Default | Deviate when | Why (section) |
|---|---|---|---|
| Transport | SSE over plain HTTP, one event per token, `id:` on every event | Client must signal mid-stream (barge-in, multiplexing): WebSocket. Voice audio: WebRTC over UDP | [2](02-the-streaming-model.md) |
| Session state | Server-side store: Redis hot path, Postgres durable backing | Simple bot, no long sessions or resumability: client-side transcript | [3](03-session-and-memory.md) |
| Context policy | Summarize oldest turns at a threshold | Early context genuinely irrelevant: sliding window of last k turns | [3](03-session-and-memory.md) |
| Prefill cost | Prefix caching plus session-id sticky routing, treated as best-effort | Single replica: stickiness is free and automatic | [3](03-session-and-memory.md), [6](06-serving-and-scaling.md) |
| Backpressure | Bounded per-stream buffer; block the decode loop for text | Hopelessly slow consumer: abort the stream. Audio: drop frames | [4](04-backpressure-and-concurrency.md) |
| Cancellation | Propagate cancel and disconnect to the inference engine; heartbeat on the SSE channel | Never. Orphaned slots are a capacity leak at any scale | [4](04-backpressure-and-concurrency.md) |
| Recovery | Write-only-on-completion; idempotency key on retries; resume via `Last-Event-ID` plus a short ring buffer | Replies run tens of seconds: checkpoint at sentence boundaries | [5](05-reliability.md) |
| Overload | Shed with HTTP 429 and `Retry-After` past a utilization threshold; fall back to a smaller model | Never queue silently. Unbounded queues amplify retries | [4](04-backpressure-and-concurrency.md), [6](06-serving-and-scaling.md) |

The cancellation row is the one beginners skip and regret: a stream that nobody
is reading looks like a no-op, but it pins an inference slot for its entire
decode duration. Wiring cancel-on-disconnect before launch is capacity
management, not hygiene.

## The complete build

Return to the scenario from [section 1](01-clarifying-requirements.md): a
server-stateful, multi-turn text chat product, tens of thousands of concurrent
streams at peak, p95 time-to-first-token under one second, sessions that survive
browser closes and device switches, and a stop button that works. Conversations
run five to fifteen turns, with power users reaching thirty or forty. Here is
the whole system with every choice committed and the reason it won.

| Decision | Choice | Why it won |
|---|---|---|
| Transport | SSE, one event per token, `id:` field on every event | Token delivery is one-directional; plain HTTP passes every proxy; resume-by-id comes with the standard |
| Session state | Server-side: Redis keyed by session id, Postgres behind it | Multi-device resumability rules out client-held history; the durable copy survives Redis eviction |
| Routing | Consistent hash on session id, gateway to inference replica | Prefix caching only pays when the follow-up turn finds its warm KV cache; hashing rebalances a fraction, not all, on scale-out |
| Context policy | Prefix caching for the stable head; summarize oldest turns at a threshold | Bounds per-turn prefill and keeps forty-turn sessions alive without hitting the context limit |
| Backpressure | Bounded per-stream buffer, block the decode loop, abort a hopelessly slow consumer | Dropping tokens garbles text; a small buffer surfaces dead consumers fast |
| Cancellation | Stop button and disconnect both propagate an abort to the inference engine; SSE heartbeat | Freed slots are the capacity unit; a heartbeat catches half-open TCP the socket never reports |
| Recovery | Write-only-on-completion, client request id for dedup, short token ring buffer for resume | Replies are short, so a lost generation is a cheap retry; the ring buffer makes network blips invisible |
| Overload | Shed with 429 plus `Retry-After` past the threshold; fall back to a smaller model under sustained spike | Visible degradation beats a silent hang; halving parameters roughly doubles decode speed |

**Concurrent streams per replica.** The capacity unit is concurrent decodes, not
QPS ([section 6](06-serving-and-scaling.md)). Illustrative: a GPU replica
sustaining ~3,000 tok/s aggregate under continuous batching, serving streams
that decode at ~30 tok/s each, carries about 100 concurrent streams. Peak load
of 20,000 concurrent streams then needs on the order of 200 replicas. The
fallback model is the cheapest lever on this number: half the parameters
roughly doubles decode speed, so the same fleet carries roughly twice the
streams during a spike, at a visible quality dip instead of a blank screen.

**Context growth per session.** At the chapter's illustrative average of 200
tokens per turn ([section 3](03-session-and-memory.md)), a fifteen-turn session
re-reads about 3,000 tokens of transcript at turn fifteen, and a forty-turn
power session about 8,000. Without prefix caching the cumulative prefill over
forty turns is the triangular sum, roughly 200 x 40 x 41 / 2, about 164,000
tokens for one conversation. With prefix caching and sticky routing the
marginal cost per turn collapses toward the new message alone, and
summarization at the threshold resets the transcript length so the tail of a
long session stops growing. This pair of decisions is most of the
infrastructure bill.

**TTFT budget.** The one-second p95 target decomposes into queueing plus
prefill plus transport ([section 2](02-the-streaming-model.md)). Illustrative
warm path: ~50ms network and gateway, single-digit ms for the Redis transcript
read, ~100-200ms prefill over the new message with a warm prefix cache,
near-zero admission wait below saturation; comfortably a few hundred ms. The
budget must also absorb the cold path: a rehashed or reconnected session pays
full prefill over its ~3,000-token transcript, several hundred ms more, still
inside one second. Past the saturation point none of this matters: TTFT is
determined by queue length, and the only levers are shedding, fallback, and
replicas.

**What breaks in month one.** Three failure modes dominate early operations, so
wire their signals before launch: prefix-cache hit rate (a dip that recovers
within a turn or two is rebalancing; a persistently low rate is a routing bug
sending the same session to a different replica every turn, see
[section 8](08-interview-qa.md)), orphaned streams (slot utilization high while
user QPS is low means disconnects are not propagating aborts, see
[section 6](06-serving-and-scaling.md)), and retry amplification (a TTFT spike
that triggers client retries deepens itself; watch queue depth against the shed
threshold and confirm 429s carry `Retry-After` and clients back off with
jitter).

## The same techniques under different constraints

The review question that matters in practice is not "SSE or WebSocket" but "SSE
or WebSocket under my constraints." Here is the same serving layer built three
times. Only the consumer column is the build above; the other two keep the
identical stage interfaces and swap nearly every implementation choice.

| | Internal tools bot | Consumer chat (this chapter) | Voice assistant |
|---|---|---|---|
| Traffic | A few hundred concurrent streams | Tens of thousands of concurrent streams | Thousands of concurrent calls, each a continuous audio session |
| Transport | SSE; nothing else earns its keep | SSE with `id:` events and a resume buffer | WebRTC over UDP; TCP head-of-line blocking stalls audio ([7](07-how-teams-do-it-in-production.md)) |
| Session state | One Redis, no stickiness; cold prefill on short transcripts is cheap | Redis + Postgres, consistent-hash sticky routing | Per-call state on the media server; turn detection is part of the session |
| Context policy | Sliding window of last k turns | Prefix caching + summarize at threshold | Short rolling context; the binding sum is L_STT + L_turn + L_LLM + L_TTS |
| Backpressure / overload | FIFO queue, block on slow consumers, no tiers | Bounded buffer, shed with 429, priority queue, model fallback | Drop late audio frames; smoothness beats completeness; eager end-of-turn overlaps dead time |
| Reliability | Write-only-on-completion plus an idempotency key | Ring-buffer resume via `Last-Event-ID`, dedup on request id | No replay: stale audio is worthless; reconnect restarts the turn |
| What would be over-engineering | Consistent hashing, priority tiers, model fallback, resume buffers | Checkpoint-at-sentence-boundaries for short replies | Resume-by-id token replay, blocking backpressure, any TCP transport |

Two lessons fall out. First, the internal-bot column is mostly deletions: at a
few hundred streams a single replica or two absorbs peak, cold prefill over
short transcripts costs milliseconds, and every affinity and tiering mechanism
is dead weight; only cancellation-on-disconnect survives, because orphaned
slots leak capacity at any scale. Second, the voice column shows the transport
and the backpressure policy flipping together: once frames must arrive on time
rather than in order, UDP replaces TCP, dropping replaces blocking, and the
latency war moves from prefill to the four-term pipeline sum of
[section 7](07-how-teams-do-it-in-production.md).

## What each constraint decides

The compressed decision guide. Read the left column off your requirements; the
right columns say which lever it moves before you compare any tools.

| Your constraint | Lever it moves | Rule of thumb |
|---|---|---|
| TTFT target | Prefix caching + sticky routing, prefill size | Under ~1s p95: warm-cache path mandatory for multi-turn; budget the cold path too, it fires on every rebalance |
| Peak concurrent streams | Replica count, batching, fallback tier | Streams per replica ~ aggregate tok/s over per-stream tok/s; halve the model to roughly double the ceiling in a spike |
| Session length distribution | Context policy | Short-memory product: sliding window. Long sessions users return to: summarize at a threshold, never grow unbounded |
| Mid-stream client signaling | Transport | Server-to-client only: SSE. Barge-in, live interrupt, multiplexing: WebSocket. Audio: WebRTC over UDP |
| Flaky client networks | Event ids + replay buffer | Tag every SSE event with `id:`, keep a short per-stream ring buffer; reconnect without it is a silent correctness bug |
| Multi-device resumability | Where state lives | Session id tied to the account, transcript in a durable store; the cache changes latency only, never correctness |
| User tiers | Queue policy | Same SLA for everyone: FIFO. Paid tiers: priority queue at admission, not preemption mid-decode |
| Overload profile | Shed threshold + fallback | Brief spikes: shed with `Retry-After` and back off with jitter. Sustained: fallback model first, then replicas |
| Slow or dead consumers | Buffer bound + abort policy | Text: small buffer, block, abort the hopeless case. Audio: drop frames. Never buffer without a bound |

## The smallest runnable stream

The chapter's core claim is easy to state and easy to disbelieve: an unbounded
buffer between a fast decoder and a slow client is a memory leak, and the
bounded fix trades that memory for slot time. So here is the whole tradeoff in
one file with zero installs. Every production component is swapped for the
smallest thing with the same interface: the inference engine becomes a clock
that emits a token every 20 ms, the slow client becomes a clock that drains one
every ~100 ms with jitter, and the gateway's per-stream buffer becomes a
counter. One run uses an unbounded queue; the other bounds it at 32 tokens and
blocks the decode loop when full, the text-chat policy from
[section 4](04-backpressure-and-concurrency.md).

```python
"""One token stream through a gateway buffer: unbounded queue vs bounded backpressure."""
import random

DECODE_MS = 20        # decoder emits a token every 20 ms (50 tok/s)
CONSUME_MS = 100      # a slow client drains a token every ~100 ms (10 tok/s)
TOTAL_TOKENS = 300    # one assistant reply

def stream(bound=None, seed=7):
    """Simulate one generation end to end.
    bound=None: unbounded gateway queue (production: an unlimited socket buffer).
    bound=B:    the decode loop blocks while the per-stream buffer holds B tokens."""
    rng = random.Random(seed)
    buffered = produced = delivered = peak = 0
    now = 0.0                       # simulated clock, ms
    next_produce = DECODE_MS
    next_consume = CONSUME_MS * rng.uniform(0.5, 1.5)
    slot_freed = None               # when the last token decodes, the slot returns
    while delivered < TOTAL_TOKENS:
        can_produce = produced < TOTAL_TOKENS and (bound is None or buffered < bound)
        if can_produce and next_produce <= next_consume:
            now = next_produce
            produced += 1
            buffered += 1
            peak = max(peak, buffered)
            next_produce = now + DECODE_MS
            if produced == TOTAL_TOKENS:
                slot_freed = now
        else:
            now = max(next_consume, now)
            if buffered:
                buffered -= 1
                delivered += 1
            next_consume = now + CONSUME_MS * rng.uniform(0.5, 1.5)
            next_produce = max(next_produce, now)   # a blocked decoder resumes on drain
    return peak, slot_freed / 1000, now / 1000

for label, bound in [("unbounded queue", None), ("bounded, B = 32", 32)]:
    peak, slot_s, done_s = stream(bound)
    print(f"{label}: peak buffer {peak:3d} tokens | "
          f"slot freed at {slot_s:5.1f}s | last token at {done_s:5.1f}s")
```

Run it and the two lines are the whole of section 4 in numbers. The unbounded
run peaks at 237 buffered tokens, nearly the entire 300-token reply sitting in
gateway memory, because the decoder finishes at 6.0 seconds while the client
drains until 29.6; multiply that buffer by tens of thousands of concurrent
streams and the gateway's memory bill is the reply corpus of every slow client
at once. The bounded run caps the buffer at exactly 32, but the blocked decode
loop holds the inference slot until 26.2 seconds instead of 6.0, which is the
tradeoff [section 4](04-backpressure-and-concurrency.md) states and why the
abort-on-hopeless-consumer policy exists at all. Delivery time is identical in
both runs, 29.6 seconds, because the client, not the buffer policy, sets the
end-to-end pace. Swap the producer clock for a real inference engine, the
consumer clock for an SSE connection, the counter for a per-stream buffer with
a write timeout, and the `slot_freed` timestamp for the engine's abort-and-free
call, and you have rebuilt this chapter's capacity model.
