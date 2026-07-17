# 8. Interview Q&A

The questions interviewers actually ask about realtime streaming chat, grouped
by how they are used. The commonly-missed ones are where interviews are won or
lost.

## Commonly asked

**Q: Why stream tokens instead of returning the full reply at the end?**

A: Perceived latency. The model decodes one token at a time, so the first token
is available almost immediately after prefill while the last may arrive seconds
later. Streaming delivers each token as it is emitted, so the user sees the
reply building in real time. Time-to-first-token (TTFT) is the metric users
feel; total generation time is secondary. A five-second reply that starts in
300ms feels faster than a two-second reply that shows nothing until it is
complete.

**Q: SSE or WebSockets? Which would you pick?**

A: SSE for text-only chat. Token delivery is one-directional (server to client),
and SSE is purpose-built for that: plain HTTP, no upgrade handshake, works
through standard proxies, simple to debug. WebSockets add full-duplex capability
you do not need for basic chat. Pick WebSockets when the product requires
mid-stream client-to-server signaling (live interrupts, audio, multiplexed
concurrent requests over one connection). Reach for WebRTC over UDP for voice,
where ordered TCP delivery stalls audio on packet loss.

**Q: Why does turn five cost more than turn one?**

A: Each turn runs prefill over the full conversation transcript. The transcript
grows by roughly one user message plus one assistant reply per turn, so by turn
five the model is reading five turns of history. Prefill cost scales with
context length. Prefix caching mitigates this by reusing the KV cache for the
stable head of the conversation, but it only helps if the follow-up turn lands
on the same replica (sticky routing). Without caching and stickiness, every turn
pays the full prefill cost over the growing transcript.

**Q: A user clicks stop mid-generation. Walk me through what happens.**

A: The client sends a cancel signal (or closes the SSE connection). The gateway
detects the cancel, sends a cancellation signal to the inference engine, and
the engine aborts the decode and frees the inference slot. The gateway writes
nothing to the session store for this turn (the partial reply is discarded). The
slot is returned to the pool immediately. Not propagating the cancel to the
inference engine is the critical mistake: the model keeps decoding tokens nobody
will read, burning a slot that could serve another user.

**Q: How do you handle a user who closes the browser tab mid-generation?**

A: The gateway detects a write error when it tries to push the next token (the
TCP connection is gone). It aborts the generation, frees the slot, and discards
the partial reply. A heartbeat on the SSE channel (a periodic comment line)
ensures the gateway detects a dead connection even when the OS does not signal a
clean close. On reconnect, the client sends `Last-Event-ID` (standard SSE
behavior), and the gateway resumes from that point if the generation is still in
flight, or serves the completed reply from the session store if it finished.

## Tricky (the follow-ups that separate people)

**Q: You claimed prefix caching cuts multi-turn latency. Traffic spikes and you
have to add new inference replicas. What happens to your cache?**

A: The cache on the new replicas is cold. Sessions that rehash to a new replica
(due to consistent-hash rebalancing) pay a full-prefill cache miss on their next
turn. The correct response is to treat the cache as a best-effort optimization:
correctness is not affected (the session store has the transcript), and the next
turn on the new replica warms its cache. You monitor the cache hit rate and
alert if it drops persistently (which would signal a routing bug, not a
rebalancing event).
**Why persistence is the tell:** a rebalancing miss is self-healing, since the
very turn that misses also warms the new replica, so the hit rate dips once and
recovers within a turn or two; a routing bug sends the same session to a
different replica on every turn, so each turn misses again and the rate stays
low. The shape of the curve over time distinguishes the two causes.

**Q: You have 10,000 concurrent streams. One user is on turn 200 of a very long
conversation. Their generation is slow because the context is huge. Does it
hurt other users?**

A: Under continuous batching, the scheduler interleaves decode steps across all
in-flight streams, so a single slow stream does not starve others. The slow
stream's issue is that its prefill is expensive (long context), which delays its
TTFT on each turn. The other 9,999 streams are not blocked.
**Why the isolation holds:** the batch is re-formed at every decode step, so no
stream ever holds the GPU for its whole generation; the one risk point is the
long prefill burst, which modern engines bound by splitting a large prefill into
chunks interleaved with everyone else's decode steps, so the huge context slows
that user's own first token rather than inflating inter-token latency across the
fleet. The mitigation for
the slow user is server-side summarization to bound their context length, or
explicit per-session rate limiting to prevent one user from claiming
disproportionate GPU time on each turn.

**Q: Why not cache the full conversation on the client and send it every turn?
That avoids the sticky-routing requirement.**

A: It is a valid design and is used by some simple deployments. The cost is
that per-request payload size grows with history, the server cannot do
server-side summarization or context management, and a large transcript becomes
a significant serialization and network cost per turn. The bigger issue is
trust: if the client sends the history, the server must validate it on every
request (the client could truncate or tamper with it). Server-side state is
the right call for products with long sessions, resumability requirements, or
security constraints.
**Why tampering is more than a billing nuisance:** the transcript is the model's
entire grounding for the turn, so a client that edits history can fabricate
earlier assistant commitments ("you already agreed to waive the fee") or smuggle
in instructions the moderation layer never saw on any real turn; trusting
client-held history turns every past message into an unauthenticated input.

**Q: Your voice product has 600ms end-to-end latency. Where do you cut it?**

A: Start with the pipeline sum: STT plus turn detection plus LLM first-token
plus TTS first-byte plus network. Measure each stage. Common high-value cuts:
(a) eager turn detection (start the LLM on medium-confidence transcript, cancel
if the user resumes), which shaves 200 to 400ms off the turn-detection wait;
(b) streaming TTS that starts playback before the full sentence is synthesized;
(c) moving to a fused speech-to-speech model if you can give up the inspectable
intermediate transcript. Cutting network means multi-region deployment with
traffic routed to the nearest region.
**Why eager turn detection is the biggest single cut:** a turn detector can only
be sure the user finished by waiting through a silence threshold, and that wait
is pure dead time added to every single turn; starting the LLM speculatively
overlaps the dead time with useful compute, and the gamble is cheap because a
wrong guess is cancelled before anything was spoken back to the user.

**Q: Time-to-first-token and inter-token latency both measure streaming speed;
when does the difference actually matter?**
A: Both show up in "the stream feels slow," and dashboards often blur them into
one latency number. They are produced by different phases with different
bottlenecks. TTFT is dominated by queueing plus prefill: it grows with context
length and admission delay, and it is what prefix caching, sticky routing, and
chunked prefill improve. Inter-token latency is set by decode throughput: it
grows with batch size and model size, and it is what batching policy and
hardware choice improve. The difference matters the moment you tune anything:
raising the batch ceiling improves cost per token while worsening ITL and
leaving TTFT roughly alone, while a prefix cache improves TTFT and does nothing
for ITL. A single blended latency metric hides which knob to turn; the fix for
one phase routinely makes the other slightly worse.

**Q: The user's network blips mid-stream and the SSE connection drops. How does the
standard help you resume without replaying the whole answer?**

A: Server-Sent Events (W3C/WHATWG) has reconnection built in: the browser's
`EventSource` automatically reconnects when the connection drops, and if the server
tagged each event with an `id:` field, the client sends the last id it saw in a
`Last-Event-ID` request header on reconnect. The server can then resume streaming
from the token after that id rather than from the start, provided it buffered the
in-flight generation keyed by a stream id (the transcript in the session store is the
durable copy). Without server-side buffering the reconnect still fires but there is
nothing to resume from, so the client either replays from scratch or waits for the
persisted final message. This is why production streaming assigns per-event ids even
though the happy path never reads them back.

## Commonly answered wrong

**Q: Should I use WebSockets for voice to get lower latency than SSE?**

A: No. The transport choice for voice is WebRTC over UDP, not WebSocket.
WebSocket runs on TCP, which guarantees ordered delivery by retransmitting lost
packets. For text tokens, a retransmit is a slight delay; for audio frames, a
200ms retransmit stall destroys conversational flow (head-of-line blocking).
WebRTC uses UDP: a lost 20ms audio frame is dropped, which is barely audible.
WebRTC also bundles jitter buffering, echo cancellation, congestion control, and
NAT traversal. Using WebSocket for voice is the single most common transport
mistake in this interview.

**Q: To handle overload, I will queue all excess requests and serve them when
slots free up.**

A: Queueing without bound is worse than shedding, because a long queue means
the requests at the back of the queue are sitting there for minutes by the time
they get served. The correct overload response is: queue with an honest
displayed wait, shed beyond a threshold with a clear retry signal (HTTP 429 with
a `Retry-After` header), and fall back to a smaller model to raise the slot
capacity under a spike. Silent indefinite queueing looks like a hang to the
user and amplifies load from retries.
**Why the amplification happens:** users and client libraries time out and
retry, so every silently queued request spawns duplicates that join the same
queue behind it; arrival rate rises just as effective capacity is exhausted, and
much of what the queue eventually serves belongs to users who already gave up,
which is pure wasted GPU time. Shedding early keeps the work the system does
aligned with users still waiting for it.

**Q: I will add more context to the session store and include it in every prompt
to make the model smarter about long conversations.**

A: Growing the stored context without bound is the failure mode, not the fix.
Every extra token stored is re-read in prefill on the next turn. The correct
design bounds context growth with summarization or a sliding window, and invests
in prefix caching plus sticky routing so the re-read cost stays low for the
stable head. Adding more context without a bound is how you arrive at a product
that errors out mid-conversation when it hits the context limit.
**Why more context does not even buy the quality it promises:** old turns
compete with the current question for the model's attention, and retrieval of a
relevant detail degrades as the transcript grows around it; a bounded,
summarized context often answers better, not just cheaper, because the signal
the model needs is no longer buried in turns that stopped mattering.

**Q: I will use session affinity in the load balancer to make prefix caching
work.**

A: Sticky load balancing is a mechanism, but it is not sufficient on its own.
If a user opens two tabs, or reconnects after a disconnect, or the load balancer
re-routes them (on a node restart, or because the sticky session expired), the
session lands on a cold replica. The correct design treats stickiness as
best-effort: attempt to route to the warm replica, but accept that cache misses
happen and make sure the cold-cache path is correct and merely slower, not
broken.
**Why the cold path must be first-class:** the session-to-replica mapping lives
in the load balancer and is not durable, while the events that break it
(restarts, scale-ups, failovers) cluster at exactly the moments of highest
stress; a design that is only correct on the warm path therefore fails
preferentially during incidents. Correctness has to live in the session store,
with the cache changing only latency.

**Q: Streaming reduces the total time to generate the answer, right?**

A: No. Streaming does not change how long the model takes to produce the last token;
decode throughput is identical whether or not you stream. What it changes is perceived
latency: it delivers the first token as soon as prefill finishes (TTFT) instead of
withholding every token until the full completion is ready, so the user starts reading
while generation continues. Time to the final token, and total cost, are unchanged.
Confusing time-to-first-token with time-to-last-token leads people to promise a
throughput win streaming does not deliver; the win is entirely in responsiveness,
which is why TTFT and inter-token latency are the metrics that matter rather than a
single end-to-end number.
