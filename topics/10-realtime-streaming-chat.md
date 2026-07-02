# 10 - Realtime streaming chat

> **Interviewer:** "Design the serving and application layer for a multi-turn
> streaming chat product. I am interested in the transport and state around the
> model: how tokens reach the user, how conversation state is managed, and how the
> system holds up under load and disconnects."

This question is about everything wrapping the model, not the model internals. It
is where backend engineering meets LLM serving: streaming transport, session
state, sticky routing, backpressure, and graceful degradation. The cost model
from [topic 02](02-long-context-and-kv-cache.md) shows up here as a product
problem, because the transcript grows every turn and someone has to pay for it.

## 1. Clarify and scope

- **Conversation length?** A few turns or long sessions? This decides how hard the
  growing-context problem bites.
- **Latency target?** Time-to-first-token is the number users feel in chat. State
  a target (for example under 1 second p95 to first token).
- **Scale and concurrency?** Peak concurrent streams, not just QPS. A streaming
  connection holds a slot for the whole generation, which changes capacity math.
- **Statefulness?** Does the server hold conversation state, or does the client
  send the full transcript each turn? Both are valid; they trade differently.
- **Multi-device / resumable?** Does a session need to survive a reconnect or move
  between devices?

## 2. Requirements

**Functional**
- Stream tokens to the client as they are generated
- Maintain multi-turn conversation context
- Let the user cancel a generation mid-stream
- Handle disconnects and (optionally) resume

**Non-functional**
- Low time-to-first-token at p95
- High concurrent-stream capacity per GPU
- Bounded cost per conversation as it grows
- Graceful degradation under overload rather than hard failures

## 3. High-level data flow

```mermaid
flowchart LR
  C["client"] -->|"SSE / websocket"| G["gateway /<br/>streaming layer"]
  G --> P["inference pool<br/>(continuous batching)"]
  G -->|"token stream"| C
  G --> S["session store<br/>(transcript, summary,<br/>routing key)"]
  P --> S
```

## 4. Deep dives

### Why stream, and how

You stream because **perceived latency** is dominated by time-to-first-token. A
model that takes several seconds to finish feels fast if the first words appear in
well under a second. Two transports:

- **Server-sent events (SSE):** one-way server-to-client, runs over plain HTTP,
  simple, and a natural fit because token output is one-directional. The common
  default.
- **WebSockets:** full duplex. Worth it when you need rich client-to-server
  signaling mid-stream (live interrupts, audio, collaborative state). More
  overhead to operate.

Default to SSE unless you genuinely need duplex.

### Conversation state and the growing-context cost

Every turn, the model must see the conversation so far, and that transcript grows.
Two consequences:

- **Cost grows per turn.** Each turn re-processes the history in prefill, so a long
  chat gets more expensive with every message. This is the cost model of
  [topic 02](02-long-context-and-kv-cache.md) showing up as a product bill.
- **You eventually hit the context limit.**

Mitigations:

- **Prefix caching.** The system prompt and the stable head of the conversation
  repeat every turn, so cache their KV state and reuse it instead of recomputing
  prefill ([topic 02](02-long-context-and-kv-cache.md)). This is the single
  biggest win for multi-turn cost and latency.
- **Summarization / truncation.** Once the history is long, summarize older turns
  or drop the least relevant ones to bound growth. State the tradeoff: you lose
  some fidelity to save cost and stay under the limit.

Where state lives is a design choice: a stateless server (the client sends the
full transcript each turn) is simple and horizontally trivial but moves cost and
trust to the client; a stateful server (a session store holds the transcript)
enables server-side summarization and smaller payloads but needs the routing
below.

### Sticky routing for cache reuse

Prefix caching only helps if the follow-up turn lands on the **same replica** that
holds the cached KV state. So route a session consistently to its replica (a
sticky routing key). Without stickiness, every turn is a cache miss and you pay
full prefill each time. Balance this against load: a hot session must still be
movable, so treat the cache as an optimization, not a hard pin.

### Backpressure and cancellation

A streaming generation holds an inference slot for its entire duration, so freeing
slots promptly is a capacity issue, not just hygiene:

- **Cancellation.** When the user clicks stop, propagate the cancel to the
  inference engine and free the slot immediately. Do not keep generating tokens
  nobody will read.
- **Disconnect detection.** If the client drops mid-stream, detect the closed
  connection and abort generation. Orphaned streams silently eat capacity.
- **Backpressure.** If the client cannot consume tokens as fast as they are
  produced, you need bounded buffering and a policy for a slow consumer.

### Graceful degradation under overload

Concurrent streams are the binding constraint because each holds a slot. When the
inference pool saturates:

- **Queue** with an honest wait, surfaced to the user, rather than hanging.
- **Shed load** with a clear retry signal instead of timing out silently.
- **Fall back** to a smaller, cheaper model to keep the product responsive under
  spikes, accepting a quality dip.

Continuous batching ([topic 02](02-long-context-and-kv-cache.md), topic 04) on the
inference pool is what lets many streams share GPUs in the first place.

## 5. Bottlenecks and scaling

| Bottleneck | Cause | Fix | Tradeoff |
|---|---|---|---|
| Time-to-first-token | Large prefill (long history) | Prefix caching; summarize history | Some fidelity loss |
| Cost per long chat | History reprocessed each turn | Prefix caching; truncate / summarize | Fidelity vs cost |
| Concurrent-stream capacity | Each stream holds a slot | Continuous batching; cancel on disconnect | Infra complexity |
| Cache misses across turns | Non-sticky routing | Session-sticky routing key | Load-balancing flexibility |
| Overload | More streams than slots | Queue, shed, or fall back to a smaller model | Latency or quality dip |

## 6. Failure modes

- **Orphaned generations.** Client disconnects but the server keeps generating,
  burning slots. Detect and abort.
- **Cache stampede on a hot session.** Many turns pinned to one replica overload
  it. Allow migration; treat the cache as best-effort.
- **Unbounded history.** No summarization, so cost and latency climb until the
  context limit errors out mid-conversation. Bound it proactively.
- **Silent overload.** Hanging instead of a clear queue or retry. Degrade
  visibly.

## 7. Likely follow-ups

- "Why does turn five cost more than turn one?" The whole transcript is
  reprocessed in prefill each turn; history grows, so prefill grows.
- "How do you cut multi-turn latency and cost?" Prefix caching plus sticky
  routing so the cache actually hits, plus summarization to bound growth.
- "A user closes the tab mid-generation. What happens?" Detect the dropped
  connection, abort generation, free the slot.
- "Traffic spikes 5x. How does the product stay up?" Continuous batching for
  throughput, queue with honest waits, shed or fall back to a smaller model.

---

## Seen in production

Real systems that ship the patterns above. Each is a first-party engineering
writeup; read them for what an interview answer skips: who the system serves,
the product design, the eval bar, and the deployment shape.

### The shared pipeline

Every system here carries the same skeleton: an LLM emits tokens, a transport streams them to the client, and the client renders incrementally while session memory feeds history back into the next turn. The transport is the main fork. Text chat leans on SSE or WebSockets, while live voice moves to WebRTC because ordered TCP delivery stalls audio on packet loss. Voice agents wrap the LLM in an STT front end and a TTS back end, and add a turn-detection stage that decides when the user has finished speaking so the model can start.

```mermaid
flowchart LR
  U["user"] --> LLM["LLM token stream"]
  LLM --> T["transport<br/>(SSE / WebSocket / WebRTC)"]
  T --> R["client render"]
  M["session memory"] --> LLM
  BP["backpressure"] --> T
  subgraph voice["voice path"]
    STT["STT"] --> TD["turn detection"] --> LLM2["LLM"] --> TTS["TTS"]
  end
```

### How they differ

| System | Transport | Text / voice | Turn detection / endpointing | Session memory or scale concern | When it wins | When it breaks / watch out |
|---|---|---|---|---|---|---|
| LinkedIn | realtime messaging (WebSocket) | text | n/a | history in shared prompt templates | streaming plus progressive parsing hides latency on structured output | mid-stream parsing of partial output is fragile; shared templates couple history |
| Cloudflare | WebSocket via Durable Objects | text | n/a | persistent connections for concurrent streams | many long-lived concurrent streams needing stateful edge connections | a Durable Object per connection adds coordination and cost overhead |
| Vercel | native streaming, throttled fallback | text | n/a | cross-platform stream delivery | one chat UI shipping across platforms that stream unevenly | the throttled fallback trades token smoothness for reach |
| OpenAI | Realtime speech-to-speech | voice | model-side | audio snapshots for STT and TTS | lowest-latency voice when one model handles speech in and out | model-side turn detection is a black box you cannot tune per product |
| LiveKit | WebRTC (WebSocket for signaling) | voice | semantic end-of-turn model | packet loss and jitter tolerance | real-world networks with packet loss, jitter, and congestion | WebRTC is heavier to operate; raw model APIs lack transport, echo, and turn logic |
| Deepgram | streaming STT | voice | eager end-of-turn on medium confidence | overlap LLM start with speech | shaving latency by starting the LLM before the user fully stops | eager start can act on a half-finished utterance and misfire |
| AssemblyAI | streaming STT | voice | intelligent endpointing (~300ms) | immutable transcripts | stable transcripts fast with no later rewrites | the ~300ms endpointing floor adds to every turn |
| ElevenLabs | streaming TTS | voice | n/a | TTS time-to-first-byte | responsive spoken replies where the first audio byte matters | TTS time-to-first-byte piles onto the tail after the LLM |
| Cartesia | streaming TTS | voice | n/a | 135ms model latency | the tightest voice latency budgets | a state-space TTS trades some quality for that latency |
| Krisp | CPU turn model | voice | 6M-weight turn detection | speak, listen, or wait decision | cheap CPU turn-taking with no GPU in the loop | a 6M-weight model bounds how nuanced the decision can be |
| Twilio | WebSocket Media Streams | voice | n/a | raw call audio fork, bidirectional | bridging telephony and PSTN call audio into an LLM | it forks raw audio only; STT, turn detection, and TTS are on you |
| Vapi | streaming STT plus VAD | voice | VAD and endpointing | inference coordination | wanting an assembled pipeline instead of wiring parts yourself | coordinating STT, VAD, LLM, and TTS multiplies moving parts |
| Daily / Pipecat | WebRTC, open benchmark | voice | Smart Turn v3 semantic VAD (12ms CPU) | latency benchmarking | open-source semantic turn detection with measurable latency | you self-host and assemble the whole stack |
| Slack | WebSocket gateway | text | n/a | stateful channel servers, 500ms global delivery | stateful channels with predictable global delivery | stateful servers complicate scaling and failover |
| Discord | WebSocket (Elixir GenServer) | text | n/a | 5M concurrent, Manifold fan-out | massive concurrent fan-out to many listeners at once | per-session actors and fan-out need a specialized runtime (Elixir/BEAM) |

The core dividing line is the medium: text systems optimize concurrent-connection scale over SSE or WebSocket, while voice systems switch to WebRTC and bolt on STT, turn-detection, and TTS stages to fight per-hop latency.

### The systems

- **LinkedIn** [Musings on building a Generative AI product](https://www.linkedin.com/blog/engineering/generative-ai/musings-on-building-a-generative-ai-product): End-to-end token streaming and progressive parsing to cut perceived latency. *(deployment)*
- **Cloudflare** [Durable Objects for WebSockets and auth in AI Gateway](https://blog.cloudflare.com/do-it-again/): Scaling persistent WebSocket connections for concurrent AI inference streams. *(deployment)*
- **Vercel** [Chat SDK brings agents to your users](https://vercel.com/blog/chat-sdk-brings-agents-to-your-users): Streaming responses cross-platform via native streaming versus a throttled fallback. *(product design)*

- **OpenAI** [Updates for developers building with voice](https://developers.openai.com/blog/updates-audio-models): New audio model snapshots for STT, TTS, and realtime speech-to-speech. *(product design)*
- **LiveKit** [Why WebRTC beats WebSockets for realtime voice AI](https://livekit.com/blog/why-webrtc-beats-websockets-for-voice-ai-agents): WebRTC handles packet loss, jitter, and congestion better than TCP. *(deployment)*
- **LiveKit** [Why you shouldn't build voice agents directly on model APIs](https://livekit.com/blog/real-time-voice-agents-vs-model-apis): Model APIs lack transport, echo cancellation, and turn detection. *(deployment)*
- **Deepgram** [Optimize voice agent latency with eager end of turn](https://developers.deepgram.com/docs/flux/voice-agent-eager-eot): Start the LLM on medium-confidence transcripts to overlap with speech. *(deployment)*
- **AssemblyAI** [Universal-Streaming: ultra-fast speech-to-text for voice agents](https://www.assemblyai.com/blog/introducing-universal-streaming): Immutable streaming transcripts in about 300ms with intelligent endpointing. *(eval bar)*
- **ElevenLabs** [Enhancing conversational AI latency with efficient TTS](https://elevenlabs.io/blog/enhancing-conversational-ai-latency-with-efficient-tts-pipelines): Reducing streaming TTS time-to-first-byte for responsive conversation. *(deployment)*
- **Daily** [Benchmarking LLMs for voice agent use cases](https://www.daily.co/blog/benchmarking-llms-for-voice-agent-use-cases/): An open benchmark for latency, tool calling, and instruction adherence in voice. *(eval bar)*
- **Cartesia** [Announcing Sonic: a low-latency voice model](https://cartesia.ai/blog/sonic): A state-space TTS hitting 135ms model latency for streaming voice agents. *(product design)*
- **Krisp** [A 6M-weight turn-taking model for voice AI agents](https://krisp.ai/blog/turn-taking-for-voice-ai/): A tiny CPU turn-detection model deciding when agents speak, listen, or wait. *(product design)*
- **Twilio** [Introducing Media Streams](https://www.twilio.com/en-us/blog/media-streams-public-beta): Forks raw call audio over WebSockets for real-time bidirectional voice apps. *(deployment)*
- **Vapi** [How we built Vapi's voice AI pipeline (part 2)](https://vapi.ai/blog/how-we-built-vapi-s-voice-ai-pipeline-part-2): VAD, endpointing, streaming STT, and inference coordination for low-latency voice. *(deployment)*
- **Daily (Pipecat)** [Smart Turn v3, with CPU inference in 12ms](https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/): An open-source semantic-VAD turn-detection model, 8MB, 23 languages, CPU-friendly. *(product design)*
- **Slack** [Real-time Messaging](https://slack.engineering/real-time-messaging/): A stateful WebSocket gateway and channel servers deliver messages globally in 500ms. *(deployment)*
- **Discord** [How Discord Scaled Elixir to 5,000,000 Concurrent Users](https://discord.com/blog/how-discord-scaled-elixir-to-5-000-000-concurrent-users): Elixir GenServer sessions and Manifold fan-out for millions of concurrent WebSockets. *(deployment)*

More production case studies: the [Evidently AI ML system design database](https://www.evidentlyai.com/ml-system-design) (800 case studies from 150+
companies) is the broadest curated index; this section pulls the ones that map
directly onto this topic.

---
## Trace the architectures

The transport and state layer is wrapped around one expensive thing: the decoder
generating tokens. The per-turn cost you are managing comes straight from its
attention block building the KV cache, so it helps to see the actual model the
stream is carrying.

- **The decoder behind the stream (Llama-3 8B):**
  [open it live](https://www.neurarch.com/?import=https://raw.githubusercontent.com/neurarch-ai/awesome-llm-model-zoo/main/architectures/llama3-8b/model.json).
  Find the attention block: that is what builds the KV cache whose reuse (prefix
  caching) and growth (summarization) this whole topic is about. Its grouped-query
  attention keeps the KV cache *memory* small as context grows; the per-turn
  prefill cost (re-reading the transcript each turn) is bounded by prefix caching
  and summarization, not by GQA. Keep the two distinct.

  ![Llama-3 8B](https://raw.githubusercontent.com/neurarch-ai/awesome-llm-model-zoo/main/architectures/llama3-8b/assets/diagram.png)

This is a validated reference graph at real dimensions, shape-checked end to end,
not a screenshot. All 92 architectures live in the
[Model Zoo](https://github.com/neurarch-ai/awesome-llm-model-zoo)
([gallery](https://neurarch-ai.github.io/awesome-llm-model-zoo)). Built by
[Neurarch](https://www.neurarch.com).

## Related deep-dive drills

Rapid-fire questions that probe the modeling and systems underneath this topic, from [deep-dives.md](../deep-dives.md):

- [Decoding and sampling](../deep-dives.md#decoding-and-sampling)
- [Inference, quantization, and serving math](../deep-dives.md#inference-quantization-and-serving-math)
- [Commonly asked, commonly missed](../deep-dives.md#commonly-asked-commonly-missed)
