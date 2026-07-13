# 7. How teams do it in production

Every shipping system carries the same spine: an LLM emits tokens, a transport
carries them to the client, the client renders incrementally, and session memory
feeds the history back into the next turn. What differs between teams is the
transport choice, the pipeline shape (text versus voice, fused versus
componentized), how they manage backpressure, and where they invest in
reliability.

## Where real systems diverge

| System | Transport | Text / voice | Session / state | Backpressure / overload | Signature choice |
|---|---|---|---|---|---|
| LinkedIn | streaming over HTTP | text | shared prompt templates carry history | progressive parsing fires downstream calls mid-stream | progressive parsing: skill calls fire as their params appear in the stream, not after the full reply |
| Cloudflare AI Gateway | WebSocket via Durable Objects | text | per-connection UUID, Durable Object holds lifecycle | eventId on every chunk for multiplex demux | one persistent socket for many inference requests; eventId solves attribution |
| Vercel Chat SDK | native streaming; throttled edit loop on non-streaming platforms | text | Redis or Postgres for distributed locks and kv state | throttled fallback rate-limits edits to platform limits | one agent codebase, many platform adapters; the fallback repeatedly edits a message |
| OpenAI gpt-realtime-mini | Realtime API (WebSocket) | voice: speech-to-speech | audio snapshots (model-side turn detection) | model-side, not tunable externally | fused speech-to-speech; turn detection is inside the model, not a separate endpointer |
| LiveKit | WebRTC over UDP | voice | multi-region SFU routes to nearest node | congestion control before packet loss occurs | WebRTC for voice: UDP drops a lost 20ms frame instead of stalling the whole stream |
| Deepgram Flux | streaming STT | voice (STT front-end) | EagerEndOfTurn on medium confidence; TurnResumed cancels | speculative LLM call cancelled on TurnResumed | eager end-of-turn: start the LLM before the user has fully stopped speaking |
| AssemblyAI Universal-Streaming | streaming STT | voice (STT front-end) | immutable transcripts, never revised | ~300ms endpointing floor | immutable words on first emission; acoustic + semantic + silence endpointing |
| ElevenLabs | streaming TTS | voice (TTS back-end) | adaptive buffering per network conditions | parallel preprocessing, synthesis, rendering | stream audio chunks before the full sentence is synthesized |
| Cartesia Sonic | streaming TTS | voice (TTS back-end) | 135ms model latency target | state-space model for speed | state-space architecture trades some quality for 135ms TTS first-byte |
| Slack | WebSocket gateway | text | stateful channel servers, ~500ms global delivery | stateful servers complicate scaling and failover | persistent WebSocket channels with stateful servers per channel |
| Discord | WebSocket via Elixir GenServer | text | 5M concurrent sessions, Manifold fan-out | Elixir BEAM handles massive concurrent actor count | per-session GenServer actors; Manifold for fan-out to many listeners |
| Daily / Pipecat Smart Turn v3 | WebRTC, open stack | voice | semantic VAD, 12ms CPU inference, 8MB model | open-source, self-hosted | smallest semantic turn-detection model publicly benchmarked |
| Krisp | CPU turn-detection model | voice | 6M-weight model, no GPU required | fully CPU, cheap | tiny CPU model decides speak / listen / wait per frame |

## The core dividing line

Text systems and voice systems differ in one deep way: text tokens tolerate
ordered TCP delivery. A delayed token just makes the user wait. Audio frames
cannot wait: a 200ms TCP retransmit stalls all buffered audio while the gap is
filled, destroying conversational flow. That head-of-line blocking problem is
why voice moves to WebRTC over UDP, where a lost 20ms frame is dropped
(barely audible) rather than recovered (catastrophic stall).

Voice also adds two new latency terms that text does not have: speech-to-text on
the front end and text-to-speech on the back end. These sum:

$$L_{\text{voice}} = L_{\text{STT}} + L_{\text{turn}} + L_{\text{LLM}} + L_{\text{TTS}} + L_{\text{net}}$$

A 300ms endpointing floor (AssemblyAI) plus a 200ms STT latency plus a 400ms
LLM first-token plus a 135ms TTS first-byte (Cartesia) already adds up to over
one second before any network. Every team building a conversational voice product
is managing this sum tightly.

## The systems

First-party engineering writeups. Each is a production deployment, not a
demo.

- **LinkedIn** [Musings on building a Generative AI product](https://www.linkedin.com/blog/engineering/generative-ai/musings-on-building-a-generative-ai-product): end-to-end token streaming and progressive parsing to cut perceived latency. *(deployment)*
- **Cloudflare** [Durable Objects for WebSockets and auth in AI Gateway](https://blog.cloudflare.com/do-it-again/): scaling persistent WebSocket connections for concurrent AI inference streams. *(deployment)*
- **Vercel** [Chat SDK brings agents to your users](https://vercel.com/blog/chat-sdk-brings-agents-to-your-users): streaming responses cross-platform via native streaming versus a throttled fallback. *(product design)*
- **OpenAI** [Updates for developers building with voice](https://developers.openai.com/blog/updates-audio-models): audio model snapshots for STT, TTS, and realtime speech-to-speech. *(product design)*
- **LiveKit** [Why WebRTC beats WebSockets for realtime voice AI](https://livekit.com/blog/why-webrtc-beats-websockets-for-voice-ai-agents): WebRTC handles packet loss, jitter, and congestion better than TCP for audio. *(deployment)*
- **LiveKit** [Why you shouldn't build voice agents directly on model APIs](https://livekit.com/blog/real-time-voice-agents-vs-model-apis): model APIs lack transport, echo cancellation, and turn detection. *(deployment)*
- **Deepgram** [Optimize voice agent latency with eager end of turn](https://developers.deepgram.com/docs/flux/voice-agent-eager-eot): start the LLM on medium-confidence transcripts to overlap with speech. *(deployment)*
- **AssemblyAI** [Universal-Streaming: ultra-fast speech-to-text for voice agents](https://www.assemblyai.com/blog/introducing-universal-streaming): immutable streaming transcripts in about 300ms with intelligent endpointing. *(eval bar)*
- **ElevenLabs** [Enhancing conversational AI latency with efficient TTS](https://elevenlabs.io/blog/enhancing-conversational-ai-latency-with-efficient-tts-pipelines): reducing streaming TTS time-to-first-byte for responsive conversation. *(deployment)*
- **Cartesia** [Announcing Sonic: a low-latency voice model](https://cartesia.ai/blog/sonic): a state-space TTS hitting 135ms model latency for streaming voice agents. *(product design)*
- **Krisp** [A 6M-weight turn-taking model for voice AI agents](https://krisp.ai/blog/turn-taking-for-voice-ai/): a tiny CPU turn-detection model deciding when agents speak, listen, or wait. *(product design)*
- **Twilio** [Introducing Media Streams](https://www.twilio.com/en-us/blog/media-streams-public-beta): forks raw call audio over WebSockets for real-time bidirectional voice apps. *(deployment)*
- **Vapi** [How we built Vapi's voice AI pipeline (part 2)](https://vapi.ai/blog/how-we-built-vapi-s-voice-ai-pipeline-part-2): VAD, endpointing, streaming STT, and inference coordination for low-latency voice. *(deployment)*
- **Daily (Pipecat)** [Smart Turn v3, with CPU inference in 12ms](https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/): an open-source semantic-VAD turn-detection model, 8MB, 23 languages, CPU-friendly. *(product design)*
- **Slack** [Real-time Messaging](https://slack.engineering/real-time-messaging/): a stateful WebSocket gateway and channel servers deliver messages globally in 500ms. *(deployment)*
- **Discord** [How Discord Scaled Elixir to 5,000,000 Concurrent Users](https://discord.com/blog/how-discord-scaled-elixir-to-5-000-000-concurrent-users): Elixir GenServer sessions and Manifold fan-out for millions of concurrent WebSockets. *(deployment)*
