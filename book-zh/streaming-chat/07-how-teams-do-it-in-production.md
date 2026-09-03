# 7. 真实团队在生产环境里怎么做

每一个真正上线的系统都长着同一根脊椎：LLM 吐 token，某种传输把 token 送到客户端，客户端逐步渲染，
session 记忆再把历史喂回下一轮。团队之间的差别在于：选哪种传输、流水线长什么样（文本还是语音，融合式还是组件式）、
背压怎么管，以及可靠性上把钱花在哪儿。

## 真实系统的分歧在哪里

| 系统 | 传输 | 文本 / 语音 | Session 与状态 | 背压 / 过载 | 标志性选择 |
|---|---|---|---|---|---|
| LinkedIn | HTTP 流式 | 文本 | 共享的 prompt 模板携带历史 | 渐进式解析在流还没结束时就触发下游调用 | 渐进式解析：技能调用的参数一在流里出现就发出去，不等整条回复 |
| Cloudflare AI Gateway | 基于 Durable Objects 的 WebSocket | 文本 | 每条连接一个 UUID，Durable Object 管生命周期 | 每个 chunk 带 eventId，用来做多路复用的解复用 | 一条长连接承载多个推理请求；eventId 解决归属问题 |
| Vercel Chat SDK | 原生流式；在不支持流式的平台上退化成限速的编辑循环 | 文本 | Redis 或 Postgres 存分布式锁和 kv 状态 | 限速的兜底路径把编辑频率压到平台上限之内 | 一份 agent 代码，多个平台适配器；兜底路径靠反复编辑同一条消息 |
| OpenAI gpt-realtime-mini | Realtime API（WebSocket） | 语音：speech-to-speech | 音频模型快照（轮次检测在模型侧） | 在模型侧，外部调不了 | 融合式 speech-to-speech；轮次检测长在模型里，不是一个独立的 endpointer |
| LiveKit | 基于 UDP 的 WebRTC | 语音 | 多区域 SFU 路由到最近的节点 | 在丢包真的发生之前就做拥塞控制 | 语音走 WebRTC：丢了一个 20ms 的帧就丢了，而不是把整条流卡住 |
| Deepgram Flux | 流式 STT | 语音（STT 前端） | 中等置信度就触发 EagerEndOfTurn；TurnResumed 负责取消 | 投机性的 LLM 调用在 TurnResumed 时被取消 | 抢跑式的轮次结束判定：用户还没完全说完就先把 LLM 发出去 |
| AssemblyAI Universal-Streaming | 流式 STT | 语音（STT 前端） | 转写结果不可变，从不回改 | 大约 300ms 的 endpointing 下限 | 词一旦吐出就不再改；声学 + 语义 + 静音三路 endpointing |
| ElevenLabs | 流式 TTS | 语音（TTS 后端） | 按网络状况自适应缓冲 | 预处理、合成、渲染并行跑 | 整句还没合成完就先把音频块推出去 |
| Cartesia Sonic | 流式 TTS | 语音（TTS 后端） | 模型延迟目标 135ms | 用状态空间模型换速度 | 状态空间架构拿一点质量换来 135ms 的 TTS 首字节 |
| Slack | WebSocket 网关 | 文本 | 有状态的频道服务器，全球送达约 500ms | 有状态服务器让扩容和故障切换更麻烦 | 长连 WebSocket 频道，每个频道配有状态的服务器 |
| Discord | 基于 Elixir GenServer 的 WebSocket | 文本 | 500 万并发 session，用 Manifold 做扇出 | Elixir BEAM 扛得住海量并发 actor | 每个 session 一个 GenServer actor；扇出给大量监听者靠 Manifold |
| Daily / Pipecat Smart Turn v3 | WebRTC，开放技术栈 | 语音 | 语义 VAD，CPU 上 12ms 推理，模型 8MB | 开源，自托管 | 公开做过 benchmark 的语义轮次检测模型里最小的一个 |
| Krisp | 跑在 CPU 上的轮次检测模型 | 语音 | 600 万权重的模型，不需要 GPU | 全 CPU，便宜 | 一个极小的 CPU 模型逐帧判定该说、该听，还是该等 |

## 最核心的那条分界线

文本系统和语音系统有一处很深的差别：文本 token 受得了 TCP 的有序投递，一个 token 晚到了，无非是让用户多等一会儿。
音频帧等不起：一次 200ms 的 TCP 重传会在补空洞的这段时间里把所有缓冲的音频全卡住，对话的流畅感就毁了。
正是这个队头阻塞问题，把语音推向了基于 UDP 的 WebRTC：丢了一个 20ms 的帧就让它丢掉（几乎听不出来），
而不是去把它救回来（代价是灾难性的卡顿）。

语音还比文本多出两项延迟：前端的语音转文字和后端的文字转语音。它们是累加的：

$$L_{\text{voice}} = L_{\text{STT}} + L_{\text{turn}} + L_{\text{LLM}} + L_{\text{TTS}} + L_{\text{net}}$$

300ms 的 endpointing 下限（AssemblyAI）加 200ms 的 STT 延迟，加 400ms 的 LLM 首 token，
再加 135ms 的 TTS 首字节（Cartesia），网络还没算进去就已经超过一秒了。
每一个做对话式语音产品的团队，都在死死盯着这个和。

## 这些系统

一手的工程记录。每一条都是真实的生产部署，不是 demo。

- **LinkedIn** [Musings on building a Generative AI product](https://www.linkedin.com/blog/engineering/generative-ai/musings-on-building-a-generative-ai-product)：端到端的 token 流式输出，加上渐进式解析来压低感知延迟。*(部署)*
- **Cloudflare** [Durable Objects for WebSockets and auth in AI Gateway](https://blog.cloudflare.com/do-it-again/)：为并发的 AI 推理流扩展长连 WebSocket 连接。*(部署)*
- **Vercel** [Chat SDK brings agents to your users](https://vercel.com/blog/chat-sdk-brings-agents-to-your-users)：跨平台的流式回复，原生流式和限速兜底两条路。*(产品设计)*
- **OpenAI** [Updates for developers building with voice](https://developers.openai.com/blog/updates-audio-models)：面向 STT、TTS 和实时 speech-to-speech 的音频模型快照。*(产品设计)*
- **LiveKit** [Why WebRTC beats WebSockets for realtime voice AI](https://livekit.com/blog/why-webrtc-beats-websockets-for-voice-ai-agents)：在音频这件事上，WebRTC 处理丢包、抖动和拥塞都比 TCP 好。*(部署)*
- **LiveKit** [Why you shouldn't build voice agents directly on model APIs](https://livekit.com/blog/real-time-voice-agents-vs-model-apis)：模型 API 不管传输、回声消除和轮次检测。*(部署)*
- **Deepgram** [Optimize voice agent latency with eager end of turn](https://developers.deepgram.com/docs/flux/voice-agent-eager-eot)：拿中等置信度的转写就先发 LLM，让它和用户说话的过程重叠起来。*(部署)*
- **AssemblyAI** [Universal-Streaming: ultra-fast speech-to-text for voice agents](https://www.assemblyai.com/blog/introducing-universal-streaming)：大约 300ms 的不可变流式转写，配上智能 endpointing。*(评估标准)*
- **ElevenLabs** [Enhancing conversational AI latency with efficient TTS](https://elevenlabs.io/blog/enhancing-conversational-ai-latency-with-efficient-tts-pipelines)：压低流式 TTS 的首字节时间，让对话跟得上。*(部署)*
- **Cartesia** [Announcing Sonic: a low-latency voice model](https://cartesia.ai/blog/sonic)：一个状态空间的 TTS，为流式语音 agent 做到 135ms 的模型延迟。*(产品设计)*
- **Krisp** [A 6M-weight turn-taking model for voice AI agents](https://krisp.ai/blog/turn-taking-for-voice-ai/)：一个极小的 CPU 轮次检测模型，决定 agent 什么时候说、什么时候听、什么时候等。*(产品设计)*
- **Twilio** [Introducing Media Streams](https://www.twilio.com/en-us/blog/media-streams-public-beta)：把通话的原始音频通过 WebSocket 分流出来，做实时双向语音应用。*(部署)*
- **Vapi** [How we built Vapi's voice AI pipeline (part 2)](https://vapi.ai/blog/how-we-built-vapi-s-voice-ai-pipeline-part-2)：为低延迟语音做的 VAD、endpointing、流式 STT 和推理协同。*(部署)*
- **Daily (Pipecat)** [Smart Turn v3, with CPU inference in 12ms](https://www.daily.co/blog/announcing-smart-turn-v3-with-cpu-inference-in-just-12ms/)：一个开源的语义 VAD 轮次检测模型，8MB，23 种语言，CPU 就能跑。*(产品设计)*
- **Slack** [Real-time Messaging](https://slack.engineering/real-time-messaging/)：有状态的 WebSocket 网关加频道服务器，全球范围 500ms 内送达消息。*(部署)*
- **Discord** [How Discord Scaled Elixir to 5,000,000 Concurrent Users](https://discord.com/blog/how-discord-scaled-elixir-to-5-000-000-concurrent-users)：用 Elixir GenServer 承载 session，用 Manifold 做扇出，支撑数百万并发 WebSocket。*(部署)*
