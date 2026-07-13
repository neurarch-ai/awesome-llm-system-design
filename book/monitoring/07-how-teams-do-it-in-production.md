# 7. How Teams Do It in Production

Every production LLM observability system converges on the same skeleton: emit a
cheap trace per request, fan expensive quality checks off that stream
asynchronously and sampled, roll results into rates, and alert on the delta after
any model or prompt change. What actually differs between companies is three
decisions: **trace granularity** (span-per-step or per-message), **quality signal**
(which proxy they trust), and **build-versus-adopt** (custom judge pipeline vs
auto-instrumentation SDK).

## Where the real designs diverge

| System | Trace granularity | Quality signal | Build or adopt | Distinctive choice |
|---|---|---|---|---|
| Datadog (RAG) | span per step with claim-level detail | LLM judge (GPT-4o, two-stage) + deterministic checks | build: custom judge + ETL | flags contradiction vs unsupported separately; quotes the exact offending claim for fast triage |
| Datadog (judge eval) | same spans as above | F1 against human-labeled HaluBench + RAGTruth | build | two-stage prompt (free reasoning then small-model reformat) achieves 0.810 F1; hardest lesson: synthetic benchmark overestimates real-world performance |
| Honeycomb | OTel span-per-step (one line of instrumentation) | SLO on 80% success over 7-day window; thumbs + error taxonomy | adopt: OTel auto-instrument | logs raw LLM output AND corrected executed query separately to separate model error from fixup logic |
| Uber Genie | request + conversation log via Kafka to Hive | Slack 4-way ratings (Resolved / Helpful / Not Helpful / Not Relevant) + Michelangelo ETL LLM judge | build: Michelangelo ETL | adds a document-quality workflow; retrieval quality is capped by source doc quality so the feedback loop improves the knowledge base not just the model |
| Grafana Labs | OTel via OpenLIT SDK (auto-instrument 50+ tools) | GenAI Evaluations dashboard (hallucination, bias, toxicity flags) | adopt: OpenLIT SDK + OTLP gateway | first-class cost dashboard (gen_ai_usage_cost_USD_sum) and TTFT alongside quality; onboarding is four lines of code |
| LangChain / LangSmith | span per tool call (args, result, error) | online LLM-judge + code checks + Insights clustering | build: custom evaluators | encodes every found failure as a permanent offline eval so regressions cannot recur silently |
| Twilio Segment | conversation-id-linked product-analytics events | implicit behavioral signals (component clicks, business events) + engagement | adopt: Segment SDK spec | translates copilot interactions into business events (stock purchase, chart view); rich implicit signal but does not replace grounding or faithfulness checks |

The dividing line is simple: **data and calibration buy the quality ceiling;
tracing granularity and sampling rate buy the cost and detection latency.** A
complete answer picks a point on both and justifies the choice from the product's
risk tolerance and observation budget.

## The systems (first-party writeups)

- **Datadog** [Detect hallucinations in your RAG LLM applications](https://www.datadoghq.com/blog/llm-observability-hallucination-detection/): span-level contradiction and unsupported-claim detection for production RAG apps; flags the exact offending claim. *(product design)*
- **Datadog** [Detecting hallucinations with LLM-as-a-judge](https://www.datadoghq.com/blog/ai/llm-hallucination-detection/): how they built and benchmarked the two-stage GPT-4o judge; 0.810 F1 on HaluBench and RAGTruth. *(eval bar)*
- **Honeycomb** [Improving LLMs in Production With Observability](https://www.honeycomb.io/blog/improving-llms-production-observability): OTel spans capture input, output, errors, latency, tokens, and user feedback for their Query Assistant; 80% success SLO over 7-day windows. *(deployment)*
- **Uber** [Genie: Uber's Gen AI On-Call Copilot](https://www.uber.com/us/en/blog/genie-ubers-gen-ai-on-call-copilot/): a Slack-deployed RAG copilot handling 70,000+ questions; Kafka to Hive pipeline feeds both a Michelangelo ETL judge and a document-quality feedback loop. *(product design)*
- **Grafana Labs** [Monitor LLMs in production with Grafana Cloud, OpenLIT, and OpenTelemetry](https://grafana.com/blog/ai-observability-llms-in-production/): auto-instrument with OpenLIT, route OTLP to managed Prometheus and Tempo, get first-class dashboards for cost, TTFT, and evaluation scores in four lines of code. *(deployment)*
- **LangChain** [The agent improvement loop starts with a trace](https://www.langchain.com/blog/traces-start-agent-improvement-loop): production traces as the input to a continuous improvement loop; encode every failure as a permanent offline eval so it cannot regress silently. *(deployment)*
- **Twilio Segment** [Instrumenting User Insights for your AI Copilot](https://www.twilio.com/en-us/blog/insights/ai/instrumenting-user-insights-for-your-ai-copilot/): a standardized "AI Copilot spec" that stitches conversation events with a stable ID and translates UI interactions into business events for product analytics. *(product design)*

For the full decision matrix with math and the quadrant plot, see the dense
reference at [topics/12-production-monitoring-and-observability.md](../../topics/12-production-monitoring-and-observability.md).
