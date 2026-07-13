# 7. How teams do it in production

Every serious LLM team converges on the same skeleton: an offline suite gates the
change, an online loop checks the gate was honest, and the gap between them
recalibrates the suite. What actually differs between companies is two decisions:
**how they score open-ended output** (which judge, how validated, how calibrated)
and **what they use as the online ground truth** (A/B, shadow, canary, human
sign-off). The architecture everyone shares; the leverage is in the calibration
and the online proof.

## Where the real designs diverge

| System | Offline signal | Judge calibration | Online proof | Gating | When it wins | Watch out |
|---|---|---|---|---|---|---|
| DoorDash (chatbot) | Simulated multi-turn conversations | LLM judge calibrated to human grades | Pre-release quality bar | Sim flywheel gates before release | Multi-turn agents where pre-launch real traffic is scarce | Simulator drifts from real user behavior; a sim pass can overstate readiness |
| DoorDash (AutoEval, search) | Whole-page relevance scoring | Fine-tuned LLM raters, human in the loop | Search result page quality | Human-in-loop grades gate | Whole-page relevance where per-item labels miss the gestalt | Fine-tuned rater needs retraining as catalog and query mix shift |
| GitHub Copilot | 4000+ offline tests, ~100 broken repos | LLM judge for chat, plus manual review | Internal canary on live Hubbers | CI gate, daily regression vs prod | Large stable code-task suites with executable pass/fail signal | Thousands of cases are costly to maintain; broken-repo fixtures go stale |
| Spotify | Offline evals as pre-experiment funnel | Judge recalibrated when it misses A/B | A/B user outcome | Evals filter before A/B; gap drives recalibration | High A/B volume where offline can cheaply filter candidates first | Needs enough A/B throughput to recalibrate; the offline-online gap must be watched |
| Thomson Reuters | Public benchmarks + semi-automated task eval | Human A/B as final arbiter | Human preference A/B | Three-stage gate, human sign-off | High-stakes expert domains (legal) where errors are costly | Human sign-off is slow and expensive; does not scale to daily edits |
| Uber (uReview) | Generated-comment scoring | LLM grader with confidence scores | Posted-comment usefulness | Confidence threshold gates what posts | High-volume generation where low-confidence output can be suppressed | Confidence scores need calibration; a bad threshold floods or starves output |
| Discord | Critic-LLM prompt review | Critic LLM assists human review | A/B rollout outcome | Critic screen before A/B | Fast prompt iteration where a critic cheaply catches obvious regressions | Critic is advisory, not a hard gate; weak prompts can still reach A/B |
| Ramp | Judge on agent classifications | LLM judge vs human labels | Shadow mode on live traffic | Shadow plus judge before rollout | Agent actions you can run silently on real traffic before enabling them | Shadow mode needs traffic and infra to mirror; yields no user signal yet |
| GitLab Duo | CEF prompt library, daily regression | LLM judge at scale | Model comparison across iterations | Daily automated regression | Many models and prompts compared under one shared harness | Judge drift hits every team at once; framework is upfront investment |
| Booking.com | Golden datasets | LLM-as-judge for monitoring | Production quality monitoring | Judge plus golden set watches drift | Ongoing production monitoring to catch drift on a stable task | Golden sets go stale; monitoring catches drift after it ships |
| Pinterest | Fine-tuned open-source relevance judge | 73.7% exact match vs human (XLM-RoBERTa) | A/B search ranking experiments | Stratified per-category + FDR control | High-volume search ranking where fine-tuned judge beats general model | Fine-tuned judge needs retraining as query mix and catalog shift |

## The dividing line

The dividing line is simple: **judge calibration and the online proof buy the
gate's trustworthiness; the golden set and task metrics buy the cost efficiency.**
A complete answer picks a point on both axes and justifies it from the task's
checkability (can you measure it without a judge?) and the change's blast radius
(how bad is a regression that ships?).

## The systems (first-party write-ups)

- **DoorDash** [A Simulation and Evaluation Flywheel to Develop LLM Chatbots at Scale](https://careersatdoordash.com/blog/doordash-simulation-evaluation-flywheel-to-develop-llm-chatbots-at-scale/): Simulated multi-turn conversations graded by an LLM judge calibrated to humans before release.
- **DoorDash** [How DoorDash leverages LLMs to evaluate search result pages](https://careersatdoordash.com/blog/doordash-llms-to-evaluate-search-result-pages/): AutoEval: fine-tuned LLM raters with a human in the loop for whole-page relevance.
- **GitHub** [How we evaluate AI models and LLMs for GitHub Copilot](https://github.blog/ai-and-ml/generative-ai/how-we-evaluate-models-for-github-copilot/): 4000+ offline tests plus broken-repo fixing, LLM-as-judge for chat, and daily regression vs production.
- **Spotify** [Better experiments with LLM evals: a funnel, not a fork](https://engineering.atspotify.com/2026/5/better-experiments-with-llm-evals-a-funnel-not-a-fork): Offline evals calibrated against online A/B as a sequential funnel.
- **Spotify** [Profile-aware LLM-as-a-Judge for Podcasts](https://research.atspotify.com/2025/9/profile-aware-llm-as-a-judge-for-podcasts-a-better-middle-ground-between): An LLM judge bridging offline metrics and costly A/B tests for podcast quality.
- **Thomson Reuters** [Efficiently evaluating LLMs for legal tasks](https://legal.thomsonreuters.com/blog/evaluating-llms-legal-tasks/): Three-stage gate: public benchmarks, semi-automated task eval, then human A/B sign-off.
- **Uber** [uReview: scalable, trustworthy GenAI for code review](https://www.uber.com/us/en/blog/ureview/): An LLM grader scores generated comments; confidence thresholds gate what gets posted.
- **Discord** [Developing Rapidly with Generative AI](https://discord.com/blog/developing-rapidly-with-generative-ai): A critic-LLM AI-assisted eval of prompts before A/B rollout.
- **Ramp** [How Ramp Fixes Merchant Matches with AI](https://builders.ramp.com/post/fixing-merchant-classifications-with-ai): Shadow mode plus an LLM judge evaluate agent classifications against humans pre-rollout.
- **Microsoft** [LLM-Rubric: a multidimensional, calibrated approach to automated evaluation](https://www.microsoft.com/en-us/research/publication/llm-rubric-a-multidimensional-calibrated-approach-to-automated-evaluation-of-natural-language-texts/): A calibrated multi-dimension rubric judge predicts human satisfaction for a dialogue system.
- **GitLab** [Developing GitLab Duo: validating and testing AI models at scale](https://about.gitlab.com/blog/developing-gitlab-duo-how-we-validate-and-test-ai-models-at-scale/): A central eval framework with an LLM judge runs daily regression across dozens of features.
- **Booking.com** [LLM Evaluation: practical tips at Booking.com](https://booking.ai/llm-evaluation-practical-tips-at-booking-com-1b038a0d6662): LLM-as-judge plus golden datasets for ongoing production quality monitoring.
- **Pinterest** [LLM-Powered Relevance Assessment for Pinterest Search](https://medium.com/pinterest-engineering/llm-powered-relevance-assessment-for-pinterest-search-b846489e358d): Fine-tuned XLM-RoBERTa judge labels search relevance to evaluate ranking A/B experiments at scale.
- **Honeycomb** [So we shipped an AI product. Did it work?](https://www.honeycomb.io/blog/we-shipped-ai-product): Post-launch product eval via activation and adoption metrics, minimal offline gate.
- **Instacart** [Scaling Catalog Attribute Extraction with Multi-modal LLMs](https://company.instacart.com/tech-innovation/scaling-catalog-attribute-extraction-with-multi-modal-llms): LLM-as-judge auto-eval monitors attribute-extraction quality alongside human auditors.
- **LinkedIn** [How we engineered LinkedIn's Hiring Assistant](https://www.linkedin.com/blog/engineering/ai/how-we-engineered-linkedins-hiring-assistant): A quality framework pairs product policy with LLM judges scoring coherence and factuality.
- **Wayfair** [How AI understands what you're looking for](https://www.aboutwayfair.com/careers/tech-blog/smarter-shopping-starts-here-how-ai-understands-what-youre-looking-for): LLM-as-judge validation tasks periodically evaluate AI-generated customer interests offline.

For the full reference (divergence math, quadrant plot, all case studies):
[topics/06-evaluation-system.md](../../topics/06-evaluation-system.md).
