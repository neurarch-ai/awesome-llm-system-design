# 7. How teams do it in production

World models moved from research demo to a named industry program (world-action
models) in 2024 and 2025. Here is where the major efforts diverge, with first-party
sources.

## Meta: V-JEPA 2 (predict in representation space)

Meta's V-JEPA 2 is a self-supervised video world model that predicts in embedding
space rather than pixels. It is pretrained on roughly a million hours of internet
video and then adapted with a comparatively tiny amount (on the order of tens of
hours) of unlabeled robot footage, after which its action-conditioned variant
plans real-robot actions in a new lab zero-shot using model-predictive control.
Meta also released physical-reasoning benchmarks (IntPhys 2, Minimal Video Pairs,
CausalVQA) to measure understanding, not just generation. This is the clearest
public instance of the pretrain-on-video, adapt-on-robots recipe this chapter
describes.
Source: [V-JEPA 2 (arXiv:2506.09985)](https://arxiv.org/abs/2506.09985).

## NVIDIA: Cosmos and GR00T (generate and act)

NVIDIA frames the space as **world-action models**: a pretrained generative world
model (imagine) fine-tuned into an action model (act). Cosmos world foundation
models generate physically grounded video for two production jobs, synthetic
training data and robot policy evaluation in simulation, and serve as the backbone
for downstream WAMs; Isaac GR00T is the vision-language-action model for humanoids.
This is the offline-engine view from section 6: the world model is often more
valuable as a bulk data and evaluation generator than as an on-robot planner.
[Cosmos-Drive-Dreams (arXiv:2506.09042)](https://arxiv.org/abs/2506.09042) is the
concrete instance of the first job: driving data generated at scale by a world
foundation model, which is the clearest published account of what "the world model as
a data engine" actually produces.
Sources: [NVIDIA Cosmos](https://www.nvidia.com/en-us/ai/cosmos/),
[The Rise of World-Action Models](https://developer.nvidia.com/blog/pretrained-to-imagine-fine-tuned-to-act-the-rise-of-world-action-models/),
[Cosmos-Drive-Dreams (arXiv:2506.09042)](https://arxiv.org/abs/2506.09042).

## DeepMind: Genie (action-controllable generated worlds)

DeepMind's Genie line generates *playable* environments: Genie learns
action-controllable world generation from unlabeled video (arXiv:2402.15391), and
Genie 2 generates diverse, playable 3D worlds from a single image. The emphasis is
interactivity, a generated world an agent can act inside, which makes it a training
and evaluation ground rather than an on-robot controller.
Source: [Genie (arXiv:2402.15391)](https://arxiv.org/abs/2402.15391).

## Wayve: GAIA (a domain-specific driving world model)

Wayve's GAIA-1 is a generative world model for autonomous driving that predicts
future driving scenes conditioned on video, text, and action. It is the clearest
example of a *domain-specialized* world model: narrower than a general video model,
but tuned to the states and actions that matter for one embodiment.
[GAIA-2 (arXiv:2503.20523)](https://arxiv.org/abs/2503.20523) is the successor and
sharpens the point: it is controllable and multi-view, which is what a driving stack
actually needs, because a scenario you cannot specify and cannot see from every camera
is not a test case. Sources:
[GAIA-1 (arXiv:2309.17080)](https://arxiv.org/abs/2309.17080),
[GAIA-2 (arXiv:2503.20523)](https://arxiv.org/abs/2503.20523).

## Physical Intelligence and the VLA line

Vision-language-action models take the end-to-end route: map observation plus a
language goal directly to actions. OpenVLA is the open reference point
(arXiv:2406.09246), and Physical Intelligence's pi-0 is a widely cited robot
foundation policy. These ship actions today; the world-action framing is about
adding an explicit predictive model on top so the policy can plan, not only react.
Sources: [OpenVLA (arXiv:2406.09246)](https://arxiv.org/abs/2406.09246),
[Physical Intelligence](https://www.physicalintelligence.company/).

## The 2026 turn: real-time and interactive

The chapter's serving section treats the world model as an offline engine, and that was
the right default when a rollout took seconds per step. A 2026 line of work is attacking
exactly that constraint, and it changes which of the designs above can run online:

- **Streaming interactive generation.** [Matrix-Game 3.5
  (arXiv:2608.29910)](https://arxiv.org/abs/2608.29910) targets real-time streaming
  interactive world models, using patch memory to hold context without paying for it on
  every frame. The interesting part for a serving answer is the memory structure, not
  the generator: it is the same problem a KV cache solves for text, arriving here.
- **Full-stack open implementations.** [minWM
  (arXiv:2605.30263)](https://arxiv.org/abs/2605.30263) and [Nano World Models
  (arXiv:2605.23993)](https://arxiv.org/abs/2605.23993) publish minimal end-to-end
  implementations, which is what makes the latency claims in this area checkable rather
  than quotable.
- **Reproducible evaluation as a platform.** [stable-worldmodel
  (arXiv:2605.21800)](https://arxiv.org/abs/2605.21800) is a platform for reproducible
  world-modeling research and evaluation, which is the same move
  [Inspect](https://inspect.aisi.org.uk/) made for agents: fix the harness so a number
  measures the model rather than the setup.

Treat the latency claims the way section 6 says to treat any of them: ask what the
action space and horizon were, and whether the number is a single step or a full
rollout. "Real-time" in this literature usually means interactive frame generation, not
a planning loop closed on hardware.

## Where they agree and diverge

- **Agree:** pretrain broad (video), adapt narrow (robot); evaluate on embodied
  tasks in sim and, at milestones, on real hardware.
- **Diverge on state:** pixels (Cosmos, Genie, GAIA) for fidelity and synthetic
  data, versus embeddings (V-JEPA 2) or compact latents (Dreamer line) for cheap
  on-robot planning.
- **Diverge on role:** the world model as an *offline* data and eval engine
  (NVIDIA) versus an *online* planner on the robot (Meta's action-conditioned
  V-JEPA 2, the Dreamer line).

For a continuously updated map of this fast-moving area, the community reading list
[Awesome-WAM](https://github.com/OpenMOSS/Awesome-WAM) tracks world-action-model
papers and the embodied evaluation benchmarks (for example WorldArena) referenced in
section 5.
