# 2. Framing as an ML task

Every world model learns the same object, a transition function that maps a state
and an action to a next state (and often a reward or a next observation):

$$\hat{s}_{t+1} = f_\theta(s_t, a_t)$$

What differs is the space $s$ lives in and how you supervise $f_\theta$. Four
paradigms dominate, and naming them is the framing move.

## The four paradigms

**Generative / video world models.** Predict future *observations* (frames)
directly, conditioned on a reference frame and often on an action or a language
instruction. The state is a grid of visual tokens; the model is a large
transformer or diffusion model. Examples: DeepMind's Genie (action-controllable
generated worlds, arXiv:2402.15391), Wayve's GAIA-1 for driving (arXiv:2309.17080),
OpenAI's positioning of Sora-style video generators as world simulators (2024), and
NVIDIA Cosmos world foundation models. Strength: photoreal, general, great for
synthetic data. Weakness: expensive, and visual fidelity does not guarantee control
usefulness.

**Latent-dynamics (model-based RL) world models.** Learn a compact recurrent latent
state and its transition, then train a policy by "imagining" rollouts entirely
inside the latent model. The canonical line is Ha and Schmidhuber's World Models
(arXiv:1803.10122), Dreamer (DreamerV3, arXiv:2301.04104), and MuZero, which learns
a model only of the quantities needed for planning (arXiv:1911.08265). Strength:
sample-efficient control, cheap rollouts. Weakness: the latent is abstract, so it
transfers less obviously across embodiments.

**JEPA-style predictive world models.** Predict in *representation* space rather
than pixel space: encode observations, then predict the future embedding, avoiding
the cost and distraction of reconstructing every pixel. This is the
joint-embedding-predictive-architecture line (LeCun's 2022 position paper), realized
for video and robots in Meta's V-JEPA 2 (arXiv:2506.09985), whose action-conditioned
variant plans with model-predictive control. Strength: efficient, self-supervised
from passive video, and empirically strong for planning. Weakness: no human-legible
predicted image to inspect.

**VLA / world-action models.** A vision-language-action model maps observation plus
a language goal directly to actions, sometimes with an internal world model,
sometimes without. Examples: OpenVLA (arXiv:2406.09246), Physical Intelligence's
pi-0, and NVIDIA Isaac GR00T. The "world-action model" framing couples a pretrained
world model (imagine) with a fine-tuned action head (act). Strength: end-to-end,
directly deployable. Weakness: without an explicit predictive model, it plans less
and reacts more.

## Compare and contrast: world model vs VLA policy

The two get conflated because from the outside both are "a big pretrained model
that makes a robot act," both consume the same observations, and both lean on
large-scale video pretraining. The mechanics of how an action comes out are
opposite: one predicts futures and selects an action by comparing them, the
other maps straight from observation to action with no future ever computed.

| Dimension | World model (predict, then plan) | VLA policy (direct mapping) |
|---|---|---|
| Inputs at run time | current observation (plus goal for the planner) | same: current observation plus a goal, often in language |
| Pretraining diet | large passive video, then action-conditioned adaptation | large vision-language corpora plus teleoperation demonstrations |
| What the network outputs | a predicted next state (pixels, latent, or embedding) | an action, directly |
| Where the action comes from | a planner scores candidate actions by their imagined outcomes | the forward pass is the decision; no candidate futures exist |
| Handling a situation outside the training data | can still evaluate novel actions by simulating their consequences | must interpolate from demonstrated behavior; nothing scores unseen actions |
| Run-time cost per control step | model rollouts times candidate actions, so latency scales with planning effort | one forward pass, fixed and fast |
| Failure mode | model error is exploited by the planner (plans that work only in imagination) | distribution shift: confident actions in states no demonstration covered |

The difference changes the design when the deployment demands either tight
latency (a fixed forward pass fits a control loop; a planning loop may not) or
generalization beyond the demonstration set (imagination can rank actions no
one ever demonstrated); the production trend of coupling a pretrained world
model with a VLA-style action head exists precisely to buy both.

## When to use which

| Reach for | When | Instead of |
|---|---|---|
| Generative video WM | you need synthetic training data, driving/scene simulation, or a human-inspectable predicted future | latent-dynamics, when you only need control and pixels are wasted compute |
| Latent-dynamics WM | sample-efficient control in a fixed embodiment; cheap imagined rollouts for policy learning | video WMs, when photorealism is irrelevant to the task |
| JEPA-predictive WM | self-supervised pretraining from large passive video, then zero-shot planning on a robot | pixel-reconstructing models, when reconstruction cost buys you nothing for control |
| VLA / world-action model | you want an end-to-end policy that follows language goals and ships to hardware now | a pure predictor, when the deliverable is actions, not forecasts |

**Tools.** Latent-dynamics: the DreamerV3 reference implementation, and MuZero via
open reimplementations (muzero-general, EfficientZero). Sim + RL plumbing: Gymnasium,
Stable-Baselines3, DeepMind's dm_control, and NVIDIA Isaac Lab for GPU-parallel
robot simulation. Generative world models: NVIDIA Cosmos (open world foundation
models plus tokenizers). VLA: the OpenVLA release and NVIDIA Isaac GR00T. Video
encoders reuse the multimodal stack (CLIP (OpenAI), SigLIP (Google), and the V-JEPA
2 weights from Meta).

**Provenance.** The four paradigms trace to distinct lineages. The latent-dynamics
row runs from World Models (Ha and Schmidhuber, 2018) through Dreamer/DreamerV3
(DeepMind) and MuZero (DeepMind); the JEPA-predictive row is V-JEPA 2 (Meta, 2025);
the generative-video row includes NVIDIA Cosmos, Genie (DeepMind, 2024), and GAIA-1
(Wayve, 2023); and the VLA row is OpenVLA (2024) and NVIDIA Isaac GR00T. Run-time
planning on top of these uses model-predictive control (classical control).

**Worked example.** A manipulation team wants a robot to follow spoken instructions
in a warehouse it was never trained in, with a small teleoperation dataset. Because
action-labeled data is scarce and the target is zero-shot transfer, they pretrain a
JEPA-predictive world model on large passive video (learning physical dynamics
broadly), then add an action-conditioned head trained on the little robot data, and
plan with model-predictive control at run time. They keep a generative video WM in
the toolbox for a different purpose, generating synthetic warehouse scenes to widen
coverage, and evaluate the final system by real-robot success rate, not by how
realistic either model's predicted frames look.
