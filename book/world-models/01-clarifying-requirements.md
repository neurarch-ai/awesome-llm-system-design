# 1. Clarifying the requirements

The phrase "world model" hides at least four different products. The first job of
the interview is to pin which one, because the data, the architecture, and above
all the evaluation change completely between them.

**Candidate:** When you say world model, do you want a model that *predicts the
future*, or one that also *acts*? A pure predictor is judged on prediction; a
world-action model is judged on task success.

**Interviewer:** Assume the end goal is control. We want a robot to accomplish
manipulation tasks it was not explicitly trained on. The world model is a means to
that end.

**Candidate:** Then the north-star metric is downstream task success, not video
quality. Two clarifiers. First, what is the observation and action space:
third-person video, first-person camera plus proprioception, joint torques,
end-effector deltas? Second, where do we evaluate: only in simulation, or on real
hardware too? Those two answers decide almost everything.

**Interviewer:** First-person RGB plus proprioception in, end-effector deltas out.
Evaluate in simulation during development, but the acceptance bar is real-robot
success in a new lab the model has never seen.

**Candidate:** That last constraint is the hard one: zero-shot transfer to a new
real environment. It pushes us toward pretraining a general dynamics model on
large-scale passive video (so it learns physics broadly), then adapting it with a
small amount of action-labeled robot data, rather than training a policy from
scratch in one lab. It also means the evaluation has to measure the sim-to-real
gap explicitly, not just report a simulator number.

## What the answers pin down

| Clarifier | If the answer is... | Consequence |
|---|---|---|
| Predict or act? | act (control) | success rate is the metric; video fidelity is only a proxy |
| Observation space | pixels vs low-dim state | pixels need a tokenizer or latent encoder; low-dim state can use a compact dynamics model |
| Action-labeled data volume | scarce | pretrain on passive video, adapt on the little action data you have |
| Evaluation surface | sim and real | you must report the sim-to-real gap, not a single simulator score |
| Open-loop or closed-loop | closed-loop control | short-horizon prediction with replanning beats long-horizon video accuracy |
| Latency budget on the robot | tens of milliseconds per step | planning compute (rollout count times horizon) becomes a hard design constraint |

## The trap to avoid

The classic wrong answer is to optimize the world model as a video generator:
chase FVD (Frechet Video Distance) and photorealism, show a beautiful predicted
clip, and stop. A model can produce gorgeous, physically plausible video and still
be useless for control if the predicted future does not respond correctly to the
agent's chosen action. Pin the control objective first; every later section is
downstream of it.

## Requirements to write down

- **Functional:** given the current observation and a candidate action sequence,
  predict future states well enough to choose a good next action; execute in
  closed loop; generalize to unseen objects, scenes, and (ideally) new hardware.
- **Non-functional:** per-step planning latency inside the robot's control budget;
  a training data budget dominated by cheap passive video plus a small
  action-labeled set; an evaluation that runs in sim continuously and on real
  hardware at milestones; a reported sim-to-real gap.
