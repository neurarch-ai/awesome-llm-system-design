# 8. Interview Q&A

## Commonly asked

**Q: What is a world model, in one sentence?**
A learned predictor of how an environment evolves given the current state and an
action, so an agent can imagine the consequences of a plan before acting.

**Q: Why not just train a policy directly with reinforcement learning?**
Because real-world action-labeled data is scarce and real rollouts are slow and
risky. A world model lets you pretrain dynamics on abundant passive video and then
plan or learn a policy inside cheap imagined rollouts, spending precious real data
on adaptation and evaluation rather than on training from scratch.

**Q: Why is action-conditioning the crucial step?**
An unconditioned model learns what tends to happen; only a model conditioned on the
agent's action learns what happens *if I do this*. Control needs the second. That is
why systems pretrain on passive video for generic dynamics, then adapt on
action-labeled data to make the model controllable.

## Tricky

**Q: Your video world model has a state-of-the-art FVD. Is it a good world model?**
Not necessarily. FVD measures how realistic the generated video looks, which is the
perception axis. A model can be photoreal and still fail the decision axis if the
predicted future does not respond correctly to the agent's action. Report
action-faithfulness and downstream task success, not just FVD.
**Why the two axes can diverge so completely:** FVD compares the distribution of
generated futures to the distribution of real ones, and a model can match that
distribution while ignoring its action input entirely, since predicting the most
typical continuation is often the easiest way to look realistic. Planning asks
the opposite question, how the future changes across different actions, which is
exactly the signal a typicality-driven objective never rewards.

**Q: Your model plans well in simulation. Why might it fail on the real robot?**
The sim-to-real gap. Simulated contact dynamics, friction, sensor noise, and
lighting differ from reality, so a model that overfits simulator quirks transfers
poorly. Always report success in both settings and the gap between them; domain
randomization and better contact physics narrow it.
**Why the model latches onto simulator quirks:** the simulator's regularities
(idealized friction, noiseless sensors, repeated textures) are cheaper to learn
than true physics, and nothing in the training objective distinguishes them from
real dynamics. Domain randomization works by making those quirks unreliable
across randomized worlds, so the only predictor that survives training is the
physics you actually wanted the model to learn.

**Q: Why prefer a short planning horizon when a longer one sees further?**
Compounding error. A model accurate for one step drifts over a long imagined
rollout as small errors feed back on themselves, so a long open-loop plan is often
less accurate than a short horizon with frequent replanning, and it costs more
compute per control step.
**Why the errors compound rather than average out:** past the first step the
model predicts from its own previous outputs, states it never saw during
training, so each step's error both adds noise and pushes the rollout further
off-distribution, where the model is even less accurate; growth is closer to
multiplicative than additive. Frequent replanning works because each new plan
re-anchors on a real observation, resetting the accumulated drift to zero.

**Q: Model-based and model-free RL both end in a policy that maximizes the same
reward; when does the difference actually matter?**
Both learn from interaction and both can reach the same optimal behavior in the
limit; on a cheap, fast simulator they can look interchangeable. The difference
is what each interaction is spent on. Model-free methods push reward signal
directly into policy or value estimates, so every improvement costs fresh
environment samples; model-based methods spend samples learning the transition
function once, then mint unlimited imagined experience from it. The difference
matters when real samples are expensive, slow, or dangerous (robotics, driving),
which is where world models earn their complexity. It cuts the other way too:
the learned model becomes part of the optimization target, so a policy trained
in imagination will exploit the model's flaws where they are exploitable, and
when a near-perfect simulator is already cheap, model-free's freedom from model
bias can make it the better call.

**Q: MuZero plans without ever reconstructing the environment's observations. How
does it learn a model useful for planning then?**
MuZero (DeepMind) trains a latent dynamics model end-to-end so that only the
quantities planning needs are predicted: from an encoded state it learns a transition
function plus reward, policy, and value heads, and it optimizes these purely so the
predicted rewards, policies, and values match what actually happened along real
trajectories. It has no decoder and never tries to reproduce pixels, so the latent
state is free to discard everything visually salient but decision-irrelevant, and its
tree search plans entirely in that learned latent space. The lesson for world models:
a model can be excellent for control while useless as a video generator, because
reconstruction and decision utility are different objectives. That is the same split
behind JEPA-predictive models like V-JEPA 2 (Meta, 2025), which predict in embedding
space rather than pixel space.

```mermaid
flowchart LR
  O["observation"] --> E["encoder"]
  E --> S["latent state"]
  S --> D["dynamics:<br/>predict next latent"]
  D --> S
  S --> R["reward head"]
  S --> P["policy head"]
  S --> V["value head"]
  N["no decoder:<br/>pixels never reconstructed"] -.-> S
```

*MuZero learns only the quantities planning needs (next latent, reward, policy,
value) and never reconstructs observations, so its state is optimized for decisions
rather than for looking realistic.*

## Commonly answered wrong

**Q: Should the world model run on the robot or in the data center?**
"On the robot" is only half right. On the robot, you run a cheap-state model (latent
or embedding) inside the control loop under a hard latency budget. In the data
center, you run the (often heavier, generative) model in bulk to make synthetic data
and to evaluate policies. Many production programs get more value from the offline
role than the online one.
**Why the split falls out of the constraints:** the control loop has a hard
per-step latency budget that a heavy generative model cannot meet, while
synthetic data generation and policy evaluation are throughput problems with no
latency constraint at all, so they can batch the best available model on cheap
off-peak compute; the same model family ends up in two deployments because the
two jobs bind on different resources.

**Q: How do you evaluate it?**
The wrong answer is a single generative metric. The right answer is two axes,
perception fidelity (rollout drift, plausibility, causal prediction) and decision
utility (action-faithfulness, planning and policy success), measured continuously in
simulation and at milestones on real hardware, with the sim-to-real gap reported
explicitly as the release gate.
**Why one metric cannot stand in for both axes:** perception fidelity and
decision utility are produced by different objectives, so improvement on one
carries no evidence about the other; a gate built on a single axis will pass
models that regressed on the axis it never measured, and the regression surfaces
only on hardware, which is the most expensive place to find it.

**Q: Is a vision-language-action model a world model?**
Not by itself. A VLA maps observation plus goal to actions; it reacts. It becomes a
world-*action* model when you couple it with an explicit predictive model so it can
imagine and plan, not only react.
**Why the distinction is more than terminology:** a reactive mapping can only
reproduce and interpolate the behaviors in its demonstrations, because nothing
in it can score an action it has never seen labeled; a predictive model lets the
system evaluate candidate actions by their imagined consequences, which is the
capability that novel situations demand and imitation alone cannot supply.

**Q: A model that generates realistic long videos is a world model, right?**
Not on its own. A world model must be action-conditioned: given the current state
and an action, it predicts the next state, so a planner can ask "what happens if I do
this?" A pure video generator produces one plausible continuation but not the
counterfactual futures for different actions, which is exactly what planning needs.
That is the difference that makes Genie (DeepMind, 2024) a world model: it learns a
latent action space so the generated frames respond to chosen actions, rather than
rolling forward a single unconditioned continuation. Realism (FVD) measures the
perception axis; action-conditioning is what buys the decision axis. A generator with
no action input is a data source at best, not something you can plan through.
