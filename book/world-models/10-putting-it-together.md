# 10. Putting it together: the complete build

Sections 1 through 6 taught each stage with its options and tradeoffs; section 7
showed where the frontier teams diverge. This capstone makes every choice once:
an opinionated default stack so paradigm paralysis never blocks a first build,
the chapter's scenario committed end to end with the data, planning, and
evaluation arithmetic, the same techniques rebuilt under two other constraint
sets, and the smallest runnable planner, one file, no installs.

## The default stack: start here, deviate with reason

The four paradigms and three data tiers give a first-time builder a dozen
plausible combinations before a single rollout runs. Skip the survey. The stack
below is a sane default for a control-oriented world model; each row names when
to deviate and which section explains why. The paradigms will keep shifting
under you, but the interfaces (data pyramid in, dynamics model out, planner
around it, two-axis eval over it) do not.

| Stage | Default | Deviate when | Why (section) |
|---|---|---|---|
| Paradigm | Latent-dynamics or JEPA-predictive model for control | The product is synthetic data or video, not control: generative-video | [2](02-frame-as-ml-task.md) |
| Pretraining data | Abundant passive video, self-supervised | Low-dim state observations: skip video, learn compact dynamics directly | [3](03-data.md) |
| Adaptation data | Small action-labeled robot set, teleoperation plus failed autonomous rollouts | Demos only look smooth and near-optimal: mix in recovery data deliberately | [3](03-data.md) |
| Simulation | Middle tier for cheap action-labeled rollouts and continuous eval, with domain randomization | Sim quirks leak into the model: raise realism budget or randomize harder | [3](03-data.md), [5](05-evaluation.md) |
| Planner | Batched CEM inside MPC: imagine, score, execute one action, replan | A reactive learned policy meets the task: distill the planner away | [4](04-model-development.md), [6](06-serving-and-scaling.md) |
| Horizon | Short, with replanning every step | Long open-loop plans demanded by the task: budget for compounding error explicitly | [6](06-serving-and-scaling.md) |
| Evaluation | Two axes always: perception fidelity and decision utility, sim continuously, real hardware at milestones | Never drop the decision-utility axis. Video quality is not control quality | [5](05-evaluation.md) |
| Release gate | Real-robot success at the bar and a small reported sim-to-real gap | Never. A single simulator score is not a gate | [5](05-evaluation.md) |

The last two rows are where beginners sink: a beautiful predicted clip feels
like progress, but the chapter's north star is task success, and the only
number that gates a release is measured on hardware the model has never seen.

## The complete build

Return to the scenario from [section 1](01-clarifying-requirements.md):
first-person RGB plus proprioception in, end-effector deltas out, manipulation
tasks the robot was not explicitly trained on, developed in simulation with the
acceptance bar set by real-robot success in a new lab. Here is the whole system
with every choice committed and the reason it won.

| Decision | Choice | Why it won |
|---|---|---|
| Paradigm | JEPA-predictive dynamics with an action-conditioned head | Control is the goal; predicting latents skips pixel rendering the planner never needs |
| Pretraining | Self-supervised on the passive-video tier | Physics is learnable from watching; a million hours of video exists, a million hours of teleop never will |
| Adaptation | Action-conditioning on the scarce robot tier | The only data that teaches what the robot's own actions do; too small to train from scratch |
| Sim tier | Domain-randomized simulator for rollouts and continuous eval | Unlimited action labels at near-zero cost; randomization buys worst-case transfer |
| Planner | Batched CEM in an MPC loop on the robot's GPU | Rollouts are independent, so one batched forward pass fits the control budget |
| Horizon | Short, replan every step | Compounding prediction error makes the long open-loop plan both slower and wronger |
| Eval | Rollout drift and action-faithfulness in sim per commit; real-robot success at milestones | The two axes fail independently; only measuring both catches a pretty-but-uncontrollable model |
| Gate | Real success at the bar, sim-to-real gap reported | Zero-shot transfer to a new lab is the stated acceptance test |

**Data budget.** The pyramid from [section 3](03-data.md) sets the volumes: on
the order of a million hours of passive video for pretraining, unlimited
simulator rollouts in the middle, and tens of hours of action-labeled robot
data for adaptation (the V-JEPA 2 proportions). The recipe is forced, not
chosen: no lab can collect the top tier on hardware, and the bottom tier is too
small to teach physics, so pretrain-then-adapt is the only path through the
pyramid.

**Planning budget.** The [section 6](06-serving-and-scaling.md) cost model:
model calls per control step scale as n x horizon x iterations, so 200 samples
at horizon 5 for 3 CEM iterations is about 3,000 world-model evaluations per
action. At a 10 Hz control loop that is 100 ms for all of them, which is
exactly why the on-robot model rolls a small latent vector (a batched forward
pass over 200 rollouts) while the generative-video sibling stays in the data
center.

**Eval budget.** Sim eval is continuous and effectively free on GPU-parallel
simulators (thousands of environments at once); real-robot trials are the
scarce resource, so they are spent at milestones and on the release gate, not
per commit. The gap between the two numbers (sim success minus real success) is
itself a tracked metric: a growing gap means the model is learning the
simulator, not the world.

**What breaks in month one.** Three signals to wire before the first hardware
milestone: rollout drift in sim (per-step prediction error over the horizon;
when it grows, plans that score well in imagination start failing on execution),
the sim-to-real gap trend (a widening gap flags simulator overfitting before a
milestone fails), and planner latency on the robot (a p99 over the control
budget shows up as jerky or late actions long before anyone profiles it).

## The same techniques under different constraints

The middle column is the build above. The other two keep the same interfaces
(pyramid, dynamics model, planner, two-axis eval) and swap nearly every
implementation choice.

| | Sim-only research agent | Lab-to-real manipulation (this chapter) | Fleet vehicle world model |
|---|---|---|---|
| Goal | Beat benchmarks in simulation | Zero-shot manipulation in a new real lab | Predict road scenes for policy training and eval |
| Paradigm | Latent-dynamics (Dreamer-class), trained in the loop | JEPA-predictive plus action head | Generative-video (GAIA/Cosmos-class) |
| Data | Simulator only; the pyramid collapses to its middle tier | Full pyramid: video pretrain, sim middle, robot adaptation | Fleet camera logs at scale plus sim for rare scenarios |
| Planner | CEM or learned policy in imagination; latency is irrelevant | Batched CEM under a 100 ms budget on the robot | None online; the model runs offline as data and eval engine |
| Real-hardware eval | None | The release gate | Shadow-mode comparison against the driving stack, Illustrative |
| Sim-to-real gap | Undefined and unreported | The tracked gate metric | The product itself: closing scenario coverage the fleet never logged |
| What would be over-engineering | Domain randomization, real-robot milestones | Photoreal generative rendering on the hot path | An on-vehicle planner in the control loop |

Two lessons fall out. First, the sim-only column deletes the hardest parts
(randomization, gap tracking, hardware milestones), which is exactly why a
sim-only result does not transfer as a claim: the deleted parts are where
embodiment lives. Second, the vehicle column shows the offline regime from
[section 6](06-serving-and-scaling.md) standing alone: a world model too slow
to plan with is still the cheapest source of rare-scenario data and policy
evaluation a fleet can buy.

## What each constraint decides

| Your constraint | Lever it moves | Rule of thumb |
|---|---|---|
| Predict or act | Paradigm and metric | Control: success rate is the metric and video fidelity only a proxy; content: generative-video and fidelity metrics |
| Observation space | Model family | Pixels: tokenizer or latent encoder; low-dim state: compact dynamics model, skip video entirely |
| Action-labeled data volume | Training recipe | Scarce (it always is): pretrain on passive video, adapt on the little you have |
| Per-step latency budget | Planner shape | Tens of ms: latent rollouts, batched CEM, short horizon; no budget: plan or generate offline at leisure |
| Horizon the task demands | Replanning cadence | Error compounds per step: keep the horizon short and replan, rather than trusting a long open-loop plan |
| Evaluation surface | The gate | Real hardware in the loop: gate on real success and report the gap; sim-only: say so, and claim nothing about transfer |
| Embodiment diversity | Data coverage | New arm or new kinematics: report which embodiments the data covers; do not assume transfer |
| Sim realism budget | Randomization | Cheaper than photorealism: domain randomization trades average realism for worst-case transfer |

## The smallest runnable planner

Every component of the on-robot loop has a smallest version: the world model
becomes a one-line dynamics function with a deliberately wrong drag constant
(the per-step model error every learned model has), the CEM planner becomes a
few lines of sample-sort-refit, and the robot becomes a 1D point mass. The
shape is the lesson: imagine in the model, act in the world, and watch what
model error does to a plan you trust for too long.

```python
"""Open-loop planning vs MPC replanning inside an imperfect world model."""
import random

DT, GOAL = 0.2, 1.0

def true_step(pos, vel, force):
    """The real environment: a 1D point mass with drag the model gets wrong."""
    vel += (force - 0.5 * vel) * DT
    return pos + vel * DT, vel

def model_step(pos, vel, force, eps):
    """The learned dynamics: drag misestimated by eps, the per-step model error."""
    vel += (force - (0.5 + eps) * vel) * DT
    return pos + vel * DT, vel

def imagine(pos, vel, seq, eps):
    for a in seq:
        pos, vel = model_step(pos, vel, a, eps)
    return abs(GOAL - pos) + 0.1 * abs(vel)          # cost of the imagined endpoint

def cem_plan(pos, vel, horizon, eps, rng, n=60, iters=3, elite=10):
    """Cross-entropy method: sample n sequences, refit on the elite, repeat.
    Cost per control step ~ n x horizon x iters world-model calls (section 6)."""
    mean, std = [0.0] * horizon, [1.5] * horizon
    for _ in range(iters):
        seqs = [[rng.gauss(mean[t], std[t]) for t in range(horizon)] for _ in range(n)]
        seqs.sort(key=lambda s: imagine(pos, vel, s, eps))
        top = seqs[:elite]
        mean = [sum(s[t] for s in top) / elite for t in range(horizon)]
        std = [max(0.1, (sum((s[t] - mean[t]) ** 2 for s in top) / elite) ** 0.5)
               for t in range(horizon)]
    return mean

def run(horizon, eps, replan, seed=0):
    rng = random.Random(seed)
    pos, vel = 0.0, 0.0
    seq = cem_plan(pos, vel, horizon, eps, rng)
    for t in range(horizon):
        if replan and t > 0:             # MPC: fresh real observation, replan the tail
            seq = cem_plan(pos, vel, horizon - t, eps, rng)
            action = seq[0]
        else:                            # open-loop: trust the imagined trajectory
            action = seq[t]
        pos, vel = true_step(pos, vel, action)
    return abs(GOAL - pos)

print(f"final |distance to goal| after executing a horizon-length plan (goal {GOAL}):")
print(f"{'model err':>10} {'horizon':>8} {'open-loop':>10} {'MPC replan':>11}")
for eps in (0.0, 0.4, 0.8):
    for horizon in (4, 8, 16):
        print(f"{eps:>10.1f} {horizon:>8} {run(horizon, eps, False):>10.3f} "
              f"{run(horizon, eps, True):>11.3f}")
```

Run it and the table tells the chapter's story in nine rows. With a perfect
model (error 0.0) the open-loop plan lands within 0.05 of the goal at every
horizon: when imagination matches reality, trusting the plan is fine. With
model error, open-loop misses grow with the horizon (at error 0.8: 0.098 at
horizon 4, 0.453 at 8, 0.823 at 16), which is compounding rollout error made
visible: each imagined step conditions on the last imagined step, so the drag
misestimate accumulates. MPC replanning holds the miss near zero at every error
level and every horizon, because each step starts from a fresh real observation
and the error never gets more than one step to compound. That is
[section 6](06-serving-and-scaling.md)'s rule (short horizon, frequent
replanning) as an experiment you can rerun. Swap `true_step` for a robot,
`model_step` for a learned latent-dynamics model, and `cem_plan` for its
batched GPU version, and you have rebuilt this chapter's control loop.
