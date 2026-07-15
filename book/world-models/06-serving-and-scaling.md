# 6. Serving and scaling

World models are served in two very different regimes, and conflating them is a
design error. Online, on a robot, the model runs inside a real-time control loop
with a hard latency budget. Offline, in a data center, the same model is run in bulk
to generate synthetic data and to evaluate policies, where throughput matters and
latency does not.

## Online: the planning-latency budget

At control time the cost is dominated by the planner, not a single forward pass. A
cross-entropy-method planner evaluates roughly

$$\text{model calls per step} \approx n \times \text{horizon} \times \text{iters}$$

so a modest plan (200 samples, horizon 5, 3 refinement iterations) is about 3000
world-model evaluations for every single action the robot takes. If the control loop
runs at, say, 10 Hz, all of that has to finish in around 100 milliseconds. Three
levers make this feasible:

- **Cheap state.** Latent-dynamics and JEPA models roll forward a small vector, not a
  full frame, which is why they dominate real-time control while generative-video
  models mostly serve offline.
- **Batched rollouts.** The `n` sampled sequences are independent, so evaluate them
  as one batched forward pass on the GPU, not a Python loop.
- **Shorter horizon, more replanning.** Because prediction error compounds (section
  5), a short horizon with frequent replanning is both more accurate and cheaper than
  a long open-loop plan.

## Offline: world models as data and eval engines

The second regime is often the higher-value one in production. A generative world
model that is too slow to plan with is still extremely useful run in bulk:

- **Synthetic data generation.** Generate action-labeled trajectories and rare or
  dangerous scenarios that are expensive to collect on hardware, widening coverage
  for policy training. NVIDIA Cosmos is explicitly positioned for this.
- **Policy evaluation in simulation.** Run a candidate policy against the world model
  (or a simulator built from it) thousands of times to estimate success before
  touching real hardware, which is what turns the expensive real-robot track from
  section 5 into a cheap sim proxy.

Here you scale for throughput: many parallel environments on the GPU (Isaac Lab and
similar GPU-parallel simulators run thousands of environments at once), large batch
sizes, and no per-request latency constraint.

## When to use which serving regime

| Regime | Use when | Tools |
|---|---|---|
| On-robot planning (online) | closed-loop control with a hard per-step latency budget | latent-dynamics or JEPA model, batched CEM/MPC on the robot's GPU |
| Bulk generation (offline) | synthetic data or scenario coverage for policy training | generative video WM (Cosmos), GPU-parallel simulation |
| Bulk policy eval (offline) | scoring many policies cheaply before hardware trials | GPU-parallel simulators (Isaac Lab), the world model as environment |

**Tools.** Online control: the DreamerV3 and V-JEPA 2 action-conditioned models with
a batched cross-entropy-method planner; NVIDIA Isaac GR00T for VLA policies on
humanoids. Offline generation and eval: NVIDIA Cosmos world foundation models, and
GPU-parallel simulators (NVIDIA Isaac Lab, dm_control, ManiSkill) for running
thousands of environments in parallel.

## Bottlenecks

| Bottleneck | Symptom | Fix |
|---|---|---|
| Planner blows the control budget | jerky or late actions on the robot | shrink `n` or horizon, batch rollouts, use a cheaper latent state |
| Compounding rollout error | plans look good but fail over a long horizon | shorten horizon, replan more often, add rollout-drift to eval |
| Sim-to-real gap | high sim success, low real success | domain randomization, better contact physics, more real adaptation data |
| Offline generation too slow | synthetic-data pipeline starves policy training | quantize the generative model, scale horizontally, cache reusable rollouts |
