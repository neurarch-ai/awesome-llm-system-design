# 4. Model development

Three architectural decisions define a world model: what the state is, how the
transition is parameterized, and how actions enter. Then a fourth piece, the
planner, turns the trained predictor into a controller.

## The state representation

- **Token grid (generative).** The observation is patch-tokenized (as in a vision
  transformer) and the model predicts the next frame's tokens autoregressively or by
  diffusion. Human-inspectable, expensive, high-fidelity.
- **Recurrent latent (latent-dynamics).** A compact vector carries the state; a
  recurrent core (the recurrent state-space model in the Dreamer line) predicts the
  next latent. Cheap to roll out, which is what makes learning in imagination
  practical.
- **Joint embedding (JEPA).** An encoder maps the observation to an embedding and a
  predictor forecasts the *future embedding*, trained to match the encoder's output
  on the real future. No pixels are reconstructed, which removes the cost of modeling
  irrelevant visual detail.

## Action conditioning

A predictor of the world is not yet controllable. You make it controllable by
feeding the action into the transition, $\hat{s}_{t+1} = f_\theta(s_t, a_t)$, and
training on action-labeled data so the model learns the *effect* of each action.
This is exactly the pretraining-then-adaptation split: the passive-video pretraining
learns $f_\theta(s_t, \cdot)$ as generic dynamics, and the action-conditioned
adaptation (for example V-JEPA 2's action-conditioned variant) learns the dependence
on $a_t$. An unconditioned model can dream plausible futures; only an
action-conditioned one can answer "what if I do this."

## The planner: turning prediction into control

Given an action-conditioned model, planning is search over action sequences: imagine
the rollout for each candidate, score it against the goal, and execute the first
action of the best sequence, then replan (model-predictive control). The
**cross-entropy method** is the standard sampling planner: sample action sequences,
keep the elite fraction by imagined return, refit the sampling distribution to the
elites, and repeat.

```python
import numpy as np
def cem_plan(state, dynamics, reward, horizon=5, n=200, elite=20, iters=3, act_dim=2, seed=0):
    r = np.random.default_rng(seed)
    mu, sig = np.zeros((horizon, act_dim)), np.ones((horizon, act_dim))
    for _ in range(iters):
        acts = mu + sig * r.standard_normal((n, horizon, act_dim))   # sample action sequences
        rets = np.zeros(n)
        for i in range(n):
            s = state
            for t in range(horizon):
                s = dynamics(s, acts[i, t])       # imagined rollout inside the world model
                rets[i] += reward(s)
        idx = rets.argsort()[-elite:]             # keep the elite sequences
        mu, sig = acts[idx].mean(0), acts[idx].std(0) + 1e-6   # refit toward the elites
    return mu[0]                                  # execute the first action, then replan (MPC)
# point mass at (5,5), dynamics s'=s+a, reward=-dist(origin): cem_plan returns an action with both
# components negative, i.e. a step toward the goal.
```

The planner is where the world model pays off: the same trained $f_\theta$ supports
any goal you can write as a reward, without retraining. It is also where the latency
budget bites, because cost is roughly `n * horizon * iters` model evaluations per
control step (section 6).

## Training objectives, by paradigm

- **Generative:** next-frame likelihood (autoregressive cross-entropy over visual
  tokens) or a diffusion denoising loss.
- **Latent-dynamics:** a reconstruction or reward-prediction term plus a latent
  transition (KL) term; the policy is then trained by backpropagating through
  imagined rollouts.
- **JEPA:** predict the future embedding and minimize distance to the (stop-gradient)
  encoder's embedding of the true future, with an anti-collapse mechanism so the
  encoder does not output a constant.

The common failure across all three is **compounding error**: a model accurate for
one step drifts over a long imagined rollout as small errors feed back on
themselves. This is why short horizons with frequent replanning usually beat long
open-loop predictions, and why section 5 measures multi-step rollout drift directly.
