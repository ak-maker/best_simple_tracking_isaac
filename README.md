# best_simple_tracking — Active Multi-Target Tracking on Isaac Lab

Isaac Lab port of the `best_simple_reward` model-based agent from the
`RL_Active_Multi_Target_Tracking` research repository. Swaps the original
pure-2D `SimpleEnvAtt` /
`MultiRobotEnv` for an Isaac Lab `TrackingEnv` so the agent trains against
quadruped (Go2) locomotion and a PhysX-backed sheep population. The
agent, training loop, reward, optimizer and checkpoint format are
byte-identical to the source repository.

## What's included

```
best_simple_tracking/
├── isaac_env/                        Isaac Lab integration
│   ├── managed_env_cfg.py            ManagerBasedRLEnv cfg (2× Go2 + 5× sheep)
│   ├── scene_cfg.py                  Scene config (cameras, terrain, lights)
│   └── managed_tracking_env.py       TrackingEnv wrapper — exposes the
│                                     original env API (reset/step → mu_real, v, x, done)
├── tracking/                         Agent, verbatim port
│   ├── model_based_agent.py          ModelBasedAgent / ModelBasedAgentAtt
│   ├── policy_net.py                 PolicyNet (MLP)
│   ├── policy_net_att.py             PolicyNetAtt (attention)
│   ├── replay_buffer.py
│   └── utils.py                      SE2_kinematics, landmark_motion, triangle_SDF, ...
├── params/params.yaml                Hyperparameters (A, B, W, FoV, lr, ...)
└── assets/                           Sheep USD + URDF
scripts/
├── train.py                          Headless training entry point
└── eval.py                           Headless evaluation entry point
```

## Sheep dynamics

Linear, identical to `simple_env.py:120–125` in the source repository:
```
mu_new = clip( mu @ A.T + v @ B.T + N(0, W),  -env_half,  +env_half )
v_new  = (rand + bias - 0.5) * landmark_motion_scale
```
The per-episode `bias ∈ [-1, 1]^2` is resampled at every `env.reset()` and
held constant for the rest of the episode (matches `simple_env.py:60`).
Sheep are kinematic rigid bodies in Isaac Sim — they are teleported to the
new `mu_real` once per env step; no PhysX integration between updates.

## Prerequisites

1. **Isaac Sim 4.5+** (recommended pip-install path).
2. **Isaac Lab** (https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
   — clone and install per their docs.

A trained Go2 rough-terrain locomotion policy (`go2_locomotion.pt`, JIT
TorchScript, ~1.2 MB) is **bundled** in `best_simple_tracking/assets/`.
`TrackingEnv` loads it by default. To use your own, pass
`locomotion_policy_path=/path/to/your/policy.pt` to `TrackingEnv`, or use
the `--locomotion-policy` CLI flag in the training / evaluation scripts.

## Install

```bash
# 1) Activate the conda env Isaac Lab installed for you (typically called
#    'isaaclab' if you used the official installer).
conda activate isaaclab

# 2) Install this package (editable):
cd /path/to/best_simple_tracking_isaac
pip install -e .
```

## Run

```bash
# Headless training, full config from params/params.yaml:
cd /path/to/IsaacLab
./isaaclab.sh -p /path/to/scripts/train.py --headless

# Quick test (100 epoch × 10 batch × horizon 25, ~30 min):
./isaaclab.sh -p /path/to/scripts/train.py \
    --headless --max-epoch 100 --batch-size 10 --horizon 25

# CLI flags:
#   --seed              random seed
#   --num-robots        number of trackers (default 2)
#   --num-clusters      sheep spawn clusters (default 2)
#   --clustering-prob   prob of clustered (vs uniform) spawn per episode (default 0.65)
#   --max-epoch         override params.yaml
#   --batch-size        override params.yaml
#   --horizon           override params.yaml
#   --checkpoint-dir    where best/final .pth land (default ./checkpoints)
#   --tensorboard-dir   tb log dir (default ./tensorboard)
#   --resume PATH       resume from saved policy weights
```

Checkpoints are written to `--checkpoint-dir`:
- `best_seed{N}.pth` — best mean reward so far
- `final_seed{N}.pth` — last epoch

Monitor training:
```bash
tensorboard --logdir ./tensorboard
```

## Evaluate a trained checkpoint

```bash
./isaaclab.sh -p path/to/scripts/eval.py --headless \
    --checkpoint ./checkpoints/best_seed0.pth \
    --num-trials 20 \
    --locomotion-policy /path/to/go2_policy.pt
```

Prints per-trial reward and a summary (mean / std / median / min / max) at
the end. Uses the same env config and KF predict as `train.py` —
only the optimizer step is skipped and the policy is in eval mode under
`torch.no_grad()`.

## Env API (matches `SimpleEnvAtt`)

```python
from best_simple_tracking.isaac_env.managed_tracking_env import TrackingEnv

env = TrackingEnv(horizon=50, tau=1.0, A=A, B=B, V=V, W=W, ...)

mu_real, v, x, done = env.reset()
#  mu_real: (1, L, 2)  — sheep positions
#  v:       (1, L, 2)  — sheep velocities
#  x:       (1, R, 3)  — robot poses (x_local, y_local, yaw)
#  done:    (1,) bool

while not done.any():
    action = agent.plan(v, x)            # (R, 2)  v_lin, v_ang
    mu_real, v, x, done = env.step(action)
```

Outputs are CPU tensors with a leading `num_envs` dim (always 1 here). Drop
the leading dim if your agent expects the single-env shapes:
`mu_real[0]`, `v[0]`, `x[0]`.

## Notes on the Isaac Sim integration

- The wrapper runs the inner Isaac Lab env at 200 Hz (50 control sub-steps
  per `tau=1.0` env step) but only updates sheep positions on sub-step 0,
  matching the source's "once per env step" semantics.
- Sheep collide with each other and the Go2 quadrupeds via PhysX, but
  between teleports the wrapper writes zero velocity so PhysX does not
  drift them.
- A soft "go home" boundary (`DOG_BOUND_R=10 m`) prevents the Go2 from
  drifting outside the arena and destabilising the locomotion policy. The
  override is invisible to the high-level agent (it only sees the policy's
  commanded velocity).

## Citation

Please cite the original RL_Active_Multi_Target_Tracking paper / repo if
you build on this. The Isaac Lab integration is a thin wrapper around the
source code.
