"""Train the ModelBasedAgent on Isaac Lab TrackingEnv.

This is a port of
  RL_Active_Multi_Target_Tracking/best_simple_reward/scripts/run_model_based_training.py
that swaps the source pure-2D SimpleEnvAtt / MultiRobotEnv for an Isaac Lab
TrackingEnv (Go2 quadruped locomotion + linear sheep dynamics). Everything
outside the env (agent, training loop, reward, optimizer, checkpoint format)
is byte-identical to the source run_model_based_training.py.

Sheep follow the linear dynamics from simple_env.py:120-125:
    mu_new = clip( mu @ A.T + v @ B.T + N(0, W),  -env_half,  +env_half )
    v_new  = (rand - 0.5 + bias) * landmark_motion_scale

Default config matches params/params.yaml (mirrors best_simple_reward).

Usage (headless training, no rendering):
    cd /path/to/IsaacLab
    ./isaaclab.sh -p path/to/scripts/train.py --headless

Quick test (100 epoch x 10 batch x horizon 25, ~30 min):
    ./isaaclab.sh -p path/to/scripts/train.py \\
        --headless --max-epoch 100 --batch-size 10 --horizon 25
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# ───── CLI ─────
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--num-robots", type=int, default=2)
parser.add_argument("--num-clusters", type=int, default=2,
                    help="Number of spawn clusters for sheep initial positions.")
parser.add_argument("--clustering-prob", type=float, default=0.65,
                    help="Per-episode prob of clustered (vs uniform) spawn.")
parser.add_argument("--max-epoch", type=int, default=None,
                    help="Override params.yaml max_epoch")
parser.add_argument("--batch-size", type=int, default=None,
                    help="Override params.yaml batch_size")
parser.add_argument("--horizon", type=int, default=None,
                    help="Override params.yaml horizon")
parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints")
parser.add_argument("--tensorboard-dir", type=str, default="./tensorboard")
parser.add_argument("--params-yaml", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "..", "best_simple_tracking", "params",
                                         "params.yaml"))
parser.add_argument("--resume", type=str, default=None,
                    help="Resume from policy weights .pth")
parser.add_argument("--locomotion-policy", type=str, default=None,
                    help="Path to the Go2 rough-terrain locomotion policy (.pt JIT). "
                         "If not given, falls back to the DEFAULT_GO2_POLICY constant "
                         "in best_simple_tracking/isaac_env/managed_tracking_env.py.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

# Launch Isaac Sim BEFORE importing anything that touches USD/Omni.
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# ───── normal imports below this line ─────
import yaml
import time
import gc
import os as _os
import numpy as np
import torch
from torch import tensor
from torch.utils.tensorboard import SummaryWriter

from best_simple_tracking.tracking import ModelBasedAgentAtt
from best_simple_tracking.isaac_env.managed_tracking_env import TrackingEnv

# Line-buffer stdout/stderr so each `print(...)` reaches the log immediately.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

try:
    import psutil
    _PROC = psutil.Process(_os.getpid())
    def _rss_gb():
        return _PROC.memory_info().rss / (1024 ** 3)
except ImportError:
    def _rss_gb():
        return -1.0

torch.manual_seed(args.seed)


def main():
    with open(args.params_yaml) as f:
        params = yaml.safe_load(f)

    max_num_landmarks = params["max_num_landmarks"]
    horizon = args.horizon if args.horizon is not None else params["horizon"]
    tau = params["tau"]

    A = torch.zeros((2, 2))
    A[0, 0] = params["motion"]["A"]["_1"]
    A[1, 1] = params["motion"]["A"]["_2"]
    B = torch.zeros((2, 2))
    B[0, 0] = params["motion"]["B"]["_1"]
    B[1, 1] = params["motion"]["B"]["_2"]
    W = torch.zeros(2)
    W[0] = params["motion"]["W"]["_1"]
    W[1] = params["motion"]["W"]["_2"]
    landmark_motion_scale = params["motion"]["landmark_motion_scale"]

    init_info = params["init_info"]
    radius = params["FoV"]["radius"]
    psi = tensor([params["FoV"]["psi"]])
    kappa = params["FoV"]["kappa"]
    V = torch.zeros(2)
    V[0] = params["FoV"]["V"]["_1"]
    V[1] = params["FoV"]["V"]["_2"]

    lr = params["lr"]
    max_epoch = args.max_epoch if args.max_epoch is not None else params["max_epoch"]
    batch_size = args.batch_size if args.batch_size is not None else params["batch_size"]

    # ───── env (Isaac Lab) + agent (best_simple_reward) ─────
    env_kwargs = dict(
        horizon=horizon, tau=tau, A=A, B=B, V=V, W=W,
        landmark_motion_scale=landmark_motion_scale, psi=psi, radius=radius,
        num_clusters=args.num_clusters, clustering_prob=args.clustering_prob,
        device="cuda:0",
    )
    if args.locomotion_policy is not None:
        env_kwargs["locomotion_policy_path"] = args.locomotion_policy
    env = TrackingEnv(**env_kwargs)

    agent = ModelBasedAgentAtt(
        max_num_landmarks=max_num_landmarks, init_info=init_info,
        A=A, B=B, W=W, radius=radius, psi=psi, kappa=kappa, V=V, lr=lr,
        num_robots=args.num_robots, uncertainty_threshold=15.0,
        tau=tau,
    )
    print(f"[INFO] sheep dynamics = linear, filter = KF")

    if args.resume is not None and os.path.exists(args.resume):
        agent.load_policy_state_dict(args.resume)
        print(f"[resume] loaded {args.resume}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.tensorboard_dir, exist_ok=True)
    best_path = os.path.join(args.checkpoint_dir, f"best_seed{args.seed}.pth")
    final_path = os.path.join(args.checkpoint_dir, f"final_seed{args.seed}.pth")
    writer = SummaryWriter(args.tensorboard_dir)

    print(f"[config] max_epoch={max_epoch}  batch_size={batch_size}  horizon={horizon}")
    print(f"[config] num_robots={args.num_robots}  num_clusters={args.num_clusters}")
    total_steps = max_epoch * batch_size * horizon
    print(f"[estimate] total tracking steps = {total_steps:,}")

    # ───── training loop (verbatim from source) ─────
    agent.train_policy()
    reward_list = np.empty((max_epoch, batch_size))
    best_reward = 1.0
    t0 = time.time()

    for i in range(max_epoch):
        agent.set_policy_grad_to_zero()

        for j in range(batch_size):
            mu_real, v, x, done = env.reset()
            # Env is vectorised (leading num_envs dim); agent expects the
            # single-env shapes (L, 2) / (R, 3) / scalar bool. Squeeze for
            # num_envs=1 (the only supported size for the source agent).
            mu_real = mu_real.squeeze(0)
            v = v.squeeze(0)
            x = x.squeeze(0)
            done = bool(done.squeeze(0).item())
            num_landmarks = mu_real.size()[0]
            agent.reset_estimate_mu(mu_real)
            agent.reset_agent_info()
            while not done:
                action = agent.plan(v, x)
                mu_real, v, x, done = env.step(action)
                mu_real = mu_real.squeeze(0)
                v = v.squeeze(0)
                x = x.squeeze(0)
                done = bool(done.squeeze(0).item())
                agent.update_info_mu(mu_real, x)

            reward_list[i, j] = agent.update_policy_grad() / num_landmarks
            writer.add_scalar("Reward/per_episode", reward_list[i, j],
                              i * batch_size + j)

        agent.policy_step(debug=False)

        mean_reward = float(np.mean(reward_list[i]))
        median_reward = float(np.median(reward_list[i]))
        writer.add_scalar("Reward/epoch_mean", mean_reward, i)
        writer.add_scalar("Reward/epoch_median", median_reward, i)

        elapsed = time.time() - t0
        eta_sec = elapsed * (max_epoch - i - 1) / (i + 1)
        rss = _rss_gb()
        cuda_gb = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0
        print(f"[epoch {i+1:4d}/{max_epoch}] reward mean={mean_reward:+7.3f} "
              f"median={median_reward:+7.3f}  "
              f"elapsed={elapsed/3600:.2f}h  ETA={eta_sec/3600:.2f}h  "
              f"RSS={rss:.2f}GB  CUDA_peak={cuda_gb:.2f}GB")

        if mean_reward > best_reward:
            torch.save(agent.get_policy_state_dict(), best_path)
            best_reward = mean_reward
            print(f"           ↑ new best, saved → {best_path}")

        # Force GC + CUDA cache release each epoch — prevents Isaac Lab's
        # 200Hz physics buffers and PyTorch's autograd graph from
        # accumulating across batches.
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        writer.flush()

    torch.save(agent.get_policy_state_dict(), final_path)

    total = time.time() - t0
    hh, rem = divmod(int(total), 3600)
    mm, ss = divmod(rem, 60)
    print(f"[done] final weights → {final_path}")
    print(f"[time] total wall-clock = {hh:d}h {mm:02d}m {ss:02d}s "
          f"({total:.0f} sec)")
    print(f"[time] mean per epoch = {total / max_epoch:.1f} sec")
    print(f"[time] best reward = {best_reward:.3f} → {best_path}")
    writer.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
