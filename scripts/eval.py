"""Evaluate a trained ModelBasedAgent policy on Isaac Lab TrackingEnv.

Mirrors the training rollout in `train.py`, but skips the optimizer step
and instead reports mean / std / median reward across several test
episodes. Same env, same agent, same KF predict step — just frozen
weights.

Usage:
    ./isaaclab.sh -p path/to/scripts/eval.py \\
        --checkpoint /path/to/best_seedX.pth \\
        --num-trials 20 --horizon 50 \\
        --locomotion-policy /path/to/go2_policy.pt
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True,
                    help="Path to a saved policy .pth (best_seedX.pth or final_seedX.pth).")
parser.add_argument("--num-trials", type=int, default=10,
                    help="Number of test episodes to roll out.")
parser.add_argument("--horizon", type=int, default=None,
                    help="Per-episode step horizon; defaults to params.yaml.")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num-robots", type=int, default=2)
parser.add_argument("--num-clusters", type=int, default=2)
parser.add_argument("--clustering-prob", type=float, default=0.65)
parser.add_argument("--params-yaml", type=str,
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "..", "best_simple_tracking", "params",
                                         "params.yaml"))
parser.add_argument("--locomotion-policy", type=str, default=None,
                    help="Path to the Go2 rough-terrain locomotion policy (.pt JIT).")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import yaml
import numpy as np
import torch
from torch import tensor

from best_simple_tracking.tracking import ModelBasedAgentAtt
from best_simple_tracking.isaac_env.managed_tracking_env import TrackingEnv

sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

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
        A=A, B=B, W=W, radius=radius, psi=psi, kappa=kappa, V=V, lr=0.0,
        num_robots=args.num_robots, uncertainty_threshold=15.0,
        tau=tau,
    )
    agent.load_policy_state_dict(args.checkpoint)
    agent.eval_policy()
    print(f"[INFO] checkpoint = {args.checkpoint}")
    print(f"[INFO] {args.num_trials} test episodes, horizon = {horizon}")

    rewards = np.empty(args.num_trials)
    with torch.no_grad():
        for k in range(args.num_trials):
            mu_real, v, x, done = env.reset()
            mu_real = mu_real.squeeze(0); v = v.squeeze(0)
            x = x.squeeze(0); done = bool(done.squeeze(0).item())
            num_landmarks = mu_real.size()[0]
            agent.reset_estimate_mu(mu_real)
            agent.reset_agent_info()
            while not done:
                action = agent.plan(v, x)
                mu_real, v, x, done = env.step(action)
                mu_real = mu_real.squeeze(0); v = v.squeeze(0)
                x = x.squeeze(0); done = bool(done.squeeze(0).item())
                agent.update_info_mu(mu_real, x)
            ep_reward = agent.update_policy_grad(train=False) / num_landmarks
            rewards[k] = ep_reward
            print(f"[trial {k+1:3d}/{args.num_trials}] reward = {ep_reward:+.3f}")

    print()
    print("=" * 50)
    print(f"Mean reward   : {rewards.mean():+.3f}")
    print(f"Std reward    : {rewards.std():+.3f}")
    print(f"Median reward : {float(np.median(rewards)):+.3f}")
    print(f"Min / Max     : {rewards.min():+.3f}  /  {rewards.max():+.3f}")
    print("=" * 50)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
