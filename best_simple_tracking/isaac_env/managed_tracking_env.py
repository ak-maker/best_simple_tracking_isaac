"""Wrapper: SimpleEnv API backed by Isaac Lab's ManagerBasedRLEnv.

Vectorised version — all I/O is shape (num_envs, …). When num_envs=1 the
behaviour is identical to the original single-env implementation, so the
ModelBasedAgent that expects shapes (num_landmarks, 2) / (num_robots, 3) can
still be used by squeezing the leading dim.

Parameters ALIGNED with the source best_simple_reward / params.yaml:
  num_robots = 2
  num_landmarks = 5
  tau = 1.0
  V = 0.04  (sensor noise)
  W = 0.0025  (motion noise)
  landmark_motion_scale = 1.0
  init_info = 0.5
  psi = 0.785 (45 deg half-angle)
  radius = 4.0
  kappa = 0.4

Sheep dynamics — linear model (1:1 port of simple_env.py:120-125):
  mu_new = clip( mu @ A.T + v @ B.T + N(0, W),  -env_half,  +env_half )
  v_new  = (rand + bias - 0.5) * landmark_motion_scale
The per-env `bias` is re-sampled from Uniform[-1, 1]^2 at every reset and
held constant for the rest of the episode (matches simple_env.py:60).

Architecture:
  - Inner env: ManagerBasedRLEnv (Isaac Lab Go2 rough env) — handles robot_0 fully
  - Wrapper: manages robot_1 manually (obs construction, policy, actions)
  - Sheep: kinematic rigid bodies, teleported each env step to the new
           mu_real position (no PhysX integration for the targets).
"""

from __future__ import annotations

import math
import os
from typing import Tuple

import gymnasium as gym
import torch

import isaaclab_tasks  # noqa: F401

from best_simple_tracking.isaac_env.managed_env_cfg import ActiveTrackingManagedEnvCfg

# Default Go2 locomotion policy — bundled with the package, see assets/.
DEFAULT_GO2_POLICY = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "assets", "go2_locomotion.pt",
)


def _quat_to_yaw(q: torch.Tensor) -> torch.Tensor:
    """Quaternion (..., 4 = wxyz) -> yaw scalar (...,) about world Z."""
    qw, qx, qy, qz = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return torch.atan2(siny_cosp, cosy_cosp)


def _generate_clusters(num_landmarks: int, num_clusters: int, env_size: torch.Tensor,
                       robot_xy: torch.Tensor, min_robot_dist: float,
                       device: str) -> torch.Tensor:
    """Port of the source _generate_clusters (single env). Caller batches.

    Places num_clusters cluster centers inside the arena with a minimum
    spacing from each other and from robot 0, then scatters num_landmarks
    sheep into the clusters with small Gaussian jitter.
    """
    centers = torch.zeros((num_clusters, 2), device=device)
    env_half = env_size * 0.5
    min_center_sep = min_robot_dist * 0.5

    for i in range(num_clusters):
        best_center = None
        best_min_dist = -1.0
        for _ in range(50):
            candidate = (torch.rand(2, device=device) - 0.5) * env_size
            d_to_robot = torch.linalg.norm(robot_xy - candidate).item()
            if i > 0:
                d_to_centers = torch.linalg.norm(centers[:i] - candidate, dim=1)
                if d_to_centers.min().item() < min_center_sep:
                    continue
            if d_to_robot >= min_robot_dist:
                best_center = candidate
                break
            if d_to_robot > best_min_dist:
                best_min_dist = d_to_robot
                best_center = candidate
        centers[i] = best_center if best_center is not None else torch.clamp(candidate, -env_half, env_half)

    mu = torch.zeros((num_landmarks, 2), device=device)
    points_per = num_landmarks // num_clusters
    rem = num_landmarks % num_clusters
    start = 0
    for i in range(num_clusters):
        cnt = points_per + (1 if i < rem else 0)
        mu[start:start + cnt] = centers[i] + torch.randn((cnt, 2), device=device) * (env_size[0].item() / 15.0)
        start += cnt
    return torch.clamp(mu, -env_half, env_half)


class TrackingEnv:
    """Vectorised Isaac Lab env with N parallel instances of (2 Go2 + 5 sheep).

    All public tensors carry a leading num_envs dim. For backwards-compatible
    use with the single-env agent code, instantiate with the inner env
    config's num_envs=1 — outputs will have a leading 1 the caller can
    squeeze.
    """

    NUM_ROBOTS = 2
    NUM_TARGETS = 5

    def __init__(
        self,
        horizon: int = 50,
        tau: float = 1.0,
        A: torch.Tensor | None = None,
        B: torch.Tensor | None = None,
        V: torch.Tensor | None = None,
        W: torch.Tensor | None = None,
        landmark_motion_scale: float = 1.0,
        psi: torch.Tensor | float = 0.785,
        radius: float = 4.0,
        kappa: float = 0.4,
        init_info: float = 0.5,
        env_size: float = 10.0,
        num_clusters: int = 2,
        clustering_prob: float = 0.65,
        device: str = "cuda:0",
        locomotion_policy_path: str = DEFAULT_GO2_POLICY,
    ):
        self._num_robots = self.NUM_ROBOTS
        self._num_landmarks = self.NUM_TARGETS
        self._horizon = horizon
        self._tau = tau
        self._A = A if A is not None else torch.eye(2)
        self._B = B if B is not None else torch.eye(2)
        self._V = V if V is not None else torch.tensor([0.04, 0.04])
        self._W = W if W is not None else torch.tensor([0.0025, 0.0025])
        self._landmark_motion_scale = landmark_motion_scale
        self._psi = psi
        self._radius = radius
        self._kappa = kappa
        self._init_info = init_info
        self._env_size = torch.tensor([env_size, env_size])
        self._device = device
        self._num_clusters = num_clusters
        self._clustering_prob = clustering_prob
        self._go2_max_vx = 1.0
        self._go2_max_wz = 1.0

        # ───── Build inner env ─────
        cfg = ActiveTrackingManagedEnvCfg()
        cfg.sim.device = device
        self._inner_env = gym.make("Isaac-Velocity-Rough-Unitree-Go2-v0", cfg=cfg).unwrapped

        self._num_envs = int(self._inner_env.scene.num_envs)
        N = self._num_envs

        self._robots = [
            self._inner_env.scene["robot"],
            self._inner_env.scene["robot_1"],
        ]
        self._scanners = [
            self._inner_env.scene.sensors["height_scanner"],
            self._inner_env.scene.sensors["height_scanner_1"],
        ]
        self._targets = [self._inner_env.scene[f"target_{i}"] for i in range(self._num_landmarks)]

        if not os.path.exists(locomotion_policy_path):
            raise FileNotFoundError(f"Go2 policy not found: {locomotion_policy_path}")
        self._go2_policy = torch.jit.load(locomotion_policy_path, map_location=device).eval()

        control_dt = self._inner_env.step_dt
        self._control_steps_per_env_step = max(1, int(round(self._tau / control_dt)))

        # Per-env, per-robot last joint command buffer (used for obs construction
        # of robot_1; robot_0 reads from Isaac Lab's last_action observation term).
        self._last_action = torch.zeros(N, self._num_robots, 12, device=device)

        # Tracking state — all leading num_envs dim.
        self._mu_real: torch.Tensor | None = None     # (N, L, 2)
        self._v: torch.Tensor | None = None            # (N, L, 2)
        self._step_num: torch.Tensor = torch.zeros(N, dtype=torch.long, device=device)

        # ───── Pre-allocated buffers (reused every control sub-step) ─────
        L = self._num_landmarks
        R = self._num_robots
        self._sheep_xyz_buf = torch.zeros(N, L, 3, device=device)
        self._sheep_vel_buf = torch.zeros(N, L, 3, device=device)
        self._dog_xyz_buf = torch.zeros(N, R, 3, device=device)
        # Per-env motion bias for the biased random walk.
        # Re-sampled at every reset to a uniform value in [-1, 1]; biases
        # each env's per-step velocity sampling so the whole flock has a
        # preferred drift direction throughout the episode (matches
        # simple_env.py:60 in the source repo).
        self._landmark_motion_bias = torch.zeros(N, 2, device=device)
        # write_root_velocity_to_sim takes shape (num_envs, 6) per call
        # (per rigid object). For batched write across envs we still call
        # once per sheep, but pass the full (N, 6) batch.
        self._sheep_v6_buf = torch.zeros(N, 6, device=device, dtype=torch.float32)
        self._sheep_pose_buf = torch.zeros(N, 7, device=device, dtype=torch.float32)
        self._env_ids_buf = torch.arange(N, device=device, dtype=torch.long)
        self._env_half = (self._env_size / 2).to(device)

    # ─────────── API: reset ───────────
    def reset(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Reset ALL envs. Returns batched tensors:
          mu_real: (N, L, 2)
          v:       (N, L, 2)
          x:       (N, R, 3)
          done:    (N,) bool
        """
        self._inner_env.reset()
        self._last_action.zero_()

        N = self._num_envs
        L = self._num_landmarks
        env_size_dev = self._env_size.to(self._device)

        # Per-env landmark init. Each env independently rolls clustered
        # (with clustering_prob) vs uniform spawn. Same logic as the source
        # simple_env.py reset().
        mu_per_env = []
        env_origins = self._inner_env.scene.env_origins  # (N, 3)
        robot0_xy_all = self._robots[0].data.root_pos_w[:, :2] - env_origins[:, :2]
        for e in range(N):
            if torch.rand(1).item() < self._clustering_prob:
                robot_xy = robot0_xy_all[e].to(self._device)
                num_clusters = min(self._num_clusters, L)
                min_robot_dist = 0.3 * torch.min(self._env_size).item()
                mu = _generate_clusters(L, num_clusters, env_size_dev,
                                        robot_xy, min_robot_dist, self._device)
            else:
                mu = (torch.rand((L, 2), device=self._device) - 0.5) * env_size_dev
            mu_per_env.append(mu)
        self._mu_real = torch.stack(mu_per_env, dim=0)        # (N, L, 2)
        self._v = torch.zeros((N, L, 2), device=self._device)
        self._step_num.zero_()

        # Re-sample per-env motion bias (per-episode drift bias).
        # Each env gets a constant random drift direction in [-1, 1]^2,
        # which biases the per-step uniform velocity sampling so the whole
        # flock has a preferred drift direction throughout this episode
        # (matches simple_env.py:60 in the original repo).
        self._landmark_motion_bias = (
            torch.rand((N, 2), device=self._device) - 0.5
        ) * 2.0   # in [-1, 1]
        # simple_env.py:61 ALSO initialises velocity at reset (not just at
        # the first step). Match that so the very first step's position
        # update uses a sensible v_0.
        bias_expanded = self._landmark_motion_bias.unsqueeze(1)        # (N, 1, 2)
        self._v = (
            torch.rand((N, L, 2), device=self._device) - 0.5
            + bias_expanded
        ) * self._landmark_motion_scale

        self._write_target_poses(self._mu_real)
        self._write_target_velocities(self._v)
        x = self._read_robot_poses()
        done = torch.zeros(N, dtype=torch.bool, device=self._device)
        return (
            self._mu_real.detach().clone().cpu(),
            self._v.detach().clone().cpu(),
            x.cpu(),
            done.cpu(),
        )

    # ─────────── API: step ───────────
    def step(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        action: (N, R, 2) where last dim = (v_linear, v_angular). For
                backwards compat, (R, 2) is also accepted and broadcast to all envs.
        returns: (mu_real, v, x, done) all leading num_envs dim.
        """
        N = self._num_envs
        R = self._num_robots
        action = action.to(self._device)
        if action.dim() == 2:                 # (R, 2) → (N, R, 2)
            action = action.unsqueeze(0).expand(N, -1, -1)
        action = action.reshape(N, R, 2)

        # PolicyNetAtt (best_simple_reward scaling) outputs v_lin in [0, 2]
        # and v_ang in [-pi/6, pi/6]. Rescale to Go2 locomotion-policy
        # range [-1, 1].
        POLICY_VLIN_MAX = 2.0
        POLICY_VANG_MAX = math.pi / 6.0
        v_lin = (action[..., 0] / POLICY_VLIN_MAX).clamp(-self._go2_max_vx, self._go2_max_vx)  # (N, R)
        v_ang = (action[..., 1] / POLICY_VANG_MAX).clamp(-self._go2_max_wz, self._go2_max_wz)  # (N, R)

        control_dt = self._inner_env.step_dt
        env_half = self._env_half
        done_global = torch.zeros(N, dtype=torch.bool, device=self._device)

        env_origins = self._inner_env.scene.env_origins  # (N, 3)

        # Soft boundary: if a Go2 ever leaves DOG_BOUND_R, override its
        # velocity command to walk back home until it's inside the arena.
        # Without this, a runaway Go2 can drift far enough that the
        # Isaac Lab Go2 locomotion policy goes unstable.
        DOG_BOUND_R = 10.0
        DOG_HOME_VLIN = 1.0    # forward speed to use when "going home"

        for substep_idx in range(self._control_steps_per_env_step):
            # ── 1. read robot poses (need them for boundary check) ──
            poses = self._read_robot_poses()                       # (N, R, 3)
            robot_xy_local = poses[..., :2]
            robot_yaw = poses[..., 2]
            robot_dist = robot_xy_local.norm(dim=-1)               # (N, R)
            out_of_bound = robot_dist > DOG_BOUND_R                # (N, R) bool

            # ── 2. compute v_cmd: policy command, OR home-bound override ──
            # When out of bound, turn fast and stop moving until aligned
            # with home direction; then walk back. The Go2 locomotion
            # policy was trained with v_ang in ±1 rad/s, so we can exceed
            # the policy's ±pi/6 output limit for the home-bound override.
            target_yaw = torch.atan2(-robot_xy_local[..., 1],
                                     -robot_xy_local[..., 0])      # (N, R)
            yaw_err = (target_yaw - robot_yaw + math.pi) % (2 * math.pi) - math.pi
            # Forward speed scales with alignment to home direction:
            # cos(yaw_err) = 1 -> full speed home; cos = 0 (sideways) -> stop;
            # cos < 0 (facing away) -> also stop (don't reverse).
            align = torch.cos(yaw_err).clamp(min=0.0)
            v_lin_home = align * DOG_HOME_VLIN
            v_ang_home = (yaw_err / 0.3).clamp(-1.0, 1.0)
            v_lin_eff = torch.where(out_of_bound, v_lin_home, v_lin)
            v_ang_eff = torch.where(out_of_bound, v_ang_home, v_ang)

            # ── 3. write Go2 velocity commands ──
            cmd_0 = self._inner_env.command_manager.get_term("base_velocity").command   # (N, 3)
            cmd_1 = self._inner_env.command_manager.get_term("base_velocity_1").command  # (N, 3)
            cmd_0[:, 0] = v_lin_eff[:, 0]; cmd_0[:, 1] = 0.0; cmd_0[:, 2] = v_ang_eff[:, 0]
            cmd_1[:, 0] = v_lin_eff[:, 1]; cmd_1[:, 1] = 0.0; cmd_1[:, 2] = v_ang_eff[:, 1]

            # ── 4. read sheep pos + vel from PhysX ──
            sheep_xyz = self._sheep_xyz_buf
            sheep_vel = self._sheep_vel_buf
            for i, tgt in enumerate(self._targets):
                sheep_xyz[:, i, :] = tgt.data.root_pos_w - env_origins
                sheep_vel[:, i, :] = tgt.data.root_lin_vel_w

            # ── 5. Linear sheep dynamics — 1:1 port of simple_env.py:120-125 ───
            # Original (per env step, ONCE):
            #   self._mu_real = clip(landmark_motion_real(mu, v, A, B, W),
            #                        -env_half, env_half)
            #   self._v = (rand + bias - 0.5) * landmark_motion_scale
            # Where landmark_motion_real(mu, v, A, B, W) =
            #   mu @ A.T + v @ B.T + N(0, W)
            #
            # This update runs only on the FIRST control sub-step of each
            # env step (substep counter == 0), because:
            #   - simple_env.py:120 updates once per env step, not per
            #     control sub-step. Running 50× per env step would
            #     accumulate ~50× noise and break the variance scaling.
            #   - Between updates, sheep should stay put: target velocity
            #     is written as zero on every sub-step so PhysX does not
            #     drift them.
            # The update teleports sheep via _write_target_poses,
            # bypassing PhysX integration (matches the source's pure-
            # kinematic update).
            if substep_idx == 0:
                A_dev = self._A.to(self._device)
                B_dev = self._B.to(self._device)
                W_sqrt = torch.sqrt(self._W).to(self._device)            # (2,)

                # 1. compute new position using the OLD velocity:
                #    mu_new = mu @ A.T + v @ B.T + N(0, sqrt(W))
                mu = sheep_xyz[..., :2]                                  # (N, L, 2)
                pos_noise = torch.randn_like(mu) * W_sqrt
                mu_new = mu @ A_dev.T + self._v @ B_dev.T + pos_noise

                # 2. clip to env boundary (paper does this explicitly)
                env_half_dev = (self._env_size * 0.5).to(self._device)
                mu_new = torch.maximum(mu_new, -env_half_dev)
                mu_new = torch.minimum(mu_new,  env_half_dev)

                # 3. resample velocity for the NEXT env step:
                #    v_new = (rand + bias - 0.5) * landmark_motion_scale
                bias_expanded = self._landmark_motion_bias.unsqueeze(1)
                v_new = (
                    torch.rand_like(mu) - 0.5 + bias_expanded
                ) * self._landmark_motion_scale

                # 4. cache for return; teleport sheep in PhysX
                self._mu_real = mu_new.clone()
                self._v = v_new
                new_pos = torch.zeros_like(sheep_xyz)
                new_pos[..., :2] = mu_new
                new_pos[..., 2] = sheep_xyz[..., 2]                      # keep z
                self._write_target_poses(new_pos)
                new_vel = torch.zeros_like(sheep_xyz)                    # PhysX vel = 0
            else:
                # No movement on the other (N_substeps - 1) sub-steps.
                # Velocity stays at zero so PhysX doesn't drift the sheep.
                new_pos = sheep_xyz
                new_vel = torch.zeros_like(sheep_xyz)

            # ── 6. write velocity to PhysX (kept at 0; sheep are kinematic) ──
            self._write_target_velocities(new_vel[..., :2])

            # ── 7. build obs for both robots ──
            # robot_0: Isaac Lab's ObservationManager (cheap, already batched)
            obs_0 = self._inner_env.observation_manager.compute()["policy"]  # (N, 235)
            # robot_1: manual batched obs (matches Isaac Lab layout exactly)
            obs_1 = self._build_locomotion_obs_batched(1, v_lin[:, 1], v_ang[:, 1])  # (N, 235)

            # ── 8. run Go2 policy on both, batched ──
            with torch.no_grad():
                joint_action_0 = self._go2_policy(obs_0)   # (N, 12)
                joint_action_1 = self._go2_policy(obs_1)   # (N, 12)
            self._last_action[:, 0, :] = joint_action_0
            self._last_action[:, 1, :] = joint_action_1
            combined = torch.cat([joint_action_0, joint_action_1], dim=-1)  # (N, 24)

            _, _, terminated, truncated, _ = self._inner_env.step(combined)
            # Isaac Lab returns (N,) bool tensors here.
            done_global = done_global | terminated | truncated

        # Per-env step counter increment (whole tracking step done).
        self._step_num += 1
        done_global = done_global | (self._step_num >= self._horizon)

        x = self._read_robot_poses()
        return (
            self._mu_real.detach().clone().cpu(),
            self._v.detach().clone().cpu(),
            x.cpu(),
            done_global.cpu(),
        )

    # ─────────── Helpers ───────────
    def _read_robot_poses(self) -> torch.Tensor:
        """Return shape (N, R, 3) = (x_local, y_local, yaw_world)."""
        env_origins = self._inner_env.scene.env_origins  # (N, 3)
        poses = []
        for robot in self._robots:
            pos_local = robot.data.root_pos_w[:, :2] - env_origins[:, :2]  # (N, 2)
            yaw = _quat_to_yaw(robot.data.root_quat_w).unsqueeze(-1)        # (N, 1)
            poses.append(torch.cat([pos_local, yaw], dim=-1))                # (N, 3)
        return torch.stack(poses, dim=1)  # (N, R, 3)

    def _write_target_velocities(self, vel_xy: torch.Tensor):
        """Write planar velocity for the sheep rigid bodies. Zero each
        sub-step keeps the sheep kinematic (no PhysX drift between
        teleports).
        """
        env_ids = self._env_ids_buf
        v6 = self._sheep_v6_buf
        for i, tgt in enumerate(self._targets):
            v6.zero_()
            v6[:, 0] = vel_xy[:, i, 0]
            v6[:, 1] = vel_xy[:, i, 1]
            tgt.write_root_velocity_to_sim(v6, env_ids)

    def _write_target_poses(self, mu_xy: torch.Tensor):
        """Write sheep XY positions (z fixed = 0.3) for all envs. mu_xy: (N, L, 2)."""
        env_origins = self._inner_env.scene.env_origins  # (N, 3)
        env_ids = self._env_ids_buf
        pose = self._sheep_pose_buf
        for i, tgt in enumerate(self._targets):
            pose.zero_()
            pose[:, 0] = mu_xy[:, i, 0] + env_origins[:, 0]
            pose[:, 1] = mu_xy[:, i, 1] + env_origins[:, 1]
            pose[:, 2] = 0.3
            pose[:, 3] = 1.0  # qw (identity)
            tgt.write_root_pose_to_sim(pose, env_ids)

    def _build_locomotion_obs_batched(self, idx: int,
                                       v_lin: torch.Tensor,
                                       v_ang: torch.Tensor) -> torch.Tensor:
        """Build 235-d Go2 obs for robot `idx` across all envs.

        v_lin, v_ang: (N,) — commanded velocity for this robot in each env.
        Returns (N, 235).
        """
        robot = self._robots[idx]
        scanner = self._scanners[idx]

        base_lin_vel_b = robot.data.root_lin_vel_b              # (N, 3)
        base_ang_vel_b = robot.data.root_ang_vel_b              # (N, 3)
        projected_gravity_b = robot.data.projected_gravity_b    # (N, 3)
        vel_cmd = torch.stack([v_lin, torch.zeros_like(v_lin), v_ang], dim=-1)  # (N, 3)
        joint_pos_rel = robot.data.joint_pos - robot.data.default_joint_pos       # (N, 12)
        joint_vel_rel = robot.data.joint_vel - robot.data.default_joint_vel       # (N, 12)
        last_action = self._last_action[:, idx, :]                                # (N, 12)
        sensor_z = scanner.data.pos_w[:, 2:3]                                     # (N, 1)
        ray_hits_z = scanner.data.ray_hits_w[..., 2]                               # (N, 187)
        height_scan = (sensor_z - ray_hits_z - 0.5).clamp(-1.0, 1.0)              # (N, 187)
        return torch.cat([
            base_lin_vel_b, base_ang_vel_b, projected_gravity_b, vel_cmd,
            joint_pos_rel, joint_vel_rel, last_action, height_scan,
        ], dim=-1)

    def close(self):
        self._inner_env.close()
