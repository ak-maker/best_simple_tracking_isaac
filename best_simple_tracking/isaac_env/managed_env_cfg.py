"""ManagerBasedRLEnv config — extends Isaac Lab's Go2 rough locomotion env.

Adds:
  - 2nd Go2 robot (robot_1)   (1st Go2 keeps key "robot", maps to robot_0 in wrapper)
  - 5 sheep targets (UsdGeom.Mesh-based rigid bodies)
Keeps Isaac Lab's events/rewards/terminations/commands for robot_0.
Wrapper handles robot_1 observations / actions manually.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
)
from isaaclab.envs.mdp import JointPositionActionCfg
from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG

from best_simple_tracking.isaac_env.scene_cfg import (
    ROUGH_GRASS_TERRAIN_CFG,
    SHEEP_USD_PATH,
    make_sheep_target_cfg,
)


@configclass
class ActiveTrackingManagedEnvCfg(UnitreeGo2RoughEnvCfg):
    """2-Go2 + 5 sheep-target version, parameters aligned with the source
    SimpleEnvAtt / best_simple_reward defaults.
    """

    def __post_init__(self):
        super().__post_init__()

        # ModelBasedAgent is single-env. Keep num_envs=1 so the TrackingEnv
        # wrapper's outputs squeeze cleanly to the (L, 2)/(R, 3) shapes the
        # agent expects.
        self.scene.num_envs = 1
        self.scene.env_spacing = 25.0
        # Must be ≥ horizon (50) × tau (1.0) + safety margin.
        # Inner env's time_out terminates at episode_length_s / step_dt inner-steps,
        # which translates to (episode_length_s / tau) TrackingEnv steps.
        # At least 50 TrackingEnv steps → episode_length_s ≥ 50 * 1.0 = 50s.
        self.episode_length_s = 60.0

        # ───── Terrain override: green grass ─────
        self.scene.terrain = ROUGH_GRASS_TERRAIN_CFG

        if hasattr(self, "curriculum") and self.curriculum is not None:
            if hasattr(self.curriculum, "terrain_levels"):
                self.curriculum.terrain_levels = None

        # Inherits Isaac Lab's base_contact termination (Go2 ends episode
        # if its base touches the ground). Trackingenv keeps the high-level
        # velocity commands inside the Go2 locomotion policy's trained
        # range so this termination is rarely triggered.

        # ───── Second Go2 (robot_1) ─────
        self.scene.robot_1 = UNITREE_GO2_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot_1"
        )
        # Its own height_scanner
        self.scene.height_scanner_1 = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/Robot_1/base",
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
            ray_alignment="yaw",
            pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
            debug_vis=False,
            mesh_prim_paths=["/World/ground"],
        )

        # ───── Second Go2 action term ─────
        self.actions.joint_pos_1 = JointPositionActionCfg(
            asset_name="robot_1",
            joint_names=[".*"],
            scale=0.25,
            use_default_offset=True,
        )

        # ───── Reset pose range matches the source env_size = 10 m (±5 m) ─────
        # SimpleEnvAtt resets robots uniformly in [-env_size/2, +env_size/2]
        # on each axis, with yaw uniform in [-pi, pi].
        self.events.reset_base.params["pose_range"] = {
            "x": (-5.0, 5.0), "y": (-5.0, 5.0), "yaw": (-3.14, 3.14)
        }
        # Spawn at rest (matches the source env).
        self.events.reset_base.params["velocity_range"] = {
            "x": (0, 0), "y": (0, 0), "z": (0, 0),
            "roll": (0, 0), "pitch": (0, 0), "yaw": (0, 0)
        }

        # ───── Reset event for robot_1 (also ±5m range) ─────
        self.events.reset_robot_1_base = EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot_1"),
                "pose_range": {"x": (-5.0, 5.0), "y": (-5.0, 5.0), "yaw": (-3.14, 3.14)},
                "velocity_range": {"x": (0, 0), "y": (0, 0), "z": (0, 0),
                                   "roll": (0, 0), "pitch": (0, 0), "yaw": (0, 0)},
            },
        )
        self.events.reset_robot_1_joints = EventTermCfg(
            func=mdp.reset_joints_by_scale,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot_1"),
                "position_range": (1.0, 1.0),
                "velocity_range": (0, 0),
            },
        )

        # ───── 5 sheep targets — matches the source num_landmarks = 5 ─────
        # Initial poses are placeholders only; TrackingEnv.reset() teleports
        # each sheep to the cluster spawn position computed by
        # _generate_clusters() before the first env step. z = 0.3 matches the
        # sheep body center height in the USD.
        target_positions = [(0.0, 3.0, 0.3), (2.0, -3.0, 0.3), (-2.0, -3.0, 0.3),
                            (3.0, 2.0, 0.3), (-3.0, 2.0, 0.3)]
        for i in range(5):
            setattr(self.scene, f"target_{i}", make_sheep_target_cfg(
                prim_path=f"{{ENV_REGEX_NS}}/Target_{i}",
                init_pos=target_positions[i],
            ))

        # ───── Command override settings (wrapper controls vel_cmd) ─────
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.resampling_time_range = (1e9, 1e9)
        self.commands.base_velocity.rel_standing_envs = 0.0

        # ───── Add command term for robot_1 (so it also gets debug arrows) ─────
        import math as _math
        self.commands.base_velocity_1 = mdp.UniformVelocityCommandCfg(
            asset_name="robot_1",
            resampling_time_range=(1e9, 1e9),
            rel_standing_envs=0.0,
            rel_heading_envs=1.0,
            heading_command=False,
            debug_vis=True,
            ranges=mdp.UniformVelocityCommandCfg.Ranges(
                lin_vel_x=(-1.0, 1.0), lin_vel_y=(-1.0, 1.0),
                ang_vel_z=(-1.0, 1.0), heading=(-_math.pi, _math.pi),
            ),
        )

        # ───── obs.actions must reference ONLY robot_0's joint_pos action (12 dim), ─────
        # otherwise after adding joint_pos_1 the ActionManager.action is 24-dim and obs breaks.
        self.observations.policy.actions.params = {"action_name": "joint_pos"}

        # ───── Disable observation noise for inference (matches _PLAY mode) ─────
        self.observations.policy.enable_corruption = False
