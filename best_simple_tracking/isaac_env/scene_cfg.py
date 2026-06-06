"""Scene configuration for Active Tracking env (Go2 UGVs + sheep targets + rough grass).

Uses proper configclass pattern with {ENV_REGEX_NS} prim paths for clone_environments.
Sheep target USD is at best_simple_tracking/assets/sheep.usd (converted from MQE URDF).
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR

# Path to sheep USD (converted from MQE sheep.urdf via Isaac Lab convert_urdf.py).
SHEEP_USD_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "assets",
    "sheep.usd",
)


# ───────── Terrain ─────────

ROUGH_GRASS_TERRAIN_CFG = TerrainImporterCfg(
    prim_path="/World/ground",
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
        # 1 tile is enough since we run num_envs=1. (The original 8x8 grid was
        # for vectorised Go2 locomotion training, which needs many terrain
        # variants for curriculum / randomisation. Single-env tracking only
        # needs one 25x25 m patch of grass.)
        size=(25.0, 25.0),
        border_width=20.0,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.1,
        vertical_scale=0.005,
        slope_threshold=0.75,
        color_scheme="height",
        sub_terrains={
            "grass": terrain_gen.HfRandomUniformTerrainCfg(
                proportion=1.0,
                noise_range=(0.01, 0.06),  # 1-6cm bumps (matches Go2 rough training)
                noise_step=0.01,
                border_width=0.25,
            ),
        },
    ),
    use_terrain_origins=False,  # IMPORTANT: use flat env_spacing grid, not terrain tiles
    max_init_terrain_level=None,
    collision_group=-1,
    physics_material=sim_utils.RigidBodyMaterialCfg(
        friction_combine_mode="multiply",
        restitution_combine_mode="multiply",
        static_friction=1.0,
        dynamic_friction=1.0,
    ),
    visual_material=sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.18, 0.5, 0.12),
        roughness=0.9,
    ),
)


# ───────── Go2 UGV ─────────

def make_go2_cfg(prim_path: str, init_pos: tuple[float, float, float],
                 diffuse_color: tuple[float, float, float]) -> ArticulationCfg:
    """Unitree Go2 with gravity enabled (real walking via locomotion policy)."""
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd",
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=4,
                solver_velocity_iteration_count=0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=diffuse_color,
                metallic=0.3,
                roughness=0.5,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=init_pos,
            joint_pos={
                ".*L_hip_joint": 0.1,
                ".*R_hip_joint": -0.1,
                "F[L,R]_thigh_joint": 0.8,
                "R[L,R]_thigh_joint": 1.0,
                ".*_calf_joint": -1.5,
            },
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "base_legs": DCMotorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                effort_limit=23.5,
                saturation_effort=23.5,
                velocity_limit=30.0,
                stiffness=25.0,
                damping=0.5,
                friction=0.0,
            ),
        },
    )


# ───────── Sheep Target ─────────

def make_sheep_target_cfg(prim_path: str, init_pos: tuple[float, float, float]) -> RigidObjectCfg:
    """Target = sheep USD (Mesh-based).

    Dynamic body so PhysX naturally resolves sheep-sheep and sheep-Go2 collisions.
    Gravity disabled (we'd rather sheep float at z=0.3 than fall into terrain bumps).
    Collision shape is a bounding sphere of the visible body sphere (r=0.20).
    Position is rewritten each env step by managed_tracking_env._write_target_poses
    (the linear sheep update teleports them); velocity is set to zero between
    teleports so PhysX does not drift them.
    """
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=SHEEP_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                kinematic_enabled=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=5.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=init_pos),
        collision_group=-1,
    )


# ───────── Full Scene ─────────

@configclass
class ActiveTrackingSceneCfg(InteractiveSceneCfg):
    """Scene: rough grass terrain + 2 Go2 + 3 sheep targets."""

    # terrain (shared across envs)
    terrain: TerrainImporterCfg = ROUGH_GRASS_TERRAIN_CFG

    # robots (2 Go2)
    robot_0: ArticulationCfg = make_go2_cfg(
        prim_path="{ENV_REGEX_NS}/Robot_0",
        init_pos=(-3.0, 0.0, 0.5),
        diffuse_color=(0.0, 0.4, 1.0),
    )
    robot_1: ArticulationCfg = make_go2_cfg(
        prim_path="{ENV_REGEX_NS}/Robot_1",
        init_pos=(3.0, 0.0, 0.5),
        diffuse_color=(0.0, 0.3, 0.9),
    )

    # height scanners (one per Go2, matches rough locomotion training)
    height_scanner_0: RayCasterCfg = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot_0/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    height_scanner_1: RayCasterCfg = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot_1/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    # targets (5 sheep, matches num_landmarks=5 in env).
    # z=0.3 matches MQE sheep_origin z and the URDF collision cylinder height.
    # Initial positions are overwritten by env.reset() before training begins.
    target_0: RigidObjectCfg = make_sheep_target_cfg(
        prim_path="{ENV_REGEX_NS}/Target_0", init_pos=(0.0, 3.0, 0.3),
    )
    target_1: RigidObjectCfg = make_sheep_target_cfg(
        prim_path="{ENV_REGEX_NS}/Target_1", init_pos=(2.0, -3.0, 0.3),
    )
    target_2: RigidObjectCfg = make_sheep_target_cfg(
        prim_path="{ENV_REGEX_NS}/Target_2", init_pos=(-2.0, -3.0, 0.3),
    )
    target_3: RigidObjectCfg = make_sheep_target_cfg(
        prim_path="{ENV_REGEX_NS}/Target_3", init_pos=(3.0, 2.0, 0.3),
    )
    target_4: RigidObjectCfg = make_sheep_target_cfg(
        prim_path="{ENV_REGEX_NS}/Target_4", init_pos=(-3.0, 1.0, 0.3),
    )
