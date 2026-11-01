# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration for the velocity-tracking locomotion environments on rough terrain."""

from dataclasses import MISSING

import instinctlab.envs.mdp as instinct_mdp
from instinctlab.envs.manager_based_rl_env_cfg import InstinctLabRLEnvCfg
from instinctlab.monitors import MonitorTermCfg as MonitorTerm
from instinctlab.motion_reference import MotionReferenceManagerCfg
from instinctlab.terrains.terrain_generator_cfg import FiledTerrainGeneratorCfg
from instinctlab.terrains.terrain_importer_cfg import TerrainImporterCfg
from instinctlab.terrains.trimesh import MotionMatchedTerrainCfg

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg

# from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg

# from isaaclab.terrains import TerrainImporterCfg
# from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import UniformNoiseCfg as Unoise
from isaaclab_newton.physics import (
    KaminoPADMMSolverCfg,
    MJWarpSolverCfg,
    NewtonCfg,
    NewtonCollisionPipelineCfg,
    NewtonShapeCfg,
)
from isaaclab_ov.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab_tasks.utils import PresetCfg

from . import mdp

PROPRIO_HISTORY_LENGTH = 8

##
# Physics backend presets
##


@configclass
class RoughPhysicsCfg(PresetCfg):
    """Shared backend presets for locomotion velocity environments."""

    isaacsim_physx = PhysxCfg(
        gpu_max_rigid_patch_count=10 * 2**15,
        gpu_max_rigid_contact_count=2**27,
        gpu_collision_stack_size=2**27,
    )
    ovphysx = OvPhysxCfg(gpu_max_rigid_patch_count=10 * 2**15)
    physx = PhysxAutoCfg(isaacsim_physx=isaacsim_physx, ovphysx=ovphysx)
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=1000,
            nconmax=300,
            cone="pyramidal",
            impratio=1.0,
            integrator="implicitfast",
            use_mujoco_contacts=False,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(max_triangle_pairs=2_500_000),
        num_substeps=2,
        debug_mode=False,
        default_shape_cfg=NewtonShapeCfg(margin=0.0, ke=160000.0, kd=1100.0),
    )
    newton_kamino = NewtonCfg(solver_cfg=KaminoPADMMSolverCfg(max_contacts_per_world=64))
    default = newton_mjwarp


##
# Scene definition
##


@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    env_spacing = 4.0
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="hacked_generator",
        terrain_generator=FiledTerrainGeneratorCfg(
            size=(9, 12),
            border_width=0.0,
            border_height=0.0,
            curriculum=True,
            num_rows=23,
            num_cols=1,
            sub_terrains={
                "motion_matched": MotionMatchedTerrainCfg(
                    proportion=1.0,
                ),
            },
        ),
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = MISSING
    # full-robot reference visualization, unmanaged by the scene articulation lifecycle
    robot_reference: AssetBaseCfg | None = None
    # motion reference
    motion_reference: MotionReferenceManagerCfg = MISSING  # type: ignore
    # sensors: the concrete implementation is selected automatically from the active physics backend;
    # backend-specific fields such as ``global_world_only`` are ignored by the backends that do not use them
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/[^/]*", history_length=3, track_air_time=True)
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    def __post_init__(self):
        if type(self.motion_reference) is type(MISSING) or not self.motion_reference.debug_vis:
            delattr(self, "robot_reference")


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    position_ref_command = instinct_mdp.PositionRefCommandCfg(
        realtime_mode=True,
        current_state_command=False,
        anchor_frame="robot",
    )
    position_b_ref_command = instinct_mdp.PositionRefCommandCfg(
        realtime_mode=True,
        current_state_command=False,
        anchor_frame="reference",
    )
    rotation_ref_command = instinct_mdp.RotationRefCommandCfg(
        realtime_mode=True,
        current_state_command=False,
        in_base_frame=True,
        rotation_mode="tannorm",
    )
    joint_pos_ref_command = instinct_mdp.JointPosRefCommandCfg(current_state_command=False)
    joint_vel_ref_command = instinct_mdp.JointVelRefCommandCfg(current_state_command=False)


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos_ref = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "joint_pos_ref_command"},
            history_length=1
        )
        joint_vel_ref = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "joint_vel_ref_command"},
            history_length=1
        )
        position_ref = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "position_ref_command"},
            noise=Unoise(n_min=-0.25, n_max=0.25),
            history_length=1,
        )
        rotation_ref = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "rotation_ref_command"},
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=1,
        )
        # proprioception
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            history_length=PROPRIO_HISTORY_LENGTH,
        )
        # base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            history_length=PROPRIO_HISTORY_LENGTH,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            history_length=PROPRIO_HISTORY_LENGTH,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            noise=Unoise(n_min=-0.5, n_max=0.5),
            history_length=PROPRIO_HISTORY_LENGTH,
        )
        last_action = ObsTerm(
            func=mdp.last_action,
            history_length=PROPRIO_HISTORY_LENGTH,
        )
        link_pos = ObsTerm(
            func=instinct_mdp.link_pos_b,
            params={"asset_cfg": SceneEntityCfg(name="robot", body_names=MISSING, preserve_order=True)},
            history_length=1,
        )
        link_rot = ObsTerm(
            func=instinct_mdp.link_tannorm_b,
            params={"asset_cfg": SceneEntityCfg(name="robot", body_names=MISSING, preserve_order=True)},
            history_length=1,
        )
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            history_length=PROPRIO_HISTORY_LENGTH,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()

    @configclass
    class CriticObsCfg(ObsGroup):
        """Critic observations for BeyondMimic."""

        # BeyondMimic specific reference observations
        joint_pos_ref = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "joint_pos_ref_command"},
            history_length = 1
        )
        joint_vel_ref = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "joint_vel_ref_command"},
            history_length = 1
        )
        position_ref = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "position_ref_command"},
            history_length = 1
        )
        rotation_ref = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "rotation_ref_command"},
            history_length=1,
        )

        # proprioception
        link_pos = ObsTerm(
            func=instinct_mdp.link_pos_b,
            params={"asset_cfg": SceneEntityCfg(name="robot", body_names=MISSING, preserve_order=True)},
            history_length=1,
        )
        link_rot = ObsTerm(
            func=instinct_mdp.link_tannorm_b,
            params={"asset_cfg": SceneEntityCfg(name="robot", body_names=MISSING, preserve_order=True)},
            history_length=1,
        )
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            history_length=PROPRIO_HISTORY_LENGTH,
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            history_length=PROPRIO_HISTORY_LENGTH,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            history_length=PROPRIO_HISTORY_LENGTH,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            history_length=PROPRIO_HISTORY_LENGTH,
        )
        last_action = ObsTerm(
            func=mdp.last_action,
            history_length=PROPRIO_HISTORY_LENGTH,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    critic: CriticObsCfg = CriticObsCfg()

@configclass
class EventsCfg:
    """Configuration for events."""

    match_motion_ref_with_scene = EventTerm(
        func=instinct_mdp.match_motion_ref_with_scene,
        mode="startup",
        params={
            "motion_ref_cfg": SceneEntityCfg("motion_reference"),
        },
    )
    bin_fail_counter_smoothing = EventTerm(
        func=instinct_mdp.beyondmimic_bin_fail_counter_smoothing,
        mode="interval",
        interval_range_s=(0.02, 0.02),  # every environment step
        params={
            "curriculum_name": "beyond_adaptive_sampling",
        },
    )
    reset_robot = EventTerm(
        func=instinct_mdp.reset_robot_state_by_reference,
        mode="reset",
        params={
            "motion_ref_cfg": SceneEntityCfg("motion_reference"),
            "asset_cfg": SceneEntityCfg("robot"),
            # reset with position offset to put the robot_reference in scene.
            "position_offset": [0.0, 0.0, 0.0],
            "dof_vel_ratio": 1.0,
            "base_lin_vel_ratio": 1.0,
            "base_ang_vel_ratio": 1.0,
            # Pose randomization (+-5cm position, +-6degrees rotation)
            "randomize_pose_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
            },
            # Velocity randomization (+-0.1 m/s linear, +-0.1 rad/s angular)
            "randomize_velocity_range": {},
            # Joint position randomization (+-0.1 rad)
            "randomize_joint_pos_range": (-0.0, 0.0),
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    base_position_imitation_gauss = RewTerm(
        func=instinct_mdp.base_position_imitation_gauss,
        weight=0.5,
        params={
            "std": 0.3,
        },
    )
    base_rot_imitation_gauss = RewTerm(
        func=instinct_mdp.base_rot_imitation_gauss,
        weight=0.5,
        params={
            "std": 0.4,
            "difference_type": "axis_angle",
        },
    )
    link_pos_imitation_gauss = RewTerm(
        func=instinct_mdp.link_pos_imitation_gauss,
        weight=1.0,
        params={
            "combine_method": "mean_prod",
            "in_base_frame": False,
            "in_relative_world_frame": True,
            "std": 0.3,
        },
    )
    link_rot_imitation_gauss = RewTerm(
        func=instinct_mdp.link_rot_imitation_gauss,
        weight=1.0,
        params={
            "combine_method": "mean_prod",
            "in_base_frame": False,
            "in_relative_world_frame": True,
            "std": 0.4,
        },
    )
    link_lin_vel_imitation_gauss = RewTerm(
        func=instinct_mdp.link_lin_vel_imitation_gauss,
        weight=1.0,
        params={
            "combine_method": "mean_prod",
            "std": 1.0,
        },
    )
    link_ang_vel_imitation_gauss = RewTerm(
        func=instinct_mdp.link_ang_vel_imitation_gauss,
        weight=1.0,
        params={
            "combine_method": "mean_prod",
            "std": 3.14,
        },
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    illegal_reset_contact = DoneTerm(
        func=instinct_mdp.illegal_reset_contact,
        time_out=True,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$).+$"
                ],
            ),
            "threshold": 500,
            "episode_length_threshold": 2,
        },
    )
    base_pos_too_far = DoneTerm(
        func=instinct_mdp.pos_far_from_ref,
        time_out=False,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "reference_cfg": SceneEntityCfg("motion_reference"),
            "distance_threshold": 0.25,
            "check_at_keyframe_threshold": -1,
            "print_reason": False,
            "height_only": False,
        },
    )
    base_pg_too_far = DoneTerm(
        func=instinct_mdp.projected_gravity_far_from_ref,
        time_out=False,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "reference_cfg": SceneEntityCfg("motion_reference"),
            "projected_gravity_threshold": 0.8,
            "check_at_keyframe_threshold": -1,
            "z_only": False,
            "print_reason": False,
        },
    )
    link_pos_too_far = DoneTerm(
        func=instinct_mdp.link_pos_far_from_ref,
        time_out=False,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "reference_cfg": SceneEntityCfg(
                "motion_reference",
                body_names=[
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
                preserve_order=True,
            ),
            "distance_threshold": 0.25,
            "in_base_frame": False,
            "check_at_keyframe_threshold": -1,
            "height_only": True,
            "print_reason": False,
        },
    )

    dataset_exhausted = DoneTerm(
        func=instinct_mdp.dataset_exhausted,
        time_out=True,
        params={
            "reference_cfg": SceneEntityCfg("motion_reference"),
            "print_reason": False,
        },
    )
    out_of_border = DoneTerm(
        func=instinct_mdp.terrain_out_of_bounds,
        time_out=True,
        params={"asset_cfg": SceneEntityCfg("robot"), "print_reason": False, "distance_buffer": 0.1},
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    beyond_adaptive_sampling = CurrTerm(func=instinct_mdp.BeyondConcatMotionAdaptiveWeighting)   # type: ignore


##
# Monitor configuration
##


@configclass
class MonitorCfg:
    dataset = MonitorTerm(
        func="instinctlab.monitors.monitors:MotionReferenceMonitorTerm",
        params=dict(
            asset_cfg=SceneEntityCfg("motion_reference"),
            sample_stat_interval=500,
            top_n_samples=5,
        ),
    )
    shadowing_position = MonitorTerm(
        func="instinctlab.monitors.monitors:ShadowingPositionMonitorTerm",
        params=dict(
            robot_cfg=SceneEntityCfg("robot"),
            motion_reference_cfg=SceneEntityCfg("motion_reference"),
            in_base_frame=True,
            check_at_keyframe_threshold=0.03,
        ),
    )
    shadowing_rotation = MonitorTerm(
        func="instinctlab.monitors.monitors:ShadowingRotationMonitorTerm",
        params=dict(
            robot_cfg=SceneEntityCfg("robot"),
            motion_reference_cfg=SceneEntityCfg("motion_reference"),
            masking=True,
        ),
    )
    shadowing_joint_pos = MonitorTerm(
        func="instinctlab.monitors.monitors:ShadowingJointPosMonitorTerm",
        params=dict(
            robot_cfg=SceneEntityCfg("robot"),
            motion_reference_cfg=SceneEntityCfg("motion_reference"),
            masking=True,
        ),
    )
    shadowing_joint_vel = MonitorTerm(
        func="instinctlab.monitors.monitors:ShadowingJointVelMonitorTerm",
        params=dict(
            robot_cfg=SceneEntityCfg("robot"),
            motion_reference_cfg=SceneEntityCfg("motion_reference"),
            masking=True,
        ),
    )
    shadowing_link_pos_b = MonitorTerm(
        func="instinctlab.monitors.monitors:ShadowingLinkPosMonitorTerm",
        params=dict(
            robot_cfg=SceneEntityCfg("robot"),
            motion_reference_cfg=SceneEntityCfg("motion_reference"),
            in_base_frame=True,
            masking=True,
        ),
    )
    shadowing_link_pos_w = MonitorTerm(
        func="instinctlab.monitors.monitors:ShadowingLinkPosMonitorTerm",
        params=dict(
            robot_cfg=SceneEntityCfg("robot"),
            motion_reference_cfg=SceneEntityCfg("motion_reference"),
            in_base_frame=False,
            masking=True,
        ),
    )


##
# Environment configuration
##


@configclass
class PerceptiveShadowingEnvCfg(InstinctLabRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Simulation settings
    sim: SimulationCfg = SimulationCfg(physics=RoughPhysicsCfg())
    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    monitors: MonitorCfg = MonitorCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material

    def play_mode(self):
        """Play-mode overrides shared by the velocity-tracking environments."""
        super().play_mode()
