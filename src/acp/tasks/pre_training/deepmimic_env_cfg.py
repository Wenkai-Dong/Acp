# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Base configuration for the velocity-tracking locomotion environments on rough terrain."""

import math
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
from isaaclab.managers import SceneEntityCfg, RecorderManagerBaseCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.physics import PhysxAutoCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
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

from acp.envs import mdp
from acp.envs.mdp import ContactRecorderManagerCfg
from acp.utils.noise import ContactLabelUniformNoiseCfg as ContactLabelUnoise
from acp.utils.noise import GoalRotUniformNoiseCfg as GoalRotUnoise

PROPRIO_HISTORY_LENGTH = 8

##
# Physics backend presets
##


@configclass
class RoughPhysicsCfg(PresetCfg):
    """Shared backend presets for locomotion velocity environments."""

    isaacsim_physx = PhysxCfg(
        gpu_max_rigid_patch_count=10 * 2**16,
        gpu_found_lost_aggregate_pairs_capacity=2**27,
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
            num_rows=25,
            num_cols=16,
            sub_terrains={
                "motion_matched": MotionMatchedTerrainCfg(
                    proportion=1.0,
                ),
            },
        ),
        max_init_terrain_level=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="max",
            restitution_combine_mode="max",
            static_friction=0.3,
            dynamic_friction=0.3,
            restitution=0.0,
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
    tracker_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.8, 0.6]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        global_world_only=True,
        ray_cast_drift_range={"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
    )
    planner_height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/pelvis",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        global_world_only=True,
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/[^/]*", history_length=4, track_air_time=True)
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

    link_contact_pos_ref_command = mdp.LinkContactPosRefCommandCfg(
        sensor_name="planner_height_scanner",
        asset_name="robot",
        current_state_command=False,
        debug_vis=True,
    )
    goal_pose_command = mdp.GoalPose2dCommandCfg(
        asset_name="robot",
        resampling_time_range=(0.5, 0.5),
        debug_vis=True,
        radius=0.3,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.DelayedJointPositionActionCfg(asset_name="robot", scale=0.25, use_default_offset=True)


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class ActorPropHisObsCfg(ObsGroup):
        """Observations for policy group."""

        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.0125, n_max=0.0125))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-1.5, n_max=1.5))
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 5
            self.flatten_history_dim = True

    @configclass
    class ActorMapObsCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("tracker_height_scanner"), "offset": 0.74},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            delay_min_lag=2,
            delay_max_lag=5,
            clip=(-1.5, 1.5),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class ActorContactObsCfg(ObsGroup):
        """Observations for policy group."""

        link_contact_pos_ref_command = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "link_contact_pos_ref_command"},
            noise=ContactLabelUnoise(n_min=-0.1, n_max=0.1),
            history_length = 1,
        )
        goal_pose_commands = ObsTerm(
            func=mdp.generated_commands,
            params={"command_name": "goal_pose_command"},
            noise=GoalRotUnoise(n_min=-0.2, n_max=0.2),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    actor_prop_his: ActorPropHisObsCfg = ActorPropHisObsCfg()
    actor_map: ActorMapObsCfg = ActorMapObsCfg()
    actor_contact: ActorContactObsCfg = ActorContactObsCfg()

    @configclass
    class CriticPropObsCfg(ActorPropHisObsCfg):
        """Observations for policy group."""

        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        def __post_init__(self):
            self.enable_corruption = False
            self.history_length = None

    @configclass
    class CriticMapObsCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("planner_height_scanner"), "offset": 0.74},
            clip=(-1.5, 1.5),
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticRefKinObsCfg(ObsGroup):

        ref_base_lin_vel = ObsTerm(func=instinct_mdp.base_lin_vel_reference_as_state,)
        ref_base_ang_vel = ObsTerm(func=instinct_mdp.base_ang_vel_reference_as_state,)
        ref_joint_pos_rel = ObsTerm(func=instinct_mdp.joint_pos_rel_reference_as_state,)
        ref_joint_vel = ObsTerm(func=instinct_mdp.joint_vel_rel_reference_as_state,)
        ref_contact_link_pos = ObsTerm(
            func=mdp.ref_contact_link_pos,
            history_length=1,
            flatten_history_dim=True,
        )
        ref_contact_flags = ObsTerm(func=mdp.contact_flags_reference_as_state)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    critic_prop: CriticPropObsCfg = CriticPropObsCfg()
    critic_map: CriticMapObsCfg = CriticMapObsCfg()
    critic_ref_kin: CriticRefKinObsCfg = CriticRefKinObsCfg()

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
                "z": (-0.0, 0.05),
            },
            # Joint position randomization (+-0.2 rad)
            "randomize_joint_pos_range": (-0.2, 0.2),
        },
    )
    # Dynamics
    contact_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.2),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    torso_added_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "mass_distribution_params": (-2.0, 4.0),
            "operation": "add",
            "distribution": "uniform",
        },
    )
    Leg_link_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[".*_hip_.*_link", ".*_knee_link", ".*_ankle_.*_link"]),
            "mass_distribution_params": (1 / 1.15, 1.15),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    torso_com_offset = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )
    joint_zero_position_bias = EventTerm(
        func=mdp.joint_zero_position_bias,
        mode="startup",
        params={
            "action_name": "joint_pos",
            "non_ankle_range": (-0.01, 0.01),
            "ankle_range": (-0.03, 0.03),
        },
    )
    joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "armature_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    pd_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- Task
    track_base_pos = RewTerm(
        func=mdp.track_base_pos,
        weight=3.0,
        params={"sigma_p": 0.5, "sigma_r": 1.0,},
    )
    track_base_vel = RewTerm(
        func=mdp.track_base_vel,
        weight=1.0,
        params={"sigma_v": 1.0, "sigma_omega": 3.0,},
    )
    track_joint_pos = RewTerm(func=mdp.track_joint_pos,weight=2.5,params={"sigma_q": 2.0,},)
    track_joint_vel = RewTerm(func=mdp.track_joint_vel, weight=0.5, params={"sigma_dot_q": 10.0,},)
    track_contact_label = RewTerm(
        func=mdp.track_contact_label,
        weight=5.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces")},
    )
    track_contact_pos_base = RewTerm(
        func=mdp.track_contact_pos_base,
        weight=2.0,
        params={"sigma_c": 1.0},
    )
    track_contact_pos_global = RewTerm(
        func=mdp.track_contact_pos_global,
        weight=1.0,
        params={"sigma_c": 1.0},
    )
    # Regularization
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    action_rate_2nd = RewTerm(func=mdp.action_rate_2nd, weight=-0.05)
    joint_torque = RewTerm(func=mdp.joint_torques_l2,weight=-1e-5,)
    torque_limits = RewTerm(func=mdp.applied_torque_limits,weight=-0.001,)
    dof_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0,)
    dof_vel_limits = RewTerm(func=mdp.joint_vel_limits, weight=-1.0, params={"soft_ratio": 0.9,},)
    # Style
    contact_acc = RewTerm(func=mdp.contact_acc, weight=-0.01)


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
    base_pos_too_far = DoneTerm(
        func=instinct_mdp.pos_far_from_ref,
        time_out=False,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "reference_cfg": SceneEntityCfg("motion_reference"),
            "distance_threshold": 0.25,
            "check_at_keyframe_threshold": -1,
            "print_reason": False,
            "height_only": True,
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
            "print_reason": False,
            "z_only": True,
        },
    )
    foot_pos_too_far = DoneTerm(
        func=instinct_mdp.link_pos_far_from_ref,
        time_out=False,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "reference_cfg": SceneEntityCfg(
                "motion_reference",
                body_names=["left_ankle_roll_link","right_ankle_roll_link"],
                preserve_order=True,
            ),
            "distance_threshold": 0.5,
            "in_base_frame": False,
            "check_at_keyframe_threshold": -1,
            "height_only": False,
            "print_reason": False,
        },
    )
    base_xy_too_far = DoneTerm(
        func=mdp.pos_xy_far_from_ref,
        time_out=False,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "reference_cfg": SceneEntityCfg("motion_reference"),
            "distance_threshold": 0.5,
            "check_at_keyframe_threshold": -1,
            "print_reason": False,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    beyond_adaptive_sampling = CurrTerm(func=mdp.StartPhaseWeighting)


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


@configclass
class RecordersCfg(PresetCfg):
    default = RecorderManagerBaseCfg()
    recorder = ContactRecorderManagerCfg()


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
    # Monitor Settings
    monitors: MonitorCfg = MonitorCfg()
    # Recorders Settings
    recorders: RecordersCfg = RecordersCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 13.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material

    def play_mode(self):
        """Play-mode overrides shared by the velocity-tracking environments."""
        super().play_mode()
        self.scene.contact_pelvis = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/pelvis",
            track_contact_points=True,
            max_contact_data_count_per_prim=64,
            history_length=4,
            filter_prim_paths_expr=["/World/ground/terrain/mesh"],
        )
        self.scene.contact_left_knee = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_knee_link",
            track_contact_points=True,
            max_contact_data_count_per_prim=64,
            history_length=4,
            filter_prim_paths_expr=["/World/ground/terrain/mesh"],
        )
        self.scene.contact_right_knee = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_knee_link",
            track_contact_points=True,
            max_contact_data_count_per_prim=64,
            history_length=4,
            filter_prim_paths_expr=["/World/ground/terrain/mesh"],
        )
        self.scene.contact_left_foot = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_ankle_roll_link",
            track_contact_points=True,
            max_contact_data_count_per_prim=64,
            history_length=4,
            filter_prim_paths_expr=["/World/ground/terrain/mesh"],
        )
        self.scene.contact_right_foot = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_ankle_roll_link",
            track_contact_points=True,
            max_contact_data_count_per_prim=64,
            history_length=4,
            filter_prim_paths_expr=["/World/ground/terrain/mesh"],
        )
        self.scene.contact_left_elbow = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_elbow_link",
            track_contact_points=True,
            max_contact_data_count_per_prim=64,
            history_length=4,
            filter_prim_paths_expr=["/World/ground/terrain/mesh"],
        )
        self.scene.contact_right_elbow = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_elbow_link",
            track_contact_points=True,
            max_contact_data_count_per_prim=64,
            history_length=4,
            filter_prim_paths_expr=["/World/ground/terrain/mesh"],
        )
        self.scene.contact_left_hand = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/left_wrist_yaw_link",
            track_contact_points=True,
            max_contact_data_count_per_prim=64,
            history_length=4,
            filter_prim_paths_expr=["/World/ground/terrain/mesh"],
        )
        self.scene.contact_right_hand = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/right_wrist_yaw_link",
            track_contact_points=True,
            max_contact_data_count_per_prim=64,
            history_length=4,
            filter_prim_paths_expr=["/World/ground/terrain/mesh"],
        )