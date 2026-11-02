# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Unitree G1 velocity-tracking environment on rough terrain."""

import os

from instinctlab.assets.unitree_g1 import (
    G1_29DOF_POLICY_JOINT_ORDER_V1,
    G1_REFERENCE_CFG,
    beyondmimic_action_scale,
    beyondmimic_g1_29dof_actuators,
)
from instinctlab.monitors import MonitorTermCfg as MonitorTerm
from instinctlab.motion_reference import MotionReferenceManagerCfg
from instinctlab.motion_reference.motion_files.amass_motion_cfg import AmassMotionCfg as AmassMotionCfgBase
from instinctlab.motion_reference.motion_files.terrain_motion_cfg import TerrainMotionCfg as TerrainMotionCfgBase
from instinctlab.motion_reference.utils import motion_interpolate_bilinear
from instinctlab.tasks.utils.backends import backend_preset
from instinctlab.utils.config import set_cfg_joint_order

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from acp.assets import ACP_ASSETS_DIR
from acp.assets.robots.unitree import UNITREE_G1_29DOF_MIMIC_CFG

from ...deepmimic_env_cfg import (
    PerceptiveShadowingEnvCfg,
)

G1_CFG = UNITREE_G1_29DOF_MIMIC_CFG

MOTION_FOLDER = f"{ACP_ASSETS_DIR}/motion_data/box_climbing/climb_20"


@configclass
class TerrainMotionCfg(TerrainMotionCfgBase):
    path = os.path.expanduser(MOTION_FOLDER)
    metadata_yaml = os.path.expanduser(f"{MOTION_FOLDER}/metadata.yaml")
    max_origins_per_motion = 49

    ensure_link_below_zero_ground = False
    motion_start_from_middle_range = [0.0, 0.0]
    motion_start_height_offset = 0.0
    motion_bin_length_s = 1.0
    buffer_device = "output_device"
    motion_interpolate_func = motion_interpolate_bilinear
    velocity_estimation_method = "frontbackward"
    env_starting_stub_sampling_strategy = "concat_motion_bins"


@configclass
class AMASSMotionCfg(AmassMotionCfgBase):
    path = os.path.expanduser(MOTION_FOLDER)
    filtered_motion_selection_filepath = None
    ensure_link_below_zero_ground = False
    motion_start_from_middle_range = [0.0, 0.0]
    motion_start_height_offset = 0.0
    motion_bin_length_s = 1.0
    buffer_device = "output_device"
    motion_interpolate_func = motion_interpolate_bilinear
    velocity_estimation_method = "frontbackward"
    env_starting_stub_sampling_strategy = "concat_motion_bins"


motion_reference_cfg = MotionReferenceManagerCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    robot_model_path=G1_CFG.spawn.asset_path,
    reference_prim_path="/World/envs/env_[^/]+/RobotReference",
    link_of_interests=[
        "pelvis",
        "torso_link",
        "left_shoulder_roll_link",
        "right_shoulder_roll_link",
        "left_elbow_link",
        "right_elbow_link",
        "left_wrist_yaw_link",
        "right_wrist_yaw_link",
        "left_hip_roll_link",
        "right_hip_roll_link",
        "left_knee_link",
        "right_knee_link",
        "left_ankle_roll_link",
        "right_ankle_roll_link",
    ],
    symmetric_augmentation_link_mapping=None,
    symmetric_augmentation_joint_mapping=None,
    symmetric_augmentation_joint_reverse_buf=None,
    symmetric_augmentation_joint_names=list(G1_29DOF_POLICY_JOINT_ORDER_V1),
    frame_interval_s=0.1,
    update_period=0.02,
    num_frames=10,
    data_start_from="current_time",
    # set the robot_reference directly at where they are in the scene
    # DO NOT FORGET to change this when in actual training
    visualizing_robot_offset=(2.0, 0.0, 0.0),
    visualizing_robot_from="reference_frame",
    visualizing_marker_types=["relative_links", "links"],
    motion_buffers={
        "TerrainMotion": TerrainMotionCfg(),
    },
    mp_split_method="None",
)


@configclass
class G1PerceptiveShadowingEnvCfg(PerceptiveShadowingEnvCfg):
    """Configuration for the Unitree G1 velocity-tracking environment on rough terrain."""

    def __post_init__(self):
        super().__post_init__()

        # scene
        self.scene.robot = G1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.motion_reference = motion_reference_cfg
        self.events.reset_robot.params["position_offset"] = backend_preset([0.0, 0.0, 0.0], newton=[0.0, 0.0, 0.05])
        joint_order = list(G1_29DOF_POLICY_JOINT_ORDER_V1)
        set_cfg_joint_order(self.actions.joint_pos, joint_order)
        self.observations.policy.joint_pos.params["asset_cfg"] = set_cfg_joint_order(
            SceneEntityCfg("robot"), joint_order
        )
        self.observations.policy.joint_vel.params["asset_cfg"] = set_cfg_joint_order(
            SceneEntityCfg("robot"), joint_order
        )
        self.observations.critic.joint_pos.params["asset_cfg"] = set_cfg_joint_order(
            SceneEntityCfg("robot"), joint_order
        )
        self.observations.critic.joint_vel.params["asset_cfg"] = set_cfg_joint_order(
            SceneEntityCfg("robot"), joint_order
        )
        set_cfg_joint_order(self.commands.joint_pos_ref_command.asset_cfg, joint_order)
        set_cfg_joint_order(self.commands.joint_vel_ref_command.asset_cfg, joint_order)

        self.scene.robot.actuators = beyondmimic_g1_29dof_actuators
        # self.scene.robot.spawn.rigid_props.max_depenetration_velocity = 0.3
        self.actions.joint_pos.scale = beyondmimic_action_scale

        MOTION_NAME = list(self.scene.motion_reference.motion_buffers.keys())[0]
        self.scene.motion_reference.motion_buffers[MOTION_NAME].metadata_yaml = os.path.join(
            self.scene.motion_reference.motion_buffers[MOTION_NAME].path, "metadata.yaml"
        )
        PLANE_TERRAIN = False
        if PLANE_TERRAIN:
            self.scene.motion_reference.motion_buffers.pop(MOTION_NAME)
            self.scene.motion_reference.motion_buffers["AMASSMotion"] = AMASSMotionCfg()
            self.scene.terrain.terrain_type = "plane"
            self.scene.terrain.terrain_generator = None
        else:
            self.scene.terrain.terrain_generator.sub_terrains["motion_matched"].path = (
                self.scene.motion_reference.motion_buffers[MOTION_NAME].path
            )
            self.scene.terrain.terrain_generator.sub_terrains["motion_matched"].metadata_yaml = os.path.join(
                self.scene.motion_reference.motion_buffers[MOTION_NAME].path, "metadata.yaml"
            )

        # match key links for observation terms
        self.observations.critic.link_pos.params["asset_cfg"].body_names = self.scene.motion_reference.link_of_interests
        self.observations.critic.link_rot.params["asset_cfg"].body_names = self.scene.motion_reference.link_of_interests
        self.observations.policy.link_pos.params["asset_cfg"].body_names = self.scene.motion_reference.link_of_interests
        self.observations.policy.link_rot.params["asset_cfg"].body_names = self.scene.motion_reference.link_of_interests

        self.run_name = "g1Perceptive" + "".join(
            [
                (
                    "_concatMotionBins"
                    if self.scene.motion_reference.motion_buffers[MOTION_NAME].env_starting_stub_sampling_strategy
                    == "concat_motion_bins"
                    else "_independentMotionBins"
                ),
            ]
        )

    def play_mode(self):
        """Play-mode overrides shared by the velocity-tracking environments."""
        super().play_mode()
        # self.scene.motion_reference.debug_vis = True
        # self.scene.robot_reference = G1_REFERENCE_CFG.replace(prim_path="{ENV_REGEX_NS}/RobotReference")

        # deactivate adaptive sampling and start from the 0.0s of the motion
        self.curriculum.beyond_adaptive_sampling = None
        self.events.bin_fail_counter_smoothing = None
        MOTION_NAME = list(self.scene.motion_reference.motion_buffers.keys())[0]
        self.scene.motion_reference.motion_buffers[MOTION_NAME].motion_start_from_middle_range = [0.0, 0.0]
        self.scene.motion_reference.motion_buffers[MOTION_NAME].motion_bin_length_s = None
        self.scene.motion_reference.motion_buffers[MOTION_NAME].env_starting_stub_sampling_strategy = "independent"
        self.scene.motion_reference.motion_buffers[MOTION_NAME].metadata_yaml = os.path.join(
            self.scene.motion_reference.motion_buffers[MOTION_NAME].path, "metadata.yaml"
        )
        if self.scene.terrain.terrain_type == "hacked_generator":
            self.scene.terrain.terrain_generator.sub_terrains["motion_matched"].path = (
                self.scene.motion_reference.motion_buffers[MOTION_NAME].path
            )
            self.scene.terrain.terrain_generator.sub_terrains["motion_matched"].metadata_yaml = os.path.join(
                self.scene.motion_reference.motion_buffers[MOTION_NAME].path, "metadata.yaml"
            )

        # Use non-terrain-matching motion and plane to hack the scene.
        self.scene.terrain.terrain_generator.num_rows = 1
        self.scene.terrain.terrain_generator.num_cols = 1
        # remove some terimation terms
        self.terminations.base_pos_too_far = None
        self.terminations.base_pg_too_far = None
        self.terminations.link_pos_too_far = None

        # remove some randomizations
        self.events.add_joint_default_pos = None
        self.events.base_com = None
        self.events.physics_material = None
        self.events.push_robot = None
        self.events.reset_robot.params["randomize_pose_range"]["x"] = [0.0] * 2  # (+-0.6)
        self.events.reset_robot.params["randomize_pose_range"]["y"] = [0.0] * 2  # (+-0.6)
        self.events.reset_robot.params["randomize_pose_range"]["z"] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_pose_range"]["roll"] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_pose_range"]["pitch"] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_pose_range"]["yaw"] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_velocity_range"]["x"] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_velocity_range"]["y"] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_velocity_range"]["z"] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_velocity_range"]["roll"] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_velocity_range"]["pitch"] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_velocity_range"]["yaw"] = (0.0, 0.0)
        self.events.reset_robot.params["randomize_joint_pos_range"] = (0.0, 0.0)

        # add some additional monitor terms
        self.monitors.shadowing_position_stats = MonitorTerm(
            func="instinctlab.monitors.monitors:ShadowingBasePosMonitorTerm",
            params=dict(
                robot_cfg=SceneEntityCfg("robot"),
                motion_reference_cfg=SceneEntityCfg("motion_reference"),
            ),
        )
        self.monitors.right_ankle_pitch_actuator = MonitorTerm(
            func="instinctlab.monitors.monitors:ActuatorMonitorTerm",
            params=dict(
                asset_cfg=SceneEntityCfg("robot", joint_names="right_ankle_pitch.*"),
            ),
        )
        self.monitors.left_ankle_pitch_actuator = MonitorTerm(
            func="instinctlab.monitors.monitors:ActuatorMonitorTerm",
            params=dict(
                asset_cfg=SceneEntityCfg("robot", joint_names="left_ankle_pitch.*"),
            ),
        )
        self.monitors.right_knee_actuator = MonitorTerm(
            func="instinctlab.monitors.monitors:ActuatorMonitorTerm",
            params=dict(
                asset_cfg=SceneEntityCfg("robot", joint_names="right_knee.*"),
            ),
        )
        self.monitors.left_knee_actuator = MonitorTerm(
            func="instinctlab.monitors.monitors:ActuatorMonitorTerm",
            params=dict(
                asset_cfg=SceneEntityCfg("robot", joint_names="left_knee.*"),
            ),
        )