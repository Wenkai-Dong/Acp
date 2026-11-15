# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable reward functions.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to include
the reward introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul, wrap_to_pi
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.sensors import ContactSensor, RayCaster
    from isaaclab.envs import ManagerBasedRLEnv
    from instinctlab.motion_reference.motion_reference_manager import MotionReferenceManagerBase

"""
General.
"""


def straight_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    selected_bodies_quat = asset.data.body_quat_w.torch[:, asset_cfg.body_ids, :]
    # (num_envs, num_bodies, 3) via broadcasting, no memory copy
    gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=env.device)
    gravity_vec = gravity_vec.expand(selected_bodies_quat.shape[0], selected_bodies_quat.shape[1], 3)
    projected_gravity = quat_apply_inverse(selected_bodies_quat, gravity_vec)
    return torch.sum(torch.square(projected_gravity[..., :2]), dim=(1, 2))


def contact_acc(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]

    link_indices = asset.find_bodies(motion_reference.cfg.link_of_contacts, preserve_order=True)[0]
    return torch.sum(torch.linalg.norm(asset.data.body_lin_acc_w.torch[:, link_indices, :], dim=-1), dim=1).clamp(max=1e4)


"""
Root penalties.
"""


def track_base_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    sigma_p: float = 0.5, sigma_r: float = 1.0,
):
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]

    base_pos_ref_w = motion_reference.reference_frame.base_pos_w[:, 0]  # (batch_size, 3)
    base_pos_w = asset.data.root_pos_w.torch  # (batch_size, 3)
    base_quat_ref_w = motion_reference.reference_frame.base_quat_w[:, 0]  # (batch_size, 4)
    base_quat_w = asset.data.root_quat_w.torch  # (batch_size, 4)

    pos_diff = torch.sum(torch.square(base_pos_w - base_pos_ref_w), dim=-1) / sigma_p ** 2  # (batch_size,)
    chi_mov = (torch.linalg.norm(motion_reference.reference_frame.base_lin_vel_w[:, 0], dim=-1) > 0.1).float()
    pos_diff = pos_diff * chi_mov

    quat_diff = math_utils.quat_mul(base_quat_ref_w, math_utils.quat_conjugate(base_quat_w))  # (batch_size, 4)
    axisang = math_utils.axis_angle_from_quat(quat_diff)  # (batch_size, 3)
    ang_diff = torch.norm(axisang, dim=-1) ** 2 / sigma_r ** 2  # (batch_size,)

    return torch.exp(-pos_diff - ang_diff)


def track_base_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    sigma_v: float = 1.0, sigma_omega: float = 3.0,
):
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    base_lin_vel_w = asset.data.root_link_lin_vel_w.torch  # (batch_size, 3)
    base_lin_vel_ref_w = motion_reference.reference_frame.base_lin_vel_w[:, 0]  # (batch_size, 3)
    base_ang_vel_w = asset.data.root_link_ang_vel_w.torch  # (batch_size, 3)
    base_ang_vel_ref_w = motion_reference.reference_frame.base_ang_vel_w[:, 0]  # (batch_size, 3)

    vel_diff = torch.sum(torch.square(base_lin_vel_w - base_lin_vel_ref_w), dim=-1) / sigma_v ** 2  # (batch_size,)
    ang_diff = torch.sum(torch.square(base_ang_vel_w - base_ang_vel_ref_w), dim=-1) / sigma_omega ** 2  # (batch_size,)

    return torch.exp(-vel_diff - ang_diff)


"""
Joint penalties.
"""


def joint_torque_limits(
    env: ManagerBasedRLEnv, soft_ratio: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = (
        torch.abs(asset.data.applied_torque.torch[:, asset_cfg.joint_ids])
        - asset.data.joint_effort_limits.torch[:, asset_cfg.joint_ids] * soft_ratio
    )
    # clip to min=0.0
    out_of_limits = out_of_limits.clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def joint_deviation_l2_threshold(
    env: ManagerBasedRLEnv, threshold: float = 0.25, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one beyond a threshold."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    squared_error = torch.sum(torch.square(angle), dim=1)
    return (squared_error - threshold).clip(min=0.0)


def track_joint_pos(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    sigma_q: float = 2.0,
):
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_ids = asset_cfg.joint_ids

    joint_pos_ref = motion_reference.reference_frame.joint_pos[:, 0, joint_pos_ids]  # (batch_size, num_joints_selected)
    joint_pos = asset.data.joint_pos.torch[:, joint_pos_ids]  # (batch_size, num_joints_selected)
    joint_pos_diff = math_utils.wrap_to_pi(joint_pos - joint_pos_ref)  # (batch_size, num_joints_selected)

    return torch.exp(-torch.sum(torch.square(joint_pos_diff), dim=-1) / sigma_q ** 2)  # (batch_size,)


def track_joint_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    sigma_dot_q: float = 10.0,
):
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel_ids = asset_cfg.joint_ids

    joint_vel_ref = motion_reference.reference_frame.joint_vel[:, 0, joint_vel_ids]  # (batch_size, num_joints_selected)
    joint_vel = asset.data.joint_vel.torch[:, joint_vel_ids]  # (batch_size, num_joints_selected)

    return torch.exp(-torch.sum(torch.square(joint_vel - joint_vel_ref), dim=-1) / sigma_dot_q ** 2)  # (batch_size,)


def joint_pos_limits(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a sum of the absolute value of the difference between the joint position and the soft limits.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    out_of_limits = -(
        asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
        - asset.data.soft_joint_pos_limits.torch[:, asset_cfg.joint_ids, 0]
    ).clip(max=0.0)
    out_of_limits += (
        asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
        - asset.data.soft_joint_pos_limits.torch[:, asset_cfg.joint_ids, 1]
    ).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1).clamp(max=30.0)


"""
Action penalties.
"""


class action_rate_2nd(ManagerTermBase):

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.prev_prev_action = torch.zeros(
            env.num_envs, env.action_manager.total_action_dim, device=env.device
        )

    def reset(self, env_ids: torch.Tensor) -> None:
        self.prev_prev_action[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        a_t = env.action_manager.action
        a_t_1 = env.action_manager.prev_action
        a_t_2 = self.prev_prev_action

        reward = torch.sum(torch.square(a_t - 2 * a_t_1 + a_t_2), dim=1)  # (num_envs,)

        self.prev_prev_action[:] = a_t_1
        return reward


"""
Contact sensor.
"""


def track_contact_label(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
):
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    link_indices = contact_sensor.find_sensors(motion_reference.cfg.link_of_contacts, preserve_order=True)[0]
    contact_label = motion_reference.reference_frame.link_contact[:, 0]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_normal_forces_w_history.torch[:, :, link_indices]
    is_contact = (torch.max(net_contact_forces[..., 2], dim=1)[0] > 1.0)

    return torch.sum(is_contact * (2 * contact_label -1), dim=-1) / 9.0  # (batch_size,)


def track_contact_pos_base(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    sigma_c: float = 1.0,
):
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    link_indices = asset.find_bodies(motion_reference.cfg.link_of_contacts, preserve_order=True)[0]
    links_pos_w = asset.data.body_link_pos_w.torch[:, link_indices]  # (batch_size, num_links, 3)
    ref_link_indices = motion_reference.find_bodies(motion_reference.cfg.link_of_contacts, preserve_order=True)[0]
    ref_links_pos_w = motion_reference.reference_frame.link_pos_w[:, 0, ref_link_indices]

    num_links = links_pos_w.shape[1]

    base_pos_w = asset.data.root_pos_w.torch
    base_yaw_quat = math_utils.yaw_quat(asset.data.root_quat_w.torch)
    links_pos = math_utils.quat_apply_inverse(
        base_yaw_quat[:, None, :].expand(-1, num_links, -1),
        links_pos_w - base_pos_w[:, None, :],
    )
    ref_base_pos_w = motion_reference.reference_frame.base_pos_w[:, 0]
    ref_base_yaw_quat = math_utils.yaw_quat(motion_reference.reference_frame.base_quat_w[:, 0])
    ref_link_pos = math_utils.quat_apply_inverse(
        ref_base_yaw_quat[:, None, :].expand(-1, num_links, -1),
        ref_links_pos_w - ref_base_pos_w[:, None, :],
    )

    link_pos_square = torch.sum(torch.sum(torch.square(links_pos - ref_link_pos), dim=-1), dim=-1) / sigma_c ** 2
    return torch.exp(-link_pos_square)


def track_contact_pos_global(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    sigma_c: float = 1.0,
):
    motion_reference: MotionReferenceManagerBase = env.scene[reference_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    link_indices = asset.find_bodies(motion_reference.cfg.link_of_contacts, preserve_order=True)[0]
    links_pos_w = asset.data.body_link_pos_w.torch[:, link_indices]  # (batch_size, num_links, 3)
    ref_link_indices = motion_reference.find_bodies(motion_reference.cfg.link_of_contacts, preserve_order=True)[0]
    ref_links_pos_w = motion_reference.reference_frame.link_pos_w[:, 0, ref_link_indices]

    chi_mov = (torch.linalg.norm(motion_reference.reference_frame.base_lin_vel_w[:, 0], dim=-1) > 0.1).float()

    link_pos_square = torch.sum(torch.sum(torch.square(links_pos_w - ref_links_pos_w), dim=-1), dim=-1) / sigma_c ** 2
    return torch.exp(-chi_mov * link_pos_square)


"""
Velocity-tracking rewards.
"""


def stand_still_velocity(
    env: ManagerBasedRLEnv, command_name: str = "base_velocity", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), threshold: float = 0.05
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    reward = torch.sum(torch.abs(asset.data.joint_vel.torch), dim=1)
    cmd_norm = torch.norm(env.command_manager.get_command(command_name), dim=1)
    return reward * (cmd_norm < threshold)


"""
Pose-tracking rewards.
"""



