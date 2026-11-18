# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

import functools
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from typing_extensions import deprecated

from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.manager_base import ManagerTermBase
from isaaclab.managers.manager_term_cfg import ObservationTermCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils.images import (
    CameraFrameStack,
    is_depth_like,
    is_normals_like,
    is_rgb_like,
    normalize_camera_image,
    normalize_depth,
    normalize_normals,
    normalize_rgb,
    normalize_segmentation,
)

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.sensors import Camera, Imu, JointWrenchSensor, Pva, RayCaster, RayCasterCamera
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
    from instinctlab.motion_reference.motion_reference_manager import MotionReferenceManagerBase

from isaaclab.envs.utils.io_descriptors import (
    generic_io_descriptor,
    record_body_names,
    record_dtype,
    record_joint_names,
    record_joint_pos_offsets,
    record_joint_vel_offsets,
    record_shape,
)

"""
Root state.
"""





"""
Body state
"""


def ref_contact_link_pos(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    motion_ref_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    motion_reference: MotionReferenceManagerBase = env.scene[motion_ref_cfg.name]

    ref_link_pos_w = motion_reference.reference_frame.link_pos_w[:, 0]
    root_pos_w = robot.data.root_pos_w.torch    # (N, 3)
    yaw_q = math_utils.yaw_quat(robot.data.root_quat_w.torch)   # (N, 4)
    rel_w = ref_link_pos_w - root_pos_w[:, None, :] # (N, 9, 3)
    rel_b = math_utils.quat_apply_inverse(yaw_q[:, None, :].expand(-1, rel_w.shape[1], -1), rel_w)
    return rel_b.reshape(rel_b.shape[0], -1)    # (N, 27)


"""
Joint state.
"""





"""
Sensors.
"""


def elevation_map_vfloor(
        env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, size: tuple[int, int], offset: float = 0.825,
        z_noise: float = 0.0,
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    vfloor_z = env.vfloor_z.unsqueeze(1)  # (N, 1)

    pos_z = sensor.data.pos_w.torch[:, 2:3]  # (N, 1)
    ray_hits_z = sensor.data.ray_hits_w.torch[..., 2]   # (N, B)
    miss = ~torch.isfinite(ray_hits_z)  # (N, B)
    ray_hits_z = torch.where(miss, vfloor_z, ray_hits_z)

    # Z-axis height: height = sensor_height - hit_point_z - offset + noise
    height_scan = pos_z - ray_hits_z - offset
    height_scan += (torch.rand_like(height_scan) - 0.5) * 2 * z_noise  # (N, B)
    height_scan = height_scan.reshape(height_scan.shape[0], size[0], size[1]).unsqueeze(-1)
    elevation_map = sensor.ray_starts.torch.reshape(height_scan.shape[0], size[0], size[1], -1)[..., :2]

    return torch.cat([elevation_map, height_scan], dim=-1).permute(0, 3, 1, 2).contiguous()


def contact_flags_reference_as_state(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
):
    motion_reference: MotionReferenceManagerBase = env.scene[asset_cfg.name]
    return motion_reference.reference_frame.link_contact[:, 0]


"""
Actions.
"""





"""
Commands.
"""





"""
Time.
"""



