# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

import instinctlab.motion_reference.utils as motion_reference_utils

if TYPE_CHECKING:
    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.managers.command_manager import CommandTerm
    from isaaclab.sensors import ContactSensor
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.envs.mdp.commands.pose_command import UniformPoseCommand

    from instinctlab.motion_reference.motion_reference_manager import MotionReferenceManager

"""
MDP terminations.
"""





"""
Root terminations.
"""


def root_height_below_minimum_terrain(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene[sensor_cfg.name]
    terrain_height = torch.amax(sensor.data.ray_hits_w.torch[..., 2], dim=1)
    terrain_height = torch.nan_to_num(terrain_height, nan=-1e6, posinf=-1e6, neginf=-1e6)
    relative_height = asset.data.root_pos_w.torch[:, 2] - terrain_height
    return relative_height < minimum_height


def subterrain_out_of_bounds(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_buffer: float = 3.0
) -> torch.Tensor:
    if env.scene.cfg.terrain.terrain_type == "plane":
        # we have infinite terrain because it is a plane
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    elif env.scene.cfg.terrain.terrain_type == "generator":
        # obtain the size of the sub-terrains
        terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
        grid_width, grid_length = terrain_gen_cfg.size

        # extract the used quantities (to enable type-hinting)
        asset: RigidObject = env.scene[asset_cfg.name]
        subterrain_origins = env.scene.env_origins[:, :2]
        pos_rel = asset.data.root_pos_w.torch[:, :2] - subterrain_origins

        # check if the agent is out of bounds
        x_out_of_bounds = torch.abs(pos_rel[:, 0]) > 0.5 * grid_width - distance_buffer
        y_out_of_bounds = torch.abs(pos_rel[:, 1]) > 0.5 * grid_length - distance_buffer
        return torch.logical_or(x_out_of_bounds, y_out_of_bounds)
    else:
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")


def pos_xy_far_from_ref(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    reference_cfg: SceneEntityCfg = SceneEntityCfg("motion_reference"),
    distance_threshold: float = 0.5,
    check_at_keyframe_threshold: float = 0.01,
    print_reason: bool = False,
) -> torch.Tensor:
    distance = motion_reference_utils.get_base_position_distance(
        env,
        asset_cfg,
        reference_cfg,
        masking=True,
        squared=False,
        return_diff=True,
    )
    distance = torch.norm(distance[..., :2], dim=-1)
    distance = motion_reference_utils.matching_reference_timing(
        env,
        distance,
        env.scene[reference_cfg.name],
        check_at_keyframe_threshold=check_at_keyframe_threshold,
        multiply_by_frame_interval=False,
    )

    return_ = distance > distance_threshold
    if print_reason and return_.any():
        motion_reference: MotionReferenceManager = env.scene[reference_cfg.name]
        env_ids = torch.where(return_)[0]
        motion_identifiers = motion_reference.get_current_motion_identifiers(env_ids=env_ids)
        print(f"pos_far_from_ref: {return_.sum()} at {motion_identifiers}")
    return return_


"""
Joint terminations.
"""





"""
Contact sensor.
"""



