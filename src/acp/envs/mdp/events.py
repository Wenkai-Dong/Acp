# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to enable different events.

Events include anything related to altering the simulation state. This includes changing the physics
materials, applying external forces, and resetting the state of the asset.

The functions can be passed to the :class:`isaaclab.managers.EventTermCfg` object to enable
the event introduced by the function.
"""

from __future__ import annotations

import logging
import math
from types import ModuleType
from typing import TYPE_CHECKING, Literal

import torch

from isaaclab import sim as sim_utils
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab_physx.assets import DeformableObject

    from omni.replicator.core.scripts.utils.rng import ReplicatorRNG
    from pxr import Usd

    from isaaclab.assets import Articulation, RigidObject
    from isaaclab.terrains import TerrainImporter
    from isaaclab.envs import ManagerBasedEnv

# import logger
logger = logging.getLogger(__name__)


class randomize_virtual_floor(ManagerTermBase):
    def __init__(self, cfg: EventTermCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        vfloor_range = cfg.params.get("vfloor_range", (-0.4, -1.0))
        env.vfloor_z = math_utils.sample_uniform(*vfloor_range, (env.scene.num_envs,), device=env.device)

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        vfloor_range: tuple[float, float] = (-0.4, -1.0),
    ):
        # resolve environment ids
        if env_ids is None:
            env_ids = slice(None)
        if isinstance(env_ids, slice):
            env_ids = torch.arange(env.scene.num_envs, device=env.device)[env_ids]
        else:
            env_ids = env_ids.to(env.device)

        env.vfloor_z[env_ids] = math_utils.sample_uniform(*vfloor_range, (len(env_ids),), device=env.device)


def joint_zero_position_bias(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | slice,
    action_name,
    non_ankle_range,
    ankle_range,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    asset: Articulation = env.scene[asset_cfg.name]

    # cast env_ids to allow broadcasting
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)

    bias = torch.empty(len(env_ids), asset.num_joints, device=env.device).uniform_(*non_ankle_range)
    ankle_ids, _ = asset.find_joints(".*_ankle_.*")
    bias[:, ankle_ids] = torch.empty(len(env_ids), len(ankle_ids), device=env.device).uniform_(*ankle_range)

    # get default joint state
    default_pos = asset.data.default_joint_pos.torch
    default_pos[env_ids] += bias

    term = env.action_manager.get_term(action_name)
    term._offset[env_ids] = default_pos[env_ids][:, term._joint_ids]