# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import torch

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.terrains import TerrainImporter
    from instinctlab.motion_reference.motion_reference_manager import MotionReferenceManagerBase


def sub_terrain_levels_vel(
    env: ManagerBasedRLEnv, env_ids: Sequence[int], asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> dict[str, torch.Tensor]:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")
    # compute the distance the robot walked
    distance = torch.linalg.norm(asset.data.root_pos_w.torch[env_ids, :2] - env.scene.env_origins[env_ids, :2], dim=1)
    # robots that walked far enough progress to harder terrains
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    # robots that walked less than half of their required distance go to simpler terrains
    move_down = distance < torch.linalg.norm(command[env_ids, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up
    # update terrain levels
    terrain.update_env_origins(env_ids, move_up, move_down)
    # return the mean terrain levels of sub-terrain
    gen_cfg = terrain.cfg.terrain_generator
    if not hasattr(env, "_col_to_sub"):
        props = torch.tensor([c.proportion for c in gen_cfg.sub_terrains.values()], dtype=torch.float64)
        cum = torch.cumsum(props / props.sum(), dim=0)
        cols = torch.arange(gen_cfg.num_cols, dtype=torch.float64) / gen_cfg.num_cols + 0.001
        env._col_to_sub = torch.searchsorted(cum, cols, right=True).to(terrain.terrain_types.device)
    sub_type = env._col_to_sub[terrain.terrain_types]

    levels = {}
    for i, name in enumerate(gen_cfg.sub_terrains.keys()):
        mask = sub_type == i
        if mask.any():
            levels[name] = terrain.terrain_levels[mask].float().mean()
    levels["all"] = terrain.terrain_levels.float().mean()
    return levels


class StartPhaseWeighting(ManagerTermBase):

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self.motion_reference: MotionReferenceManagerBase = env.scene[
            cfg.params.get("reference_name", "motion_reference")
        ]

        # NOTE: For simplicity, only support one motion buffer for now.
        assert len(self.motion_reference.motion_buffers.keys()) == 1, "Only support one motion buffer for now."
        self.motion_buffer_name = list(self.motion_reference.motion_buffers.keys())[0]
        self.motion_buffer: AmassMotion = self.motion_reference.motion_buffers[self.motion_buffer_name]  # type: ignore

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | slice,
        phase_step: float = 0.05,
        max_phase: float = 0.99,
    ) -> dict[str, float] | None:
        if isinstance(env_ids, slice):
            env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
        if len(env_ids) == 0:
            raise ValueError("No env_ids")

        env_ids = env_ids[env.episode_length_buf[env_ids] > 0]
        if len(env_ids) == 0:
            return None

        motion_buffer = self.motion_buffer
        assigned_ids = motion_buffer.env_ids_to_assigned_ids(env_ids).to(motion_buffer.buffer_device)
        motion_ids = motion_buffer._assigned_env_motion_selection[assigned_ids]

        episode_failed = self._env.termination_manager.terminated[env_ids].to(motion_buffer.buffer_device)
        sign = torch.where(episode_failed, 1.0, -1.0)

        # ④ 同一条动作这一步可能有好几个环境结束：把 +1/-1 加起来，再数一共几个
        num_motions = motion_buffer.num_trajectories
        sign_sum = torch.zeros(num_motions, device=motion_buffer.buffer_device).index_add_(0, motion_ids, sign)
        count = torch.zeros(num_motions, device=motion_buffer.buffer_device).index_add_(0, motion_ids, torch.ones_like(sign))
        has = count > 0   # 这一步有环境结束的那些动作

        # ⑤ 更新相位：取平均再 × 0.05，所以一步最多移动 0.05；最后限制在 [0, 0.99]
        motion_buffer.start_phase[has] += phase_step * sign_sum[has] / count[has]
        motion_buffer.start_phase.clamp_(0.0, max_phase)

        # ⑥ 返回的数会记到日志里
        return {
            "start_phase_mean": motion_buffer.start_phase.mean().item(),
            "start_phase_max": motion_buffer.start_phase.max().item(),
            "ratio_at_phase_0": (motion_buffer.start_phase <= 0.0).float().mean().item(),
        }