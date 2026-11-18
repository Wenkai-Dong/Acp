# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import JointPositionAction
from isaaclab.utils import DelayBuffer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv
    from . import delayed_actions_cfg

logger = logging.getLogger(__name__)


class DelayedJointPositionAction(JointPositionAction):

    cfg: delayed_actions_cfg.DelayedJointPositionActionCfg

    def __init__(self, cfg: delayed_actions_cfg.DelayedJointPositionActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)

        self._delay_buffer = DelayBuffer(
            cfg.max_delay, self.num_envs, device=self.device, min_lag=cfg.min_delay, hold_prob=1.0
        )


    def apply_actions(self):
        # set position targets
        delayed = self._delay_buffer.compute(self.processed_actions)
        self._asset.actuators.target_command.set_position_index(value=delayed, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        self._delay_buffer.reset(env_ids)
