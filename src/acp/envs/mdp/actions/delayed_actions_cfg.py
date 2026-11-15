# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.utils import configclass

from isaaclab.envs.mdp import JointPositionActionCfg

if TYPE_CHECKING:
    from .delayed_joint_actions import DelayedJointPositionAction

##
# Joint actions.
##


@configclass
class DelayedJointPositionActionCfg(JointPositionActionCfg):

    class_type: type[DelayedJointPositionAction] | str = "{DIR}.delayed_joint_actions:DelayedJointPositionAction"

    min_delay: int = 0

    max_delay: int = 4

