# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

import torch

from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg

if TYPE_CHECKING:
    from .noise_model import NoiseModel, NoiseModelWithAdditiveBias


@configclass
class ContactLabelUniformNoiseCfg(UniformNoiseCfg):
    """Configuration for a additive uniform noise term."""

    func: str = "{DIR}.noise_model:contact_label_uniform_noise"

@configclass
class GoalRotUniformNoiseCfg(UniformNoiseCfg):
    """Configuration for a additive uniform noise term."""

    func: str = "{DIR}.noise_model:goal_rot_uniform_noise"


