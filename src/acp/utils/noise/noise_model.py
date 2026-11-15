# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import wrap_to_pi
from isaaclab.utils.noise import uniform_noise

if TYPE_CHECKING:
    from . import noise_cfg

##
# Noise as functions.
##


def contact_label_uniform_noise(data: torch.Tensor, cfg: noise_cfg.UniformNoiseCfg) -> torch.Tensor:
    noisy = uniform_noise(data, cfg)
    has_contact = (data != 0).any(dim=-1, keepdim=True)
    return torch.where(has_contact, noisy, data)

def goal_rot_uniform_noise(data: torch.Tensor, cfg: noise_cfg.UniformNoiseCfg) -> torch.Tensor:
    # data 的形状是 [环境数, 3]，三列依次是 [方向x, 方向y, yaw]
    delta = torch.rand(data.shape[0], device=data.device) * (cfg.n_max - cfg.n_min) + cfg.n_min
    cos_d, sin_d = torch.cos(delta), torch.sin(delta)
    x, y, yaw = data[:, 0], data[:, 1], data[:, 2]
    # 第 2 步：方向向量旋转 delta（长度不变）
    new_x = cos_d * x - sin_d * y
    new_y = sin_d * x + cos_d * y
    # 第 3 步：航向也偏 delta，再包回 [-pi, pi)
    new_yaw = wrap_to_pi(yaw + delta)
    # 返回新张量，不改原来的 data（Isaac Lab 要求噪声函数不能改输入）
    return torch.stack([new_x, new_y, new_yaw], dim=-1)