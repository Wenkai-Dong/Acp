from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING, Literal

from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from collections.abc import Callable

from instinctlab.motion_reference.motion_files.terrain_motion_cfg import TerrainMotionCfg


@configclass
class ContactMotionCfg(TerrainMotionCfg):

    class_type: type | str = "{DIR}.contact_motion:ContactMotion"

    init_start_phase: float = 0.9
