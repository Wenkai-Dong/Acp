# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.managers import CommandTermCfg, SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, FRAME_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils import configclass, replace
import isaaclab.sim as sim_utils

from instinctlab.envs.mdp import ShadowingCommandBaseCfg

if TYPE_CHECKING:
    from .shadowing_command import LinkContactPosRefCommand, GoalPose2dCommand


@configclass
class LinkContactPosRefCommandCfg(ShadowingCommandBaseCfg):

    class_type: type["LinkContactPosRefCommand"] | str = "{DIR}.shadowing_command:LinkContactPosRefCommand"

    sensor_name: str = MISSING

    visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/LinkPosRefCommand",
        markers={
            "point": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0)),
            ),
        },
    )


@configclass
class GoalPose2dCommandCfg(CommandTermCfg):

    class_type: type["GoalPose2dCommand"] | str = "{DIR}.shadowing_command:GoalPose2dCommand"

    motion_reference: SceneEntityCfg = SceneEntityCfg("motion_reference")

    sensor_name: str = MISSING

    radius: float = MISSING

    position_success_threshold: float | None = None

    goal_pose_visualizer_cfg: VisualizationMarkersCfg = replace(
        GREEN_ARROW_X_MARKER_CFG, prim_path="/Visuals/Command/pose_goal"
    )

    # Set the scale of the visualization markers to (0.2, 0.2, 0.8)
    goal_pose_visualizer_cfg.markers["arrow"].scale = (0.2, 0.2, 0.8)