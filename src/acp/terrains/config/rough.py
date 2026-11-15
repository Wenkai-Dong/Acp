# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for custom terrains."""

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

import acp.terrains as ame_terrain_gen

ACP_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=True,
    size=(9.0, 12.0),
    border_width=0.0,
    border_height=0.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=False,
    cache_dir="/tmp/IsaacLab/terrains",
    sub_terrains={
        "stepping_stones": ame_terrain_gen.MeshSteppingStonesTerrainCfg(
            proportion=1.0,
            stone_height_max=(0, 0.1),
            stone_width_range=(0.15, 0.45),
            stone_distance_range=(0.05, 0.28),
            border_width=1.0,
            max_yx_angle=False,
        ),
        "stepping_stones_narrow": ame_terrain_gen.MeshPalletsNarrowTerrainCfg(
            proportion=0.0,
            num_bars=1,
            bar_width_range=(0.16, 0.8),
            stone_length_range=(0.14, 0.8),
            stone_distance_range=(0.05, 0.28),
            stone_height_range=(0.0, 0.1),
            border_width=1.0,
        ),
        "pallets_narrow": ame_terrain_gen.MeshPalletsNarrowTerrainCfg(
            proportion=0.0,
            num_bars=1,
            bar_width_range=(0.3, 0.8),
            stone_length_range=(0.2, 0.45),
            stone_distance_range=(0.05, 0.35),
            stone_height_range=(0.0, 0.1),
            border_width=1.0,
        ),
        "beams": ame_terrain_gen.MeshStarWoGroundTerrainCfg(
            proportion=0.0,
            num_bars=1,
            bar_width_range=(0.1, 0.8),
            border_width=1.0,
        ),
    },
)