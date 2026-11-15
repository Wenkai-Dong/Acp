# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions to generate different terrains using the ``trimesh`` library."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import scipy.spatial.transform as tf
import trimesh

from isaaclab.terrains.trimesh.utils import make_border, make_box, make_plane
from isaaclab.terrains.utils import *  # noqa: F401, F403

if TYPE_CHECKING:
    from . import mesh_terrains_cfg

ground_z = -0.4

def make_flat_border(size, inner_size, position, wall_bottom=None):
    cx, cy = position[0], position[1]
    tx = (size[0] - inner_size[0]) / 2  # 左、右两条的宽
    ty = (size[1] - inner_size[1]) / 2  # 上、下两条的宽
    meshes = []
    # 上、下两条：横跨整个 x 方向
    for sign in (1, -1):
        strip = make_plane((size[0], ty), 0.0, center_zero=True)
        strip.apply_translation((cx, cy + sign * (inner_size[1] + ty) / 2, 0.0))
        meshes.append(strip)
    # 左、右两条：夹在上、下两条之间
    for sign in (1, -1):
        strip = make_plane((tx, inner_size[1]), 0.0, center_zero=True)
        strip.apply_translation((cx + sign * (inner_size[0] + tx) / 2, cy, 0.0))
        meshes.append(strip)
    # 平台内侧的 4 面竖墙（法线朝向中间的坑）
    if wall_bottom is not None:
        ix, iy = inner_size[0] / 2, inner_size[1] / 2
        corners = [(cx - ix, cy - iy), (cx + ix, cy - iy), (cx + ix, cy + iy), (cx - ix, cy + iy)]
        for i in range(4):
            (x0, y0), (x1, y1) = corners[i], corners[(i + 1) % 4]
            vertices = np.array(
                [[x0, y0, wall_bottom], [x1, y1, wall_bottom], [x1, y1, 0.0], [x0, y0, 0.0]]
            )
            meshes.append(trimesh.Trimesh(vertices=vertices, faces=np.array([[0, 2, 1], [0, 3, 2]])))
    return meshes


def stepping_stones_height_terrain(
        difficulty: float, cfg: mesh_terrains_cfg.MeshSteppingStonesTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    # resolve terrain configuration
    stone_width = cfg.stone_width_range[1] - difficulty * (cfg.stone_width_range[1] - cfg.stone_width_range[0])
    stone_distance = cfg.stone_distance_range[0] + difficulty * (
        cfg.stone_distance_range[1] - cfg.stone_distance_range[0]
    )
    stone_height_max = cfg.stone_height_max[0] + difficulty * (cfg.stone_height_max[1] - cfg.stone_height_max[0])
    if cfg.max_yx_angle:
        max_yx_angle = cfg.max_yx_angle[0] + difficulty * (cfg.max_yx_angle[1] - cfg.max_yx_angle[0])

    # initialize list of meshes
    meshes_list = list()
    # constants for terrain generation
    ob_height = 1.0
    terrain_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -ob_height / 2)
    # Generate the outer ring
    outer_size = [cfg.size[0], cfg.size[1]]
    inner_size = [cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width]
    meshes_list += make_flat_border(outer_size, inner_size, terrain_center, wall_bottom=ground_z)
    # add the stones
    start_x, start_y = 0 + cfg.border_width, 0 + cfg.border_width
    # -- if the terrain is longer than it is wide then fill the terrain column by column
    while start_y < cfg.size[1] - cfg.border_width:
        # ensure that stone stops along y-axis
        stop_y = min(cfg.size[1] - cfg.border_width, start_y + stone_width)
        # randomly sample x-position
        start_x = np.random.uniform(0 + cfg.border_width, stone_width + cfg.border_width)
        stop_x = max(0 + cfg.border_width, start_x - stone_distance)
        # fill first stone
        length = stop_x - cfg.border_width
        width = stop_y - start_y
        height = ob_height
        center = ((stop_x + cfg.border_width) / 2, (stop_y + start_y) / 2, -ob_height / 2)
        if cfg.max_yx_angle:
            stone = make_box(length=length, width=width, height=height, center=center, max_yx_angle=max_yx_angle,)
        else:
            stone = trimesh.creation.box((length, width, height), trimesh.transformations.translation_matrix(center))
        meshes_list.append(stone)
        # fill row with stones
        while start_x < cfg.size[0] - cfg.border_width:
            stop_x = min(cfg.size[0] - cfg.border_width, start_x + stone_width)
            length = stop_x - start_x
            width = stop_y - start_y
            height = ob_height
            center = ((stop_x + start_x) / 2, (stop_y + start_y) / 2,
                      -ob_height / 2 + np.random.uniform(-stone_height_max, stone_height_max))
            if stop_x >= cfg.size[0] - cfg.border_width:
                center = ((stop_x + start_x) / 2, (stop_y + start_y) / 2, -ob_height / 2)

            if cfg.max_yx_angle:
                stone = make_box(length=length, width=width, height=height, center=center, max_yx_angle=max_yx_angle, )
            else:
                stone = trimesh.creation.box((length, width, height),trimesh.transformations.translation_matrix(center))
            meshes_list.append(stone)

            start_x += stone_width + stone_distance
        # update y-position
        start_y += stone_width + stone_distance

    # Generate the ground
    ground = make_plane(inner_size, ground_z, center_zero=True)
    ground.apply_translation((terrain_center[0], terrain_center[1], 0.0))
    meshes_list.append(ground)

    # specify the origin of the terrain
    origin = np.array([terrain_center[0], terrain_center[1], 0.0])

    return meshes_list, origin


def pallets_narrow_terrain(
    difficulty: float, cfg: mesh_terrains_cfg.MeshStarGapTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:

    # resolve the terrain configuration
    bar_width = cfg.bar_width_range[1] - difficulty * (cfg.bar_width_range[1] - cfg.bar_width_range[0])
    stone_length = cfg.stone_length_range[1] - difficulty * (cfg.stone_length_range[1] - cfg.stone_length_range[0])
    stone_distance = cfg.stone_distance_range[0] + difficulty * (cfg.stone_distance_range[1] - cfg.stone_distance_range[0])
    stone_height_max = cfg.stone_height_range[0] + difficulty * (cfg.stone_height_range[1] - cfg.stone_height_range[0])

    # initialize list of meshes
    meshes_list = list()
    # constants for terrain generation
    ob_height = 1.0
    # Generate a platform in the middle
    platform_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -ob_height / 2)
    # Generate bars to connect the platform to the terrain
    transform = np.eye(4)
    transform[:3, -1] = np.asarray(platform_center)
    yaw = 0.5 * np.pi
    for _ in range(cfg.num_bars):
        # compute the length of the bar based on the yaw
        # length changes since the bar is connected to a square border
        c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
        half_inner = min(
            (0.5 * cfg.size[0] - cfg.border_width) / c if c > 1e-6 else np.inf,
            (0.5 * cfg.size[1] - cfg.border_width) / s if s > 1e-6 else np.inf,
        )
        # compute the transform of the bar
        transform[0:3, 0:3] = tf.Rotation.from_euler("z", yaw).as_matrix()
        # add the bar to the mesh
        dim = [stone_length, bar_width, ob_height]
        transform[:3, -1] = np.asarray(platform_center)
        start = 0.5 * stone_distance
        while start + stone_length <= half_inner:
            stone_center = start + 0.5 * stone_length
            transform[0, -1] = platform_center[0] - stone_center * math.cos(yaw)
            transform[1, -1] = platform_center[1] - stone_center * math.sin(yaw)
            if start + stone_length + stone_distance + stone_length > half_inner:
                transform[2, -1] = platform_center[2]
            else:
                transform[2, -1] = platform_center[2] + np.random.uniform(-stone_height_max, stone_height_max)
            bar = trimesh.creation.box(dim, transform)
            meshes_list.append(bar)
            start += stone_length + stone_distance
        transform[:3, -1] = np.asarray(platform_center)
        start = 0.5 * stone_distance
        while start + stone_length <= half_inner:
            stone_center = start + 0.5 * stone_length
            transform[0, -1] = platform_center[0] + stone_center * math.cos(yaw)
            transform[1, -1] = platform_center[1] + stone_center * math.sin(yaw)
            if start + stone_length + stone_distance + stone_length > half_inner:
                transform[2, -1] = platform_center[2]
            else:
                transform[2, -1] = platform_center[2] + np.random.uniform(-stone_height_max, stone_height_max)
            bar = trimesh.creation.box(dim, transform)
            meshes_list.append(bar)
            start += stone_length + stone_distance
        transform[:3, -1] = np.asarray(platform_center)
        # increment the yaw
        yaw += np.pi / cfg.num_bars
    # Generate the exterior border
    inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
    meshes_list += make_flat_border(cfg.size, inner_size, platform_center, wall_bottom=ground_z)

    # Generate the ground
    ground = make_plane(inner_size, ground_z, center_zero=True)
    ground.apply_translation((platform_center[0], platform_center[1], 0.0))
    meshes_list.append(ground)
    # specify the origin of the terrain
    origin = np.asarray([0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0])

    return meshes_list, origin

def star_terrain_wo_ground(
    difficulty: float, cfg: mesh_terrains_cfg.MeshStarTerrainCfg
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    # resolve the terrain configuration
    bar_width = cfg.bar_width_range[1] - difficulty * (cfg.bar_width_range[1] - cfg.bar_width_range[0])

    # initialize list of meshes
    meshes_list = list()
    # constants for terrain generation
    ob_height = 1.0
    # Generate a platform in the middle
    platform_center = (0.5 * cfg.size[0], 0.5 * cfg.size[1], -ob_height / 2)
    # Generate bars to connect the platform to the terrain
    transform = np.eye(4)
    transform[:3, -1] = np.asarray(platform_center)
    yaw = 0.5 * np.pi
    for _ in range(cfg.num_bars):
        # compute the length of the bar based on the yaw
        # length changes since the bar is connected to a square border
        c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
        half_inner = min(
            (0.5 * cfg.size[0] - cfg.border_width) / c if c > 1e-6 else np.inf,
            (0.5 * cfg.size[1] - cfg.border_width) / s if s > 1e-6 else np.inf,
        )
        # compute the transform of the bar
        transform[0:3, 0:3] = tf.Rotation.from_euler("z", yaw).as_matrix()
        # add the bar to the mesh
        dim = [2 * half_inner, bar_width, ob_height]
        bar = trimesh.creation.box(dim, transform)
        meshes_list.append(bar)
        # increment the yaw
        yaw += np.pi / cfg.num_bars
    # Generate the exterior border
    inner_size = (cfg.size[0] - 2 * cfg.border_width, cfg.size[1] - 2 * cfg.border_width)
    meshes_list += make_flat_border(cfg.size, inner_size, platform_center, wall_bottom=ground_z)
    # specify the origin of the terrain
    origin = np.asarray([0.5 * cfg.size[0], 0.5 * cfg.size[1], 0.0])

    # Generate the ground
    ground = make_plane(inner_size, ground_z, center_zero=True)
    ground.apply_translation((platform_center[0], platform_center[1], 0.0))
    meshes_list.append(ground)

    return meshes_list, origin