from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.envs.mdp import UniformPose2dCommand
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.managers import CommandTerm
from isaaclab.markers.visualization_markers import VisualizationMarkers
from isaaclab.utils import index_fill_
from isaaclab.utils.math import quat_apply_inverse, quat_from_euler_xyz, wrap_to_pi, yaw_quat

import instinctlab.utils.math as instinct_math_utils
from instinctlab.envs.mdp import ShadowingCommandBase, BaseLinVelRefCommand

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv
    from isaaclab.sensors import RayCaster, TiledCamera

    from instinctlab.motion_reference.motion_reference_manager import MotionReferenceManagerBase

    from .commands_cfg import (
        LinkContactPosRefCommandCfg, GoalPose2dCommandCfg, BaseLinAngRefCommandCfg
    )


class LinkContactPosRefCommand(ShadowingCommandBase):

    cfg: LinkContactPosRefCommandCfg

    def __init__(self, cfg: LinkContactPosRefCommandCfg, env: ManagerBasedRLEnv, ):
        super().__init__(cfg, env)
        # -- planner_height_scanner
        self.sensor: RayCaster = env.scene.sensors[cfg.sensor_name]

        starts = self.sensor.ray_starts.torch[0, :, :2]
        self.sensor_min = starts.min(dim=0).values
        self.sensor_max = starts.max(dim=0).values

        data_dims = (3,)
        self._command = torch.zeros(
            (self.num_envs, self._motion_reference.num_frames, self._motion_reference.num_link_to_contact, *data_dims),
            device=self.device,
        )
        self._mask = torch.ones(
            (self.num_envs, self._motion_reference.num_frames, self._motion_reference.num_link_to_contact),
            device=self.device,
            dtype=torch.bool,
        )  # (num_envs, num_frames, num_links) # (position mask)
        # generate state buffer in case we need additional frame as a pseudo command frame
        if self.cfg.current_state_command:
            self._current_state = torch.zeros(
                (self.num_envs, 1, self._motion_reference.num_link_to_contact, *data_dims),
                device=self.device,
            )
            self._current_state_mask = torch.ones(
                (self.num_envs, 1, self._motion_reference.num_link_to_contact),
                device=self.device,
                dtype=torch.bool,
            )  # (num_envs, 1, num_links, 1) # (position mask)
        self._update_command()

    @property
    def mask(self) -> torch.Tensor:
        mask = torch.logical_and(
            self._mask,
            self._motion_reference.data.validity.unsqueeze(-1),
        )
        if self.cfg.current_state_command:
            return torch.cat([mask, self._current_state_mask], dim=1)
        return mask

    def _update_command_by_env_ids(self, env_ids: Sequence[int] | torch.Tensor):
        link_contact_next_pos_w = self._motion_reference.data.link_contact_next_pos_w[env_ids]

        sensor_pos_w = self.sensor.data.pos_w.torch[env_ids]
        sensor_quat_w = self.sensor.data.quat_w.torch[env_ids]
        rel_w = link_contact_next_pos_w - sensor_pos_w[:, None, None, :]
        link_contact_next_pos_t = math_utils.quat_apply_inverse(
            math_utils.yaw_quat(sensor_quat_w)[:, None, None, :].expand(-1, rel_w.shape[1], rel_w.shape[2], -1), rel_w
        )
        # out of planner_height_scanner
        in_sensor = (
            (link_contact_next_pos_t[..., 0] >= self.sensor_min[0]) & (link_contact_next_pos_t[..., 0] <= self.sensor_max[0])
            & (link_contact_next_pos_t[..., 1] >= self.sensor_min[1]) & (link_contact_next_pos_t[..., 1] <= self.sensor_max[1])
        )

        root_pos_w = self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w.torch[env_ids]
        root_quat_w = self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w.torch[env_ids]
        rel_root_w = link_contact_next_pos_w - root_pos_w[:, None, None, :]
        link_contact_next_pos_b = math_utils.quat_apply_inverse(
            math_utils.yaw_quat(root_quat_w)[:, None, None, :].expand(-1, rel_root_w.shape[1], rel_root_w.shape[2], -1), rel_root_w
        )

        link_contact_next_pos_b = torch.where(in_sensor.unsqueeze(-1), link_contact_next_pos_b, torch.full_like(link_contact_next_pos_b, float("nan")))

        self._command[env_ids, :, :, :] = (
            torch.nan_to_num(link_contact_next_pos_b, nan=0.0)
            * self._motion_reference.data.validity[env_ids].unsqueeze(-1).unsqueeze(-1)
            * self._mask[env_ids].unsqueeze(-1)
        )

    def _debug_vis_callback(self, event):
        if not hasattr(self, "_vis_link_contact_pos_w"):
            return
        self._visualizer.visualize(
            translations=self._vis_link_contact_pos_w,
        )

    def _compute_debug_vis_data(self):
        root_pos_w = self._env.scene[self.cfg.asset_cfg.name].data.root_pos_w.torch
        yaw_q = math_utils.yaw_quat(self._env.scene[self.cfg.asset_cfg.name].data.root_quat_w.torch)

        p_b = self._command[:, 0]
        valid = self._mask[:, 0]

        yaw_q = yaw_q[:, None, :].expand(-1, p_b.shape[1], -1)
        p_w = math_utils.quat_apply(yaw_q, p_b) + root_pos_w[:, None, :]

        self._vis_link_contact_pos_w = p_w[valid]


class GoalPose2dCommand(CommandTerm):

    cfg: GoalPose2dCommandCfg

    def __init__(self, cfg: GoalPose2dCommandCfg, env: ManagerBasedEnv):
        # initialize the base class
        super().__init__(cfg, env)
        self._motion_reference: MotionReferenceManagerBase = env.scene[cfg.motion_reference.name]

        # obtain the robot and terrain assets
        # -- planner_height_scanner
        self.sensor: RayCaster = env.scene.sensors[cfg.sensor_name]

        # crete buffers to store the command
        # -- commands: (x, y, z, heading)
        self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
        self.pos_command_b = torch.zeros(self.num_envs, 2, device=self.device)
        self.heading_command_b = torch.zeros_like(self.heading_command_w)
        # -- metrics
        self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)

        # adds (optional) cmd kind and element names for leapp export
        # during export, semantic data about this command will be used to annotate the command input
        self.cfg.cmd_kind = self.cfg.cmd_kind or "command/body/pose"
        self.cfg.element_names = self.cfg.element_names or ["x", "y", "heading"]

    def __str__(self) -> str:
        msg = "GoalPositionCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired 2D-pose in base frame. Shape is (num_envs, 3)."""
        return torch.cat([self.pos_command_b, self.heading_command_b.unsqueeze(1)], dim=1)

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        # logs data
        self.metrics["error_pos"] = torch.linalg.norm(
            self.pos_command_w[:, :2] - self.sensor.data.pos_w.torch[:, :2], dim=1
        )
        self.metrics["error_heading"] = torch.abs(wrap_to_pi(
            self.heading_command_w - math_utils.euler_xyz_from_quat(self.sensor.data.quat_w.torch)[2])
        )

    def _resample_command(self, env_ids: Sequence[int]):
        goal_base_pos_w = self._motion_reference.data.goal_base_pos_w.torch[env_ids][:, 0]
        goal_base_quat_w = self._motion_reference.data.goal_base_quat_w.torch[env_ids][:, 0]
        self.pos_command_w[env_ids] = goal_base_pos_w
        self.heading_command_w[env_ids] = math_utils.euler_xyz_from_quat(goal_base_quat_w)[2]
        valid = self._motion_reference.data.validity.torch[env_ids][:, 0]
        self.time_left[env_ids] = torch.where(
            valid, self.time_left[env_ids], torch.zeros_like(self.time_left[env_ids])
        )

    def _update_command(self):
        base_pos_w = self.sensor.data.pos_w.torch
        base_quat_w = self.sensor.data.quat_w.torch
        base_yaw = math_utils.euler_xyz_from_quat(base_quat_w)[2]

        target_vec = self.pos_command_w - base_pos_w
        target_vec[:, 2] = 0.0
        d_xy = quat_apply_inverse(yaw_quat(base_quat_w), target_vec)[:, :2]

        dist = torch.linalg.norm(d_xy, dim=1, keepdim=True)
        self.pos_command_b[:] = d_xy / torch.clamp(dist, min=self.cfg.radius)

        self.heading_command_b[:] = wrap_to_pi(self.heading_command_w - base_yaw)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer = VisualizationMarkers(self.cfg.goal_pose_visualizer_cfg)
            # set their visibility to true
            self.goal_pose_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pose_visualizer"):
                self.goal_pose_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the box marker
        self.goal_pose_visualizer.visualize(
            translations=self.pos_command_w,
            orientations=quat_from_euler_xyz(
                torch.zeros_like(self.heading_command_w),
                torch.zeros_like(self.heading_command_w),
                self.heading_command_w,
            ),
            environment_ids=self._env.scene._ALL_INDICES,
        )
