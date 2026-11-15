from __future__ import annotations

import inspect
import os
from collections.abc import Sequence
from copy import copy
from prettytable import PrettyTable
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import isaaclab.utils.string as string_utils
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveScene
from isaaclab.sensors import SensorBase
from isaaclab.utils.buffers import TimestampedBuffer

from instinctlab.motion_reference.motion_reference_data import MotionReferenceData, MotionReferenceState
from instinctlab.utils.backend_dispatch import create_backend_component
from instinctlab.utils.prims import articulation_name_order

if TYPE_CHECKING:
    from instinctlab.motion_reference.motion_buffer import MotionBuffer
    from instinctlab.motion_reference.motion_reference_cfg import MotionReferenceManagerCfg

import numpy as np
import torch
import torch.distributed as dist

import pytorch_kinematics as pk
import warp as wp

from instinctlab.motion_reference.motion_reference_manager import MotionReferenceManager, MotionReferenceManagerBase


class ContactReferenceManagerBase(MotionReferenceManagerBase):

    @property
    def num_link_to_contact(self) -> int:
        if self.cfg.robot_model_path is None or self.cfg.link_of_contacts is None:
            return 0
        else:
            return len(self.cfg.link_of_contacts)

    def _initialize_motion_buffers(self):
        assert len(self.cfg.motion_buffers) > 0, "At least one motion buffer should be specified."
        for buffer_cfg in self.cfg.motion_buffers.values():
            buffer_cfg.link_of_contacts = self.cfg.link_of_contacts
        super()._initialize_motion_buffers()   # 原来的流程一点不动

    def _prepare_data_class_kwargs(self) -> dict:
        make_empty_kwargs = super()._prepare_data_class_kwargs()
        make_empty_kwargs["num_contact_links"] = self.num_link_to_contact
        return make_empty_kwargs

    def _set_debug_vis_impl(self, debug_vis: bool):
        super()._set_debug_vis_impl(debug_vis)
        if debug_vis and self.cfg.contact_visualizing_marker_types:
            if not hasattr(self, "_contact_visualizer"):
                self._contact_visualizer = VisualizationMarkers(self.cfg.contact_visualizer_cfg)
            # set their visibility to true
            self._contact_visualizer.set_visibility(True)
        else:
            if hasattr(self, "_contact_visualizer"):
                self._contact_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        super()._debug_vis_callback(event)
        if self.cfg.contact_visualizing_marker_types:
            if not hasattr(self, "_view") or not hasattr(self, "_contact_visualizer"):
                return
            aiming_frame_idx = self.aiming_frame_idx

            pos_list = []
            index_list = []

            if "link_contact_pos_w" in self.cfg.contact_visualizing_marker_types:
                contact = self.data.link_contact[self.ALL_INDICES, aiming_frame_idx]
                link_contact_pos_w = self.data.link_contact_pos_w[self.ALL_INDICES, aiming_frame_idx][contact > 0.5]
                contact_indices = torch.zeros(link_contact_pos_w.shape[0], device=self.device, dtype=torch.long)

                pos_list.append(link_contact_pos_w.reshape(-1, 3))
                index_list.append(contact_indices.reshape(-1))

            if pos_list:
                pos_w = torch.cat(pos_list, dim=0)
                indices = torch.cat(index_list, dim=0)
                self._contact_visualizer.visualize(translations=pos_w, marker_indices=indices)


class ContactReferenceManager(MotionReferenceManager):
    """Construct the motion-reference manager for the active physics backend."""

    def __new__(cls, cfg: MotionReferenceManagerCfg):
        return create_backend_component(
            cfg,
            {
                "physx": "acp.motion_reference.physx:PhysxContactReferenceManager",
                "ovphysx": "acp.motion_reference.ovphysx:OvPhysxContactReferenceManager",
                "newton": "acp.motion_reference.newton:NewtonContactReferenceManager",
            },
        )
