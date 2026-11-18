from __future__ import annotations

import torch
from collections.abc import Sequence
from dataclasses import dataclass, fields

import warp as wp

from isaaclab.utils.warp import ProxyArray

import instinctlab.utils.torch as torch_utils
from instinctlab.motion_reference.motion_reference_data import MotionReferenceData


@dataclass
class ContactSequence:

    link_contact: torch.Tensor = None  # type: ignore

    link_contact_pos_w: torch.Tensor = None  # type: ignore

    link_contact_next_frame: torch.Tensor = None  # type: ignore

    framerate: torch.Tensor = None  # type: ignore
    """ The framerate of the motion data.

    Shape: [N,], N is the batch size.
    """

    buffer_length: torch.Tensor = None  # type: ignore
    """ The number of frames of each motion data across batch

    Shape: [N,], N is the batch size.
    """

    attrs_with_frame_dim: tuple = (
        "link_contact",
        "link_contact_pos_w",
        "link_contact_next_frame",
    )
    attrs_only_batch_dim: tuple = ("framerate", "buffer_length")

    @staticmethod
    def make_empty(
        batch_size: int,
        num_frames: int,
        num_contact_links: int,
        device=torch.device("cpu"),
    ) -> ContactSequence:
        return ContactSequence(
            link_contact=torch.zeros(batch_size, num_frames, num_contact_links, device=device),
            link_contact_pos_w=torch.zeros(batch_size, num_frames, num_contact_links, 3, device=device),
            link_contact_next_frame=torch.zeros(batch_size, num_frames, num_contact_links, device=device),
            framerate=torch.zeros(batch_size, dtype=torch.int, device=device),
            buffer_length=torch.zeros(batch_size, dtype=torch.int, device=device),
        )

    @staticmethod
    def make_emtpy_concat_batch(
        buffer_lengths: Sequence[int],
        num_contact_links: int,
        device=torch.device("cpu"),
    ) -> ContactSequence:
        return ContactSequence(
            link_contact=torch_utils.ConcatBatchTensor(
                batch_sizes=buffer_lengths, data_shape=(num_contact_links,), device=device
            ),  # type: ignore
            link_contact_pos_w=torch_utils.ConcatBatchTensor(
                batch_sizes=buffer_lengths, data_shape=(num_contact_links, 3), device=device
            ),  # type: ignore
            link_contact_next_frame=torch_utils.ConcatBatchTensor(
                batch_sizes=buffer_lengths, data_shape=(num_contact_links,), device=device, dtype=torch.long,
            ),  # type: ignore
            framerate=torch.zeros((len(buffer_lengths),), device=device),
            buffer_length=torch.tensor(buffer_lengths, device=device),
        )

    def __post_init__(self):
        if not self.link_contact is None:
            # all attributes should have the same leading dimensions
            assert all(
                getattr(self, attr).shape[0] == self.link_contact.shape[0] for attr in self.attrs_with_frame_dim
            ), "All attributes should have the same leading dimension."
        else:
            # all attributes should be None
            assert all(
                getattr(self, attr) is None for attr in (self.attrs_with_frame_dim + self.attrs_only_batch_dim)
            ), "All attributes should be None or not None."


@dataclass
class ContactReferenceData(MotionReferenceData):

    link_contact: torch.Tensor = None   # type: ignore

    link_contact_pos_w: torch.Tensor = None # type: ignore

    link_contact_next_pos_w: torch.Tensor = None    # type: ignore

    goal_base_pos_w: torch.Tensor = None    # type: ignore

    goal_base_quat_w: torch.Tensor = None   # type: ignore

    goal_time_ahead: torch.Tensor = None    # type: ignore

    @staticmethod
    def make_empty(
        num_envs: int,
        num_frames: int,
        num_joints: int,
        num_links: int,
        num_contact_links: int,
        device=torch.device("cpu"),
        **kwargs,
    ) -> ContactReferenceData:
        base = MotionReferenceData.make_empty(num_envs, num_frames, num_joints, num_links, device=device, **kwargs)
        values = {f.name: getattr(base, f.name) for f in fields(MotionReferenceData)}
        values["link_contact"] = ProxyArray(wp.zeros((num_envs, num_frames, num_contact_links), device=str(device)))
        values["link_contact_pos_w"] = ProxyArray(wp.zeros((num_envs, num_frames, num_contact_links, 3), device=str(device)))
        values["link_contact_next_pos_w"] = ProxyArray(wp.zeros((num_envs, num_frames, num_contact_links, 3), device=str(device)))
        values["goal_base_pos_w"] = ProxyArray(wp.zeros((num_envs, 1, 3), device=str(device)))
        values["goal_base_quat_w"] = ProxyArray(wp.zeros((num_envs, 1, 4), device=str(device)))
        values["goal_time_ahead"] = ProxyArray(wp.zeros((num_envs, 1), device=str(device)))
        return ContactReferenceData(**values)
