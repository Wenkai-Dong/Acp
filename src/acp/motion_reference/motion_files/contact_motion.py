from __future__ import annotations

import os
import numpy as np
import torch
import yaml
from collections.abc import Sequence
from typing import TYPE_CHECKING, Callable, Protocol
from typing import TYPE_CHECKING

import joblib

from instinctlab.motion_reference import MotionSequence
from instinctlab.motion_reference.motion_files.amass_motion import AmassMotion
from instinctlab.motion_reference.motion_files.terrain_motion import TerrainMotion
from instinctlab.motion_reference.motion_buffer import MotionReferenceArticulationAccess

from acp.motion_reference.contact_reference_data import ContactSequence, ContactReferenceData

if TYPE_CHECKING:
    from instinctlab.motion_reference.motion_reference_cfg import MotionBufferCfg


class ContactMotion(TerrainMotion):

    def __init__(
        self,
        cfg: MotionBufferCfg,
        articulation_access: MotionReferenceArticulationAccess,
        link_of_interests: Sequence[str],
        forward_kinematics_func: Callable[..., torch.Tensor],
        device: torch.device,
    ) -> None:
        super().__init__(cfg, articulation_access, link_of_interests, forward_kinematics_func, device)
        self.link_of_contacts = cfg.link_of_contacts
        self.start_phase = torch.full((self.num_trajectories,), cfg.init_start_phase, dtype=torch.float, device=self.buffer_device,)

    @property
    def num_link_to_contact(self) -> int:
        """The number of links to reference in the motion buffer."""
        return len(self.link_of_contacts)

    def _load_motion_sequences(self):
        super()._load_motion_sequences()

        all_contact_sequences = list(map(self._read_contact_file, range(len(self._all_motion_files))))
        print(f"[AMASS Motion] All {len(all_contact_sequences)} motion files loaded.")

        # fill the motion sequences into a single pytorch tensor buffer, to accelerate the access
        # max_buffer_length = max([int(contact.buffer_length) for contact in all_contact_sequences])
        # self.all_contact_sequences = ContactSequence.make_empty(
        #     len(all_contact_sequences),
        #     max_buffer_length,
        #     num_contact_links=self.num_link_to_contact,
        #     device=self.buffer_device
        # )
        print(
            "[AMASS Contact] buffer lengths statistics:"
            f" mean: {np.array([contact.buffer_length for contact in all_contact_sequences]).mean()},"
            f" max: {np.array([contact.buffer_length for contact in all_contact_sequences]).max()},"
            f" min: {np.array([contact.buffer_length for contact in all_contact_sequences]).min()},"
        )
        self._all_contact_sequences = ContactSequence.make_emtpy_concat_batch(
            buffer_lengths=[int(contact.buffer_length) for contact in all_contact_sequences],
            num_contact_links=self.num_link_to_contact,
            device=self.buffer_device,
        )
        for i, contact in enumerate(all_contact_sequences):
            for attr in self._all_contact_sequences.attrs_with_frame_dim:
                getattr(self._all_contact_sequences, attr)[i, : contact.buffer_length] = getattr(contact, attr)
            for attr in self._all_contact_sequences.attrs_only_batch_dim:
                getattr(self._all_contact_sequences, attr)[i] = getattr(contact, attr)

    def _read_contact_file(self, motion_file_idx: int) -> ContactSequence:
        filepath = self._all_motion_files[motion_file_idx]
        for suffix in ("retargetted.npz", "retargeted.npz", "soma.csv"):
            if filepath.endswith(suffix):
                filepath = filepath[: -len(suffix)]
                break
        contact_path = filepath + "contact.npz"
        if not os.path.exists(contact_path):
            raise FileNotFoundError(f"Cannot find the contact file：{contact_path}")

        raw_data = np.load(contact_path, mmap_mode="r", allow_pickle=True)
        framerate = raw_data["framerate"].item()
        link_names = (
            raw_data["link_names"] if isinstance(raw_data["link_names"], list) else raw_data["link_names"].tolist()
        )
        link_contact = torch.as_tensor(raw_data["link_contact"], device=self.buffer_device, dtype=torch.float)
        link_contact_pos_w = torch.as_tensor(raw_data["link_contact_pos_w"], device=self.buffer_device, dtype=torch.float)

        link_ids = [link_names.index(j_name) for j_name in self.link_of_contacts]
        link_contact = link_contact[:, link_ids]
        link_contact_pos_w = link_contact_pos_w[:, link_ids]

        # calculate link contact next frame
        link_contact_next_frame = self._calculate_link_contact_next_frame(link_contact, float(framerate))

        motion_length = int(self._all_motion_sequences.buffer_length[motion_file_idx])
        motion_fps = float(self._all_motion_sequences.framerate[motion_file_idx])
        name = os.path.basename(contact_path)
        if link_contact.shape[0] - motion_length > 1:
            raise ValueError(
                f"{name}: contact {link_contact.shape[0]} frames, motion {motion_length} frames (after interpolation)")
        if float(framerate) != motion_fps:
            raise ValueError(f"{name}: contact {float(framerate)} fps, motion {motion_fps} fps")

        return ContactSequence(
            link_contact=link_contact,
            link_contact_pos_w=link_contact_pos_w,
            link_contact_next_frame=link_contact_next_frame,
            framerate=torch.as_tensor(framerate),
            buffer_length=torch.as_tensor(link_contact.shape[0]),
        )

    def _calculate_link_contact_next_frame(
            self, link_contact: torch.Tensor, framerate: float, horizon_s: float = 1.5
    ) -> torch.Tensor:
        c = link_contact > 0.5
        T, L = c.shape

        onset = torch.zeros_like(c)
        onset[1:] = c[1:] & ~c[:-1]

        BIG = T + 1
        idx = torch.arange(T, device=c.device).unsqueeze(-1).expand(T, L)
        onset_idx = torch.where(onset, idx, torch.full_like(idx, BIG))

        suffix_min = torch.flip(torch.cummin(torch.flip(onset_idx, [0]), dim=0).values, [0])

        next_frame = torch.full_like(onset_idx, BIG)
        next_frame[:-1] = suffix_min[1:]

        H = int(round(horizon_s * framerate))
        ok = (next_frame < BIG) & (next_frame - idx <= H)
        return torch.where(ok, next_frame, torch.zeros_like(next_frame))


    def fill_motion_data(
        self,
        env_ids: torch.Tensor,
        sample_timestamp: torch.Tensor,
        env_origins: torch.Tensor,
        data_buffer: ContactReferenceData,
    ) -> None:
        super().fill_motion_data(env_ids, sample_timestamp, env_origins, data_buffer)
        assigned_ids = self.env_ids_to_assigned_ids(env_ids).to(self.buffer_device)

        # compute the frame selection for each env
        frame_selections = torch.round(
            (self._motion_buffer_start_time_s[assigned_ids].unsqueeze(-1) + sample_timestamp.to(self.buffer_device))
            * self._all_motion_sequences.framerate[self._assigned_env_motion_selection[assigned_ids]].unsqueeze(-1)
        )  # shape [num_env_ids, num_frames]

        # check if the sample_frame_idx is out of range
        frame_selections = torch.where(
            data_buffer.validity[env_ids].to(self.buffer_device),
            frame_selections,
            self._all_motion_sequences.buffer_length[self._assigned_env_motion_selection[assigned_ids]].unsqueeze(-1)
            - 1,
        ).to(torch.long)

        # flattened indexing to get the data from all motion sequences
        assigned_ids_across_frame = assigned_ids.unsqueeze(-1).repeat(
            1, frame_selections.shape[1]
        )  # shape [num_env_ids, num_frames]
        assigned_ids_across_frame_ = assigned_ids_across_frame.flatten()
        frame_selections_ = frame_selections.flatten()

        # contact
        data_buffer.link_contact[env_ids] = (
            self._all_contact_sequences.link_contact[
                self._assigned_env_motion_selection[assigned_ids_across_frame_],
                frame_selections_,
            ]
            .reshape(env_ids.numel(), *data_buffer.link_contact.shape[1:])
            .to(self.output_device)
        )
        data_buffer.link_contact_pos_w[env_ids] = (
            self._all_contact_sequences.link_contact_pos_w[
                self._assigned_env_motion_selection[assigned_ids_across_frame_],
                frame_selections_,
            ]
            .reshape(env_ids.numel(), *data_buffer.link_contact_pos_w.shape[1:])
            .to(self.output_device)
        )

        # update the base_pos_w and link_pos_w with the env_origins
        data_buffer.link_contact_pos_w[env_ids] += (
            self._get_motion_based_origin(env_origins, env_ids).unsqueeze(1).unsqueeze(1)
        )  # avoiding inplace operation

        # link contact next position world
        data_buffer.link_contact_next_pos_w[env_ids] = (
            self._get_link_contact_next_pos_w(assigned_ids_across_frame_, frame_selections_)
            .reshape(env_ids.numel(), *data_buffer.link_contact_next_pos_w.shape[1:])
            .to(self.output_device)
        )
        # update the link_contact_next_pos_w with the env_origins
        data_buffer.link_contact_next_pos_w[env_ids] += (
            self._get_motion_based_origin(env_origins, env_ids).unsqueeze(1).unsqueeze(1)
        )

        # compute the frame selection for each env
        frame_simple_ = self._frame_simple(assigned_ids=assigned_ids, sample_timestamp=sample_timestamp, goal_range=(0.5,1.5))
        # goal torso_link position world
        data_buffer.goal_base_pos_w[env_ids] = (
            self._all_motion_sequences.link_pos_w[
                self._assigned_env_motion_selection[assigned_ids_across_frame_],
                frame_simple_,
            ][:, None, self.link_of_interests.index("torso_link")]
            .reshape(env_ids.numel(), *data_buffer.goal_base_pos_w.shape[1:])
            .to(self.output_device)
        )
        # update the goal_base_pos_w with the env_origins
        data_buffer.goal_base_pos_w[env_ids] += (self._get_motion_based_origin(env_origins, env_ids).unsqueeze(1))

        # goal torso_link quat world
        data_buffer.goal_base_quat_w[env_ids] = (
            self._all_motion_sequences.link_quat_w[
                self._assigned_env_motion_selection[assigned_ids_across_frame_],
                frame_simple_,
            ][:, None, self.link_of_interests.index("torso_link")]
            .reshape(env_ids.numel(), *data_buffer.goal_base_quat_w.shape[1:])
            .to(self.output_device)
        )


    def _get_link_contact_next_pos_w(self, assigned_ids_across_frame_, frame_selections_):
        # link contact next position world
        link_contact_next_frame = self._all_contact_sequences.link_contact_next_frame[
            self._assigned_env_motion_selection[assigned_ids_across_frame_],
            frame_selections_,
        ]
        valid = (link_contact_next_frame >= 0.5).unsqueeze(-1)

        link_contact_pos_w = self._all_contact_sequences.link_contact_pos_w
        row = link_contact_pos_w._batch_starts[self._assigned_env_motion_selection[assigned_ids_across_frame_]].unsqueeze(-1) + link_contact_next_frame
        body = torch.arange(link_contact_next_frame.shape[-1], device=row.device)
        link_contact_next_pos_w = link_contact_pos_w._concatenated_tensor[row, body]
        return torch.where(valid, link_contact_next_pos_w, torch.full_like(link_contact_next_pos_w, float("nan")))

    def _frame_simple(
        self,
        assigned_ids,
        sample_timestamp,
        goal_range,
        ):
        sample_timestamp = sample_timestamp[:, :1].to(self.buffer_device)
        goal_timestamp = sample_timestamp + torch.rand_like(sample_timestamp) * (goal_range[1] - goal_range[0]) + goal_range[0]
        # compute the frame selection for each env
        frame_selections = torch.round(
            (self._motion_buffer_start_time_s[assigned_ids].unsqueeze(-1) + goal_timestamp)
            * self._all_motion_sequences.framerate[self._assigned_env_motion_selection[assigned_ids]].unsqueeze(-1)
        )  # shape [num_env_ids, num_frames]

        validity = (
            (
                frame_selections
                < self._all_motion_sequences.buffer_length[self._assigned_env_motion_selection[assigned_ids]].unsqueeze(
                    -1
                )
            )
            .to(torch.bool)
            .to(self.output_device)
        )

        # check if the sample_frame_idx is out of range
        frame_selections = torch.where(
            validity.to(self.buffer_device),
            frame_selections,
            self._all_motion_sequences.buffer_length[self._assigned_env_motion_selection[assigned_ids]].unsqueeze(-1)
            - 1,
        ).to(torch.long)

        return frame_selections.flatten()

    def _sample_assigned_env_motion_selection(self, assigned_ids: torch.Tensor) -> None:
        """Resample the motion selection for the buffer, build the motion selection if not built.
        This samples the env_motion_selection independent from the start time.
        """
        if len(assigned_ids) == 0:
            return
        weights = self._motion_weights * (self.start_phase + 0.1)
        self._assigned_env_motion_selection[assigned_ids] = torch.multinomial(
            weights,
            len(assigned_ids),
            replacement=True,
        ).to(self.buffer_device)

    def _sample_env_motion_start_time(self, assigned_ids: torch.Tensor) -> None:
        assert assigned_ids is not None  # type: ignore

        motion_ids = self._assigned_env_motion_selection[assigned_ids]
        self._motion_buffer_start_time_s[assigned_ids] = (
            self.start_phase[motion_ids] * self.complete_motion_lengths[assigned_ids]
        )
