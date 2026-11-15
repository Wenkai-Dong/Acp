from __future__ import annotations

from instinctlab.motion_reference.physx import PhysxMotionReferenceManager

from acp.motion_reference.contact_reference_manager import ContactReferenceManagerBase


class PhysxContactReferenceManager(ContactReferenceManagerBase, PhysxMotionReferenceManager):
    pass
