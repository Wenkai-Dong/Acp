from __future__ import annotations

from instinctlab.motion_reference.ovphysx import OvPhysxMotionReferenceManager

from acp.motion_reference.contact_reference_manager import ContactReferenceManagerBase


class OvPhysxContactReferenceManager(ContactReferenceManagerBase, OvPhysxMotionReferenceManager):
    pass
