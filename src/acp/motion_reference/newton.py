from __future__ import annotations

from instinctlab.motion_reference.newton import NewtonMotionReferenceManager

from acp.motion_reference.contact_reference_manager import ContactReferenceManagerBase


class NewtonContactReferenceManager(ContactReferenceManagerBase, NewtonMotionReferenceManager):
    pass
