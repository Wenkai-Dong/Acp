"""The implementation for reading and returning motion reference data.

FAQ:
    - How to use this module in AMP/WASABI implementation?
        * Use the motion reference to build discriminator observation and feed to the algorithm
        * Use the discriminator observation and discriminator (in algo) to label the discrimintaor
        reward.
"""

from .contact_reference_cfg import ContactReferenceManagerCfg
from .contact_reference_data import ContactReferenceData, ContactSequence

__all__ = [
    "ContactReferenceData",
    "ContactReferenceManagerCfg",
    "ContactSequence",
]
