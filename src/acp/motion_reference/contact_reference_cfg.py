from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.sensors import SensorBaseCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.configclass import configclass

from instinctlab.motion_reference.motion_reference_cfg import MotionReferenceManagerCfg


@configclass
class ContactReferenceManagerCfg(MotionReferenceManagerCfg):

    link_of_contacts: Sequence[str] | None = None

    class_type: type | str = "{DIR}.contact_reference_manager:ContactReferenceManager"

    data_class_type: type | str = "{DIR}.contact_reference_data:ContactReferenceData"

    contact_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/ContactReference",
        markers={
            "link_contact_pos_w": sim_utils.SphereCfg(
                radius=0.05,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        },
    )

    contact_visualizing_marker_types: list[str] = []
    """List of marker types to contact visualize.
    ## Available options:
        - 'link_contact_pos_w' for link contact pos world
    """

