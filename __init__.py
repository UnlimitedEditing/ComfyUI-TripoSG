import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .nodes import (
    NODE_CLASS_MAPPINGS         as _MAPS,
    NODE_DISPLAY_NAME_MAPPINGS  as _DMAPS,
)
from .nodes_rigging import (
    NODE_CLASS_MAPPINGS         as _RMAPS,
    NODE_DISPLAY_NAME_MAPPINGS  as _RDMAPS,
)

NODE_CLASS_MAPPINGS          = {**_MAPS,  **_RMAPS}
NODE_DISPLAY_NAME_MAPPINGS   = {**_DMAPS, **_RDMAPS}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
