import os
import sys
import traceback

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from .nodes import (
    NODE_CLASS_MAPPINGS        as _MAPS,
    NODE_DISPLAY_NAME_MAPPINGS as _DMAPS,
)

try:
    from .nodes_rigging import (
        NODE_CLASS_MAPPINGS        as _RMAPS,
        NODE_DISPLAY_NAME_MAPPINGS as _RDMAPS,
    )
    print(f"[ComfyUI-TripoSG] nodes_rigging loaded — {len(_RMAPS)} sprite nodes registered")
except Exception as _e:
    print(f"[ComfyUI-TripoSG] WARNING: nodes_rigging failed to load: {_e}")
    traceback.print_exc()
    _RMAPS  = {}
    _RDMAPS = {}

NODE_CLASS_MAPPINGS        = {**_MAPS,  **_RMAPS}
NODE_DISPLAY_NAME_MAPPINGS = {**_DMAPS, **_RDMAPS}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
