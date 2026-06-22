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

try:
    from .nodes_concept import (
        NODE_CLASS_MAPPINGS        as _CMAPS,
        NODE_DISPLAY_NAME_MAPPINGS as _CDMAPS,
    )
    print(f"[ComfyUI-TripoSG] nodes_concept loaded — {len(_CMAPS)} lyric concept nodes registered")
except Exception as _ce:
    print(f"[ComfyUI-TripoSG] WARNING: nodes_concept failed to load: {_ce}")
    _CMAPS  = {}
    _CDMAPS = {}

NODE_CLASS_MAPPINGS        = {**_MAPS, **_RMAPS, **_CMAPS}
NODE_DISPLAY_NAME_MAPPINGS = {**_DMAPS, **_RDMAPS, **_CDMAPS}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
