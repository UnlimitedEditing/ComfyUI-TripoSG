"""
nodes_rigging.py — Rigging and animated sprite-sheet nodes for ComfyUI-TripoSG.

Adds four nodes under the TripoSG/Sprite category:

  AutoRigHumanoid          Fits a 24-joint SMPL-style skeleton to a T-pose
                           TRIMESH and computes per-vertex LBS skin weights.
                           Works without any neural network — just bounding-box
                           proportions calibrated for TripoSG output.

  LoadBVHAnimation         Parses a BVH motion-capture file (URL or local path)
                           into per-frame local rotation matrices.  Supports CMU
                           and Mixamo joint naming out of the box.

  AnimatedTurntableRender  Applies the animation to the rigged mesh frame-by-frame
                           via Linear Blend Skinning and renders every
                           (frame × azimuth) combination with pyrender.
                           Output is a flat IMAGE batch ready for SpriteSheetCompose.

  SpriteSheetCompose       Arranges an IMAGE batch into a rows×cols grid.
                           Wire view_count → cols to get a sheet where each row is
                           one animation frame and each column is a cardinal direction.
"""

from __future__ import annotations

import math
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import trimesh as Trimesh
import torch


# ── SMPL-24 skeleton definition ───────────────────────────────────────────────

JOINT_NAMES: List[str] = [
    "pelvis",         # 0
    "left_hip",       # 1
    "right_hip",      # 2
    "spine1",         # 3
    "left_knee",      # 4
    "right_knee",     # 5
    "spine2",         # 6
    "left_ankle",     # 7
    "right_ankle",    # 8
    "spine3",         # 9
    "left_foot",      # 10
    "right_foot",     # 11
    "neck",           # 12
    "left_collar",    # 13
    "right_collar",   # 14
    "head",           # 15
    "left_shoulder",  # 16
    "right_shoulder", # 17
    "left_elbow",     # 18
    "right_elbow",    # 19
    "left_wrist",     # 20
    "right_wrist",    # 21
    "left_hand",      # 22
    "right_hand",     # 23
]
N_JOINTS = len(JOINT_NAMES)

JOINT_PARENTS: List[int] = [
    -1,  # 0  pelvis  (root)
     0,  # 1  left_hip
     0,  # 2  right_hip
     0,  # 3  spine1
     1,  # 4  left_knee
     2,  # 5  right_knee
     3,  # 6  spine2
     4,  # 7  left_ankle
     5,  # 8  right_ankle
     6,  # 9  spine3
     7,  # 10 left_foot
     8,  # 11 right_foot
     9,  # 12 neck
     9,  # 13 left_collar
     9,  # 14 right_collar
    12,  # 15 head
    13,  # 16 left_shoulder
    14,  # 17 right_shoulder
    16,  # 18 left_elbow
    17,  # 19 right_elbow
    18,  # 20 left_wrist
    19,  # 21 right_wrist
    20,  # 22 left_hand
    21,  # 23 right_hand
]

# Template joint positions as (x, y, z) fractions of the mesh bounding box.
#   X: 0 = left fingertip  →  1 = right fingertip  (full arm span in T-pose)
#   Y: 0 = bottom of feet  →  1 = top of head
#   Z: 0 = back            →  1 = front  (0.5 = centre depth)
TEMPLATE_FRACS = np.array([
    [0.500, 0.510, 0.500],  # 0  pelvis
    [0.440, 0.490, 0.500],  # 1  left_hip
    [0.560, 0.490, 0.500],  # 2  right_hip
    [0.500, 0.560, 0.500],  # 3  spine1
    [0.440, 0.310, 0.500],  # 4  left_knee
    [0.560, 0.310, 0.500],  # 5  right_knee
    [0.500, 0.625, 0.500],  # 6  spine2
    [0.440, 0.110, 0.520],  # 7  left_ankle
    [0.560, 0.110, 0.520],  # 8  right_ankle
    [0.500, 0.680, 0.500],  # 9  spine3
    [0.440, 0.025, 0.560],  # 10 left_foot
    [0.560, 0.025, 0.560],  # 11 right_foot
    [0.500, 0.845, 0.500],  # 12 neck
    [0.462, 0.810, 0.500],  # 13 left_collar
    [0.538, 0.810, 0.500],  # 14 right_collar
    [0.500, 0.945, 0.500],  # 15 head
    [0.390, 0.795, 0.500],  # 16 left_shoulder
    [0.610, 0.795, 0.500],  # 17 right_shoulder
    [0.250, 0.770, 0.500],  # 18 left_elbow
    [0.750, 0.770, 0.500],  # 19 right_elbow
    [0.110, 0.745, 0.500],  # 20 left_wrist
    [0.890, 0.745, 0.500],  # 21 right_wrist
    [0.040, 0.730, 0.500],  # 22 left_hand
    [0.960, 0.730, 0.500],  # 23 right_hand
], dtype=np.float32)


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class RiggedMesh:
    """A TripoSG TRIMESH with an attached SMPL-24 skeleton and LBS weights."""
    mesh:        Trimesh.Trimesh   # base T-pose mesh (vertex colors preserved)
    joints:      np.ndarray        # (24, 3)  rest-pose joint world positions
    parents:     List[int]         # (24,)    SMPL parent indices
    weights:     np.ndarray        # (N_verts, 24)  normalised LBS weights
    joint_names: List[str] = field(default_factory=lambda: list(JOINT_NAMES))


@dataclass
class BVHAnimation:
    """Per-frame rotation matrices parsed from a BVH motion-capture file."""
    joint_names:       List[str]    # joint names in BVH hierarchy order
    frame_time:        float        # seconds per frame
    local_rotations:   np.ndarray   # (N_frames, N_bvh_joints, 3, 3)
    root_translations: np.ndarray   # (N_frames, 3)  root world translation

    @property
    def n_frames(self) -> int:
        return self.local_rotations.shape[0]

    @property
    def duration(self) -> float:
        return self.n_frames * self.frame_time


# ── Rotation / kinematics math ────────────────────────────────────────────────

def _euler_to_rot(rx: float, ry: float, rz: float, order: str) -> np.ndarray:
    """Euler angles (degrees) → 3×3 rotation matrix.

    order is the channel sequence from the BVH file, e.g. 'ZXY' means
    R = Rz @ Rx @ Ry (each successive rotation in local frame).
    """
    def Rx(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float64)
    def Ry(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float64)
    def Rz(a):
        c, s = math.cos(a), math.sin(a)
        return np.array([[c,-s,0],[s,c,0],[0,0,1]], dtype=np.float64)

    mats = {
        "X": Rx(math.radians(rx)),
        "Y": Ry(math.radians(ry)),
        "Z": Rz(math.radians(rz)),
    }
    R = np.eye(3, dtype=np.float64)
    for ch in order:
        R = R @ mats[ch]
    return R


def _forward_kinematics(
    joints_rest:      np.ndarray,   # (J, 3)
    parents:          List[int],
    local_rotations:  np.ndarray,   # (J, 3, 3)
    root_translation: np.ndarray,   # (3,)
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute global rotation matrices and world joint positions.

    Returns
    -------
    global_R : (J, 3, 3)
    global_T : (J, 3)   world position of each joint
    """
    J = len(parents)
    global_R = np.zeros((J, 3, 3), dtype=np.float64)
    global_T = np.zeros((J, 3),    dtype=np.float64)

    for j in range(J):
        p = parents[j]
        if p == -1:
            global_R[j] = local_rotations[j]
            global_T[j] = joints_rest[j] + root_translation
        else:
            global_R[j] = global_R[p] @ local_rotations[j]
            offset       = joints_rest[j] - joints_rest[p]
            global_T[j]  = global_T[p] + global_R[p] @ offset

    return global_R, global_T


def _lbs(
    verts:       np.ndarray,   # (N, 3) rest-pose vertices
    joints_rest: np.ndarray,   # (J, 3)
    weights:     np.ndarray,   # (N, J)
    global_R:    np.ndarray,   # (J, 3, 3)
    global_T:    np.ndarray,   # (J, 3)
) -> np.ndarray:
    """Vectorised Linear Blend Skinning.  Returns (N, 3) posed vertices."""
    # v_rel[j,n] = vertex n relative to joint j's rest position
    v_rel  = verts[None] - joints_rest[:, None]           # (J, N, 3)
    v_rot  = np.einsum("jkl,jnl->jnk", global_R, v_rel)  # (J, N, 3)
    v_rot += global_T[:, None]                            # add world pos
    return np.einsum("nj,jnk->nk", weights, v_rot)        # (N, 3)


# ── BVH parser ────────────────────────────────────────────────────────────────

def _parse_bvh(text: str) -> BVHAnimation:
    """Parse a BVH file string into a BVHAnimation.

    Handles CMU-style (Zrotation Xrotation Yrotation) and Mixamo exports.
    """
    lines = text.splitlines()
    joint_names:    List[str]              = []
    joint_channels: List[Tuple[str, ...]]  = []  # channel names per joint

    i = 0
    # ── HIERARCHY section ─────────────────────────────────────────────────────
    while i < len(lines):
        tok = lines[i].split()
        if not tok:
            i += 1; continue

        if tok[0] in ("ROOT", "JOINT"):
            joint_names.append(tok[1])
            joint_channels.append(())   # placeholder

        elif tok[0] == "CHANNELS":
            n_ch    = int(tok[1])
            ch_list = tuple(tok[2 : 2 + n_ch])
            # Fill the most recently added joint
            if joint_channels:
                joint_channels[-1] = ch_list

        elif tok[0] == "MOTION":
            i += 1
            break
        i += 1

    # ── MOTION section ────────────────────────────────────────────────────────
    n_frames   = 0
    frame_time = 1.0 / 30.0
    frames_data: List[List[float]] = []

    while i < len(lines):
        tok = lines[i].split()
        if not tok:
            i += 1; continue

        if tok[0] == "Frames:":
            n_frames = int(tok[1])
        elif tok[0] == "Frame" and len(tok) > 2 and tok[1] == "Time:":
            frame_time = float(tok[2])
        else:
            try:
                vals = [float(v) for v in tok]
                if vals:
                    frames_data.append(vals)
            except ValueError:
                pass
        i += 1

    n_bvh_joints = len(joint_names)
    # Pad channel list if parse was incomplete
    while len(joint_channels) < n_bvh_joints:
        joint_channels.append(())

    local_rots  = np.zeros((len(frames_data), n_bvh_joints, 3, 3), dtype=np.float32)
    root_trans  = np.zeros((len(frames_data), 3),                   dtype=np.float32)

    # Initialise all frames to identity
    for j in range(n_bvh_joints):
        local_rots[:, j] = np.eye(3)

    for fi, frame_vals in enumerate(frames_data):
        ch_cursor = 0
        for ji, ch_names in enumerate(joint_channels):
            rx = ry = rz = 0.0
            tx = ty = tz = 0.0
            rot_order: List[str] = []

            for ch in ch_names:
                v = frame_vals[ch_cursor] if ch_cursor < len(frame_vals) else 0.0
                ch_cursor += 1
                if   ch == "Xposition": tx = v
                elif ch == "Yposition": ty = v
                elif ch == "Zposition": tz = v
                elif ch == "Xrotation": rx = v; rot_order.append("X")
                elif ch == "Yrotation": ry = v; rot_order.append("Y")
                elif ch == "Zrotation": rz = v; rot_order.append("Z")

            order_str = "".join(rot_order) or "ZXY"
            local_rots[fi, ji] = _euler_to_rot(rx, ry, rz, order_str).astype(np.float32)

            if ji == 0:
                root_trans[fi] = [tx, ty, tz]

    return BVHAnimation(
        joint_names       = joint_names,
        frame_time        = frame_time,
        local_rotations   = local_rots,
        root_translations = root_trans,
    )


# ── BVH → SMPL joint name map ─────────────────────────────────────────────────

# Maps common BVH joint names → SMPL-24 joint names.
# Covers CMU motion capture and Mixamo FBX-to-BVH exports.
BVH_TO_SMPL: dict = {
    # CMU / standard
    "Hips":             "pelvis",
    "LeftUpLeg":        "left_hip",
    "RightUpLeg":       "right_hip",
    "Spine":            "spine1",
    "LeftLeg":          "left_knee",
    "RightLeg":         "right_knee",
    "Spine1":           "spine2",
    "LeftFoot":         "left_ankle",
    "RightFoot":        "right_ankle",
    "Spine2":           "spine3",
    "LeftToeBase":      "left_foot",
    "RightToeBase":     "right_foot",
    "Neck":             "neck",
    "LeftShoulder":     "left_collar",
    "RightShoulder":    "right_collar",
    "Head":             "head",
    "LeftArm":          "left_shoulder",
    "RightArm":         "right_shoulder",
    "LeftForeArm":      "left_elbow",
    "RightForeArm":     "right_elbow",
    "LeftHand":         "left_wrist",
    "RightHand":        "right_wrist",
    # Mixamo prefix variants
    "mixamorig:Hips":           "pelvis",
    "mixamorig:LeftUpLeg":      "left_hip",
    "mixamorig:RightUpLeg":     "right_hip",
    "mixamorig:Spine":          "spine1",
    "mixamorig:LeftLeg":        "left_knee",
    "mixamorig:RightLeg":       "right_knee",
    "mixamorig:Spine1":         "spine2",
    "mixamorig:LeftFoot":       "left_ankle",
    "mixamorig:RightFoot":      "right_ankle",
    "mixamorig:Spine2":         "spine3",
    "mixamorig:LeftToeBase":    "left_foot",
    "mixamorig:RightToeBase":   "right_foot",
    "mixamorig:Neck":           "neck",
    "mixamorig:LeftShoulder":   "left_collar",
    "mixamorig:RightShoulder":  "right_collar",
    "mixamorig:Head":           "head",
    "mixamorig:LeftArm":        "left_shoulder",
    "mixamorig:RightArm":       "right_shoulder",
    "mixamorig:LeftForeArm":    "left_elbow",
    "mixamorig:RightForeArm":   "right_elbow",
    "mixamorig:LeftHand":       "left_wrist",
    "mixamorig:RightHand":      "right_wrist",
}


def _build_bvh_to_rig_map(bvh_joint_names: List[str]) -> np.ndarray:
    """Returns (N_bvh,) int32 array mapping BVH joint index → SMPL-24 index.
    -1 means no match (joint will be ignored during retargeting)."""
    smpl_idx = {name: i for i, name in enumerate(JOINT_NAMES)}
    out = np.full(len(bvh_joint_names), -1, dtype=np.int32)
    for bi, bname in enumerate(bvh_joint_names):
        smpl_name = BVH_TO_SMPL.get(bname)
        if smpl_name is None:
            # Last-ditch: strip namespace prefix and lower-case
            smpl_name = bname.split(":")[-1]
        if smpl_name in smpl_idx:
            out[bi] = smpl_idx[smpl_name]
    n_matched = int((out >= 0).sum())
    print(f"[BVH→SMPL] Matched {n_matched}/{len(bvh_joint_names)} BVH joints to SMPL-24 skeleton")
    return out


# ── LBS weight computation ────────────────────────────────────────────────────

def _compute_lbs_weights(
    verts:   np.ndarray,   # (N, 3)
    joints:  np.ndarray,   # (J, 3)
    parents: List[int],
    falloff: float = 2.5,
) -> np.ndarray:
    """Per-vertex LBS weights via bone-segment inverse-distance.

    For each bone (parent→child segment), compute the closest-point distance
    from every vertex to the segment, then weight by exp(-falloff * dist²).
    Normalises so weights sum to 1 per vertex.
    """
    N, J = len(verts), len(joints)
    W = np.zeros((N, J), dtype=np.float32)

    for j in range(J):
        p = parents[j]
        if p == -1:
            # Root joint: use point distance
            d2 = np.sum((verts - joints[j]) ** 2, axis=1)
        else:
            # Bone segment: parent → child
            a, b  = joints[p].astype(np.float64), joints[j].astype(np.float64)
            ab    = b - a
            ab_l2 = float(np.dot(ab, ab))
            if ab_l2 < 1e-8:
                d2 = np.sum((verts - a) ** 2, axis=1)
            else:
                t       = np.clip(((verts - a) @ ab) / ab_l2, 0.0, 1.0)   # (N,)
                closest = a + t[:, None] * ab                               # (N, 3)
                d2      = np.sum((verts - closest) ** 2, axis=1)

        W[:, j] = np.exp(-falloff * d2)

    W_sum = np.maximum(W.sum(axis=1, keepdims=True), 1e-8)
    return W / W_sum


# ── pyrender scene helpers ────────────────────────────────────────────────────

def _lookat(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    """Build a 4×4 camera pose matrix from eye position, target, and up vector."""
    z = eye - target
    z_n = np.linalg.norm(z)
    z = z / z_n if z_n > 1e-8 else np.array([0, 0, 1.0])
    x = np.cross(up, z)
    x_n = np.linalg.norm(x)
    x = x / x_n if x_n > 1e-8 else np.array([1, 0, 0.0])
    y = np.cross(z, x)
    pose = np.eye(4)
    pose[:3, 0] = x
    pose[:3, 1] = y
    pose[:3, 2] = z
    pose[:3, 3] = eye
    return pose


def _camera_pose(azimuth_deg: float, elevation_deg: float, dist: float) -> np.ndarray:
    az  = math.radians(azimuth_deg)
    el  = math.radians(elevation_deg)
    eye = dist * np.array([
        math.cos(el) * math.sin(az),
        math.sin(el),
        math.cos(el) * math.cos(az),
    ])
    return _lookat(eye, np.zeros(3), np.array([0.0, 1.0, 0.0]))


def _light_pose(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    return _camera_pose(azimuth_deg, elevation_deg, dist=1.0)


def _render_mesh_multiview(
    mesh:          Trimesh.Trimesh,
    azimuths_deg:  List[float],
    elevation_deg: float,
    cam_dist:      float,
    fov_deg:       float,
    width:         int,
    height:        int,
    key_intensity: float,
    fill_intensity: float,
) -> List[np.ndarray]:
    """Render a trimesh from multiple azimuths.

    Returns a list of (H, W, 3) uint8 numpy arrays.
    pyrender uses EGL on Linux (Graydient) automatically when a display is absent.
    """
    try:
        import pyrender
    except ImportError:
        raise ImportError("pyrender is required — add it to pip requirements")

    scene = pyrender.Scene(
        bg_color       = [0, 0, 0, 0],
        ambient_light  = [0.1, 0.1, 0.1],
    )

    # Mesh
    if (hasattr(mesh, "visual") and
            hasattr(mesh.visual, "vertex_colors") and
            mesh.visual.vertex_colors is not None):
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor = [1.0, 1.0, 1.0, 1.0],
            metallicFactor  = 0.0,
            roughnessFactor = 1.0,
        )
        pr_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)
    else:
        pr_mesh = pyrender.Mesh.from_trimesh(mesh, smooth=True)

    mesh_node = scene.add(pr_mesh)

    # 3-point lighting: key + fill
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=key_intensity),
              pose=_light_pose(45,  60))
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=fill_intensity),
              pose=_light_pose(-60, 30))

    camera   = pyrender.PerspectiveCamera(
        yfov        = math.radians(fov_deg),
        aspectRatio = width / height,
    )
    renderer = pyrender.OffscreenRenderer(width, height)
    frames: List[np.ndarray] = []

    try:
        for az in azimuths_deg:
            cam_node        = scene.add(camera, pose=_camera_pose(az, elevation_deg, cam_dist))
            color, _depth   = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            scene.remove_node(cam_node)
            frames.append(color[:, :, :3])   # drop alpha → (H, W, 3) uint8
    finally:
        renderer.delete()

    scene.remove_node(mesh_node)
    return frames


# ── Node: AutoRigHumanoid ─────────────────────────────────────────────────────

class AutoRigHumanoid:
    """
    Fits a SMPL-24 skeleton to a T-pose humanoid TRIMESH (TripoSG output) and
    computes per-vertex Linear Blend Skinning weights.

    No external model required — joint positions are derived from the mesh
    bounding box using proportions calibrated for TripoSG's T-pose output.
    weight_falloff controls how sharply weights drop off with distance from
    each bone segment; higher values give crisper joints.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "weight_falloff": ("FLOAT", {
                    "default": 2.5,
                    "min": 0.2, "max": 20.0, "step": 0.1,
                    "tooltip": (
                        "Controls how sharply skinning weights fall off with "
                        "distance from each bone segment.  Higher = crisper "
                        "joints.  Lower = smoother cross-joint blending."
                    ),
                }),
            }
        }

    RETURN_TYPES  = ("RIGGED_MESH",)
    RETURN_NAMES  = ("rigged_mesh",)
    FUNCTION      = "rig"
    CATEGORY      = "TripoSG/Sprite"
    DESCRIPTION   = (
        "Fits a SMPL-24 humanoid skeleton to a T-pose TRIMESH and computes "
        "LBS skin weights.  Input should be a T-pose humanoid such as those "
        "produced by TripoSGInference."
    )

    def rig(self, trimesh, weight_falloff: float):
        verts = np.array(trimesh.vertices, dtype=np.float32)
        mn, mx = verts.min(axis=0), verts.max(axis=0)
        span   = np.maximum(mx - mn, 1e-6)

        # Place template joints proportionally inside the bounding box
        joints = (mn + TEMPLATE_FRACS * span).astype(np.float32)

        print(f"[AutoRigHumanoid] Mesh: {len(verts):,} verts | "
              f"bbox {span.round(3).tolist()} | computing LBS weights …")
        weights = _compute_lbs_weights(verts, joints, JOINT_PARENTS, falloff=weight_falloff)
        print("[AutoRigHumanoid] Done.")

        return (RiggedMesh(
            mesh        = trimesh,
            joints      = joints,
            parents     = JOINT_PARENTS,
            weights     = weights,
        ),)


# ── Node: LoadBVHAnimation ────────────────────────────────────────────────────

class LoadBVHAnimation:
    """
    Loads a BVH motion-capture file from a URL or local path and returns a
    BVH_ANIMATION containing per-frame local rotation matrices.

    Supports CMU motion capture joint names and Mixamo BVH exports.
    Use max_frames and frame_skip to trim or down-sample long clips before
    passing to AnimatedTurntableRender.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url_or_path": ("STRING", {
                    "default": "",
                    "tooltip": "HTTPS URL or absolute local path to a .bvh file",
                }),
                "max_frames": ("INT", {
                    "default": 0,
                    "min": 0, "max": 10000,
                    "tooltip": "Trim to this many frames after skipping.  0 = keep all.",
                }),
                "frame_skip": ("INT", {
                    "default": 1,
                    "min": 1, "max": 32,
                    "tooltip": (
                        "Use every Nth frame (1 = all, 2 = half rate, etc.). "
                        "Useful for long CMU clips that run at 120 fps."
                    ),
                }),
            }
        }

    RETURN_TYPES  = ("BVH_ANIMATION",)
    RETURN_NAMES  = ("animation",)
    FUNCTION      = "load"
    CATEGORY      = "TripoSG/Sprite"
    DESCRIPTION   = "Parses a BVH motion-capture file into per-frame rotation matrices."

    def load(self, url_or_path: str, max_frames: int, frame_skip: int):
        src = url_or_path.strip()
        if not src:
            raise ValueError("[LoadBVHAnimation] url_or_path is empty")

        if src.startswith("http://") or src.startswith("https://"):
            import requests
            resp = requests.get(src, timeout=30)
            resp.raise_for_status()
            text = resp.text
        else:
            text = Path(src).read_text(encoding="utf-8", errors="replace")

        anim = _parse_bvh(text)
        print(f"[LoadBVHAnimation] Parsed: {anim.n_frames} frames @ "
              f"{1.0/anim.frame_time:.1f} fps, {len(anim.joint_names)} joints")

        # Down-sample and trim
        indices = list(range(0, anim.n_frames, frame_skip))
        if max_frames > 0:
            indices = indices[:max_frames]

        anim.local_rotations   = anim.local_rotations[indices]
        anim.root_translations = anim.root_translations[indices]
        print(f"[LoadBVHAnimation] Using {anim.n_frames} frames "
              f"({anim.duration:.2f}s after skip={frame_skip})")

        return (anim,)


# ── Node: AnimatedTurntableRender ─────────────────────────────────────────────

class AnimatedTurntableRender:
    """
    Applies a BVH animation to a rigged mesh via Linear Blend Skinning and
    renders every (animation frame × azimuth angle) combination with pyrender.

    Output IMAGE batch layout  (index = frame * num_views + view):
        frame 0 | az 0 … az N-1
        frame 1 | az 0 … az N-1
        …

    Wire view_count → SpriteSheetCompose.cols to assemble into a standard
    sprite sheet where each row is one animation frame and each column is a
    cardinal direction.

    root_scale = 0 locks the character to the origin (recommended for sprites).
    Set to 1 if you want the BVH root translation applied at full scale.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "rigged_mesh":    ("RIGGED_MESH",),
                "animation":      ("BVH_ANIMATION",),
                "num_views":      ("INT",   {"default": 8,    "min": 1,     "max": 64}),
                "start_azimuth":  ("FLOAT", {"default": 0.0,  "min": -180.0,"max": 180.0,"step": 1.0}),
                "elevation_deg":  ("FLOAT", {"default": 15.0, "min": -90.0, "max": 90.0, "step": 1.0}),
                "cam_dist":       ("FLOAT", {"default": 3.2,  "min": 0.5,   "max": 20.0, "step": 0.1}),
                "fov_deg":        ("FLOAT", {"default": 24.0, "min": 5.0,   "max": 120.0,"step": 1.0}),
                "frame_w":        ("INT",   {"default": 128,  "min": 16,    "max": 2048}),
                "frame_h":        ("INT",   {"default": 192,  "min": 16,    "max": 2048}),
                "key_intensity":  ("FLOAT", {"default": 3.0,  "min": 0.0,   "max": 20.0, "step": 0.1}),
                "fill_intensity": ("FLOAT", {"default": 1.2,  "min": 0.0,   "max": 20.0, "step": 0.1}),
                "root_scale":     ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0, "max": 10.0, "step": 0.01,
                    "tooltip": (
                        "Scale applied to BVH root translation.  "
                        "0 = keep character centred (best for sprites).  "
                        "1 = use BVH translation at full scale."
                    ),
                }),
            }
        }

    RETURN_TYPES  = ("IMAGE", "INT", "INT")
    RETURN_NAMES  = ("frames", "anim_frame_count", "view_count")
    FUNCTION      = "render"
    CATEGORY      = "TripoSG/Sprite"
    DESCRIPTION   = (
        "Renders a rigged mesh for every animation frame × azimuth angle. "
        "Output is a flat IMAGE batch; feed into SpriteSheetCompose with "
        "cols=view_count to assemble the final sprite sheet."
    )

    def render(
        self,
        rigged_mesh:    RiggedMesh,
        animation:      BVHAnimation,
        num_views:      int,
        start_azimuth:  float,
        elevation_deg:  float,
        cam_dist:       float,
        fov_deg:        float,
        frame_w:        int,
        frame_h:        int,
        key_intensity:  float,
        fill_intensity: float,
        root_scale:     float,
    ):
        azimuths = [start_azimuth + i * (360.0 / num_views) for i in range(num_views)]

        # Build BVH → SMPL joint index map
        bvh_to_smpl = _build_bvh_to_rig_map(animation.joint_names)

        verts_rest  = np.array(rigged_mesh.mesh.vertices, dtype=np.float64)
        joints_rest = rigged_mesh.joints.astype(np.float64)
        parents     = rigged_mesh.parents
        weights     = rigged_mesh.weights.astype(np.float64)
        base_colors = (
            rigged_mesh.mesh.visual.vertex_colors
            if (hasattr(rigged_mesh.mesh, "visual") and
                hasattr(rigged_mesh.mesh.visual, "vertex_colors"))
            else None
        )
        faces = rigged_mesh.mesh.faces.copy()

        n_frames     = animation.n_frames
        total_renders = n_frames * num_views
        print(f"[AnimatedTurntableRender] {n_frames} frames × {num_views} views "
              f"= {total_renders} renders at {frame_w}×{frame_h}")

        try:
            import comfy.utils
            pbar = comfy.utils.ProgressBar(n_frames)
        except Exception:
            pbar = None

        all_frames: List[np.ndarray] = []

        for fi in range(n_frames):
            # Build SMPL-order local rotations for this frame
            local_R = np.stack([np.eye(3, dtype=np.float64)] * N_JOINTS)
            for bi, smpl_idx in enumerate(bvh_to_smpl):
                if smpl_idx >= 0:
                    local_R[smpl_idx] = animation.local_rotations[fi, bi].astype(np.float64)

            root_t = animation.root_translations[fi].astype(np.float64) * root_scale

            global_R, global_T = _forward_kinematics(joints_rest, parents, local_R, root_t)
            verts_posed         = _lbs(verts_rest, joints_rest, weights, global_R, global_T)

            posed_mesh = Trimesh.Trimesh(
                vertices     = verts_posed,
                faces        = faces,
                vertex_colors= base_colors,
                process      = False,
            )

            renders = _render_mesh_multiview(
                posed_mesh, azimuths, elevation_deg,
                cam_dist, fov_deg, frame_w, frame_h,
                key_intensity, fill_intensity,
            )
            all_frames.extend(renders)

            if pbar:
                pbar.update(1)

        # Stack → [N_total, H, W, 3] float32 0–1
        arr    = np.stack(all_frames, axis=0).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr)
        print(f"[AnimatedTurntableRender] Done — tensor shape {list(tensor.shape)}")

        return (tensor, n_frames, num_views)


# ── Node: SpriteSheetCompose ──────────────────────────────────────────────────

class SpriteSheetCompose:
    """
    Arranges an IMAGE batch into a rows × cols grid sprite sheet.

    Default usage: set cols = view_count from AnimatedTurntableRender.
    Each row becomes one animation frame seen from all directions.
    padding adds black pixels between cells.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames":  ("IMAGE",),
                "cols":    ("INT", {
                    "default": 8,
                    "min": 1, "max": 256,
                    "tooltip": "Frames per row.  Set to view_count for a directions×frames sheet.",
                }),
                "padding": ("INT", {
                    "default": 0,
                    "min": 0, "max": 64,
                    "tooltip": "Pixels of black padding between sprite cells.",
                }),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("sprite_sheet",)
    FUNCTION      = "compose"
    CATEGORY      = "TripoSG/Sprite"
    DESCRIPTION   = "Arranges an IMAGE batch into a rows×cols sprite sheet grid."

    def compose(self, frames: torch.Tensor, cols: int, padding: int):
        N, H, W, C = frames.shape
        rows = math.ceil(N / cols)

        sheet_h = rows * H + max(0, rows - 1) * padding
        sheet_w = cols * W + max(0, cols - 1) * padding
        sheet   = torch.zeros((1, sheet_h, sheet_w, C), dtype=frames.dtype)

        for i in range(N):
            r  = i // cols
            c  = i  % cols
            y0 = r * (H + padding)
            x0 = c * (W + padding)
            sheet[0, y0:y0 + H, x0:x0 + W, :] = frames[i]

        print(f"[SpriteSheetCompose] {N} frames → {cols}×{rows} grid "
              f"({sheet_w}×{sheet_h}px)")
        return (sheet,)


# ── Registration ──────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "AutoRigHumanoid":         AutoRigHumanoid,
    "LoadBVHAnimation":        LoadBVHAnimation,
    "AnimatedTurntableRender": AnimatedTurntableRender,
    "SpriteSheetCompose":      SpriteSheetCompose,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AutoRigHumanoid":         "Auto Rig Humanoid (SMPL-24)",
    "LoadBVHAnimation":        "Load BVH Animation",
    "AnimatedTurntableRender": "Animated Turntable Render",
    "SpriteSheetCompose":      "Sprite Sheet Compose",
}
