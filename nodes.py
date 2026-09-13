import os
import numpy as np
import torch
from io import BytesIO
from pathlib import Path
from PIL import Image
from typing import Dict, Any

import folder_paths
import comfy.utils
import comfy.model_management as mm

gpu = mm.get_torch_device()
cpu = torch.device("cpu")

# Heavy 3D pipeline imports — optional.
# LoadImageFromURL and LoadVideoFromURL work without them.
_TRIPOSG_AVAILABLE = False
try:
    import cv2
    import trimesh as Trimesh
    from huggingface_hub import snapshot_download
    from comfy_extras.nodes_hunyuan3d import MESH
    from .triposg.pipelines.pipeline_triposg import TripoSGPipeline
    from .triposg.pipelines.pipeline_triposg_scribble import TripoSGScribblePipeline
    from .partcrafter.pipelines.pipeline_partcrafter import PartCrafterPipeline
    from .partcrafter.utils.data_utils import (
        get_colored_mesh_composition, scene_to_parts, load_surfaces,
    )
    _TRIPOSG_AVAILABLE = True
except ImportError as e:
    print(
        f"[ComfyUI-TripoSG] 3D pipeline unavailable ({e}); "
        "LoadImageFromURL and LoadVideoFromURL still active."
    )


def pil2numpy(image: Image.Image):
    return np.array(image).astype(np.float32) / 255.0


def numpy2pil(image: np.ndarray, mode=None):
    return Image.fromarray(np.clip(255.0 * image, 0, 255).astype(np.uint8), mode)


def pil2tensor(image: Image.Image):
    return torch.from_numpy(pil2numpy(image)).unsqueeze(0)


def tensor2pil(image: torch.Tensor, mode=None):
    return numpy2pil(image.cpu().numpy().squeeze(), mode=mode)


def simplify_mesh(mesh, n_faces: int):
    # Assume mesh.vertices: (1, N, 3), mesh.faces: (1, M, 3)
    v = mesh.vertices[0].cpu().numpy()
    f = mesh.faces[0].cpu().numpy()

    if f.shape[0] <= n_faces or n_faces == 0:
        # No simplification needed, just return original
        vertices = mesh.vertices
        faces = mesh.faces
    else:
        try:
            import pymeshlab
        except ImportError:
            raise ImportError("pymeshlab is not installed. Please install it with `pip install pymeshlab`.")
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=v, face_matrix=f))
        ms.meshing_merge_close_vertices()
        ms.meshing_decimation_quadric_edge_collapse(targetfacenum=n_faces)
        m = ms.current_mesh()
        vertices = torch.from_numpy(m.vertex_matrix()).float().unsqueeze(0)
        faces = torch.from_numpy(m.face_matrix()).long().unsqueeze(0)
    return MESH(vertices=vertices, faces=faces)


class TripoSGModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (
                    ["VAST-AI/TripoSG", "VAST-AI/TripoSG-scribble", "wgsxm/PartCrafter"],
                    {"default": "VAST-AI/TripoSG"},
                )
            },
            "optional": {
                "model_override": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "When non-empty, overrides the model dropdown above with this repo id. "
                                   "Lets a Graydient slot (e.g. via PartCrafterModelSelect) drive the model "
                                   "choice at runtime without rewiring the graph.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("TRIPOSG",)
    FUNCTION = "load_model"
    CATEGORY = "TripoSG"

    def load_model(self, model, model_override=""):
        if model_override and model_override.strip():
            model = model_override.strip()
        model_name = model.split("/")[-1]
        model_dir = os.path.join(folder_paths.models_dir, "3D", model_name)
        os.makedirs(model_dir, exist_ok=True)
        if not os.path.exists(model_dir) or not os.listdir(model_dir):
            print(f"Downloading {model} to {model_dir}")
            snapshot_download(repo_id=model, local_dir=model_dir, local_dir_use_symlinks=False)

        if model == "VAST-AI/TripoSG":
            pipe = TripoSGPipeline.from_pretrained(model_dir).to(gpu, torch.float16)
        elif model == "VAST-AI/TripoSG-scribble":
            pipe = TripoSGScribblePipeline.from_pretrained(model_dir).to(gpu, torch.float16)
        elif model == "wgsxm/PartCrafter":
            import shutil

            custom_model_index_path = os.path.join(
                os.path.dirname(__file__), "partcrafter", "models", "model_index.json"
            )
            target_model_index_path = os.path.join(model_dir, "model_index.json")
            shutil.copy2(custom_model_index_path, target_model_index_path)

            pipe = PartCrafterPipeline.from_pretrained(model_dir).to(gpu, torch.float16)
        else:
            raise ValueError(f"Unknown model: {model}")

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        return (pipe,)


class TripoSGInference:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("TRIPOSG",),
                "image": ("IMAGE",),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "The random seed used for creating the noise.",
                    },
                ),
                "steps": (
                    "INT",
                    {
                        "default": 50,
                        "min": 1,
                        "max": 10000,
                        "tooltip": "The number of steps used in the denoising process.",
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 7,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "round": 0.01,
                        "tooltip": "The Classifier-Free Guidance scale balances creativity and adherence to the prompt. Higher values result in images more closely matching the prompt however too high values will negatively impact quality.",
                    },
                ),
            },
            "optional": {
                "conditioning": ("TRIPOSG_CONDITIONING",),
            },
        }

    RETURN_TYPES = ("TRIMESH", "TRIMESH")
    RETURN_NAMES = ("trimesh", "parts")
    OUTPUT_IS_LIST = (False, True)
    FUNCTION = "run_inference"
    CATEGORY = "TripoSG"

    def run_inference(
        self,
        model,
        image,
        seed,
        steps,
        cfg,
        conditioning=None,
    ):
        pil_image = tensor2pil(image)

        pipe_class = model.__class__.__name__
        generator = torch.Generator(device=model.device).manual_seed(seed)
        pbar = comfy.utils.ProgressBar(steps + 1)

        def step_callback(pipe, step, t, callback_kwargs):
            pbar.update(1)
            return callback_kwargs

        if pipe_class == "TripoSGPipeline":
            outputs = model(
                image=pil_image,
                generator=generator,
                num_inference_steps=steps,
                guidance_scale=cfg,
                callback_on_step_end=step_callback,
            )
        elif pipe_class == "TripoSGScribblePipeline":
            if not conditioning:
                raise ValueError("TripoSGScribbleConditioning must be provided")

            if not isinstance(conditioning, TripoSGScribbleConditioning):
                raise ValueError("Conditioning must be a TripoSGScribbleConditioning")

            # Empty prompt is allowed — text encoder handles "" as no text guidance

            outputs = model(
                image=pil_image,
                generator=generator,
                num_inference_steps=steps,
                guidance_scale=0,  # CFG-distilled model
                use_flash_decoder=False,
                callback_on_step_end=step_callback,
                **conditioning.to_dict(),
            )
        elif pipe_class == "PartCrafterPipeline":
            if not conditioning:
                raise ValueError("PartCrafterConditioning must be provided")

            if not isinstance(conditioning, PartCrafterConditioning):
                raise ValueError("Conditioning must be a PartCrafterConditioning")

            outputs = model(
                image=[pil_image] * conditioning.attention_kwargs["num_parts"],
                generator=generator,
                num_inference_steps=steps,
                guidance_scale=cfg,
                use_flash_decoder=False,
                callback_on_step_end=step_callback,
                **conditioning.to_dict(),
            )
        else:
            raise ValueError(f"Unknown pipeline type: {pipe_class}")

        parts = [m for m in outputs.meshes if m is not None]

        if len(parts) == 1:
            mesh = parts[0]
        else:
            mesh = get_colored_mesh_composition(parts)

        return (mesh, parts)


class SimplifyMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESH",),
                "faces": (
                    "INT",
                    {
                        "min": 0.0,
                        "max": 0xFFFFFFFFFFFFFFF,
                        "step": 1,
                        "default": 0,
                        "tooltip": "The number of faces to simplify the mesh to. 0 means no simplification.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("MESH",)
    FUNCTION = "simplify_mesh"
    CATEGORY = "TripoSG"

    def simplify_mesh(self, mesh, faces):
        if faces == 0 or faces > mesh.faces.shape[0]:
            return (mesh,)

        return (simplify_mesh(mesh, faces),)


class TripoSGPrepareImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "prepare"
    CATEGORY = "TripoSG"

    def prepare(self, image, mask=None):
        # image: [1, H, W, C] or [H, W, C], float32, 0-1
        # mask: [1, H, W] or [H, W], float32, 0-1 or 0-255
        if image.ndim == 4:
            image = image[0]
        if image.ndim != 3:
            raise ValueError(f"Image tensor must be [H, W, C], got {image.shape}")
        H, W, C = image.shape
        image_np = (image.cpu().numpy() * 255).astype(np.uint8)
        alpha = None

        # Handle channels
        if C == 1:
            rgb_image = np.repeat(image_np, 3, axis=2)  # HWC
        elif C == 3:
            rgb_image = image_np  # HWC
        elif C == 4:
            rgb_image = image_np[:, :, :3]  # HWC
            alpha = image_np[:, :, 3]
        else:
            raise ValueError(f"Unsupported channel count: {C}")

        # Resize if too large
        H, W = rgb_image.shape[:2]
        max_side = max(H, W)
        if max_side > 2000:
            scale = 2000 / max_side
            new_H, new_W = int(H * scale), int(W * scale)
            rgb_image = cv2.resize(rgb_image, (new_W, new_H), interpolation=cv2.INTER_AREA)
            if alpha is not None:
                alpha = cv2.resize(alpha, (new_W, new_H), interpolation=cv2.INTER_NEAREST)
            H, W = new_H, new_W

        # Alpha validation
        def is_valid_alpha(alpha, min_ratio=0.01):
            hist = cv2.calcHist([alpha], [0], None, [20], [0, 256])
            min_hist_val = alpha.shape[0] * alpha.shape[1] * min_ratio
            return hist[0] >= min_hist_val and hist[-1] >= min_hist_val

        if alpha is not None and not is_valid_alpha(alpha):
            alpha = None

        if alpha is None and mask is None:
            # Auto-remove white background for images without alpha
            white_mask = np.all(image_np >= 250, axis=2)
            alpha = np.where(white_mask, 0, 255).astype(np.uint8)
            if not is_valid_alpha(alpha):
                # No valid subject found (e.g. blank canvas for text-only scribble).
                # Fall back to treating the entire image as the subject.
                alpha = np.full(image_np.shape[:2], 255, dtype=np.uint8)

        if alpha is None:
            if mask.ndim == 3:
                mask = mask[0]
            if mask.shape != (H, W):
                raise ValueError(f"Mask shape {mask.shape} does not match image shape {(H, W)}")
            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
            alpha = mask_np

        # Find bounding box
        if np.any(alpha > 0):
            x, y, w, h = self.find_bounding_box(alpha)
        else:
            raise ValueError("input image too small or empty mask")

        # Compose with white background
        alpha_f = alpha.astype(np.float32) / 255.0
        rgb_f = rgb_image.astype(np.float32) / 255.0
        bg_color = np.ones(3, dtype=np.float32)  # [1,1,1]
        out_rgb = rgb_f * alpha_f[..., None] + bg_color * (1 - alpha_f[..., None])

        # Crop to bbox
        cropped = out_rgb[y : y + h, x : x + w, :]

        # Dynamic padding based on aspect ratio
        pad_ratio = 0.1
        if w > h:
            pad_h = int(w * pad_ratio)
            pad_w = int(w * pad_ratio)
            size = w + 2 * pad_w
            y_off = int(pad_h + (w - h) / 2)
            x_off = pad_w
        else:
            pad_h = int(h * pad_ratio)
            pad_w = int(h * pad_ratio)
            size = h + 2 * pad_h
            y_off = pad_h
            x_off = int(pad_w + (h - w) / 2)
        
        padded = np.ones((size, size, 3), dtype=np.float32)
        padded[y_off : y_off + h, x_off : x_off + w, :] = cropped

        # To tensor [1, H, W, 3]
        tensor = torch.from_numpy(padded).unsqueeze(0).contiguous().float()
        return (tensor,)

    @staticmethod
    def find_bounding_box(gray_image):
        # gray_image: HxW uint8
        _, binary_image = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return 0, 0, gray_image.shape[1], gray_image.shape[0]
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        return x, y, w, h


class BaseConditioning:
    def get(self, key: str, default=None):
        return getattr(self, key, default)

    def __getitem__(self, key: str):
        return getattr(self, key)

    def __setitem__(self, key: str, value):
        setattr(self, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the conditioning object to a dictionary."""
        return {key: value for key, value in self.__dict__.items()}


class TripoSGScribbleConditioning(BaseConditioning):
    def __init__(self, prompt: str, attention_kwargs: Dict[str, Any]):
        self.prompt = prompt
        self.attention_kwargs = attention_kwargs


class TripoSGScribbleConditioningNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {"multiline": True}),
                "prompt_confidence": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "scribble_confidence": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 10.0, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("TRIPOSG_CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "conditioning"
    CATEGORY = "TripoSG"

    def conditioning(self, prompt, prompt_confidence, scribble_confidence):
        return (
            TripoSGScribbleConditioning(
                prompt=prompt,
                attention_kwargs={
                    "cross_attention_scale": prompt_confidence,
                    "cross_attention_2_scale": scribble_confidence,
                },
            ),
        )


class PartCrafterConditioning(BaseConditioning):
    def __init__(self, num_tokens: int, max_num_expanded_coords: int, attention_kwargs: Dict[str, Any]):
        self.num_tokens = num_tokens
        self.max_num_expanded_coords = max_num_expanded_coords
        self.attention_kwargs = attention_kwargs


class PartCrafterConditioningNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "num_parts": ("INT", {"default": 1, "min": 1, "max": 100}),
                "num_tokens": ("INT", {"default": 1024, "min": 1, "max": 4096}),
                "max_num_expanded_coords": ("INT", {"default": 1e8, "min": 1, "max": 1e10}),
            },
        }

    RETURN_TYPES = ("TRIPOSG_CONDITIONING",)
    RETURN_NAMES = ("conditioning",)
    FUNCTION = "conditioning"
    CATEGORY = "TripoSG"

    def conditioning(self, num_parts, num_tokens, max_num_expanded_coords):
        return (
            PartCrafterConditioning(
                num_tokens=num_tokens,
                max_num_expanded_coords=max_num_expanded_coords,
                attention_kwargs={"num_parts": num_parts},
            ),
        )


class PartCrafterModelSelect:
    """Turns a plain Graydient slot INT (0/1) into the TripoSG model repo id.

    Wire its STRING output into TripoSGModelLoader's optional `model_override`
    input. 0 -> VAST-AI/TripoSG (default), 1 -> wgsxm/PartCrafter. Mirrors
    Meshsmuggler's MeshSmuggleGate: a plain-value toggle instead of a graph
    rewire, since Graydient can only patch literal field values.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "enable_partcrafter": ("INT", {"default": 0, "min": 0, "max": 1, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("model",)
    FUNCTION = "select"
    CATEGORY = "TripoSG"

    def select(self, enable_partcrafter):
        return ("wgsxm/PartCrafter" if int(enable_partcrafter) != 0 else "VAST-AI/TripoSG",)


class SimplifyMeshFacesSelect:
    """Turns a plain Graydient slot INT (0/1) into a target face count for SimplifyMesh.

    0 -> 0 (SimplifyMesh no-ops, mesh passes through untouched).
    1 -> target_faces (decimate to this count). Same slot-toggle pattern as
    PartCrafterModelSelect / MeshSmuggleGate.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "enable_simplify": ("INT", {"default": 0, "min": 0, "max": 1, "step": 1}),
                "target_faces": ("INT", {"default": 50000, "min": 1, "max": 0xFFFFFFF, "step": 1}),
            },
        }

    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("faces",)
    FUNCTION = "select"
    CATEGORY = "TripoSG"

    def select(self, enable_simplify, target_faces):
        return (target_faces if int(enable_simplify) != 0 else 0,)


class TrimeshToMESH:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            }
        }

    RETURN_TYPES = ("MESH",)
    OUTPUT_TOOLTIPS = ("MESH object containing vertices and faces as torch tensors.",)

    FUNCTION = "load"
    CATEGORY = "TripoSG"
    DESCRIPTION = "Converts trimesh object to ComfyUI MESH object, which only includes mesh data"

    def load(self, trimesh):
        vertices = torch.tensor(trimesh.vertices, dtype=torch.float32)
        faces = torch.tensor(trimesh.faces, dtype=torch.float32)
        mesh = MESH(vertices.unsqueeze(0), faces.unsqueeze(0))

        return (mesh,)


class MESHToTrimesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESH",),
            }
        }

    RETURN_TYPES = ("TRIMESH",)
    OUTPUT_TOOLTIPS = ("TRIMESH object containing vertices and faces as torch tensors.",)

    FUNCTION = "load"
    CATEGORY = "TripoSG"
    DESCRIPTION = "Converts trimesh object to ComfyUI MESH object, which only includes mesh data"

    def load(self, mesh):
        mesh_output = Trimesh.Trimesh(mesh.vertices[0], mesh.faces[0])
        return (mesh_output,)


class SaveTrimesh:
    _FORMATS = ["glb", "obj", "ply", "stl", "3mf", "dae"]

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "filename_prefix": ("STRING", {"default": "3D/TripoSG"}),
                "file_format": (SaveTrimesh._FORMATS,),
            },
            "optional": {
                "also_save": (["none"] + SaveTrimesh._FORMATS, {"default": "none"}),
                "save_file": ("BOOLEAN", {"default": True, "label_on": "output", "label_off": "temp"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("file_path", "also_path")
    FUNCTION = "process"
    CATEGORY = "TripoSG"
    OUTPUT_NODE = True
    DESCRIPTION = "Export trimesh object to one or two model files simultaneously"

    def process(self, trimesh, filename_prefix, file_format, also_save="none", save_file=True):
        save_dir = folder_paths.get_output_directory() if save_file else folder_paths.get_temp_directory()
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, save_dir
        )

        def write(fmt):
            path = Path(full_output_folder, f"{filename}_{counter:05}_.{fmt}")
            path.parent.mkdir(parents=True, exist_ok=True)
            trimesh.export(str(path), file_type=fmt)
            print(f"[SaveTrimesh] wrote {fmt.upper()} → {path} ({path.stat().st_size} bytes)")
            return str(Path(subfolder) / f"{filename}_{counter:05}_.{fmt}")

        primary_path = write(file_format)
        also_path = write(also_save) if also_save != "none" and also_save != file_format else ""

        return (primary_path, also_path)


class BakeVertexColorsFromViews:
    """
    Orthographic vertex-colour bake from front + optional back view images.
    Samples pixel colours per vertex using X/Y projection, blended by the
    vertex normal Z component so front-facing verts get the front image and
    back-facing verts get the back image with a smooth transition at the sides.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh":     ("TRIMESH",),
                "front_image": ("IMAGE",),
                "cam_dist": ("FLOAT", {
                    "default": 2.5, "min": 0.5, "max": 10.0, "step": 0.1,
                    "tooltip": "Virtual camera distance along +Z. Controls perspective correction depth. "
                               "Increase if texture is too zoomed on edges; decrease if too stretched. "
                               "Match to TripoSG training camera (~2.0-3.5)."
                }),
                "back_image_url": ("STRING", {
                    "default": "",
                    "tooltip": "Optional URL for a back-view image. Leave empty to mirror the front image."
                }),
            },
            "optional": {
                "back_image":  ("IMAGE",),
            },
        }

    RETURN_TYPES  = ("TRIMESH",)
    RETURN_NAMES  = ("trimesh",)
    FUNCTION      = "bake"
    CATEGORY      = "TripoSG"
    DESCRIPTION   = ("Bakes front/back view images onto mesh vertices via "
                     "orthographic projection weighted by vertex normals.")

    def bake(self, trimesh, front_image, cam_dist=2.5, back_image_url="", back_image=None):
        verts   = trimesh.vertices        # (N, 3)
        normals = trimesh.vertex_normals  # (N, 3) — auto-computed

        def to_u8(t):
            return (t[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        front_np = to_u8(front_image)
        if back_image is not None:
            back_np = to_u8(back_image)
        elif back_image_url and back_image_url.strip():
            import requests as _requests
            resp = _requests.get(back_image_url.strip(), timeout=30)
            resp.raise_for_status()
            back_pil = Image.open(BytesIO(resp.content)).convert("RGB")
            back_np = np.array(back_pil).astype(np.uint8)
        else:
            back_np = front_np[:, ::-1, :].copy()   # mirror front as fallback

        # --- Perspective projection onto the Z=0 reference plane ----------------
        # Orthographic (raw XY) is perfect at the closest face but drifts with Z
        # depth because the source image was rendered with a perspective camera.
        # Dividing by (cam_dist - z) and rescaling to z=0 undoes foreshortening.
        z     = verts[:, 2]
        depth = np.maximum(cam_dist - z, 1e-3)        # never divide by zero
        x_p   = verts[:, 0] / depth * cam_dist        # perspective-correct X
        y_p   = verts[:, 1] / depth * cam_dist        # perspective-correct Y

        # --- UV mapping that matches TripoSGPrepareImage exactly ----------------
        # PrepareImage adds pad_ratio (10 %) on every side and squares by the
        # dominant dimension — replicate that here on the projected coords.
        pad   = 0.1
        inner = 1.0 - 2.0 * pad
        xr = float(x_p.max() - x_p.min()) or 1.0
        yr = float(y_p.max() - y_p.min()) or 1.0

        if xr <= yr:   # tall / square — Y dominant
            v      = pad + (1.0 - (y_p - y_p.min()) / yr) * inner
            x_span = (xr / yr) * inner
            u      = 0.5 - x_span * 0.5 + (x_p - x_p.min()) / xr * x_span
        else:           # wide — X dominant
            u      = pad + (x_p - x_p.min()) / xr * inner
            y_span = (yr / xr) * inner
            v      = 0.5 - y_span * 0.5 + (1.0 - (y_p - y_p.min()) / yr) * y_span
        # ------------------------------------------------------------------------

        def sample(img, uc, vc):
            H, W = img.shape[:2]
            ix = np.clip((uc * (W - 1)).astype(np.int32), 0, W - 1)
            iy = np.clip((vc * (H - 1)).astype(np.int32), 0, H - 1)
            return img[iy, ix]              # (N, 3) uint8

        front_col = sample(front_np, u,       v)
        back_col  = sample(back_np,  1.0 - u, v)   # mirror X for back view

        # nz=+1 → fully front, nz=-1 → fully back
        w   = np.clip((normals[:, 2] + 1.0) / 2.0, 0.0, 1.0)[:, np.newaxis]
        rgb = (front_col.astype(np.float32) * w +
               back_col.astype(np.float32)  * (1.0 - w)).clip(0, 255).astype(np.uint8)

        alpha         = np.full((len(verts), 1), 255, dtype=np.uint8)
        vertex_colors = np.concatenate([rgb, alpha], axis=1)   # (N, 4) RGBA

        out = Trimesh.Trimesh(
            vertices=verts.copy(),
            faces=trimesh.faces.copy(),
            vertex_colors=vertex_colors,
            process=False,
        )
        return (out,)


class UVUnwrapAndBakeTexture:
    """
    xatlas UV-unwraps the mesh, then bakes a real basecolor texture into that
    UV space from front/back view images (same perspective-projection sampling
    as BakeVertexColorsFromViews, but rasterized per-pixel in UV space instead
    of per-vertex). Attaches the result as a glTF PBR material (baseColorTexture
    + constant roughness/metallic factors) instead of vertex colours — this is
    a real textured UV mesh, not a learned PBR material estimate.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh":     ("TRIMESH",),
                "front_image": ("IMAGE",),
                "cam_dist": ("FLOAT", {
                    "default": 2.5, "min": 0.5, "max": 10.0, "step": 0.1,
                    "tooltip": "Virtual camera distance along +Z, same convention as Bake Vertex Colors. "
                               "Match to TripoSG training camera (~2.0-3.5)."
                }),
                "texture_size": ("INT", {"default": 1024, "min": 256, "max": 2048, "step": 256}),
                "roughness": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05}),
                "metallic":  ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "back_image_url": ("STRING", {
                    "default": "",
                    "tooltip": "Optional URL for a back-view image. Leave empty to mirror the front image."
                }),
            },
            "optional": {
                "back_image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "bake"
    CATEGORY = "TripoSG"
    DESCRIPTION = ("xatlas UV-unwraps the mesh and bakes a real basecolor texture from "
                   "front/back photos into a glTF PBR material.")

    @staticmethod
    def _project_to_image_uv(pos, normal, cam_dist):
        # Same perspective-correct projection + 10% pad-square mapping used in
        # BakeVertexColorsFromViews — kept in sync so bakes match vertex-colour
        # results at the same cam_dist.
        z     = pos[:, 2]
        depth = np.maximum(cam_dist - z, 1e-3)
        x_p   = pos[:, 0] / depth * cam_dist
        y_p   = pos[:, 1] / depth * cam_dist

        pad   = 0.1
        inner = 1.0 - 2.0 * pad
        xr = float(x_p.max() - x_p.min()) or 1.0
        yr = float(y_p.max() - y_p.min()) or 1.0

        if xr <= yr:
            v      = pad + (1.0 - (y_p - y_p.min()) / yr) * inner
            x_span = (xr / yr) * inner
            u      = 0.5 - x_span * 0.5 + (x_p - x_p.min()) / xr * x_span
        else:
            u      = pad + (x_p - x_p.min()) / xr * inner
            y_span = (yr / xr) * inner
            v      = 0.5 - y_span * 0.5 + (1.0 - (y_p - y_p.min()) / yr) * y_span

        w = np.clip((normal[:, 2] + 1.0) / 2.0, 0.0, 1.0)  # front/back blend weight
        return u, v, w

    def bake(self, trimesh, front_image, cam_dist=2.5, texture_size=1024,
              roughness=0.6, metallic=0.0, back_image_url="", back_image=None):
        import xatlas

        def to_u8(t):
            return (t[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        front_np = to_u8(front_image)
        if back_image is not None:
            back_np = to_u8(back_image)
        elif back_image_url and back_image_url.strip():
            import requests as _requests
            resp = _requests.get(back_image_url.strip(), timeout=30)
            resp.raise_for_status()
            back_pil = Image.open(BytesIO(resp.content)).convert("RGB")
            back_np = np.array(back_pil).astype(np.uint8)
        else:
            back_np = front_np[:, ::-1, :].copy()

        verts = trimesh.vertices.astype(np.float32)
        faces = trimesh.faces.astype(np.uint32)
        src_normals = trimesh.vertex_normals.astype(np.float32)

        vmapping, indices, uvs = xatlas.parametrize(verts, faces)
        new_verts   = verts[vmapping]
        new_normals = src_normals[vmapping]
        new_faces   = indices.astype(np.int64).reshape(-1, 3)
        uvs = uvs.astype(np.float32)  # (M, 2) in [0,1]

        T = texture_size
        tex   = np.zeros((T, T, 3), dtype=np.uint8)
        filled = np.zeros((T, T), dtype=np.uint8)

        # UV -> pixel space; V flipped so v=0 is bottom (image row 0 is top).
        px = uvs[:, 0] * (T - 1)
        py = (1.0 - uvs[:, 1]) * (T - 1)

        def sample_img(img, uc, vc):
            H, W = img.shape[:2]
            ix = np.clip((uc * (W - 1)).astype(np.int32), 0, W - 1)
            iy = np.clip((vc * (H - 1)).astype(np.int32), 0, H - 1)
            return img[iy, ix].astype(np.float32)

        for tri in new_faces:
            tpx, tpy = px[tri], py[tri]
            x0, x1 = int(np.floor(tpx.min())), int(np.ceil(tpx.max()))
            y0, y1 = int(np.floor(tpy.min())), int(np.ceil(tpy.max()))
            x0, y0 = max(x0, 0), max(y0, 0)
            x1, y1 = min(x1, T - 1), min(y1, T - 1)
            if x1 < x0 or y1 < y0:
                continue

            xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
            xs = xs.astype(np.float32) + 0.5
            ys = ys.astype(np.float32) + 0.5

            x1v, y1v, x2v, y2v, x3v, y3v = tpx[0], tpy[0], tpx[1], tpy[1], tpx[2], tpy[2]
            denom = (y2v - y3v) * (x1v - x3v) + (x3v - x2v) * (y1v - y3v)
            if abs(denom) < 1e-8:
                continue
            w1 = ((y2v - y3v) * (xs - x3v) + (x3v - x2v) * (ys - y3v)) / denom
            w2 = ((y3v - y1v) * (xs - x3v) + (x1v - x3v) * (ys - y3v)) / denom
            w3 = 1.0 - w1 - w2

            inside = (w1 >= -1e-4) & (w2 >= -1e-4) & (w3 >= -1e-4)
            if not inside.any():
                continue

            v0, v1_, v2_ = new_verts[tri[0]], new_verts[tri[1]], new_verts[tri[2]]
            n0, n1_, n2_ = new_normals[tri[0]], new_normals[tri[1]], new_normals[tri[2]]

            pos_pix = (w1[..., None] * v0 + w2[..., None] * v1_ + w3[..., None] * v2_)
            nrm_pix = (w1[..., None] * n0 + w2[..., None] * n1_ + w3[..., None] * n2_)

            flat_pos = pos_pix[inside]
            flat_nrm = nrm_pix[inside]
            u_img, v_img, blend = self._project_to_image_uv(flat_pos, flat_nrm, cam_dist)

            front_c = sample_img(front_np, u_img, v_img)
            back_c  = sample_img(back_np, 1.0 - u_img, v_img)
            rgb = front_c * blend[:, None] + back_c * (1.0 - blend[:, None])

            py_idx = (ys[inside] - 0.5).astype(np.int32)
            px_idx = (xs[inside] - 0.5).astype(np.int32)
            tex[py_idx, px_idx] = rgb.clip(0, 255).astype(np.uint8)
            filled[py_idx, px_idx] = 255

        # Seam/gap padding — push filled colour into unbaked UV-chart borders
        # so bilinear texture sampling doesn't pick up black at seams.
        if (filled == 0).any() and (filled != 0).any():
            mask = (filled == 0).astype(np.uint8) * 255
            tex = cv2.inpaint(tex, mask, 3, cv2.INPAINT_TELEA)

        tex_img = Image.fromarray(tex, mode="RGB")

        material = Trimesh.visual.material.PBRMaterial(
            baseColorTexture=tex_img,
            roughnessFactor=float(roughness),
            metallicFactor=float(metallic),
        )
        visual = Trimesh.visual.TextureVisuals(uv=uvs, material=material)

        out = Trimesh.Trimesh(
            vertices=new_verts,
            faces=new_faces,
            visual=visual,
            process=False,
        )
        return (out,)


class LoadImageFromURL:
    """Load an image directly from a URL, bypassing ComfyUI's local-file validation."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "url": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "load"
    CATEGORY = "TripoSG"

    def load(self, url):
        import requests

        url = url.strip()
        if not url:
            # No URL supplied — return a blank white 512×512 image so
            # text-only runs (e.g. TripoSG-scribble with prompt only) don't crash.
            blank = np.ones((512, 512, 3), dtype=np.float32)
            image = torch.from_numpy(blank).unsqueeze(0)
            mask  = torch.zeros((1, 512, 512), dtype=torch.float32)
            return (image, mask)

        # Graydient's image-upload field mapping (as opposed to a fetchable
        # init_image_url) sometimes drops the file into ComfyUI's local input/
        # directory and passes back a bare filename, not a URL — architecturally
        # unpredictable per submission path, confirmed on the same local_field
        # across different jobs (see KI-007 §2/§7). requests.get() on a bare
        # filename raises MissingSchema, so treat anything without an http(s)
        # scheme as a local upload and resolve it through folder_paths instead
        # of fetching it — same pattern as TripoSGLoadVideoFromURL below.
        is_remote = url.startswith("http://") or url.startswith("https://")

        if is_remote:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert("RGBA")
        else:
            local_path = folder_paths.get_annotated_filepath(url)
            if not os.path.isfile(local_path):
                raise RuntimeError(f"local image upload not found: {url}")
            img = Image.open(local_path).convert("RGBA")
        arr = np.array(img).astype(np.float32) / 255.0

        # IMAGE: [1, H, W, 3] RGB float32 0–1
        image = torch.from_numpy(arr[:, :, :3]).unsqueeze(0)
        # MASK: [1, H, W] ComfyUI convention — 0 = opaque, 1 = transparent
        mask = torch.from_numpy(1.0 - arr[:, :, 3]).unsqueeze(0)

        return (image, mask)


class TripoSGLoadVideoFromURL:
    """Download a video from URL and decode frames as an IMAGE batch."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url":        ("STRING",  {"default": ""}),
                "max_frames": ("INT",     {"default": 81,   "min": 1,   "max": 1000}),
                "fps":        ("FLOAT",   {"default": 16.0, "min": 1.0, "max": 60.0}),
                "width":      ("INT",     {"default": 832,  "min": 64,  "max": 4096, "step": 8}),
                "height":     ("INT",     {"default": 480,  "min": 64,  "max": 4096, "step": 8}),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("frames",)
    FUNCTION      = "load"
    CATEGORY      = "TripoSG"

    def load(self, url, max_frames=81, fps=16.0, width=832, height=480):
        import requests
        import shutil
        import subprocess
        import tempfile

        url = url.strip()
        if not url:
            blank = np.zeros((max_frames, height, width, 3), dtype=np.float32)
            return (torch.from_numpy(blank),)

        # Graydient's video-upload field mapping (as opposed to a fetchable
        # init_image_url) drops the file into ComfyUI's local input/ directory
        # and passes back a bare filename, not a URL — same convention as the
        # native LoadVideo widget. requests.get() on a bare filename raises
        # MissingSchema, so treat anything without an http(s) scheme as a local
        # upload and resolve it through folder_paths instead of fetching it.
        is_remote = url.startswith("http://") or url.startswith("https://")

        tmp_dir    = tempfile.mkdtemp()
        tmp_video  = os.path.join(tmp_dir, "input.mp4")
        frames_dir = os.path.join(tmp_dir, "frames")
        os.makedirs(frames_dir)

        try:
            if is_remote:
                resp = requests.get(url, timeout=120, stream=True)
                resp.raise_for_status()
                with open(tmp_video, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=65536):
                        fh.write(chunk)
            else:
                local_path = folder_paths.get_annotated_filepath(url)
                if not os.path.isfile(local_path):
                    raise RuntimeError(f"local video upload not found: {url}")
                shutil.copyfile(local_path, tmp_video)

            subprocess.check_call([
                "ffmpeg", "-y", "-i", tmp_video,
                "-vf", f"fps={fps},scale={width}:{height}:flags=lanczos",
                "-frames:v", str(max_frames),
                "-pix_fmt", "rgb24", "-f", "image2",
                os.path.join(frames_dir, "frame_%05d.png"),
            ], stderr=subprocess.DEVNULL)

            frame_paths = sorted(Path(frames_dir).glob("frame_*.png"))
            if not frame_paths:
                raise RuntimeError("ffmpeg produced no frames from the video")

            frames = [np.array(Image.open(p).convert("RGB")) for p in frame_paths[:max_frames]]
            while len(frames) < max_frames:
                frames.append(frames[-1])

            return (torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0),)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _render_text_card(text: str, title: str = "", width: int = 1200) -> torch.Tensor:
    """Render a string as a dark-themed card IMAGE tensor [1,H,W,3]."""
    from PIL import Image, ImageDraw, ImageFont

    LINE_H, HEADER_H, PADDING = 30, 70, 40
    lines = text.splitlines()
    card_h = max(600, HEADER_H + len(lines) * LINE_H + PADDING)

    img  = Image.new("RGB", (width, card_h), color=(11, 14, 18))
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.load_default(size=22)
        font_body  = ImageFont.load_default(size=16)
    except TypeError:
        font_title = font_body = ImageFont.load_default()

    if title:
        draw.text((PADDING, 16), title, fill=(108, 71, 255), font=font_title)
    draw.line([(PADDING, 56), (width - PADDING, 56)], fill=(30, 33, 40), width=1)

    y = HEADER_H
    for line in lines:
        draw.text((PADDING, y), line, fill=(210, 215, 225), font=font_body)
        y += LINE_H
        if y > card_h - LINE_H:
            draw.text((PADDING, y), "… (truncated)", fill=(60, 70, 90), font=font_body)
            break

    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _encode_string_as_image(text: str) -> torch.Tensor:
    """Losslessly encode a UTF-8 string as an RGB IMAGE tensor.
    Format: 4-byte big-endian length header + UTF-8 bytes, 3 bytes per pixel.
    ForgeExpress decodes by reading pixel values back to bytes."""
    import math
    import struct

    payload = struct.pack(">I", len(text.encode("utf-8"))) + text.encode("utf-8")
    while len(payload) % 3:
        payload += b"\x00"

    n_pixels = len(payload) // 3
    side     = max(1, math.ceil(math.sqrt(n_pixels)))
    padded   = payload + b"\x00" * (side * side * 3 - len(payload))

    arr = np.frombuffer(padded, dtype=np.uint8).reshape(side, side, 3).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────

class TranscribeAudioFromURL:
    """Download audio from URL and transcribe to timed lyrics using faster-whisper.
    Returns a lyrics card IMAGE (suitable as Graydient workflow output) and the raw JSON STRING."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url":        ("STRING", {"default": ""}),
                "model_size": (["large-v3", "medium", "small", "base"], {}),
                "language":   ("STRING", {"default": ""}),
            },
            "optional": {
                # Graydient's init_<type>_* is a {bool, filename, url} triplet and
                # it's unconfirmed which of filename/url gets populated for a given
                # submission path -- accept both, first non-empty wins.
                "filename": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES  = ("IMAGE", "STRING")
    RETURN_NAMES  = ("lyrics_card", "lyrics_json")
    FUNCTION      = "transcribe"
    CATEGORY      = "TripoSG"

    def transcribe(self, url, model_size="large-v3", language="", filename=""):
        import json
        import os
        import shutil
        import tempfile

        import requests
        import torch
        from PIL import Image, ImageDraw, ImageFont

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            raise RuntimeError(
                "faster-whisper not installed. Add 'faster-whisper' to the workflow's pip requirements."
            )

        # ── Resolve audio source ──────────────────────────────────────────────
        # init_audio_url may pass a URL (direct) OR a local filename that
        # Graydient pre-downloaded (e.g. "init_audio__httpsapi.telegram.org...mp3").
        url_clean = (url or "").strip() or (filename or "").strip()

        if not url_clean:
            # No URL supplied — return blank outputs so the workflow doesn't crash
            blank = np.ones((512, 512, 3), dtype=np.float32)
            card_t = torch.from_numpy(blank).unsqueeze(0)
            import json as _json
            empty = _json.dumps({"language": "unknown", "duration": 0.0, "timeline": []})
            return (card_t, empty)

        tmp_dir   = tempfile.mkdtemp()
        ext       = os.path.splitext(url_clean.split("?")[0])[-1] or ".mp3"
        tmp_path  = os.path.join(tmp_dir, f"audio{ext}")

        try:
            if url_clean.startswith(("http://", "https://")):
                # Direct URL — download ourselves
                resp = requests.get(url_clean, timeout=120, stream=True)
                resp.raise_for_status()
                with open(tmp_path, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=65536):
                        fh.write(chunk)
            else:
                # Graydient pre-downloaded the file; resolve against ComfyUI input dir.
                # Use isfile() not exists() — avoids matching the input directory itself
                # when url_clean is empty or resolves to a directory.
                import folder_paths as _fp
                candidate = os.path.join(_fp.get_input_directory(), url_clean)
                if os.path.isfile(candidate):
                    tmp_path = candidate
                elif os.path.isfile(url_clean):
                    tmp_path = url_clean
                else:
                    raise FileNotFoundError(
                        f"Audio file not found: {url_clean!r}\n"
                        f"(looked in {_fp.get_input_directory()} and cwd)"
                    )

            # ── Transcribe ────────────────────────────────────────────────────
            device       = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"

            # If concept_mapping pre-downloaded the model weights, load from local path.
            # Graydient puts concept_mapping files in {ComfyUI}/models/.
            local_model_dir = os.path.join(
                folder_paths.models_dir, "whisper", model_size
            )
            load_from = local_model_dir if os.path.isfile(
                os.path.join(local_model_dir, "model.bin")
            ) else model_size

            model = WhisperModel(load_from, device=device, compute_type=compute_type)

            lang   = language.strip() if language.strip() else None
            raw, info = model.transcribe(
                tmp_path,
                language               = lang,
                beam_size              = 5,
                # word_timestamps=True: use actual word boundaries rather than Whisper's
                # segment-level timestamps. This fixes the "intro declared as speech"
                # problem where Whisper reports a segment starting at 0:00 even when the
                # first word is at 0:12 — giving gap detection a real gap to work with.
                word_timestamps            = True,
                no_speech_threshold        = 0.7,
                condition_on_previous_text = False,
            )
            lyrical = []
            for s in raw:
                text = s.text.strip()
                if not text:
                    continue
                # Adjust start/end to actual word boundaries if available
                if s.words:
                    seg_start = s.words[0].start
                    seg_end   = s.words[-1].end
                    words = [{"start": round(w.start, 2), "end": round(w.end, 2),
                              "text": w.word.strip()} for w in s.words if w.word.strip()]
                else:
                    seg_start = s.start
                    seg_end   = s.end
                    words = []
                lyrical.append({
                    "start": round(seg_start, 2),
                    "end":   round(seg_end,   2),
                    "text":  text,
                    "words": words,
                })

        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        # ── Gap detection — build annotated timeline ───────────────────────────
        # Any window > GAP_THRESH seconds between vocal segments becomes an
        # instrumental entry. These are natural B-roll / scenic shot slots.
        GAP_THRESH = 1.5
        timeline   = []
        prev_end   = 0.0

        for seg in lyrical:
            gap = round(seg["start"] - prev_end, 2)
            if gap > GAP_THRESH:
                timeline.append({
                    "start": round(prev_end, 2),
                    "end":   round(seg["start"], 2),
                    "type":  "instrumental",
                })
            timeline.append({**seg, "type": "lyric"})
            prev_end = seg["end"]

        # Outro gap
        if info.duration - prev_end > GAP_THRESH:
            timeline.append({
                "start": round(prev_end, 2),
                "end":   round(info.duration, 2),
                "type":  "instrumental",
            })

        n_lyric = sum(1 for e in timeline if e["type"] == "lyric")
        n_instr = sum(1 for e in timeline if e["type"] == "instrumental")

        lyrics_data = {
            "language": info.language,
            "duration": round(info.duration, 2),
            "timeline": timeline,
        }
        lyrics_json = json.dumps(lyrics_data, indent=2, ensure_ascii=False)

        # ── Render lyrics card ────────────────────────────────────────────────
        CARD_W   = 1200
        LINE_H   = 36
        HEADER_H = 70
        PADDING  = 40
        card_h   = max(600, HEADER_H + len(timeline) * LINE_H + PADDING)

        img  = Image.new("RGB", (CARD_W, card_h), color=(11, 14, 18))
        draw = ImageDraw.Draw(img)

        try:
            font_ts   = ImageFont.load_default(size=17)
            font_text = ImageFont.load_default(size=19)
            font_hdr  = ImageFont.load_default(size=22)
        except TypeError:
            font_ts = font_text = font_hdr = ImageFont.load_default()

        meta = (f"lang:{info.language}  dur:{info.duration:.1f}s  "
                f"lyric:{n_lyric}  instrumental:{n_instr}")
        draw.text((PADDING, 16), "TIMED LYRICS", fill=(108, 71, 255), font=font_hdr)
        draw.text((PADDING + 280, 20), meta, fill=(80, 90, 110), font=font_ts)
        draw.line([(PADDING, 56), (CARD_W - PADDING, 56)], fill=(30, 33, 40), width=1)

        y = HEADER_H
        for entry in timeline:
            s, e = entry["start"], entry["end"]
            ts = f"[{int(s)//60:02d}:{s%60:05.2f}→{int(e)//60:02d}:{e%60:05.2f}]"
            if entry["type"] == "instrumental":
                draw.rectangle([(PADDING - 4, y - 2), (CARD_W - PADDING, y + LINE_H - 6)],
                                fill=(20, 22, 28))
                draw.text((PADDING, y), ts, fill=(50, 60, 75), font=font_ts)
                draw.text((PADDING + 320, y), f"── instrumental  ({e - s:.1f}s) ──",
                          fill=(55, 70, 90), font=font_text)
            else:
                draw.text((PADDING, y), ts, fill=(108, 71, 255), font=font_ts)
                draw.text((PADDING + 320, y), entry["text"], fill=(210, 215, 225), font=font_text)
            y += LINE_H
            if y > card_h - LINE_H:
                draw.text((PADDING, y), "… (truncated)", fill=(60, 70, 90), font=font_ts)
                break

        img_np = np.array(img).astype(np.float32) / 255.0
        img_t  = torch.from_numpy(img_np).unsqueeze(0)   # [1, H, W, 3]

        return (img_t, lyrics_json)


class LoadVideoFromURLAuto:
    """Download a video from URL and decode frames as an IMAGE batch, auto-detecting
    the source resolution via ffprobe (aspect-ratio preserving, capped to max_side)
    instead of forcing a fixed width/height like TripoSGLoadVideoFromURL. Used by the
    subtitle-burn pipeline so on-screen text can be sized/positioned relative to the
    video's real frame dimensions rather than a hardcoded default."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url":        ("STRING", {"default": ""}),
                "max_frames": ("INT",   {"default": 720, "min": 1, "max": 3000}),
                "fps":        ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0}),
                "max_side":   ("INT",   {"default": 1280, "min": 64, "max": 4096, "step": 8}),
            },
            "optional": {
                # Graydient's init_video_* is a {bool, filename, url} triplet and it's
                # unconfirmed which of filename/url gets populated for a given
                # submission path -- accept both, first non-empty wins.
                "filename": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("frames",)
    FUNCTION      = "load"
    CATEGORY      = "TripoSG"

    def load(self, url, max_frames=720, fps=24.0, max_side=1280, filename=""):
        import json as _json
        import shutil
        import subprocess
        import tempfile

        import requests

        url = (url or "").strip() or (filename or "").strip()
        if not url:
            blank = np.zeros((1, 480, 832, 3), dtype=np.float32)
            return (torch.from_numpy(blank),)

        is_remote = url.startswith("http://") or url.startswith("https://")
        tmp_dir    = tempfile.mkdtemp()
        tmp_video  = os.path.join(tmp_dir, "input.mp4")
        frames_dir = os.path.join(tmp_dir, "frames")
        os.makedirs(frames_dir)

        try:
            if is_remote:
                resp = requests.get(url, timeout=120, stream=True)
                resp.raise_for_status()
                with open(tmp_video, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=65536):
                        fh.write(chunk)
            else:
                # Graydient's video-upload field mapping drops the file into ComfyUI's
                # local input/ directory and passes back a bare filename, not a URL.
                local_path = folder_paths.get_annotated_filepath(url)
                if not os.path.isfile(local_path):
                    raise RuntimeError(f"local video upload not found: {url}")
                shutil.copyfile(local_path, tmp_video)

            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height",
                 "-of", "json", tmp_video],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
            try:
                info = _json.loads(probe.stdout or b"{}")
                streams = info.get("streams", [{}])
                src_w = int(streams[0].get("width", 1280))
                src_h = int(streams[0].get("height", 720))
            except Exception:
                src_w, src_h = 1280, 720

            scale = min(1.0, max_side / max(src_w, src_h))
            # h264 requires even dimensions
            out_w = max(2, int(round(src_w * scale / 2)) * 2)
            out_h = max(2, int(round(src_h * scale / 2)) * 2)

            subprocess.check_call([
                "ffmpeg", "-y", "-i", tmp_video,
                "-vf", f"fps={fps},scale={out_w}:{out_h}:flags=lanczos",
                "-frames:v", str(max_frames),
                "-pix_fmt", "rgb24", "-f", "image2",
                os.path.join(frames_dir, "frame_%05d.png"),
            ], stderr=subprocess.DEVNULL)

            frame_paths = sorted(Path(frames_dir).glob("frame_*.png"))
            if not frame_paths:
                raise RuntimeError("ffmpeg produced no frames from the video")

            frames = [np.array(Image.open(p).convert("RGB")) for p in frame_paths[:max_frames]]
            return (torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0),)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)


_FONT_CACHE = {}


def _resolve_font_path(family: str):
    """Map a font family name to a bundled TTF path + horizontal squeeze factor
    (used to fake a 'condensed' variant without shipping a separate font file).
    Sourced from matplotlib's bundled DejaVu fonts, which ship as package data
    with a very common pip dependency rather than relying on system fonts being
    present in an ephemeral container."""
    if family in _FONT_CACHE:
        return _FONT_CACHE[family]

    mpl_ttf_dir = None
    try:
        import matplotlib
        mpl_ttf_dir = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    except Exception:
        pass

    fname_map = {
        "sans":      "DejaVuSans-Bold.ttf",
        "serif":     "DejaVuSerif-Bold.ttf",
        "mono":      "DejaVuSansMono-Bold.ttf",
        "condensed": "DejaVuSans-Bold.ttf",   # squeezed horizontally, see below
    }
    squeeze = 0.78 if family == "condensed" else 1.0
    fname   = fname_map.get(family, fname_map["sans"])

    search_dirs = [mpl_ttf_dir] if mpl_ttf_dir else []
    search_dirs += ["/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/truetype/liberation"]

    for d in search_dirs:
        if not d:
            continue
        p = os.path.join(d, fname)
        if os.path.isfile(p):
            _FONT_CACHE[family] = (p, squeeze)
            return _FONT_CACHE[family]

    _FONT_CACHE[family] = (None, squeeze)
    return _FONT_CACHE[family]


def _hex_to_rgb(s: str, default=(255, 255, 255)):
    s = (s or "").strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        return default
    try:
        return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))
    except ValueError:
        return default


def _wrap_lines(text, font, max_width, draw):
    words = text.split()
    lines, cur = [], ""
    for word in words:
        trial = (cur + " " + word).strip()
        bbox = draw.textbbox((0, 0), trial, font=font)
        if bbox[2] - bbox[0] <= max_width or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def _render_subtitle_layer(lines, font, color, style, outline_w, squeeze, max_w):
    """Render wrapped subtitle lines onto a transparent RGBA layer, applying the
    chosen style (outline / box / shadow). squeeze < 1.0 fakes a condensed font by
    horizontally compressing the finished layer instead of shipping a separate font."""
    from PIL import Image, ImageDraw

    pad = outline_w + 4
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    line_boxes = [probe.textbbox((0, 0), ln, font=font) for ln in lines]
    line_w = [b[2] - b[0] for b in line_boxes]
    ascent, descent = font.getmetrics()
    line_step = ascent + descent + max(2, int(font.size * 0.15))

    layer_w = min(max_w, max(line_w) + pad * 2) if line_w else pad * 2
    layer_h = line_step * len(lines) + pad * 2
    layer = Image.new("RGBA", (max(1, layer_w), max(1, layer_h)), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    y = pad
    for ln, lw in zip(lines, line_w):
        x = (layer_w - lw) // 2
        if style == "box":
            draw.rectangle(
                [(pad // 2, y - pad // 3), (layer_w - pad // 2, y + line_step - pad // 3)],
                fill=(0, 0, 0, 150),
            )
            draw.text((x, y), ln, font=font, fill=(*color, 255))
        elif style == "shadow":
            dx = dy = max(1, outline_w)
            draw.text((x + dx, y + dy), ln, font=font, fill=(0, 0, 0, 200))
            draw.text((x, y), ln, font=font, fill=(*color, 255))
        else:  # "outline" — thin black stroke, universal readability, the default
            draw.text((x, y), ln, font=font, fill=(*color, 255),
                       stroke_width=outline_w, stroke_fill=(0, 0, 0, 255))
        y += line_step

    if squeeze < 0.999:
        layer = layer.resize((max(1, int(layer_w * squeeze)), layer_h), Image.LANCZOS)

    return layer


def _rechunk_by_speed(entries, speed):
    """Regroup each lyric entry's words into shorter caption chunks as speed rises.
    speed<=1 is a no-op (natural Whisper sentence segments = 'normal pacing'); higher
    speed caps each chunk to fewer words, so captions cut faster and show less text
    at once. Entries with no word-level timing (older lyrics_json, or a segment
    faster-whisper didn't return words for) pass through unchanged."""
    if speed <= 1.01:
        return entries

    max_words = max(1, round(9.0 / speed))
    out = []
    for e in entries:
        words = e.get("words") or []
        if len(words) <= max_words:
            out.append(e)
            continue
        for i in range(0, len(words), max_words):
            chunk = words[i:i + max_words]
            out.append({
                "start": chunk[0]["start"],
                "end":   chunk[-1]["end"],
                "text":  " ".join(w["text"] for w in chunk).strip(),
                "type":  "lyric",
            })
    return out


class BurnSubtitlesFromTimeline:
    """Burn dynamically positioned/sized subtitles onto a video frame batch using a
    Whisper timeline (as produced by TranscribeAudioFromURL's lyrics_json output).
    Font size and bottom margin scale with the actual frame resolution rather than a
    fixed pixel value, so the same workflow looks right at any aspect ratio. Styling
    (font family / size / colour / outline-vs-box-vs-shadow) is exposed as plain
    widgets so Graydient can wire them to slot1-slot4. `speed` (slot5) controls how
    aggressively long Whisper segments get split into shorter word-chunks — 1 keeps
    natural sentence-length captions, 4 cuts to ~2 words per caption."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames":      ("IMAGE",),
                "lyrics_json": ("STRING", {"default": "", "multiline": True}),
                "fps":         ("FLOAT", {"default": 24.0, "min": 1.0, "max": 60.0}),
                "font_family": (["sans", "serif", "mono", "condensed"], {"default": "sans"}),
                "font_scale":  ("FLOAT", {"default": 1.0, "min": 0.4, "max": 2.5, "step": 0.05}),
                "text_color":  ("STRING", {"default": "#FFFFFF"}),
                "style":       (["outline", "box", "shadow"], {"default": "outline"}),
                "speed":       ("FLOAT", {"default": 1.0, "min": 1.0, "max": 4.0, "step": 0.1}),
            }
        }

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("frames",)
    FUNCTION      = "burn"
    CATEGORY      = "TripoSG"

    def burn(self, frames, lyrics_json, fps=24.0, font_family="sans", font_scale=1.0,
             text_color="#FFFFFF", style="outline", speed=1.0):
        import json

        from PIL import Image, ImageDraw, ImageFont

        n, h, w, _ = frames.shape

        try:
            data = json.loads(lyrics_json) if lyrics_json and lyrics_json.strip() else {"timeline": []}
        except Exception:
            data = {"timeline": []}
        entries = [e for e in data.get("timeline", [])
                   if e.get("type") == "lyric" and e.get("text")]
        entries = _rechunk_by_speed(entries, speed)
        entries.sort(key=lambda e: e["start"])

        if not entries:
            return (frames,)

        font_path, squeeze = _resolve_font_path(font_family)
        base_size = max(10, int(round(h * 0.052 * font_scale)))
        font = (ImageFont.truetype(font_path, base_size)
                if font_path else ImageFont.load_default(size=base_size))

        color = _hex_to_rgb(text_color)
        margin_bottom  = max(4, int(round(h * 0.06)))
        max_text_width = max(20, int(round(w * 0.88 / squeeze)))
        outline_w      = max(1, int(round(base_size * 0.07)))

        frames_np = (frames.cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        out = np.empty_like(frames_np)

        probe_draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))

        cur_entry, cur_layer = None, None
        e_i, n_entries = 0, len(entries)

        for i in range(n):
            t = i / fps
            while e_i < n_entries and entries[e_i]["end"] <= t:
                e_i += 1
            entry = entries[e_i] if e_i < n_entries and entries[e_i]["start"] <= t < entries[e_i]["end"] else None

            frame_img = Image.fromarray(frames_np[i]).convert("RGBA")

            if entry is not None:
                if entry is not cur_entry:
                    cur_entry = entry
                    lines = _wrap_lines(entry["text"], font, max_text_width, probe_draw)
                    cur_layer = _render_subtitle_layer(
                        lines, font, color, style, outline_w, squeeze, w
                    )
                if cur_layer is not None:
                    lw, lh = cur_layer.size
                    px = (w - lw) // 2
                    py = h - margin_bottom - lh
                    frame_img.alpha_composite(cur_layer, (px, max(0, py)))

            out[i] = np.array(frame_img.convert("RGB"))

        result = torch.from_numpy(out.astype(np.float32) / 255.0)
        return (result,)


class HFTextGenerate:
    """Run text-only inference with any HuggingFace instruction-tuned model.
    Returns the response as a STRING and as an encoded data IMAGE (for Graydient output).
    Default model: Qwen/Qwen2.5-7B-Instruct (~15 GB)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id":       ("STRING", {"default": "Qwen/Qwen2.5-7B-Instruct"}),
                "system_prompt":  ("STRING", {"multiline": True,
                                              "default": "You are a helpful assistant. Respond with valid JSON only."}),
                "user_prompt":    ("STRING", {"multiline": True, "default": ""}),
                "max_new_tokens": ("INT",    {"default": 2048, "min": 64, "max": 8192}),
                "temperature":    ("FLOAT",  {"default": 0.3,  "min": 0.0, "max": 2.0, "step": 0.05}),
            }
        }

    RETURN_TYPES  = ("STRING", "IMAGE", "IMAGE")
    RETURN_NAMES  = ("response_text", "response_card", "response_data")
    FUNCTION      = "generate"
    CATEGORY      = "TripoSG"

    def generate(self, model_id, system_prompt, user_prompt, max_new_tokens, temperature):
        import torch

        try:
            from transformers import pipeline as hf_pipeline
        except ImportError:
            raise RuntimeError("transformers not installed.")

        device     = "cuda" if torch.cuda.is_available() else "cpu"
        dtype      = torch.bfloat16 if device == "cuda" else torch.float32

        pipe = hf_pipeline(
            "text-generation",
            model=model_id,
            torch_dtype=dtype,
            device_map="auto",
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ]

        do_sample = temperature > 0
        out = pipe(
            messages,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
        )
        response = out[0]["generated_text"][-1]["content"]

        del pipe
        if device == "cuda":
            torch.cuda.empty_cache()

        card = _render_text_card(response, title=f"LLM: {model_id.split('/')[-1]}")
        data = _encode_string_as_image(response)
        return (response, card, data)


class VLMInferFromURL:
    """Multimodal inference with Qwen2.5-VL (or Qwen2-VL) on an image from URL.
    Returns the response as STRING, a readable card IMAGE, and an encoded data IMAGE.
    Default model: Qwen/Qwen2.5-VL-7B-Instruct (~15 GB)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_url":      ("STRING", {"default": ""}),
                "model_id":       ("STRING", {"default": "Qwen/Qwen2.5-VL-7B-Instruct"}),
                "system_prompt":  ("STRING", {"multiline": True,
                                              "default": "You are a helpful assistant. Respond with valid JSON only."}),
                "user_prompt":    ("STRING", {"multiline": True, "default": "Describe this image."}),
                "max_new_tokens": ("INT",    {"default": 2048, "min": 64, "max": 4096}),
            }
        }

    RETURN_TYPES  = ("STRING", "IMAGE", "IMAGE")
    RETURN_NAMES  = ("response_text", "response_card", "response_data")
    FUNCTION      = "infer"
    CATEGORY      = "TripoSG"

    def infer(self, image_url, model_id, system_prompt, user_prompt, max_new_tokens):
        import torch

        try:
            from qwen_vl_utils import process_vision_info
        except ImportError:
            raise RuntimeError("qwen-vl-utils not installed. Add 'qwen-vl-utils' to pip requirements.")

        # Support both Qwen2-VL and Qwen2.5-VL
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration as QwenVLModel
        except ImportError:
            from transformers import Qwen2VLForConditionalGeneration as QwenVLModel
        from transformers import AutoProcessor

        model = QwenVLModel.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_id)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_url},
                    {"type": "text",  "text":  user_prompt},
                ],
            },
        ]

        text_input  = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        img_inputs, vid_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text_input], images=img_inputs, videos=vid_inputs,
            padding=True, return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            gen_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

        trimmed  = [g[len(i):] for i, g in zip(inputs.input_ids, gen_ids)]
        response = processor.batch_decode(trimmed, skip_special_tokens=True)[0]

        del model, processor, inputs
        torch.cuda.empty_cache()

        card = _render_text_card(response, title=f"VLM: {model_id.split('/')[-1]}")
        data = _encode_string_as_image(response)
        return (response, card, data)


class ConcatStrings:
    """Concatenate up to 8 STRING inputs in order. Empty parts are skipped.
    Each part can be left as a literal widget value or overridden by a link, same as
    any other plain STRING widget (see HFTextGenerate.user_prompt) -- no forceInput,
    so this doubles as a way to splice a fixed literal template around dynamic
    (LLM-generated or field-mapped) text without a separate templating node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            f"part{i}": ("STRING", {"multiline": True, "default": ""})
            for i in range(1, 9)
        }}

    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("text",)
    FUNCTION      = "concat"
    CATEGORY      = "TripoSG"

    def concat(self, part1, part2, part3, part4, part5, part6, part7, part8):
        return ("".join(p for p in (part1, part2, part3, part4, part5, part6, part7, part8) if p),)


class LoadVideoFromURLAsVideo:
    """Download a video from a URL (or resolve a local Graydient upload filename)
    and wrap it as ComfyUI's native VIDEO type -- unlike LoadVideoFromURL/
    TripoSGLoadVideoFromURL, which only ever return a decoded IMAGE frame batch,
    this preserves the container's real audio track and container-reported fps
    so downstream GetVideoComponents (and any node expecting a real VIDEO
    object) works exactly as it does with the core LoadVideo node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"url": ("STRING", {"default": ""})}}

    RETURN_TYPES = ("VIDEO",)
    RETURN_NAMES = ("video",)
    FUNCTION     = "load"
    CATEGORY     = "TripoSG"

    def load(self, url):
        import requests
        import tempfile
        import shutil
        from comfy_api.latest import InputImpl

        url = url.strip()
        is_remote = url.startswith("http://") or url.startswith("https://")

        tmp_dir = tempfile.mkdtemp()
        tmp_video = os.path.join(tmp_dir, "input.mp4")

        if is_remote:
            resp = requests.get(url, timeout=120, stream=True)
            resp.raise_for_status()
            with open(tmp_video, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=65536):
                    fh.write(chunk)
        else:
            local_path = folder_paths.get_annotated_filepath(url)
            if not os.path.isfile(local_path):
                raise RuntimeError(f"local video upload not found: {url}")
            shutil.copyfile(local_path, tmp_video)

        # Deliberately not cleaned up here -- VideoFromFile streams from this
        # path lazily as the graph executes; the OS temp dir is reclaimed on
        # container restart, matching every other tmp_dir use in this file.
        return (InputImpl.VideoFromFile(tmp_video),)


_BPY_RENDER_SCRIPT = r'''
import bpy, time, sys, os, math, subprocess, shutil
import mathutils

GLTF_PATH, OUT_PATH, WIDTH, HEIGHT, FRAME_COUNT, FPS, ENGINE, SAMPLES = sys.argv[1:9]
WIDTH, HEIGHT, FRAME_COUNT, FPS, SAMPLES = int(WIDTH), int(HEIGHT), int(FRAME_COUNT), int(FPS), int(SAMPLES)

t_start = time.time()
bpy.ops.wm.read_factory_settings(use_empty=True)

t0 = time.time()
bpy.ops.import_scene.gltf(filepath=GLTF_PATH)
t_import = time.time() - t0

scene = bpy.context.scene
min_co = mathutils.Vector((float("inf"),) * 3)
max_co = mathutils.Vector((float("-inf"),) * 3)
for obj in scene.objects:
    if obj.type != "MESH":
        continue
    for corner in obj.bound_box:
        w = obj.matrix_world @ mathutils.Vector(corner)
        min_co.x, min_co.y, min_co.z = min(min_co.x, w.x), min(min_co.y, w.y), min(min_co.z, w.z)
        max_co.x, max_co.y, max_co.z = max(max_co.x, w.x), max(max_co.y, w.y), max(max_co.z, w.z)

center = (min_co + max_co) / 2
size = max_co - min_co
radius = max(size.x, size.y, size.z, 0.01)

cam_data = bpy.data.cameras.new("Cam")
cam_obj = bpy.data.objects.new("Cam", cam_data)
scene.collection.objects.link(cam_obj)
scene.camera = cam_obj
cam_distance = radius * 2.2
cam_obj.location = (center.x + cam_distance * 0.7, center.y - cam_distance * 0.9, center.z + cam_distance * 0.5)
cam_obj.rotation_euler = (center - cam_obj.location).to_track_quat("-Z", "Y").to_euler()

light_data = bpy.data.lights.new("Sun", type="SUN")
light_data.energy = 3.0
light_obj = bpy.data.objects.new("Sun", light_data)
light_obj.rotation_euler = (math.radians(50), 0, math.radians(30))
scene.collection.objects.link(light_obj)

scene.render.engine = ENGINE
gpu_found = False
if ENGINE == "CYCLES":
    scene.cycles.samples = SAMPLES
    scene.cycles.device = "GPU"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "CUDA"
    prefs.get_devices()
    for device in prefs.devices:
        if device.type == "CUDA":
            device.use = True
            gpu_found = True
        else:
            device.use = False

scene.render.resolution_x = WIDTH
scene.render.resolution_y = HEIGHT
scene.render.resolution_percentage = 100
scene.render.fps = FPS
scene.frame_start = 1
scene.frame_end = FRAME_COUNT

frames_dir = OUT_PATH + "_frames"
if os.path.isdir(frames_dir):
    shutil.rmtree(frames_dir)
os.makedirs(frames_dir)
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = frames_dir + "/frame_"

t0 = time.time()
bpy.ops.render.render(animation=True)
t_render = time.time() - t0

t0 = time.time()
mux_ok = True
try:
    subprocess.check_call([
        "ffmpeg", "-y", "-framerate", str(FPS),
        "-i", frames_dir + "/frame_%04d.png",
        "-pix_fmt", "yuv420p", "-c:v", "libx264",
        OUT_PATH,
    ], stderr=subprocess.DEVNULL)
except Exception as e:
    mux_ok = False
t_mux = time.time() - t0

t_total = time.time() - t_start
print("===== BPY RENDER RESULTS =====")
print(f"engine: {ENGINE}")
print(f"gpu_device_found: {gpu_found}")
print(f"resolution: {WIDTH}x{HEIGHT}")
print(f"frames: {FRAME_COUNT} @ {FPS}fps")
print(f"import_time_s: {t_import:.2f}")
print(f"render_time_s: {t_render:.2f}")
print(f"render_time_per_frame_s: {t_render/FRAME_COUNT:.3f}")
print(f"mux_time_s: {t_mux:.2f}")
print(f"mux_ok: {mux_ok}")
print(f"total_time_s: {t_total:.2f}")
print(f"output: {OUT_PATH}")
'''


class BpyRenderTest:
    """Provisions an isolated Python 3.13 (bpy's pip wheels only exist for
    cp311/cp313 -- Graydient's ComfyUI runs 3.12, so bpy cannot be imported
    in-process) inside the ephemeral container, pip-installs bpy into it,
    downloads a glTF test asset, and renders it headless via Blender's Python
    API -- timing every phase separately (env provisioning, bpy import,
    glTF import, render, ffmpeg mux) so the real per-job cost is measured,
    not just render time. Tries EEVEE or CYCLES per the engine widget."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "gltf_url":    ("STRING", {"default": "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/main/Models/Fox/glTF-Binary/Fox.glb"}),
            "width":       ("INT", {"default": 1920, "min": 64, "max": 3840}),
            "height":      ("INT", {"default": 1080, "min": 64, "max": 2160}),
            "frame_count": ("INT", {"default": 48, "min": 1, "max": 4096}),
            "fps":         ("INT", {"default": 24, "min": 1, "max": 60}),
            "engine":      (["BLENDER_EEVEE", "CYCLES"], {"default": "BLENDER_EEVEE"}),
            "samples":     ("INT", {"default": 32, "min": 1, "max": 4096}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION     = "run"
    CATEGORY     = "TripoSG"
    OUTPUT_NODE  = True

    PYTHON_URL = (
        "https://github.com/astral-sh/python-build-standalone/releases/download/"
        "20260901/cpython-3.13.15%2B20260901-x86_64-unknown-linux-gnu-install_only.tar.gz"
    )

    @staticmethod
    async def _run_subprocess(args):
        """Runs a subprocess without blocking the asyncio event loop (unlike
        subprocess.run, which stalls ComfyUI's entire async executor for the
        whole child lifetime -- suspected cause of jobs reporting timed_out
        with no error even though the work itself finished per the logs).
        Launches in its own process group and explicitly kills that group
        afterward so no orphaned Blender/ffmpeg children are left holding
        GPU/file-descriptor resources that could make the container look
        "still busy" to Graydient's own health check after we've returned."""
        import asyncio, os as _os, signal

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        try:
            stdout, stderr = await proc.communicate()
        finally:
            if proc.returncode is None:
                try:
                    _os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                # Reap any stray children left in this process group (e.g.
                # ffmpeg spawned by bpy's own subprocess call) even though
                # the direct child already exited cleanly.
                try:
                    _os.killpg(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        return proc.returncode, stdout.decode(errors="replace"), stderr.decode(errors="replace")

    async def run(self, gltf_url, width, height, frame_count, fps, engine, samples):
        import asyncio, time, tempfile, tarfile, urllib.request

        report_lines = []

        def log(line):
            print(line)
            report_lines.append(line)

        work_dir = os.path.join(tempfile.gettempdir(), "bpy_render_test")
        os.makedirs(work_dir, exist_ok=True)
        py_dir = os.path.join(work_dir, "python3.13")
        py_bin = os.path.join(py_dir, "bin", "python3.13")

        t0 = time.time()
        if not os.path.isfile(py_bin):
            tar_path = os.path.join(work_dir, "python.tar.gz")
            await asyncio.to_thread(urllib.request.urlretrieve, self.PYTHON_URL, tar_path)
            with tarfile.open(tar_path) as tf:
                tf.extractall(work_dir)
            extracted = os.path.join(work_dir, "python")
            if os.path.isdir(extracted) and not os.path.isdir(py_dir):
                os.rename(extracted, py_dir)
        t_python_provision = time.time() - t0

        t0 = time.time()
        bpy_check_rc, _, _ = await self._run_subprocess([py_bin, "-c", "import bpy"])
        bpy_already = bpy_check_rc == 0
        if not bpy_already:
            install_rc, install_out, install_err = await self._run_subprocess(
                [py_bin, "-m", "pip", "install", "--quiet", "bpy"]
            )
            if install_rc != 0:
                log(f"bpy_install_FAILED: rc={install_rc}\n{install_err}")
        t_bpy_install = time.time() - t0

        gltf_path = os.path.join(work_dir, "asset.glb")
        t0 = time.time()
        await asyncio.to_thread(urllib.request.urlretrieve, gltf_url, gltf_path)
        t_gltf_download = time.time() - t0

        script_path = os.path.join(work_dir, "render_test.py")
        with open(script_path, "w") as f:
            f.write(_BPY_RENDER_SCRIPT)

        out_path = os.path.join(work_dir, "out.mp4")
        returncode, stdout, stderr = await self._run_subprocess([
            py_bin, script_path, gltf_path, out_path,
            str(width), str(height), str(frame_count), str(fps), engine, str(samples),
        ])

        log(f"python_provision_time_s: {t_python_provision:.2f} (cached: {os.path.isfile(py_bin) and t_python_provision < 1})")
        log(f"bpy_install_time_s: {t_bpy_install:.2f} (already_installed: {bpy_already})")
        log(f"gltf_download_time_s: {t_gltf_download:.2f}")
        log("----- subprocess stdout -----")
        log(stdout)
        if returncode != 0:
            log("----- subprocess stderr -----")
            log(stderr)

        report = "\n".join(report_lines)
        return {"ui": {"text": [report]}, "result": (report,)}


_BPY_PROCEDURAL_FIELD_SCRIPT = r'''
import bpy, math, sys, time, os, shutil, subprocess

OUT_PATH, WIDTH, HEIGHT, FRAME_COUNT, FPS, GRID_N = sys.argv[1:7]
WIDTH, HEIGHT, FRAME_COUNT, FPS, GRID_N = int(WIDTH), int(HEIGHT), int(FRAME_COUNT), int(FPS), int(GRID_N)

t_start = time.time()
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene

SPACING = 0.6
AMPLITUDE = 1.2
WAVELENGTH = 6.0
RIPPLE_PERIOD_FRAMES = 60
ORBIT_PERIOD_FRAMES = 240

base_mesh = bpy.data.meshes.new("sphere_mesh")
tmp = bpy.data.objects.new("tmp", base_mesh)
scene.collection.objects.link(tmp)
bpy.context.view_layer.objects.active = tmp
bpy.ops.object.select_all(action="DESELECT")
tmp.select_set(True)
bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=1, radius=0.18)
generated = bpy.context.active_object
base_mesh = generated.data
bpy.data.objects.remove(tmp, do_unlink=True)

objs = []
centers = []
half = (GRID_N - 1) / 2.0
for ix in range(GRID_N):
    for iy in range(GRID_N):
        x = (ix - half) * SPACING
        y = (iy - half) * SPACING
        obj = bpy.data.objects.new(f"sph_{ix}_{iy}", base_mesh)
        obj.location = (x, y, 0)
        scene.collection.objects.link(obj)
        objs.append(obj)
        centers.append((x, y, math.sqrt(x * x + y * y)))
bpy.data.objects.remove(generated, do_unlink=True)

mat = bpy.data.materials.new("Field")
mat.use_nodes = True
nt = mat.node_tree
nt.nodes.clear()
obj_info = nt.nodes.new("ShaderNodeObjectInfo")
sep = nt.nodes.new("ShaderNodeSeparateXYZ")
map_range = nt.nodes.new("ShaderNodeMapRange")
map_range.inputs["From Min"].default_value = -AMPLITUDE
map_range.inputs["From Max"].default_value = AMPLITUDE
ramp = nt.nodes.new("ShaderNodeValToRGB")
ramp.color_ramp.elements[0].color = (0.05, 0.05, 0.6, 1)
ramp.color_ramp.elements[1].color = (1.0, 0.25, 0.05, 1)
emission = nt.nodes.new("ShaderNodeEmission")
emission.inputs["Strength"].default_value = 4.0
output = nt.nodes.new("ShaderNodeOutputMaterial")
nt.links.new(obj_info.outputs["Location"], sep.inputs[0])
nt.links.new(sep.outputs["Z"], map_range.inputs["Value"])
nt.links.new(map_range.outputs["Result"], ramp.inputs["Fac"])
nt.links.new(ramp.outputs["Color"], emission.inputs["Color"])
nt.links.new(emission.outputs["Emission"], output.inputs["Surface"])
base_mesh.materials.append(mat)

light_data = bpy.data.lights.new("Sun", type="SUN")
light_data.energy = 1.0
light_obj = bpy.data.objects.new("Sun", light_data)
light_obj.rotation_euler = (math.radians(60), 0, math.radians(20))
scene.collection.objects.link(light_obj)

world = bpy.data.worlds.new("W")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.01, 0.012, 0.02, 1)
scene.world = world

orbit_empty = bpy.data.objects.new("Orbit", None)
scene.collection.objects.link(orbit_empty)
cam_data = bpy.data.cameras.new("Cam")
cam_obj = bpy.data.objects.new("Cam", cam_data)
scene.collection.objects.link(cam_obj)
cam_obj.location = (14, 0, 9)
cam_obj.parent = orbit_empty
scene.camera = cam_obj
target = bpy.data.objects.new("Target", None)
scene.collection.objects.link(target)
con = cam_obj.constraints.new("TRACK_TO")
con.target = target
con.track_axis = "TRACK_NEGATIVE_Z"
con.up_axis = "UP_Y"

def update_frame(scene_):
    f = scene_.frame_current
    t = f / RIPPLE_PERIOD_FRAMES
    for obj, (x, y, dist) in zip(objs, centers):
        z = AMPLITUDE * math.sin(2 * math.pi * (t - dist / WAVELENGTH))
        obj.location.z = z
        s = 0.7 + 0.3 * (z / AMPLITUDE)
        obj.scale = (s, s, s)
    orbit_empty.rotation_euler.z = 2 * math.pi * (f / ORBIT_PERIOD_FRAMES)

bpy.app.handlers.frame_change_pre.append(update_frame)
update_frame(scene)

scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x = WIDTH
scene.render.resolution_y = HEIGHT
scene.render.resolution_percentage = 100
scene.render.fps = FPS
scene.frame_start = 1
scene.frame_end = FRAME_COUNT

frames_dir = OUT_PATH + "_frames"
if os.path.isdir(frames_dir):
    shutil.rmtree(frames_dir)
os.makedirs(frames_dir)
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = frames_dir + "/frame_"

t0 = time.time()
bpy.ops.render.render(animation=True)
t_render = time.time() - t0

t0 = time.time()
mux_ok = True
try:
    subprocess.check_call([
        "ffmpeg", "-y", "-framerate", str(FPS),
        "-i", frames_dir + "/frame_%04d.png",
        "-pix_fmt", "yuv420p", "-c:v", "libx264",
        OUT_PATH,
    ], stderr=subprocess.DEVNULL)
except Exception:
    mux_ok = False
t_mux = time.time() - t0

t_total = time.time() - t_start
print("===== BPY PROCEDURAL FIELD RESULTS =====")
print(f"instances: {GRID_N * GRID_N}")
print(f"resolution: {WIDTH}x{HEIGHT}")
print(f"frames: {FRAME_COUNT} @ {FPS}fps")
print(f"render_time_s: {t_render:.2f}")
print(f"render_time_per_frame_s: {t_render/FRAME_COUNT:.3f}")
print(f"mux_time_s: {t_mux:.2f}")
print(f"mux_ok: {mux_ok}")
print(f"total_time_s: {t_total:.2f}")
print(f"output: {OUT_PATH}")
'''


class BpyProceduralField:
    """Self-contained procedural motion-graphics test: a grid of glowing
    instanced spheres animated as a traveling ripple (frame_change_pre
    handler, no keyframe baking), shared single material driven per-instance
    by each object's own Z location via the Object Info node, camera slowly
    orbiting. No external assets, no diffusion -- pure algorithmic content,
    proving the 'native Blender rendering' half of the render pipeline can
    stand on its own. Outputs a real VIDEO (unlike BpyRenderTest/
    SystemDiagnostics, which only return text reports)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "width":       ("INT", {"default": 1280, "min": 64, "max": 3840}),
            "height":      ("INT", {"default": 720, "min": 64, "max": 2160}),
            "frame_count": ("INT", {"default": 240, "min": 1, "max": 4096}),
            "fps":         ("INT", {"default": 24, "min": 1, "max": 60}),
            "grid_n":      ("INT", {"default": 16, "min": 2, "max": 64}),
        }}

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "report")
    FUNCTION     = "run"
    CATEGORY     = "TripoSG"
    OUTPUT_NODE  = True

    PYTHON_URL = BpyRenderTest.PYTHON_URL
    _run_subprocess = staticmethod(BpyRenderTest._run_subprocess)

    async def run(self, width, height, frame_count, fps, grid_n):
        import asyncio, time, tempfile, tarfile, urllib.request
        from comfy_api.latest import InputImpl

        report_lines = []

        def log(line):
            print(line)
            report_lines.append(line)

        work_dir = os.path.join(tempfile.gettempdir(), "bpy_render_test")
        os.makedirs(work_dir, exist_ok=True)
        py_dir = os.path.join(work_dir, "python3.13")
        py_bin = os.path.join(py_dir, "bin", "python3.13")

        t0 = time.time()
        if not os.path.isfile(py_bin):
            tar_path = os.path.join(work_dir, "python.tar.gz")
            await asyncio.to_thread(urllib.request.urlretrieve, self.PYTHON_URL, tar_path)
            with tarfile.open(tar_path) as tf:
                tf.extractall(work_dir)
            extracted = os.path.join(work_dir, "python")
            if os.path.isdir(extracted) and not os.path.isdir(py_dir):
                os.rename(extracted, py_dir)
        t_python_provision = time.time() - t0

        t0 = time.time()
        bpy_check_rc, _, _ = await self._run_subprocess([py_bin, "-c", "import bpy"])
        bpy_already = bpy_check_rc == 0
        if not bpy_already:
            install_rc, install_out, install_err = await self._run_subprocess(
                [py_bin, "-m", "pip", "install", "--quiet", "bpy"]
            )
            if install_rc != 0:
                log(f"bpy_install_FAILED: rc={install_rc}\n{install_err}")
        t_bpy_install = time.time() - t0

        script_path = os.path.join(work_dir, "procedural_field.py")
        with open(script_path, "w") as f:
            f.write(_BPY_PROCEDURAL_FIELD_SCRIPT)

        out_path = os.path.join(work_dir, "procedural_field.mp4")
        returncode, stdout, stderr = await self._run_subprocess([
            py_bin, script_path, out_path,
            str(width), str(height), str(frame_count), str(fps), str(grid_n),
        ])

        log(f"python_provision_time_s: {t_python_provision:.2f}")
        log(f"bpy_install_time_s: {t_bpy_install:.2f} (already_installed: {bpy_already})")
        log("----- subprocess stdout -----")
        log(stdout)
        if returncode != 0:
            log("----- subprocess stderr -----")
            log(stderr)

        report = "\n".join(report_lines)
        video = InputImpl.VideoFromFile(out_path) if returncode == 0 and os.path.isfile(out_path) else None
        return {"ui": {"text": [report]}, "result": (video, report)}


class SystemDiagnostics:
    """Runs a battery of read-only shell/environment checks relevant to evaluating whether
    a Windows/D3D12-only tool (NVIDIA NGX, Wine/Proton, VKD3D-Proton, DXVK-NVAPI) could be
    made to run on this Graydient container. Everything here is print()'d during execution
    so it lands in the job's stdout log even with no downstream node attached, and also
    returned as one STRING for optional SaveText/output-node use."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION     = "run"
    CATEGORY     = "TripoSG"
    OUTPUT_NODE  = True

    def run(self):
        import subprocess, os, shutil, glob

        def sh(cmd):
            try:
                out = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=15)
                return (out.stdout + out.stderr).strip() or "(empty)"
            except Exception as e:
                return f"(error: {e})"

        sections = []

        sections.append(("nvidia-smi", sh("nvidia-smi")))
        sections.append(("nvidia driver version",
                          sh("cat /proc/driver/nvidia/version 2>/dev/null || nvidia-smi --query-gpu=driver_version --format=csv,noheader")))
        sections.append(("libnvidia-ngx.so present",
                          sh("ldconfig -p | grep -i ngx || find / -iname 'libnvidia-ngx*' 2>/dev/null")))
        sections.append(("vulkan support",
                          sh("vulkaninfo --summary 2>&1 | head -40 || echo 'vulkaninfo not installed'")))
        sections.append(("wine present", sh("which wine wine64 2>/dev/null || echo 'not found'")))
        sections.append(("dxvk/vkd3d-proton on disk",
                          sh("find / -iname '*vkd3d*' -o -iname '*dxvk*' 2>/dev/null | head -20")))
        sections.append(("kernel / distro", sh("uname -a && cat /etc/os-release 2>/dev/null")))
        sections.append(("gpu render nodes", sh("ls -la /dev/dri 2>/dev/null")))
        sections.append(("free disk on /", sh("df -h / 2>/dev/null")))
        sections.append(("apt/dpkg present", sh("which apt-get dpkg 2>/dev/null || echo 'none'")))

        report_lines = []
        for title, body in sections:
            report_lines.append(f"===== {title} =====\n{body}\n")
        report = "\n".join(report_lines)

        print(report)
        return {"ui": {"text": [report]}, "result": (report,)}


class EncodeStringAsImage:
    """Encode a STRING as a lossless RGB data IMAGE for Graydient output.
    ForgeExpress decodes it by reading pixel values back to UTF-8 bytes."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"multiline": True, "default": ""})}}

    RETURN_TYPES  = ("IMAGE",)
    RETURN_NAMES  = ("data_image",)
    FUNCTION      = "encode"
    CATEGORY      = "TripoSG"

    def encode(self, text):
        return (_encode_string_as_image(text),)


class AudioAnalyze:
    """Download audio from URL, compute energy timeline / BPM / beats / section boundaries.
    Returns analysis_json (STRING), annotated spectrogram (IMAGE), encoded data (IMAGE).
    Dependencies: librosa, matplotlib (both pip-installable on Graydient)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"url": ("STRING", {"default": ""})}}

    RETURN_TYPES  = ("STRING", "IMAGE", "IMAGE")
    RETURN_NAMES  = ("analysis_json", "spectrogram", "analysis_data")
    FUNCTION      = "analyze"
    CATEGORY      = "TripoSG"

    def analyze(self, url: str):
        import json
        import os
        import shutil
        import tempfile

        import requests
        import torch

        try:
            import librosa
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.gridspec as gridspec
            import matplotlib.pyplot as plt
        except ImportError:
            raise RuntimeError("librosa / matplotlib not installed. Add them to pip requirements.")

        # ── Resolve audio source (same logic as TranscribeAudioFromURL) ─────────
        url_clean = url.strip()
        tmp_dir   = tempfile.mkdtemp()
        ext       = os.path.splitext(url_clean.split("?")[0])[-1] or ".mp3"
        tmp_path  = os.path.join(tmp_dir, f"audio{ext}")
        try:
            if url_clean.startswith(("http://", "https://")):
                resp = requests.get(url_clean, timeout=120, stream=True)
                resp.raise_for_status()
                with open(tmp_path, "wb") as fh:
                    for chunk in resp.iter_content(65536):
                        fh.write(chunk)
            else:
                import folder_paths as _fp
                candidate = os.path.join(_fp.get_input_directory(), url_clean)
                if os.path.isfile(candidate):
                    tmp_path = candidate
                elif os.path.isfile(url_clean):
                    tmp_path = url_clean
                else:
                    raise FileNotFoundError(f"Audio file not found: {url_clean!r}")

            # ── Audio analysis ────────────────────────────────────────────────
            y, sr    = librosa.load(tmp_path, sr=22050, mono=True)
            duration = float(librosa.get_duration(y=y, sr=sr))
            hop      = 512

            # BPM + beats — np.asarray().item() handles both scalar and 0-d array
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
            beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=hop).tolist()
            bpm = float(np.asarray(tempo).item())

            # RMS energy (normalised 0-1)
            rms_raw = librosa.feature.rms(y=y, hop_length=hop)[0]
            rms_max = float(rms_raw.max()) or 1.0
            frame_times = librosa.frames_to_time(
                np.arange(len(rms_raw)), sr=sr, hop_length=hop
            )

            energy_timeline = [
                {"time": round(float(t), 2), "energy": round(float(e) / rms_max, 3)}
                for t, e in zip(frame_times, rms_raw)
                if float(t) <= duration
            ]

            # Section segmentation (MFCC recurrence matrix)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=hop)
            R    = librosa.segment.recurrence_matrix(mfcc, mode="affinity", sym=True)
            seg_frames = librosa.segment.agglomerative(R, k=min(8, R.shape[0] - 1))
            section_times = sorted(set(
                round(float(t), 2)
                for t in librosa.frames_to_time(seg_frames, sr=sr, hop_length=hop)
                if 0 < float(t) < duration
            ))

            analysis = {
                "duration":         round(duration, 2),
                "bpm":              round(bpm, 1),
                "energy_timeline":  energy_timeline,
                "beat_times":       [round(t, 3) for t in beat_times],
                "section_times":    section_times,
            }
            analysis_json = json.dumps(analysis, ensure_ascii=False)

            # ── Annotated spectrogram ─────────────────────────────────────────
            fig = plt.figure(figsize=(16, 8), facecolor="#0b0e12")
            gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.12)

            ax1 = fig.add_subplot(gs[0])
            S   = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=hop)
            librosa.display.specshow(
                librosa.power_to_db(S, ref=np.max),
                sr=sr, hop_length=hop, x_axis="time", y_axis="mel",
                ax=ax1, cmap="magma",
            )
            ax1.set_facecolor("#0b0e12")
            ax1.tick_params(colors="#4a5168")
            ax1.set_ylabel("Hz", color="#4a5168", fontsize=8)
            ax1.set_xlabel("")
            for bt in beat_times:
                ax1.axvline(bt, color="#6c47ff", alpha=0.25, linewidth=0.5)
            for st in section_times:
                ax1.axvline(st, color="#5bc8e0", alpha=0.8, linewidth=1.5, linestyle="--")
            ax1.set_title(f"BPM {bpm:.1f}  ·  dur {duration:.1f}s  ·  {len(section_times)} sections",
                          color="#9099a8", fontsize=9, pad=4)

            ax2 = fig.add_subplot(gs[1])
            ax2.set_facecolor("#0b0e12")
            rms_norm = rms_raw / rms_max
            ax2.fill_between(frame_times[:len(rms_norm)], rms_norm,
                             color="#6c47ff", alpha=0.5)
            ax2.plot(frame_times[:len(rms_norm)], rms_norm,
                     color="#6c47ff", linewidth=0.8)
            for st in section_times:
                ax2.axvline(st, color="#5bc8e0", alpha=0.8, linewidth=1.5, linestyle="--")
            ax2.set_ylabel("Energy", color="#4a5168", fontsize=8)
            ax2.set_xlabel("Time (s)", color="#4a5168", fontsize=8)
            ax2.tick_params(colors="#4a5168")
            ax2.set_xlim(0, duration)
            ax2.set_ylim(0, 1.05)

            plt.tight_layout(pad=0.4)
            fig.canvas.draw()
            w, h  = fig.canvas.get_width_height()
            buf   = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
            plt.close(fig)

            spec_t   = torch.from_numpy(buf.astype(np.float32) / 255.0).unsqueeze(0)
            encoded  = _encode_string_as_image(analysis_json)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        return (analysis_json, spec_t, encoded)


class LoadAudioFromURLStereo:
    """Download audio from URL and decode as full-fidelity stereo AUDIO.

    Unlike ComfyUI-MisoTTS's LoadAudioFromURL (hardcoded mono/24kHz, tuned for
    voice cloning), this preserves stereo channels and a configurable sample
    rate — required for Matchering, which needs the target's real stereo image
    and headroom, not a downsampled mono copy."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "url": ("STRING", {"default": ""}),
                "sample_rate": ("INT", {"default": 44100, "min": 8000, "max": 192000, "step": 1}),
            },
            "optional": {
                # Graydient's init_<type>_* is a {bool, filename, url} triplet and
                # it's unconfirmed which of filename/url gets populated for a given
                # submission path -- accept both, first non-empty wins.
                "filename": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "load"
    CATEGORY = "TripoSG"

    def load(self, url, sample_rate, filename=""):
        import subprocess
        import tempfile
        import urllib.request

        import numpy as np
        import torch

        source = (url or "").strip() or (filename or "").strip()
        if not source:
            raise RuntimeError("LoadAudioFromURLStereo: no url or filename supplied")

        if source.startswith("http://") or source.startswith("https://"):
            with urllib.request.urlopen(source) as resp:
                data = resp.read()
        else:
            # Bare filename -- Graydient pre-staged this into ComfyUI's input/ dir.
            # urllib.request.urlopen() would raise "unknown url type" on this, the
            # exact anti-pattern documented elsewhere in this file for LoadImage/
            # LoadVideoFromURL -- read it directly instead of fetching it.
            local_path = folder_paths.get_annotated_filepath(source)
            if not os.path.isfile(local_path):
                raise FileNotFoundError(f"local audio upload not found: {source!r}")
            with open(local_path, "rb") as fh:
                data = fh.read()

        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(source.split("?")[0])[1] or ".bin") as tmp:
            tmp.write(data)
            tmp.flush()
            cmd = [
                "ffmpeg", "-v", "error", "-i", tmp.name,
                "-f", "s16le", "-acodec", "pcm_s16le",
                "-ar", str(sample_rate), "-ac", "2", "-",
            ]
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if proc.returncode != 0:
                raise RuntimeError(f"ffmpeg failed to decode audio from {source}: {proc.stderr.decode(errors='replace')}")

        arr = np.frombuffer(proc.stdout, dtype="<i2").astype("float32") / 32768.0
        arr = arr.reshape(-1, 2).T  # [channels=2, samples]
        waveform = torch.from_numpy(arr.copy())
        return ({"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate},)


# Node registration — URL loaders always available; 3D nodes require full pip deps
NODE_CLASS_MAPPINGS = {
    "LoadImageFromURL":       LoadImageFromURL,
    "TripoSGLoadVideoFromURL": TripoSGLoadVideoFromURL,
    "LoadVideoFromURLAuto":   LoadVideoFromURLAuto,
    "TranscribeAudioFromURL": TranscribeAudioFromURL,
    "BurnSubtitlesFromTimeline": BurnSubtitlesFromTimeline,
    "HFTextGenerate":         HFTextGenerate,
    "VLMInferFromURL":        VLMInferFromURL,
    "EncodeStringAsImage":    EncodeStringAsImage,
    "ConcatStrings":          ConcatStrings,
    "AudioAnalyze":           AudioAnalyze,
    "LoadAudioFromURLStereo": LoadAudioFromURLStereo,
    "SystemDiagnostics":      SystemDiagnostics,
    "LoadVideoFromURLAsVideo": LoadVideoFromURLAsVideo,
    "BpyRenderTest":          BpyRenderTest,
    "BpyProceduralField":     BpyProceduralField,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageFromURL":       "Load Image From URL",
    "TripoSGLoadVideoFromURL": "TripoSG Load Video From URL",
    "LoadVideoFromURLAuto":   "Load Video From URL (Auto Res)",
    "TranscribeAudioFromURL": "Transcribe Audio From URL",
    "BurnSubtitlesFromTimeline": "Burn Subtitles From Timeline",
    "HFTextGenerate":         "HF Text Generate",
    "VLMInferFromURL":        "VLM Infer From URL",
    "EncodeStringAsImage":    "Encode String As Image",
    "ConcatStrings":          "Concat Strings",
    "AudioAnalyze":           "Audio Analyze",
    "LoadAudioFromURLStereo": "Load Audio From URL (Stereo)",
    "SystemDiagnostics":      "System Diagnostics (DLSS5 feasibility)",
    "LoadVideoFromURLAsVideo": "Load Video From URL (as VIDEO)",
    "BpyRenderTest":          "Bpy Render Test (Blender headless render-time budget)",
    "BpyProceduralField":     "Bpy Procedural Field (ripple field demo render)",
}

if _TRIPOSG_AVAILABLE:
    NODE_CLASS_MAPPINGS.update({
        "TripoSGModelLoader": TripoSGModelLoader,
        "TripoSGInference": TripoSGInference,
        "TripoSGPrepareImage": TripoSGPrepareImage,
        "TripoSGConditioning": TripoSGScribbleConditioningNode,
        "PartCrafterConditioning": PartCrafterConditioningNode,
        "PartCrafterModelSelect": PartCrafterModelSelect,
        "SimplifyMeshFacesSelect": SimplifyMeshFacesSelect,
        "SimplifyMesh": SimplifyMesh,
        "MESHToTrimesh": MESHToTrimesh,
        "TrimeshToMESH": TrimeshToMESH,
        "SaveTrimesh": SaveTrimesh,
        "BakeVertexColorsFromViews": BakeVertexColorsFromViews,
        "UVUnwrapAndBakeTexture": UVUnwrapAndBakeTexture,
    })
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "TripoSGModelLoader": "TripoSG Model Loader",
        "TripoSGInference": "TripoSG Inference",
        "TripoSGConditioning": "TripoSG Scribble Conditioning",
        "PartCrafterConditioning": "PartCrafter Conditioning",
        "PartCrafterModelSelect": "PartCrafter Model Select (slot toggle)",
        "SimplifyMeshFacesSelect": "Simplify Mesh Faces Select (slot toggle)",
        "TripoSGPrepareImage": "TripoSG Prepare Image",
        "SimplifyMesh": "Simplify Mesh",
        "MESHToTrimesh": "Mesh to Trimesh",
        "TrimeshToMESH": "Trimesh to Mesh",
        "SaveTrimesh": "Save Trimesh",
        "BakeVertexColorsFromViews": "Bake Vertex Colors From Views",
        "UVUnwrapAndBakeTexture": "UV Unwrap and Bake Texture (PBR)",
    })
