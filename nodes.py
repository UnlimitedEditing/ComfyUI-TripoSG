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
            }
        }

    RETURN_TYPES = ("TRIPOSG",)
    FUNCTION = "load_model"
    CATEGORY = "TripoSG"

    def load_model(self, model):
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

        if not url.strip():
            # No URL supplied — return a blank white 512×512 image so
            # text-only runs (e.g. TripoSG-scribble with prompt only) don't crash.
            blank = np.ones((512, 512, 3), dtype=np.float32)
            image = torch.from_numpy(blank).unsqueeze(0)
            mask  = torch.zeros((1, 512, 512), dtype=torch.float32)
            return (image, mask)

        response = requests.get(url.strip(), timeout=30)
        response.raise_for_status()

        img = Image.open(BytesIO(response.content)).convert("RGBA")
        arr = np.array(img).astype(np.float32) / 255.0

        # IMAGE: [1, H, W, 3] RGB float32 0–1
        image = torch.from_numpy(arr[:, :, :3]).unsqueeze(0)
        # MASK: [1, H, W] ComfyUI convention — 0 = opaque, 1 = transparent
        mask = torch.from_numpy(1.0 - arr[:, :, 3]).unsqueeze(0)

        return (image, mask)


class LoadVideoFromURL:
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

        if not url.strip():
            blank = np.zeros((max_frames, height, width, 3), dtype=np.float32)
            return (torch.from_numpy(blank),)

        tmp_dir    = tempfile.mkdtemp()
        tmp_video  = os.path.join(tmp_dir, "input.mp4")
        frames_dir = os.path.join(tmp_dir, "frames")
        os.makedirs(frames_dir)

        try:
            resp = requests.get(url.strip(), timeout=120, stream=True)
            resp.raise_for_status()
            with open(tmp_video, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=65536):
                    fh.write(chunk)

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
            }
        }

    RETURN_TYPES  = ("IMAGE", "STRING")
    RETURN_NAMES  = ("lyrics_card", "lyrics_json")
    FUNCTION      = "transcribe"
    CATEGORY      = "TripoSG"

    def transcribe(self, url, model_size="large-v3", language=""):
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

        # ── Download audio ────────────────────────────────────────────────────
        tmp_dir  = tempfile.mkdtemp()
        # Keep original extension so ffmpeg/whisper can sniff format
        ext      = os.path.splitext(url.strip().split("?")[0])[-1] or ".mp3"
        tmp_path = os.path.join(tmp_dir, f"audio{ext}")

        try:
            resp = requests.get(url.strip(), timeout=120, stream=True)
            resp.raise_for_status()
            with open(tmp_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=65536):
                    fh.write(chunk)

            # ── Transcribe ────────────────────────────────────────────────────
            device       = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            model = WhisperModel(model_size, device=device, compute_type=compute_type)

            lang   = language.strip() if language.strip() else None
            raw, info = model.transcribe(
                tmp_path,
                language               = lang,
                beam_size              = 5,
                word_timestamps        = False,
                no_speech_threshold    = 0.6,
                temperature            = 0,     # greedy — no hallucination fallback
                condition_on_previous_text = False,
                # VAD pre-filters music/silence so Whisper only sees real speech.
                # This prevents the 2-minute context-overflow hallucination cliff
                # and naturally produces instrumental gaps without a manual scanner.
                vad_filter             = True,
                vad_parameters         = dict(
                    min_silence_duration_ms = 500,  # gaps < 500ms stay joined
                    speech_pad_ms           = 200,  # pad each speech window by 200ms
                ),
            )
            lyrical = [
                {"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()}
                for s in raw if s.text.strip()
            ]

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


# Node registration — URL loaders always available; 3D nodes require full pip deps
NODE_CLASS_MAPPINGS = {
    "LoadImageFromURL":       LoadImageFromURL,
    "LoadVideoFromURL":       LoadVideoFromURL,
    "TranscribeAudioFromURL": TranscribeAudioFromURL,
    "HFTextGenerate":         HFTextGenerate,
    "VLMInferFromURL":        VLMInferFromURL,
    "EncodeStringAsImage":    EncodeStringAsImage,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadImageFromURL":       "Load Image From URL",
    "LoadVideoFromURL":       "Load Video From URL",
    "TranscribeAudioFromURL": "Transcribe Audio From URL",
    "HFTextGenerate":         "HF Text Generate",
    "VLMInferFromURL":        "VLM Infer From URL",
    "EncodeStringAsImage":    "Encode String As Image",
}

if _TRIPOSG_AVAILABLE:
    NODE_CLASS_MAPPINGS.update({
        "TripoSGModelLoader": TripoSGModelLoader,
        "TripoSGInference": TripoSGInference,
        "TripoSGPrepareImage": TripoSGPrepareImage,
        "TripoSGConditioning": TripoSGScribbleConditioningNode,
        "PartCrafterConditioning": PartCrafterConditioningNode,
        "SimplifyMesh": SimplifyMesh,
        "MESHToTrimesh": MESHToTrimesh,
        "TrimeshToMESH": TrimeshToMESH,
        "SaveTrimesh": SaveTrimesh,
        "BakeVertexColorsFromViews": BakeVertexColorsFromViews,
    })
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "TripoSGModelLoader": "TripoSG Model Loader",
        "TripoSGInference": "TripoSG Inference",
        "TripoSGConditioning": "TripoSG Scribble Conditioning",
        "PartCrafterConditioning": "PartCrafter Conditioning",
        "TripoSGPrepareImage": "TripoSG Prepare Image",
        "SimplifyMesh": "Simplify Mesh",
        "MESHToTrimesh": "Mesh to Trimesh",
        "TrimeshToMESH": "Trimesh to Mesh",
        "SaveTrimesh": "Save Trimesh",
        "BakeVertexColorsFromViews": "Bake Vertex Colors From Views",
    })
