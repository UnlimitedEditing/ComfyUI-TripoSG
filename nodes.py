import os
import cv2
import numpy as np
import torch
import trimesh as Trimesh
from io import BytesIO
from pathlib import Path
from huggingface_hub import snapshot_download
from PIL import Image
from typing import Dict, Any

import folder_paths
import comfy.utils
import comfy.model_management as mm
from comfy_extras.nodes_hunyuan3d import MESH

# Import pipeline classes
from .triposg.pipelines.pipeline_triposg import TripoSGPipeline
from .triposg.pipelines.pipeline_triposg_scribble import TripoSGScribblePipeline
from .partcrafter.pipelines.pipeline_partcrafter import PartCrafterPipeline
from .partcrafter.utils.data_utils import get_colored_mesh_composition, scene_to_parts, load_surfaces

gpu = mm.get_torch_device()
cpu = torch.device("cpu")


def pil2numpy(image: Image.Image):
    return np.array(image).astype(np.float32) / 255.0


def numpy2pil(image: np.ndarray, mode=None):
    return Image.fromarray(np.clip(255.0 * image, 0, 255).astype(np.uint8), mode)


def pil2tensor(image: Image.Image):
    return torch.from_numpy(pil2numpy(image)).unsqueeze(0)


def tensor2pil(image: torch.Tensor, mode=None):
    return numpy2pil(image.cpu().numpy().squeeze(), mode=mode)


def simplify_mesh(mesh: MESH, n_faces: int):
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

            if not conditioning.prompt:
                raise ValueError("Prompt is required for TripoSGScribblePipeline")

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
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "filename_prefix": ("STRING", {"default": "3D/TripoSG"}),
                "file_format": (["glb", "obj", "ply", "stl", "3mf", "dae"],),
            },
            "optional": {
                "save_file": ("BOOLEAN", {"default": True, "label_on": "output", "label_off": "temp"}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)
    FUNCTION = "process"
    CATEGORY = "TripoSG"
    OUTPUT_NODE = True
    DESCRIPTION = "Export trimesh object to model file"

    def process(self, trimesh, filename_prefix, file_format, save_file=True):
        save_dir = folder_paths.get_output_directory() if save_file else folder_paths.get_temp_directory()
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
            filename_prefix, save_dir
        )
        output_glb_path = Path(full_output_folder, f"{filename}_{counter:05}_.{file_format}")
        output_glb_path.parent.mkdir(exist_ok=True)

        trimesh.export(output_glb_path, file_type=file_format)
        relative_path = Path(subfolder) / f"{filename}_{counter:05}_.{file_format}"

        return (str(relative_path),)


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

    def bake(self, trimesh, front_image, back_image=None, cam_dist=2.5):
        verts   = trimesh.vertices        # (N, 3)
        normals = trimesh.vertex_normals  # (N, 3) — auto-computed

        def to_u8(t):
            return (t[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        front_np = to_u8(front_image)
        back_np  = to_u8(back_image) if back_image is not None \
                   else front_np[:, ::-1, :].copy()   # mirror front as fallback

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


# Node registration
NODE_CLASS_MAPPINGS = {
    "TripoSGModelLoader": TripoSGModelLoader,
    "TripoSGInference": TripoSGInference,
    "TripoSGPrepareImage": TripoSGPrepareImage,
    "TripoSGConditioning": TripoSGScribbleConditioningNode,
    "PartCrafterConditioning": PartCrafterConditioningNode,
    "SimplifyMesh": SimplifyMesh,
    "MESHToTrimesh": MESHToTrimesh,
    "TrimeshToMESH": TrimeshToMESH,
    "SaveTrimesh": SaveTrimesh,
    "LoadImageFromURL": LoadImageFromURL,
    "BakeVertexColorsFromViews": BakeVertexColorsFromViews,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TripoSGModelLoader": "TripoSG Model Loader",
    "TripoSGInference": "TripoSG Inference",
    "TripoSGConditioning": "TripoSG Scribble Conditioning",
    "PartCrafterConditioning": "PartCrafter Conditioning",
    "TripoSGPrepareImage": "TripoSG Prepare Image",
    "SimplifyMesh": "Simplify Mesh",
    "MESHToTrimesh": "Mesh to Trimesh",
    "TrimeshToMESH": "Trimesh to Mesh",
    "SaveTrimesh": "Save Trimesh",
    "LoadImageFromURL": "Load Image From URL",
    "BakeVertexColorsFromViews": "Bake Vertex Colors From Views",
}
