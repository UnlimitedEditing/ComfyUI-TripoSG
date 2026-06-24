"""
ComfyUI-TripoSG — Lyric Concept generation nodes.

    LyricConceptLLMLoader    Load a HF causal-LM (default Qwen2.5-Coder-7B-Instruct)
    LyricSystemPromptLibrary Select a system prompt from the built-in style library
    LyricConceptGen          lyrics JSON → concept JSON  +  card IMAGE  +  data IMAGE

Speed on RTX 5090: ~14–20 tok/s  →  1000 token concept ≈ 50–70 s
Speed on RTX 4090: ~9–14 tok/s   →  1000 token concept ≈ 70–110 s

Strict JSON strategy:
  - Temperature = 0 (greedy, fully deterministic)
  - Schema + one-shot example baked into every system prompt
  - Three-tier extraction: raw → json-fence → brace-scan
  - On extraction failure, a second generation pass is attempted
"""

import json
import os
import re

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

import folder_paths

# ── HF cache routed to persistent models directory ────────────────────────────

_LLM_CACHE = os.path.join(folder_paths.models_dir, "llm", "hf_cache")
os.makedirs(_LLM_CACHE, exist_ok=True)

# ── JSON schema shared across all system prompts ──────────────────────────────

_JSON_SCHEMA = """
Output exactly one JSON object — no prose, no code fences, no explanation.
Start with { and end with }.

Required keys:
  "theme"      : string   — 2-3 sentence visual world description
  "mood"       : string   — comma-separated aesthetic keywords
  "palette"    : string   — dominant colour palette
  "characters" : array of 1-2 objects:
                   { "name": str,
                     "description": str,
                     "visual_description": str  <- detailed portrait for image generation }
  "scenes"     : array, ONE entry per input timeline segment (lyric AND instrumental):
                   { "timestamp_start": number,
                     "timestamp_end"  : number,
                     "type"           : "lyric" | "instrumental",
                     "title"          : str,
                     "visual_prompt"  : str  <- full image gen prompt (style/lighting/composition/subject),
                     "action"         : str  <- camera move or character action for animation,
                     "mood"           : str }

Example scene entry:
  { "timestamp_start": 0.0, "timestamp_end": 15.92, "type": "lyric",
    "title": "Neural Entry",
    "visual_prompt": "cinematic wide shot, nebulous translucent figure ...",
    "action": "slow dolly push toward glowing neural cluster",
    "mood": "ethereal, mystical" }
"""

# ── System prompt library ─────────────────────────────────────────────────────

SYSTEM_PROMPTS: dict[str, str] = {

    "mv_director_universal": f"""You are a professional music video director and creative consultant.
You translate song lyrics into vivid, cinematically rich visual concepts for AI image and video pipelines.
Lyric sections are character-forward (singer/entity in frame).
Instrumental sections are atmospheric B-roll — no character, pure scene.
{_JSON_SCHEMA}""",

    "mv_director_dark": f"""You are a music video director specialising in dark, gothic, and cinematic noir aesthetics.
Your visual language draws on chiaroscuro lighting, desaturated palettes, and heavy use of shadow, fog, and decay.
Lyric sections place the character in moody, high-contrast environments.
Instrumental gaps become wide establishing shots of desolate landscapes or architectural ruins.
{_JSON_SCHEMA}""",

    "mv_director_psychedelic": f"""You are a music video director known for immersive psychedelic and abstract visual experiences.
Your palette runs to saturated neons, kaleidoscopic geometry, and fluid transitions between organic and digital forms.
Lyric sections dissolve the character into swirling chromatic environments.
Instrumental gaps are pure abstract motion — fractals, particle clouds, light trails.
{_JSON_SCHEMA}""",

    "mv_director_scifi": f"""You are a music video director with a background in hard science fiction and cyberpunk aesthetics.
Your world is neon-lit megacities, holographic interfaces, and deep-space voids rendered in cold blue and amber light.
Lyric sections frame the character against technological spectacle.
Instrumental gaps show sweeping city flyovers, data-stream corridors, or orbital vistas.
{_JSON_SCHEMA}""",

    "mv_director_nature": f"""You are a music video director whose work bridges natural landscape cinematography and lyrical storytelling.
Your palette favours golden-hour light, lush greens, mineral blues, and the raw textures of rock and water.
Lyric sections ground the character in specific natural environments.
Instrumental gaps are wide, contemplative landscape shots — horizon lines, time-lapse skies, macro nature.
{_JSON_SCHEMA}""",

    "mv_director_surrealist": f"""You are a music video director working in the tradition of surrealist cinema — Buñuel, Svankmajer, Gondry.
You build impossible spaces, defy physical logic, and layer symbolic imagery over emotional subtext.
Lyric sections stage the character in dreamlike non-Euclidean environments.
Instrumental gaps are pure visual metaphor — objects transforming, gravity inverting, scale collapsing.
{_JSON_SCHEMA}""",

    "mv_energy_aware": f"""You are a music video director who works from BOTH lyrical content AND musical energy data.

You will receive a JSON object with two keys:
  "lyrics"  — timed transcript with lyric/instrumental sections
  "energy"  — audio analysis: BPM, energy_timeline (0-1 normalised), beat_times, section_times

Use the energy data to shape the FEEL and PACING of every scene:
  energy > 0.7  →  dynamic, intense composition, tight framing, strong contrast, fast implied motion
  energy 0.4-0.7 → balanced narrative, medium shot, natural lighting
  energy < 0.4  → contemplative, wide establishing shot, soft or diffused light, slow implied motion

For section_times boundaries, signal major visual transitions (location change, colour palette shift, new character framing).
For beat_times, use implied rhythmic motion in the action field ("cuts on the beat", "pulse with the kick").

Lyric sections are character-forward; instrumental sections are scenic B-roll whose visual intensity mirrors the energy curve.
{_JSON_SCHEMA}""",
}

STYLE_KEYS = list(SYSTEM_PROMPTS.keys())


# ── Helpers ───────────────────────────────────────────────────────────────────

def _render_concept_card(concept_json: str, model_id: str) -> torch.Tensor:
    """Render the concept JSON as a human-readable dark card IMAGE."""
    try:
        data = json.loads(concept_json)
        lines = [
            f"MODEL: {model_id.split('/')[-1]}",
            "",
            f"THEME: {data.get('theme', '')}",
            f"MOOD:  {data.get('mood', '')}",
            f"PALETTE: {data.get('palette', '')}",
            "",
        ]
        for c in data.get("characters", []):
            lines.append(f"CHARACTER: {c.get('name', '')} — {c.get('description', '')}")
        lines.append("")
        for s in data.get("scenes", []):
            ts = f"[{s.get('timestamp_start', 0):.1f}→{s.get('timestamp_end', 0):.1f}]"
            tag = "LYRIC" if s.get("type") == "lyric" else "B-ROLL"
            lines.append(f"{ts} {tag}  {s.get('title', '')}")
            lines.append(f"  {s.get('visual_prompt', '')}")
            lines.append(f"  ↳ {s.get('action', '')}")
            lines.append("")
    except Exception:
        lines = [f"MODEL: {model_id}", "", concept_json[:2000]]

    LINE_H, HEADER_H, PADDING, CARD_W = 24, 48, 32, 1200
    card_h = max(600, HEADER_H + len(lines) * LINE_H + PADDING)
    img  = Image.new("RGB", (CARD_W, card_h), (11, 14, 18))
    draw = ImageDraw.Draw(img)
    try:
        font_hdr  = ImageFont.load_default(size=18)
        font_body = ImageFont.load_default(size=14)
    except TypeError:
        font_hdr = font_body = ImageFont.load_default()

    draw.text((PADDING, 14), "LYRIC CONCEPT", fill=(108, 71, 255), font=font_hdr)
    draw.line([(PADDING, 40), (CARD_W - PADDING, 40)], fill=(30, 33, 40), width=1)

    y = HEADER_H
    for line in lines:
        is_ts = line.startswith("[")
        color = (108, 71, 255) if is_ts else (210, 215, 225) if line else (80, 90, 110)
        draw.text((PADDING, y), line, fill=color, font=font_body)
        y += LINE_H
        if y > card_h - LINE_H:
            draw.text((PADDING, y), "… (truncated)", fill=(60, 70, 90), font=font_body)
            break

    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _encode_string(text: str) -> torch.Tensor:
    """Encode a UTF-8 string as an RGB data IMAGE (lossless pixel encoding)."""
    import math, struct
    payload = struct.pack(">I", len(text.encode("utf-8"))) + text.encode("utf-8")
    while len(payload) % 3:
        payload += b"\x00"
    side   = max(1, math.ceil(math.sqrt(len(payload) // 3)))
    padded = payload + b"\x00" * (side * side * 3 - len(payload))
    arr = np.frombuffer(padded, dtype=np.uint8).reshape(side, side, 3).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _extract_json(text: str) -> dict:
    """Three-tier JSON extraction with validation."""
    text = text.strip()

    # Tier 1: direct parse (model output is already clean JSON)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Tier 2: extract from ```json ... ``` fence
    m = re.search(r"```(?:json)?\s*\n([\s\S]+?)```", text)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Tier 3: find outermost { ... } block
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"No valid JSON found in model output. First 400 chars:\n{text[:400]}")


def _generate(model_pack, messages: list[dict], max_tokens: int) -> str:
    """Run a greedy (T=0) chat completion."""
    model, tokenizer = model_pack
    text   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    print(f"[LyricConceptGen] prompt_tokens={inputs.input_ids.shape[1]}  max_new_tokens={max_tokens}")
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens = max_tokens,
            do_sample      = False,           # greedy — deterministic JSON
            pad_token_id   = tokenizer.eos_token_id,
            eos_token_id   = tokenizer.eos_token_id,
        )
    new_ids = out_ids[0][inputs.input_ids.shape[1]:]
    raw = tokenizer.decode(new_ids, skip_special_tokens=True)
    print(f"[LyricConceptGen] generated {new_ids.shape[0]} tokens")
    return raw


# ── Node: LyricConceptLLMLoader ───────────────────────────────────────────────

class LyricConceptLLMLoader:
    """
    Load a HuggingFace causal-LM for lyric concept generation.
    Default: Qwen/Qwen2.5-Coder-7B-Instruct  (~14 GB, bfloat16).
    Model is cached to {ComfyUI}/models/llm/hf_cache/ between runs.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_id": ("STRING", {"default": "Qwen/Qwen2.5-Coder-7B-Instruct"}),
                "dtype":    (["bfloat16", "float16", "float32"], {"default": "bfloat16"}),
            }
        }

    RETURN_TYPES  = ("LYRIC_LLM",)
    RETURN_NAMES  = ("model",)
    FUNCTION      = "load"
    CATEGORY      = "LyricConcept"

    def load(self, model_id: str, dtype: str):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        os.environ["HF_HOME"] = _LLM_CACHE

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
        print(f"[LyricConceptLLMLoader] loading {model_id!r} ({dtype})")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model     = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype_map[dtype], device_map="auto"
        )
        model.eval()
        print(f"[LyricConceptLLMLoader] ready on {next(model.parameters()).device}")
        return ((model, tokenizer),)


# ── Node: LyricSystemPromptLibrary ───────────────────────────────────────────

class LyricSystemPromptLibrary:
    """
    Select a system prompt from the built-in style library.
    The selected prompt is passed as a STRING to LyricConceptGen.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style": (STYLE_KEYS, {"default": STYLE_KEYS[0]}),
            }
        }

    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("system_prompt",)
    FUNCTION      = "get_prompt"
    CATEGORY      = "LyricConcept"
    OUTPUT_NODE   = False

    def get_prompt(self, style: str):
        return (SYSTEM_PROMPTS.get(style, SYSTEM_PROMPTS[STYLE_KEYS[0]]),)


# ── Node: LyricConceptGen ────────────────────────────────────────────────────

class LyricConceptGen:
    """
    Generate a structured music video concept from timed lyrics JSON.

    Inputs:
      model         — from LyricConceptLLMLoader
      system_prompt — from LyricSystemPromptLibrary or any STRING node
      lyrics_json   — timed lyrics JSON string from TranscribeAudioFromURL
      max_tokens    — cap on generated tokens (default 1536 ≈ 90–160 s)

    Outputs:
      concept_json  — raw JSON string  (STRING)
      concept_card  — human-readable dark card  (IMAGE)
      concept_data  — pixel-encoded JSON for ForgeExpress decode  (IMAGE)
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":         ("LYRIC_LLM",),
                "system_prompt": ("STRING", {"forceInput": True}),
                "lyrics_json":   ("STRING", {"forceInput": True}),
                "max_tokens":    ("INT",    {"default": 1536, "min": 512, "max": 4096}),
            }
        }

    RETURN_TYPES  = ("STRING", "IMAGE", "IMAGE")
    RETURN_NAMES  = ("concept_json", "concept_card", "concept_data")
    FUNCTION      = "generate"
    CATEGORY      = "LyricConcept"

    def generate(self, model, system_prompt: str, lyrics_json: str, max_tokens: int):
        model_obj, tokenizer = model
        model_id = getattr(model_obj.config, "_name_or_path", "unknown")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": (
                "Here are the timed lyrics from a song. "
                "Create a complete music video concept covering every timeline segment.\n\n"
                f"{lyrics_json}"
            )},
        ]

        # First attempt
        raw = _generate((model_obj, tokenizer), messages, max_tokens)
        try:
            data = _extract_json(raw)
        except ValueError:
            # Second attempt — explicitly prompt for JSON repair
            print("[LyricConceptGen] first attempt failed extraction, retrying with correction prompt")
            messages.append({"role": "assistant", "content": raw})
            messages.append({"role": "user", "content":
                "Your response was not valid JSON. Return the concept as a single JSON object "
                "starting with { and ending with }, no other text."})
            raw = _generate((model_obj, tokenizer), messages, max_tokens)
            data = _extract_json(raw)   # raises if still invalid

        concept_json = json.dumps(data, ensure_ascii=False)
        card = _render_concept_card(concept_json, model_id)
        encoded = _encode_string(concept_json)
        return (concept_json, card, encoded)


class ScenePromptGen(LyricConceptGen):
    """Energy-aware scene generation from combined lyrics + audio analysis.
    Accepts lyrics_json AND energy_json separately and merges them before the LLM."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":         ("LYRIC_LLM",),
                "system_prompt": ("STRING", {"forceInput": True}),
                "lyrics_json":   ("STRING", {"forceInput": True}),
                "energy_json":   ("STRING", {"forceInput": True}),
                "max_tokens":    ("INT", {"default": 2048, "min": 512, "max": 4096}),
            }
        }

    RETURN_TYPES  = ("STRING", "IMAGE", "IMAGE")
    RETURN_NAMES  = ("concept_json", "concept_card", "concept_data")
    FUNCTION      = "generate_with_energy"
    CATEGORY      = "LyricConcept"

    def generate_with_energy(self, model, system_prompt, lyrics_json, energy_json, max_tokens):
        import json
        # Combine into a single structured context object for the LLM
        try:
            combined = json.dumps({"lyrics": json.loads(lyrics_json),
                                   "energy": json.loads(energy_json)},
                                  ensure_ascii=False)
        except Exception:
            combined = f'{{"lyrics": {lyrics_json}, "energy": {energy_json}}}'
        # Reuse parent logic with combined context as the lyrics_json input
        return self.generate(model, system_prompt, combined, max_tokens)


# ── Registration ──────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "LyricConceptLLMLoader":    LyricConceptLLMLoader,
    "LyricSystemPromptLibrary": LyricSystemPromptLibrary,
    "LyricConceptGen":          LyricConceptGen,
    "ScenePromptGen":           ScenePromptGen,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LyricConceptLLMLoader":    "Lyric Concept LLM Loader",
    "LyricSystemPromptLibrary": "Lyric System Prompt Library",
    "LyricConceptGen":          "Lyric Concept Gen",
    "ScenePromptGen":           "Scene Prompt Gen (Energy-Aware)",
}
