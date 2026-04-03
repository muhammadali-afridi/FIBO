"""
FIBO transformer AOT (torch.export + AOTInductor) with **dynamic resolution** support.

1) Capture inputs for all warmup resolutions in one shot:

   cd /data/Ali/FIBO && uv run python src/fibo_inference/aot_export.py --capture

2) Build the .pt2 package (slow; requires CUDA):

   uv run python src/fibo_inference/aot_export.py

The compiled artifact now handles multiple resolutions (1024x1024, 1344x768, 768x1344)
via torch.export Dim() constraints — no need to recompile per resolution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import ujson

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
from torch._inductor import aoti_compile_and_package  # noqa: E402
from torch.export import export, Dim  # noqa: E402

from src.fibo_inference.aot_transformer import BriaFiboTransformerAOTWrapper  # noqa: E402
from src.fibo_inference.inference import create_pipeline  # noqa: E402
from src.fibo_inference.parse_caption import clean_json  # noqa: E402

# --- CONFIG ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch._inductor.config.max_autotune = True
torch._inductor.config.triton.cudagraphs = False
torch._dynamo.config.cache_size_limit = 64

DEFAULT_INPUTS_PATH = "fibo_transformer_aot_inputs.pt"
DEFAULT_PACKAGE_PATH = "fibo_transformer_aot.pt2"

# ---------------------------------------------------------------------------
# Warmup prompts & resolutions (from warmup())
# ---------------------------------------------------------------------------
WARMUP_CONFIGS: list[tuple[dict, int, int]] = [
    # (json_prompt, width, height)
    (
        {
            "short_description": "A graceful faun stands amidst a carpet of green fallen leaves, its gaze directed towards the right. The scene evokes a sense of serene autumnal beauty, with the soft, diffused light. The composition is balanced, drawing the viewer's eye to the central figure and its gentle expression. The overall mood is peaceful and contemplative, capturing a moment of quiet observation in a natural setting.",
            "objects": [
                {
                    "description": "A young faun with delicate features, prominent horns, and soft, speckled fur.",
                    "location": "center",
                    "relationship": "The faun is the primary subject, standing in the middle of the leaf-covered ground.",
                    "relative_size": "medium-to-large within frame",
                    "shape_and_color": "Elegant, humanoid shape with animalistic features. Predominantly light brown and white speckled fur, with darker brown horns.",
                    "texture": "soft, slightly coarse fur",
                    "appearance_details": "The horns are curved and textured. Its cheeks are slightly flushed.",
                    "number_of_objects": 1,
                    "pose": "Standing upright, with its body slightly turned to the left and its head looking to the right.",
                    "expression": "curious and serene",
                    "action": "observing something in the distance",
                    "gender": "male",
                    "skin_tone_and_texture": "N/A (animal fur)",
                    "orientation": "vertical",
                },
                {
                    "description": "A dense scattering of fallen autumn leaves in various shades of brown, yellow, and orange.",
                    "location": "foreground and midground",
                    "relationship": "These leaves form the ground cover upon which the faun stands.",
                    "relative_size": "small individual leaves, large collective area",
                    "shape_and_color": "Varied leaf shapes. Colors range from deep brown to golden yellow and reddish-orange.",
                    "texture": "dry, brittle, papery",
                    "appearance_details": "Some leaves are whole, others are broken or curled.",
                    "orientation": "scattered, mostly flat",
                },
            ],
            "background_setting": "A soft-focus forest floor and underbrush, with hints of green foliage.",
            "lighting": {"conditions": "soft, diffused daylight", "direction": "side-lit from the left", "shadows": "soft, elongated"},
            "aesthetics": {"composition": "rule of thirds", "color_scheme": "warm autumnal palette", "mood_atmosphere": "peaceful, serene", "aesthetic_score": "very high", "preference_score": "very high"},
            "photographic_characteristics": {"depth_of_field": "shallow", "focus": "sharp on faun", "camera_angle": "eye-level", "lens_focal_length": "50mm"},
            "style_medium": "photograph",
            "context": "Nature-themed photograph.",
            "artistic_style": "realistic, painterly",
        },
        1024,
        1024,
    ),
    (
        {
            "short_description": "A modern cityscape at sunset with tall glass buildings reflecting warm orange and pink hues.",
            "objects": [
                {
                    "description": "Multiple skyscrapers with glass facades and geometric designs.",
                    "location": "background and midground",
                    "relationship": "The buildings form the main architectural elements.",
                    "relative_size": "large",
                    "shape_and_color": "Rectangular shapes in blue, silver, and warm sunset colors.",
                    "texture": "smooth, reflective glass",
                    "appearance_details": "Windows reflect sunset. Some interiors illuminated.",
                    "number_of_objects": "multiple",
                    "orientation": "vertical",
                }
            ],
            "background_setting": "Urban skyline with dramatic sunset.",
            "lighting": {"conditions": "warm sunset", "direction": "backlit", "shadows": "long, dramatic"},
            "aesthetics": {"composition": "wide panoramic", "color_scheme": "sunset palette", "mood_atmosphere": "dramatic, vibrant", "aesthetic_score": "very high", "preference_score": "very high"},
            "photographic_characteristics": {"depth_of_field": "deep", "focus": "sharp throughout", "camera_angle": "slightly elevated", "lens_focal_length": "24mm"},
            "style_medium": "photograph",
            "context": "Urban photography.",
            "artistic_style": "realistic, cinematic",
        },
        1344,
        768,
    ),
    (
        {
            "short_description": "A close-up portrait of a person with expressive eyes in soft natural lighting.",
            "objects": [
                {
                    "description": "A person with clear, expressive eyes and natural features.",
                    "location": "center",
                    "relationship": "Primary subject.",
                    "relative_size": "large, filling frame",
                    "shape_and_color": "Natural human features, warm skin tones",
                    "texture": "smooth skin",
                    "appearance_details": "Eyes well-lit. Hair has natural texture.",
                    "number_of_objects": 1,
                    "pose": "facing camera with slight head tilt",
                    "expression": "calm and confident",
                    "action": "looking at viewer",
                    "orientation": "vertical",
                }
            ],
            "background_setting": "Soft, blurred neutral background.",
            "lighting": {"conditions": "soft window light", "direction": "front-lit with side", "shadows": "soft, subtle"},
            "aesthetics": {"composition": "centered portrait", "color_scheme": "warm natural tones", "mood_atmosphere": "intimate, professional", "aesthetic_score": "very high", "preference_score": "very high"},
            "photographic_characteristics": {"depth_of_field": "shallow", "focus": "sharp on eyes", "camera_angle": "eye-level", "lens_focal_length": "85mm"},
            "style_medium": "photograph",
            "context": "Professional headshot.",
            "artistic_style": "realistic, professional",
        },
        768,
        1344,
    ),
]


def _ensure_cuda_home() -> None:
    if os.environ.get("CUDA_HOME"):
        return
    for root in ("/usr/lib/cuda", "/usr/local/cuda"):
        inc = os.path.join(root, "include")
        if os.path.isdir(inc) and os.path.isfile(os.path.join(inc, "cuda.h")):
            os.environ["CUDA_HOME"] = root
            bin_dir = os.path.join(root, "bin")
            if os.path.isdir(bin_dir):
                os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
            return


def _clone_cpu(t: torch.Tensor) -> torch.Tensor:
    return t.detach().clone().cpu()


# ---------------------------------------------------------------------------
# Capture: run the pipeline once per resolution, save all inputs
# ---------------------------------------------------------------------------

def _capture_one_forward(
    pipe,
    prompt_payload: str,
    negative_payload: str,
    width: int,
    height: int,
    guidance_scale: float,
    num_inference_steps: int,
    seed: int,
    device: str,
) -> dict:
    """Run one pipeline forward and capture the transformer's input tensors."""
    dtype = next(pipe.transformer.parameters()).dtype
    captured: dict = {}

    def pre_hook(_module, _args, kwargs):
        if captured.get("_done"):
            return
        captured["_done"] = True
        layers = kwargs["text_encoder_layers"]
        captured["hidden_states"] = _clone_cpu(kwargs["hidden_states"])
        captured["encoder_hidden_states"] = _clone_cpu(kwargs["encoder_hidden_states"])
        captured["timestep"] = _clone_cpu(kwargs["timestep"])
        captured["img_ids"] = _clone_cpu(kwargs["img_ids"])
        captured["txt_ids"] = _clone_cpu(kwargs["txt_ids"])
        captured["attention_mask"] = _clone_cpu(kwargs["joint_attention_kwargs"]["attention_mask"])
        captured["text_encoder_layers"] = tuple(_clone_cpu(x) for x in layers)

    hook_handle = pipe.transformer.register_forward_pre_hook(pre_hook, with_kwargs=True)
    try:
        generator = None
        if seed >= 0:
            generator = torch.Generator(device=device).manual_seed(seed)
        with torch.inference_mode():
            pipe(
                prompt_payload,
                num_inference_steps=num_inference_steps,
                negative_prompt=negative_payload,
                generator=generator,
                width=width,
                height=height,
                guidance_scale=guidance_scale,
            )
    finally:
        hook_handle.remove()

    if not captured.get("_done"):
        raise RuntimeError("Capture hook never ran; transformer was not called.")
    del captured["_done"]
    return captured


def capture_warmup_inputs(
    *,
    pipeline_name: str,
    device: str,
    guidance_scale: float,
    num_inference_steps: int,
    seed: int,
    lora_path: str | None,
    out_path: Path,
) -> None:
    """Capture transformer inputs for every warmup resolution."""
    pipe = create_pipeline(pipeline_name, device, lora_path)

    all_captures: list[dict] = []
    for json_prompt, width, height in WARMUP_CONFIGS:
        prompt_str = ujson.dumps(json_prompt, escape_forward_slashes=False)
        print(f"  Capturing {width}x{height} ...")
        cap = _capture_one_forward(
            pipe,
            prompt_payload=prompt_str,
            negative_payload="",
            width=width,
            height=height,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            device=device,
        )
        cap["meta"] = {
            "pipeline_name": pipeline_name,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "dtype": str(next(pipe.transformer.parameters()).dtype),
            "num_caption_layers": len(cap["text_encoder_layers"]),
        }
        all_captures.append(cap)

    torch.save({"captures": all_captures}, out_path)
    print(f"Saved {len(all_captures)} captures to {out_path.resolve()}")


# ---------------------------------------------------------------------------
# Build flat tensor tuples & detect dynamic dims
# ---------------------------------------------------------------------------

def _make_flat(
    cap: dict,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    def to_model(t: torch.Tensor) -> torch.Tensor:
        return t.to(device=device, dtype=dtype)

    layers = tuple(to_model(x) for x in cap["text_encoder_layers"])
    return (
        to_model(cap["hidden_states"]),
        to_model(cap["encoder_hidden_states"]),
        to_model(cap["timestep"]),
        to_model(cap["img_ids"]),
        to_model(cap["txt_ids"]),
        to_model(cap["attention_mask"]),
        *layers,
    )


def _build_dynamic_shapes(
    flats: list[tuple[torch.Tensor, ...]],
    num_named_args: int = 6,
) -> tuple | None:
    """Compare tensor shapes across captures and build a dynamic_shapes spec.

    Dimensions that differ across captures become Dim() objects. Dimensions with
    identical sizes across all captures stay static (None).

    Dims that always take the same set of sizes are assumed to represent the same
    underlying dynamic quantity and share one Dim object (important for export).

    IMPORTANT: torch.export maps dynamic_shapes against the **function signature**,
    not the flat arg count. BriaFiboTransformerAOTWrapper.forward has 7 params:
      (hidden_states, encoder_hidden_states, timestep, img_ids, txt_ids,
       attention_mask, *text_encoder_layers)
    So we return a 7-element tuple where the last element is a *list* of specs
    covering the varargs.
    """
    n_tensors = len(flats[0])

    # 1. Collect sizes per (tensor_index, dim_index) across all captures
    size_vecs: dict[tuple[int, int], list[int]] = {}
    for ti in range(n_tensors):
        ndim = flats[0][ti].ndim
        for di in range(ndim):
            sizes = [flat[ti].shape[di] for flat in flats]
            if len(set(sizes)) > 1:  # this dim actually varies
                size_vecs[(ti, di)] = sizes

    if not size_vecs:
        print("All tensor shapes are identical across captures — no dynamic dims needed.")
        return None

    # 2. Group dimensions that always have the same sizes -> same Dim object.
    group_map: dict[tuple[int, ...], Dim] = {}
    dim_for: dict[tuple[int, int], Dim] = {}
    dim_counter = 0

    for key, sizes in size_vecs.items():
        sizes_tuple = tuple(sizes)
        if sizes_tuple not in group_map:
            lo, hi = min(sizes), max(sizes)
            dim_obj = Dim(f"dyn_{dim_counter}", min=lo, max=hi)
            group_map[sizes_tuple] = dim_obj
            dim_counter += 1
            print(
                f"  Dynamic dim dyn_{dim_counter - 1}: "
                f"sizes={list(set(sizes))}, min={lo}, max={hi}"
            )
        dim_for[key] = group_map[sizes_tuple]

    # 3. Build per-tensor specs
    def _spec_for(ti: int) -> dict[int, Dim] | None:
        per_tensor: dict[int, Dim] = {}
        ndim = flats[0][ti].ndim
        for di in range(ndim):
            if (ti, di) in dim_for:
                per_tensor[di] = dim_for[(ti, di)]
        return per_tensor if per_tensor else None

    # 4. Assemble into the signature-matching structure:
    #    (spec0, spec1, ..., spec5, [spec6, spec7, ..., spec_N])
    #    First `num_named_args` are individual, the rest form a list for *varargs.
    named_specs = [_spec_for(ti) for ti in range(num_named_args)]
    vararg_specs = [_spec_for(ti) for ti in range(num_named_args, n_tensors)]

    return tuple(named_specs) + (tuple(vararg_specs),)


# ---------------------------------------------------------------------------
# AOT compile with dynamic shapes
# ---------------------------------------------------------------------------

def compile_aot(
    *,
    pipeline_name: str,
    device: str,
    lora_path: str | None,
    inputs_path: Path,
    package_path: Path,
    fast_compile: bool,
) -> None:
    _ensure_cuda_home()
    if not os.environ.get("CUDA_HOME"):
        raise SystemExit(
            "CUDA_HOME is not set and no toolkit found under /usr/lib/cuda or /usr/local/cuda."
        )
    if not torch.cuda.is_available():
        raise SystemExit("AOT compile requires CUDA.")

    if fast_compile:
        torch._inductor.config.max_autotune = False
    else:
        torch._inductor.config.max_autotune = True

    pipe = create_pipeline(pipeline_name, device, lora_path)
    transformer = pipe.transformer
    dtype = next(transformer.parameters()).dtype
    dev = next(transformer.parameters()).device

    # --- Load all captured resolutions ---
    blob = torch.load(inputs_path, map_location="cpu", weights_only=False)

    # Support both old single-capture format and new multi-capture format
    if "captures" in blob:
        captures = blob["captures"]
    else:
        # Legacy single capture
        captures = [blob]

    flats = [_make_flat(cap, dtype=dtype, device=dev) for cap in captures]
    print(f"Loaded {len(flats)} capture(s). Shapes per capture:")
    for i, flat in enumerate(flats):
        meta = captures[i].get("meta", {})
        res = f"{meta.get('width', '?')}x{meta.get('height', '?')}"
        shapes = [tuple(t.shape) for t in flat[:6]]  # first 6 named tensors
        print(f"  [{i}] {res}: hs={shapes[0]}, enc={shapes[1]}, ts={shapes[2]}, "
              f"img_ids={shapes[3]}, txt_ids={shapes[4]}, mask={shapes[5]}")

    # --- Detect dynamic dims ---
    dynamic_shapes = _build_dynamic_shapes(flats)

    # --- Export ---
    wrapper = BriaFiboTransformerAOTWrapper(transformer).to(dev).eval()

    # Use the first capture as the example input for export
    example_flat = flats[0]

    if dynamic_shapes is not None:
        print("Exporting with dynamic shapes ...")
    else:
        print("Exporting with static shapes (no resolution variation detected) ...")

    exported_model = export(
        wrapper,
        example_flat,
        dynamic_shapes=dynamic_shapes,
        strict=False,
    )

    # --- Compile ---
    print("Compiling to package (AOTInductor) ...")
    if fast_compile:
        print("  --fast-compile: lower RAM / faster compile; slower runtime.")
        inductor_configs = {
            "max_autotune": False,
            "max_autotune_gemm": False,
            "triton.cudagraphs": True,
            "coordinate_descent_tuning": False,
            "epilogue_fusion": True,
        }
    else:
        print(
            "  Full max_autotune — can be slow and spike RAM. If OOM: "
            "dmesg | tail -20 | grep -i -E 'killed process|out of memory'"
        )
        inductor_configs = {
            "max_autotune": True,
            "max_autotune_gemm": True,
            "triton.cudagraphs": True,
            "coordinate_descent_tuning": True,
            "epilogue_fusion": True,
        }

    out = aoti_compile_and_package(
        exported_model,
        package_path=str(package_path),
        inductor_configs=inductor_configs,
    )
    print(f"SUCCESS: wrote package to {out}")

    # --- Extract to RAM disk ---
    _extract_to_ramdisk(package_path)


def _extract_to_ramdisk(package_path: Path) -> str:
    cache_dir = "/dev/shm/aoti_cache"
    os.makedirs(cache_dir, exist_ok=True)

    stat = os.stat(package_path)
    file_id = f"{os.path.abspath(package_path)}_{stat.st_size}_{stat.st_mtime}"
    file_hash = hashlib.md5(file_id.encode("utf-8")).hexdigest()[:12]

    extract_dir = os.path.join(cache_dir, file_hash)
    marker = os.path.join(extract_dir, ".done")

    if not os.path.exists(marker):
        print(f"Extracting {package_path} to RAM disk ...")
        start = time.perf_counter()
        with zipfile.ZipFile(package_path, "r") as zf:
            zf.extractall(extract_dir)
        open(marker, "w").close()
        print(f"[Timing] Unzip took {time.perf_counter() - start:.4f}s")
    else:
        print(f"Found cached extraction at {extract_dir}, skipping unzip.")
    return extract_dir


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="FIBO transformer AOT capture / compile.")
    parser.add_argument(
        "--capture", action="store_true",
        help="Run pipeline for all warmup resolutions and save transformer inputs.",
    )
    parser.add_argument("--pipeline-name", type=str, default="briaai/FIBO")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=1, help="Steps for capture (keep low).")
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--inputs-path", type=Path, default=Path(DEFAULT_INPUTS_PATH))
    parser.add_argument("--package-path", type=Path, default=Path(DEFAULT_PACKAGE_PATH))
    parser.add_argument(
        "--fast-compile", action="store_true",
        help="Disable aggressive autotune (less RAM / faster compile; slower inference).",
    )
    args = parser.parse_args()

    if args.capture:
        capture_warmup_inputs(
            pipeline_name=args.pipeline_name,
            device=args.device,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_steps,
            seed=args.seed,
            lora_path=args.lora_path,
            out_path=args.inputs_path,
        )
        return

    if not args.inputs_path.is_file():
        raise SystemExit(f"Missing inputs file {args.inputs_path}; run with --capture first.")

    compile_aot(
        pipeline_name=args.pipeline_name,
        device=args.device,
        lora_path=args.lora_path,
        inputs_path=args.inputs_path,
        package_path=args.package_path,
        fast_compile=args.fast_compile,
    )


if __name__ == "__main__":
    main()