"""
FIBO transformer AOT (torch.export + AOTInductor), following the same flow as the
GeometryCrafter template: capture real inputs -> export -> aoti_compile_and_package.

1) Capture one real forward's tensors (run from repo root):

   cd /data/Ali/FIBO && uv run python src/fibo_inference/test.py --capture

2) Build the .pt2 package (slow; requires CUDA):

   uv run python src/fibo_inference/test.py

3) Inference: pass the package path to generate.py --aot-transformer-package, or call
   attach_aot_transformer() from run() / your own entrypoint.

The compiled artifact is tied to the capture resolution, guidance (batch 1 vs 2), and
model checkpoint. Re-capture and recompile if those change. For multiple resolutions you
can extend this script with torch.export Dim() constraints (see your GeometryCrafter template).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import hashlib
import zipfile
import time
# Running `python src/fibo_inference/test.py` puts the script dir on sys.path[0], not the
# repo root, so `import src...` fails unless we add the project root (parents[2] = FIBO/).
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch  # noqa: E402
from torch._inductor import aoti_compile_and_package  # noqa: E402
from torch.export import export  # noqa: E402

from src.fibo_inference.aot_transformer import BriaFiboTransformerAOTWrapper  # noqa: E402
from src.fibo_inference.inference import create_pipeline  # noqa: E402
from src.fibo_inference.parse_caption import clean_json  # noqa: E402

# --- CONFIG (override via CLI) ---
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch._inductor.config.max_autotune = True
torch._inductor.config.triton.cudagraphs = False
torch._dynamo.config.cache_size_limit = 64

DEFAULT_INPUTS_PATH = "fibo_transformer_aot_inputs.pt"
DEFAULT_PACKAGE_PATH = "fibo_transformer_aot.pt2"


def _ensure_cuda_home() -> None:
    """PyTorch Inductor subprocesses need CUDA_HOME; set a sane default if missing."""
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


def capture_inputs(
    *,
    pipeline_name: str,
    device: str,
    width: int,
    height: int,
    prompt_payload: str,
    negative_payload: str,
    num_inference_steps: int,
    guidance_scale: float,
    seed: int,
    lora_path: str | None,
    out_path: Path,
) -> None:
    pipe = create_pipeline(pipeline_name, device, lora_path)
    dtype = next(pipe.transformer.parameters()).dtype
    captured: dict = {}

    def pre_hook(_module, _args, kwargs):  # noqa: ANN001
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
        captured["meta"] = {
            "pipeline_name": pipeline_name,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "dtype": str(dtype),
            "num_caption_layers": len(layers),
        }

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
    torch.save(captured, out_path)
    print(f"Saved capture to {out_path.resolve()}")


def _load_inputs_for_export(
    inputs_path: Path,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple:
    blob = torch.load(inputs_path, map_location="cpu", weights_only=False)

    def to_model(t: torch.Tensor) -> torch.Tensor:
        return t.to(device=device, dtype=dtype)

    layers = tuple(to_model(x) for x in blob["text_encoder_layers"])
    flat = (
        to_model(blob["hidden_states"]),
        to_model(blob["encoder_hidden_states"]),
        to_model(blob["timestep"]),
        to_model(blob["img_ids"]),
        to_model(blob["txt_ids"]),
        to_model(blob["attention_mask"]),
        *layers,
    )
    return flat


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
            "CUDA_HOME is not set and no toolkit found under /usr/lib/cuda or /usr/local/cuda. "
            "Install the CUDA toolkit and export CUDA_HOME, e.g. export CUDA_HOME=/usr/lib/cuda"
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

    flat = _load_inputs_for_export(inputs_path, dtype=dtype, device=dev)
    wrapper = BriaFiboTransformerAOTWrapper(transformer).to(dev).eval()

    print("Exporting (static shapes for this capture; re-capture for other resolutions)...")
    exported_model = export(wrapper, flat, dynamic_shapes=None, strict=False)

    print("Compiling to package (AOTInductor)...")
    if fast_compile:
        print(
            "Using --fast-compile (lower peak RAM / shorter compile; slower runtime than full autotune)."
        )
    else:
        print(
            "Full max_autotune can take a long time and spike RAM. If the process exits with no "
            "Python traceback, check OOM: dmesg | tail -20 | grep -i -E 'killed process|out of memory'"
        )
    # Maximum-throughput-oriented Inductor settings for datacenter GPUs (e.g. H100).
    # Expect much longer compiles and higher peak RAM during autotune; runtime is the target.
    #
    # CUTLASS: pip wheels default cutlass_dir to a path that does not exist. Install either:
    #   uv pip install nvidia-cutlass
    # or clone https://github.com/NVIDIA/cutlass and set TORCHINDUCTOR_CUTLASS_DIR to the repo root.
    #
    # CUDA graphs: AOTInductor always uses the C++ wrapper; PyTorch then forces triton.cudagraphs=False
    # (see torch._inductor.compile_fx.get_cpp_wrapper_config). Do not set triton.cudagraphs here—it only
    # logs "skipping cudagraphs due to cpp wrapper enabled". For graph capture use torch.cuda.CUDAGraph
    # outside the packaged model, or torch.compile (non-AOT) if you need Inductor cudagraphs.
    if fast_compile:
        inductor_configs = {
            "max_autotune": False,
            "max_autotune_gemm": False,
            "triton.cudagraphs": True,
            "coordinate_descent_tuning": False,
            "epilogue_fusion": True,
        }
    else:
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
    print("Extracting package to RAM disk...")
    # 1. Setup RAM disk caching directory
    cache_dir = "/dev/shm/aoti_cache"
    os.makedirs(cache_dir, exist_ok=True)

    # FAST HASHING: Hash the path, size, and mod time instead of the gigabytes of content
    stat = os.stat(package_path)
    file_id = f"{os.path.abspath(package_path)}_{stat.st_size}_{stat.st_mtime}"
    file_hash = hashlib.md5(file_id.encode('utf-8')).hexdigest()[:12]

    extract_dir = os.path.join(cache_dir, file_hash)
    marker = os.path.join(extract_dir, ".done")

    # 2. Extract only if it hasn't been extracted yet
    if not os.path.exists(marker):
        print(f"Extracting {package_path} to RAM disk...")
        start_unzip = time.perf_counter()
        with zipfile.ZipFile(package_path, 'r') as zf:
            zf.extractall(extract_dir)
        open(marker, 'w').close()
        print(f"[Timing] Unzipping took {time.perf_counter() - start_unzip:.4f} seconds")
    else:
        print(f"Found cached extraction at {extract_dir}, skipping unzip.")



def main() -> None:
    parser = argparse.ArgumentParser(description="FIBO transformer AOT capture / compile.")
    parser.add_argument("--capture", action="store_true", help="Run pipeline once and save transformer inputs.")
    parser.add_argument("--pipeline-name", type=str, default="briaai/FIBO")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--resolution", type=str, default="1024 1024", help="'W H'")
    parser.add_argument("--prompt-json", type=Path, default=Path("default_json_caption.json"))
    parser.add_argument("--negative-prompt", type=str, default="", help="JSON string, same as generate.py.")
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-steps", type=int, default=1, help="Steps for capture only (keep low).")
    parser.add_argument("--lora-path", type=str, default=None)
    parser.add_argument("--inputs-path", type=Path, default=Path(DEFAULT_INPUTS_PATH))
    parser.add_argument("--package-path", type=Path, default=Path(DEFAULT_PACKAGE_PATH))
    parser.add_argument(
        "--fast-compile",
        action="store_true",
        help="Disable aggressive autotune (less RAM / faster compile; slower inference).",
    )
    args = parser.parse_args()

    parts = args.resolution.replace(",", " ").split()
    if len(parts) != 2:
        raise SystemExit("--resolution must be 'W H'")
    width, height = int(parts[0]), int(parts[1])

    if args.capture:
        if not args.prompt_json.is_file():
            raise SystemExit(f"Prompt JSON not found: {args.prompt_json}")
        with args.prompt_json.open(encoding="utf-8") as f:
            prompt_payload = clean_json(json.load(f))
        neg = args.negative_prompt
        capture_inputs(
            pipeline_name=args.pipeline_name,
            device=args.device,
            width=width,
            height=height,
            prompt_payload=prompt_payload,
            negative_payload=neg,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
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
