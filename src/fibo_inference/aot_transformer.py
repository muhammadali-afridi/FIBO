"""Ahead-of-time Inductor packaging for `BriaFiboTransformer2DModel`.

The diffusers transformer takes a list of caption tensors and a dict
`joint_attention_kwargs`, which `torch.export` handles poorly. The export
wrapper flattens those into plain tensor arguments.
"""

from __future__ import annotations

import os
import hashlib
import zipfile
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


class BriaFiboTransformerAOTWrapper(nn.Module):
    """Tensor-only forward for `torch.export` / AOTInductor."""

    def __init__(self, transformer: nn.Module) -> None:
        super().__init__()
        self.transformer = transformer
        self._num_caption_layers = len(transformer.transformer_blocks) + len(transformer.single_transformer_blocks)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        img_ids: torch.Tensor,
        txt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        *text_encoder_layers: torch.Tensor,
    ) -> torch.Tensor:
        if len(text_encoder_layers) != self._num_caption_layers:
            raise ValueError(
                f"expected {self._num_caption_layers} caption layer tensors, got {len(text_encoder_layers)}"
            )
        joint_attention_kwargs: Dict[str, Any] = {"attention_mask": attention_mask}
        return self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            text_encoder_layers=list(text_encoder_layers),
            joint_attention_kwargs=joint_attention_kwargs,
            img_ids=img_ids,
            txt_ids=txt_ids,
            return_dict=False,
        )[0]


class _BlocksLen:
    """Pipeline only uses `len(transformer.transformer_blocks)`; avoid holding full blocks."""

    __slots__ = ("n",)

    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n


class AOTFiboTransformerRunner(nn.Module):
    """Same call signature as `BriaFiboTransformer2DModel.forward` for the pipeline."""

    def __init__(self, compiled: Any, num_caption_layers: int) -> None:
        super().__init__()
        self._compiled = compiled
        self._num_caption_layers = num_caption_layers
        self._dtype: torch.dtype = torch.float32

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        text_encoder_layers: list = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        return_dict: bool = True,
    ):
        from diffusers.models.modeling_outputs import Transformer2DModelOutput

        if joint_attention_kwargs is None:
            raise ValueError("joint_attention_kwargs is required")
        attention_mask = joint_attention_kwargs.get("attention_mask")
        if attention_mask is None:
            raise ValueError("joint_attention_kwargs['attention_mask'] is required for AOT inference")

        if text_encoder_layers is None or len(text_encoder_layers) != self._num_caption_layers:
            raise ValueError(
                f"text_encoder_layers must be a list of length {self._num_caption_layers}, "
                f"got {0 if text_encoder_layers is None else len(text_encoder_layers)}"
            )

        inputs = [
            hidden_states,
            encoder_hidden_states,
            timestep,
            img_ids,
            txt_ids,
            attention_mask,
            *text_encoder_layers,
        ]

        outputs = self._compiled.run(inputs)
        sample = outputs[0]

        if return_dict:
            return Transformer2DModelOutput(sample=sample)
        return (sample,)


# ---------------------------------------------------------------------------
# Helpers: locate .so inside an extracted directory
# ---------------------------------------------------------------------------

def _find_so(directory: str) -> tuple[str, str]:
    """Walk `directory` and return (so_path, cubin_dir)."""
    for root, _dirs, files in os.walk(directory):
        for f in files:
            if f.endswith(".so"):
                return os.path.join(root, f), root
    raise FileNotFoundError(f"No .so found in {directory}")


def _extract_pt2_to_cache(package_path: str) -> str:
    """Extract .pt2 zip to /dev/shm cache, returning the extraction directory."""
    import time

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
        print(f"[Timing] Unzipping took {time.perf_counter() - start:.4f}s")
    else:
        print(f"[Timing] Found cached extraction at {extract_dir}, skipping unzip.")

    return extract_dir


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def attach_aot_transformer(
    pipeline: Any,
    package_path: str | None = None,
    extracted_dir: str | None = None,
) -> None:
    """Replace `pipeline.transformer` with an AOT-loaded runner.

    Two modes:
      1. **Pre-extracted directory** (fast — no unzipping):
             attach_aot_transformer(pipe, extracted_dir="/data/aot_extracted")
         Use this on fal / modal / any deploy where you upload the extracted
         files to a persistent volume.

      2. **Raw .pt2 file** (extracts to /dev/shm on first call):
             attach_aot_transformer(pipe, package_path="model.pt2")
         Convenience for local dev; 24s unzip on first run, cached after.
    """
    import time

    if extracted_dir is None and package_path is None:
        raise ValueError("Provide either extracted_dir or package_path")

    transformer = pipeline.transformer
    n = len(transformer.transformer_blocks) + len(transformer.single_transformer_blocks)
    device = next(transformer.parameters()).device

    # --- Resolve the directory containing the .so ---
    if extracted_dir is not None:
        # Fast path: already extracted on a volume, no unzipping
        so_dir = extracted_dir
    else:
        # Fallback: extract .pt2 to /dev/shm cache
        so_dir = _extract_pt2_to_cache(package_path)

    so_path, cubin_dir = _find_so(so_dir)

    # --- Load the .so ---
    device_str = str(device)
    print(f"Loading bare .so via AOTIModelContainerRunner ({device_str}) ...")
    start_load = time.perf_counter()
    if device.type == "cuda":
        compiled = torch._C._aoti.AOTIModelContainerRunnerCuda(so_path, 1, device_str, cubin_dir)
    else:
        compiled = torch._C._aoti.AOTIModelContainerRunnerCpu(so_path, 1, device_str, cubin_dir)
    print(f"[Timing] .so loading took {time.perf_counter() - start_load:.4f}s")

    # --- Attach runner to pipeline ---
    runner = AOTFiboTransformerRunner(compiled, n)
    runner.config = transformer.config
    runner.transformer_blocks = _BlocksLen(len(transformer.transformer_blocks))
    runner.single_transformer_blocks = _BlocksLen(len(transformer.single_transformer_blocks))
    runner._dtype = next(transformer.parameters()).dtype
    pipeline.transformer = runner