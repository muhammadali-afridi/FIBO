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

        # The low-level AOTIModelContainerRunner requires a single list of inputs
        inputs = [
            hidden_states,
            encoder_hidden_states,
            timestep,
            img_ids,
            txt_ids,
            attention_mask,
            *text_encoder_layers,
        ]

        # Use the .run() method of the C++ runner
        outputs = self._compiled.run(inputs)
        sample = outputs[0]

        if return_dict:
            return Transformer2DModelOutput(sample=sample)
        return (sample,)


def attach_aot_transformer(pipeline: Any, package_path: str) -> None:
    """Replace `pipeline.transformer` with an AOT-loaded runner, caching extraction to RAM."""
    import time
    
    transformer = pipeline.transformer
    n = len(transformer.transformer_blocks) + len(transformer.single_transformer_blocks)
    device = next(transformer.parameters()).device

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
        print(f"[Timing] Found cached extraction at {extract_dir}, skipping unzip.")

    # 3. Locate the .so file and cubin directory
    so_path, cubin_dir = None, None
    for root, dirs, files in os.walk(extract_dir):
        for f in files:
            if f.endswith('.so'):
                so_path = os.path.join(root, f)
                cubin_dir = root  # Triton cubins are in the same dir as the .so
                break
        if so_path:
            break

    if not so_path:
        raise FileNotFoundError("No .so found in pt2 archive")

    # 4. Load the .so directly via the low-level runner, bypassing python metadata parsing
    device_str = str(device)
    print(f"Loading bare .so file via AOTIModelContainerRunner ({device_str})...")
    start_load = time.perf_counter()
    if device.type == "cuda":
        compiled = torch._C._aoti.AOTIModelContainerRunnerCuda(so_path, 1, device_str, cubin_dir)
    else:
        compiled = torch._C._aoti.AOTIModelContainerRunnerCpu(so_path, 1, device_str, cubin_dir)
    print(f"[Timing] .so loading took {time.perf_counter() - start_load:.4f} seconds")

    # 5. Attach runner to pipeline
    runner = AOTFiboTransformerRunner(compiled, n)
    runner.config = transformer.config
    runner.transformer_blocks = _BlocksLen(len(transformer.transformer_blocks))
    runner.single_transformer_blocks = _BlocksLen(len(transformer.single_transformer_blocks))
    runner._dtype = next(transformer.parameters()).dtype
    pipeline.transformer = runner