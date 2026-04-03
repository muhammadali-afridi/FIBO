"""
Extract a .pt2 AOT package into a directory suitable for uploading
to a persistent volume (e.g. fal, modal, etc.).

Usage:
    python extract_pt2_for_volume.py fibo_transformer_aot.pt2 --out-dir ./aot_extracted

Then upload the output directory to your volume:
    # fal example:
    fal files upload ./aot_extracted data/aot_extracted

At runtime, pass the directory (not .pt2) to your loading code:
    --aot-transformer-dir /data/aot_extracted
"""

import argparse
import os
import zipfile
import time
from pathlib import Path


def extract(pt2_path: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    print(f"Extracting {pt2_path} -> {out_dir} ...")
    start = time.perf_counter()
    with zipfile.ZipFile(pt2_path, "r") as zf:
        zf.extractall(out_dir)
    elapsed = time.perf_counter() - start
    print(f"Done in {elapsed:.2f}s")

    # Find and print the .so path so you know what to expect at runtime
    for root, _dirs, files in os.walk(out_dir):
        for f in files:
            if f.endswith(".so"):
                so_path = os.path.join(root, f)
                print(f"\n.so file:   {so_path}")
                print(f"cubin dir:  {root}")
                print(f"\nUpload '{out_dir}' to your volume, then at runtime load with:")
                print(f'  AOTIModelContainerRunnerCuda("{so_path}", 1, "cuda:0", "{root}")')
                return

    print("WARNING: no .so found in archive")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("pt2_path", help="Path to the .pt2 package")
    p.add_argument("--out-dir", default="./aot_extracted", help="Output directory")
    extract(**vars(p.parse_args()))