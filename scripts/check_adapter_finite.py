# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Scan a checkpoint dir for non-finite (NaN/inf) and suspiciously large weights.

A fine-tune that produced ``inf``/``nan`` logits at inference (the usual cause of a vLLM
``EngineCore`` death like "probability tensor contains either inf, nan or element < 0") often
already shows it in the saved weights, or at least an extreme max abs value. This walks every
``*.safetensors`` shard without moving tensors to GPU and reports any non-finite entries plus the
largest magnitude per tensor.

Run with::

    python scripts/check_adapter_finite.py /path/to/checkpoint_or_adapter_dir

Point it at the LoRA adapter dir you are serving (the one with ``adapter_model.safetensors``), or
at a merged/exported full model dir.
"""

import sys
from pathlib import Path

import torch
from safetensors import safe_open


# Anything past this absolute value is a red flag for bf16 inference overflow, even if "finite".
LARGE_ABS_WARN = 1.0e4


def scan_dir(path: str) -> int:
    shards = sorted(Path(path).glob("*.safetensors"))
    if not shards:
        print(f"No *.safetensors files under {path!r} — wrong directory?")
        return 2

    bad_tensors = 0
    global_max = 0.0
    global_max_name = ""
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as f:
            for key in f.keys():
                t = f.get_tensor(key).float()  # upcast so we test what inference sees, not bf16 quirks
                n_nan = int(torch.isnan(t).sum())
                n_inf = int(torch.isinf(t).sum())
                max_abs = float(t.abs().max()) if t.numel() else 0.0
                if max_abs > global_max:
                    global_max, global_max_name = max_abs, f"{shard.name}::{key}"
                if n_nan or n_inf:
                    bad_tensors += 1
                    print(f"  NON-FINITE  {shard.name}::{key}  nan={n_nan} inf={n_inf} max_abs={max_abs:.3e}")
                elif max_abs > LARGE_ABS_WARN:
                    print(f"  LARGE       {shard.name}::{key}  max_abs={max_abs:.3e}")

    print(f"\nScanned {len(shards)} shard(s). Largest |w| = {global_max:.3e} in {global_max_name}.")
    if bad_tensors:
        print(f"RESULT: {bad_tensors} tensor(s) contain NaN/inf — the checkpoint is corrupt. This is the crash.")
        return 1
    if global_max > LARGE_ABS_WARN:
        print("RESULT: no NaN/inf, but some weights are very large — possible bf16 inference overflow.")
        return 0
    print("RESULT: weights are finite and within a normal range — the crash is likely runtime, not the weights.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(scan_dir(sys.argv[1]))
