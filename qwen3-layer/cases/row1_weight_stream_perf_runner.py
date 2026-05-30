"""NPU runner for the row1 full-layer Q4NX weight fanout performance slice."""

from __future__ import annotations

from pathlib import Path

import npu_build
import numpy as np
import torch
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

from cases import row1_weight_stream_perf_generate as generate
from cases.full_layer_engine_reference import TOTAL_WEIGHT_AND_AUX_I32, TOTAL_WEIGHT_BF16
from projection_schedule import FULL_LAYER_TOTAL_WEIGHT_CHUNKS

CASE_NAME = generate.CASE_NAME
EXPERIMENT_DIR = Path(__file__).parent.parent
WEIGHT_BYTES = TOTAL_WEIGHT_BF16 * 2
DONE_DWORDS = generate.DONE_DWORDS


def _reject_options(current_token: int | None, model_path: Path | None, layer: int, download_model: bool) -> None:
    if current_token is not None:
        raise ValueError(f"--current-token is not used by {CASE_NAME}")
    if model_path is not None or download_model:
        raise ValueError(f"model options are not used by {CASE_NAME}")
    if layer != 0:
        raise ValueError(f"--layer is not used by {CASE_NAME}")


def build_kernel(build_name: str = CASE_NAME) -> tuple[Path, Path]:
    build_dir = EXPERIMENT_DIR / "build" / build_name
    build_dir.mkdir(parents=True, exist_ok=True)
    mlir_path = build_dir / "design.mlir"
    xclbin_path = build_dir / "design.xclbin"
    insts_path = build_dir / "design.bin"

    mlir_text = generate.generate_mlir()
    mlir_path.write_text(mlir_text)
    errors = generate.validate_generated_mlir(mlir_text)
    if errors:
        raise RuntimeError("\n".join(f"  ROW1 WEIGHT PERF STRUCTURE FAIL: {error}" for error in errors))
    npu_build.compile_mlir(mlir_path, xclbin_path, insts_path)
    return xclbin_path, insts_path


def check_only(
    current_token: int | None = None,
    model_path: Path | None = None,
    layer: int = 0,
    download_model: bool = False,
) -> bool:
    _reject_options(current_token, model_path, layer, download_model)
    errors = generate.validate_generated_mlir(generate.generate_mlir())
    if errors:
        for error in errors:
            print(f"  ROW1 WEIGHT PERF FAIL: {error}")
        return False
    print("  PASS: row1 weight-stream perf MLIR uses full-layer weight fanout ABI")
    print(f"  chunks_per_main_tile={FULL_LAYER_TOTAL_WEIGHT_CHUNKS}")
    print(f"  total_weight_bytes={WEIGHT_BYTES}")
    return True


def build_only(
    current_token: int | None = None,
    model_path: Path | None = None,
    layer: int = 0,
    download_model: bool = False,
) -> bool:
    _reject_options(current_token, model_path, layer, download_model)
    xclbin_path, insts_path = build_kernel()
    print(f"  PASS: built {xclbin_path}")
    print(f"  PASS: built {insts_path}")
    return True


def run(
    current_token: int | None = None,
    model_path: Path | None = None,
    layer: int = 0,
    download_model: bool = False,
) -> bool:
    _reject_options(current_token, model_path, layer, download_model)
    print("=" * 78)
    print(f"qwen3-layer: {CASE_NAME}")
    print("=" * 78)
    print("  route=host/shim -> row1 S2MM4/5 -> row1 MM2S0..3 -> main16 DMA1 sink")
    print("  compute=disabled; main16 returns done through packetized row1 -> bridge -> shim_out")
    print(f"  chunks_per_main_tile={FULL_LAYER_TOTAL_WEIGHT_CHUNKS}")
    print(f"  total_weight_bytes={WEIGHT_BYTES}")
    print()

    xclbin_path, insts_path = build_kernel()
    print("  Loading NPU kernel...")
    handle = npu_build.load_kernel(xclbin_path, insts_path)
    weights = np.zeros((TOTAL_WEIGHT_AND_AUX_I32,), dtype=np.int32)
    weights_buf = XRTTensor.from_torch(torch.from_numpy(weights).to(torch.int32))
    done_buf = XRTTensor((DONE_DWORDS,), dtype=np.int32)

    print("  Running on NPU...")
    result = npu_build.run(handle, [weights_buf, done_buf])
    got_done = done_buf.to_torch().numpy().astype(np.int32)
    if not np.all(got_done == FULL_LAYER_TOTAL_WEIGHT_CHUNKS):
        print(f"  done[0:8]: {got_done[:8].tolist()}")
        return False
    seconds = result.npu_time / 1e9
    gib_per_sec = (WEIGHT_BYTES / 2**30) / seconds if seconds > 0.0 else 0.0
    print(f"  NPU time: {result.npu_time / 1e3:.1f} us")
    print(f"  done[0:8]: {got_done[:8].tolist()}")
    print(
        "  perf_budget: "
        f"row1_weight_stream: bytes={WEIGHT_BYTES} chunks={FULL_LAYER_TOTAL_WEIGHT_CHUNKS} "
        f"gib_per_sec={gib_per_sec:.3f}"
    )
    print("  PASS: row1 full-layer weight stream reached all main16 DMA1 sinks")
    return True
