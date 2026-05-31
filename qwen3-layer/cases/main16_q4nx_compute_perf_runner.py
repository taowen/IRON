"""NPU runner for the main16 Q4NX compute-only performance slice."""

from __future__ import annotations

from pathlib import Path

import npu_build
import numpy as np
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

from cases import main16_q4nx_compute_perf_generate as generate
from projection_schedule import FULL_LAYER_TOTAL_WEIGHT_CHUNKS

CASE_NAME = generate.CASE_NAME
EXPERIMENT_DIR = Path(__file__).parent.parent
TOTAL_TILE_CHUNKS = generate.MAIN16_TILES * FULL_LAYER_TOTAL_WEIGHT_CHUNKS


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
        raise RuntimeError("\n".join(f"  MAIN16 Q4NX PERF STRUCTURE FAIL: {error}" for error in errors))
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
            print(f"  MAIN16 Q4NX PERF FAIL: {error}")
        return False
    print("  PASS: main16 Q4NX hot-body perf MLIR isolates the callable Q4NX body")
    print("  note=this is a hot-body microbench, not the generated layer scheduler path")
    print(f"  chunks_per_tile={FULL_LAYER_TOTAL_WEIGHT_CHUNKS}")
    print(f"  main16_tiles={generate.MAIN16_TILES}")
    print(f"  total_tile_chunks={TOTAL_TILE_CHUNKS}")
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


def _done_errors(done: np.ndarray) -> list[str]:
    expected = np.full((generate.DONE_DWORDS,), FULL_LAYER_TOTAL_WEIGHT_CHUNKS, dtype=np.int32)
    mismatches = np.flatnonzero(done != expected)
    errors: list[str] = []
    for idx in mismatches[:16]:
        errors.append(
            f"done[{int(idx)}] expected={int(expected[idx])} got={int(done[idx])}"
        )
    if mismatches.size > 16:
        errors.append(f"{int(mismatches.size - 16)} additional done mismatches")
    return errors


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
    print(f"  main16_kernel={generate.MAIN16_KERNEL_OBJECT}")
    print("  route=main16 core callable Q4NX hot body -> shim done output")
    print("  scope=hot-body microbench; production scheduler evidence comes from full-layer-qkv-prefix/full decode")
    print("  DMA=disabled for activation/weight; each tile reuses one local Q4NX chunk")
    print(f"  chunks_per_tile={FULL_LAYER_TOTAL_WEIGHT_CHUNKS}")
    print(f"  total_tile_chunks={TOTAL_TILE_CHUNKS}")
    print()

    xclbin_path, insts_path = build_kernel()
    print("  Loading NPU kernel...")
    handle = npu_build.load_kernel(xclbin_path, insts_path)
    done_buf = XRTTensor((generate.DONE_DWORDS,), dtype=np.int32)

    print("  Running on NPU...")
    result = npu_build.run(handle, [done_buf])
    done = done_buf.to_torch().numpy().astype(np.int32)
    seconds = result.npu_time / 1e9
    tile_chunks_per_sec = TOTAL_TILE_CHUNKS / seconds if seconds > 0.0 else 0.0
    print(f"  NPU time: {result.npu_time / 1e3:.1f} us")
    print(f"  done[0:8]: {done[:8].tolist()}")
    print(
        "  perf_budget: "
        f"main16_q4nx_compute: chunks_per_tile={FULL_LAYER_TOTAL_WEIGHT_CHUNKS} "
        f"kernel={generate.MAIN16_KERNEL_OBJECT} "
        f"total_tile_chunks={TOTAL_TILE_CHUNKS} tile_chunks_per_sec={tile_chunks_per_sec:.1f}"
    )
    errors = _done_errors(done)
    if errors:
        print(f"  FAIL: {len(errors)} main16 completion mismatches")
        for error in errors:
            print(f"    {error}")
        return False
    print("  PASS: all main16 tiles completed the full-layer Q4NX compute loop")
    return True
