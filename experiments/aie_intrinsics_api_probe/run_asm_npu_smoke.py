#!/usr/bin/env python3
"""Run a minimal NPU smoke test for source assembly linked into an AIE core."""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from ml_dtypes import bfloat16

os.environ["PATH"] = "/var/opt/xilinx/xrt/bin:" + os.environ.get("PATH", "")

REPO_ROOT = Path(__file__).resolve().parents[2]
QWEN3_LAYER_DIR = REPO_ROOT / "qwen3-layer"
sys.path.insert(0, str(QWEN3_LAYER_DIR))

import aie.utils as aie_utils
from aie.utils.config import peano_install_dir, root_path
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from aie.utils.npukernel import NPUKernel

from mlir_utils import flow, npu_address_patch, npu_push_queue, npu_sync, npu_writebd
from run_asm_link_probe import compile_and_link, run_command, toolchain

CASE_NAME = "asm-q4nx-exact-npu-smoke"
BUILD_DIR = Path(__file__).resolve().parent / "build" / CASE_NAME
SRC_DWORDS = 1280
DST_DWORDS = 64
PACKED_GROUP_BYTES = 0x200
SCALE_OFFSET_BYTES = 0x1000
OFFSET_OFFSET_BYTES = 0x1100
ACTIVATION_OFFSET_BYTES = 0x1200
Q4_GROUPS = 8
ROWS_PER_LANE = 16
GROUP_SIZE = 32
HOST_INPUT_CHANNEL = 0
TILE_INPUT_CHANNEL = 1
TILE_OUTPUT_CHANNEL = 0
HOST_OUTPUT_CHANNEL = 1


def generate_mlir(link_object: Path) -> str:
    return f"""module {{
  aie.device(npu2) {{
    %shim = aie.tile(1, 0)
    %tile = aie.tile(1, 2)

    // case marker {CASE_NAME}
{flow("shim", HOST_INPUT_CHANNEL, "tile", TILE_INPUT_CHANNEL)}
{flow("tile", TILE_OUTPUT_CHANNEL, "shim", HOST_OUTPUT_CHANNEL)}

    func.func private @probe_asm_q4_exact_lane8_loop_release(memref<{SRC_DWORDS}xi32>, memref<{DST_DWORDS}xi32>) attributes {{link_with = "{link_object.resolve()}"}}

    %src = aie.buffer(%tile) {{sym_name = "src"}} : memref<{SRC_DWORDS}xi32>
    %dst = aie.buffer(%tile) {{sym_name = "dst"}} : memref<{DST_DWORDS}xi32>
    %src_empty = aie.lock(%tile, 0) {{init = 1 : i32, sym_name = "src_empty"}}
    %src_full = aie.lock(%tile, 1) {{init = 0 : i32, sym_name = "src_full"}}
    %dst_empty = aie.lock(%tile, 4) {{init = 1 : i32, sym_name = "dst_empty"}}
    %dst_full = aie.lock(%tile, 5) {{init = 0 : i32, sym_name = "dst_full"}}

    %tile_core = aie.core(%tile) {{
      aie.use_lock(%src_full, AcquireGreaterEqual, 1)
      aie.use_lock(%dst_empty, AcquireGreaterEqual, 1)
      func.call @probe_asm_q4_exact_lane8_loop_release(%src, %dst)
        : (memref<{SRC_DWORDS}xi32>, memref<{DST_DWORDS}xi32>) -> ()
      aie.use_lock(%src_empty, Release, 1)
      aie.end
    }}

    %tile_mem = aie.mem(%tile) {{
      %src_dma = aie.dma_start(S2MM, {TILE_INPUT_CHANNEL}, ^src_in, ^dst_start)
    ^src_in:
      aie.use_lock(%src_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%src : memref<{SRC_DWORDS}xi32>, 0, {SRC_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%src_full, Release, 1)
      aie.next_bd ^src_end
    ^src_end:
      aie.end

    ^dst_start:
      %dst_dma = aie.dma_start(MM2S, {TILE_OUTPUT_CHANNEL}, ^dst_out, ^end)
    ^dst_out:
      aie.use_lock(%dst_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%dst : memref<{DST_DWORDS}xi32>, 0, {DST_DWORDS}) {{bd_id = 2 : i32}}
      aie.use_lock(%dst_empty, Release, 1)
      aie.next_bd ^end
    ^end:
      aie.end
    }}

    aie.runtime_sequence(%src_arg: memref<{SRC_DWORDS}xi32>, %dst_arg: memref<{DST_DWORDS}xi32>) {{
{npu_writebd(1, 0, SRC_DWORDS, 0)}
{npu_address_patch(1, 0, 0, 0)}
{npu_push_queue(1, "MM2S", HOST_INPUT_CHANNEL, 0)}
{npu_writebd(1, 2, DST_DWORDS, 0)}
{npu_address_patch(1, 2, 1, 0)}
{npu_push_queue(1, "S2MM", HOST_OUTPUT_CHANNEL, 2)}
{npu_sync(1, HOST_OUTPUT_CHANNEL)}
{npu_sync(1, HOST_INPUT_CHANNEL, direction=1)}
    }}
  }}
}}
"""


def bf16_words(values: np.ndarray) -> np.ndarray:
    return np.frombuffer(values.astype(bfloat16).tobytes(), dtype=np.int32).copy()


def prepare_q4nx_smoke_input() -> tuple[np.ndarray, np.ndarray]:
    src_bytes = bytearray(SRC_DWORDS * 4)
    for idx in range(Q4_GROUPS * PACKED_GROUP_BYTES):
        src_bytes[idx] = 0x11

    scale = np.array(
        [
            bfloat16(0.0625 + group / 64.0 + lane / 256.0)
            for group in range(Q4_GROUPS)
            for lane in range(ROWS_PER_LANE)
        ],
        dtype=bfloat16,
    )
    offset = np.array(
        [
            bfloat16(-0.03125 + group / 512.0 + lane / 512.0)
            for group in range(Q4_GROUPS)
            for lane in range(ROWS_PER_LANE)
        ],
        dtype=bfloat16,
    )
    activation = np.array(
        [
            bfloat16((((dim * 5 + group * 7) % 17) - 8) / 8.0)
            for group in range(Q4_GROUPS)
            for dim in range(GROUP_SIZE)
        ],
        dtype=bfloat16,
    )

    src_bytes[SCALE_OFFSET_BYTES:SCALE_OFFSET_BYTES + scale.nbytes] = scale.tobytes()
    src_bytes[OFFSET_OFFSET_BYTES:OFFSET_OFFSET_BYTES + offset.nbytes] = offset.tobytes()
    src_bytes[
        ACTIVATION_OFFSET_BYTES:ACTIVATION_OFFSET_BYTES + activation.nbytes
    ] = activation.tobytes()

    scale_by_group = scale.astype(np.float32).reshape(Q4_GROUPS, ROWS_PER_LANE)
    offset_by_group = offset.astype(np.float32).reshape(Q4_GROUPS, ROWS_PER_LANE)
    activation_by_group = activation.astype(np.float32).reshape(Q4_GROUPS, GROUP_SIZE)
    output_fp32 = np.zeros((ROWS_PER_LANE,), dtype=np.float32)
    for group in range(Q4_GROUPS):
        scaled = (scale_by_group[group] * np.float32(1.0)).astype(bfloat16).astype(np.float32)
        dequant = (scaled + offset_by_group[group]).astype(bfloat16).astype(np.float32)
        activation_sum = np.sum(activation_by_group[group], dtype=np.float32)
        np.add(output_fp32, dequant * activation_sum, out=output_fp32)
    output = output_fp32.astype(bfloat16)
    expected = np.zeros((DST_DWORDS,), dtype=np.int32)
    expected[:8] = bf16_words(output)
    return np.frombuffer(bytes(src_bytes), dtype=np.int32).copy(), expected


def compile_mlir(mlir_path: Path, xclbin_path: Path, insts_path: Path) -> None:
    mlir_aie_dir = Path(root_path())
    peano_dir = Path(peano_install_dir())
    aiecc = mlir_aie_dir / "bin" / "aiecc"
    tmpdir = BUILD_DIR / "aiecc"
    tmpdir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(aiecc),
        "-v",
        "-j1",
        f"--tmpdir={tmpdir}",
        "--no-compile-host",
        "--no-xchesscc",
        "--no-xbridge",
        "--alloc-scheme=basic-sequential",
        "--peano",
        str(peano_dir),
        "--aie-generate-xclbin",
        f"--xclbin-name={xclbin_path}",
        "--xclbin-kernel-name=MLIR_AIE",
        "--aie-generate-npu-insts",
        f"--npu-insts-name={insts_path}",
        str(mlir_path),
    ]
    print("  Compiling MLIR...")
    run_command(tuple(cmd))


def load_kernel(xclbin_path: Path, insts_path: Path):
    kernel = NPUKernel(
        xclbin_path=str(xclbin_path),
        kernel_name="MLIR_AIE",
        insts_path=str(insts_path),
    )
    return aie_utils.DefaultNPURuntime.load(kernel)


def run_smoke() -> bool:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    tools = toolchain(Path(peano_install_dir()))
    artifacts = compile_and_link(
        tools=tools,
        mlir_aie_install=Path(root_path()),
        output_dir=BUILD_DIR,
    )
    mlir_path = BUILD_DIR / "design.mlir"
    xclbin_path = BUILD_DIR / "design.xclbin"
    insts_path = BUILD_DIR / "design.bin"
    mlir_path.write_text(generate_mlir(artifacts.combined_object))
    compile_mlir(mlir_path, xclbin_path, insts_path)

    src, expected = prepare_q4nx_smoke_input()
    src_buf = XRTTensor.from_torch(torch.from_numpy(src.copy()).to(torch.int32))
    dst_buf = XRTTensor((DST_DWORDS,), dtype=np.int32)

    print("  Loading NPU kernel...")
    handle = load_kernel(xclbin_path, insts_path)
    print("  Running on NPU...")
    result = aie_utils.DefaultNPURuntime.run(handle, [src_buf, dst_buf])
    got = dst_buf.to_torch().numpy().astype(np.int32)
    print(f"  NPU time: {result.npu_time / 1e3:.1f} us")
    print(f"  expected[0:8]: {expected[:8].tolist()}")
    print(f"  got[0:8]: {got[:8].tolist()}")
    mismatches = np.flatnonzero(got != expected)
    if mismatches.size:
        print(f"  FAIL: {int(mismatches.size)} asm Q4NX exact smoke mismatches")
        for idx in mismatches[:16]:
            print(f"    dst[{int(idx)}] expected={int(expected[idx])} got={int(got[idx])}")
        return False
    print("  PASS: source-assembly Q4NX exact lane8 loop matched the BF16 reference on NPU")
    return True


def main() -> int:
    try:
        return 0 if run_smoke() else 1
    except Exception as exc:
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1
    finally:
        aie_utils.DefaultNPURuntime.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
