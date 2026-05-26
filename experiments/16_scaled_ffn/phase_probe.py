#!/usr/bin/env python3
"""Phase-level diagnostics for experiment 16.

The full FFN has many moving pieces.  These probes keep the same shim,
memtile fanout, core DMA, output packet path, and Q4NX kernel, but stop at a
small phase boundary so the first bad boundary is reproducible.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from ml_dtypes import bfloat16

os.environ["PATH"] = "/var/opt/xilinx/xrt/bin:" + os.environ.get("PATH", "")

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

import aie.utils as aie_utils
from aie.utils.config import peano_install_dir, root_path
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from aie.utils.npukernel import NPUKernel

import generate
from preflight import preflight_check
from reference import pack_q4nx_chunk, q4nx_matvec_single
from run_npu import compile_kernels, compile_mlir


EXPERIMENT_DIR = Path(__file__).parent.resolve()
COPY_OBJ = EXPERIMENT_DIR / "copy_kernel.o"


def _compile_copy_kernel() -> bool:
    peano_dir = Path(peano_install_dir())
    mlir_aie_dir = Path(root_path())
    clang = peano_dir / "bin" / "clang++"
    include_path = mlir_aie_dir / "include"
    runtime_lib_include = mlir_aie_dir / "aie_runtime_lib" / "AIE2P"
    cmd = [
        str(clang),
        "-O2",
        "-std=c++20",
        "--target=aie2p-none-unknown-elf",
        "-Wno-parentheses",
        "-Wno-attributes",
        "-Wno-macro-redefined",
        "-Wno-empty-body",
        "-Wno-missing-template-arg-list-after-template-kw",
        f"-I{include_path}",
        f"-I{runtime_lib_include}",
        "-c",
        str(EXPERIMENT_DIR / "copy_kernel.cc"),
        "-o",
        str(COPY_OBJ),
    ]
    return os.system(" ".join(cmd)) == 0


def _q4_loop(prefix: str, phase_offset: int, out_name: str) -> str:
    return f"""
      scf.for %chunk = %c0 to %num_chunks step %c1 {{
        %chunk_i32 = arith.index_cast %chunk : index to i32
        %act_offset = arith.muli %chunk_i32, %k_chunk_i32 : i32
        %shifted = arith.addi %chunk_i32, %phase_offset_i32 : i32
        %rem = arith.remsi %shifted, %c2_i32 : i32
        %is_pong = arith.cmpi eq, %rem, %c1_i32 : i32

        scf.if %is_pong {{
          aie.use_lock(%{prefix}_wt_pong_full, AcquireGreaterEqual, 1)
          func.call @q4nx_chunk_accum_offset(%{prefix}_wt_pong, %{prefix}_act, %act_offset, %m_i32, %{prefix}_accum_buf)
            : (memref<{generate.CHUNK_BF16}xbf16>, memref<{generate.ACT_BF16}xbf16>, i32, i32, memref<{generate.ACCUM_F32}xf32>) -> ()
          aie.use_lock(%{prefix}_wt_pong_empty, Release, 1)
        }} else {{
          aie.use_lock(%{prefix}_wt_ping_full, AcquireGreaterEqual, 1)
          func.call @q4nx_chunk_accum_offset(%{prefix}_wt_ping, %{prefix}_act, %act_offset, %m_i32, %{prefix}_accum_buf)
            : (memref<{generate.CHUNK_BF16}xbf16>, memref<{generate.ACT_BF16}xbf16>, i32, i32, memref<{generate.ACCUM_F32}xf32>) -> ()
          aie.use_lock(%{prefix}_wt_ping_empty, Release, 1)
        }}
      }}
      func.call @q4nx_flush_output(%{prefix}_{out_name}, %m_i32, %{prefix}_accum_buf)
        : (memref<{generate.OUT_BF16}xbf16>, i32, memref<{generate.ACCUM_F32}xf32>) -> ()
"""


def _probe_core_program(mode: str, col: int, row: int, prefix: str) -> str:
    del col, row
    core_mode = mode.removesuffix("-guard").removesuffix("-extra")
    gate_to = "out" if core_mode == "gate-direct" else "gate_buf"
    body = _q4_loop(prefix, 0, gate_to)
    if core_mode == "gate-after-up":
        body += _q4_loop(prefix, generate.NUM_CHUNKS % 2, "up_buf")

    if core_mode in {"gate-buffer", "gate-after-up"}:
        body += f"""
      aie.use_lock(%{prefix}_out_prod, AcquireGreaterEqual, 1)
      func.call @copy_section(%{prefix}_gate_buf, %{prefix}_out, %c0_i32)
        : (memref<{generate.OUT_BF16}xbf16>, memref<{generate.OUT_BF16}xbf16>, i32) -> ()
      aie.use_lock(%{prefix}_out_cons, Release, 1)
"""
    else:
        body = body.replace(
            f"func.call @q4nx_flush_output(%{prefix}_out, %m_i32, %{prefix}_accum_buf)",
            f"aie.use_lock(%{prefix}_out_prod, AcquireGreaterEqual, 1)\n"
            f"      func.call @q4nx_flush_output(%{prefix}_out, %m_i32, %{prefix}_accum_buf)",
        )
        body += f"      aie.use_lock(%{prefix}_out_cons, Release, 1)\n"

    return f"""    %core_{prefix} = aie.core(%{prefix}) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2_i32 = arith.constant 2 : i32
      %c1_i32 = arith.constant 1 : i32
      %c0_i32 = arith.constant 0 : i32
      %num_chunks = arith.constant {generate.NUM_CHUNKS} : index
      %k_chunk_i32 = arith.constant {generate.K_CHUNK} : i32
      %m_i32 = arith.constant {generate.M_PER_TILE} : i32
      %phase_offset_i32 = arith.constant 0 : i32

      aie.use_lock(%{prefix}_act_full, AcquireGreaterEqual, 1)
{body}
      aie.use_lock(%{prefix}_act_empty, Release, 1)
      aie.end
    }}"""


def _configure_generate(pushes_per_col: int) -> None:
    generate.PUSHES_PER_COL = pushes_per_col
    generate.PER_COL_WT_BF16 = pushes_per_col * generate.FAT_CHUNK_BF16
    generate.TOTAL_WT_I32 = (generate.NUM_COLS * generate.PER_COL_WT_BF16 * 2) // 4


def _generate_probe_mlir(mode: str) -> str:
    original = generate._core_program
    generate._core_program = lambda col, row, prefix: _probe_core_program(mode, col, row, prefix)
    try:
        mlir_text = generate.generate_mlir()
    finally:
        generate._core_program = original

    declaration = (
        f"    func.func private @copy_section(memref<{generate.OUT_BF16}xbf16>, "
        f"memref<{generate.OUT_BF16}xbf16>, i32) attributes "
        f'{{link_with = "{COPY_OBJ}"}}\n'
    )
    mlir_text = mlir_text.replace("    // === Core programs ===", declaration + "\n    // === Core programs ===")
    if mode.endswith("-guard"):
        mlir_text = mlir_text.replace(
            f"memref<{generate.CHUNK_BF16}xbf16>",
            f"memref<{generate.CHUNK_BF16 * 2}xbf16>",
        )
    return mlir_text


def _make_data() -> tuple[np.ndarray, list[list[list[np.ndarray]]], list[list[list[np.ndarray]]], list[list[list[np.ndarray]]]]:
    np.random.seed(42)
    activation = np.random.randn(generate.K_HIDDEN).astype(bfloat16)
    scales = [[None] * generate.ROWS_PER_COL for _ in range(generate.NUM_COLS)]
    zeros = [[None] * generate.ROWS_PER_COL for _ in range(generate.NUM_COLS)]
    int4 = [[None] * generate.ROWS_PER_COL for _ in range(generate.NUM_COLS)]
    for col in range(generate.NUM_COLS):
        for row in range(generate.ROWS_PER_COL):
            scales[col][row] = [
                np.random.uniform(
                    0.01,
                    0.5,
                    (generate.M_PER_TILE, generate.K_HIDDEN // generate.GROUP_SIZE),
                ).astype(bfloat16)
                for _ in range(generate.NUM_PHASES_PROJ)
            ]
            zeros[col][row] = [
                np.random.uniform(
                    4,
                    12,
                    (generate.M_PER_TILE, generate.K_HIDDEN // generate.GROUP_SIZE),
                ).astype(bfloat16)
                for _ in range(generate.NUM_PHASES_PROJ)
            ]
            int4[col][row] = [
                np.random.randint(
                    0,
                    16,
                    (generate.M_PER_TILE, generate.K_HIDDEN),
                    dtype=np.uint8,
                )
                for _ in range(generate.NUM_PHASES_PROJ)
            ]
    return activation, scales, zeros, int4


def _pack_projection_weights(
    phases: list[int],
    scales: list[list[list[np.ndarray]]],
    zeros: list[list[list[np.ndarray]]],
    int4: list[list[list[np.ndarray]]],
) -> np.ndarray:
    groups_per_row = generate.K_CHUNK // generate.GROUP_SIZE
    chunks: list[np.ndarray] = []
    for col in range(generate.NUM_COLS):
        for phase in phases:
            for chunk_idx in range(generate.NUM_CHUNKS):
                for row in range(generate.ROWS_PER_COL):
                    g0 = chunk_idx * groups_per_row
                    g1 = g0 + groups_per_row
                    k0 = chunk_idx * generate.K_CHUNK
                    k1 = k0 + generate.K_CHUNK
                    chunks.append(
                        pack_q4nx_chunk(
                            scales[col][row][phase][:, g0:g1],
                            zeros[col][row][phase][:, g0:g1],
                            int4[col][row][phase][:, k0:k1],
                            generate.M_PER_TILE,
                            generate.K_CHUNK,
                            generate.GROUP_SIZE,
                        )
                    )
    packed = np.concatenate(chunks)
    expected_bytes = generate.TOTAL_WT_I32 * 4
    if packed.shape[0] != expected_bytes:
        raise RuntimeError(f"packed weight size {packed.shape[0]} != {expected_bytes}")
    return packed


def _gate_reference(packed: np.ndarray, activation: np.ndarray) -> np.ndarray:
    chunk_bytes = generate.CHUNK_BF16 * 2
    fat_chunk_bytes = generate.FAT_CHUNK_BF16 * 2
    out = np.zeros(generate.TOTAL_OUTPUT, dtype=bfloat16)
    for col in range(generate.NUM_COLS):
        for row in range(generate.ROWS_PER_COL):
            row_chunks = bytearray()
            for chunk_idx in range(generate.NUM_CHUNKS):
                fat_offset = col * generate.PER_COL_WT_BF16 * 2 + chunk_idx * fat_chunk_bytes
                row_offset = fat_offset + row * chunk_bytes
                row_chunks.extend(packed[row_offset:row_offset + chunk_bytes])
            row_packed = np.frombuffer(bytes(row_chunks), dtype=np.uint8)
            idx = (col * generate.ROWS_PER_COL + row) * generate.M_PER_TILE
            out[idx:idx + generate.M_PER_TILE] = q4nx_matvec_single(
                row_packed,
                activation,
                generate.K_HIDDEN,
                generate.K_CHUNK,
                generate.GROUP_SIZE,
            )
    return out


def _run(mode: str) -> bool:
    phases = [0, 1] if "extra" in mode or mode.startswith("gate-after-up") else [0]
    _configure_generate(len(phases) * generate.NUM_CHUNKS)

    build_dir = EXPERIMENT_DIR / f"build_probe_{mode.replace('-', '_')}"
    build_dir.mkdir(exist_ok=True)
    mlir_path = build_dir / "design.mlir"
    mlir_text = _generate_probe_mlir(mode)
    mlir_path.write_text(mlir_text)

    errors = preflight_check(mlir_text)
    if errors:
        print("PREFLIGHT FAILED:")
        for error in errors:
            print(f"  {error}")
        return False

    if not compile_kernels(build_dir):
        return False
    if not _compile_copy_kernel():
        return False
    ok, xclbin_path, insts_path = compile_mlir(build_dir, mlir_path)
    if not ok:
        return False

    activation, scales, zeros, int4 = _make_data()
    packed = _pack_projection_weights(phases, scales, zeros, int4)
    ref = _gate_reference(packed, activation)

    kernel = NPUKernel(
        xclbin_path=str(xclbin_path),
        kernel_name="MLIR_AIE",
        insts_path=str(insts_path),
    )
    handle = aie_utils.DefaultNPURuntime.load(kernel)
    wt_buf = XRTTensor.from_torch(torch.from_numpy(packed.view(np.int32).copy()).to(torch.int32))
    act_i32 = np.frombuffer(activation.tobytes(), dtype=np.int32)
    act_buf = XRTTensor.from_torch(torch.from_numpy(act_i32.copy()).to(torch.int32))
    out_buf = XRTTensor((generate.OUT_TOTAL_I32,), dtype=np.int32)

    result = aie_utils.DefaultNPURuntime.run(handle, [wt_buf, act_buf, out_buf])
    npu = np.frombuffer(out_buf.to_torch().numpy().tobytes(), dtype=bfloat16)
    print(f"mode={mode} npu_time_us={result.npu_time / 1e3:.1f}")
    print(f"ref[0:8]={ref[:8]}")
    print(f"npu[0:8]={npu[:8]}")

    ref_f32 = ref.astype(np.float32)
    npu_f32 = npu.astype(np.float32)
    finite = np.isfinite(ref_f32) & np.isfinite(npu_f32)
    abs_err = np.abs(ref_f32 - npu_f32)
    max_abs = np.max(abs_err[finite]) if np.any(finite) else np.inf
    bad = (~finite) | (abs_err > 1.0)
    print(f"finite={np.all(finite)} max_abs={max_abs:.4f} bad={int(np.sum(bad))}/{ref.shape[0]}")
    if np.any(bad):
        for idx in np.where(bad)[0][:12]:
            print(f"  [{idx}] ref={ref_f32[idx]:.4f} npu={npu_f32[idx]:.4f} err={abs_err[idx]:.4f}")
    return not np.any(bad)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=[
            "gate-direct",
            "gate-buffer",
            "gate-buffer-extra",
            "gate-buffer-extra-guard",
            "gate-after-up",
        ],
    )
    args = parser.parse_args()
    try:
        return 0 if _run(args.mode) else 1
    finally:
        aie_utils.DefaultNPURuntime.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
