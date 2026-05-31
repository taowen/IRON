#!/usr/bin/env python3
"""Run an NPU smoke test for callable in-place FP32 accumulator assembly."""

from __future__ import annotations

import os
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path

os.environ["PATH"] = "/var/opt/xilinx/xrt/bin:" + os.environ.get("PATH", "")
REPO_ROOT = Path(__file__).resolve().parents[2]
QWEN3_LAYER_DIR = REPO_ROOT / "qwen3-layer"
sys.path.insert(0, str(QWEN3_LAYER_DIR))

import aie.utils as aie_utils
import numpy as np
import torch
from aie.utils.config import peano_install_dir, root_path
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from aie.utils.npukernel import NPUKernel

from mlir_utils import flow, npu_address_patch, npu_push_queue, npu_sync, npu_writebd

EXPERIMENT_DIR = Path(__file__).resolve().parent
BUILD_DIR = EXPERIMENT_DIR / "build"
CPP_SOURCE = EXPERIMENT_DIR / "accum_probe.cc"
ASM_SOURCE = EXPERIMENT_DIR / "accum_probe.s"
CASE_NAME = "aie2p-float-accum-inplace-asm"
DST_DWORDS = 16
HOST_INPUT_CHANNEL = 0
TILE_INPUT_CHANNEL = 1
TILE_OUTPUT_CHANNEL = 0
HOST_OUTPUT_CHANNEL = 1


@dataclass(frozen=True)
class Toolchain:
    clang: Path
    clangxx: Path
    linker: Path
    objdump: Path


@dataclass(frozen=True)
class Artifacts:
    combined_object: Path
    disasm: Path
    mlir: Path
    xclbin: Path
    insts: Path


def checked_run(cmd: tuple[str, ...]) -> str:
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "command failed:\n"
            + " ".join(cmd)
            + "\nstdout:\n"
            + completed.stdout
            + "\nstderr:\n"
            + completed.stderr
        )
    return completed.stdout


def toolchain() -> Toolchain:
    peano = Path(peano_install_dir())
    return Toolchain(
        clang=peano / "bin/clang",
        clangxx=peano / "bin/clang++",
        linker=peano / "bin/ld.lld",
        objdump=peano / "bin/llvm-objdump",
    )


def compile_role_object(tools: Toolchain) -> Artifacts:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    cpp_object = BUILD_DIR / "accum_probe.cc.o"
    asm_object = BUILD_DIR / "accum_probe.s.o"
    combined_object = BUILD_DIR / "accum_probe.o"
    disasm = BUILD_DIR / "accum_probe.disasm"
    mlir = BUILD_DIR / "design.mlir"
    xclbin = BUILD_DIR / "design.xclbin"
    insts = BUILD_DIR / "design.bin"
    mlir_aie = Path(root_path())
    include_dir = mlir_aie / "include"
    runtime_include = mlir_aie / "aie_runtime_lib/AIE2P"
    checked_run(
        (
            str(tools.clangxx),
            "-O2",
            "-std=c++20",
            "--target=aie2p-none-unknown-elf",
            "-ffunction-sections",
            "-fdata-sections",
            f"-I{include_dir}",
            f"-I{runtime_include}",
            "-c",
            str(CPP_SOURCE),
            "-o",
            str(cpp_object),
        )
    )
    checked_run(
        (
            str(tools.clang),
            "--target=aie2p-none-unknown-elf",
            "-c",
            str(ASM_SOURCE),
            "-o",
            str(asm_object),
        )
    )
    checked_run((str(tools.linker), "-r", str(cpp_object), str(asm_object), "-o", str(combined_object)))
    disasm.write_text(
        checked_run(
            (
                str(tools.objdump),
                "--triple=aie2p",
                "-dr",
                "--no-print-imm-hex",
                str(combined_object),
            )
        )
    )
    return Artifacts(
        combined_object=combined_object,
        disasm=disasm,
        mlir=mlir,
        xclbin=xclbin,
        insts=insts,
    )


def generate_mlir(link_object: Path) -> str:
    return f"""module {{
  aie.device(npu2) {{
    %shim = aie.tile(1, 0)
    %tile = aie.tile(1, 2)

    // case marker {CASE_NAME}
{flow("shim", HOST_INPUT_CHANNEL, "tile", TILE_INPUT_CHANNEL)}
{flow("tile", TILE_OUTPUT_CHANNEL, "shim", HOST_OUTPUT_CHANNEL)}

    func.func private @cpp_call_asm_float_accum_inplace(memref<{DST_DWORDS}xf32>) attributes {{link_with = "{link_object.resolve()}"}}

    %dst = aie.buffer(%tile) {{sym_name = "dst"}} : memref<{DST_DWORDS}xf32>
    %dst_empty = aie.lock(%tile, 0) {{init = 1 : i32, sym_name = "dst_empty"}}
    %dst_full = aie.lock(%tile, 1) {{init = 0 : i32, sym_name = "dst_full"}}
    %out_full = aie.lock(%tile, 2) {{init = 0 : i32, sym_name = "out_full"}}

    %tile_core = aie.core(%tile) {{
      aie.use_lock(%dst_full, AcquireGreaterEqual, 1)
      func.call @cpp_call_asm_float_accum_inplace(%dst) : (memref<{DST_DWORDS}xf32>) -> ()
      aie.use_lock(%out_full, Release, 1)
      aie.end
    }}

    %tile_mem = aie.mem(%tile) {{
      %in_dma = aie.dma_start(S2MM, {TILE_INPUT_CHANNEL}, ^dst_in, ^out_start)
    ^dst_in:
      aie.use_lock(%dst_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%dst : memref<{DST_DWORDS}xf32>, 0, {DST_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%dst_full, Release, 1)
      aie.next_bd ^dst_in_end
    ^dst_in_end:
      aie.end

    ^out_start:
      %out_dma = aie.dma_start(MM2S, {TILE_OUTPUT_CHANNEL}, ^dst_out, ^end)
    ^dst_out:
      aie.use_lock(%out_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%dst : memref<{DST_DWORDS}xf32>, 0, {DST_DWORDS}) {{bd_id = 2 : i32}}
      aie.use_lock(%dst_empty, Release, 1)
      aie.next_bd ^end
    ^end:
      aie.end
    }}

    aie.runtime_sequence(%dst_in_arg: memref<{DST_DWORDS}xi32>, %dst_out_arg: memref<{DST_DWORDS}xi32>) {{
{npu_writebd(1, 0, DST_DWORDS, 0)}
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


def compile_mlir(artifacts: Artifacts) -> None:
    mlir_aie = Path(root_path())
    peano = Path(peano_install_dir())
    aiecc = mlir_aie / "bin/aiecc"
    tmpdir = BUILD_DIR / "aiecc"
    tmpdir.mkdir(parents=True, exist_ok=True)
    checked_run(
        (
            str(aiecc),
            "-v",
            "-j1",
            f"--tmpdir={tmpdir}",
            "--no-compile-host",
            "--no-xchesscc",
            "--no-xbridge",
            "--alloc-scheme=basic-sequential",
            "--peano",
            str(peano),
            "--aie-generate-xclbin",
            f"--xclbin-name={artifacts.xclbin}",
            "--xclbin-kernel-name=MLIR_AIE",
            "--aie-generate-npu-insts",
            f"--npu-insts-name={artifacts.insts}",
            str(artifacts.mlir),
        )
    )


def load_kernel(artifacts: Artifacts):
    kernel = NPUKernel(
        xclbin_path=str(artifacts.xclbin),
        kernel_name="MLIR_AIE",
        insts_path=str(artifacts.insts),
    )
    return aie_utils.DefaultNPURuntime.load(kernel)


def run_smoke() -> bool:
    artifacts = compile_role_object(toolchain())
    artifacts.mlir.write_text(generate_mlir(artifacts.combined_object))
    compile_mlir(artifacts)
    input_values = np.zeros((DST_DWORDS,), dtype=np.float32)
    expected = np.ones((DST_DWORDS,), dtype=np.float32)
    input_buf = XRTTensor.from_torch(torch.from_numpy(input_values.view(np.int32).copy()).to(torch.int32))
    output_buf = XRTTensor((DST_DWORDS,), dtype=np.int32)
    print("  Loading NPU kernel...")
    handle = load_kernel(artifacts)
    print("  Running on NPU...")
    result = aie_utils.DefaultNPURuntime.run(handle, [input_buf, output_buf])
    got_bits = output_buf.to_torch().numpy().astype(np.int32)
    got = got_bits.view(np.float32)
    print(f"  NPU time: {result.npu_time / 1e3:.1f} us")
    print(f"  got[0:8]: {got[:8].tolist()}")
    mismatches = np.flatnonzero(got_bits != expected.view(np.int32))
    if mismatches.size:
        print(f"  FAIL: {int(mismatches.size)} FP32 accumulator mismatches")
        for idx in mismatches[:16]:
            print(
                f"    dst[{int(idx)}] expected={float(expected[idx])} "
                f"got={float(got[idx])} bits=0x{int(got_bits[idx]) & 0xffffffff:08x}"
            )
        return False
    print("  PASS: callable source assembly loaded, updated, and stored FP32 accumulator in place")
    print(f"  disasm={artifacts.disasm}")
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
