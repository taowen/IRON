#!/usr/bin/env python3
"""Calibrate AIE2P vmac.f operand layout on real NPU."""

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
CPP_SOURCE = EXPERIMENT_DIR / "operand_probe.cc"
ASM_SOURCE = EXPERIMENT_DIR / "operand_probe.s"
WORKSPACE_DWORDS = 1024
OUTPUT_BLOCK_FLOATS = 16
OUTPUT_BLOCKS = 39
ACTIVATION_DWORD_OFFSET = 640
ZERO_DWORD_OFFSET = 768
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
    build_dir: Path
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
    build_dir = EXPERIMENT_DIR / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    cpp_object = build_dir / "operand_probe.cc.o"
    asm_object = build_dir / "operand_probe.s.o"
    combined_object = build_dir / "operand_probe.o"
    disasm = build_dir / "operand_probe.disasm"
    mlir = build_dir / "design.mlir"
    xclbin = build_dir / "design.xclbin"
    insts = build_dir / "design.bin"
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
    checked_run((str(tools.clang), "--target=aie2p-none-unknown-elf", "-c", str(ASM_SOURCE), "-o", str(asm_object)))
    checked_run((str(tools.linker), "-r", str(cpp_object), str(asm_object), "-o", str(combined_object)))
    disasm.write_text(
        checked_run((str(tools.objdump), "--triple=aie2p", "-dr", "--no-print-imm-hex", str(combined_object)))
    )
    return Artifacts(build_dir, combined_object, disasm, mlir, xclbin, insts)


def generate_mlir(link_object: Path) -> str:
    return f"""module {{
  aie.device(npu2) {{
    %shim = aie.tile(1, 0)
    %tile = aie.tile(1, 2)

    // case marker aie2p-vmac-operand-layout
{flow("shim", HOST_INPUT_CHANNEL, "tile", TILE_INPUT_CHANNEL)}
{flow("tile", TILE_OUTPUT_CHANNEL, "shim", HOST_OUTPUT_CHANNEL)}

    func.func private @cpp_call_vmac_operand_layout(memref<{WORKSPACE_DWORDS}xi32>) attributes {{link_with = "{link_object.resolve()}"}}

    %workspace = aie.buffer(%tile) {{sym_name = "workspace"}} : memref<{WORKSPACE_DWORDS}xi32>
    %workspace_empty = aie.lock(%tile, 0) {{init = 1 : i32, sym_name = "workspace_empty"}}
    %workspace_full = aie.lock(%tile, 1) {{init = 0 : i32, sym_name = "workspace_full"}}
    %out_full = aie.lock(%tile, 2) {{init = 0 : i32, sym_name = "out_full"}}

    %tile_core = aie.core(%tile) {{
      aie.use_lock(%workspace_full, AcquireGreaterEqual, 1)
      func.call @cpp_call_vmac_operand_layout(%workspace) : (memref<{WORKSPACE_DWORDS}xi32>) -> ()
      aie.use_lock(%out_full, Release, 1)
      aie.end
    }}

    %tile_mem = aie.mem(%tile) {{
      %in_dma = aie.dma_start(S2MM, {TILE_INPUT_CHANNEL}, ^workspace_in, ^out_start)
    ^workspace_in:
      aie.use_lock(%workspace_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%workspace : memref<{WORKSPACE_DWORDS}xi32>, 0, {WORKSPACE_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%workspace_full, Release, 1)
      aie.next_bd ^workspace_in_end
    ^workspace_in_end:
      aie.end

    ^out_start:
      %out_dma = aie.dma_start(MM2S, {TILE_OUTPUT_CHANNEL}, ^workspace_out, ^end)
    ^workspace_out:
      aie.use_lock(%out_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%workspace : memref<{WORKSPACE_DWORDS}xi32>, 0, {WORKSPACE_DWORDS}) {{bd_id = 2 : i32}}
      aie.use_lock(%workspace_empty, Release, 1)
      aie.next_bd ^end
    ^end:
      aie.end
    }}

    aie.runtime_sequence(%workspace_in_arg: memref<{WORKSPACE_DWORDS}xi32>, %workspace_out_arg: memref<{WORKSPACE_DWORDS}xi32>) {{
{npu_writebd(1, 0, WORKSPACE_DWORDS, 0)}
{npu_address_patch(1, 0, 0, 0)}
{npu_push_queue(1, "MM2S", HOST_INPUT_CHANNEL, 0)}
{npu_writebd(1, 2, WORKSPACE_DWORDS, 0)}
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
    tmpdir = artifacts.build_dir / "aiecc"
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


def bf16_words(values: np.ndarray) -> np.ndarray:
    return (values.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def make_workspace() -> np.ndarray:
    workspace = np.zeros((WORKSPACE_DWORDS,), dtype=np.int32)
    activation = np.arange(1, 33, dtype=np.float32)
    workspace.view(np.uint16)[
        ACTIVATION_DWORD_OFFSET * 2 : ACTIVATION_DWORD_OFFSET * 2 + activation.shape[0]
    ] = bf16_words(activation)
    workspace[ZERO_DWORD_OFFSET : ZERO_DWORD_OFFSET + OUTPUT_BLOCK_FLOATS] = 0
    return workspace


def load_kernel(artifacts: Artifacts):
    kernel = NPUKernel(
        xclbin_path=str(artifacts.xclbin),
        kernel_name="MLIR_AIE",
        insts_path=str(artifacts.insts),
    )
    return aie_utils.DefaultNPURuntime.load(kernel)


def validate_output(got_workspace: np.ndarray) -> bool:
    blocks = got_workspace[: OUTPUT_BLOCKS * OUTPUT_BLOCK_FLOATS].view(np.float32).reshape(
        OUTPUT_BLOCKS, OUTPUT_BLOCK_FLOATS
    )
    block_names = ["qreg_x2", "qreg_x3", "qreg_x5", "qreg_x7", "qreg_x9"]
    block_names.extend(f"vext_lane{lane}" for lane in range(32))
    block_names.extend(("correction_2x32", "sum_32_vext_lanes"))
    expected_first_lane = np.empty((OUTPUT_BLOCKS,), dtype=np.float32)
    expected_first_lane[:5] = 1.0
    expected_first_lane[5:37] = np.arange(1, 33, dtype=np.float32)
    expected_first_lane[37] = 64.0
    expected_first_lane[38] = 528.0
    print("  operand_blocks:")
    ok = True
    for idx, name in enumerate(block_names):
        first = float(blocks[idx, 0])
        print(f"    {idx}:{name}: first={first} lanes0_3={blocks[idx, :4].tolist()}")
        if first != float(expected_first_lane[idx]):
            ok = False
    if not ok:
        print(f"  expected_first_lanes={expected_first_lane.tolist()}")
        return False
    print("  PASS: vmac operand layout matches the simple lane model")
    return True


def run_experiment() -> bool:
    artifacts = compile_role_object(toolchain())
    artifacts.mlir.write_text(generate_mlir(artifacts.combined_object))
    compile_mlir(artifacts)
    input_buf = XRTTensor.from_torch(torch.from_numpy(make_workspace()).to(torch.int32))
    output_buf = XRTTensor((WORKSPACE_DWORDS,), dtype=np.int32)
    print("  Loading NPU kernel...")
    handle = load_kernel(artifacts)
    print("  Running on NPU...")
    result = aie_utils.DefaultNPURuntime.run(handle, [input_buf, output_buf])
    got = output_buf.to_torch().numpy().astype(np.int32)
    print(f"  NPU time: {result.npu_time / 1e3:.1f} us")
    print(f"  disasm={artifacts.disasm}")
    return validate_output(got)


def main() -> int:
    try:
        return 0 if run_experiment() else 1
    except Exception as exc:
        print(f"\nFAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 1
    finally:
        aie_utils.DefaultNPURuntime.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
