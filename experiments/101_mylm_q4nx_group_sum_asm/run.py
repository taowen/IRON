#!/usr/bin/env python3
"""Run the first MyLM-style Q4NX group-sum source-assembly body on NPU."""

from __future__ import annotations

import os
import re
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
CPP_SOURCE = EXPERIMENT_DIR / "group_sum_probe.cc"
ASM_SOURCE = EXPERIMENT_DIR / "group_sum_probe.s"
WORKSPACE_DWORDS = 128
DST_DWORDS = 16
DST_DWORD_OFFSET = 0
GROUP_SUM_DWORD_OFFSET = 16
ACTIVATION_DWORD_OFFSET = 32
EXPECTED_PER_GROUP = 96.0
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
    cpp_object = build_dir / "group_sum_probe.cc.o"
    asm_object = build_dir / "group_sum_probe.s.o"
    combined_object = build_dir / "group_sum_probe.o"
    disasm = build_dir / "group_sum_probe.disasm"
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
    return Artifacts(build_dir, combined_object, disasm, mlir, xclbin, insts)


def generate_mlir(link_object: Path) -> str:
    return f"""module {{
  aie.device(npu2) {{
    %shim = aie.tile(1, 0)
    %tile = aie.tile(1, 2)

    // case marker mylm-q4nx-group-sum-source-asm
{flow("shim", HOST_INPUT_CHANNEL, "tile", TILE_INPUT_CHANNEL)}
{flow("tile", TILE_OUTPUT_CHANNEL, "shim", HOST_OUTPUT_CHANNEL)}

    func.func private @cpp_call_mylm_q4_group_sum_body(memref<{WORKSPACE_DWORDS}xi32>) attributes {{link_with = "{link_object.resolve()}"}}

    %workspace = aie.buffer(%tile) {{sym_name = "workspace"}} : memref<{WORKSPACE_DWORDS}xi32>
    %workspace_empty = aie.lock(%tile, 0) {{init = 1 : i32, sym_name = "workspace_empty"}}
    %workspace_full = aie.lock(%tile, 1) {{init = 0 : i32, sym_name = "workspace_full"}}
    %out_full = aie.lock(%tile, 2) {{init = 0 : i32, sym_name = "out_full"}}

    %tile_core = aie.core(%tile) {{
      aie.use_lock(%workspace_full, AcquireGreaterEqual, 1)
      func.call @cpp_call_mylm_q4_group_sum_body(%workspace) : (memref<{WORKSPACE_DWORDS}xi32>) -> ()
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


def bf16_words(value: float, count: int) -> np.ndarray:
    fp32 = np.full((count,), value, dtype=np.float32)
    return (fp32.view(np.uint32) >> 16).astype(np.uint16)


def make_workspace() -> np.ndarray:
    workspace = np.zeros((WORKSPACE_DWORDS,), dtype=np.int32)
    workspace.view(np.uint16)[GROUP_SUM_DWORD_OFFSET * 2 : GROUP_SUM_DWORD_OFFSET * 2 + 8] = bf16_words(32.0, 8)
    workspace.view(np.uint16)[
        ACTIVATION_DWORD_OFFSET * 2 : ACTIVATION_DWORD_OFFSET * 2 + 32
    ] = bf16_words(1.0, 32)
    return workspace


def validate_shape(artifacts: Artifacts) -> list[str]:
    text = artifacts.disasm.read_text()
    errors: list[str] = []
    loop_count = parse_loop_count(text)
    static_vmac = text.count("vmac.f")
    static_vext = text.count("vextbcst.16")
    static_vups = text.count("vups.4x")
    static_group_loads = text.count("lda.s16")
    static_vst = text.count("\tvst") + text.count(";		vst")
    static_vconv = text.count("vconv.bf16.fp32")
    print("  instruction_shape:")
    print(f"    lc={loop_count}")
    print(f"    static_vmac.f={static_vmac}")
    print(f"    dynamic_vmac.f={static_vmac * loop_count}")
    print(f"    static_vextbcst.16={static_vext}")
    print(f"    dynamic_vextbcst.16={static_vext * loop_count}")
    print(f"    static_vups.4x={static_vups}")
    print(f"    static_group_sum_loads={static_group_loads}")
    print(f"    dynamic_group_sum_loads={static_group_loads * loop_count}")
    print(f"    static_vst={static_vst}")
    print(f"    static_vconv.bf16.fp32={static_vconv}")
    if static_vmac != 33:
        errors.append(f"static vmac mismatch: {static_vmac} != 33")
    if static_vext != 32:
        errors.append(f"static vextbcst.16 mismatch: {static_vext} != 32")
    if static_vups != 0:
        errors.append(f"static vups.4x mismatch: {static_vups} != 0")
    if static_group_loads != 1:
        errors.append(f"static group-sum load mismatch: {static_group_loads} != 1")
    if static_vst != 1:
        errors.append(f"function should only store final accumulator once, got vst={static_vst}")
    if static_vconv != 0:
        errors.append(f"body should not use bf16 conversion traffic, got {static_vconv}")
    return errors


def parse_loop_count(disasm: str) -> int:
    match = re.search(r"\bmova\s+r15,\s*#(\d+)\n\s+[0-9a-f]+:.*\badd\.nc\s+lc,\s*r15,\s*#0\b", disasm)
    if match is None:
        raise ValueError("failed to parse source-assembly LC setup")
    return int(match.group(1), 10)


def load_kernel(artifacts: Artifacts):
    kernel = NPUKernel(
        xclbin_path=str(artifacts.xclbin),
        kernel_name="MLIR_AIE",
        insts_path=str(artifacts.insts),
    )
    return aie_utils.DefaultNPURuntime.load(kernel)


def validate_output(got_workspace: np.ndarray, loop_count: int) -> bool:
    dst = got_workspace[DST_DWORD_OFFSET : DST_DWORD_OFFSET + DST_DWORDS].view(np.float32)
    expected = np.full((DST_DWORDS,), EXPECTED_PER_GROUP * loop_count, dtype=np.float32)
    print(f"  dst[0:8]: {dst[:8].tolist()}")
    print(f"  expected[0:8]: {expected[:8].tolist()}")
    mismatches = np.flatnonzero(dst != expected)
    if mismatches.size:
        print(f"  FAIL: {int(mismatches.size)} accumulator mismatches")
        for idx in mismatches[:16]:
            print(f"    dst[{int(idx)}] expected={float(expected[idx])} got={float(dst[idx])}")
        return False
    print("  PASS: group-sum source-assembly body produced exact synthetic output")
    return True


def run_experiment() -> bool:
    artifacts = compile_role_object(toolchain())
    shape_errors = validate_shape(artifacts)
    loop_count = parse_loop_count(artifacts.disasm.read_text())
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
    if shape_errors:
        print("  FAIL: instruction-shape errors")
        for error in shape_errors:
            print(f"    {error}")
        return False
    return validate_output(got, loop_count)


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
