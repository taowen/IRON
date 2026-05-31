#!/usr/bin/env python3
"""Measure small AIE2P vector dependency gaps on real NPU."""

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
CPP_SOURCE = EXPERIMENT_DIR / "latency_probe.cc"
WORKSPACE_DWORDS = 1024
OUTPUT_BLOCK_FLOATS = 16
GAPS = tuple(range(9))
CATEGORIES = ("vext_to_vmac", "vbcst_to_vmac", "vldb_to_vext", "vmac_to_vst")
OUTPUT_BLOCKS = len(GAPS) * len(CATEGORIES)
ACTIVATION_DWORD_OFFSET = 640
ZERO_DWORD_OFFSET = 768
HOST_INPUT_CHANNEL = 0
TILE_INPUT_CHANNEL = 1
TILE_OUTPUT_CHANNEL = 0
HOST_OUTPUT_CHANNEL = 1
EXPECTED = 3.0


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
    asm_source: Path
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


def nops(count: int) -> str:
    return "\n".join("\tnop" for _ in range(count))


def emit_store() -> str:
    return "\n".join(
        (
            "\tvst\tbmll1, [p4, #0]",
            nops(8),
            "\tpadda\t[p4], m0",
        )
    )


def emit_reset() -> str:
    return "\n".join(
        (
            "\tvlda\tbmll1, [p3, #0]",
            nops(8),
        )
    )


def emit_case(category: str, gap: int) -> str:
    if category == "vext_to_vmac":
        body = (
            "\tvbcst.16\tx0, r11",
            "\tvextbcst.16\tx0, x11, #0",
            nops(gap),
            "\tvmac.f\tdm1, dm1, x2, x0, r4",
            nops(8),
            emit_store(),
        )
    elif category == "vbcst_to_vmac":
        body = (
            "\tvbcst.16\tx0, r11",
            "\tvbcst.16\tx0, r10",
            nops(gap),
            "\tvmac.f\tdm1, dm1, x2, x0, r4",
            nops(8),
            emit_store(),
        )
    elif category == "vldb_to_vext":
        body = (
            "\tvbcst.16\tx11, r11",
            "\tvldb\tx11, [p1, #0]",
            nops(gap),
            "\tvextbcst.16\tx0, x11, #0",
            nops(4),
            "\tvmac.f\tdm1, dm1, x2, x0, r4",
            nops(8),
            emit_store(),
        )
    elif category == "vmac_to_vst":
        body = (
            "\tvmac.f\tdm1, dm1, x2, x3, r4",
            nops(gap),
            emit_store(),
        )
    else:
        raise ValueError(f"unknown category {category}")
    return "\n".join((f"\n\t// {category} gap={gap}", emit_reset(), *body))


def generate_asm() -> str:
    cases = [emit_case(category, gap) for category in CATEGORIES for gap in GAPS]
    return "\n".join(
        (
            "\t.text",
            "",
            "\t.macro SAVE_LATENCY_STATE",
            "\tpaddxm\t[sp], #0x80",
            "\tst\tr0, [sp, #-0x80]",
            "\tst\tr4, [sp, #-0x7c]",
            "\tst\tr10, [sp, #-0x78]",
            "\tst\tr11, [sp, #-0x74]",
            "\tst\tr12, [sp, #-0x70]",
            "\tst\tp0, [sp, #-0x6c]",
            "\tst\tp1, [sp, #-0x68]",
            "\tst\tp3, [sp, #-0x64]",
            "\tst\tp4, [sp, #-0x60]",
            "\tst\tm0, [sp, #-0x5c]",
            "\tst\tlr, [sp, #-0x58]",
            "\t.endm",
            "",
            "\t.macro RESTORE_LATENCY_STATE",
            "\tlda\tlr, [sp, #-0x58]",
            "\tlda\tm0, [sp, #-0x5c]",
            "\tlda\tp4, [sp, #-0x60]",
            "\tlda\tp3, [sp, #-0x64]",
            "\tlda\tp1, [sp, #-0x68]",
            "\tlda\tp0, [sp, #-0x6c]",
            "\tlda\tr12, [sp, #-0x70]",
            "\tlda\tr11, [sp, #-0x74]",
            "\tlda\tr10, [sp, #-0x78]",
            "\tlda\tr4, [sp, #-0x7c]",
            "\tlda\tr0, [sp, #-0x80]",
            "\tpaddxm\t[sp], #-0x80",
            "\t.endm",
            "",
            "\t.section\t.text.asm_aie2p_latency_probe,\"ax\",@progbits",
            "\t.globl\tasm_aie2p_latency_probe",
            "\t.p2align\t4",
            "\t.type\tasm_aie2p_latency_probe,@function",
            "asm_aie2p_latency_probe:",
            "\tSAVE_LATENCY_STATE",
            "\tmov\tp4, p0",
            "\tmov\tp1, p0",
            "\tmovxm\tr0, #0xa00",
            "\tmovs\tm0, r0",
            "\tpadda\t[p1], m0",
            "\tmov\tp3, p0",
            "\tmovxm\tr0, #0xc00",
            "\tmovs\tm0, r0",
            "\tpadda\t[p3], m0",
            "\tmova\tr0, #0x40",
            "\tmovs\tm0, r0",
            "\tmov\tcrrnd, #0xc",
            "\tmova\tr4, #0x33c",
            "\tmovxm\tr10, #0x4040",
            "\tmovxm\tr11, #0x4110",
            "\tmovxm\tr12, #0x3f80",
            "\tvbcst.16\tx2, r12",
            "\tvbcst.16\tx3, r10",
            "\tvldb\tx11, [p1, #0]",
            nops(8),
            *cases,
            "\tRESTORE_LATENCY_STATE",
            "\tret\tlr",
            "",
        )
    )


def compile_role_object(tools: Toolchain) -> Artifacts:
    build_dir = EXPERIMENT_DIR / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    cpp_object = build_dir / "latency_probe.cc.o"
    asm_source = build_dir / "latency_probe.s"
    asm_object = build_dir / "latency_probe.s.o"
    combined_object = build_dir / "latency_probe.o"
    disasm = build_dir / "latency_probe.disasm"
    mlir = build_dir / "design.mlir"
    xclbin = build_dir / "design.xclbin"
    insts = build_dir / "design.bin"
    asm_source.write_text(generate_asm())
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
    checked_run((str(tools.clang), "--target=aie2p-none-unknown-elf", "-c", str(asm_source), "-o", str(asm_object)))
    checked_run((str(tools.linker), "-r", str(cpp_object), str(asm_object), "-o", str(combined_object)))
    disasm.write_text(
        checked_run((str(tools.objdump), "--triple=aie2p", "-dr", "--no-print-imm-hex", str(combined_object)))
    )
    return Artifacts(build_dir, combined_object, asm_source, disasm, mlir, xclbin, insts)


def generate_mlir(link_object: Path) -> str:
    return f"""module {{
  aie.device(npu2) {{
    %shim = aie.tile(1, 0)
    %tile = aie.tile(1, 2)

    // case marker aie2p-dependency-latency
{flow("shim", HOST_INPUT_CHANNEL, "tile", TILE_INPUT_CHANNEL)}
{flow("tile", TILE_OUTPUT_CHANNEL, "shim", HOST_OUTPUT_CHANNEL)}

    func.func private @cpp_call_aie2p_latency_probe(memref<{WORKSPACE_DWORDS}xi32>) attributes {{link_with = "{link_object.resolve()}"}}

    %workspace = aie.buffer(%tile) {{sym_name = "workspace"}} : memref<{WORKSPACE_DWORDS}xi32>
    %workspace_empty = aie.lock(%tile, 0) {{init = 1 : i32, sym_name = "workspace_empty"}}
    %workspace_full = aie.lock(%tile, 1) {{init = 0 : i32, sym_name = "workspace_full"}}
    %out_full = aie.lock(%tile, 2) {{init = 0 : i32, sym_name = "out_full"}}

    %tile_core = aie.core(%tile) {{
      aie.use_lock(%workspace_full, AcquireGreaterEqual, 1)
      func.call @cpp_call_aie2p_latency_probe(%workspace) : (memref<{WORKSPACE_DWORDS}xi32>) -> ()
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
    activation = np.full((32,), EXPECTED, dtype=np.float32)
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
    ok = True
    block_index = 0
    for category in CATEGORIES:
        first_passing_gap = None
        print(f"  {category}:")
        for gap in GAPS:
            value = float(blocks[block_index, 0])
            matches = value == EXPECTED
            print(f"    gap={gap}: first={value} {'PASS' if matches else 'MISS'}")
            if matches and first_passing_gap is None:
                first_passing_gap = gap
            block_index += 1
        if first_passing_gap is None:
            ok = False
            print("    no passing gap")
        else:
            print(f"    first_passing_gap={first_passing_gap}")
    return ok


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
    print(f"  asm={artifacts.asm_source}")
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
