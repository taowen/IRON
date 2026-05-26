#!/usr/bin/env python3
"""Diagnostic: multi-push memtile weight fanout ordering.

This isolates the shared mt_wt_full/mt_wt_empty protocol used by
generate.py. Two fat chunks are sent through one memtile ping/pong pair and
split to four core rows. Each core discards chunk 0 and returns chunk 1.
Expected row markers are 200, 201, 202, 203.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from ml_dtypes import bfloat16

os.environ["PATH"] = "/var/opt/xilinx/xrt/bin:" + os.environ.get("PATH", "")

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

import aie.utils as aie_utils
from aie.utils.config import peano_install_dir, root_path
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from aie.utils.npukernel import NPUKernel


NUM_ROWS = 4
ROW_SIZE = 32
FAT_SIZE = NUM_ROWS * ROW_SIZE
NUM_PUSHES = 2
WT_I32 = NUM_PUSHES * FAT_SIZE * 2 // 4
OUT_I32 = FAT_SIZE * 2 // 4
EXPERIMENT_DIR = Path(__file__).parent.resolve()


def _shim_bd_address(column: int, bd_id: int) -> int:
    return column * 0x02000000 + 0x1D004 + bd_id * 0x20


def _writebd(bd_id: int, length_i32: int, offset_bytes: int, direction: str, channel: int) -> str:
    return f"""
      aiex.npu.writebd {{bd_id = {bd_id} : i32, buffer_length = {length_i32} : i32, buffer_offset = {offset_bytes} : i32, burst_length = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}}
      aiex.npu.address_patch {{addr = {_shim_bd_address(0, bd_id)} : ui32, arg_idx = {0 if direction == "MM2S" else 1} : i32, arg_plus = {offset_bytes} : i32}}
      aiex.npu.push_queue(0, 0, {direction} : {channel}) {{bd_id = {bd_id} : i32, issue_token = {"true" if direction == "S2MM" else "false"}, repeat_count = 0 : i32}}"""


def generate_mlir() -> str:
    rows = range(NUM_ROWS)
    tiles = "\n".join(f"    %c{r} = aie.tile(0, {r + 2})" for r in rows)
    buffers = "\n".join(
        [
            f"    %c{r}_wt_ping = aie.buffer(%c{r}) {{sym_name = \"c{r}_wt_ping\"}} : memref<{ROW_SIZE}xbf16>\n"
            f"    %c{r}_wt_pong = aie.buffer(%c{r}) {{sym_name = \"c{r}_wt_pong\"}} : memref<{ROW_SIZE}xbf16>\n"
            f"    %c{r}_out = aie.buffer(%c{r}) {{sym_name = \"c{r}_out\"}} : memref<{ROW_SIZE}xbf16>"
            for r in rows
        ]
    )
    locks = "\n".join(
        [
            f"    %c{r}_wt_empty = aie.lock(%c{r}, 0) {{init = 2 : i32, sym_name = \"c{r}_wt_empty\"}}\n"
            f"    %c{r}_wt_full = aie.lock(%c{r}, 1) {{init = 0 : i32, sym_name = \"c{r}_wt_full\"}}\n"
            f"    %c{r}_out_prod = aie.lock(%c{r}, 2) {{init = 1 : i32, sym_name = \"c{r}_out_prod\"}}\n"
            f"    %c{r}_out_cons = aie.lock(%c{r}, 3) {{init = 0 : i32, sym_name = \"c{r}_out_cons\"}}"
            for r in rows
        ]
    )
    flows = "\n".join(
        [
            "    aie.flow(%shim, DMA : 1, %mt, DMA : 1)",
            *[f"    aie.flow(%mt, DMA : {r + 1}, %c{r}, DMA : 1)" for r in rows],
            *[
                f"    aie.packet_flow({r}) {{\n"
                f"      aie.packet_source<%c{r}, DMA : 0>\n"
                f"      aie.packet_dest<%mt, DMA : 2>\n"
                f"    }}"
                for r in rows
            ],
            "    aie.flow(%mt, DMA : 5, %shim, DMA : 0)",
        ]
    )
    core_programs = "\n".join(
        f"""    %prog{r} = aie.core(%c{r}) {{
      %zero = arith.constant 0 : i32
      aie.use_lock(%c{r}_wt_full, AcquireGreaterEqual, 1)
      aie.use_lock(%c{r}_wt_empty, Release, 1)
      aie.use_lock(%c{r}_wt_full, AcquireGreaterEqual, 1)
      aie.use_lock(%c{r}_out_prod, AcquireGreaterEqual, 1)
      func.call @copy_section(%c{r}_wt_pong, %c{r}_out, %zero)
        : (memref<{ROW_SIZE}xbf16>, memref<{ROW_SIZE}xbf16>, i32) -> ()
      aie.use_lock(%c{r}_wt_empty, Release, 1)
      aie.use_lock(%c{r}_out_cons, Release, 1)
      aie.end
    }}"""
        for r in rows
    )
    core_dmas = "\n".join(
        f"""    %mem{r} = aie.mem(%c{r}) {{
      %0 = aie.dma_start(S2MM, 1, ^wt_ping, ^out_start)
    ^wt_ping:
      aie.use_lock(%c{r}_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%c{r}_wt_ping : memref<{ROW_SIZE}xbf16>, 0, {ROW_SIZE}) {{bd_id = 1 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%c{r}_wt_full, Release, 1)
      aie.next_bd ^wt_pong
    ^wt_pong:
      aie.use_lock(%c{r}_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%c{r}_wt_pong : memref<{ROW_SIZE}xbf16>, 0, {ROW_SIZE}) {{bd_id = 2 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%c{r}_wt_full, Release, 1)
      aie.next_bd ^wt_ping
    ^out_start:
      %1 = aie.dma_start(MM2S, 0, ^out, ^end)
    ^out:
      aie.use_lock(%c{r}_out_cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%c{r}_out : memref<{ROW_SIZE}xbf16>, 0, {ROW_SIZE}) {{bd_id = 3 : i32, next_bd_id = 3 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {r}>}}
      aie.use_lock(%c{r}_out_prod, Release, 1)
      aie.next_bd ^out
    ^end:
      aie.end
    }}"""
        for r in rows
    )
    memtile_slices = "\n".join(
        f"""    ^wt_r{r}_ping:
      aie.use_lock(%mt_wt_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt_wt_ping : memref<{FAT_SIZE}xbf16>, {r * ROW_SIZE}, {ROW_SIZE}) {{bd_id = {30 + r * 2 if r % 2 == 0 else 3 + (r - 1) * 2} : i32, next_bd_id = {31 + r * 2 if r % 2 == 0 else 4 + (r - 1) * 2} : i32}}
      aie.use_lock(%mt_wt_empty, Release, 1)
      aie.next_bd ^wt_r{r}_pong
    ^wt_r{r}_pong:
      aie.use_lock(%mt_wt_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt_wt_pong : memref<{FAT_SIZE}xbf16>, {r * ROW_SIZE}, {ROW_SIZE}) {{bd_id = {31 + r * 2 if r % 2 == 0 else 4 + (r - 1) * 2} : i32, next_bd_id = {30 + r * 2 if r % 2 == 0 else 3 + (r - 1) * 2} : i32}}
      aie.use_lock(%mt_wt_empty, Release, 1)
      aie.next_bd ^wt_r{r}_ping"""
        for r in rows
    )
    # The BD ids above follow the same parity split shape as generate.py:
    # row0 odd ch1 -> 30/31, row1 even ch2 -> 3/4, row2 odd ch3 -> 34/35,
    # row3 even ch4 -> 7/8.
    memtile_slices = memtile_slices.replace("34 : i32", "32 : i32").replace("35 : i32", "33 : i32").replace("7 : i32", "5 : i32").replace("8 : i32", "6 : i32")
    runtime = "\n".join(
        [
            f"    aie.runtime_sequence(%wt_bo: memref<{WT_I32}xi32>, %out_bo: memref<{OUT_I32}xi32>) {{",
            _writebd(3, OUT_I32, 0, "S2MM", 0),
            _writebd(1, FAT_SIZE * 2 // 4, 0, "MM2S", 1),
            _writebd(2, FAT_SIZE * 2 // 4, FAT_SIZE * 2, "MM2S", 1),
            "      aiex.npu.sync {channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}",
            "    }",
        ]
    )
    return f"""module {{
  aie.device(npu2) {{
    %shim = aie.tile(0, 0)
    %mt = aie.tile(0, 1)
{tiles}
    %mt_wt_ping = aie.buffer(%mt) {{sym_name = "mt_wt_ping"}} : memref<{FAT_SIZE}xbf16>
    %mt_wt_pong = aie.buffer(%mt) {{sym_name = "mt_wt_pong"}} : memref<{FAT_SIZE}xbf16>
    %mt_out = aie.buffer(%mt) {{sym_name = "mt_out"}} : memref<{FAT_SIZE}xbf16>
{buffers}
    %mt_wt_empty = aie.lock(%mt, 0) {{init = {2 * NUM_ROWS} : i32, sym_name = "mt_wt_empty"}}
    %mt_wt_full = aie.lock(%mt, 1) {{init = 0 : i32, sym_name = "mt_wt_full"}}
    %mt_out_empty = aie.lock(%mt, 2) {{init = {NUM_ROWS} : i32, sym_name = "mt_out_empty"}}
    %mt_out_full = aie.lock(%mt, 3) {{init = 0 : i32, sym_name = "mt_out_full"}}
{locks}
{flows}
    func.func private @copy_section(memref<{ROW_SIZE}xbf16>, memref<{ROW_SIZE}xbf16>, i32) attributes {{link_with = "{EXPERIMENT_DIR}/copy_kernel.o"}}
{core_programs}
    %mtdma = aie.memtile_dma(%mt) {{
      %0 = aie.dma_start(S2MM, 1, ^wt_s2mm_ping, ^out_s2mm_start)
    ^wt_s2mm_ping:
      aie.use_lock(%mt_wt_empty, AcquireGreaterEqual, {NUM_ROWS})
      aie.dma_bd(%mt_wt_ping : memref<{FAT_SIZE}xbf16>, 0, {FAT_SIZE}) {{bd_id = 24 : i32, next_bd_id = 25 : i32}}
      aie.use_lock(%mt_wt_full, Release, {NUM_ROWS})
      aie.next_bd ^wt_s2mm_pong
    ^wt_s2mm_pong:
      aie.use_lock(%mt_wt_empty, AcquireGreaterEqual, {NUM_ROWS})
      aie.dma_bd(%mt_wt_pong : memref<{FAT_SIZE}xbf16>, 0, {FAT_SIZE}) {{bd_id = 25 : i32, next_bd_id = 24 : i32}}
      aie.use_lock(%mt_wt_full, Release, {NUM_ROWS})
      aie.next_bd ^wt_s2mm_ping
    ^out_s2mm_start:
      %1 = aie.dma_start(S2MM, 2, ^out0, ^wt_mm2s_r0_start)
    ^out0:
      aie.use_lock(%mt_out_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt_out : memref<{FAT_SIZE}xbf16>, 0, {ROW_SIZE}) {{bd_id = 7 : i32, next_bd_id = 8 : i32}}
      aie.use_lock(%mt_out_full, Release, 1)
      aie.next_bd ^out1
    ^out1:
      aie.use_lock(%mt_out_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt_out : memref<{FAT_SIZE}xbf16>, {ROW_SIZE}, {ROW_SIZE}) {{bd_id = 8 : i32, next_bd_id = 9 : i32}}
      aie.use_lock(%mt_out_full, Release, 1)
      aie.next_bd ^out2
    ^out2:
      aie.use_lock(%mt_out_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt_out : memref<{FAT_SIZE}xbf16>, {2 * ROW_SIZE}, {ROW_SIZE}) {{bd_id = 9 : i32, next_bd_id = 10 : i32}}
      aie.use_lock(%mt_out_full, Release, 1)
      aie.next_bd ^out3
    ^out3:
      aie.use_lock(%mt_out_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt_out : memref<{FAT_SIZE}xbf16>, {3 * ROW_SIZE}, {ROW_SIZE}) {{bd_id = 10 : i32, next_bd_id = 7 : i32}}
      aie.use_lock(%mt_out_full, Release, 1)
      aie.next_bd ^out0
    ^wt_mm2s_r0_start:
      %2 = aie.dma_start(MM2S, 1, ^wt_r0_ping, ^wt_mm2s_r1_start)
{memtile_slices.split('    ^wt_r1_ping:')[0]}
    ^wt_mm2s_r1_start:
      %3 = aie.dma_start(MM2S, 2, ^wt_r1_ping, ^wt_mm2s_r2_start)
    ^wt_r1_ping:{memtile_slices.split('    ^wt_r1_ping:')[1].split('    ^wt_r2_ping:')[0]}
    ^wt_mm2s_r2_start:
      %4 = aie.dma_start(MM2S, 3, ^wt_r2_ping, ^wt_mm2s_r3_start)
    ^wt_r2_ping:{memtile_slices.split('    ^wt_r2_ping:')[1].split('    ^wt_r3_ping:')[0]}
    ^wt_mm2s_r3_start:
      %5 = aie.dma_start(MM2S, 4, ^wt_r3_ping, ^out_mm2s_start)
    ^wt_r3_ping:{memtile_slices.split('    ^wt_r3_ping:')[1]}
    ^out_mm2s_start:
      %6 = aie.dma_start(MM2S, 5, ^out_send, ^end)
    ^out_send:
      aie.use_lock(%mt_out_full, AcquireGreaterEqual, {NUM_ROWS})
      aie.dma_bd(%mt_out : memref<{FAT_SIZE}xbf16>, 0, {FAT_SIZE}) {{bd_id = 34 : i32, next_bd_id = 34 : i32}}
      aie.use_lock(%mt_out_empty, Release, {NUM_ROWS})
      aie.next_bd ^out_send
    ^end:
      aie.end
    }}
{core_dmas}
{runtime}
  }}
}}
"""


def compile_and_run() -> bool:
    build_dir = EXPERIMENT_DIR / "build_weight_fanout_sequence"
    build_dir.mkdir(exist_ok=True)
    mlir_path = build_dir / "design.mlir"
    mlir_path.write_text(generate_mlir())

    mlir_aie_dir = Path(root_path())
    peano_dir = Path(peano_install_dir())
    aiecc = mlir_aie_dir / "bin" / "aiecc"
    xclbin_path = build_dir / "design.xclbin"
    insts_path = build_dir / "design.bin"
    cmd = [
        str(aiecc), "-v", "-j1", "--no-compile-host", "--no-xchesscc",
        "--no-xbridge", "--peano", str(peano_dir), "--aie-generate-xclbin",
        f"--xclbin-name={xclbin_path}", "--xclbin-kernel-name=MLIR_AIE",
        "--aie-generate-npu-insts", f"--npu-insts-name={insts_path}",
        str(mlir_path),
    ]
    print("Compiling diagnostic MLIR...")
    if os.system(" ".join(cmd)) != 0:
        return False

    wt = np.zeros(NUM_PUSHES * FAT_SIZE, dtype=bfloat16)
    for row in range(NUM_ROWS):
        wt[row * ROW_SIZE:(row + 1) * ROW_SIZE] = bfloat16(100 + row)
        base = FAT_SIZE + row * ROW_SIZE
        wt[base:base + ROW_SIZE] = bfloat16(200 + row)
    wt_i32 = np.frombuffer(wt.tobytes(), dtype=np.int32)

    npu_kernel = NPUKernel(
        xclbin_path=str(xclbin_path), kernel_name="MLIR_AIE",
        insts_path=str(insts_path),
    )
    handle = aie_utils.DefaultNPURuntime.load(npu_kernel)
    wt_buf = XRTTensor.from_torch(torch.from_numpy(wt_i32.copy()).to(torch.int32))
    out_buf = XRTTensor((OUT_I32,), dtype=np.int32)
    print("Running diagnostic...")
    try:
        result = aie_utils.DefaultNPURuntime.run(handle, [wt_buf, out_buf])
        print(f"  NPU time: {result.npu_time / 1e3:.1f} us")
    finally:
        aie_utils.DefaultNPURuntime.cleanup()

    out = np.frombuffer(out_buf.to_torch().numpy().tobytes(), dtype=bfloat16)
    ok = True
    for row in range(NUM_ROWS):
        got = out[row * ROW_SIZE:(row + 1) * ROW_SIZE]
        expected = np.full(ROW_SIZE, bfloat16(200 + row), dtype=bfloat16)
        row_ok = np.array_equal(got.view(np.uint16), expected.view(np.uint16))
        ok = ok and row_ok
        print(f"  row{row}: got {got[:8]} expected {expected[:4]} {'OK' if row_ok else 'BAD'}")
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    sys.exit(0 if compile_and_run() else 1)
