#!/usr/bin/env python3
"""Test: 1 column (4 cores), full pipeline but smallest possible sizes.

Tests the COMPLETE per-column datapath at minimal scale:
- Shim → memtile activation
- Memtile multicast activation → 4 cores
- Shim → memtile weights (1 fat chunk, single push, no ping-pong)
- Memtile → per-core weight slices
- Core copies activation[offset:offset+32] → inter_buf (mimics gate/up+swiglu)
- Core sends intermediate via packet → memtile S2MM ch3 (gather)
- Memtile multicasts gathered → 4 cores
- Core copies gathered[0:32] → out_buf (mimics down projection)
- Core sends output via packet → memtile S2MM ch2
- Memtile forwards aggregated output → shim

This exercises ALL channel interactions in a single column
but uses trivial copy kernels to eliminate Q4NX as a variable.
"""

import sys, os, numpy as np
os.environ["PATH"] = "/var/opt/xilinx/xrt/bin:" + os.environ.get("PATH", "")
from pathlib import Path

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

from ml_dtypes import bfloat16
import torch
import aie.utils as aie_utils
from aie.utils.config import peano_install_dir, root_path
from aie.utils.npukernel import NPUKernel
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

NUM_CORES = 4
M = 32
ACT_BF16 = 128  # small activation (not 4096 — just enough for 4 cores × 32)
WT_PER_CORE = 64  # small weight per core
FAT_WT = WT_PER_CORE * NUM_CORES  # 256 bf16 total
GATHERED = NUM_CORES * M  # 128
TOTAL_OUT = GATHERED  # 128

ACT_I32 = ACT_BF16 * 2 // 4
WT_I32 = FAT_WT * 2 // 4
OUT_I32 = TOTAL_OUT * 2 // 4

EXPERIMENT_DIR = Path(__file__).parent.resolve()


def _shim_bd_address(column, bd_id):
    return column * 0x02000000 + 0x1D004 + bd_id * 0x20


def generate_mlir():
    lines = []
    lines.append("module {")
    lines.append("  aie.device(npu2) {")
    lines.append("    %shim = aie.tile(0, 0)")
    lines.append("    %mt = aie.tile(0, 1)")
    for r in range(NUM_CORES):
        lines.append(f"    %core{r} = aie.tile(0, {r+2})")

    # Memtile buffers
    lines.append(f"    %mt_act = aie.buffer(%mt) {{sym_name = \"mt_act\"}} : memref<{ACT_BF16}xbf16>")
    lines.append(f"    %mt_wt = aie.buffer(%mt) {{sym_name = \"mt_wt\"}} : memref<{FAT_WT}xbf16>")
    lines.append(f"    %mt_gathered = aie.buffer(%mt) {{sym_name = \"mt_gathered\"}} : memref<{GATHERED}xbf16>")
    lines.append(f"    %mt_out = aie.buffer(%mt) {{sym_name = \"mt_out\"}} : memref<{TOTAL_OUT}xbf16>")

    # Memtile locks
    lines.append(f"    %mt_act_empty = aie.lock(%mt, 0) {{init = 1 : i32, sym_name = \"mt_act_empty\"}}")
    lines.append(f"    %mt_act_full  = aie.lock(%mt, 1) {{init = 0 : i32, sym_name = \"mt_act_full\"}}")
    lines.append(f"    %mt_wt_empty  = aie.lock(%mt, 2) {{init = {NUM_CORES} : i32, sym_name = \"mt_wt_empty\"}}")
    lines.append(f"    %mt_wt_full   = aie.lock(%mt, 3) {{init = 0 : i32, sym_name = \"mt_wt_full\"}}")
    lines.append(f"    %mt_gathered_empty = aie.lock(%mt, 4) {{init = {NUM_CORES} : i32, sym_name = \"mt_gathered_empty\"}}")
    lines.append(f"    %mt_gathered_full  = aie.lock(%mt, 5) {{init = 0 : i32, sym_name = \"mt_gathered_full\"}}")
    lines.append(f"    %mt_out_empty = aie.lock(%mt, 6) {{init = {NUM_CORES} : i32, sym_name = \"mt_out_empty\"}}")
    lines.append(f"    %mt_out_full  = aie.lock(%mt, 7) {{init = 0 : i32, sym_name = \"mt_out_full\"}}")

    # Core buffers and locks
    for r in range(NUM_CORES):
        lines.append(f"    %c{r}_act = aie.buffer(%core{r}) {{sym_name = \"c{r}_act\"}} : memref<{ACT_BF16}xbf16>")
        lines.append(f"    %c{r}_wt = aie.buffer(%core{r}) {{sym_name = \"c{r}_wt\"}} : memref<{WT_PER_CORE}xbf16>")
        lines.append(f"    %c{r}_inter = aie.buffer(%core{r}) {{sym_name = \"c{r}_inter\"}} : memref<{M}xbf16>")
        lines.append(f"    %c{r}_gathered = aie.buffer(%core{r}) {{sym_name = \"c{r}_gathered\"}} : memref<{GATHERED}xbf16>")
        lines.append(f"    %c{r}_out = aie.buffer(%core{r}) {{sym_name = \"c{r}_out\"}} : memref<{M}xbf16>")
        lines.append(f"    %c{r}_act_empty = aie.lock(%core{r}, 0) {{init = 1 : i32, sym_name = \"c{r}_act_empty\"}}")
        lines.append(f"    %c{r}_act_full  = aie.lock(%core{r}, 1) {{init = 0 : i32, sym_name = \"c{r}_act_full\"}}")
        lines.append(f"    %c{r}_wt_empty  = aie.lock(%core{r}, 2) {{init = 1 : i32, sym_name = \"c{r}_wt_empty\"}}")
        lines.append(f"    %c{r}_wt_full   = aie.lock(%core{r}, 3) {{init = 0 : i32, sym_name = \"c{r}_wt_full\"}}")
        lines.append(f"    %c{r}_inter_prod = aie.lock(%core{r}, 4) {{init = 1 : i32, sym_name = \"c{r}_inter_prod\"}}")
        lines.append(f"    %c{r}_inter_cons = aie.lock(%core{r}, 5) {{init = 0 : i32, sym_name = \"c{r}_inter_cons\"}}")
        lines.append(f"    %c{r}_gathered_empty = aie.lock(%core{r}, 6) {{init = 1 : i32, sym_name = \"c{r}_gathered_empty\"}}")
        lines.append(f"    %c{r}_gathered_full  = aie.lock(%core{r}, 7) {{init = 0 : i32, sym_name = \"c{r}_gathered_full\"}}")
        lines.append(f"    %c{r}_out_prod = aie.lock(%core{r}, 8) {{init = 1 : i32, sym_name = \"c{r}_out_prod\"}}")
        lines.append(f"    %c{r}_out_cons = aie.lock(%core{r}, 9) {{init = 0 : i32, sym_name = \"c{r}_out_cons\"}}")

    # Flows: shim → memtile
    lines.append("    aie.flow(%shim, DMA : 0, %mt, DMA : 0)")  # activation
    lines.append("    aie.flow(%shim, DMA : 1, %mt, DMA : 1)")  # weights

    # Memtile activation multicast → 4 cores S2MM ch0
    for r in range(NUM_CORES):
        lines.append(f"    aie.flow(%mt, DMA : 0, %core{r}, DMA : 0)")

    # Memtile weight distribution: MM2S ch(1+r) → core r S2MM ch1
    for r in range(NUM_CORES):
        lines.append(f"    aie.flow(%mt, DMA : {1+r}, %core{r}, DMA : 1)")

    # Core → Memtile: intermediate via packet (all to memtile S2MM ch3)
    for r in range(NUM_CORES):
        lines.append(f"    aie.packet_flow({r}) {{")
        lines.append(f"      aie.packet_source<%core{r}, DMA : 1>")
        lines.append(f"      aie.packet_dest<%mt, DMA : 3>")
        lines.append(f"    }}")

    # Core → Memtile: output via packet (all to memtile S2MM ch2)
    for r in range(NUM_CORES):
        lines.append(f"    aie.packet_flow({NUM_CORES + r}) {{")
        lines.append(f"      aie.packet_source<%core{r}, DMA : 0>")
        lines.append(f"      aie.packet_dest<%mt, DMA : 2>")
        lines.append(f"    }}")

    # Memtile → shim: output forward
    lines.append("    aie.flow(%mt, DMA : 5, %shim, DMA : 0)")

    # Kernel declaration
    lines.append(f'    func.func private @copy_section(memref<{ACT_BF16}xbf16>, memref<{M}xbf16>, i32) attributes {{link_with = "{EXPERIMENT_DIR}/copy_kernel.o"}}')
    lines.append(f'    func.func private @copy_gathered(memref<{GATHERED}xbf16>, memref<{M}xbf16>, i32) attributes {{link_with = "{EXPERIMENT_DIR}/copy_kernel.o"}}')

    # Core programs
    for r in range(NUM_CORES):
        lines.append(f"""    %prog{r} = aie.core(%core{r}) {{
      %offset = arith.constant {r * M} : i32

      // Phase 1: receive activation, compute "intermediate" (copy slice)
      aie.use_lock(%c{r}_act_full, AcquireGreaterEqual, 1)
      aie.use_lock(%c{r}_wt_full, AcquireGreaterEqual, 1)

      aie.use_lock(%c{r}_inter_prod, AcquireGreaterEqual, 1)
      func.call @copy_section(%c{r}_act, %c{r}_inter, %offset)
        : (memref<{ACT_BF16}xbf16>, memref<{M}xbf16>, i32) -> ()
      aie.use_lock(%c{r}_wt_empty, Release, 1)
      aie.use_lock(%c{r}_act_empty, Release, 1)
      aie.use_lock(%c{r}_inter_cons, Release, 1)

      // Phase 2: receive gathered, compute "output" (copy first 32 of gathered)
      %zero = arith.constant 0 : i32
      aie.use_lock(%c{r}_gathered_full, AcquireGreaterEqual, 1)
      aie.use_lock(%c{r}_out_prod, AcquireGreaterEqual, 1)
      func.call @copy_gathered(%c{r}_gathered, %c{r}_out, %zero)
        : (memref<{GATHERED}xbf16>, memref<{M}xbf16>, i32) -> ()
      aie.use_lock(%c{r}_out_cons, Release, 1)
      aie.use_lock(%c{r}_gathered_empty, Release, 1)

      aie.end
    }}""")

    # Core DMAs
    for r in range(NUM_CORES):
        lines.append(f"""    %mem{r} = aie.mem(%core{r}) {{
      // S2MM ch0: activation → gathered (sequential chain)
      %0 = aie.dma_start(S2MM, 0, ^act_bd, ^wt_start)
    ^act_bd:
      aie.use_lock(%c{r}_act_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%c{r}_act : memref<{ACT_BF16}xbf16>, 0, {ACT_BF16}) {{bd_id = 0 : i32, next_bd_id = 5 : i32}}
      aie.use_lock(%c{r}_act_full, Release, 1)
      aie.next_bd ^gathered_bd
    ^gathered_bd:
      aie.use_lock(%c{r}_gathered_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%c{r}_gathered : memref<{GATHERED}xbf16>, 0, {GATHERED}) {{bd_id = 5 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%c{r}_gathered_full, Release, 1)
      aie.next_bd ^act_bd

      // S2MM ch1: weight (single transfer)
    ^wt_start:
      %1 = aie.dma_start(S2MM, 1, ^wt_bd, ^out_start)
    ^wt_bd:
      aie.use_lock(%c{r}_wt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%c{r}_wt : memref<{WT_PER_CORE}xbf16>, 0, {WT_PER_CORE}) {{bd_id = 1 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%c{r}_wt_full, Release, 1)
      aie.next_bd ^wt_bd

      // MM2S ch0: output (packet to memtile S2MM ch2)
    ^out_start:
      %2 = aie.dma_start(MM2S, 0, ^out_bd, ^inter_start)
    ^out_bd:
      aie.use_lock(%c{r}_out_cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%c{r}_out : memref<{M}xbf16>, 0, {M}) {{bd_id = 2 : i32, next_bd_id = 2 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {NUM_CORES + r}>}}
      aie.use_lock(%c{r}_out_prod, Release, 1)
      aie.next_bd ^out_bd

      // MM2S ch1: intermediate (packet to memtile S2MM ch3)
    ^inter_start:
      %3 = aie.dma_start(MM2S, 1, ^inter_bd, ^end)
    ^inter_bd:
      aie.use_lock(%c{r}_inter_cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%c{r}_inter : memref<{M}xbf16>, 0, {M}) {{bd_id = 3 : i32, next_bd_id = 3 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {r}>}}
      aie.use_lock(%c{r}_inter_prod, Release, 1)
      aie.next_bd ^inter_bd
    ^end:
      aie.end
    }}""")

    # Memtile DMA
    lines.append(f"""    %mtdma = aie.memtile_dma(%mt) {{
      // S2MM ch0 (even, BD 0): activation from shim
      %0 = aie.dma_start(S2MM, 0, ^act_recv, ^wt_recv_start)
    ^act_recv:
      aie.use_lock(%mt_act_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt_act : memref<{ACT_BF16}xbf16>, 0, {ACT_BF16}) {{bd_id = 0 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%mt_act_full, Release, 1)
      aie.next_bd ^act_recv

      // S2MM ch1 (odd, BD 24): weight from shim (single transfer)
    ^wt_recv_start:
      %1 = aie.dma_start(S2MM, 1, ^wt_recv, ^out_recv_start)
    ^wt_recv:
      aie.use_lock(%mt_wt_empty, AcquireGreaterEqual, {NUM_CORES})
      aie.dma_bd(%mt_wt : memref<{FAT_WT}xbf16>, 0, {FAT_WT}) {{bd_id = 24 : i32, next_bd_id = 24 : i32}}
      aie.use_lock(%mt_wt_full, Release, {NUM_CORES})
      aie.next_bd ^wt_recv

      // S2MM ch2 (even, BD 7-10): output collect from 4 cores via packet
    ^out_recv_start:
      %2 = aie.dma_start(S2MM, 2, ^out_r0, ^inter_recv_start)
    ^out_r0:
      aie.use_lock(%mt_out_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt_out : memref<{TOTAL_OUT}xbf16>, 0, {M}) {{bd_id = 7 : i32, next_bd_id = 8 : i32}}
      aie.use_lock(%mt_out_full, Release, 1)
      aie.next_bd ^out_r1
    ^out_r1:
      aie.use_lock(%mt_out_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt_out : memref<{TOTAL_OUT}xbf16>, {M}, {M}) {{bd_id = 8 : i32, next_bd_id = 9 : i32}}
      aie.use_lock(%mt_out_full, Release, 1)
      aie.next_bd ^out_r2
    ^out_r2:
      aie.use_lock(%mt_out_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt_out : memref<{TOTAL_OUT}xbf16>, {2*M}, {M}) {{bd_id = 9 : i32, next_bd_id = 10 : i32}}
      aie.use_lock(%mt_out_full, Release, 1)
      aie.next_bd ^out_r3
    ^out_r3:
      aie.use_lock(%mt_out_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt_out : memref<{TOTAL_OUT}xbf16>, {3*M}, {M}) {{bd_id = 10 : i32, next_bd_id = 7 : i32}}
      aie.use_lock(%mt_out_full, Release, 1)
      aie.next_bd ^out_r0

      // S2MM ch3 (odd, BD 26-29): intermediate gather from 4 cores via packet
    ^inter_recv_start:
      %3 = aie.dma_start(S2MM, 3, ^inter_r0, ^act_send_start)
    ^inter_r0:
      aie.use_lock(%mt_gathered_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt_gathered : memref<{GATHERED}xbf16>, 0, {M}) {{bd_id = 26 : i32, next_bd_id = 27 : i32}}
      aie.use_lock(%mt_gathered_full, Release, 1)
      aie.next_bd ^inter_r1
    ^inter_r1:
      aie.use_lock(%mt_gathered_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt_gathered : memref<{GATHERED}xbf16>, {M}, {M}) {{bd_id = 27 : i32, next_bd_id = 28 : i32}}
      aie.use_lock(%mt_gathered_full, Release, 1)
      aie.next_bd ^inter_r2
    ^inter_r2:
      aie.use_lock(%mt_gathered_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt_gathered : memref<{GATHERED}xbf16>, {2*M}, {M}) {{bd_id = 28 : i32, next_bd_id = 29 : i32}}
      aie.use_lock(%mt_gathered_full, Release, 1)
      aie.next_bd ^inter_r3
    ^inter_r3:
      aie.use_lock(%mt_gathered_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt_gathered : memref<{GATHERED}xbf16>, {3*M}, {M}) {{bd_id = 29 : i32, next_bd_id = 26 : i32}}
      aie.use_lock(%mt_gathered_full, Release, 1)
      aie.next_bd ^inter_r0

      // MM2S ch0 (even, BD 1→2): act THEN gathered, multicast to 4 cores
    ^act_send_start:
      %4 = aie.dma_start(MM2S, 0, ^act_send, ^wt_send_r0_start)
    ^act_send:
      aie.use_lock(%mt_act_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt_act : memref<{ACT_BF16}xbf16>, 0, {ACT_BF16}) {{bd_id = 1 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%mt_act_empty, Release, 1)
      aie.next_bd ^gathered_send
    ^gathered_send:
      aie.use_lock(%mt_gathered_full, AcquireGreaterEqual, {NUM_CORES})
      aie.dma_bd(%mt_gathered : memref<{GATHERED}xbf16>, 0, {GATHERED}) {{bd_id = 2 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%mt_gathered_empty, Release, {NUM_CORES})
      aie.next_bd ^act_send

      // MM2S ch1 (odd, BD 30): weight slice row0
    ^wt_send_r0_start:
      %5 = aie.dma_start(MM2S, 1, ^wt_send_r0, ^wt_send_r1_start)
    ^wt_send_r0:
      aie.use_lock(%mt_wt_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt_wt : memref<{FAT_WT}xbf16>, 0, {WT_PER_CORE}) {{bd_id = 30 : i32, next_bd_id = 30 : i32}}
      aie.use_lock(%mt_wt_empty, Release, 1)
      aie.next_bd ^wt_send_r0

      // MM2S ch2 (even, BD 3): weight slice row1
    ^wt_send_r1_start:
      %6 = aie.dma_start(MM2S, 2, ^wt_send_r1, ^wt_send_r2_start)
    ^wt_send_r1:
      aie.use_lock(%mt_wt_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt_wt : memref<{FAT_WT}xbf16>, {WT_PER_CORE}, {WT_PER_CORE}) {{bd_id = 3 : i32, next_bd_id = 3 : i32}}
      aie.use_lock(%mt_wt_empty, Release, 1)
      aie.next_bd ^wt_send_r1

      // MM2S ch3 (odd, BD 32): weight slice row2
    ^wt_send_r2_start:
      %7 = aie.dma_start(MM2S, 3, ^wt_send_r2, ^wt_send_r3_start)
    ^wt_send_r2:
      aie.use_lock(%mt_wt_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt_wt : memref<{FAT_WT}xbf16>, {2*WT_PER_CORE}, {WT_PER_CORE}) {{bd_id = 32 : i32, next_bd_id = 32 : i32}}
      aie.use_lock(%mt_wt_empty, Release, 1)
      aie.next_bd ^wt_send_r2

      // MM2S ch4 (even, BD 5): weight slice row3
    ^wt_send_r3_start:
      %8 = aie.dma_start(MM2S, 4, ^wt_send_r3, ^out_send_start)
    ^wt_send_r3:
      aie.use_lock(%mt_wt_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt_wt : memref<{FAT_WT}xbf16>, {3*WT_PER_CORE}, {WT_PER_CORE}) {{bd_id = 5 : i32, next_bd_id = 5 : i32}}
      aie.use_lock(%mt_wt_empty, Release, 1)
      aie.next_bd ^wt_send_r3

      // MM2S ch5 (odd, BD 34): output forward to shim
    ^out_send_start:
      %9 = aie.dma_start(MM2S, 5, ^out_send, ^end)
    ^out_send:
      aie.use_lock(%mt_out_full, AcquireGreaterEqual, {NUM_CORES})
      aie.dma_bd(%mt_out : memref<{TOTAL_OUT}xbf16>, 0, {TOTAL_OUT}) {{bd_id = 34 : i32, next_bd_id = 34 : i32}}
      aie.use_lock(%mt_out_empty, Release, {NUM_CORES})
      aie.next_bd ^out_send
    ^end:
      aie.end
    }}""")

    # Runtime sequence
    act_addr = _shim_bd_address(0, 0)
    wt_addr = _shim_bd_address(0, 1)
    out_addr = _shim_bd_address(0, 3)

    lines.append(f"""    aie.runtime_sequence(%wt_bo: memref<{WT_I32}xi32>, %act_bo: memref<{ACT_I32}xi32>, %out_bo: memref<{OUT_I32}xi32>) {{
      // Push activation: shim MM2S ch0, bd0
      aiex.npu.writebd {{bd_id = 0 : i32, buffer_length = {ACT_I32} : i32, buffer_offset = 0 : i32, burst_length = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}}
      aiex.npu.address_patch {{addr = {act_addr} : ui32, arg_idx = 1 : i32, arg_plus = 0 : i32}}
      aiex.npu.push_queue(0, 0, MM2S : 0) {{bd_id = 0 : i32, issue_token = false, repeat_count = 0 : i32}}

      // Push weight: shim MM2S ch1, bd1
      aiex.npu.writebd {{bd_id = 1 : i32, buffer_length = {WT_I32} : i32, buffer_offset = 0 : i32, burst_length = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}}
      aiex.npu.address_patch {{addr = {wt_addr} : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}}
      aiex.npu.push_queue(0, 0, MM2S : 1) {{bd_id = 1 : i32, issue_token = false, repeat_count = 0 : i32}}

      // Receive output: shim S2MM ch0, bd3
      aiex.npu.writebd {{bd_id = 3 : i32, buffer_length = {OUT_I32} : i32, buffer_offset = 0 : i32, burst_length = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}}
      aiex.npu.address_patch {{addr = {out_addr} : ui32, arg_idx = 2 : i32, arg_plus = 0 : i32}}
      aiex.npu.push_queue(0, 0, S2MM : 0) {{bd_id = 3 : i32, issue_token = true, repeat_count = 0 : i32}}

      // Sync
      aiex.npu.sync {{channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}}
    }}""")

    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


def main():
    kernel_src = EXPERIMENT_DIR / "copy_kernel.cc"
    kernel_src.write_text('''#include <aie_api/aie.hpp>
extern "C" void copy_section(bfloat16* src, bfloat16* dst, int offset) {
    for (int i = 0; i < 32; i++) {
        dst[i] = src[offset + i];
    }
}
extern "C" void copy_gathered(bfloat16* src, bfloat16* dst, int offset) {
    for (int i = 0; i < 32; i++) {
        dst[i] = src[offset + i];
    }
}
''')

    # Compile kernel
    peano_dir = Path(peano_install_dir())
    mlir_aie_dir = Path(root_path())
    clang = peano_dir / "bin" / "clang++"
    include_path = mlir_aie_dir / "include"
    runtime_lib_include = mlir_aie_dir / "aie_runtime_lib" / "AIE2P"

    obj = EXPERIMENT_DIR / "copy_kernel.o"
    cmd = [
        str(clang), "-O2", "-std=c++20",
        "--target=aie2p-none-unknown-elf",
        "-Wno-parentheses", "-Wno-attributes",
        "-Wno-macro-redefined", "-Wno-empty-body",
        "-Wno-missing-template-arg-list-after-template-kw",
        f"-I{include_path}", f"-I{runtime_lib_include}",
        "-c", str(kernel_src), "-o", str(obj),
    ]
    print("Compiling copy_kernel...")
    ret = os.system(" ".join(cmd))
    if ret != 0:
        print("FAILED: kernel compilation")
        return False

    # Generate and compile MLIR
    mlir = generate_mlir()
    build_dir = EXPERIMENT_DIR / "build_core2mt"
    build_dir.mkdir(exist_ok=True)
    mlir_path = build_dir / "design.mlir"
    mlir_path.write_text(mlir)

    aiecc = mlir_aie_dir / "bin" / "aiecc"
    xclbin_path = build_dir / "design.xclbin"
    insts_path = build_dir / "design.bin"
    cmd = [
        str(aiecc), "-v", "-j1",
        "--no-compile-host", "--no-xchesscc", "--no-xbridge",
        "--peano", str(peano_dir),
        "--aie-generate-xclbin", f"--xclbin-name={xclbin_path}",
        "--xclbin-kernel-name=MLIR_AIE",
        "--aie-generate-npu-insts", f"--npu-insts-name={insts_path}",
        str(mlir_path),
    ]
    print("Compiling MLIR...")
    ret = os.system(" ".join(cmd))
    if ret != 0:
        print("FAILED: MLIR compilation")
        return False

    # Run on NPU
    print("\nRunning on NPU...")
    dev = aie_utils.DefaultNPURuntime.device()

    # Input: activation = [0,1,2,...,127]
    activation = np.arange(ACT_BF16, dtype=np.float32).astype(bfloat16)
    act_i32 = np.frombuffer(activation.tobytes(), dtype=np.int32)

    # Weights: dummy (not used by copy kernel but must exist)
    weights = np.zeros(FAT_WT, dtype=bfloat16)
    wt_i32 = np.frombuffer(weights.tobytes(), dtype=np.int32)

    npu_kernel = NPUKernel(xclbin_path=str(xclbin_path), kernel_name="MLIR_AIE", insts_path=str(insts_path))
    handle = aie_utils.DefaultNPURuntime.load(npu_kernel)

    wt_buf = XRTTensor.from_torch(torch.from_numpy(wt_i32.copy()).to(torch.int32))
    act_buf = XRTTensor.from_torch(torch.from_numpy(act_i32.copy()).to(torch.int32))
    out_buf = XRTTensor((OUT_I32,), dtype=np.int32)

    print("  Executing...")
    try:
        result = aie_utils.DefaultNPURuntime.run(handle, [wt_buf, act_buf, out_buf])
        print(f"  Time: {result.npu_time/1e3:.1f} us")
    except Exception as e:
        print(f"  FAILED: {e}")
        return False
    finally:
        aie_utils.DefaultNPURuntime.cleanup()

    # Verify
    output_i32 = out_buf.to_torch().numpy()
    npu_output = np.frombuffer(output_i32.tobytes(), dtype=bfloat16)

    # Expected: each core copies act[r*32:(r+1)*32] → intermediate
    # Then gathered = [0..31, 32..63, 64..95, 96..127] (same as activation)
    # Then each core copies gathered[0:32] → output
    # So output = 4 copies of [0,1,2,...,31]
    expected = np.tile(np.arange(32, dtype=np.float32).astype(bfloat16), 4)

    print(f"  Output[0:4]: {npu_output[:4]}")
    print(f"  Output[32:36]: {npu_output[32:36]}")
    print(f"  Output[64:68]: {npu_output[64:68]}")
    print(f"  Output[96:100]: {npu_output[96:100]}")
    print(f"  Expected[0:4]: {expected[:4]}")
    print(f"  Expected[32:36]: {expected[32:36]}")

    ref_f32 = expected.astype(np.float32)
    npu_f32 = npu_output.astype(np.float32)
    max_err = np.max(np.abs(ref_f32 - npu_f32))
    if max_err == 0:
        print("\nPASS: Full 1-column pipeline works!")
    else:
        print(f"\nFAIL: max_err = {max_err}")
        # Show mismatches
        for i in range(0, TOTAL_OUT, M):
            chunk = npu_f32[i:i+4]
            ref_chunk = ref_f32[i:i+4]
            if not np.array_equal(chunk, ref_chunk):
                print(f"  [{i}:{i+4}] got={chunk} expected={ref_chunk}")
    return max_err == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
