#!/usr/bin/env python3
"""Minimal test: 4-core output via packet-switch through memtile to shim.

Tests ONLY the output path (no weights, no gather, no compute):
- Shim sends 4×32 bf16 activation (multicast to 4 cores)
- Each core copies its section to output buffer
- Core sends output via packet to memtile S2MM ch2
- Memtile aggregates 4×32=128 bf16 and forwards to shim S2MM ch0
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
TOTAL_OUT = NUM_CORES * M  # 128 bf16
ACT_SIZE = TOTAL_OUT  # 128 bf16 input (each core gets 32)
ACT_I32 = ACT_SIZE * 2 // 4  # 64 i32
OUT_I32 = TOTAL_OUT * 2 // 4  # 64 i32

EXPERIMENT_DIR = Path(__file__).parent.resolve()


def generate_mlir():
    """Generate minimal MLIR for output-via-memtile-packet test."""
    lines = []
    lines.append("module {")
    lines.append("  aie.device(npu2) {")
    lines.append("    %shim = aie.tile(0, 0)")
    lines.append("    %mt = aie.tile(0, 1)")
    for r in range(NUM_CORES):
        lines.append(f"    %core{r} = aie.tile(0, {r+2})")

    # Memtile buffers
    lines.append(f"    %mt_act = aie.buffer(%mt) {{sym_name = \"mt_act\"}} : memref<{ACT_SIZE}xbf16>")
    lines.append(f"    %mt_out = aie.buffer(%mt) {{sym_name = \"mt_out\"}} : memref<{TOTAL_OUT}xbf16>")

    # Memtile locks
    lines.append(f"    %mt_act_empty = aie.lock(%mt, 0) {{init = 1 : i32, sym_name = \"mt_act_empty\"}}")
    lines.append(f"    %mt_act_full  = aie.lock(%mt, 1) {{init = 0 : i32, sym_name = \"mt_act_full\"}}")
    lines.append(f"    %mt_out_empty = aie.lock(%mt, 2) {{init = {NUM_CORES} : i32, sym_name = \"mt_out_empty\"}}")
    lines.append(f"    %mt_out_full  = aie.lock(%mt, 3) {{init = 0 : i32, sym_name = \"mt_out_full\"}}")

    # Core buffers and locks
    for r in range(NUM_CORES):
        lines.append(f"    %c{r}_act = aie.buffer(%core{r}) {{sym_name = \"c{r}_act\"}} : memref<{ACT_SIZE}xbf16>")
        lines.append(f"    %c{r}_out = aie.buffer(%core{r}) {{sym_name = \"c{r}_out\"}} : memref<{M}xbf16>")
        lines.append(f"    %c{r}_act_empty = aie.lock(%core{r}, 0) {{init = 1 : i32, sym_name = \"c{r}_act_empty\"}}")
        lines.append(f"    %c{r}_act_full  = aie.lock(%core{r}, 1) {{init = 0 : i32, sym_name = \"c{r}_act_full\"}}")
        lines.append(f"    %c{r}_out_prod  = aie.lock(%core{r}, 2) {{init = 1 : i32, sym_name = \"c{r}_out_prod\"}}")
        lines.append(f"    %c{r}_out_cons  = aie.lock(%core{r}, 3) {{init = 0 : i32, sym_name = \"c{r}_out_cons\"}}")

    # Flows
    lines.append("    aie.flow(%shim, DMA : 0, %mt, DMA : 0)")  # act: shim → memtile
    # Multicast activation from memtile to all cores
    for r in range(NUM_CORES):
        lines.append(f"    aie.flow(%mt, DMA : 0, %core{r}, DMA : 0)")
    # Output: core → memtile via packet (memtile S2MM ch2)
    for r in range(NUM_CORES):
        lines.append(f"    aie.packet_flow({r}) {{")
        lines.append(f"      aie.packet_source<%core{r}, DMA : 1>")
        lines.append(f"      aie.packet_dest<%mt, DMA : 3>")
        lines.append(f"    }}")
    # Memtile → shim output
    lines.append("    aie.flow(%mt, DMA : 5, %shim, DMA : 0)")

    # Copy kernel declaration
    lines.append(f'    func.func private @copy_section(memref<{ACT_SIZE}xbf16>, memref<{M}xbf16>, i32) attributes {{link_with = "{EXPERIMENT_DIR}/copy_kernel.o"}}')

    # Core programs
    for r in range(NUM_CORES):
        lines.append(f"""    %prog{r} = aie.core(%core{r}) {{
      %offset = arith.constant {r * M} : i32
      aie.use_lock(%c{r}_act_full, AcquireGreaterEqual, 1)
      aie.use_lock(%c{r}_out_prod, AcquireGreaterEqual, 1)
      func.call @copy_section(%c{r}_act, %c{r}_out, %offset)
        : (memref<{ACT_SIZE}xbf16>, memref<{M}xbf16>, i32) -> ()
      aie.use_lock(%c{r}_act_empty, Release, 1)
      aie.use_lock(%c{r}_out_cons, Release, 1)
      aie.end
    }}""")

    # Core DMAs
    for r in range(NUM_CORES):
        lines.append(f"""    %mem{r} = aie.mem(%core{r}) {{
      %0 = aie.dma_start(S2MM, 0, ^act, ^out_start)
    ^act:
      aie.use_lock(%c{r}_act_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%c{r}_act : memref<{ACT_SIZE}xbf16>, 0, {ACT_SIZE}) {{bd_id = 0 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%c{r}_act_full, Release, 1)
      aie.next_bd ^act
    ^out_start:
      %1 = aie.dma_start(MM2S, 1, ^out, ^end)
    ^out:
      aie.use_lock(%c{r}_out_cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%c{r}_out : memref<{M}xbf16>, 0, {M}) {{bd_id = 1 : i32, next_bd_id = 1 : i32, packet = #aie.packet_info<pkt_type = 0, pkt_id = {r}>}}
      aie.use_lock(%c{r}_out_prod, Release, 1)
      aie.next_bd ^out
    ^end:
      aie.end
    }}""")

    # Memtile DMA
    lines.append(f"""    %mtdma = aie.memtile_dma(%mt) {{
      // S2MM ch0 (even, BD 0): activation from shim
      %0 = aie.dma_start(S2MM, 0, ^act_recv, ^out_recv_start)
    ^act_recv:
      aie.use_lock(%mt_act_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt_act : memref<{ACT_SIZE}xbf16>, 0, {ACT_SIZE}) {{bd_id = 0 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%mt_act_full, Release, 1)
      aie.next_bd ^act_recv

      // S2MM ch3 (odd, BD 26-29): output collect from 4 cores via packet
    ^out_recv_start:
      %1 = aie.dma_start(S2MM, 3, ^out_r0, ^act_send_start)
    ^out_r0:
      aie.use_lock(%mt_out_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt_out : memref<{TOTAL_OUT}xbf16>, 0, {M}) {{bd_id = 26 : i32, next_bd_id = 27 : i32}}
      aie.use_lock(%mt_out_full, Release, 1)
      aie.next_bd ^out_r1
    ^out_r1:
      aie.use_lock(%mt_out_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt_out : memref<{TOTAL_OUT}xbf16>, {M}, {M}) {{bd_id = 27 : i32, next_bd_id = 28 : i32}}
      aie.use_lock(%mt_out_full, Release, 1)
      aie.next_bd ^out_r2
    ^out_r2:
      aie.use_lock(%mt_out_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt_out : memref<{TOTAL_OUT}xbf16>, {2*M}, {M}) {{bd_id = 28 : i32, next_bd_id = 29 : i32}}
      aie.use_lock(%mt_out_full, Release, 1)
      aie.next_bd ^out_r3
    ^out_r3:
      aie.use_lock(%mt_out_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt_out : memref<{TOTAL_OUT}xbf16>, {3*M}, {M}) {{bd_id = 29 : i32, next_bd_id = 26 : i32}}
      aie.use_lock(%mt_out_full, Release, 1)
      aie.next_bd ^out_r0

      // MM2S ch0 (even, BD 1): activation multicast to all cores
    ^act_send_start:
      %2 = aie.dma_start(MM2S, 0, ^act_send, ^out_send_start)
    ^act_send:
      aie.use_lock(%mt_act_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%mt_act : memref<{ACT_SIZE}xbf16>, 0, {ACT_SIZE}) {{bd_id = 1 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%mt_act_empty, Release, 1)
      aie.next_bd ^act_send

      // MM2S ch5 (odd, BD 34): output forward to shim
    ^out_send_start:
      %3 = aie.dma_start(MM2S, 5, ^out_send, ^end)
    ^out_send:
      aie.use_lock(%mt_out_full, AcquireGreaterEqual, {NUM_CORES})
      aie.dma_bd(%mt_out : memref<{TOTAL_OUT}xbf16>, 0, {TOTAL_OUT}) {{bd_id = 34 : i32, next_bd_id = 34 : i32}}
      aie.use_lock(%mt_out_empty, Release, {NUM_CORES})
      aie.next_bd ^out_send
    ^end:
      aie.end
    }}""")

    # Runtime sequence
    lines.append(f"""    aie.runtime_sequence(%act_bo: memref<{ACT_I32}xi32>, %out_bo: memref<{OUT_I32}xi32>) {{
      // Push activation: shim MM2S ch0
      aiex.npu.writebd {{bd_id = 0 : i32, buffer_length = {ACT_I32} : i32, buffer_offset = 0 : i32, burst_length = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}}
      aiex.npu.address_patch {{addr = 118788 : ui32, arg_idx = 0 : i32, arg_plus = 0 : i32}}
      aiex.npu.push_queue(0, 0, MM2S : 0) {{bd_id = 0 : i32, issue_token = false, repeat_count = 0 : i32}}

      // Receive output: shim S2MM ch0
      aiex.npu.writebd {{bd_id = 3 : i32, buffer_length = {OUT_I32} : i32, buffer_offset = 0 : i32, burst_length = 0 : i32, column = 0 : i32, d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, enable_packet = 0 : i32, iteration_current = 0 : i32, iteration_size = 0 : i32, iteration_stride = 0 : i32, lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, next_bd = 0 : i32, out_of_order_id = 0 : i32, packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}}
      aiex.npu.address_patch {{addr = 118884 : ui32, arg_idx = 1 : i32, arg_plus = 0 : i32}}
      aiex.npu.push_queue(0, 0, S2MM : 0) {{bd_id = 3 : i32, issue_token = true, repeat_count = 0 : i32}}

      // Sync
      aiex.npu.sync {{channel = 0 : i32, column = 0 : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}}
    }}""")

    lines.append("  }")
    lines.append("}")
    return "\n".join(lines)


def main():
    # Write copy kernel
    kernel_src = EXPERIMENT_DIR / "copy_kernel.cc"
    if not kernel_src.exists():
        kernel_src.write_text('''#include <aie_api/aie.hpp>
extern "C" void copy_section(bfloat16* src, bfloat16* dst, int offset) {
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
    build_dir = EXPERIMENT_DIR / "build_output_test"
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

    # Input: 128 bf16 = [0, 1, 2, ..., 127] as bf16
    activation = np.arange(TOTAL_OUT, dtype=np.float32).astype(bfloat16)
    act_i32 = np.frombuffer(activation.tobytes(), dtype=np.int32)

    npu_kernel = NPUKernel(xclbin_path=str(xclbin_path), kernel_name="MLIR_AIE", insts_path=str(insts_path))
    handle = aie_utils.DefaultNPURuntime.load(npu_kernel)

    act_buf = XRTTensor.from_torch(torch.from_numpy(act_i32.copy()).to(torch.int32))
    out_buf = XRTTensor((OUT_I32,), dtype=np.int32)

    print("  Executing...")
    try:
        result = aie_utils.DefaultNPURuntime.run(handle, [act_buf, out_buf])
        print(f"  Time: {result.npu_time/1e3:.1f} us")
    except Exception as e:
        print(f"  FAILED: {e}")
        return False
    finally:
        aie_utils.DefaultNPURuntime.cleanup()

    # Verify
    output_i32 = out_buf.to_torch().numpy()
    npu_output = np.frombuffer(output_i32.tobytes(), dtype=bfloat16)
    print(f"  Input[0:4]: {activation[:4]}")
    print(f"  Output[0:4]: {npu_output[:4]}")
    print(f"  Output[32:36]: {npu_output[32:36]}")
    print(f"  Output[64:68]: {npu_output[64:68]}")
    print(f"  Output[96:100]: {npu_output[96:100]}")

    # Expected: output = input (identity copy through fabric)
    ref = activation.astype(np.float32)
    npu_f32 = npu_output.astype(np.float32)
    max_err = np.max(np.abs(ref - npu_f32))
    if max_err == 0:
        print("\nPASS: Output-via-memtile-packet works!")
    else:
        print(f"\nFAIL: max_err = {max_err}")
    return max_err == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
