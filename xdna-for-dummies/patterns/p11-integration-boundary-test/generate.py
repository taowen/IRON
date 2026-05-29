"""Generate MLIR-AIE for P11: integration boundary testing demo."""

from __future__ import annotations

from pathlib import Path

from reference import DATA_DWORDS

EXPERIMENT_DIR = Path(__file__).parent.resolve()


def _shim_bd_address(column: int, bd_id: int) -> int:
    return column * 0x02000000 + 0x1D004 + bd_id * 0x20


def npu_writebd(column: int, bd_id: int, buffer_length: int, buffer_offset: int) -> str:
    return (
        f"      aiex.npu.writebd {{bd_id = {bd_id} : i32, "
        f"buffer_length = {buffer_length} : i32, buffer_offset = {buffer_offset} : i32, "
        f"burst_length = 64 : i32, column = {column} : i32, "
        f"d0_size = 0 : i32, d0_stride = 0 : i32, "
        f"d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, "
        f"d1_size = 0 : i32, d1_stride = 0 : i32, "
        f"d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, "
        f"d2_size = 0 : i32, d2_stride = 0 : i32, "
        f"d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, "
        f"enable_packet = 0 : i32, iteration_current = 0 : i32, "
        f"iteration_size = 0 : i32, iteration_stride = 0 : i32, "
        f"lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, "
        f"lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, "
        f"next_bd = 0 : i32, out_of_order_id = 0 : i32, "
        f"packet_id = 0 : i32, packet_type = 0 : i32, "
        f"row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}}"
    )


def npu_address_patch(column: int, bd_id: int, arg_idx: int, arg_plus_bytes: int) -> str:
    return (
        f"      aiex.npu.address_patch {{addr = {_shim_bd_address(column, bd_id)} : ui32, "
        f"arg_idx = {arg_idx} : i32, arg_plus = {arg_plus_bytes} : i32}}"
    )


def npu_push_queue(column: int, direction: str, channel: int, bd_id: int, repeat: int = 0) -> str:
    return (
        f"      aiex.npu.push_queue({column}, 0, {direction} : {channel}) "
        f"{{bd_id = {bd_id} : i32, issue_token = true, repeat_count = {repeat} : i32}}"
    )


def npu_sync(column: int, channel: int, direction: int = 0) -> str:
    return (
        f"      aiex.npu.sync {{channel = {channel} : i32, column = {column} : i32, "
        f"column_num = 1 : i32, direction = {direction} : i32, row = 0 : i32, row_num = 1 : i32}}"
    )


def generate_mlir() -> str:
    return f"""module {{
  aie.device(npu2) {{
    %shim = aie.tile(2, 0)
    %compute = aie.tile(2, 2)

    aie.flow(%shim, DMA : 0, %compute, DMA : 0)
    aie.flow(%compute, DMA : 1, %shim, DMA : 0)

    func.func private @transform(memref<{DATA_DWORDS}xi32>, memref<{DATA_DWORDS}xi32>, i32) attributes {{link_with = "{EXPERIMENT_DIR}/kernel.o"}}

    %in_buf = aie.buffer(%compute) {{sym_name = "in_buf"}} : memref<{DATA_DWORDS}xi32>
    %out_buf = aie.buffer(%compute) {{sym_name = "out_buf"}} : memref<{DATA_DWORDS}xi32>
    %in_empty = aie.lock(%compute, 0) {{init = 1 : i32, sym_name = "in_empty"}}
    %in_full = aie.lock(%compute, 1) {{init = 0 : i32, sym_name = "in_full"}}
    %out_empty = aie.lock(%compute, 2) {{init = 1 : i32, sym_name = "out_empty"}}
    %out_full = aie.lock(%compute, 3) {{init = 0 : i32, sym_name = "out_full"}}

    %core = aie.core(%compute) {{
      %len = arith.constant {DATA_DWORDS} : i32
      aie.use_lock(%in_full, AcquireGreaterEqual, 1)
      aie.use_lock(%out_empty, AcquireGreaterEqual, 1)
      func.call @transform(%in_buf, %out_buf, %len) : (memref<{DATA_DWORDS}xi32>, memref<{DATA_DWORDS}xi32>, i32) -> ()
      aie.use_lock(%in_empty, Release, 1)
      aie.use_lock(%out_full, Release, 1)
      aie.end
    }}

    %mem = aie.mem(%compute) {{
      %0 = aie.dma_start(S2MM, 0, ^recv, ^send_start)
    ^recv:
      aie.use_lock(%in_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%in_buf : memref<{DATA_DWORDS}xi32>, 0, {DATA_DWORDS}) {{bd_id = 0 : i32}}
      aie.use_lock(%in_full, Release, 1)
      aie.next_bd ^recv
    ^send_start:
      %1 = aie.dma_start(MM2S, 1, ^send, ^end)
    ^send:
      aie.use_lock(%out_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_buf : memref<{DATA_DWORDS}xi32>, 0, {DATA_DWORDS}) {{bd_id = 1 : i32}}
      aie.use_lock(%out_empty, Release, 1)
      aie.next_bd ^send
    ^end:
      aie.end
    }}

    aie.runtime_sequence(%input: memref<{DATA_DWORDS}xi32>, %output: memref<{DATA_DWORDS}xi32>) {{
{npu_writebd(2, 0, DATA_DWORDS, 0)}
{npu_address_patch(2, 0, 0, 0)}
{npu_push_queue(2, "MM2S", 0, 0, repeat=0)}
{npu_writebd(2, 1, DATA_DWORDS, 0)}
{npu_address_patch(2, 1, 1, 0)}
{npu_push_queue(2, "S2MM", 0, 1, repeat=0)}
{npu_sync(2, 0, direction=0)}
    }}
  }}
}}
"""


def validate_generated_mlir(mlir: str) -> list[str]:
    errors: list[str] = []
    required = [
        "aie.flow(%shim, DMA : 0, %compute, DMA : 0)",
        "aie.flow(%compute, DMA : 1, %shim, DMA : 0)",
        "@transform",
        "kernel.o",
        "in_buf",
        "out_buf",
    ]
    for marker in required:
        if marker not in mlir:
            errors.append(f"missing: {marker}")
    return errors
