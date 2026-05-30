"""Generate MLIR-AIE for the real Qwen3 c1r2 input RMSNorm replay boundary."""

from __future__ import annotations

from pathlib import Path

from contract import C1R2_PACKET_DWORDS
from mlir_utils import (
    flow,
    npu_address_patch,
    npu_push_queue,
    npu_sync,
    npu_writebd,
    require_count,
    require_dma_bd_lock_balance,
    require_max_address_patch_arg,
    require_npu_writebd_field_ranges,
    require_npu_writebd_id_limit,
)
from q4nx_reference import HIDDEN_DWORDS

CASE_NAME = "qwen3-8b-c1r2-input-norm-replay"
REPLAY_PAYLOAD_DWORDS = C1R2_PACKET_DWORDS - 1


def _runtime_sequence() -> str:
    return "\n".join(
        (
            f"    aie.runtime_sequence(%hidden: memref<{HIDDEN_DWORDS}xi32>, "
            f"%input_norm: memref<{HIDDEN_DWORDS}xi32>, "
            f"%output: memref<{REPLAY_PAYLOAD_DWORDS}xi32>) {{",
            npu_writebd(1, 0, HIDDEN_DWORDS, 0),
            npu_address_patch(1, 0, 0, 0),
            npu_push_queue(1, "MM2S", 0, 0),
            npu_writebd(1, 1, HIDDEN_DWORDS, 0),
            npu_address_patch(1, 1, 1, 0),
            npu_push_queue(1, "MM2S", 0, 1),
            npu_writebd(1, 2, REPLAY_PAYLOAD_DWORDS, 0),
            npu_address_patch(1, 2, 2, 0),
            npu_push_queue(1, "S2MM", 1, 2),
            npu_sync(1, 1),
            npu_sync(1, 0, direction=1),
            "    }",
        )
    )


def _full_vector_replay() -> str:
    return f"""
    %full_hidden = aie.buffer(%full) {{sym_name = "full_hidden"}} : memref<{HIDDEN_DWORDS}xi32>
    %full_norm = aie.buffer(%full) {{sym_name = "full_norm"}} : memref<{HIDDEN_DWORDS}xi32>
    %full_replay = aie.buffer(%full) {{sym_name = "full_replay"}} : memref<{C1R2_PACKET_DWORDS}xi32>
    %full_hidden_empty = aie.lock(%full, 0) {{init = 1 : i32, sym_name = "full_hidden_empty"}}
    %full_hidden_full = aie.lock(%full, 1) {{init = 0 : i32, sym_name = "full_hidden_full"}}
    %full_norm_empty = aie.lock(%full, 2) {{init = 1 : i32, sym_name = "full_norm_empty"}}
    %full_norm_full = aie.lock(%full, 3) {{init = 0 : i32, sym_name = "full_norm_full"}}
    %full_replay_empty = aie.lock(%full, 4) {{init = 1 : i32, sym_name = "full_replay_empty"}}
    %full_replay_full = aie.lock(%full, 5) {{init = 0 : i32, sym_name = "full_replay_full"}}

    %full_core = aie.core(%full) {{
      %payload_i32 = arith.constant {REPLAY_PAYLOAD_DWORDS} : i32
      aie.use_lock(%full_hidden_full, AcquireGreaterEqual, 1)
      aie.use_lock(%full_norm_full, AcquireGreaterEqual, 1)
      aie.use_lock(%full_replay_empty, AcquireGreaterEqual, 1)
      func.call @full_c1r2_make_input_norm_replay(%full_hidden, %full_norm, %full_replay, %payload_i32)
        : (memref<{HIDDEN_DWORDS}xi32>, memref<{HIDDEN_DWORDS}xi32>, memref<{C1R2_PACKET_DWORDS}xi32>, i32) -> ()
      aie.use_lock(%full_hidden_empty, Release, 1)
      aie.use_lock(%full_norm_empty, Release, 1)
      aie.use_lock(%full_replay_full, Release, 1)
      aie.end
    }}

    %full_mem = aie.mem(%full) {{
      %input_dma = aie.dma_start(S2MM, 1, ^hidden_in, ^replay_out_start)
    ^hidden_in:
      aie.use_lock(%full_hidden_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%full_hidden : memref<{HIDDEN_DWORDS}xi32>, 0, {HIDDEN_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%full_hidden_full, Release, 1)
      aie.next_bd ^norm_in
    ^norm_in:
      aie.use_lock(%full_norm_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%full_norm : memref<{HIDDEN_DWORDS}xi32>, 0, {HIDDEN_DWORDS}) {{bd_id = 1 : i32}}
      aie.use_lock(%full_norm_full, Release, 1)
      aie.next_bd ^input_end
    ^input_end:
      aie.end

    ^replay_out_start:
      %replay_dma = aie.dma_start(MM2S, 0, ^replay_out, ^end)
    ^replay_out:
      aie.use_lock(%full_replay_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%full_replay : memref<{C1R2_PACKET_DWORDS}xi32>, 1, {REPLAY_PAYLOAD_DWORDS}) {{bd_id = 2 : i32}}
      aie.use_lock(%full_replay_empty, Release, 1)
      aie.next_bd ^end
    ^end:
      aie.end
    }}
"""


def generate_mlir() -> str:
    experiment_dir = Path(__file__).parent.parent.resolve()
    return f"""module {{
  aie.device(npu2) {{
    %shim = aie.tile(1, 0)
    %full = aie.tile(1, 2)

    // case marker {CASE_NAME}
{flow("shim", 0, "full", 1)}
{flow("full", 0, "shim", 1)}

    func.func private @full_c1r2_make_input_norm_replay(memref<{HIDDEN_DWORDS}xi32>, memref<{HIDDEN_DWORDS}xi32>, memref<{C1R2_PACKET_DWORDS}xi32>, i32) attributes {{link_with = "{experiment_dir}/full_vector_station.o"}}

{_full_vector_replay()}
{_runtime_sequence()}
  }}
}}
"""


def validate_generated_mlir(mlir: str) -> list[str]:
    required = (
        f"case marker {CASE_NAME}",
        "full_c1r2_make_input_norm_replay",
        "full_vector_station.o",
        f"memref<{HIDDEN_DWORDS}xi32>",
        f"memref<{REPLAY_PAYLOAD_DWORDS}xi32>",
        "aie.dma_start(S2MM, 1, ^hidden_in, ^replay_out_start)",
        "aie.dma_start(MM2S, 0, ^replay_out, ^end)",
        f"aie.dma_bd(%full_replay : memref<{C1R2_PACKET_DWORDS}xi32>, 1, {REPLAY_PAYLOAD_DWORDS})",
        "arg_idx = 0 : i32",
        "arg_idx = 1 : i32",
        "arg_idx = 2 : i32",
    )
    errors = [f"missing c1r2 replay marker: {marker}" for marker in required if marker not in mlir]
    errors.extend(require_count(CASE_NAME, "tile flow", mlir.count("aie.flow("), 2))
    errors.extend(require_count(CASE_NAME, "address patches", mlir.count("aiex.npu.address_patch"), 3))
    errors.extend(require_count(CASE_NAME, "role object call", mlir.count("full_c1r2_make_input_norm_replay"), 2))
    errors.extend(require_dma_bd_lock_balance(CASE_NAME, mlir))
    errors.extend(require_max_address_patch_arg(CASE_NAME, mlir, 2))
    errors.extend(require_npu_writebd_id_limit(CASE_NAME, mlir, 15))
    errors.extend(require_npu_writebd_field_ranges(CASE_NAME, mlir))
    return errors


if __name__ == "__main__":
    print(generate_mlir())
