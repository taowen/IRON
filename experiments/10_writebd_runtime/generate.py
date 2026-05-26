"""
Experiment 10: writebd Runtime with Single KV Cache BO

Same static infrastructure as exp 09 (memtiles, cores, flows, locks), but replaces
the runtime_sequence with writebd/address_patch/push_queue/sync — the actual NPU
instruction format used by FastFlowLM. 4 planes are offsets into a single KV cache BO.
"""

from pathlib import Path
from math import ceil

NUM_KV_HEADS_PER_GROUP = 4
HEAD_DIM = 32
TOKENS_PER_TILE = 16
PLANE_TILE_DWORDS = NUM_KV_HEADS_PER_GROUP * HEAD_DIM * TOKENS_PER_TILE  # 2048


def _shim_bd_address(column: int, bd_id: int) -> int:
    return column * 0x02000000 + 0x1D004 + bd_id * 0x20


def _npu_writebd(column: int, bd_id: int, buffer_length: int, buffer_offset: int) -> str:
    return (
        f"      aiex.npu.writebd {{bd_id = {bd_id} : i32, "
        f"buffer_length = {buffer_length} : i32, buffer_offset = {buffer_offset} : i32, "
        f"burst_length = 0 : i32, "
        f"column = {column} : i32, "
        f"d0_size = 0 : i32, d0_stride = 0 : i32, d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, "
        f"d1_size = 0 : i32, d1_stride = 0 : i32, d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, "
        f"d2_size = 0 : i32, d2_stride = 0 : i32, d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, "
        f"enable_packet = 0 : i32, iteration_current = 0 : i32, "
        f"iteration_size = 0 : i32, iteration_stride = 0 : i32, "
        f"lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, "
        f"lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, "
        f"next_bd = 0 : i32, out_of_order_id = 0 : i32, "
        f"packet_id = 0 : i32, packet_type = 0 : i32, "
        f"row = 0 : i32, use_next_bd = 0 : i32, valid_bd = 1 : i32}}"
    )


def _npu_address_patch(column: int, bd_id: int, arg_idx: int, arg_plus_bytes: int) -> str:
    addr = _shim_bd_address(column, bd_id)
    return f"      aiex.npu.address_patch {{addr = {addr} : ui32, arg_idx = {arg_idx} : i32, arg_plus = {arg_plus_bytes} : i32}}"


def _npu_push_queue(column: int, direction: str, channel: int, bd_id: int, issue_token: bool) -> str:
    token = "true" if issue_token else "false"
    return f"      aiex.npu.push_queue({column}, 0, {direction} : {channel}) {{bd_id = {bd_id} : i32, issue_token = {token}, repeat_count = 0 : i32}}"


def _npu_sync(column: int, channel: int = 0) -> str:
    return f"      aiex.npu.sync {{channel = {channel} : i32, column = {column} : i32, column_num = 1 : i32, direction = 0 : i32, row = 0 : i32, row_num = 1 : i32}}"


def generate_mlir(L: int) -> str:
    """Generate MLIR for effective token length L."""
    num_tiles = ceil(L / TOKENS_PER_TILE)
    last_valid = L - (num_tiles - 1) * TOKENS_PER_TILE
    total_plane_elems = num_tiles * PLANE_TILE_DWORDS
    total_plane_bytes = total_plane_elems * 4
    total_kv_elems = total_plane_elems * 4
    experiment_dir = Path(__file__).parent.resolve()

    # Plane offsets in bytes within the single KV cache BO
    k03_offset = 0
    v03_offset = total_plane_bytes
    k47_offset = total_plane_bytes * 2
    v47_offset = total_plane_bytes * 3

    # Build runtime_sequence with writebd ops
    rt_lines = []
    rt_lines.append(f"    aie.runtime_sequence(%kv_cache: memref<{total_kv_elems}xi32>, %output: memref<256xi32>) {{")

    # k03: shim0 MM2S ch0, bd_id=0
    rt_lines.append(_npu_writebd(0, 0, total_plane_elems, k03_offset))
    rt_lines.append(_npu_address_patch(0, 0, 0, k03_offset))
    rt_lines.append(_npu_push_queue(0, "MM2S", 0, 0, False))

    # k47: shim0 MM2S ch1, bd_id=1
    rt_lines.append(_npu_writebd(0, 1, total_plane_elems, k47_offset))
    rt_lines.append(_npu_address_patch(0, 1, 0, k47_offset))
    rt_lines.append(_npu_push_queue(0, "MM2S", 1, 1, False))

    # v03: shim1 MM2S ch0, bd_id=0
    rt_lines.append(_npu_writebd(1, 0, total_plane_elems, v03_offset))
    rt_lines.append(_npu_address_patch(1, 0, 0, v03_offset))
    rt_lines.append(_npu_push_queue(1, "MM2S", 0, 0, False))

    # v47: shim1 MM2S ch1, bd_id=1
    rt_lines.append(_npu_writebd(1, 1, total_plane_elems, v47_offset))
    rt_lines.append(_npu_address_patch(1, 1, 0, v47_offset))
    rt_lines.append(_npu_push_queue(1, "MM2S", 1, 1, False))

    # output worker0: shim0 S2MM ch0, bd_id=2
    rt_lines.append(_npu_writebd(0, 2, 128, 0))
    rt_lines.append(_npu_address_patch(0, 2, 1, 0))
    rt_lines.append(_npu_push_queue(0, "S2MM", 0, 2, True))

    # output worker1: shim1 S2MM ch0, bd_id=2
    rt_lines.append(_npu_writebd(1, 2, 128, 512))
    rt_lines.append(_npu_address_patch(1, 2, 1, 512))
    rt_lines.append(_npu_push_queue(1, "S2MM", 0, 2, True))

    # Wait for both outputs
    rt_lines.append(_npu_sync(0, 0))
    rt_lines.append(_npu_sync(1, 0))

    rt_lines.append("    }")
    runtime_sequence = "\n".join(rt_lines)

    return f"""module {{
  aie.device(npu2) {{
    // === Tile declarations ===
    // Column 0: K planes
    %shim0 = aie.tile(0, 0)
    %mem0  = aie.tile(0, 1)
    %worker0 = aie.tile(0, 2)

    // Column 1: V planes
    %shim1 = aie.tile(1, 0)
    %mem1  = aie.tile(1, 1)
    %worker1 = aie.tile(1, 2)

    // === Memtile 0 (col 0) buffers: ring A (k03), ring B (k47) ===
    %m0_a_ping = aie.buffer(%mem0) {{sym_name = "m0_a_ping"}} : memref<{PLANE_TILE_DWORDS}xi32>
    %m0_a_pong = aie.buffer(%mem0) {{sym_name = "m0_a_pong"}} : memref<{PLANE_TILE_DWORDS}xi32>
    %m0_b_ping = aie.buffer(%mem0) {{sym_name = "m0_b_ping"}} : memref<{PLANE_TILE_DWORDS}xi32>
    %m0_b_pong = aie.buffer(%mem0) {{sym_name = "m0_b_pong"}} : memref<{PLANE_TILE_DWORDS}xi32>

    // Memtile 0 locks: ring A (locks 0,1), ring B (locks 2,3)
    %m0_a_empty = aie.lock(%mem0, 0) {{init = 2 : i32, sym_name = "m0_a_empty"}}
    %m0_a_full  = aie.lock(%mem0, 1) {{init = 0 : i32, sym_name = "m0_a_full"}}
    %m0_b_empty = aie.lock(%mem0, 2) {{init = 2 : i32, sym_name = "m0_b_empty"}}
    %m0_b_full  = aie.lock(%mem0, 3) {{init = 0 : i32, sym_name = "m0_b_full"}}

    // === Memtile 1 (col 1) buffers: ring A (v03), ring B (v47) ===
    %m1_a_ping = aie.buffer(%mem1) {{sym_name = "m1_a_ping"}} : memref<{PLANE_TILE_DWORDS}xi32>
    %m1_a_pong = aie.buffer(%mem1) {{sym_name = "m1_a_pong"}} : memref<{PLANE_TILE_DWORDS}xi32>
    %m1_b_ping = aie.buffer(%mem1) {{sym_name = "m1_b_ping"}} : memref<{PLANE_TILE_DWORDS}xi32>
    %m1_b_pong = aie.buffer(%mem1) {{sym_name = "m1_b_pong"}} : memref<{PLANE_TILE_DWORDS}xi32>

    // Memtile 1 locks: ring A (locks 0,1), ring B (locks 2,3)
    %m1_a_empty = aie.lock(%mem1, 0) {{init = 2 : i32, sym_name = "m1_a_empty"}}
    %m1_a_full  = aie.lock(%mem1, 1) {{init = 0 : i32, sym_name = "m1_a_full"}}
    %m1_b_empty = aie.lock(%mem1, 2) {{init = 2 : i32, sym_name = "m1_b_empty"}}
    %m1_b_full  = aie.lock(%mem1, 3) {{init = 0 : i32, sym_name = "m1_b_full"}}

    // === Worker 0 (col 0, row 2) buffers ===
    %w0_k_ping = aie.buffer(%worker0) {{sym_name = "w0_k_ping"}} : memref<{PLANE_TILE_DWORDS}xi32>
    %w0_k_pong = aie.buffer(%worker0) {{sym_name = "w0_k_pong"}} : memref<{PLANE_TILE_DWORDS}xi32>
    %w0_v_ping = aie.buffer(%worker0) {{sym_name = "w0_v_ping"}} : memref<{PLANE_TILE_DWORDS}xi32>
    %w0_v_pong = aie.buffer(%worker0) {{sym_name = "w0_v_pong"}} : memref<{PLANE_TILE_DWORDS}xi32>
    %w0_out    = aie.buffer(%worker0) {{sym_name = "w0_out"}} : memref<128xi32>

    // Worker 0 locks
    %w0_k_empty = aie.lock(%worker0, 0) {{init = 2 : i32, sym_name = "w0_k_empty"}}
    %w0_k_full  = aie.lock(%worker0, 1) {{init = 0 : i32, sym_name = "w0_k_full"}}
    %w0_v_empty = aie.lock(%worker0, 2) {{init = 2 : i32, sym_name = "w0_v_empty"}}
    %w0_v_full  = aie.lock(%worker0, 3) {{init = 0 : i32, sym_name = "w0_v_full"}}
    %w0_out_prod = aie.lock(%worker0, 4) {{init = 1 : i32, sym_name = "w0_out_prod"}}
    %w0_out_cons = aie.lock(%worker0, 5) {{init = 0 : i32, sym_name = "w0_out_cons"}}

    // === Worker 1 (col 1, row 2) buffers ===
    %w1_k_ping = aie.buffer(%worker1) {{sym_name = "w1_k_ping"}} : memref<{PLANE_TILE_DWORDS}xi32>
    %w1_k_pong = aie.buffer(%worker1) {{sym_name = "w1_k_pong"}} : memref<{PLANE_TILE_DWORDS}xi32>
    %w1_v_ping = aie.buffer(%worker1) {{sym_name = "w1_v_ping"}} : memref<{PLANE_TILE_DWORDS}xi32>
    %w1_v_pong = aie.buffer(%worker1) {{sym_name = "w1_v_pong"}} : memref<{PLANE_TILE_DWORDS}xi32>
    %w1_out    = aie.buffer(%worker1) {{sym_name = "w1_out"}} : memref<128xi32>

    // Worker 1 locks
    %w1_k_empty = aie.lock(%worker1, 0) {{init = 2 : i32, sym_name = "w1_k_empty"}}
    %w1_k_full  = aie.lock(%worker1, 1) {{init = 0 : i32, sym_name = "w1_k_full"}}
    %w1_v_empty = aie.lock(%worker1, 2) {{init = 2 : i32, sym_name = "w1_v_empty"}}
    %w1_v_full  = aie.lock(%worker1, 3) {{init = 0 : i32, sym_name = "w1_v_full"}}
    %w1_out_prod = aie.lock(%worker1, 4) {{init = 1 : i32, sym_name = "w1_out_prod"}}
    %w1_out_cons = aie.lock(%worker1, 5) {{init = 0 : i32, sym_name = "w1_out_cons"}}

    // === Flows ===
    // Shim → Memtile (4 plane inputs)
    aie.flow(%shim0, DMA : 0, %mem0, DMA : 0)    // k03 → memtile0 ring A
    aie.flow(%shim0, DMA : 1, %mem0, DMA : 1)    // k47 → memtile0 ring B
    aie.flow(%shim1, DMA : 0, %mem1, DMA : 0)    // v03 → memtile1 ring A
    aie.flow(%shim1, DMA : 1, %mem1, DMA : 1)    // v47 → memtile1 ring B

    // Memtile → Workers (cross-column routing)
    aie.flow(%mem0, DMA : 0, %worker0, DMA : 0)  // k03 → worker0 K
    aie.flow(%mem1, DMA : 0, %worker0, DMA : 1)  // v03 → worker0 V
    aie.flow(%mem0, DMA : 1, %worker1, DMA : 0)  // k47 → worker1 K
    aie.flow(%mem1, DMA : 1, %worker1, DMA : 1)  // v47 → worker1 V

    // Workers → Shim (output)
    aie.flow(%worker0, DMA : 0, %shim0, DMA : 0) // worker0 out
    aie.flow(%worker1, DMA : 0, %shim1, DMA : 0) // worker1 out

    // === External kernel declarations ===
    func.func private @zero_output(memref<128xi32>) attributes {{link_with = "{experiment_dir}/kv_attention.o"}}
    func.func private @kv_attention(memref<{PLANE_TILE_DWORDS}xi32>, memref<{PLANE_TILE_DWORDS}xi32>, memref<128xi32>, i32, i32, i32, i32) attributes {{link_with = "{experiment_dir}/kv_attention.o"}}

    // === Worker 0 core: heads 0-3 (head_offset=0) ===
    %core_w0 = aie.core(%worker0) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2_i32 = arith.constant 2 : i32
      %c1_i32 = arith.constant 1 : i32
      %num_tiles_i32 = arith.constant {num_tiles} : i32
      %last_valid_i32 = arith.constant {last_valid} : i32
      %head_offset_i32 = arith.constant 0 : i32
      %num_tiles_idx = arith.constant {num_tiles} : index

      aie.use_lock(%w0_out_prod, AcquireGreaterEqual, 1)
      func.call @zero_output(%w0_out) : (memref<128xi32>) -> ()

      scf.for %i = %c0 to %num_tiles_idx step %c1 {{
        %i_i32 = arith.index_cast %i : index to i32
        %rem = arith.remsi %i_i32, %c2_i32 : i32
        %is_pong = arith.cmpi eq, %rem, %c1_i32 : i32

        aie.use_lock(%w0_k_full, AcquireGreaterEqual, 1)
        aie.use_lock(%w0_v_full, AcquireGreaterEqual, 1)

        scf.if %is_pong {{
          func.call @kv_attention(%w0_k_pong, %w0_v_pong, %w0_out, %i_i32, %num_tiles_i32, %last_valid_i32, %head_offset_i32)
            : (memref<{PLANE_TILE_DWORDS}xi32>, memref<{PLANE_TILE_DWORDS}xi32>, memref<128xi32>, i32, i32, i32, i32) -> ()
        }} else {{
          func.call @kv_attention(%w0_k_ping, %w0_v_ping, %w0_out, %i_i32, %num_tiles_i32, %last_valid_i32, %head_offset_i32)
            : (memref<{PLANE_TILE_DWORDS}xi32>, memref<{PLANE_TILE_DWORDS}xi32>, memref<128xi32>, i32, i32, i32, i32) -> ()
        }}

        aie.use_lock(%w0_k_empty, Release, 1)
        aie.use_lock(%w0_v_empty, Release, 1)
      }}

      aie.use_lock(%w0_out_cons, Release, 1)
      aie.end
    }}

    // === Worker 1 core: heads 4-7 (head_offset=4) ===
    %core_w1 = aie.core(%worker1) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2_i32 = arith.constant 2 : i32
      %c1_i32 = arith.constant 1 : i32
      %num_tiles_i32 = arith.constant {num_tiles} : i32
      %last_valid_i32 = arith.constant {last_valid} : i32
      %head_offset_i32 = arith.constant 4 : i32
      %num_tiles_idx = arith.constant {num_tiles} : index

      aie.use_lock(%w1_out_prod, AcquireGreaterEqual, 1)
      func.call @zero_output(%w1_out) : (memref<128xi32>) -> ()

      scf.for %i = %c0 to %num_tiles_idx step %c1 {{
        %i_i32 = arith.index_cast %i : index to i32
        %rem = arith.remsi %i_i32, %c2_i32 : i32
        %is_pong = arith.cmpi eq, %rem, %c1_i32 : i32

        aie.use_lock(%w1_k_full, AcquireGreaterEqual, 1)
        aie.use_lock(%w1_v_full, AcquireGreaterEqual, 1)

        scf.if %is_pong {{
          func.call @kv_attention(%w1_k_pong, %w1_v_pong, %w1_out, %i_i32, %num_tiles_i32, %last_valid_i32, %head_offset_i32)
            : (memref<{PLANE_TILE_DWORDS}xi32>, memref<{PLANE_TILE_DWORDS}xi32>, memref<128xi32>, i32, i32, i32, i32) -> ()
        }} else {{
          func.call @kv_attention(%w1_k_ping, %w1_v_ping, %w1_out, %i_i32, %num_tiles_i32, %last_valid_i32, %head_offset_i32)
            : (memref<{PLANE_TILE_DWORDS}xi32>, memref<{PLANE_TILE_DWORDS}xi32>, memref<128xi32>, i32, i32, i32, i32) -> ()
        }}

        aie.use_lock(%w1_k_empty, Release, 1)
        aie.use_lock(%w1_v_empty, Release, 1)
      }}

      aie.use_lock(%w1_out_cons, Release, 1)
      aie.end
    }}

    // === Memtile 0 DMA (col 0): ring A (k03) + ring B (k47) ===
    %memtile_dma_0 = aie.memtile_dma(%mem0) {{
      // Ring A: S2MM ch0 (shim → memtile, k03)
      %0 = aie.dma_start(S2MM, 0, ^a_s2mm_ping, ^b_start)
    ^a_s2mm_ping:
      aie.use_lock(%m0_a_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%m0_a_ping : memref<{PLANE_TILE_DWORDS}xi32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%m0_a_full, Release, 1)
      aie.next_bd ^a_s2mm_pong
    ^a_s2mm_pong:
      aie.use_lock(%m0_a_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%m0_a_pong : memref<{PLANE_TILE_DWORDS}xi32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%m0_a_full, Release, 1)
      aie.next_bd ^a_s2mm_ping

      // Ring A: MM2S ch0 (memtile → worker0 K)
    ^b_start:
      %1 = aie.dma_start(MM2S, 0, ^a_mm2s_ping, ^c_start)
    ^a_mm2s_ping:
      aie.use_lock(%m0_a_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%m0_a_ping : memref<{PLANE_TILE_DWORDS}xi32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 2 : i32, next_bd_id = 3 : i32}}
      aie.use_lock(%m0_a_empty, Release, 1)
      aie.next_bd ^a_mm2s_pong
    ^a_mm2s_pong:
      aie.use_lock(%m0_a_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%m0_a_pong : memref<{PLANE_TILE_DWORDS}xi32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 3 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%m0_a_empty, Release, 1)
      aie.next_bd ^a_mm2s_ping

      // Ring B: S2MM ch1 (shim → memtile, k47)
    ^c_start:
      %2 = aie.dma_start(S2MM, 1, ^b_s2mm_ping, ^d_start)
    ^b_s2mm_ping:
      aie.use_lock(%m0_b_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%m0_b_ping : memref<{PLANE_TILE_DWORDS}xi32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 24 : i32, next_bd_id = 25 : i32}}
      aie.use_lock(%m0_b_full, Release, 1)
      aie.next_bd ^b_s2mm_pong
    ^b_s2mm_pong:
      aie.use_lock(%m0_b_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%m0_b_pong : memref<{PLANE_TILE_DWORDS}xi32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 25 : i32, next_bd_id = 24 : i32}}
      aie.use_lock(%m0_b_full, Release, 1)
      aie.next_bd ^b_s2mm_ping

      // Ring B: MM2S ch1 (memtile → worker1 K)
    ^d_start:
      %3 = aie.dma_start(MM2S, 1, ^b_mm2s_ping, ^end)
    ^b_mm2s_ping:
      aie.use_lock(%m0_b_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%m0_b_ping : memref<{PLANE_TILE_DWORDS}xi32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 26 : i32, next_bd_id = 27 : i32}}
      aie.use_lock(%m0_b_empty, Release, 1)
      aie.next_bd ^b_mm2s_pong
    ^b_mm2s_pong:
      aie.use_lock(%m0_b_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%m0_b_pong : memref<{PLANE_TILE_DWORDS}xi32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 27 : i32, next_bd_id = 26 : i32}}
      aie.use_lock(%m0_b_empty, Release, 1)
      aie.next_bd ^b_mm2s_ping
    ^end:
      aie.end
    }}

    // === Memtile 1 DMA (col 1): ring A (v03) + ring B (v47) ===
    %memtile_dma_1 = aie.memtile_dma(%mem1) {{
      // Ring A: S2MM ch0 (shim → memtile, v03)
      %0 = aie.dma_start(S2MM, 0, ^a_s2mm_ping, ^b_start)
    ^a_s2mm_ping:
      aie.use_lock(%m1_a_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%m1_a_ping : memref<{PLANE_TILE_DWORDS}xi32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%m1_a_full, Release, 1)
      aie.next_bd ^a_s2mm_pong
    ^a_s2mm_pong:
      aie.use_lock(%m1_a_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%m1_a_pong : memref<{PLANE_TILE_DWORDS}xi32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%m1_a_full, Release, 1)
      aie.next_bd ^a_s2mm_ping

      // Ring A: MM2S ch0 (memtile → worker0 V)
    ^b_start:
      %1 = aie.dma_start(MM2S, 0, ^a_mm2s_ping, ^c_start)
    ^a_mm2s_ping:
      aie.use_lock(%m1_a_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%m1_a_ping : memref<{PLANE_TILE_DWORDS}xi32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 2 : i32, next_bd_id = 3 : i32}}
      aie.use_lock(%m1_a_empty, Release, 1)
      aie.next_bd ^a_mm2s_pong
    ^a_mm2s_pong:
      aie.use_lock(%m1_a_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%m1_a_pong : memref<{PLANE_TILE_DWORDS}xi32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 3 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%m1_a_empty, Release, 1)
      aie.next_bd ^a_mm2s_ping

      // Ring B: S2MM ch1 (shim → memtile, v47)
    ^c_start:
      %2 = aie.dma_start(S2MM, 1, ^b_s2mm_ping, ^d_start)
    ^b_s2mm_ping:
      aie.use_lock(%m1_b_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%m1_b_ping : memref<{PLANE_TILE_DWORDS}xi32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 24 : i32, next_bd_id = 25 : i32}}
      aie.use_lock(%m1_b_full, Release, 1)
      aie.next_bd ^b_s2mm_pong
    ^b_s2mm_pong:
      aie.use_lock(%m1_b_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%m1_b_pong : memref<{PLANE_TILE_DWORDS}xi32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 25 : i32, next_bd_id = 24 : i32}}
      aie.use_lock(%m1_b_full, Release, 1)
      aie.next_bd ^b_s2mm_ping

      // Ring B: MM2S ch1 (memtile → worker1 V)
    ^d_start:
      %3 = aie.dma_start(MM2S, 1, ^b_mm2s_ping, ^end)
    ^b_mm2s_ping:
      aie.use_lock(%m1_b_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%m1_b_ping : memref<{PLANE_TILE_DWORDS}xi32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 26 : i32, next_bd_id = 27 : i32}}
      aie.use_lock(%m1_b_empty, Release, 1)
      aie.next_bd ^b_mm2s_pong
    ^b_mm2s_pong:
      aie.use_lock(%m1_b_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%m1_b_pong : memref<{PLANE_TILE_DWORDS}xi32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 27 : i32, next_bd_id = 26 : i32}}
      aie.use_lock(%m1_b_empty, Release, 1)
      aie.next_bd ^b_mm2s_ping
    ^end:
      aie.end
    }}

    // === Worker 0 DMA: S2MM ch0 (K from mem0), S2MM ch1 (V from mem1), MM2S ch0 (output) ===
    %mem_w0 = aie.mem(%worker0) {{
      %0 = aie.dma_start(S2MM, 0, ^k_ping, ^v_start)
    ^k_ping:
      aie.use_lock(%w0_k_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%w0_k_ping : memref<{PLANE_TILE_DWORDS}xi32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%w0_k_full, Release, 1)
      aie.next_bd ^k_pong
    ^k_pong:
      aie.use_lock(%w0_k_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%w0_k_pong : memref<{PLANE_TILE_DWORDS}xi32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%w0_k_full, Release, 1)
      aie.next_bd ^k_ping
    ^v_start:
      %1 = aie.dma_start(S2MM, 1, ^v_ping, ^out_start)
    ^v_ping:
      aie.use_lock(%w0_v_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%w0_v_ping : memref<{PLANE_TILE_DWORDS}xi32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 2 : i32, next_bd_id = 3 : i32}}
      aie.use_lock(%w0_v_full, Release, 1)
      aie.next_bd ^v_pong
    ^v_pong:
      aie.use_lock(%w0_v_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%w0_v_pong : memref<{PLANE_TILE_DWORDS}xi32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 3 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%w0_v_full, Release, 1)
      aie.next_bd ^v_ping
    ^out_start:
      %2 = aie.dma_start(MM2S, 0, ^out_bd, ^end)
    ^out_bd:
      aie.use_lock(%w0_out_cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%w0_out : memref<128xi32>, 0, 128) {{bd_id = 4 : i32}}
      aie.use_lock(%w0_out_prod, Release, 1)
      aie.next_bd ^out_bd
    ^end:
      aie.end
    }}

    // === Worker 1 DMA: S2MM ch0 (K from mem0), S2MM ch1 (V from mem1), MM2S ch0 (output) ===
    %mem_w1 = aie.mem(%worker1) {{
      %0 = aie.dma_start(S2MM, 0, ^k_ping, ^v_start)
    ^k_ping:
      aie.use_lock(%w1_k_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%w1_k_ping : memref<{PLANE_TILE_DWORDS}xi32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%w1_k_full, Release, 1)
      aie.next_bd ^k_pong
    ^k_pong:
      aie.use_lock(%w1_k_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%w1_k_pong : memref<{PLANE_TILE_DWORDS}xi32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%w1_k_full, Release, 1)
      aie.next_bd ^k_ping
    ^v_start:
      %1 = aie.dma_start(S2MM, 1, ^v_ping, ^out_start)
    ^v_ping:
      aie.use_lock(%w1_v_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%w1_v_ping : memref<{PLANE_TILE_DWORDS}xi32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 2 : i32, next_bd_id = 3 : i32}}
      aie.use_lock(%w1_v_full, Release, 1)
      aie.next_bd ^v_pong
    ^v_pong:
      aie.use_lock(%w1_v_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%w1_v_pong : memref<{PLANE_TILE_DWORDS}xi32>, 0, {PLANE_TILE_DWORDS}) {{bd_id = 3 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%w1_v_full, Release, 1)
      aie.next_bd ^v_ping
    ^out_start:
      %2 = aie.dma_start(MM2S, 0, ^out_bd, ^end)
    ^out_bd:
      aie.use_lock(%w1_out_cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%w1_out : memref<128xi32>, 0, 128) {{bd_id = 4 : i32}}
      aie.use_lock(%w1_out_prod, Release, 1)
      aie.next_bd ^out_bd
    ^end:
      aie.end
    }}

    // === Runtime sequence: writebd + address_patch + push_queue + sync ===
{runtime_sequence}
  }}
}}
"""


if __name__ == "__main__":
    import sys
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    print(generate_mlir(L))
