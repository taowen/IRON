"""
Experiment 09: Four-Plane Fanout with GQA Head-Group Mapping

Generates low-level MLIR-AIE with:
- 2 columns (col 0 = K planes, col 1 = V planes)
- 4 shim MM2S channels (k03, k47, v03, v47)
- 2 memtiles, each with 2 independent BD rings (ring A + ring B)
- 2 worker cores: worker0 (heads 0-3, k03+v03), worker1 (heads 4-7, k47+v47)
- Cross-column routing (worker0 gets K from col0, V from col1)
- Standard ping-pong lock per ring (no split needed — each ring has 1 consumer)
"""

from pathlib import Path
from math import ceil

NUM_KV_HEADS_PER_GROUP = 4
HEAD_DIM = 32
TOKENS_PER_TILE = 16
PLANE_TILE_DWORDS = NUM_KV_HEADS_PER_GROUP * HEAD_DIM * TOKENS_PER_TILE  # 2048


def _input_bd_dims(num_tiles: int) -> str:
    """Compute BD dimension array for plane input descriptor."""
    if PLANE_TILE_DWORDS <= 1023:
        return (f"[<size = 1, stride = 0>, <size = 1, stride = 0>, "
                f"<size = {num_tiles}, stride = {PLANE_TILE_DWORDS}>, "
                f"<size = {PLANE_TILE_DWORDS}, stride = 1>]")
    else:
        inner = 512
        outer = PLANE_TILE_DWORDS // inner
        return (f"[<size = 1, stride = 0>, <size = {num_tiles}, stride = {PLANE_TILE_DWORDS}>, "
                f"<size = {outer}, stride = {inner}>, <size = {inner}, stride = 1>]")


def generate_mlir(L: int) -> str:
    """Generate MLIR for effective token length L."""
    num_tiles = ceil(L / TOKENS_PER_TILE)
    last_valid = L - (num_tiles - 1) * TOKENS_PER_TILE
    total_plane_elems = num_tiles * PLANE_TILE_DWORDS
    input_dims = _input_bd_dims(num_tiles)
    experiment_dir = Path(__file__).parent.resolve()

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

    // === Shim DMA allocations ===
    aie.shim_dma_allocation @k03_alloc(%shim0, MM2S, 0)
    aie.shim_dma_allocation @k47_alloc(%shim0, MM2S, 1)
    aie.shim_dma_allocation @v03_alloc(%shim1, MM2S, 0)
    aie.shim_dma_allocation @v47_alloc(%shim1, MM2S, 1)
    aie.shim_dma_allocation @out0_alloc(%shim0, S2MM, 0)
    aie.shim_dma_allocation @out1_alloc(%shim1, S2MM, 0)

    // === Runtime sequence: 5 buffer args (XRT limit) ===
    aie.runtime_sequence(%k03: memref<{total_plane_elems}xi32>, %v03: memref<{total_plane_elems}xi32>, %k47: memref<{total_plane_elems}xi32>, %v47: memref<{total_plane_elems}xi32>, %output: memref<256xi32>) {{
      // 4 input descriptors
      %t0 = aiex.dma_configure_task_for @k03_alloc {{
        aie.dma_bd(%k03 : memref<{total_plane_elems}xi32>, 0, {total_plane_elems}, {input_dims}) {{burst_length = 0 : i32}}
        aie.end
      }}
      %t1 = aiex.dma_configure_task_for @k47_alloc {{
        aie.dma_bd(%k47 : memref<{total_plane_elems}xi32>, 0, {total_plane_elems}, {input_dims}) {{burst_length = 0 : i32}}
        aie.end
      }}
      %t2 = aiex.dma_configure_task_for @v03_alloc {{
        aie.dma_bd(%v03 : memref<{total_plane_elems}xi32>, 0, {total_plane_elems}, {input_dims}) {{burst_length = 0 : i32}}
        aie.end
      }}
      %t3 = aiex.dma_configure_task_for @v47_alloc {{
        aie.dma_bd(%v47 : memref<{total_plane_elems}xi32>, 0, {total_plane_elems}, {input_dims}) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%t0)
      aiex.dma_start_task(%t1)
      aiex.dma_start_task(%t2)
      aiex.dma_start_task(%t3)

      // 2 output descriptors: both reference %output but at different offsets
      %t4 = aiex.dma_configure_task_for @out0_alloc {{
        aie.dma_bd(%output : memref<256xi32>, 0, 128, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      %t5 = aiex.dma_configure_task_for @out1_alloc {{
        aie.dma_bd(%output : memref<256xi32>, 128, 128, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%t4)
      aiex.dma_start_task(%t5)
      aiex.dma_await_task(%t4)
      aiex.dma_await_task(%t5)
      aiex.dma_free_task(%t0)
      aiex.dma_free_task(%t1)
      aiex.dma_free_task(%t2)
      aiex.dma_free_task(%t3)
    }}
  }}
}}
"""


if __name__ == "__main__":
    import sys
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    print(generate_mlir(L))
