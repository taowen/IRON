"""
Experiment 08: FastFlowLM KV Scan Mechanism

Generates low-level MLIR-AIE with:
- Static BD ring in memtile with SPLIT: one 4096-dword KV tile → K half (2048) + V half (2048)
- 4-lock protocol for safe multi-consumer split (ping_rdy/pong_rdy/ping_done/pong_done)
- Core receives K on S2MM ch0, V on S2MM ch1
- Core computes simplified attention: out[h][d] += dot(q[h], k[t,h]) * v[t,h][d]
- Tail masking: only L valid tokens from ceil(L/16)*16 rounded tiles
- Runtime: ONE dma_configure_task_for per stream (no per-tile host tasks)
"""

from pathlib import Path
from math import ceil

NUM_HEADS = 4
HEAD_DIM = 32
TOKENS_PER_TILE = 16
KV_TILE_DWORDS = 4096
HALF_TILE_DWORDS = 2048


def _input_bd_dims(num_tiles: int) -> str:
    """Compute BD dimension array for KV input, respecting max-size-1023 limit."""
    total = num_tiles * KV_TILE_DWORDS
    if KV_TILE_DWORDS <= 1023:
        return (f"[<size = 1, stride = 0>, <size = 1, stride = 0>, "
                f"<size = {num_tiles}, stride = {KV_TILE_DWORDS}>, "
                f"<size = {KV_TILE_DWORDS}, stride = 1>]")
    else:
        inner = 512
        outer = KV_TILE_DWORDS // inner
        return (f"[<size = 1, stride = 0>, <size = {num_tiles}, stride = {KV_TILE_DWORDS}>, "
                f"<size = {outer}, stride = {inner}>, <size = {inner}, stride = 1>]")


def generate_mlir(L: int) -> str:
    """Generate MLIR for effective token length L."""
    num_tiles = ceil(L / TOKENS_PER_TILE)
    last_valid = L - (num_tiles - 1) * TOKENS_PER_TILE
    total_input_elems = num_tiles * KV_TILE_DWORDS
    input_dims = _input_bd_dims(num_tiles)
    experiment_dir = Path(__file__).parent.resolve()

    return f"""module {{
  aie.device(npu2) {{
    // Tiles
    %shim = aie.tile(0, 0)
    %mem  = aie.tile(0, 1)
    %core_tile = aie.tile(0, 2)

    // Memtile ping-pong buffers (full KV tile each)
    %buf_ping = aie.buffer(%mem) {{sym_name = "buf_ping"}} : memref<{KV_TILE_DWORDS}xi32>
    %buf_pong = aie.buffer(%mem) {{sym_name = "buf_pong"}} : memref<{KV_TILE_DWORDS}xi32>

    // Memtile locks: 4-lock protocol for safe multi-consumer split
    %ping_rdy  = aie.lock(%mem, 0) {{init = 2 : i32, sym_name = "ping_rdy"}}
    %pong_rdy  = aie.lock(%mem, 1) {{init = 2 : i32, sym_name = "pong_rdy"}}
    %ping_done = aie.lock(%mem, 2) {{init = 0 : i32, sym_name = "ping_done"}}
    %pong_done = aie.lock(%mem, 3) {{init = 0 : i32, sym_name = "pong_done"}}

    // Core tile K buffers (ping-pong, 2048 each = K half of KV tile)
    %k_ping = aie.buffer(%core_tile) {{sym_name = "k_ping"}} : memref<{HALF_TILE_DWORDS}xi32>
    %k_pong = aie.buffer(%core_tile) {{sym_name = "k_pong"}} : memref<{HALF_TILE_DWORDS}xi32>

    // Core tile V buffers (ping-pong, 2048 each = V half of KV tile)
    %v_ping = aie.buffer(%core_tile) {{sym_name = "v_ping"}} : memref<{HALF_TILE_DWORDS}xi32>
    %v_pong = aie.buffer(%core_tile) {{sym_name = "v_pong"}} : memref<{HALF_TILE_DWORDS}xi32>

    // Core tile locks for K stream
    %k_empty = aie.lock(%core_tile, 0) {{init = 2 : i32, sym_name = "k_empty"}}
    %k_full  = aie.lock(%core_tile, 1) {{init = 0 : i32, sym_name = "k_full"}}

    // Core tile locks for V stream
    %v_empty = aie.lock(%core_tile, 2) {{init = 2 : i32, sym_name = "v_empty"}}
    %v_full  = aie.lock(%core_tile, 3) {{init = 0 : i32, sym_name = "v_full"}}

    // Core tile output buffer + locks
    %out_buf = aie.buffer(%core_tile) {{sym_name = "out_buf"}} : memref<128xi32>
    %out_prod = aie.lock(%core_tile, 4) {{init = 1 : i32, sym_name = "out_prod"}}
    %out_cons = aie.lock(%core_tile, 5) {{init = 0 : i32, sym_name = "out_cons"}}

    // Flows
    aie.flow(%shim, DMA : 0, %mem, DMA : 0)        // shim → memtile (KV input)
    aie.flow(%mem, DMA : 0, %core_tile, DMA : 0)   // memtile → core (K half)
    aie.flow(%mem, DMA : 1, %core_tile, DMA : 1)   // memtile → core (V half)
    aie.flow(%core_tile, DMA : 0, %shim, DMA : 0)  // core → shim (output)

    // External kernel declarations
    func.func private @zero_output(memref<128xi32>) attributes {{link_with = "{experiment_dir}/kv_attention.o"}}
    func.func private @kv_attention(memref<{HALF_TILE_DWORDS}xi32>, memref<{HALF_TILE_DWORDS}xi32>, memref<128xi32>, i32, i32, i32) attributes {{link_with = "{experiment_dir}/kv_attention.o"}}

    // Core: loop num_tiles times, acquire K+V, compute attention, release
    %core_0_2 = aie.core(%core_tile) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_i32 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      %num_tiles_idx = arith.constant {num_tiles} : index
      %num_tiles_i32 = arith.constant {num_tiles} : i32
      %last_valid_i32 = arith.constant {last_valid} : i32

      // Acquire output buffer and zero it
      aie.use_lock(%out_prod, AcquireGreaterEqual, 1)
      func.call @zero_output(%out_buf) : (memref<128xi32>) -> ()

      // Process tiles
      scf.for %i = %c0 to %num_tiles_idx step %c1 {{
        %i_i32 = arith.index_cast %i : index to i32
        %rem = arith.remsi %i_i32, %c2_i32 : i32
        %is_pong = arith.cmpi eq, %rem, %c1_i32 : i32

        // Acquire K and V data
        aie.use_lock(%k_full, AcquireGreaterEqual, 1)
        aie.use_lock(%v_full, AcquireGreaterEqual, 1)

        // Call kernel with ping or pong buffers
        scf.if %is_pong {{
          func.call @kv_attention(%k_pong, %v_pong, %out_buf, %i_i32, %num_tiles_i32, %last_valid_i32) : (memref<{HALF_TILE_DWORDS}xi32>, memref<{HALF_TILE_DWORDS}xi32>, memref<128xi32>, i32, i32, i32) -> ()
        }} else {{
          func.call @kv_attention(%k_ping, %v_ping, %out_buf, %i_i32, %num_tiles_i32, %last_valid_i32) : (memref<{HALF_TILE_DWORDS}xi32>, memref<{HALF_TILE_DWORDS}xi32>, memref<128xi32>, i32, i32, i32) -> ()
        }}

        // Release K and V buffers
        aie.use_lock(%k_empty, Release, 1)
        aie.use_lock(%v_empty, Release, 1)
      }}

      // Signal output ready
      aie.use_lock(%out_cons, Release, 1)
      aie.end
    }}

    // Memtile DMA: S2MM ch0 receives full KV tiles (ring),
    // MM2S ch0 sends K half (offset=0), MM2S ch1 sends V half (offset=2048)
    %memtile_dma_0_1 = aie.memtile_dma(%mem) {{
      // S2MM ch0: receive full KV tile into ping-pong ring
      %0 = aie.dma_start(S2MM, 0, ^s2mm_ping, ^mm2s_k_start)
    ^s2mm_ping:
      aie.use_lock(%ping_rdy, AcquireGreaterEqual, 2)
      aie.dma_bd(%buf_ping : memref<{KV_TILE_DWORDS}xi32>, 0, {KV_TILE_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%ping_done, Release, 2)
      aie.next_bd ^s2mm_pong
    ^s2mm_pong:
      aie.use_lock(%pong_rdy, AcquireGreaterEqual, 2)
      aie.dma_bd(%buf_pong : memref<{KV_TILE_DWORDS}xi32>, 0, {KV_TILE_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%pong_done, Release, 2)
      aie.next_bd ^s2mm_ping

      // MM2S ch0: send K half (first 2048 dwords) to core
    ^mm2s_k_start:
      %1 = aie.dma_start(MM2S, 0, ^mm2s_k_ping, ^mm2s_v_start)
    ^mm2s_k_ping:
      aie.use_lock(%ping_done, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf_ping : memref<{KV_TILE_DWORDS}xi32>, 0, {HALF_TILE_DWORDS}) {{bd_id = 2 : i32, next_bd_id = 3 : i32}}
      aie.use_lock(%ping_rdy, Release, 1)
      aie.next_bd ^mm2s_k_pong
    ^mm2s_k_pong:
      aie.use_lock(%pong_done, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf_pong : memref<{KV_TILE_DWORDS}xi32>, 0, {HALF_TILE_DWORDS}) {{bd_id = 3 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%pong_rdy, Release, 1)
      aie.next_bd ^mm2s_k_ping

      // MM2S ch1: send V half (second 2048 dwords, offset=2048) to core
    ^mm2s_v_start:
      %2 = aie.dma_start(MM2S, 1, ^mm2s_v_ping, ^end)
    ^mm2s_v_ping:
      aie.use_lock(%ping_done, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf_ping : memref<{KV_TILE_DWORDS}xi32>, {HALF_TILE_DWORDS}, {HALF_TILE_DWORDS}) {{bd_id = 24 : i32, next_bd_id = 25 : i32}}
      aie.use_lock(%ping_rdy, Release, 1)
      aie.next_bd ^mm2s_v_pong
    ^mm2s_v_pong:
      aie.use_lock(%pong_done, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf_pong : memref<{KV_TILE_DWORDS}xi32>, {HALF_TILE_DWORDS}, {HALF_TILE_DWORDS}) {{bd_id = 25 : i32, next_bd_id = 24 : i32}}
      aie.use_lock(%pong_rdy, Release, 1)
      aie.next_bd ^mm2s_v_ping
    ^end:
      aie.end
    }}

    // Core tile DMA: S2MM ch0 (K), S2MM ch1 (V), MM2S ch0 (output)
    %mem_0_2 = aie.mem(%core_tile) {{
      // S2MM ch0: receive K half from memtile
      %0 = aie.dma_start(S2MM, 0, ^k_s2mm_ping, ^v_s2mm_start)
    ^k_s2mm_ping:
      aie.use_lock(%k_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%k_ping : memref<{HALF_TILE_DWORDS}xi32>, 0, {HALF_TILE_DWORDS}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%k_full, Release, 1)
      aie.next_bd ^k_s2mm_pong
    ^k_s2mm_pong:
      aie.use_lock(%k_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%k_pong : memref<{HALF_TILE_DWORDS}xi32>, 0, {HALF_TILE_DWORDS}) {{bd_id = 1 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%k_full, Release, 1)
      aie.next_bd ^k_s2mm_ping

      // S2MM ch1: receive V half from memtile
    ^v_s2mm_start:
      %1 = aie.dma_start(S2MM, 1, ^v_s2mm_ping, ^out_mm2s_start)
    ^v_s2mm_ping:
      aie.use_lock(%v_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_ping : memref<{HALF_TILE_DWORDS}xi32>, 0, {HALF_TILE_DWORDS}) {{bd_id = 2 : i32, next_bd_id = 3 : i32}}
      aie.use_lock(%v_full, Release, 1)
      aie.next_bd ^v_s2mm_pong
    ^v_s2mm_pong:
      aie.use_lock(%v_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%v_pong : memref<{HALF_TILE_DWORDS}xi32>, 0, {HALF_TILE_DWORDS}) {{bd_id = 3 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%v_full, Release, 1)
      aie.next_bd ^v_s2mm_ping

      // MM2S ch0: send output to shim
    ^out_mm2s_start:
      %2 = aie.dma_start(MM2S, 0, ^out_bd, ^end)
    ^out_bd:
      aie.use_lock(%out_cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_buf : memref<128xi32>, 0, 128) {{bd_id = 4 : i32}}
      aie.use_lock(%out_prod, Release, 1)
      aie.next_bd ^out_bd
    ^end:
      aie.end
    }}

    // Shim DMA allocations
    aie.shim_dma_allocation @kv_alloc(%shim, MM2S, 0)
    aie.shim_dma_allocation @output_alloc(%shim, S2MM, 0)

    // Runtime sequence: ONE configure per stream, no per-tile tasks
    aie.runtime_sequence(%kv_input: memref<{total_input_elems}xi32>, %output: memref<128xi32>) {{
      %0 = aiex.dma_configure_task_for @kv_alloc {{
        aie.dma_bd(%kv_input : memref<{total_input_elems}xi32>, 0, {total_input_elems}, {input_dims}) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @output_alloc {{
        aie.dma_bd(%output : memref<128xi32>, 0, 128, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 128, stride = 1>]) {{burst_length = 0 : i32}}
        aie.end
      }} {{issue_token = true}}
      aiex.dma_start_task(%1)
      aiex.dma_await_task(%1)
      aiex.dma_free_task(%0)
    }}
  }}
}}
"""


if __name__ == "__main__":
    import sys
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 17
    print(generate_mlir(L))
