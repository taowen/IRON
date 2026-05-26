"""
Experiment 07a: Memtile BD Ring Demo

Generates low-level MLIR-AIE with:
- Static BD ring in memtile (ping-pong, bd0→bd1→bd0)
- Runtime: dma_configure_task_for (single descriptor, NOT per-chunk)
- Core loops N chunks, outputs per-chunk checksum to verify ordering
- No per-chunk rt.fill — ring auto-repeats

Data path: Host DDR → Shim DMA → MemTile (ring) → Core tile
Output path: Core tile → Shim DMA → Host DDR

NOTE: This is 07a. Runtime still uses dma_configure_task_for/dma_start_task.
07b will replace with npu.writebd/address_patch/push_queue/sync.
"""

from pathlib import Path

CHUNK_ELEMS_SMOKE = 256   # int32 = 1024 bytes (fast smoke test)
CHUNK_ELEMS_REAL = 4096   # int32 = 0x4000 bytes (FastFlowLM KV plane tile size)


def _input_bd_dims(num_chunks: int, chunk_elems: int) -> str:
    """Compute BD dimension array for input descriptor, respecting max-size-1023 limit."""
    if chunk_elems <= 1023:
        return (f"[<size = 1, stride = 0>, <size = 1, stride = 0>, "
                f"<size = {num_chunks}, stride = {chunk_elems}>, "
                f"<size = {chunk_elems}, stride = 1>]")
    else:
        inner = 512
        outer = chunk_elems // inner
        return (f"[<size = 1, stride = 0>, <size = {num_chunks}, stride = {chunk_elems}>, "
                f"<size = {outer}, stride = {inner}>, <size = {inner}, stride = 1>]")


def generate_mlir(num_chunks: int, chunk_elems: int = CHUNK_ELEMS_SMOKE) -> str:
    total_input_elems = num_chunks * chunk_elems
    experiment_dir = Path(__file__).parent.resolve()
    input_dims = _input_bd_dims(num_chunks, chunk_elems)

    return f"""module {{
  aie.device(npu2) {{
    // Tiles
    %shim = aie.tile(0, 0)
    %mem  = aie.tile(0, 1)
    %core_tile = aie.tile(0, 2)

    // Memtile ping-pong buffers for input stream
    %buf_ping = aie.buffer(%mem) {{sym_name = "buf_ping"}} : memref<{chunk_elems}xi32>
    %buf_pong = aie.buffer(%mem) {{sym_name = "buf_pong"}} : memref<{chunk_elems}xi32>

    // Memtile locks for input ring: empty=2 (both free initially), full=0
    %mt_empty = aie.lock(%mem, 0) {{init = 2 : i32, sym_name = "mt_empty"}}
    %mt_full  = aie.lock(%mem, 1) {{init = 0 : i32, sym_name = "mt_full"}}

    // Core tile ping-pong buffers (receiving from memtile)
    %core_ping = aie.buffer(%core_tile) {{sym_name = "core_ping"}} : memref<{chunk_elems}xi32>
    %core_pong = aie.buffer(%core_tile) {{sym_name = "core_pong"}} : memref<{chunk_elems}xi32>

    // Core tile locks for input: empty=2, full=0
    %core_empty = aie.lock(%core_tile, 0) {{init = 2 : i32, sym_name = "core_empty"}}
    %core_full  = aie.lock(%core_tile, 1) {{init = 0 : i32, sym_name = "core_full"}}

    // Core tile output buffer (per-chunk checksums) + locks
    %out_buf = aie.buffer(%core_tile) {{sym_name = "out_buf"}} : memref<{num_chunks}xi32>
    %out_prod = aie.lock(%core_tile, 2) {{init = 1 : i32, sym_name = "out_prod"}}
    %out_cons = aie.lock(%core_tile, 3) {{init = 0 : i32, sym_name = "out_cons"}}

    // Flows: shim→memtile, memtile→core, core→shim
    aie.flow(%shim, DMA : 0, %mem, DMA : 0)
    aie.flow(%mem, DMA : 0, %core_tile, DMA : 0)
    aie.flow(%core_tile, DMA : 0, %shim, DMA : 0)

    // External kernel declaration
    func.func private @checksum_chunk(memref<{chunk_elems}xi32>, memref<{num_chunks}xi32>, i32, i32) attributes {{link_with = "{experiment_dir}/checksum.o"}}

    // Core: loop num_chunks times, acquire chunk, compute per-chunk checksum, release
    %core_0_2 = aie.core(%core_tile) {{
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c1_i32 = arith.constant 1 : i32
      %c2_i32 = arith.constant 2 : i32
      %c{chunk_elems}_i32 = arith.constant {chunk_elems} : i32
      %num_chunks = arith.constant {num_chunks} : index

      // Acquire output buffer
      aie.use_lock(%out_prod, AcquireGreaterEqual, 1)

      // Process chunks using ping-pong
      scf.for %i = %c0 to %num_chunks step %c1 {{
        // Determine ping or pong
        %i_i32 = arith.index_cast %i : index to i32
        %rem = arith.remsi %i_i32, %c2_i32 : i32
        %is_pong = arith.cmpi eq, %rem, %c1_i32 : i32

        // Acquire full (data ready)
        aie.use_lock(%core_full, AcquireGreaterEqual, 1)

        // Select buffer and compute checksum into out_buf[i]
        scf.if %is_pong {{
          func.call @checksum_chunk(%core_pong, %out_buf, %i_i32, %c{chunk_elems}_i32) : (memref<{chunk_elems}xi32>, memref<{num_chunks}xi32>, i32, i32) -> ()
        }} else {{
          func.call @checksum_chunk(%core_ping, %out_buf, %i_i32, %c{chunk_elems}_i32) : (memref<{chunk_elems}xi32>, memref<{num_chunks}xi32>, i32, i32) -> ()
        }}

        // Release empty (buffer free for reuse)
        aie.use_lock(%core_empty, Release, 1)
      }}

      // Signal output ready
      aie.use_lock(%out_cons, Release, 1)
      aie.end
    }}

    // Memtile DMA: S2MM channel 0 (shim→memtile) with BD ring
    // and MM2S channel 0 (memtile→core) with BD ring
    %memtile_dma_0_1 = aie.memtile_dma(%mem) {{
      // S2MM ch0: receive from shim into ping-pong ring
      %0 = aie.dma_start(S2MM, 0, ^s2mm_ping, ^mm2s_start)
    ^s2mm_ping:
      aie.use_lock(%mt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf_ping : memref<{chunk_elems}xi32>, 0, {chunk_elems}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%mt_full, Release, 1)
      aie.next_bd ^s2mm_pong
    ^s2mm_pong:
      aie.use_lock(%mt_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf_pong : memref<{chunk_elems}xi32>, 0, {chunk_elems}) {{bd_id = 1 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%mt_full, Release, 1)
      aie.next_bd ^s2mm_ping

      // MM2S ch0: send from memtile to core, same buffers/locks
    ^mm2s_start:
      %1 = aie.dma_start(MM2S, 0, ^mm2s_ping, ^end)
    ^mm2s_ping:
      aie.use_lock(%mt_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf_ping : memref<{chunk_elems}xi32>, 0, {chunk_elems}) {{bd_id = 2 : i32, next_bd_id = 3 : i32}}
      aie.use_lock(%mt_empty, Release, 1)
      aie.next_bd ^mm2s_pong
    ^mm2s_pong:
      aie.use_lock(%mt_full, AcquireGreaterEqual, 1)
      aie.dma_bd(%buf_pong : memref<{chunk_elems}xi32>, 0, {chunk_elems}) {{bd_id = 3 : i32, next_bd_id = 2 : i32}}
      aie.use_lock(%mt_empty, Release, 1)
      aie.next_bd ^mm2s_ping
    ^end:
      aie.end
    }}

    // Core tile DMA: S2MM ch0 (memtile→core) with BD ring
    // and MM2S ch0 (core→shim) for output
    %mem_0_2 = aie.mem(%core_tile) {{
      %0 = aie.dma_start(S2MM, 0, ^s2mm_ping, ^mm2s_start)
    ^s2mm_ping:
      aie.use_lock(%core_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%core_ping : memref<{chunk_elems}xi32>, 0, {chunk_elems}) {{bd_id = 0 : i32, next_bd_id = 1 : i32}}
      aie.use_lock(%core_full, Release, 1)
      aie.next_bd ^s2mm_pong
    ^s2mm_pong:
      aie.use_lock(%core_empty, AcquireGreaterEqual, 1)
      aie.dma_bd(%core_pong : memref<{chunk_elems}xi32>, 0, {chunk_elems}) {{bd_id = 1 : i32, next_bd_id = 0 : i32}}
      aie.use_lock(%core_full, Release, 1)
      aie.next_bd ^s2mm_ping
    ^mm2s_start:
      %1 = aie.dma_start(MM2S, 0, ^out_bd, ^end)
    ^out_bd:
      aie.use_lock(%out_cons, AcquireGreaterEqual, 1)
      aie.dma_bd(%out_buf : memref<{num_chunks}xi32>, 0, {num_chunks}) {{bd_id = 2 : i32}}
      aie.use_lock(%out_prod, Release, 1)
      aie.next_bd ^out_bd
    ^end:
      aie.end
    }}

    // Shim DMA allocations (named for runtime_sequence reference)
    aie.shim_dma_allocation @input_alloc(%shim, MM2S, 0)
    aie.shim_dma_allocation @output_alloc(%shim, S2MM, 0)

    // Runtime sequence: ONE configure per stream, no per-chunk tasks
    aie.runtime_sequence(%input: memref<{total_input_elems}xi32>, %output: memref<{num_chunks}xi32>) {{
      %0 = aiex.dma_configure_task_for @input_alloc {{
        aie.dma_bd(%input : memref<{total_input_elems}xi32>, 0, {total_input_elems}, {input_dims}) {{burst_length = 0 : i32}}
        aie.end
      }}
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @output_alloc {{
        aie.dma_bd(%output : memref<{num_chunks}xi32>, 0, {num_chunks}, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = {num_chunks}, stride = 1>]) {{burst_length = 0 : i32}}
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
    num_chunks = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    chunk_elems = int(sys.argv[2]) if len(sys.argv) > 2 else CHUNK_ELEMS_SMOKE
    print(generate_mlir(num_chunks, chunk_elems))
