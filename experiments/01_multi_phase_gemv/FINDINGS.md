# Experiment 01: Multi-Phase GEMV — Findings

## Result: IRON CAN express multi-phase workers with on-chip input reuse

The design compiled successfully to valid MLIR. The generated code shows:

### What works

1. **Input held across phases**: `X_0` is `acquire`d once (line 21) and `release`d only after
   both Q and K phases complete (line 55). The input vector stays in core local memory.

2. **Same weight FIFO serves both phases**: `W_0` is `acquire/release`d 16 times for Q,
   then 16 more times for K — all from the same ObjectFifo.

3. **Sequential DMA fills to same FIFO**: The runtime sequence issues two separate
   `dma_configure_task_for @W_0` (lines 113-122, 118-122) with different offsets
   (0 for Q weights, 16384 for K weights). Both are started immediately.

4. **Separate output FIFOs**: Q and K outputs go to different ObjectFifos (`Cq_0`, `Ck_0`)
   draining to different DDR buffers.

### Core structure verified

```
core_0_2:
  for infinite_loop:
    x = acquire(X_0)          // load input ONCE

    cq = acquire(Cq_0)       // Phase 1: Q projection
    for 16 iterations:
      w = acquire(W_0)
      matvec(w, x, cq)       // x reused
      release(W_0)
    release(Cq_0)

    ck = acquire(Ck_0)       // Phase 2: K projection
    for 16 iterations:
      w = acquire(W_0)       // SAME fifo, different data
      matvec(w, x, ck)       // x STILL held
      release(W_0)
    release(Ck_0)

    release(X_0)             // release input only after both phases
```

## Gap analysis vs Fused Layer Engine

| Aspect | This experiment proves | Fused Layer Engine needs | Gap? |
|--------|----------------------|--------------------------|------|
| Multi-phase worker | Yes, single Worker does Q then K | Q/K/V/O/up/gate/down sequentially | **No gap** — just more phases |
| On-chip input reuse | Yes, x held across phases | hidden_norm replayed for Q/K/V | **No gap** at core level |
| Sequential weight stream | Yes, same FIFO carries different weights | 608 patches through same shim BD | **No gap** in principle |
| DMA double-buffer | depth=2 on W FIFO | even/odd BD bank | **No gap** — same mechanism |

## What this does NOT test (remaining gaps)

1. **Memtile replay to multiple cores**: This experiment broadcasts x to each core independently.
   The fused engine uses a single memtile buffer that replays to 4 cores in the same column.
   Question: can ObjectFifo with multiple consumers + replay achieve this?

2. **7 phases with different output dimensions**: Q(4096), K(1024), V(1024), O(4096),
   up(12288), gate(12288), down(4096). This experiment only tested M_q == M_k.
   A single kernel signature can't serve all; need a mechanism for phase-dependent output sizing.

3. **Intermediate routing without DDR**: After Q/K/V projection, RoPE/attention must consume
   results WITHOUT going to DDR. This experiment's outputs drain to DDR (arg2, arg3).
   Need: output of phase 1 feeds into a different worker (e.g., RoPE tile) via stream/memtile.

4. **Online Q4 weight format**: Weight FIFO carries BF16 tiles. Fused engine uses INT4 packed
   chunks with inline dequant.

5. **Heterogeneous tile roles**: All cores run the same kernel here. Fused engine has
   16 main tiles + edge/aux tiles doing different work in the same graph.

## Conclusion

**IRON's ObjectFifo+Worker model CAN express the fundamental multi-phase projection pattern.**
The primary barriers to implementing a fused layer engine are at a higher level:

- The **temporal fusion** system (`FusedMLIROperator`) wraps each operator as a separate
  `aie.device` and calls them via `ConfigureOp/RunOp`, forcing DDR synchronization.
  But the underlying MLIR primitives (ObjectFifo, Worker, Runtime) can express the fused
  pattern directly — it just requires writing a single design that covers the whole layer.

- The real engineering challenges are: (a) routing intermediates between heterogeneous tiles
  without DDR, (b) supporting Q4 packed weight types, and (c) managing the complexity of
  a 7-projection + attention + FFN design in a single graph.
