# XDNA Programming Guide For The Fused-Layer Experiments

This guide explains the programming model behind the retained experiments in
this directory. It is written for readers who know Python/C++ but are not yet
comfortable reading raw MLIR-AIE, XDNA tile coordinates, DMA BDs, locks, packet
routes, and row1 memtile rings.

The short version: these experiments are not normal "call operator A, then call
operator B" code. They are attempts to build a long-lived dataflow engine for
one Qwen3 decode layer. The engine should keep the same physical AIE tiles
alive across Q, K, V, attention, O, gate, up, SwiGLU, and down phases. That is
why the code talks directly about tiles, DMA descriptors, locks, stream routes,
and phase records.

## Reading Order

Start here before reading individual experiments:

1. Read "Hardware Model" and "Runtime Model" below.
2. Read `experiments/README.md` for the retained experiment chain.
3. Read `experiments/resource_plan_audit.py` to see the computed resource plan.
4. Read the latest frontier experiments:
   - `59_mylm_exact_nblock_projection`
   - `60_mylm_chunk_ring_projection`
5. Only then read earlier retained experiments for the specific subproblem they
   isolate.

## Hardware Model

The AMD XDNA NPU is an array of AIE tiles. In these experiments we use an
8-column partition with this practical shape:

```text
row 0: shim tiles      host DDR/BO <-> AIE array DMA boundary
row 1: memtiles        larger on-array SRAM and DMA routing layer
row 2: compute tiles   AIE cores + local memory + local DMA
row 3: compute tiles
row 4: compute tiles
row 5: compute tiles
```

Tile coordinates are written as `c{column}r{row}`. For example, `c2r4` means
column 2, row 4.

Important physical facts:

- Shim row (`r0`) is where host BO data enters and leaves the AIE array.
- Memtile row (`r1`) is the right place to reshape, split, fan out, and
  ping-pong larger streams before compute tiles consume them.
- Compute rows (`r2..r5`) run C/C++ kernels compiled for AIE cores.
- Compute tiles have limited local memory and limited DMA input/output
  channels.
- Memtiles have more storage than compute tiles, but still not enough to treat
  large model patches as ordinary tensors.
- Stream routes and DMA channels are real hardware resources. A design can be
  mathematically correct and still fail because two logical streams want the
  same physical input channel or route.

## Memory Spaces

The code uses three practical memory levels:

```text
host DDR / XRT BO
  -> shim DMA
  -> row1 memtile buffers
  -> compute-tile local buffers
  -> core registers / local accumulators
```

Host-visible buffers are XRT BOs. They are large, but moving through them
between phases is slow and defeats the purpose of a fused layer engine.

Memtile buffers are useful for row1 staging, splitting, and ping-pong rings.
They are not a replacement for DDR. A full Qwen hidden-dim patch is `0x28000`
bytes; keeping many such patches resident in row1 quickly becomes the wrong
shape.

Compute-tile local buffers should hold only small working sets:

- one or two Q4NX `32 x 256` chunks,
- one `256 bf16` activation slice,
- fp32 accumulators for 32 output rows,
- small sideband/debug records.

The current resource audit computes the main Q4 local working set as about
`10944` bytes per main tile, which is small enough for a compute tile.

## DMA, BD, And Lock Basics

Most experiments are built around explicit DMA buffer descriptors, usually
called BDs.

A BD says:

- which local buffer to read or write,
- offset and length,
- optional multidimensional stride/wrap pattern,
- optional next BD,
- optional packet ID,
- which lock to acquire before running,
- which lock to release after running.

The canonical ping-pong pattern is:

```text
producer fills buffer A -> releases A_full
consumer acquires A_full -> consumes buffer A -> releases A_empty

producer fills buffer B -> releases B_full
consumer acquires B_full -> consumes buffer B -> releases B_empty
```

In MLIR-AIE this shows up as:

```mlir
aie.use_lock(%buf_empty, AcquireGreaterEqual, 1)
aie.dma_bd(%buf : memref<...>, 0, LEN) {bd_id = 0 : i32, next_bd_id = 1 : i32}
aie.use_lock(%buf_full, Release, 1)
aie.next_bd ^next
```

Locks are not optional bookkeeping. They are the synchronization mechanism
between DMA engines and workers. Most hardware timeouts in these experiments
come from one of these mistakes:

- producer releases a lock the consumer never acquires,
- consumer waits on a lock the producer never releases,
- a BD ring advances to a different buffer than the core expects,
- a channel uses a BD slot that belongs to a different channel bank,
- two producers try to feed the same worker input without one physical phase
  schedule.

## Runtime Model

There are two very different stages:

### Static Configuration

The AIE program, stream routes, static BDs, locks, and initial tile state are
compiled into the xclbin/PDI/CDO. MyLM relies heavily on this. Its layer engine
is not a host-side chain of normal kernels; it is a preconfigured dataflow
machine.

### Per-Run Runtime Sequence

At runtime, the host should mostly patch addresses/lengths and start the
hardware queues. The raw MLIR-AIE experiments use operations like:

```mlir
aiex.npu.writebd(...)
aiex.npu.address_patch(...)
aiex.npu.push_queue(...)
aiex.npu.sync(...)
```

The important distinction:

- Static BD rings describe how data moves once the stream starts.
- Runtime BD patching tells the ring which host BO address and length to use
  for this run.
- The host should not push every small chunk separately in the final design.

This is the reason experiments moved from many high-level `rt.fill(...)` calls
toward linked BDs and row1 rings. Repeated runtime tasks work for small probes,
but they do not match a high-performance layer engine.

## High-Level IRON Versus Raw MLIR-AIE

High-level IRON primitives are useful:

- `ObjectFifo` expresses producer/consumer queues.
- `Worker` expresses a core program.
- `Runtime` expresses host sequence.
- `.forward()`, `.split()`, and `.join()` can express memtile fanout/gather in
  simple cases.

But MyLM-style fused layers need lower-level ownership in some places:

- exact stream switch routing,
- exact BD slot selection,
- linked shim descriptors,
- phase-specific lock ordering,
- row1 chunk rings,
- same physical input channel reused across phases.

So the experiments use a mixed style:

- high-level ideas to design the dataflow,
- raw generated MLIR-AIE when physical resource ownership matters.

Do not read this as accidental complexity. It is the boundary where the
hardware programming model becomes visible.

## The Main16 / Edge16 Split

The current full-layer resource plan uses all 32 compute tiles:

```text
edge/aux fabric: c0,c1,c6,c7 rows 2..5  -> 16 tiles
main fabric:     c2,c3,c4,c5 rows 2..5  -> 16 tiles
```

The main16 fabric performs Q4NX projection work. It is time-reused across:

```text
Q -> K -> V -> O -> up -> gate -> down
```

This is the key insight. We do not allocate separate tiles for each projection.
The same physical projection workers consume different phase streams over time.

The edge/aux fabric handles:

- current-token K/V path,
- KV-cache scan,
- attention-side reshaping/replay,
- compact sideband or phase records,
- returning activation slices to main16 for O and later phases.

## Main Tile ABI

The main projection tile ABI used by the latest experiments is:

```text
input channel 0: activation slice
  128 dwords = 256 bf16

input channel 1: Q4NX weight chunk
  1280 dwords = 5120 bytes

output / sideband:
  compact records, often 17 dwords in debug contracts
```

One main tile logically computes 32 output rows. For a full hidden-dim
projection:

```text
K = 4096
K chunk = 256
chunks per tile phase = 16
output rows per tile = 32
```

Each chunk pairs:

```text
256 bf16 activation values
32 x 256 Q4NX weight chunk
fp32 accumulation over the tile's 32 output rows
```

Only the final accumulated output should be converted/drained or handed off.

## Q4NX Weight Format

The experiments follow the MyLM-observed Q4NX chunk shape:

```text
32 output rows x 256 input columns
group size = 32
chunk size = 5120 bytes
```

Within one chunk:

```text
scales:      32 rows x 8 groups x bf16 = 512 bytes
zero points: 32 rows x 8 groups x bf16 = 512 bytes
int4 data:   32 rows x 256 cols / 2    = 4096 bytes
total:                                      5120 bytes
```

The dequant formula is:

```text
weight = (q4 - zero_point) * scale
```

The final kernel must dequantize online and immediately MAC into accumulators.
Writing an intermediate bf16 weight tile back to DDR would lose the main
decode-speed benefit.

## Real Qwen3 Patch Schedule

MyLM-sized patches are not single Q4NX chunks. A patch is `64 output rows x
full K`.

For hidden-dim phases:

```text
64 rows x 4096 K = 0x28000 bytes
```

For down:

```text
64 rows x 12288 K = 0x78000 bytes
```

The full Qwen3 layer patch schedule is:

```text
Q    64 patches
K    16 patches
V    16 patches
O    64 patches
up   192 patches
gate 192 patches
down 64 patches
----------------
total 608 patches
```

The latest experiments prove the manifest and exact patch ordering. The next
hard part is consuming each large patch through small row1 chunk rings instead
of storing a full patch in row1.

## Row1 Memtile Responsibilities

Row1 is not just a passive buffer. In the MyLM-style design it should:

- receive coarse host patch descriptors from shim,
- split a `64-row` patch into two `32-row` streams,
- split each row stream into `5120B` Q4NX chunks,
- provide ping-pong buffering so compute can overlap with DMA,
- reshape KV-cache history tiles before attention workers consume them,
- avoid full-patch residency when possible.

Experiment `59_mylm_exact_nblock_projection` proves exact full-patch ABI with
full-patch row1 residency. Experiment `60_mylm_chunk_ring_projection` is the
frontier because it tries to replace that with a small chunk ring.

## KV Cache Shape

The KV experiments use rounded history scans. Context length `L` is rounded up
to 16-token tiles for DMA:

```text
history bytes per plane = ceil(L / 16) * 0x4000
current token offset    = (L - 1) * 0x400 within a plane
```

A 16-token, 4-head, 128-dim, bf16 plane tile is:

```text
16 tokens x 4 heads x 128 dim x 2 bytes = 0x4000 bytes
0x4000 bytes / 4 = 4096 dwords
```

The experiments often split this into two `2048 dword` half-plane streams.

Tail tokens past the true `L` are read by DMA but must be ignored by attention
math. This is normal for static tiled scans.

## Packet Routes And Sideband Records

Some experiments use packet IDs such as `14` and `15`. A packet route is a
stream-switch-level route, not a Python-level queue. Packet IDs must be unique
where routes share hardware, or different streams can collide and deadlock.

The `17 dword` sideband in the retained experiments is mostly a contract and
debug device:

```text
1 dword header
16 dword payload
```

It proves that compact phase state can move between edge and main tiles without
host-visible buffers. It is not yet the proven exact MyLM attention state. The
remaining production question is what this compact state should contain for
real attention: max/sum, score fragments, value accumulator metadata, phase
control, or some combination.

## Phase Reuse

The full-layer engine should not look like this:

```text
dispatch Q operator
dispatch K operator
dispatch V operator
dispatch attention operator
dispatch O operator
dispatch FFN operator
```

It should look closer to this:

```text
start one layer engine
  main16 phase Q consumes Q weights
  main16 phase K consumes K weights
  main16 phase V consumes V weights
  edge attention consumes current/cache streams
  main16 phase O consumes attention-replayed activation slices
  main16 phase up/gate/down consumes FFN phase streams
finish layer engine
```

The same physical main16 workers stay alive. They do not get reallocated per
projection. This is how a full layer can fit.

## How To Read An Experiment

Most retained experiments have the same file roles:

- `README.md`: what the experiment proves and does not prove.
- `generate.py`: emits raw MLIR-AIE. This is where tile placement, BDs, locks,
  flows, packet routes, and runtime patching are defined.
- `*.cc`: AIE core kernels. In many experiments these are deterministic
  contract kernels, not performance kernels.
- `reference.py`: CPU reference for the dataflow contract.
- `run_npu.py`: compiles, runs on NPU, and compares output to the reference.

When reading `generate.py`, look for these first:

1. Tile constants: which columns/rows are active?
2. Buffer sizes: are they full-patch, chunk-sized, KV-tile-sized, or records?
3. Lock pairs: who produces and who consumes each buffer?
4. `aie.dma_bd`: what does each DMA transfer, and does it chain?
5. `aie.flow` / packet flow: which tile output reaches which tile input?
6. Runtime sequence: does the host patch coarse descriptors or push each small
   chunk?

## Debugging Timeouts

An XRT timeout usually means a lock or DMA queue did not complete. Common
causes:

- A producer and consumer disagree on ping-pong parity.
- A BD has `next_bd_id` pointing to a descriptor that is never valid.
- A DMA queue is pushed on the wrong channel.
- A lock is initialized with the wrong count.
- Multiple cores consume a shared lock but the producer releases too few tokens.
- Packet IDs collide.
- A stream route is impossible, so the generated design is not the design you
  thought it was.
- A debug drain assumes host packet order that the hardware does not guarantee.

Use these debugging techniques:

- Reduce to one column or one row first.
- Use deterministic payloads with `(phase, group, row, chunk)` headers.
- Drain debug records with explicit headers, then reorder on host.
- Check generated MLIR for `address_patch`, `writebd`, `push_queue`, and
  `aie.dma_bd` counts.
- Prefer one new hardware idea per experiment.
- Run `experiments/resource_plan_audit.py` after pruning or changing retained
  milestones.

## Resource Limits In Plain Language

The important limits are not abstract compiler quirks. They correspond to real
hardware:

- Too many `rt.fill` calls means too many host-managed DMA tasks.
- Too many ObjectFifos can consume more DMA channels than a tile has.
- Full-patch row1 buffers consume too much memtile memory and hide the real
  streaming problem.
- Multiple logical producers for one worker input channel require an explicit
  phase schedule. The compiler will not infer that they are time-separated.
- Keeping gate/up/O/attention intermediates in DDR is correct but too slow.
- A design that passes for toy `K=256` can fail for `K=4096` because descriptor
  count, lock lifetime, and row1 residency scale differently.

## What The Retained Experiments Establish

The current retained chain establishes:

- edge KV row1 rings can run for rounded context lengths,
- current K/V can be produced and written into cache,
- full `32Q/8KV` attention resource mapping fits,
- main16 can be reused for full-K Q4NX phase replay,
- edge-to-main O replay can feed full-K projection slices,
- a seven-phase full-layer skeleton can run with deterministic records,
- the real Qwen3 608-patch schedule is known,
- linked descriptors can reduce runtime task count,
- exact MyLM patch units and patch order are known,
- full 512-row projection can consume exact MyLM-sized patches,
- chunk-sized row1 residency is the current unsolved frontier.

## What Is Still Not Proven

The remaining full-layer work is:

1. Make the small row1 Q4NX chunk ring run reliably for a full `0x28000` patch.
2. Replace deterministic phase records with real attention state.
3. Implement real Q/K/V current write, KV scan, online softmax, and weighted V
   in the edge/aux fabric.
4. Return attention output to main16 O phase without debug drains.
5. Implement real RMSNorm, Q/K norm, RoPE, residual, SwiGLU, and down flow.
6. Replace contract kernels with high-performance online Q4NX kernels.
7. Decide where direct CDO generation is necessary instead of relying on
   high-level routing/pathfinding.
8. Wrap the final layer engine in layer-to-layer runtime submission and lm_head.

## Practical Rule For New Experiments

A new experiment should answer one physical question:

- Does this exact BD/lock/route schedule complete?
- Does this exact handoff ABI preserve values?
- Does this exact buffer size fit?
- Does this exact phase schedule avoid extra host dispatch?

Avoid experiments that only prove a mathematical operation in isolation. The
hard part is not the math; it is making the math live in the right physical
dataflow shape.
