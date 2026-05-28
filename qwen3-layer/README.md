# qwen3-layer

This directory implements the qwen3-layer contract described in
`experiments/qwen3-dataflow.md` and contains a runnable NPU integration
backend.

The current implementation is the qwen3-dataflow physical skeleton:

- `c1r2` full-vector station with `2049`-dword packet0 replay contract.
- `c1r1` shared activation bridge for packet2/O and packet0/down.
- `c1r3` Q/K norm + RoPE postprocess station with packet14/15 current KV routes.
- `c6r1` Q fanout, attention return gather, and FFN intermediate gather.
- `c6r2` up/gate SwiGLU slice station.
- Main16 projection workers and row1 compact routes.
- Shape-A/B carrier and return windows.

## Files

- `contract.py`: Qwen3 layer constants and ABI checks.
- `dataflow.py`: typed qwen3-dataflow graph.
- `generate.py`: MLIR-AIE physical skeleton generator from the target graph.
- `check_contract.py`: integration check for contract, graph, and generated MLIR.
- `emit_mlir.py`: writes `qwen3-layer/build/qwen3_dataflow.mlir`.
- `mlir_utils.py`: shared MLIR-AIE BD, lock, queue, and runtime sequence
  helpers used by runnable cases.
- `npu_generate.py`: runnable MLIR-AIE backend for the full 608-patch
  Q/K/V/O/up/gate/down schedule.
- `qwen3_layer.cc`: AIE tile kernels for the runnable backend.
- `npu_reference.py`: CPU reference for the runnable backend.
- `current_runner.py`: runner for the current full-schedule backend.
- `npu_build.py`: shared AIE object, MLIR, xclbin, and NPU runtime helpers.
- `bridge_generate.py`: runnable MLIR-AIE for c6r1/c1r1/main16 bridge cases.
- `bridge_reference.py`: CPU reference for bridge smoke cases.
- `bridge_runner.py`: NPU runner for the shared activation bridge cases.
- `swiglu_generate.py`: runnable MLIR-AIE for main16 up/gate -> row1/c1r1
  compact -> c6r2.
- `swiglu_reference.py`: CPU reference for the up/gate compact and c6r2
  contract.
- `swiglu_runner.py`: NPU runner for the up/gate compact case.
- `c1r2_generate.py`: runnable MLIR-AIE for O compact -> c1r2 -> packet0
  replay -> main16 -> c6r2.
- `c1r2_reference.py`: CPU reference for the c1r2 full-vector replay case.
- `c1r2_runner.py`: NPU runner for the c1r2 integration case.
- `shape_generate.py`: runnable MLIR-AIE for Q fanout, left/right KV split,
  Shape-A/B carrier, packet2 O bridge, and main16 O chunk summaries.
- `shape_reference.py`: CPU reference for the Shape-A/B attention-to-O bridge
  layout contract.
- `shape_runner.py`: NPU runner for the Shape-A/B integration case.
- `qwen3_bridge.cc`: AIE tile kernels for bridge smoke cases.
- `cases/`: modular case registry and wrappers for executable integration
  boundaries, including the Q/K/V -> Shape-A/B -> O compact path.
- `run_npu.py`: thin CLI dispatcher for real NPU integration cases.

## Check

```bash
.venv/bin/python qwen3-layer/check_contract.py
```

## Run On NPU

```bash
.venv/bin/python qwen3-layer/run_npu.py --check-only
.venv/bin/python qwen3-layer/run_npu.py --build-only
.venv/bin/python qwen3-layer/run_npu.py
.venv/bin/python qwen3-layer/run_npu.py --case c1r1-o-bridge
.venv/bin/python qwen3-layer/run_npu.py --case c1r1-down-bridge
.venv/bin/python qwen3-layer/run_npu.py --case ffn-upgate-c6r2-bridge
.venv/bin/python qwen3-layer/run_npu.py --case c1r2-o-upgate-bridge
.venv/bin/python qwen3-layer/run_npu.py --case shape-attention-o-bridge
.venv/bin/python qwen3-layer/run_npu.py --case attention-kv16-o-bridge
.venv/bin/python qwen3-layer/run_npu.py --case kvscan-attention-kv16-o-bridge
.venv/bin/python qwen3-layer/run_npu.py --case qkv-shape-o-c1r2-bridge
.venv/bin/python qwen3-layer/run_npu.py --case full-layer-contract-bridge
```

This runner is intentionally an integration boundary, not a tiny unit test. It
streams the exact `608` Qwen/MyLM patch schedule through real NPU packet queues,
runs Q/K/V/O/up/gate/down, and validates the output against `npu_reference.py`.
The same script also exposes smaller executable boundaries: `--check-only`
validates the generated MLIR structure, and `--build-only` verifies routing,
core compilation, instruction generation, PDI, and xclbin generation. The
`c1r1-o-bridge` and `c1r1-down-bridge` cases verify the qwen3-dataflow shared
activation bridge: c6r1 packet2/packet0 -> c1r1 DMA4 -> c1r1 DMA1 -> all 16
main16 DMA0 activation inputs. The `ffn-upgate-c6r2-bridge` case verifies the
opposite FFN direction: all 16 main tiles emit deterministic up/gate records,
row1 column compact tiles gather per-row records, c1r1 performs the global
compact, and c6r2 receives the two 256-dword halves with low=up and high=gate.
The `c1r2-o-upgate-bridge` case verifies the longer full-vector loop: main16
emits an O compact record through row1/c1r1 into c1r2, c1r2 emits 48 packet0
full-vector payload replays, c1r1 bridges those payloads back into all 16 main
tiles, and the resulting deterministic up/gate records return through
row1/c1r1 into c6r2.
The `shape-attention-o-bridge` case verifies the first real attention fabric
handoff boundary: host Q is fanned out from c6r1, host K/V is split through
c0r1/c7r1, Shape-A produces deterministic carrier windows, Shape-B combines
those carriers with V windows, c6r1 gathers the four attention return windows
as packet2, c1r1 bridges packet2 to all 16 main tiles, and row1 drains each
main tile's O chunk summary to host output.
The `attention-kv16-o-bridge` case keeps the same packet2/O handoff but uses
production-shaped Shape-A/B windows: 8 Q heads x 128 dim, 2 KV heads x 16
tokens x 128 dim, and an 80-dword carrier split into 64 dwords of packed
softmax weights plus 16 dwords of per-head scalar state. It validates the
larger K/V scan and carrier ABI on real NPU before replacing the full-layer
contract producer.
The `kvscan-attention-kv16-o-bridge` case keeps that kv16 math and packet2/O
handoff, but changes the KV input ABI from prepacked left/right `K,V,K,V`
payloads to logical KV cache-side buffers laid out as `K0,K1,V0,V1`. The shim
issues 2048-dword BD slices for K0/V0/K1/V1 into c0r1/c7r1, and row1
memtiles reconstruct the streaming side layout consumed by Shape-A/B. This is
the boundary the full layer must use; c1r3/c1r4 must not materialize the two
8192-dword sides.
The `qkv-shape-o-c1r2-bridge` case removes the host-fed Q/K/V shortcut from
that boundary. Main16 emits deterministic Q/K/V records through row1 column
compact tiles and c1r1, c1r3 expands the global compacts into Q and split K/V
payloads, Shape-A/B returns packet2 attention data through c6r1/c1r1, main16
consumes packet2 as O chunks, and c1r2 validates the resulting O compact
summary. It is still a deterministic contract kernel: it proves the physical
cross-tile ABI and layout, not production online softmax math.
The `full-layer-contract-bridge` case extends that closed loop through c1r2
replay, main16 up/gate, row1/c1r1 compact, c6r2 SwiGLU, c6r1 packet1 down
handoff, main16 down chunks, and c1r2 final summary. It validates the
Q/K/V -> attention/O -> FFN/up/gate -> SwiGLU -> down physical wiring on the
NPU with deterministic contract math.
That case keeps three NPU constraints explicit: c1r2 packet0 replay owns
MM2S1 BD1, so final host drain uses a separate BD; c1r1 S2MM3 does not accept
the same BD bank as lower S2MM channels, so the fourth compact input group uses
high BD IDs; and c6r2 receives only the 256-dword up/gate payload halves, not
the 257-dword compact headers.

The up/gate bridge intentionally uses one S2MM channel per row in each row1
column compact tile. A single S2MM channel with multiple packet BDs is not a
valid replacement for ordered packet-id scheduling when independent main tiles
produce rows concurrently; the memtile consumes arrivals in stream order, so
row0 gate can be captured by the row1 up BD. Splitting rows across channels
keeps the packet handoff deterministic while still using legal memtile BD banks
for AIE2p.

The streaming KV scan case keeps two similar hardware constraints explicit.
First, the four row1 KV output channels cannot share one generic "payload full"
token when the shim is slicing K0/V0/K1/V1 independently; each slot needs its
own empty/full lock, otherwise an MM2S channel can consume the readiness token
for the wrong slot and the run can timeout. Second, generated runtime
descriptors must not patch an address argument that the xclbin kernel metadata
does not expose. The kvscan case therefore uses four BO arguments
`Q, left_kv_cache, right_kv_cache, output`, and `mlir_utils.py` validates the
maximum `arg_idx`.

## Emit

```bash
.venv/bin/python qwen3-layer/emit_mlir.py
```

The runnable backend proves the current executable hardware contract. The
bridge cases replace the runner's diagnostic main/edge handoff with physical
c1r1/c6r1, c1r2, and row1/c1r1/c6r2 paths specified by
`qwen3-dataflow.md`. These cases use deterministic contract kernels to verify
physical ABI and layout, not production RMSNorm, attention, or SwiGLU math.
The current attention frontier is `kvscan-attention-kv16-o-bridge`: it proves
the production-shaped Shape-A/B K/V window and carrier ABI can be fed by
shim-sliced streaming KV cache buffers on real NPU execution. Remaining work is
to connect the same boundary to main16-produced Q/current K/V and then replace
the remaining deterministic RMSNorm, SwiGLU, down, and Q4NX contract pieces
with production kernels.
Do not wire this by materializing both 8192-dword KV sides inside the c1r3
postprocess tile; that would exceed the intended compute-tile local-memory
budget. The full-layer version needs a streaming KV path through shim/row1/edge
tiles, matching the qwen3-dataflow design.
