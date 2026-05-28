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

The up/gate bridge intentionally uses one S2MM channel per row in each row1
column compact tile. A single S2MM channel with multiple packet BDs is not a
valid replacement for ordered packet-id scheduling when independent main tiles
produce rows concurrently; the memtile consumes arrivals in stream order, so
row0 gate can be captured by the row1 up BD. Splitting rows across channels
keeps the packet handoff deterministic while still using legal memtile BD banks
for AIE2p.

## Emit

```bash
.venv/bin/python qwen3-layer/emit_mlir.py
```

The runnable backend proves the current executable hardware contract. The
bridge cases replace the runner's diagnostic main/edge handoff with physical
c1r1/c6r1, c1r2, and row1/c1r1/c6r2 paths specified by
`qwen3-dataflow.md`. These cases use deterministic contract kernels to verify
physical ABI and layout, not production RMSNorm, attention, or SwiGLU math.
Remaining work is to connect these executable boundaries into the 608-patch
backend, then replace the deterministic Shape-A/B contract with production
Q/K/V scan, online softmax, and weighted-V math.
