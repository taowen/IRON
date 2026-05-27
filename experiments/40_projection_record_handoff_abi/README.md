# Experiment 40: Projection Record Handoff ABI

This experiment validates the phase-handoff shape that the MyLM/FastFlowLM
reverse engineering keeps pointing at:

- a projection tile emits a `17` dword record: one header plus `16` payload
  dwords, matching `32` bf16 projection outputs;
- four row records form one `65` dword column record;
- the same payload can be replayed as `64` data-only dwords;
- sixteen records form a `257` dword output block;
- eight blocks form a `2049` dword hidden record;
- twenty-four blocks form a `6144` dword FFN intermediate replay.

The experiment intentionally does not implement Q4NX, attention, O projection,
or FFN math. It only tests the ABI shape and route boundary that those phases
need in order to hand data off without returning to DDR between every phase.

## Dataflow

```text
src0..src3 compute tiles
  emit deterministic 17-dword records
    -> packet route to mt0

mt0 row1 memtile
  gathers 4 records into a 68-dword raw column buffer
    -> streams the raw column to agg

agg compute tile
  sorts the four routed records by row from their headers
  builds column65 and replay64
  reads a host-provided 24-block record table
  builds block257, hidden2049, and ffn6144
    -> drains one host-visible output buffer
```

The routed `4 x 17 -> 65/64` path is real AIE data movement. The larger
`257/2049/6144` ladder is computed from a host-provided record table so this
experiment can validate the full ABI sizes without overloading the route graph.

## Run

```bash
source /opt/xilinx/xrt/setup.sh
.venv/bin/python experiments/40_projection_record_handoff_abi/run_npu.py
```

Passing means every segment matches the CPU reference exactly.
