# Exp125: Main16 QKV Nocall Scheduler Contract

Goal: turn the suspected per-chunk helper-call bottleneck into a concrete
contract for the next main16 scheduler experiment.

Current active code calls `q4nx_chunk_accum_asm_zol` once per Q4NX chunk from
the C++ scheduler. For Q/K/V prefix that is `192` calls per main tile; for a
full layer it is `1472` calls per main tile. This experiment quantifies that
boundary and emits a `q4nx_main16_qkv_scheduler_nocall` assembly contract
outline. A direct linked implementation later satisfied the static no-helper
contract but timed out on the `full-layer-qkv-prefix` NPU gate, so the active
qwen3-layer code remains on the known-good C++ scheduler plus
`q4nx_chunk_accum_asm_zol` source-assembly hot body.

Run:

```bash
python3 experiments/125_main16_qkv_nocall_scheduler_contract/run.py
```

Inputs:

- `qwen3-layer/main_projection_q4nx_fast.cc`
- `qwen3-layer/main_projection_q4nx_asm.s`

Outputs:

- `main16_qkv_nocall_scheduler_contract.md`
- `main16_qkv_nocall_scheduler_contract.json`
- `q4nx_main16_qkv_scheduler_nocall_contract.s.inc`

Status:

- Useful: helper-boundary cost model and lock ABI contract.
- Not active: the first direct nocall scheduler implementation.
- Next useful step: add progress attribution or write a Peano-shaped whole
  scheduler before attempting another active replacement.
