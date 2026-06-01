# Q4NX Generated MIR High-Half Swap Gate

Experiment 044 changed operands inside a contiguous window while preserving the
original schedule shape. This experiment makes the first small schedule-changing
mutation that preserves bundle sizes.

In the local window around `0x6f6..0x700`, swap the timing of two high-half
`acc2 -> acc1` moves:

```text
original:
  0x6f6: $bmhl1 = VMOV_alu_mv_mv_x $bmhl2
  0x700: $bmhh1 = VMOV_alu_mv_mv_x $bmhh2

variant:
  0x6f6: $bmhh1 = VMOV_alu_mv_mv_x $bmhh2
  0x700: $bmhl1 = VMOV_alu_mv_mv_x $bmhl2
```

The full hot-loop range is regenerated from pre-bundled MIR and compared
against the unmodified MyLM raw reference on NPU.

Run:

```bash
python main16-exps/045_q4nx_generated_mir_highhalf_swap_gate/run.py
```

Run the stronger asymmetric-zero gate:

```bash
python main16-exps/045_q4nx_generated_mir_highhalf_swap_gate/run.py --case-set asymmetric-zero
```
