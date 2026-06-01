# Q4NX Generated MIR Hotloop Variant Gate

Experiment 040 and 041 changed one encoded source bundle directly. This
experiment takes the next step: regenerate the whole `0x260..0x1850` hot loop
from pre-bundled MIR, with one source operation changed at site `0x700`:

```text
original: $bmhh1 = VMOV_alu_mv_mv_x $bmhh2
variant:  $bmhh1 = VMOV_alu_mv_mv_x $bmll2
```

The generated hot-loop bytes replace the corresponding range in the raw MyLM
program. The direct-QKV numeric gate then compares the generated variant against
the unmodified MyLM raw program.

Run:

```bash
python main16-exps/042_q4nx_generated_mir_hotloop_variant_gate/run.py
```

Run the stronger asymmetric-zero gate:

```bash
python main16-exps/042_q4nx_generated_mir_hotloop_variant_gate/run.py --case-set asymmetric-zero
```

This is still not a performance optimization. It proves that the MIR generator
route can produce a non-byte-copy hot-loop variant that survives the current
NPU numeric gate.
