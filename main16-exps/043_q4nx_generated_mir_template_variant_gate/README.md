# Q4NX Generated MIR Template Variant Gate

Experiment 042 changed one generated MIR source bundle. This experiment changes
the whole repeated template:

```text
$bmhh1 = VMOV_alu_mv_mv_x $bmhh2
```

Every occurrence in the MyLM Q4NX hot loop is regenerated as:

```text
$bmhh1 = VMOV_alu_mv_mv_x $bmll2
```

There are 15 matching source bundles in the current MyLM hot loop.

The whole `0x260..0x1850` hot-loop range is regenerated from pre-bundled MIR,
patched into the raw program, and compared against the unmodified MyLM reference
on NPU.

Run:

```bash
python main16-exps/043_q4nx_generated_mir_template_variant_gate/run.py
```

Run the stronger asymmetric-zero gate:

```bash
python main16-exps/043_q4nx_generated_mir_template_variant_gate/run.py --case-set asymmetric-zero
```
