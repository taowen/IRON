# Q4NX Generated MIR Contiguous Window Gate

Experiment 043 changed repeated isolated source bundles. This experiment changes
a contiguous producer/consumer window in one activation group:

```text
0x6ee: $bmlh1 = VMOV_alu_mv_mv_x $bmlh2
0x6f6: $x2    = VCONV_bf16_fp32_mv_x_srs_bf $cml1 ...
0x6f6: $bmhl1 = VMOV_alu_mv_mv_x $bmhl2
0x6f6: $dm3   = VMAC_f_vmac_bf_vmul_bf_core_X_X ...
0x700: $bmhh1 = VMOV_alu_mv_mv_x $bmhh2
0x704: $dm1   = VUPS_4x_mv_ups_x2d_upsSign0 ...
0x704: $dm2   = VADD_vmac_cm2_add_reg ...
```

The schedule shape stays the same, but the three `acc2 -> acc1` quadrant moves
inside this local window are changed to use `bmll2`:

```text
$bmlh1 = VMOV_alu_mv_mv_x $bmll2
$bmhl1 = VMOV_alu_mv_mv_x $bmll2
$bmhh1 = VMOV_alu_mv_mv_x $bmll2
```

The whole hot-loop range is regenerated from pre-bundled MIR and compared
against the unmodified MyLM raw reference on NPU.

Run:

```bash
python main16-exps/044_q4nx_generated_mir_contiguous_window_gate/run.py
```

Run the stronger asymmetric-zero gate:

```bash
python main16-exps/044_q4nx_generated_mir_contiguous_window_gate/run.py --case-set asymmetric-zero
```
