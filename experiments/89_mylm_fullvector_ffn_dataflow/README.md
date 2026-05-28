# 89_mylm_fullvector_ffn_dataflow

This experiment tightens the remaining full-vector and FFN dataflow model.

Exp88 classified `c1r2` and `c6r2` as aux compute nodes. Exp89 adds
dimension-level evidence:

- `c1r0.bd0` carries the 2048-dword hidden input, matching `c1r2.ch1`.
- `c1r0.bd1` carries 4096 dwords of RMSNorm weights, matching the two
  2048-dword `c1r2.ch0` buffers.
- `c1r0.bd2` carries the 192-dword q/k norm + RoPE side input, matching
  `c1r3.ch1`.
- `c1r2.ch2` publishes 2049 dwords, matching one control dword plus the
  2048-dword final hidden output.
- `c6r2` consumes 512 dwords and emits 256 dwords. This matches one
  `SwiGLU(gate) * up` slice: 512 bf16 up + 512 bf16 gate -> 512 bf16 output.
- `c6r1` gathers/publishes 6144 dwords, exactly the 12288-bf16 FFN
  intermediate needed by down projection.

## Run

```bash
.venv/bin/python experiments/89_mylm_fullvector_ffn_dataflow/run.py --reuse
```

Artifact:

```text
/tmp/iron_exp89_mylm_fullvector_ffn_dataflow/fullvector_ffn_dataflow.txt
```

## Result

The strongest current model is:

```text
c1r2 = hidden-input / RMSNorm / residual / final-output full-vector station
c6r2 = SwiGLU slice station
c6r1 = full 12288-bf16 FFN-intermediate gather + packet0 down bridge
```

This still does not decode every value bit. Exp90 closes the c1r1/c6r1
activation bridge into main16, exp91 closes the normal `c1r2` phase order, and
exp92 closes the scheduler-critical main16 header/replay controls. The remaining
work is register-level `c1r2` value calibration and optional bit-level header
decomposition.

Exp95 later fixes the `c6r2` half order: one input BD is physically
`up[0x400]` followed by `gate[0x400]`, and the output is `SiLU(gate) * up`.
