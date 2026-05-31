# Experiment 102: AIE2P VMAC Operand Layout

Experiment 101 showed that opcode-count matching is not enough: a body with
the right `vextbcst.16 + vmac.f` shape produced the wrong synthetic value.

This experiment removes Q4NX and tests the primitive operand cases on real
NPU:

- q operand registers `x2/x3/x5/x7/x9` after `vbcst.16(1)`
- `vextbcst.16` from activation lanes `0..31`
- group-sum correction `vbcst.16(2) * vbcst.16(32)`
- a 32-lane sum using `vextbcst.16 #0..31`

The important output is not performance. The point is to prove or disprove the
simple mental model for `vmac.f #0x33c` before designing the real MyLM-style
Q4NX register plan.

Current NPU result:

```text
qreg_x2..x9 all produce first lane = 1
vext_lane0..31 produce first lanes = 1..32
correction_2x32 produces first lane = 64
sum_32_vext_lanes produces first lane = 528
PASS
```

The useful conclusion is that the primitive operand model is valid when the
`vextbcst.16` result is not consumed too early. MyLM does not pay this latency
with nops; it software-pipelines the broadcast with unpack, upshift,
conversion, moves, and MACs from neighboring groups.

Run:

```bash
.venv/bin/python experiments/102_aie2p_vmac_operand_layout/run.py
```
