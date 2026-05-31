# Experiment 101: MyLM Q4NX Group-Sum Source Assembly

This is the first implementation experiment after deciding that local tuning
of the current exact-rounding body is the wrong performance route.

The experiment runs one source-assembly body on real NPU. It is a numeric gate
for a MyLM-style hot-loop skeleton before anything can move into the production
main16 role:

- one aligned source-assembly zero-overhead loop with `lc=8`;
- one group-sum scratch load per quant group;
- `32` static `vextbcst.16 + vmac.f` main MACs per quant group;
- one offset/group-sum correction `vmac.f` per quant group;
- pre-expanded `q*scale` vectors feeding the MAC;
- no hot-loop vector store and no BF16 conversion traffic.

The numeric contract is intentionally synthetic:

```text
qscale = bf16(1.0)
activation[d] = bf16(1.0)
offset = bf16(2.0)
group_sum[g] = bf16(32.0)

per_group = 32 * 1 + 2 * 32 = 96
eight_groups = 768
```

First failed result:

```text
instruction shape:
  static vmac.f=33
  dynamic vmac.f=264
  static vextbcst.16=32
  dynamic vextbcst.16=256
  static group-sum loads=1
  dynamic group-sum loads=8
  static vst=1
  static vups.4x=0
  static vconv.bf16.fp32=0

numeric gate:
  expected dst = 768
  got dst      = 271
```

This is a real failure, and it is useful: it proves that opcode-count matching
is not enough. The source body assumes `vmac.f #0x33c` is a simple lane-wise
BF16 multiply of arbitrary `vbcst`/`vldb` vectors. The NPU result disproves
that assumption. MyLM's `q*scale` operands are not generic broadcast vectors;
they are produced by a specific `vunpack/vups/vconv` register plan, and the
MAC operand registers encode the row-lane layout.

The next step is therefore not to migrate this body. The next step is to
reverse the MyLM operand/register layout for one 32-lane quant group, then make
the synthetic body numerically pass with that exact layout before adding it to
`main_projection_q4nx_asm.s`.

Experiment 102 narrowed that failure to `vextbcst.16` use latency. A `vmac.f`
immediately after `vextbcst.16` can read the previous broadcast value. Exp102
validates the primitive group-sum formula with explicit latency spacing, but
this exp101 body still must not migrate: MyLM fills those latency windows with
useful `vunpack/vups/vconv/vmac` work and carries register state across group
boundaries, while this experiment keeps each quant group as an independent
skeleton.

An earlier draft kept `vups.4x` as a dead instruction in the same loop. That
compiled and ran, but also failed the numeric gate because the `vups`
destination aliases accumulator/control state unless it participates in the
same explicit register plan as the following MACs. That reinforces the same
conclusion: the kernel has to be designed as one register graph, not assembled
from individually legal instructions.

Run:

```bash
.venv/bin/python experiments/101_mylm_q4nx_group_sum_asm/run.py
```
