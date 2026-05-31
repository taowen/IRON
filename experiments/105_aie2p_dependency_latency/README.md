# Experiment 105: AIE2P Dependency Latency

Exp102 proved the primitive `vmac.f #0x33c` operand model when explicit latency
spacing is present. This experiment scans small producer-consumer gaps on real
NPU so the next Q4NX assembly generator has measured hazards instead of guessed
nop counts.

It measures four dependency families:

- `vextbcst.16 -> vmac.f`
- `vbcst.16 -> vmac.f`
- `vldb -> vextbcst.16 -> vmac.f`
- `vmac.f -> vst`

Each case writes one 16-float accumulator block. The runner prints the first
gap that produces the expected value. Gaps below that threshold are allowed to
fail; the point is to learn the required spacing and then fill it with useful
MyLM-style work, not to pad production code with nops.

Run:

```bash
.venv/bin/python experiments/105_aie2p_dependency_latency/run.py
```

Current result:

```text
vextbcst.16 -> vmac.f:        first passing gap = 1
vbcst.16 -> vmac.f:           first passing gap = 1
vldb -> vextbcst.16 -> vmac.f:first passing gap = 6
vmac.f -> vst:                first passing gap = 5
```

So the expensive hazards are not the broadcast-to-MAC pair itself; they are
vector-load-to-extract and MAC-to-storeback. A production Q4NX body should hide
those windows with useful unpack/upshift/convert/move/MAC work from neighboring
groups, not with local nop padding.
