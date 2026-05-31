# Experiment 98: AIE2P Float Accumulator In-Place Assembly

This experiment isolates one production blocker from `qwen3-layer`: whether a
normal C++ AIE core wrapper can call a source-assembly callee that loads an
FP32 accumulator from a tile-local buffer with `vlda`, updates it with
`vmac.f`, stores it back with `vst`, and returns through the normal C ABI.

It does not change the active Qwen3 decode path. The active main16 kernel stays
on the fastest known C++ native-MAC implementation unless this boundary is both
correct and faster in a production-shaped body.

Run:

```bash
.venv/bin/python experiments/98_aie2p_float_accum_inplace_asm/run.py
```

Observed result:

```text
NPU time: 533.7 us
got[0:8]: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
PASS: callable source assembly loaded, updated, and stored FP32 accumulator in place
```

So the direct accumulator load/store boundary is viable in isolation. The
production Q4NX direct-target timeout is more likely caused by the full asm
body's register preservation, scheduling, live ranges, or buffer lifetime.
