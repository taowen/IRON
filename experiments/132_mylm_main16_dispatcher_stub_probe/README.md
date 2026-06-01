# MyLM Main16 Dispatcher Stub Probe

This experiment replaces only the first bytes of the MyLM c2r2 raw program with
a tiny assembled caller stub. The rest of the MyLM raw program, phase bodies,
static segments, MLIR topology, DMA rings, and locks are reused from exp130.

The stub sets the caller stack slot that dispatcher `0x36d0` reads as its
phase/control pointer:

```text
caller [sp - 4] = 0x78200
local  [0x78200] = control value
call   0x36d0
```

The goal is to test whether this caller-side control path can switch the first
observed compact record header away from the exp130/exp131 `0x4` path.

Run:

```bash
./run.py --max-variants 2
```

Current result: `control=1` emits header `0x1`, while `control=0` and
`control=8` emit `0x4`. The tested caller slot is therefore a QKV-vs-alternate
gate, not a general phase-id selector.
