# 86_mylm_dispatcher_packet8_aux

This experiment narrows the remaining full-layer dataflow questions around
main16 phase dispatch and the packet8-like compact-record path.

The key result is that the post-O/FFN schedule should not be modeled as seven
independent operators or as a normal packet DMA tensor stream:

```text
main16 dispatcher:
  if *phase_flag == 1:
    0x1870 -> 0x1e80 -> 0x2490 -> 0x2aa0
  else:
    0x30c0
```

Every visible body publishes the same 17-dword record shape:

```text
body prologue: st r0, [sp, #-12]
publish:       lda el0, [sp, #-12]
record +0x00:  st el0
record +0x04: 16 bf16 values
record +0x24: 16 bf16 values
```

So the first dword is now known to be the body input register `r0`, saved to the
body stack and later republished as `el0`. Dispatcher-local evidence suggests
small immediate header candidates:

```text
0x1870: r0=0x1
0x1e80: r0=0x4
0x2490: r0=0x8
0x2aa0: r0=0x4
0x30c0: r0=0x4
```

The remaining unknown is not where the header comes from; it is the bitfield
meaning of those small control words. The direct caller at `0x160` sets local
activation, weight, and record base registers before entering dispatcher
`0x36d0`; the header candidates are generated in the dispatcher branch window,
not patched directly as a host descriptor field.

The packet8 evidence is also narrower:

```text
packet8 packet-enabled BD source: none
exact route-id 8 chain:
  c1r2 -> c2r2 -> c3r2 -> c4r2 -> c5r2 -> c6r2

main row2:
  c2r2/c3r2/c4r2/c5r2 ch2 bd4 len=17 packet_en=0

row1:
  c2r1/c3r1/c4r1/c5r1 bd0/bd1 len=17 packet_en=0
```

That means packet8 should stay classified as an unpacketized compact/aux record
path. It is not a packet14/15-style DMA packet stream.

The aux tiles are not simple transport:

```text
c1r2: scalar exp/rsqrt-style constants plus MAC/vector conversion code
c6r2: clamp/gather/vector conversion code
```

Remaining unknowns after this experiment, before exp92:

- scheduler-critical 17-dword record header values and replay counts,
- exact row1/aux consumption order for packet8-like compact records,
- exact Shape-A/B compact softmax carrier field order,
- exact placement/schedule for post-O residual, RMSNorm, up/gate, SwiGLU, and
  down over the compact records.

Exp92 resolves the first item for scheduling: normal mode uses `12 x 0x1` for
Q/K/V, `8 x 0x4` for O, `48 x 0x8` for up/gate, and `8 x 0x4` for down. The
bit-level decomposition of those header words remains optional.

Run:

```bash
.venv/bin/python experiments/86_mylm_dispatcher_packet8_aux/run.py --reuse
```
