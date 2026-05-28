# 85_mylm_main16_phase_record

This experiment tightens the main16 side of the fused layer model.

Exp83 proved that no DDR scratch exists for O/residual/RMSNorm/FFN
intermediates. The remaining ambiguity was whether the main16 17-dword output
was only a debug/sideband signal or an actual production record.

The new evidence is from `c2r2` disassembly plus the CDO local-data payload:

```text
main16 output BDs:
  bd4 0x73c1c len=17
  bd5 0x7541c len=17

core write shape:
  st el0 at record +0x00
  vst.conv.bf16.fp32 at record +0x04
  vst.conv.bf16.fp32 at record +0x24
```

So the record is exactly:

```text
1 dword control/header + 32 bf16 projection values = 17 dwords
```

That is not enough for a full 4096-vector in one tile, but it is exactly enough
for one main tile's 32 output rows. The full vector is therefore a stream of
per-tile records that aux/row1 must reorder, strip/control, and rebroadcast.

The same summary also fixes the current-Q distributor shape:

```text
c1r3 postprocesses Q/K/V
  packet14/15 -> current K/V writeback
  circuit bd1 len=2048 -> full current Q to c6r1 DMA_1

c6r1 bd24 receives full Q at 0x24000
c6r1 bd25/bd2/bd26/bd3 split it into four shape-A windows:
  window0 -> c0r2
  window1 -> c0r4
  window2 -> c7r2
  window3 -> c7r4
```

Remaining unknowns after this experiment are narrower:

- the bit-level decomposition of the first dword; exp92 later fixes the
  scheduler-critical values `0x1/0x4/0x8` and replay counts,
- the exact value-level order inside the Shape-A/B compact attention carrier,
- the post-O/FFN phase schedule that consumes and emits these 32-value records.

Run:

```bash
.venv/bin/python experiments/85_mylm_main16_phase_record/run.py --reuse
```
