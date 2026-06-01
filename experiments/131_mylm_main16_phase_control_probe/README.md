# MyLM Main16 Phase-Control Probe

This experiment reuses the record-observable c2r2 harness from
`../130_mylm_main16_record_observable_harness` and changes only the raw MyLM
static/control segment bytes loaded at `0x73c80` and `0x73d00`.

The goal is to identify whether those control words select the MyLM main16
phase body or record header. The harness keeps the same physical ABI:

```text
shim2 MM2S0 -> c2r2 DMA0 activation ring
shim2 MM2S1 -> c2r2 DMA1 weight ring
c2r2 MM2S1 -> shim3 S2MM1 17-dword compact record
```

Run:

```bash
./run.py --max-variants 3
```

The default sweep is intentionally small. A bad raw-control patch can deadlock
the core, so the script checks NPU topology between variants and stops if the
device leaves the expected 6x8 state.

Current result: patching `73d00[0]` and `73c80[0]` across the known MyLM
phase-body/dispatcher addresses still emits `0x4` for every observed record.
Those first dwords are therefore not the direct standalone phase selector.
