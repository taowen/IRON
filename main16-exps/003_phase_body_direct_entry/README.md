# 003 Phase Body Direct Entry

This experiment probes whether MyLM main16 phase bodies can be entered directly
from a raw caller stub, without going through dispatcher `0x36d0`.

It uses the same observable route as the earlier main16 harness:

```text
shim2 MM2S0 -> c2r2 DMA0 activation ring
shim2 MM2S1 -> c2r2 DMA1 weight ring
c2r2 MM2S1 -> shim3 S2MM1 compact record output
```

The probe feeds each phase body with its exact per-phase stream count and waits
for the full phase-local record count:

```text
Q/K/V:   12 records, 192 chunks
O:        8 records, 128 chunks
up/gate: 48 records, 768 chunks
down:     8 records, 384 chunks
```

Run:

```bash
./run.py
```

If a direct-entry variant times out, recover the pinned driver before running
more NPU experiments:

```bash
sudo systemctl restart amdxdna-pinned.service
```
