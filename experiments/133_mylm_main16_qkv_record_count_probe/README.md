# MyLM Main16 QKV Record-Count Probe

This experiment reuses exp132's dispatcher stub with `control=1`, which enters
the MyLM c2r2 Q/K/V body and emits header `0x1`.

The goal is to confirm the Q/K/V body emits exactly 12 compact records under the
standalone harness:

```text
control=1, wait 12 records -> should complete, all headers 0x1
control=1, wait 13 records -> should show 12 headers 0x1, then the next body header 0x4
```

Run:

```bash
./run.py
```
