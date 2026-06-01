# 004 Q4NX Hot Body Schedule

This experiment produces a local, reproducible schedule summary for the MyLM
main16 Q4NX microkernel.

It does not run the NPU. It disassembles the currently checked-in MyLM c2r2
whole-core ELF used by the observable harness and emits:

- exact dotted opcode counts for `0x260..0x1850`;
- loop count and static/dynamic instruction-shape summary;
- activation-lane group boundaries;
- a first-group register flow table with best-effort def/use columns.

Run:

```bash
./run.py
```
