# 022 Q4NX MIR Schedule Probe

This experiment tries the post-C++ compiler route selected by experiment 021.

The goal is not to implement the production Q4NX kernel yet. The goal is to
prove whether we can express the relevant AIE2P machine instructions directly
in MIR and let Peano assemble/schedule them:

- `VUPS_4x`
- `VEXTBCST_16`
- `VMAC_f`

Run:

```bash
python3 main16-exps/022_q4nx_mir_schedule_probe/run.py
```

Outputs:

- `q4nx_mir_schedule_probe.json`
- `q4nx_mir_schedule_probe.md`
- `build/*.mir`
- `build/*.s`
- `build/*.o`
- `build/*.objdump`

