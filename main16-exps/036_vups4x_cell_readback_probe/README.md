# VUPS.4x Cell Readback Probe

This experiment runs a tiny source-assembly core on the NPU to observe
`vups.4x` value effects on accumulator quadrants.

The core:

1. consumes the normal activation/weight chunk locks used by the direct-QKV
   harness;
2. initializes `dm0` quadrants with a sentinel through explicit `vups.2x`
   quadrant writes;
3. executes one target `vups.4x`;
4. stores one selected accumulator quadrant into the compact record payload.

Run:

```bash
python main16-exps/036_vups4x_cell_readback_probe/run.py
```

