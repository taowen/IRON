# VUPS.4x Lane Pattern Probe

This experiment extends the cell-level `vups.4x` readback probe with a
non-uniform source vector.

It is still a value-semantics experiment, not a performance experiment. The
goal is to distinguish lane transfer behavior for `x2d`, `cml`, and `cmh`
destinations before updating the Q4NX bundle mutation manifest.

Run the default focused case set:

```bash
python main16-exps/037_vups4x_lane_pattern_probe/run.py
```

Run every generated case:

```bash
python main16-exps/037_vups4x_lane_pattern_probe/run.py --case-set all
```

