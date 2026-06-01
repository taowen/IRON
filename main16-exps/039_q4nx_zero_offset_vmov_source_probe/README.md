# Q4NX Zero/Offset Vmov Source Probe

Experiment 038 showed that every apparent-dead `vmov bmhh1, bmhh2` candidate
fails only the `zero0_allq4` direct-QKV case.

This experiment narrows that failure by mutating one representative site:

```text
0x700: vmov bmhh1, bmhh2
```

It tests same-slot replacements:

- `nopm`
- `vmov bmhh1, bmhh1`
- `vmov bmhh1, bmll2`
- `vmov bmhh1, bmlh2`
- `vmov bmhh1, bmhl2`
- `vmov bmhh1, bmhh2`

Run:

```bash
python main16-exps/039_q4nx_zero_offset_vmov_source_probe/run.py
```

The goal is to separate a data dependence on `bmhh2` from a generic timing or
writeback dependence on `bmhh1`.
