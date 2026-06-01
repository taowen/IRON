# Q4NX MIR Dead Vmov NOPM Numeric Gate

This experiment mutates the first weak source-bundle candidate from experiment
033:

```text
0x44e: vmov bmhh1, bmhh2 -> nopm
bytes: f8 12 cb 19      -> f8 4a 03 18
```

The static proof is intentionally narrow: `acc1.bmhh` is overwritten by the next
event at `0x452` before any use.

Run:

```bash
python3 main16-exps/034_q4nx_mir_dead_vmov_nopm_numeric_gate/run.py --max-cases 0
```

A pass would show the manifest can find a real source-bundle mutation. A fail
means the manifest is still missing timing, alias, rounding, or offset-path
semantics.

Current result:

- `q4word0_allnibbles`, `q4word512_allnibbles`, and `q4word0_nibble3` pass.
- `zero0_allq4` fails with a one-step-lower payload pattern.

So the bundle is not generally removable. It appears dead for the simple q4 data
paths but still participates in the zero/offset correction path or in a
register alias not represented by the first manifest.
