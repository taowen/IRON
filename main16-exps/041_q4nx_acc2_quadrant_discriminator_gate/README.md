# Q4NX Acc2 Quadrant Discriminator Gate

Experiment 040 proved that alternate `acc2` sources at site `0x700` pass the
current four direct-QKV synthetic cases. That coverage still does not
distinguish `acc2` quadrants.

This experiment strengthens the gate with asymmetric zero cases:

- zero slot `0..15`
- low-half-only zero pair
- high-half-only zero pair

The original MyLM raw program is used as the reference for each case. Alternate
source mutations are compared to that reference, not to a hand-written formula.

Run:

```bash
python main16-exps/041_q4nx_acc2_quadrant_discriminator_gate/run.py
```

Run a smaller smoke gate:

```bash
python main16-exps/041_q4nx_acc2_quadrant_discriminator_gate/run.py --max-cases 4 --max-replacements 2
```
