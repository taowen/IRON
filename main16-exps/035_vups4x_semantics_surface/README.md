# VUPS.4x Semantics Surface

This experiment records what `llvm-aie` can tell us about `vups.4x` and what it
cannot tell us.

It compiles the four AIE2P `vups.4x` machine-instruction variants from minimal
MIR and extracts the local compiler metadata for operands, implicit registers,
and schedule resources.

Run:

```bash
python main16-exps/035_vups4x_semantics_surface/run.py
```

