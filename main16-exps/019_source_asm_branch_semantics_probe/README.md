# Source ASM Branch Semantics Probe

Experiment 018 showed non-monotonic branch results. This probe isolates scalar
comparison and branch semantics before using them in generated Q4NX arithmetic.

It emits one record with:

- raw `eq` results for equal and not-equal inputs;
- whether `jz` and `jnz` branch for scalar values 0 and 1.
