# Q4NX Dynamic Tiny Arithmetic Gate

This combines the planned 016 and 017 gates.

It keeps the isolated MyLM direct-QKV harness from experiment 015, but replaces
the generated constant record emitter with a generated source-assembly body that
reads q4/scale/zero fields from the active weight chunk and computes the compact
record payload.

Scope:

- record0 only;
- active chunk0 only;
- bf16-one activation and bf16 `1/64` scale/zero synthetic cases;
- dynamic single q4 word, dynamic high-half q4 word, dynamic single-nibble,
  dynamic two-q4word accumulation, and dynamic zero contribution.

The goal is exact record equality against the real MyLM direct `0x1870` body.
