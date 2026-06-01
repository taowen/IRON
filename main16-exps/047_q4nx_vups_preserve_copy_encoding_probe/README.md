# Q4NX VUPS Preserve-Copy Encoding Probe

Experiment 046 showed that moving `vups.4x` earlier while delaying the
`bmhh1` high-half copy breaks the zero/offset numeric contract.

This experiment tests whether we can advance `vups.4x` while preserving that
copy timing. It is a compiler/encoder integration probe:

- candidate 1 keeps `bmhh1 <- bmhh2` at `0x700` and adds `vups.4x` to the same
  bundle, leaving only `vadd` at `0x704`;
- candidate 2 moves `vups.4x` to the earlier `0x6f2` bundle and leaves only
  `vadd` at `0x704`.

The expected outcome is not a numeric result; it is whether the current
pre-bundled MIR route can encode these resource combinations.
