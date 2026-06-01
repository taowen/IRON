# Q4NX Schedule DSL Window Interface Gate

Experiment 048 added a conservative bundle-signature manifest. This experiment
adds the next small piece needed for a real schedule DSL: local window interface
checking.

For every candidate window it computes:

- live-ins: registers read before being defined inside the window;
- boundary defs: registers defined inside the window.

The first version is intentionally simple and text-derived. It does not try to
model all hardware register aliases. Its job is to catch obvious generator
mistakes before MIR encoding and NPU gates.

The gate compares:

- a safe high-half swap from experiment 045;
- the `vups.4x` advance from experiment 046, which keeps the same textual
  interface but is rejected by the resource manifest;
- a deliberately bad candidate that drops the high-half copy and should be
  rejected by the window interface.
