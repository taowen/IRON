# Q4NX Tiny Codegen Numeric Gate

This is the first generated source-assembly gate after the MyLM payload layout
probes.

It does not try to be the full high-performance Q4NX hot loop yet. Instead, it
does the smallest useful isolated proof:

- build deterministic synthetic Q4NX inputs;
- run the real MyLM direct `0x1870` Q/K/V body on NPU;
- compute the expected record0 payload from the observed layout/parity contract;
- generate a tiny whole-core source-assembly program that consumes the same
  16 stream chunks and emits the generated compact record;
- run that generated program on NPU and require exact record equality.

This proves the codegen/harness/record ABI before replacing the emit-only body
with dynamic Q4NX arithmetic.
