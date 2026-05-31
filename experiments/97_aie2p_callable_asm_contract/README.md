# Experiment 97: AIE2P Callable Assembly Contract

This experiment is a learning checkpoint before replacing any production
Qwen3 decode hot path with source assembly. It does not change `qwen3-layer`.

The goal is to make the callable-assembly contract explicit:

- which AIE2P registers the normal C ABI preserves;
- where the vector preserve-all convention exists, and where it does not;
- how a C++ role object calls a source-assembly function through relocations;
- how BF16 accumulator writeback differs from float accumulator writeback;
- why a production Q4NX body needs a full register and latency schedule, not
  only a `vextbcst.16 + vmac.f` instruction pair.

Run:

```bash
python3 experiments/97_aie2p_callable_asm_contract/run.py
```

Outputs are written under `experiments/97_aie2p_callable_asm_contract/build/`.
The important output is `build/report.md`.

## Current Interpretation

The experiment treats MyLM-style performance as a low-level scheduling problem:

1. Source assembly is viable as a C++ role-object callee.
2. Ordinary calls do not automatically preserve vector and accumulator state.
3. If a C++ caller needs vector state after an external call, Peano spills it
   around the call; the source assembly should therefore declare and respect a
   narrow clobber contract instead of relying on implicit preservation.
4. BF16 output can use `vst.conv.bf16.fp32`, while float output uses a different
   accumulator storeback shape.
5. AIE2P objdump order includes branch/return delay slots; stores after
   `ret lr` can still be semantically part of the function epilogue.
6. Load-use, MAC-use, and storeback latency must be scheduled in the assembly
   body itself.
7. The next production step should be one complete callable Q4NX lane/body with
   an explicit clobber list and exact output contract, not another partial
   intrinsic bridge.

This is deliberately independent from the existing
`aie_intrinsics_api_probe`: that directory contains many shape and NPU smoke
probes; this one is the narrower ABI contract that decides how those probes can
be safely called from the decode engine.
