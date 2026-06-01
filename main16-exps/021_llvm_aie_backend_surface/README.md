# 021 LLVM-AIE Backend Surface

This experiment answers a narrow question: if C++ cannot control the final
AIE2P assembly tightly enough, which pieces of `~/projects/llvm-aie` are worth
using for the main16 Q4NX route?

It does not copy LLVM backend code into IRON. It inventories the reusable
surfaces that can serve a small MyLM-style source-assembly or MIR generator:

- AIE2P instruction definitions.
- AIE2P instruction-selection patterns.
- AIE2P schedule and bypass metadata.
- MIR examples for `vups.4x`, `vextbcst.16`, `vmac.f`, and postpipelined GEMM.
- Backend tools and passes that should remain external services.

Run:

```bash
python3 main16-exps/021_llvm_aie_backend_surface/run.py
```

Outputs:

- `llvm_aie_backend_surface.json`
- `llvm_aie_backend_surface.md`

