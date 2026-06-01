# 020 Peano Q4NX Compiler Route

This experiment tries the compiler route for the main16 Q4NX hot body.

It compiles a narrow set of C++/Peano intrinsic candidates and scores the
generated AIE2P assembly against the MyLM static Q4NX target. The purpose is to
decide whether Peano can be used as the scheduler/register allocator for a
Q4NX-specific code generator, instead of continuing to hand-schedule raw
assembly.

Run:

```bash
python3 main16-exps/020_peano_q4nx_compiler_route/run.py --force
```

Outputs:

- `peano_q4nx_compiler_route.json`
- `peano_q4nx_compiler_route.md`
- `build/peano_q4nx_compiler_route.s`

