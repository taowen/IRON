# Experiment 111: Q4NX Dynamic Instruction Cost

The active IRON exact body and MyLM raw body both use hardware loops, so static
opcode counts are misleading. This experiment expands the hot-loop counts by
their loop trip counts and compares:

- active IRON `q4nx_chunk_accum_asm_zol`;
- MyLM main16 raw Q4NX hot loop `0x260..0x1850`;
- an exact recomposed-coefficient target model derived from exp110.

The goal is not to prove a new kernel. It is to quantify which costs are real
enough to justify a new scheduled body instead of more small edits.

Run:

```bash
python3 experiments/111_q4nx_dynamic_instruction_cost/run.py
```

It writes:

```text
experiments/111_q4nx_dynamic_instruction_cost/q4nx_dynamic_instruction_cost.md
```

Current result:

- active exact already has the right MAC count: `512` dynamic `vmac.f`.
- MyLM has `528` dynamic `vmac.f` because it adds `16` group correction MACs.
- the real active-vs-MyLM gap is coefficient construction:
  `vconv.bf16.fp32 1280 vs 272`, `vconv.fp32.bf16 768 vs 0`,
  `vmul.f 512 vs 16`, `crupsmode 128 vs 0`, and `nop 3088 vs 216`.

So the next exact-parity assembly work should not chase MAC count. It should
reduce coefficient construction/conversion/control traffic while keeping zero
inside the dequant coefficient before MAC, per exp110.
