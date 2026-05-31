# Exp127: Main16 Whole-Program Scaffold

Goal: create the first generated artifact for the MyLM-style main16 direction.

Exp126 defines the contract. This experiment turns that contract into a
compileable source-assembly scaffold with stable symbols for:

- whole-program entry;
- dispatcher;
- shared Q4 body;
- Q/K/V, O, up/gate, and down phase bodies.

The scaffold intentionally does not enter the active `qwen3-layer` path. It is
the code-generation target that will later absorb the real Q4 body. The current
scaffold already fixes the full phase schedule, the IRON compact headers
`10..15`, the `p6` FP32 accumulator residency, and the FP32-to-BF16 record
emitter contract. The shared Q4 body is generated from the same
`qwen3-layer/tools/main16_q4nx_asm_lib.py` template used by the active main16
role object, so this experiment does not maintain a second Q4NX hot body. The
phase bodies call the shared Q4 run body once per header run; that run body
owns the record/chunk loops and the DMA lock protocol.

The entry ABI matches the active `q4nx_main16_layer_scheduler` shape: `p0..p5`
carry the existing ping/pong buffers, `r0..r2` carry group/row/num_rows, and
`r3` is an exclusive phase limit. The same dispatcher can run QKV prefix
(`3`), attention-O/QKVO (`4`), upgate (`6`), or full layer (`7`) without adding
a QKV-only scheduler symbol.

The generated control flow is now a real nested-call scaffold rather than a
flat disassembly sketch. Entry, dispatcher, phase bodies, and the shared Q4
body each save and restore `lr` with a 64-byte stack frame. The record emitter
is checked as a leaf. This avoids the old compileable-but-unrunnable failure
where every `jl` overwrote the caller return address.

Run:

```bash
python3 experiments/127_main16_whole_program_scaffold/run.py
```

Outputs:

- `main16_whole_program_scaffold.s`
- `main16_whole_program_scaffold.o`
- `main16_whole_program_scaffold.md`
- `main16_whole_program_scaffold.json`
