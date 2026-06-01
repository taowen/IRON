# MyLM Main16 Standalone Raw Kernel

This experiment packages the MyLM `main16` c2r2 raw core program as a whole-core
AIE2P executable and asks MLIR-AIE/aiecc to do only the physical shell:

```text
MyLM raw bytes + fixed local static data
  -> synthetic ET_EXEC AIE2P ELF
  -> aie.core(...){ aie.end } {elf_file = ...}
  -> aiecc --no-compile xclbin/txn/insts
  -> NPUKernel load/run probe
```

The standalone harness declares tile `(2,2)`, locks `0..6`, and releases lock
id `6` in the runtime sequence. That maps to hardware lock `54`, which is the
start gate used by the MyLM dispatcher.

Run:

```bash
.venv/bin/python experiments/129_mylm_main16_standalone_raw_kernel/run.py
```

Expected interpretation:

- If packaging works, the transaction should contain three blockwrite payloads:
  `.text`, local `0x73c80`, and local `0x73d00`.
- A real run can still block after start unless the activation/weight/record DMA
  rings are also supplied; this experiment intentionally does not emulate the
  whole MyLM dataflow.
