# Source ASM Latency Model

Experiment 017 proved that source assembly can read DMA-written full-address
buffers, but only when consumers are delayed far enough after scalar loads. This
experiment turns that observation into a measured latency contract.

The goal is not performance. The goal is to stop guessing where to insert nops:

- sweep small producer/consumer gaps on real NPU;
- record the minimum gap that produces correct observable output;
- emit a latency table that later source-asm/codegen can use for hazard checks.
