# Experiment 130: MyLM Main16 Record-Observable Harness

This experiment keeps the raw whole-core route from experiment 129, but adds a
real stream/lock harness around c2r2:

```text
shim2 MM2S0 -> c2r2 DMA0 activation ring
shim2 MM2S1 -> c2r2 DMA1 weight ring
c2r2 MM2S1 -> shim3 S2MM1 17-dword record
```

The runtime feeds `16 * --records` activation chunks and the same number of
weight chunks with zero data, then waits for `--records` 17-dword compact
records. A pass here proves more than xclbin loading: the MyLM raw program
consumed MLIR-AIE DMA/lock tokens and published records through the standard
main16 ABI.
