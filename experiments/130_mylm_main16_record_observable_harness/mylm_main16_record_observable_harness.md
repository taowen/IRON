# MyLM Main16 Record-Observable Harness

- Status: `timeout_waiting_for_record`
- Route: `shim2 MM2S0/1 -> c2r2 DMA0/1; c2r2 MM2S1 -> shim3 S2MM1`
- Records requested: `69`
- Fed chunks: activation `1104`, weight `1104`
- Preserve dispatcher p5 patch: `True`
- Activation pong flag word: `1`
- ELF load segments: `3`
- Transaction blockwrite addresses: `['0x4220000', '0x4203c80', '0x4203d00', '0x421d000', '0x421d040', '0x421d080', '0x421d0a0', '0x421d060', '0x421d020']`
- Runtime probe: `timeout_waiting_for_record`
- Unique record headers: `[]`
- `xrt-smi validate`: `timeout`

## Runtime Output

```text

```

## Interpretation

The package loaded, but the host timed out waiting for the 17-dword record. That means the remaining mismatch is in route/locks/control or the raw core did not reach its first record emit under this standalone harness.

## Artifacts

- MLIR: `experiments/130_mylm_main16_record_observable_harness/design.mlir`
- ELF: `experiments/130_mylm_main16_record_observable_harness/mylm_c2r2_main16_record_exec.elf`
- XCLBIN: `experiments/130_mylm_main16_record_observable_harness/design.xclbin`
- Insts: `experiments/130_mylm_main16_record_observable_harness/design.bin`
- TXN: `experiments/130_mylm_main16_record_observable_harness/design.txn.mlir`
