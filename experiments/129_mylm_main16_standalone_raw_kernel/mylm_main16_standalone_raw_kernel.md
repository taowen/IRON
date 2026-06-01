# MyLM Main16 Standalone Raw Kernel

- Status: `run_pass`
- ELF type: `EXEC`
- ELF load segments: `3`
- Sections: `{'.text': 14868, '.mylm_static_73c80': 128, '.mylm_static_73d00': 32}`
- Transaction payload bytes: `[14868, 128, 32]`
- Transaction blockwrite addresses: `['0x4220000', '0x4203c80', '0x4203d00']`
- XCLBIN bytes: `22378`
- Insts bytes: `40`
- Runtime probe: `run_pass`
- Known-kernel probe: `run_pass`
- `xrt-smi validate`: `pass`

## What This Tests

This experiment uses the raw-core route:

```text
MyLM c2r2 raw program bytes + fixed local static data
  -> synthetic AIE2P ET_EXEC ELF with three PT_LOAD segments
  -> aie.core(...){ aie.end } {elf_file = ...}
  -> aiecc --no-compile xclbin/txn/insts
  -> NPUKernel load/run with no host buffers
```

The MLIR harness only declares tile `(2,2)`, lock ids `0..6`, and releases
lock id `6` at runtime. The MyLM core program uses the hardware lock numbers
`48..54`; MLIR lock id `6` maps to core lock `54`, which is the start gate
seen in the MyLM dispatcher.

## Artifacts

- MLIR: `experiments/129_mylm_main16_standalone_raw_kernel/design.mlir`
- ELF: `experiments/129_mylm_main16_standalone_raw_kernel/mylm_c2r2_main16_exec.elf`
- XCLBIN: `experiments/129_mylm_main16_standalone_raw_kernel/design.xclbin`
- Insts: `experiments/129_mylm_main16_standalone_raw_kernel/design.bin`
- TXN: `experiments/129_mylm_main16_standalone_raw_kernel/design.txn.mlir`

## Runtime Result

Standalone MyLM raw kernel status: `run_pass`.

```text
load_begin
load_ok
run_ok
elapsed=0.0005347728729248047
result_type=XRTKernelResult
npu_time=488013
```

Known existing xclbin control status:

```text
load_begin
load_ok
run_ok
elapsed=0.018132925033569336
result_type=XRTKernelResult
npu_time=18091776
```

`xrt-smi validate` status:

```text
WARNING: User doesn't have admin permissions to set performance mode. Running validate in Default mode
Validate Device           : [0000:c6:00.1]
    Platform              : RyzenAI-npu4
    Power Mode            : Default
-------------------------------------------------------------------------------
Test 1 [0000:c6:00.1]     : gemm 
[?25l[39m[[38;5;111m                    [39m][38;5;111m[39m: Running Test... < 0s >

[1A[?25l[39m[[38;5;111m<->                 [39m][38;5;111m[39m: Running Test... < 0s >

[1A[?25l[39m[[38;5;111m <->                [39m][38;5;111m[39m: Running Test... < 0s >

[1A[?25l[39m[[38;5;111m  <->               [39m][38;5;111m[39m: Running Test... < 0s >

[1A[?25l[39m[[38;5;111m   <->              [39m][38;5;111m[39m: Running Test... < 0s >

[1A[?25l[39m[[38;5;111m    <->             [39m][38;5;111m[39m: Running Test... < 0s >

[1A[?25l[39m[[38;5;111m     <->            [39m][38;5;111m[39m: Running Test... < 0s >

[1A[?25l[39m[[38;5;111m      <->           [39m][38;5;111m[39m: Running Test... < 0s >

[1A[?25l[39m[[38;5;111m       <->          [39m][38;5;111m[39m: Running Test... < 0s >

[1A[?25l[39m[[38;5;111m        <->         [39m][38;5;111m[39m: Running Test... < 0s >

[1A[?25l[39m[[38;5;111m         <->        [39m][38;5;111m[39m: Running Test... < 1s >

[1A[?25l[39m[[38;5;111m          <->       [39m][38;5;111m[39m: Running Test... < 1s >

[1A[?25l[39m[[38;5;111m           <->      [39m][38;5;111m[39m: Running Test... < 1s >

[1A[?25l[39m[[38;5;111m            <->     [39m][38;5;111m[39m: Running Test... < 1s >

[1A[?25l[39m[[38;5;111m             <->    [39m][38;5;111m[39m: Running Test... < 1s >

[1A[?25l[39m[[38;5;111m              <
... truncated ...
```

## Interpretation

The standalone raw MyLM kernel loaded and the no-buffer run returned.
