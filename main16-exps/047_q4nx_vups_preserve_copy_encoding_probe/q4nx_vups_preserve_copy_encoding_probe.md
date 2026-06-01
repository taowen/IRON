# Q4NX VUPS Preserve-Copy Encoding Probe

Status: `passed`

| candidate | status | unpadded rc | padded rc | first diff |
| --- | --- | ---: | ---: | --- |
| `copy_and_vups_at_700` | `llc_unpadded_failed` | `-6` | `None` | `length mismatch: got 0, expected 5616` |
| `vext_and_vups_at_6f2` | `llc_unpadded_failed` | `-6` | `None` | `length mismatch: got 0, expected 5616` |

## Interpretation

- `copy_and_vups_at_700` asks whether the current MIR route can issue the high-half copy and `vups.4x` in the same bundle.
- `vext_and_vups_at_6f2` asks whether `vups.4x` can be paired with the earlier lane broadcast while leaving the high-half copy at `0x700`.
- If both fail in the unpadded encoder, the local schedule is constrained before NPU numeric testing: preserving copy timing leaves no currently encodable early `vups.4x` slot in this small window.

## Error Tails

### copy_and_vups_at_700

```text
hon3.12/site-packages/llvm-aie/bin/../lib/libLLVM.so+0x1b419e5)
#13 0x00007fb8acab93c0 llvm::MachineFunctionPass::runOnFunction(llvm::Function&) (.part.0) MachineFunctionPass.cpp:0:0
#14 0x00007fb8ac67071c llvm::FPPassManager::runOnFunction(llvm::Function&) (/var/home/taowen/projects/IRON/.venv/lib/python3.12/site-packages/llvm-aie/bin/../lib/libLLVM.so+0xe7071c)
#15 0x00007fb8ac670b31 llvm::FPPassManager::runOnModule(llvm::Module&) (/var/home/taowen/projects/IRON/.venv/lib/python3.12/site-packages/llvm-aie/bin/../lib/libLLVM.so+0xe70b31)
#16 0x00007fb8ac67148d llvm::legacy::PassManagerImpl::run(llvm::Module&) (/var/home/taowen/projects/IRON/.venv/lib/python3.12/site-packages/llvm-aie/bin/../lib/libLLVM.so+0xe7148d)
#17 0x000000000041c60b compileModule(char**, llvm::LLVMContext&) llc.cpp:0:0
#18 0x0000000000411f37 main (/var/home/taowen/projects/IRON/.venv/lib/python3.12/site-packages/llvm-aie/bin/llc+0x411f37)
#19 0x00007fb8ab2a45b5 __libc_start_call_main (/lib64/libc.so.6+0x35b5)
#20 0x00007fb8ab2a4668 __libc_start_main@GLIBC_2.2.5 (/lib64/libc.so.6+0x3668)
#21 0x0000000000412aee _start (/var/home/taowen/projects/IRON/.venv/lib/python3.12/site-packages/llvm-aie/bin/llc+0x412aee)
```

### vext_and_vups_at_6f2

```text
hon3.12/site-packages/llvm-aie/bin/../lib/libLLVM.so+0x1b419e5)
#13 0x00007f62956b93c0 llvm::MachineFunctionPass::runOnFunction(llvm::Function&) (.part.0) MachineFunctionPass.cpp:0:0
#14 0x00007f629527071c llvm::FPPassManager::runOnFunction(llvm::Function&) (/var/home/taowen/projects/IRON/.venv/lib/python3.12/site-packages/llvm-aie/bin/../lib/libLLVM.so+0xe7071c)
#15 0x00007f6295270b31 llvm::FPPassManager::runOnModule(llvm::Module&) (/var/home/taowen/projects/IRON/.venv/lib/python3.12/site-packages/llvm-aie/bin/../lib/libLLVM.so+0xe70b31)
#16 0x00007f629527148d llvm::legacy::PassManagerImpl::run(llvm::Module&) (/var/home/taowen/projects/IRON/.venv/lib/python3.12/site-packages/llvm-aie/bin/../lib/libLLVM.so+0xe7148d)
#17 0x000000000041c60b compileModule(char**, llvm::LLVMContext&) llc.cpp:0:0
#18 0x0000000000411f37 main (/var/home/taowen/projects/IRON/.venv/lib/python3.12/site-packages/llvm-aie/bin/llc+0x411f37)
#19 0x00007f6293f995b5 __libc_start_call_main (/lib64/libc.so.6+0x35b5)
#20 0x00007f6293f99668 __libc_start_main@GLIBC_2.2.5 (/lib64/libc.so.6+0x3668)
#21 0x0000000000412aee _start (/var/home/taowen/projects/IRON/.venv/lib/python3.12/site-packages/llvm-aie/bin/llc+0x412aee)
```
