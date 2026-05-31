# Main16 QKV Nocall Scheduler Contract

This experiment quantifies the current per-chunk helper-call boundary and
defines the minimal contract for `q4nx_main16_qkv_scheduler_nocall`.

## Source Checks

- C++ helper call sites: `4`
- C++ QKV scheduler symbol references: `2`
- ASM helper symbols: `1`
- SAVE macros: `1`
- RESTORE macros: `1`
- Q4 lane macros: `1`
- Required markers present: `True`

## Current QKV Scheduler Disassembly

- Static helper call relocations: `3`
- Static record emit call relocations: `3`
- Activation full acquire shape: `acq #49` x `3`
- Weight full acquire shape: `acq #51` x `3`
- Activation empty release shape: `rel #48` x `3`
- Weight empty release shape: `rel #50` x `3`
- Record empty acquire shape: `acq #52` x `3`
- Record full release shape: `rel #53` x `3`
- Lock shape valid: `True`

## Current Dynamic Helper Boundary

| Phase | Records | Chunks/Record | Calls/Tile | Weight Base |
| --- | ---: | ---: | ---: | ---: |
| `Q` | 8 | 16 | 128 | 0 |
| `K` | 2 | 16 | 32 | 128 |
| `V` | 2 | 16 | 32 | 160 |

- QKV helper calls per tile: `192`
- QKV helper calls across main16: `3072`
- Full-layer helper calls per tile: `1472`
- Full-layer helper calls across main16: `23552`

## Per-Call Boundary Shape

- SAVE slots per call: `13`
- RESTORE slots per call: `13`
- Helper wrapper slots before macro expansion: `28`
- `Q4_EXACT_LANE_ZOL` invocations per call: `2`
- Accumulator loads per call from local memory: `2`
- Accumulator stores per call to local memory: `2`
- Exact 32-dim groups per call: `2`

## Dynamic Boundary Cost

- QKV save/restore instruction slots per tile: `4992`
- Full-layer save/restore instruction slots per tile: `38272`
- QKV accumulator load/store roundtrips per tile: `384` / `384`
- Full-layer accumulator load/store roundtrips per tile: `2944` / `2944`

## Nocall Scheduler Contract

- Keep the current MLIR topology, buffer addresses, DMA rings, and lock IDs.
- Add `q4nx_main16_qkv_scheduler_nocall` with the same C ABI as `q4nx_main16_qkv_scheduler`.
- Implement Q/K/V record-major loops inside assembly.
- Acquire activation/weight locks per chunk and release them after the inline body.
- Keep accumulators register-resident across the 16 chunks of one record.
- Emit the same 17-dword record headers and payload layout as the current C++ scheduler.
- The scheduler body must not call or jump to `q4nx_chunk_accum_asm_zol`.

## Generated Contract Include

- `/var/home/taowen/projects/IRON/experiments/125_main16_qkv_nocall_scheduler_contract/q4nx_main16_qkv_scheduler_nocall_contract.s.inc`

## Next Step

The first direct linked scheduler satisfied the static no-helper contract but
timed out on `full-layer-qkv-prefix`. Keep this experiment as the boundary cost
model, not as active production code. The next attempt should first add
progress attribution or use a Peano-shaped whole QKV scheduler so timeout can
be assigned to DMA0, DMA1, Q4 body, or record emit.
