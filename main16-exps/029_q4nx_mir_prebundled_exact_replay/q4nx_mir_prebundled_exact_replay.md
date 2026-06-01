# Q4NX MIR Prebundled Exact Replay

Status: `passed`

This experiment is a MIR usage gate, not a new NPU numeric gate.
It checks which MIR form can reproduce MyLM's `0x260..0x1850` Q4NX hot-loop bytes exactly.

## Contract Learned

- Unbundled machine MIR plus `--start-before=postmisched` is a scheduler input; opcode counts are not a numeric contract.
- Pre-bundled MIR plus `--start-after=postmisched --skip-machine-alignment` is the right encoder path for a hand-scheduled body.
- MyLM disassembly addresses include semantic padding. Address gaps must be emitted as explicit `BUNDLE { NOP }` entries.

## Results

- source raw: `experiments/130_mylm_main16_record_observable_harness/mylm_c2r2_program.bin`
- hot range: `0x260..0x1850`
- hot bytes: `5616`
- source bundles: `963`
- inserted gap bytes: `24`

| build | text bytes | matches original | first diff |
| --- | ---: | --- | --- |
| `mylm_hotloop_prebundled_unpadded` | `5600` | `False` | `0x2ba: got 0xf8, expected 0x00` |
| `mylm_hotloop_prebundled_padded` | `5616` | `True` | `None` |

## Inserted Address Gaps

- before `0x2c2`: `8` bytes
- before `0x1828`: `8` bytes
- before `0x1844`: `8` bytes

## Interpretation

- The padded pre-bundled MIR path is byte-exact against MyLM's hot loop.
- The previous MIR numeric failure is now attributable to using scheduler-generated order, not to MIR encoding capability.
- The next useful gate is to patch the byte-exact pre-bundled object through the existing 028 numeric harness. It should be a no-op replacement for MyLM hot bytes; if that passes, we can change one bundle at a time and keep byte/numeric gates tight.
