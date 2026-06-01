# Q4NX MIR Bundle Mutation Manifest

Status: `passed`

## Summary

| metric | value |
| --- | ---: |
| total bundles | 966 |
| source bundles | 963 |
| padding bundles | 3 |
| hard-safe source candidates | 0 |
| weak dead-def source candidates | 0 |
| timing-critical padding | 1 |
| byte-diff canary padding | 2 |

## Padding Results

| address | kind | length | class | ops | reason |
| --- | --- | ---: | --- | --- | --- |
| `0x2ba` | `padding` | `8` | `timing_critical_padding` | `` | failed the experiment 032 self-move numeric gate |
| `0x1820` | `padding` | `8` | `byte_diff_canary_padding` | `` | passed the experiment 032 self-move numeric gate |
| `0x183c` | `padding` | `8` | `byte_diff_canary_padding` | `` | passed the experiment 032 self-move numeric gate |

## Weak Source Candidates

These are not ready for mutation. They only pass a conservative cell-level dead-destination check.

| address | kind | length | class | ops | reason |
| --- | --- | ---: | --- | --- | --- |

## Known Failed Source Mutation Gates

These bundles may look removable under a narrow def/overwrite model, but real NPU numeric gates already failed.

| address | kind | length | class | ops | reason |
| --- | --- | ---: | --- | --- | --- |
| `0x44e` | `source` | `4` | `known_failed_source_mutation` | `vmov` | failed experiment 034 direct-QKV numeric gate after vmov-to-nopm replacement |
| `0x700` | `source` | `4` | `known_failed_source_mutation` | `vmov` | failed experiment 038 zero/offset direct-QKV numeric gate after vmov-to-nopm replacement |
| `0x9b4` | `source` | `4` | `known_failed_source_mutation` | `vmov` | failed experiment 038 zero/offset direct-QKV numeric gate after vmov-to-nopm replacement |
| `0xc68` | `source` | `4` | `known_failed_source_mutation` | `vmov` | failed experiment 038 zero/offset direct-QKV numeric gate after vmov-to-nopm replacement |
| `0xf1c` | `source` | `4` | `known_failed_source_mutation` | `vmov` | failed experiment 038 zero/offset direct-QKV numeric gate after vmov-to-nopm replacement |
| `0x11d0` | `source` | `4` | `known_failed_source_mutation` | `vmov` | failed experiment 038 zero/offset direct-QKV numeric gate after vmov-to-nopm replacement |
| `0x1484` | `source` | `4` | `known_failed_source_mutation` | `vmov` | failed experiment 038 zero/offset direct-QKV numeric gate after vmov-to-nopm replacement |
| `0x173e` | `source` | `4` | `known_failed_source_mutation` | `vmov` | failed experiment 038 zero/offset direct-QKV numeric gate after vmov-to-nopm replacement |

## Interpretation

- The manifest confirms there are no hard-safe source bundle mutations yet.
- `vups.4x` now uses the measured experiment 036/037 transfer model: `dm` updates all quadrants, `cml` updates the low half, and `cmh` updates the high half.
- Experiments 034 and 038's failed source mutations are tracked as known failed gates instead of being hidden behind a broad implicit-destination-read rule.
- `lfh*` writes are treated as external live-out because the hot-loop range ends before the phase body record emit is fully modeled.
- Padding canary sites are useful for packaging tests, but not for performance work.
- A real optimization now needs a stricter cell-level proof plus a cycle-preserving replacement bundle.
- The next real source mutation needs a fuller instruction semantics model, not another candidate from the old weak list.
