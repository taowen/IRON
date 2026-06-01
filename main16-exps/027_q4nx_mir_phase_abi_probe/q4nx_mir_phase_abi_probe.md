# Q4NX MIR Phase ABI Probe

Status: `passed`

This experiment explains why the full hot-loop MIR object from experiment 026 is not yet directly runnable.
The generated MIR covers `0x260..0x1850`; MyLM enters it through a `0x01f0` prologue and the Q/K/V phase body.

## Checks

| check | pass | detail |
| --- | --- | --- |
| `prologue_sets_lc_2` | `True` | 0x01f0 prologue owns hot-loop LC setup |
| `prologue_sets_loop_bounds` | `True` | 0x01f0 prologue sets LS/LE to the hot loop range |
| `prologue_rebases_weight_pointer` | `True` | 0x01f0 advances p0 by 0x400 before hot loop |
| `prologue_sets_constants` | `True` | hot loop constants are not provided by the external caller |
| `qkv_setup_writes_group_sums` | `True` | Q/K/V phase body writes 8 group-sum halfwords before calling 0x01f0 |
| `call_site_calls_microkernel` | `True` | Q/K/V phase body calls the shared Q4NX microkernel |
| `call_site_sets_p0_weight` | `True` | delay slot maps p0 to selected Q4NX weight buffer |
| `call_site_sets_p1_activation` | `True` | delay slot maps p1 to selected activation chunk buffer |
| `call_site_sets_p2_scratch` | `True` | delay slot maps p2 to scratch/control base |
| `call_site_sets_p3_group_sums` | `True` | delay slot maps p3 to group-sum scratch stream |
| `record_emit_uses_q4_result` | `True` | post-call phase body emits compact record payload from q4 result state |

## Range Summaries

### `microkernel_prologue`

- range: `0x1f0..0x260`
- instruction_lines: `13`
- `lda`: `1`
- `lda.s8`: `1`
- `mov`: `4`
- `mova`: `7`
- `movx`: `5`
- `movxm`: `6`
- `paddb`: `1`
- `vbcst.16`: `1`
- `vconv.fp32.bf16`: `2`

### `qkv_group_sum_setup`

- range: `0x1a1e..0x1d70`
- instruction_lines: `161`
- `lda.s8`: `1`
- `mov`: `2`
- `mova`: `1`
- `movs`: `2`
- `movx`: `4`
- `movxm`: `1`
- `nop`: `29`
- `nopb`: `1`
- `st.s16`: `8`
- `vadd.f`: `40`
- `vconv.bf16.fp32`: `8`
- `vextract.16`: `8`
- `vlda.conv.fp32.bf16`: `8`
- `vmov`: `72`
- `vshift`: `32`

### `qkv_call_delay_slots`

- range: `0x1d70..0x1d8e`
- instruction_lines: `6`
- `jl`: `1`
- `mov`: `1`
- `movs`: `2`
- `movxm`: `1`
- `nop`: `2`
- `nopa`: `1`

### `qkv_record_emit`

- range: `0x1d8c..0x1e12`
- instruction_lines: `30`
- `acq`: `1`
- `add.nc`: `2`
- `jnzd`: `1`
- `lda`: `4`
- `lda.s8`: `1`
- `lda.u8`: `1`
- `mov`: `1`
- `movx`: `1`
- `movxm`: `2`
- `nop`: `7`
- `nopa`: `1`
- `rel`: `3`
- `sel.eqz`: `1`
- `st`: `1`
- `st.s8`: `1`
- `vlda`: `2`
- `vst.conv.bf16.fp32`: `2`
- `xor`: `1`

## Interpretation

- Experiment 026 is a valid object-shape proof, but it starts too late for numeric execution.
- A runnable generated kernel must either include an equivalent `0x01f0` prologue or generate a function whose caller sets the same LC/LS/LE/constants/control registers.
- It must also include the phase-body group-sum producer and the post-call compact-record emitter, or reuse the MyLM phase body while replacing only the shared microkernel.
- The next experiment should therefore package `prologue + generated hot loop + record emit` as one tiny direct-QKV program, not call the hot-loop body alone.
