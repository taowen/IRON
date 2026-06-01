# 008 Q4NX Payload Formula Probe

This experiment extends `005_nonzero_payload_probe`.

Experiment 005 proved that MyLM direct Q/K/V phase-body entry produces non-zero
compact payloads when fed deterministic activation and Q4NX streams. This
experiment checks whether the simple synthetic cases obey the expected dot
formula:

```text
payload = active_chunks_per_record * 128 * activation_bf16 * scale_bf16 * q4_nibble
```

The probe is intentionally narrow:

- all active activation words are a uniform bf16 value;
- all active Q4NX chunks use one uniform scale and nibble;
- zero points stay zero;
- each record has 16 stream chunks;
- the effective contribution per active chunk is tested as 128 bf16 elements.

It also tests chunk/record locality:

- all chunks active;
- only the first record's 16 chunks active;
- only the first chunk active.

If these cases pass, we have a reliable scalar anchor for the MyLM Q4NX
payload/reference work. It still does not prove the production packed layout,
per-lane scale layout, or zero-point handling.

