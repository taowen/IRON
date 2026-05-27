# Experiment 64: MyLM Attention-To-O Direct Handoff

Exp39 proved a real projected-Q/current-KV attention fabric, but its attention
result was drained to host. Exp52 proved that an edge replay stream can feed an
O projection without a host-visible attention vector, but it used the older
full-K replay path instead of the exact packetized patch queue from exp63.

Exp64 closes that boundary in the current strongest projection ABI:

- edge/aux tiles produce deterministic 256-bf16 attention-result slices,
- those slices stream directly into the main16 O phase on input channel 0,
- O weights arrive through the exp63 packetized patch queue on input channel 1,
- row1 uses small Q4NX chunk rings, not full-patch residency,
- only the final O projection records drain to host for validation.

This experiment answers the full main16 handoff question directly.

## Contract

- Main columns are fixed to `c2..c5`; edge columns are fixed to `c0/c1/c6/c7`.
- One linked host `MM2S ch0` queue per main column carries two packetized
  MyLM-sized O patches.
- Shim packet IDs route patch0 to memtile `S2MM ch0` and patch1 to memtile
  `S2MM ch1`, preserving the legal BD-bank split found in exp62/63.
- Each edge tile replays one deterministic attention-result slice per K chunk
  and sends it directly to its paired main tile.
- Each main tile accumulates Q4NX O projection output from the attention slices
  and packetized O weights.

## What This Proves

- The attention-result-to-O boundary can be a direct stream, not a debug drain.
- The exp63 packetized patch ABI composes with a direct edge-to-main O handoff.
- The full main16 O phase can consume both streams continuously through row1
  small chunk rings.

This does not yet prove production attention. The edge producer is still a
deterministic contract kernel, not rounded KV scan, online softmax, or weighted
V. The next experiment should replace this replay producer with the real exp39
attention output stream while keeping the same O handoff ABI.

## Run

```bash
.venv/bin/python experiments/64_mylm_attention_to_o_direct_handoff/run_npu.py
```

Status:

```text
PASS, NPU time 9423.2 us.
```
