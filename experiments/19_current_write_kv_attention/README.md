# Experiment 19: Current Write + Four-Plane KV Attention Scan

This experiment is the retained attention/KV contract test. It consolidates the
useful lessons from earlier KV-cache prototypes and is the attention/KV
counterpart to experiment 18.

## What This Proves

- Current K/V is not assumed to already be in history. It arrives in a separate
  BO and is streamed through the AIE graph.
- Workers write current K/V back into one KV cache BO at token `L - 1`.
- The runtime synchronizes those cache writes before issuing history reads.
- History reads use the rounded `ceil(L / 16)` tile count.
- The attention worker masks the final 16-token tile with `last_valid`.
- Four cache planes are carried independently: `k03`, `v03`, `k47`, `v47`.

## What This Does Not Prove

- FastFlowLM's exact edge/aux tile attention mapping.
- Real softmax with running max/sum.
- Current K/V bypass optimization.
- Full Qwen layer integration with Q/K/V projection, RoPE, O projection, and FFN.

## Scaled-Down Shape

The real MyLM/FastFlowLM cache plane uses 4 KV heads x 128 dim.  This experiment
uses 4 KV heads x 32 dim with int32 values so the dataflow is easy to debug:

```text
one plane tile = 16 tokens x 4 heads x 32 dim = 2048 dwords
one current token per plane = 4 heads x 32 dim = 128 dwords
cache layout = k03 | v03 | k47 | v47
current layout = k03 | v03 | k47 | v47
```

## Verification

Run from this directory with the IRON environment:

```bash
/var/home/taowen/projects/IRON/.venv/bin/python run_npu.py
```

The script checks both observable effects:

- the current token slice inside the KV cache BO was updated by NPU execution
- the final attention output matches the CPU reference that reads the updated
  cache
