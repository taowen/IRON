# Experiment 100: MyLM Q4NX Body Contract

This experiment defines the next performance target for main16 Q4NX work.
It does not change the active `qwen3-layer` implementation.

The point is to stop tuning the current exact-rounding body locally and make the
MyLM-style body requirements executable:

- MyLM hot loop has `lc=2`, zero hot-loop `vst`, `528` dynamic `vmac.f` per
  32x256 chunk, and a scheduled dequant path with `64` `vunpack`, `64`
  `vups.4x`, and `136` `vconv.bf16.fp32` static slots.
- The phase body produces eight activation group sums before calling the Q4NX
  hot loop.
- The hot loop uses the group-sum correction form, so it is not bit-exact with
  the current IRON reference.
- Any production migration must therefore be validated by the single-layer/full
  decode numeric gates, not by a per-chunk exact comparison.

Run:

```bash
python3 experiments/100_mylm_q4nx_body_contract/run.py
```

Passing means the reverse-engineered MyLM contract is internally consistent and
the known group-sum numerical delta stays below `1e-2` on deterministic Q4NX
samples. It does not mean the body is implemented in production yet.
