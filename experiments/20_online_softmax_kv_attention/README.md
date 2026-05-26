# Experiment 20: Online Softmax KV Attention

This experiment builds on experiment 19.  It keeps the current-token cache write
and four-plane rounded history scan, but replaces the simplified int32
dot/reduce with tiled online softmax attention.

## What This Proves

- `running_max` and `running_sum` can stay in core-local buffers across multiple
  KV history tiles.
- A later tile can raise the running max and rescale the previous output
  accumulator correctly.
- The output is normalized at the end, matching direct softmax attention.
- Current K/V still writes into the same KV cache BO before history scan.
- The final tile still uses `last_valid` to mask rounded 16-token reads.

## What This Does Not Prove

- FastFlowLM's exact attention tile assignment.
- Q/K/V projection, Q/K norm, RoPE, or O projection.
- GQA fanout for all 32 Q heads.
- A high-performance vectorized softmax kernel.  The C kernel is intentionally
  scalar so the dataflow and online-softmax state contract are easy to inspect.
- The AIE target kernel uses an explicit scalar approximate exponential instead
  of `expf`, because scalar libm is not available in this target path.

## Shape

```text
cache layout = k03 | v03 | k47 | v47
current layout = k03 | v03 | k47 | v47
one plane tile = 16 tokens x 4 heads x 32 dim x f32
worker output = 4 heads x 32 dim x f32
```

## Verification

Run from this directory:

```bash
/var/home/taowen/projects/IRON/.venv/bin/python run_npu.py
```

The script checks both the current-token cache write and the final softmax
attention output against a CPU reference.
