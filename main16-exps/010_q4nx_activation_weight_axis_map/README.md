# 010 Q4NX Activation/Weight Axis Map

Experiment 009 showed that single active chunks contribute only on even chunk
indices:

```text
chunk 0,2,4,...,14 -> bf16 4
chunk 1,3,5,...,15 -> bf16 0
```

This experiment separates the two input axes:

- activation single chunk, weight all chunks;
- activation all chunks, weight single chunk.

The goal is to identify whether the even/odd effect belongs to the activation
stream, the weight stream, or their pairing in the direct MyLM Q/K/V body.

