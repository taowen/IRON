# 011 Q4NX Chunk Pair Matrix

Experiment 010 showed both activation and weight axes are even-only in the
direct MyLM Q/K/V harness.

This experiment checks pairing. It activates one activation chunk and one weight
chunk at a time for a small matrix:

```text
activation chunk in 0..3
weight chunk     in 0..3
```

If only matching even/even pairs contribute, the direct phase body is consuming
paired ping-side chunks. If cross even/even pairs contribute, the Q4NX body is
mixing chunks in a way that the host-side "chunk index" abstraction does not
capture.

