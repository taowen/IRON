<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Runtime Isolation for Performance Tests

Used during SageAttention bring-up.

## Symptom

Running the same pytest with multiple iterations produced a large SageAttention
latency outlier:

```text
SageAttention Latency (us): 928.6
MHA Latency (us): 439.4
Speedup: 0.473x
```

The same code path still produced correct output, and neighboring runs were
hundreds of microseconds faster.

## Cause

The test loaded SageAttention, then cleaned up before loading the MHA baseline.
It did not clean up before the next pytest iteration's SageAttention run, and
it did not clean up after the MHA run if the assertion failed.

This made the benchmark sensitive to previous runtime state. The symptom was a
performance failure, not a numerical failure.

## Fix Used

Clean up the default NPU runtime at the start of the test and again before
assertions:

```python
def test_sage_attention_faster_than_mha(seq_len, dim, aie_context):
    aie_utils.DefaultNPURuntime.cleanup()
    ...
    print(f"Speedup: {mha_latency_us / sage_latency_us:.3f}x\n")

    aie_utils.DefaultNPURuntime.cleanup()

    assert not sage_errors
    assert not mha_errors
    assert sage_latency_us < mha_latency_us
```

The cleanup before assertions matters because a failing assertion would skip
any cleanup placed after it.

## Follow-up

The benchmark also moved from five timed iterations to thirty timed iterations.
Five iterations were enough for a smoke test, but not stable enough for a
performance assertion when Sage and MHA were close at `seq_len=128`.
