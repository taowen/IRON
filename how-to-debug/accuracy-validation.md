<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Accuracy Validation

Used during SageAttention bring-up.

## Symptom

The original test verified only this relation:

```text
NPU output ~= quantized SageAttention reference
```

That catches IRON dataflow and kernel bugs, but it does not prove that the
quantized attention is close to normal bf16 attention.

## Fix Used

The reference now also computes:

```text
O_full   = softmax(Q K^T / sqrt(d)) V
K_s      = K - mean(K, token_dim)
O_smooth = softmax(Q K_s^T / sqrt(d)) V
O_quant  = softmax(dequant(Q_i8 K_i8^T) / sqrt(d)) V
```

The test checks both:

```text
O_smooth ~= O_full
O_quant  ~= O_full
```

with cosine similarity, relative L1, RMSE, and max absolute error.

## Current Result

For the current `seq_len=256`, `d=64`, synthetic test:

```text
SageAttention Quantization Accuracy:
cos=1.00000238, rel_l1=0.002262, rmse=0.007679, max_abs=0.046875

K-smoothing Invariance:
cos=1.00000703, rel_l1=0.000913, rmse=0.004697, max_abs=0.015625
```

This validates the current synthetic case. It is not yet a model-level accuracy
claim; the next step is to run the same metrics on real layer inputs from a
model using the target head dimensions.
