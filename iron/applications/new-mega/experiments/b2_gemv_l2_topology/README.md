<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# B2: GEMV L2 Split/Forward/Join Topology

Status: `pending`

Question:

```text
Can GEMV use an L3->L2->L1 topology so runtime endpoint count scales with
columns instead of compute tiles?
```

This must be proven for GEMV shape specifically. GEMM's L2 topology does not
automatically transfer because GEMV has a broadcast input vector and row-sharded
weights.
