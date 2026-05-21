<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# B3: Segment-Major Packing For The New Topology

Status: `pending`

Question:

```text
Does segment-major packing remain correct and aligned after changing GEMV shard
count and topology?
```

Acceptance requires explicit manifest offsets, alignment checks, TAP coverage
checks, monotonic pattern validation, and F.linear validation.
