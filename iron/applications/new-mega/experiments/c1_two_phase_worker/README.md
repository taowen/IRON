<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# C1: Two-Phase Same-Worker Protocol

Status: `pending`

Question:

```text
Can one Worker execute two host-scheduled phases in one dispatch with balanced
ObjectFifo acquire/release behavior?
```

This is required before a universal phase-based Worker design can be considered
real. `task_group` alone only proves DMA task grouping, not Worker phase
handshakes.
