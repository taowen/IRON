<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# A3: Dynamic Current-K/V Offset Proof

Status: `blocked by A1 rejected`

Depends on A1 and A2. A1 currently rejects the high-level RTP path as a
dynamic-position mechanism, so this experiment is blocked until a different
runtime update path exists.

Acceptance:

```text
same xclbin and runtime .bin for p and p+1
or a documented safe runtime patch/update path
cache readback matches a monotonic pattern
no future-token read and no overlapping writes
```
