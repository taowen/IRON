<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# A4: Attention Same-Artifact Two-Position Proof

Status: `blocked by A1 rejected`

Depends on A1 through A3. Since A1 rejected the current high-level RTP path,
this remains blocked until a different same-artifact position mechanism is
available.

Acceptance:

```text
same artifact runs p and p+1
attention output matches PyTorch reference at both positions
softmax row sums are valid
current K/V cache slices match reference
artifact diff shows no unproven position-specific change
```
