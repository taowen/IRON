<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# D3: End-To-End Boundary Inventory

Status: `pending`

This is required before calling a result a full inference architecture.

Record owner, shape, dtype, bytes, frequency, and measured cost for each
boundary:

```text
tokenizer -> embedding
embedding -> layer0
decode body -> final norm
final norm -> LM head
LM head -> argmax
KV cache read/write
```
