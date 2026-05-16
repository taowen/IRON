<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Qwen3 Megakernel Layout Diagnostic Methods

[Back to method index](diagnostic-methods.md).

## 11. Check ObjectFIFO Consumers In Generated MLIR

Use after adding broadcast or multiple debug/cache drains.

```bash
rg -n "objectfifo @qwen3_rc_" build_qwen3_persistent/*.mlir
```

This found a real bug before runtime: two drains were written against the same
`k_rope` and `v` FIFO consumer, but the MLIR exposed only one shim consumer.
The fix was to make cache the only drain for those streams and verify semantic
values from the cache slice.

Rule:

```text
If two logical consumers are intended, generated MLIR must show two consumer
endpoints or an explicit forward/split/join dataflow. Reusing `.cons()` twice
from the same endpoint is not proof of broadcast.
```

