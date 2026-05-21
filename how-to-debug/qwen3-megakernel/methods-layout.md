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

## 51. Treat Packet Header Changes As ABI Changes

Use when a phase-owned packet grows a metadata/residual header before a large
weight block.

Real failure caught in production D1.5v:

```text
O packet residual header changed from:
  residual_rows = q_rows_per_packet

to:
  residual_rows = fabric_group_size * q_rows_per_packet
```

The first edits updated the Python packet builder and O finalize path, but two
old assumptions remained:

```text
O partial C++ kernel weight pointer:
  packet + 2 + q_rows_per_packet

packet-level reference producer weight-block offset:
  2 + q_rows_per_packet
```

Both had to become:

```text
packet + 2 + fabric_group_size * q_rows_per_packet
```

Diagnostic checklist:

```text
1. Write the packet layout as fields, not just a total `packet_elements`.
2. Search every `+ constant` and `+ q_rows_per_packet` offset in Python and C++.
3. Update C++ signature and Python `Kernel(...)` declaration in the same patch.
4. Update packet builder, packet-level reference, semantic reference, and README
   shape printouts together.
5. Re-run `resolve_program()` before interpreting any numeric mismatch.
```

Rule:

```text
In a phase-owned megakernel, packet layout is an ABI. A passing compile with an
old offset can still be a wrong program, because the kernel may read residual
values as weights or weights as residual values without crashing.
```
