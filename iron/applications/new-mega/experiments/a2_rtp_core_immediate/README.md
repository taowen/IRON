<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# A2: RTP Replaces A Core Immediate

Status: `blocked by A1 rejected`

A1 showed that changing the RTP value through the current high-level
`inline_ops` path changes generated artifacts. Run A2 only after a real
invocation-time RTP update mechanism is found.

Question:

```text
Can a value that used to be compiled as a core-ELF immediate be read from RTP
instead, with the core ELF staying byte-identical across values?
```

Acceptance:

```text
same artifact for position=26 and position=27
core ELFs byte-identical
runtime output changes according to the RTP value
```
