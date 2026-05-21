<!--
SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# C2: Inactive FIFO / Phase Skip Protocol

Status: `blocked`

Depends on C1. This experiment checks whether a Worker can skip a phase without
requiring dummy DMA tokens on inactive FIFOs.

If dummy tokens are required, the phase design still consumes the scarce
endpoint/BD resources it was supposed to save.
