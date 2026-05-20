#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

N_LAYER_FINAL_ONLY_STAGE = "n-layer-final-only"
GENERATE_STAGE = "generate"

STAGE_CHOICES = [
    "input-rmsnorm",
    "input-rmsnorm-qkv",
    N_LAYER_FINAL_ONLY_STAGE,
    GENERATE_STAGE,
    "post-attn-rmsnorm-mlp-gate-up",
    "post-attn-mlp-down-residual",
    "post-attn-rmsnorm-full-mlp",
]
