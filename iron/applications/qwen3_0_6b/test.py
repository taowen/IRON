#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.extensive
def test_qwen3_0_6b_matches_hf_reference():
    model = os.environ.get("IRON_QWEN3_0_6B_MODEL")
    if model is None:
        pytest.skip("Set IRON_QWEN3_0_6B_MODEL to run the Qwen3-0.6B inference test")

    test_dir = Path(__file__).parent
    command = [
        sys.executable,
        str(test_dir / "qwen3_cpu.py"),
        "--model",
        model,
        "--prompt",
        "What is the capital of France? Answer with only the city name.",
        "--max-new-tokens",
        "4",
        "--verify-hf",
    ]
    subprocess.run(command, check=True)
