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


@pytest.mark.extensive
def test_qwen3_megakernel_lint():
    model = os.environ.get("IRON_QWEN3_0_6B_MODEL")
    if model is None:
        pytest.skip("Set IRON_QWEN3_0_6B_MODEL to run the Qwen3-0.6B megakernel lint")

    test_dir = Path(__file__).parent
    command = [
        sys.executable,
        str(test_dir / "qwen3_megakernel.py"),
        "--model",
        model,
        "--num-layers",
        "1",
        "--max-seq-len",
        "256",
        "--lint-only",
    ]
    subprocess.run(command, check=True)


@pytest.mark.extensive
def test_qwen3_megakernel_one_step_decode():
    model = os.environ.get("IRON_QWEN3_0_6B_MODEL")
    if model is None:
        pytest.skip(
            "Set IRON_QWEN3_0_6B_MODEL to run the Qwen3-0.6B megakernel decode test"
        )

    test_dir = Path(__file__).parent
    command = [
        sys.executable,
        str(test_dir / "qwen3_megakernel.py"),
        "--model",
        model,
        "--num-layers",
        "1",
        "--max-seq-len",
        "256",
        "--verify-one-step",
        "--verify-repeat",
        "2",
    ]
    subprocess.run(command, check=True)


@pytest.mark.extensive
def test_qwen3_persistent_input_rmsnorm():
    model = os.environ.get("IRON_QWEN3_0_6B_MODEL")
    if model is None:
        pytest.skip(
            "Set IRON_QWEN3_0_6B_MODEL to run the Qwen3-0.6B persistent bring-up test"
        )

    test_dir = Path(__file__).parent
    command = [
        sys.executable,
        str(test_dir / "qwen3_persistent.py"),
        "--model",
        model,
        "--stage",
        "input-rmsnorm",
        "--verify",
        "--verify-repeat",
        "2",
    ]
    subprocess.run(command, check=True)


@pytest.mark.extensive
def test_qwen3_persistent_input_rmsnorm_qkv():
    model = os.environ.get("IRON_QWEN3_0_6B_MODEL")
    if model is None:
        pytest.skip(
            "Set IRON_QWEN3_0_6B_MODEL to run the Qwen3-0.6B persistent QKV bring-up test"
        )

    test_dir = Path(__file__).parent
    command = [
        sys.executable,
        str(test_dir / "qwen3_persistent.py"),
        "--model",
        model,
        "--stage",
        "input-rmsnorm-qkv",
        "--verify",
        "--verify-repeat",
        "1",
    ]
    subprocess.run(command, check=True)


@pytest.mark.extensive
def test_qwen3_persistent_input_rmsnorm_qkv_rope_cache():
    model = os.environ.get("IRON_QWEN3_0_6B_MODEL")
    if model is None:
        pytest.skip(
            "Set IRON_QWEN3_0_6B_MODEL to run the Qwen3-0.6B persistent RoPE/cache bring-up test"
        )

    test_dir = Path(__file__).parent
    command = [
        sys.executable,
        str(test_dir / "qwen3_persistent.py"),
        "--model",
        model,
        "--stage",
        "input-rmsnorm-qkv-rope-cache",
        "--verify",
        "--verify-repeat",
        "1",
    ]
    subprocess.run(command, check=True)


@pytest.mark.extensive
@pytest.mark.xfail(
    reason="score/softmax persistent checkpoint compiles and runs but numeric verification is not accepted yet",
    strict=True,
)
def test_qwen3_persistent_input_rmsnorm_qkv_rope_cache_scores_softmax():
    model = os.environ.get("IRON_QWEN3_0_6B_MODEL")
    if model is None:
        pytest.skip(
            "Set IRON_QWEN3_0_6B_MODEL to run the Qwen3-0.6B persistent attention-score bring-up test"
        )

    test_dir = Path(__file__).parent
    command = [
        sys.executable,
        str(test_dir / "qwen3_persistent.py"),
        "--model",
        model,
        "--stage",
        "input-rmsnorm-qkv-rope-cache-scores-softmax",
        "--verify",
        "--verify-repeat",
        "1",
    ]
    subprocess.run(command, check=True)
