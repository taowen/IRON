#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import subprocess
import sys
from pathlib import Path

import pytest

from iron.applications.qwen3_0_6b.qwen3_preflight import (
    Qwen3PreflightError,
    run_persistent_artifact_preflight,
)


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


def test_qwen3_preflight_catches_runtime_bo_mismatch(tmp_path):
    mlir_path = tmp_path / "bad.mlir"
    mlir_path.write_text("""
module {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.objectfifo @in(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<128xbf16>>
    aie.runtime_sequence(%arg0: memref<128xbf16>, %arg1: memref<128xbf16>, %arg2: memref<128xbf16>) {
      aie.end
    }
  }
}
""")
    prj_dir = Path(str(mlir_path) + ".prj")
    prj_dir.mkdir()
    (prj_dir / "main_kernels.json").write_text(
        json.dumps(
            {
                "ps-kernels": {
                    "kernels": [
                        {
                            "arguments": [
                                {"name": "bo0", "memory-connection": "HOST"},
                                {"name": "bo1", "memory-connection": "HOST"},
                            ]
                        }
                    ]
                }
            }
        )
    )

    with pytest.raises(Qwen3PreflightError, match="Runtime BO metadata mismatch"):
        run_persistent_artifact_preflight(mlir_path=mlir_path, arg_specs=3)


def test_qwen3_preflight_catches_fifo_l1_overuse(tmp_path):
    mlir_path = tmp_path / "bad_l1.mlir"
    mlir_path.write_text("""
module {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.objectfifo @k_cache(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<256x128xbf16>>
    aie.runtime_sequence(%arg0: memref<8388608xbf16>) {
      %0 = aiex.dma_configure_task_for @k_cache {
        aie.end
      }
    }
  }
}
""")

    with pytest.raises(Qwen3PreflightError, match="ObjectFIFO L1 budget mismatch"):
        run_persistent_artifact_preflight(mlir_path=mlir_path, arg_specs=1)


def test_qwen3_preflight_catches_tile_input_overuse(tmp_path):
    mlir_path = tmp_path / "bad_inputs.mlir"
    mlir_path.write_text("""
module {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.objectfifo @in0(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<128xbf16>>
    aie.objectfifo @in1(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<128xbf16>>
    aie.objectfifo @in2(%shim_noc_tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<128xbf16>>
    aie.runtime_sequence(%arg0: memref<128xbf16>) {
      aie.end
    }
  }
}
""")

    with pytest.raises(Qwen3PreflightError, match="3 input ObjectFIFOs"):
        run_persistent_artifact_preflight(mlir_path=mlir_path, arg_specs=1)


def test_qwen3_preflight_catches_dma_task_overuse(tmp_path):
    mlir_path = tmp_path / "bad_dma.mlir"
    dma_tasks = "\n".join(
        f"      %{idx} = aiex.dma_configure_task_for @k_cache {{ aie.end }}"
        for idx in range(33)
    )
    mlir_path.write_text(f"""
module {{
  aie.device(npu2) {{
    %tile_0_2 = aie.tile(0, 2)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.objectfifo @k_cache(%shim_noc_tile_0_0, {{%tile_0_2}}, 2 : i32) : !aie.objectfifo<memref<128xbf16>>
    aie.runtime_sequence(%arg0: memref<128xbf16>) {{
{dma_tasks}
    }}
  }}
}}
""")

    with pytest.raises(Qwen3PreflightError, match="33 DMA tasks"):
        run_persistent_artifact_preflight(mlir_path=mlir_path, arg_specs=1)


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
