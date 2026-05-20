#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from iron.applications.qwen3_0_6b.persistent.ops import (
    Qwen3PersistentNLayerFinalOnly,
)
from iron.applications.qwen3_0_6b.persistent.graph_probe import layer_groups
from iron.applications.qwen3_0_6b.persistent.layout import (
    PACKED_WEIGHTS_BIN,
    PACKED_WEIGHTS_MANIFEST,
    load_packed_weight_tensor,
    pack_full_layer_weights_for_layer,
    packed_weight_layer_slice,
    validate_packed_weight_artifact,
    write_packed_weight_artifact,
)
from iron.applications.qwen3_0_6b.qwen3_cpu import DEFAULT_MODEL, resolve_model_dir
from iron.applications.qwen3_0_6b.qwen3_preflight import (
    Qwen3PreflightError,
    run_persistent_artifact_preflight,
)


def _fake_qwen3_model(num_layers=3):
    config = SimpleNamespace(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=num_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        rms_norm_eps=1e-6,
        rope_theta=1000000.0,
        vocab_size=128,
        tie_word_embeddings=True,
        torch_dtype="bfloat16",
    )
    weights = {}
    counter = 0

    def make(shape):
        nonlocal counter
        numel = math.prod(shape)
        tensor = (
            torch.arange(counter, counter + numel, dtype=torch.float32)
            .reshape(shape)
            .to(torch.bfloat16)
        )
        counter += numel + 17
        return tensor

    for layer_idx in range(num_layers):
        layer = f"model.layers.{layer_idx}"
        attn = f"{layer}.self_attn"
        mlp = f"{layer}.mlp"
        weights[f"{layer}.input_layernorm.weight"] = make((config.hidden_size,))
        weights[f"{attn}.q_proj.weight"] = make(
            (config.num_attention_heads * config.head_dim, config.hidden_size)
        )
        weights[f"{attn}.k_proj.weight"] = make(
            (config.num_key_value_heads * config.head_dim, config.hidden_size)
        )
        weights[f"{attn}.v_proj.weight"] = make(
            (config.num_key_value_heads * config.head_dim, config.hidden_size)
        )
        weights[f"{attn}.o_proj.weight"] = make(
            (config.hidden_size, config.num_attention_heads * config.head_dim)
        )
        weights[f"{attn}.q_norm.weight"] = make((config.head_dim,))
        weights[f"{attn}.k_norm.weight"] = make((config.head_dim,))
        weights[f"{layer}.post_attention_layernorm.weight"] = make(
            (config.hidden_size,)
        )
        weights[f"{mlp}.gate_proj.weight"] = make(
            (config.intermediate_size, config.hidden_size)
        )
        weights[f"{mlp}.up_proj.weight"] = make(
            (config.intermediate_size, config.hidden_size)
        )
        weights[f"{mlp}.down_proj.weight"] = make(
            (config.hidden_size, config.intermediate_size)
        )

    class FakeModel:
        def __init__(self):
            self.config = config
            self.dtype = torch.bfloat16

        def w(self, name):
            return weights[name]

    return FakeModel()


def test_qwen3_resolve_model_dir_prefers_local_hf_snapshot(tmp_path, monkeypatch):
    commit = "abc123"
    repo_cache = tmp_path / "hub" / "models--Qwen--Qwen3-0.6B"
    snapshot = repo_cache / "snapshots" / commit
    snapshot.mkdir(parents=True)
    for name in [
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
        "model.safetensors",
    ]:
        (snapshot / name).write_text("{}")
    refs = repo_cache / "refs"
    refs.mkdir()
    (refs / "main").write_text(commit)

    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path / "hub"))

    assert resolve_model_dir(DEFAULT_MODEL) == snapshot


def test_qwen3_packed_weight_artifact_roundtrip(tmp_path):
    model = _fake_qwen3_model(num_layers=3)
    expected_per_layer = pack_full_layer_weights_for_layer(model, 0).numel()
    manifest = write_packed_weight_artifact(
        model,
        tmp_path,
        expected_per_layer_numel=expected_per_layer,
    )

    assert (tmp_path / PACKED_WEIGHTS_BIN).exists()
    assert (tmp_path / PACKED_WEIGHTS_MANIFEST).exists()
    assert manifest["per_layer_numel"] == expected_per_layer
    assert (
        manifest["total_numel"] == expected_per_layer * model.config.num_hidden_layers
    )

    loaded_manifest = validate_packed_weight_artifact(
        model,
        tmp_path,
        expected_per_layer_numel=expected_per_layer,
    )
    packed = load_packed_weight_tensor(tmp_path, loaded_manifest)
    for layer_idx in range(model.config.num_hidden_layers):
        actual = packed_weight_layer_slice(packed, loaded_manifest, layer_idx)
        expected = pack_full_layer_weights_for_layer(model, layer_idx)
        assert torch.equal(actual, expected)


def test_qwen3_packed_weight_manifest_fails_fast(tmp_path):
    model = _fake_qwen3_model(num_layers=2)
    expected_per_layer = pack_full_layer_weights_for_layer(model, 0).numel()
    manifest = write_packed_weight_artifact(
        model,
        tmp_path,
        expected_per_layer_numel=expected_per_layer,
    )
    manifest["weight_order"] = list(reversed(manifest["weight_order"]))
    (tmp_path / PACKED_WEIGHTS_MANIFEST).write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="weight order"):
        validate_packed_weight_artifact(
            model,
            tmp_path,
            expected_per_layer_numel=expected_per_layer,
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


def test_qwen3_preflight_catches_probe_bd_boundary(tmp_path):
    mlir_path = tmp_path / "bad_probe_bd.mlir"
    dma_tasks = "\n".join(
        f"      %{idx} = aiex.dma_configure_task_for @current_kv {{ aie.end }}"
        for idx in range(9)
    )
    mlir_path.write_text(f"""
module {{
  aie.device(npu2) {{
    %tile_0_2 = aie.tile(0, 2)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.objectfifo @current_kv(%shim_noc_tile_0_0, {{%tile_0_2}}, 2 : i32) : !aie.objectfifo<memref<128xbf16>>
    aie.runtime_sequence(%arg0: memref<128xbf16>) {{
{dma_tasks}
    }}
  }}
}}
""")

    with pytest.raises(Qwen3PreflightError, match="9 DMA tasks"):
        run_persistent_artifact_preflight(mlir_path=mlir_path, arg_specs=1)


def test_qwen3_graph_probe_layer_groups():
    assert layer_groups(8, 4) == [(0, 4), (4, 4)]
    assert layer_groups(10, 4) == [(0, 4), (4, 4), (8, 2)]

    with pytest.raises(ValueError, match="layers must be positive"):
        layer_groups(0, 4)
    with pytest.raises(ValueError, match="group_layers must be positive"):
        layer_groups(4, 0)


def test_qwen3_n_layer_final_only_rejects_unsupported_chunk():
    Qwen3PersistentNLayerFinalOnly(layer_iterations=7)

    with pytest.raises(ValueError, match="at most 7 layers per chunk"):
        Qwen3PersistentNLayerFinalOnly(layer_iterations=8)


def test_qwen3_preflight_catches_non_advancing_acquire(tmp_path):
    mlir_path = tmp_path / "bad_acquire.mlir"
    mlir_path.write_text("""
module {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %shim_noc_tile_0_0 = aie.tile(0, 0)
    aie.objectfifo @scores(%tile_0_2, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<256xbf16>>
    aie.runtime_sequence(%arg0: memref<256xbf16>) {
      aie.end
    }
    %core_0_2 = aie.core(%tile_0_2) {
      %0 = aie.objectfifo.acquire @scores(Produce, 1) : !aie.objectfifosubview<memref<256xbf16>>
      %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<256xbf16>> -> memref<256xbf16>
      %2 = aie.objectfifo.acquire @scores(Produce, 1) : !aie.objectfifosubview<memref<256xbf16>>
      %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<256xbf16>> -> memref<256xbf16>
      aie.objectfifo.release @scores(Produce, 1)
      aie.objectfifo.release @scores(Produce, 1)
      aie.end
    }
  }
}
""")

    with pytest.raises(Qwen3PreflightError, match="does not advance"):
        run_persistent_artifact_preflight(mlir_path=mlir_path, arg_specs=1)


def test_qwen3_preflight_catches_o_proj_gemv_symbol_reuse(tmp_path):
    mlir_path = tmp_path / "bad_o_proj_gemv.mlir"
    mlir_path.write_text("""
module {
  aie.device(npu2) {
    func.func private @matvec_vectorized_bf16_bf16(i32, i32, memref<4x2048xbf16>, memref<2048xbf16>, memref<128xbf16>) attributes {link_with = "qwen3_persistent_gemv_2048k_64vs_o_proj.o"}
    aie.runtime_sequence(%arg0: memref<128xbf16>) {
      aie.end
    }
  }
}
""")

    with pytest.raises(Qwen3PreflightError, match="DIM_K=2048"):
        run_persistent_artifact_preflight(mlir_path=mlir_path, arg_specs=1)


@pytest.mark.extensive
def test_qwen3_megakernel_lint():
    model = os.environ.get("IRON_QWEN3_0_6B_MODEL")
    if model is None:
        pytest.skip("Set IRON_QWEN3_0_6B_MODEL to run the Qwen3-0.6B megakernel lint")

    test_dir = Path(__file__).parent
    command = [
        sys.executable,
        str(test_dir / "full_elf" / "main.py"),
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
        str(test_dir / "full_elf" / "main.py"),
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
        str(test_dir / "persistent" / "main.py"),
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
        str(test_dir / "persistent" / "main.py"),
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
        str(test_dir / "persistent" / "main.py"),
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
def test_qwen3_persistent_input_rmsnorm_qkv_rope_cache_scores_softmax():
    model = os.environ.get("IRON_QWEN3_0_6B_MODEL")
    if model is None:
        pytest.skip(
            "Set IRON_QWEN3_0_6B_MODEL to run the Qwen3-0.6B persistent attention-score bring-up test"
        )

    test_dir = Path(__file__).parent
    command = [
        sys.executable,
        str(test_dir / "persistent" / "main.py"),
        "--model",
        model,
        "--stage",
        "input-rmsnorm-qkv-rope-cache-scores-softmax",
        "--verify",
        "--verify-repeat",
        "1",
    ]
    subprocess.run(command, check=True)


@pytest.mark.extensive
def test_qwen3_persistent_input_rmsnorm_qkv_rope_cache_scores_softmax_context():
    model = os.environ.get("IRON_QWEN3_0_6B_MODEL")
    if model is None:
        pytest.skip(
            "Set IRON_QWEN3_0_6B_MODEL to run the Qwen3-0.6B persistent attention-context bring-up test"
        )

    test_dir = Path(__file__).parent
    command = [
        sys.executable,
        str(test_dir / "persistent" / "main.py"),
        "--model",
        model,
        "--stage",
        "input-rmsnorm-qkv-rope-cache-scores-softmax-context",
        "--verify",
        "--verify-repeat",
        "1",
    ]
    subprocess.run(command, check=True)


@pytest.mark.extensive
def test_qwen3_persistent_input_rmsnorm_qkv_rope_cache_scores_softmax_context_o_proj():
    model = os.environ.get("IRON_QWEN3_0_6B_MODEL")
    if model is None:
        pytest.skip(
            "Set IRON_QWEN3_0_6B_MODEL to run the Qwen3-0.6B persistent attention O projection bring-up test"
        )

    test_dir = Path(__file__).parent
    command = [
        sys.executable,
        str(test_dir / "persistent" / "main.py"),
        "--model",
        model,
        "--stage",
        "input-rmsnorm-qkv-rope-cache-scores-softmax-context-o-proj",
        "--verify",
        "--verify-repeat",
        "1",
    ]
    subprocess.run(command, check=True)


@pytest.mark.extensive
def test_qwen3_persistent_post_attn_rmsnorm_mlp_gate_up():
    model = os.environ.get("IRON_QWEN3_0_6B_MODEL")
    if model is None:
        pytest.skip(
            "Set IRON_QWEN3_0_6B_MODEL to run the Qwen3-0.6B persistent MLP gate/up bring-up test"
        )

    test_dir = Path(__file__).parent
    command = [
        sys.executable,
        str(test_dir / "persistent" / "main.py"),
        "--model",
        model,
        "--stage",
        "post-attn-rmsnorm-mlp-gate-up",
        "--verify",
        "--verify-repeat",
        "1",
    ]
    subprocess.run(command, check=True)


@pytest.mark.extensive
def test_qwen3_persistent_post_attn_mlp_down_residual():
    model = os.environ.get("IRON_QWEN3_0_6B_MODEL")
    if model is None:
        pytest.skip(
            "Set IRON_QWEN3_0_6B_MODEL to run the Qwen3-0.6B persistent MLP down/residual bring-up test"
        )

    test_dir = Path(__file__).parent
    command = [
        sys.executable,
        str(test_dir / "persistent" / "main.py"),
        "--model",
        model,
        "--stage",
        "post-attn-mlp-down-residual",
        "--verify",
        "--verify-repeat",
        "1",
    ]
    subprocess.run(command, check=True)


@pytest.mark.extensive
def test_qwen3_persistent_post_attn_rmsnorm_full_mlp():
    model = os.environ.get("IRON_QWEN3_0_6B_MODEL")
    if model is None:
        pytest.skip(
            "Set IRON_QWEN3_0_6B_MODEL to run the Qwen3-0.6B persistent full MLP bring-up test"
        )

    test_dir = Path(__file__).parent
    command = [
        sys.executable,
        str(test_dir / "persistent" / "main.py"),
        "--model",
        model,
        "--stage",
        "post-attn-rmsnorm-full-mlp",
        "--verify",
        "--verify-repeat",
        "1",
    ]
    subprocess.run(command, check=True)


@pytest.mark.extensive
def test_qwen3_persistent_full_layer():
    model = os.environ.get("IRON_QWEN3_0_6B_MODEL")
    if model is None:
        pytest.skip(
            "Set IRON_QWEN3_0_6B_MODEL to run the Qwen3-0.6B persistent full-layer bring-up test"
        )

    test_dir = Path(__file__).parent
    command = [
        sys.executable,
        str(test_dir / "persistent" / "main.py"),
        "--model",
        model,
        "--stage",
        "input-rmsnorm-qkv-rope-cache-scores-softmax-context-o-proj-full-mlp",
        "--verify",
        "--verify-repeat",
        "1",
    ]
    subprocess.run(command, check=True)


@pytest.mark.extensive
@pytest.mark.parametrize("layer_chunk_size", [1, 2, 4, 7])
def test_qwen3_persistent_n_layer_final_only(layer_chunk_size):
    model = os.environ.get("IRON_QWEN3_0_6B_MODEL")
    if model is None:
        pytest.skip(
            "Set IRON_QWEN3_0_6B_MODEL to run the Qwen3-0.6B persistent n-layer final-only test"
        )

    test_dir = Path(__file__).parent
    command = [
        sys.executable,
        str(test_dir / "persistent" / "main.py"),
        "--model",
        model,
        "--stage",
        "n-layer-final-only",
        "--layer-chunk-size",
        str(layer_chunk_size),
        "--verify",
        "--verify-repeat",
        "1",
    ]
    subprocess.run(command, check=True)


@pytest.mark.extensive
def test_qwen3_persistent_multi_layer_full_layer():
    model = os.environ.get("IRON_QWEN3_0_6B_MODEL")
    if model is None:
        pytest.skip(
            "Set IRON_QWEN3_0_6B_MODEL to run the Qwen3-0.6B persistent multi-layer bring-up test"
        )

    test_dir = Path(__file__).parent
    command = [
        sys.executable,
        str(test_dir / "persistent" / "main.py"),
        "--model",
        model,
        "--stage",
        "multi-layer-full-layer",
        "--num-layers",
        "2",
        "--verify",
        "--verify-repeat",
        "1",
    ]
    subprocess.run(command, check=True)


@pytest.mark.extensive
def test_qwen3_persistent_fast_generate(tmp_path):
    model = os.environ.get("IRON_QWEN3_0_6B_MODEL")
    if model is None:
        pytest.skip(
            "Set IRON_QWEN3_0_6B_MODEL to run the Qwen3-0.6B fast generate test"
        )

    test_dir = Path(__file__).parent
    packed_dir = tmp_path / "qwen3_iron_packed"
    prepare_command = [
        sys.executable,
        str(test_dir / "persistent" / "main.py"),
        "--model",
        model,
        "--prepare-weights",
        "--packed-weights-dir",
        str(packed_dir),
    ]
    subprocess.run(prepare_command, check=True)

    command = [
        sys.executable,
        str(test_dir / "persistent" / "main.py"),
        "--model",
        model,
        "--stage",
        "generate",
        "--fast-generate",
        "--verify-generate",
        "--max-new-tokens",
        "2",
        "--packed-weights-dir",
        str(packed_dir),
        "--require-packed-weights",
    ]
    subprocess.run(command, check=True)
