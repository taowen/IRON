#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path

import numpy as np
import torch

from iron.applications.qwen3_0_6b.qwen3_cpu import Qwen3ForCausalLM
from iron.applications.qwen3_0_6b.persistent.refs import rope_lut_for_position

PACKED_WEIGHTS_FORMAT = "qwen3_iron_packed_weights_v1"
PACKED_WEIGHTS_BIN = "weights.bf16.bin"
PACKED_WEIGHTS_MANIFEST = "manifest.json"
FULL_LAYER_WEIGHT_ORDER = (
    "input_norm_weight",
    "W_q",
    "W_k",
    "W_v",
    "W_q_norm",
    "W_k_norm",
    "W_o",
    "post_norm_weight",
    "W_gate",
    "W_up",
    "W_down",
)


def host_owned_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().clone().contiguous()


def pack_layer_cache(state, layer_idx: int) -> torch.Tensor:
    return torch.cat(
        [state.keys[layer_idx].flatten(), state.values[layer_idx].flatten()]
    ).contiguous()


def build_full_layer_weight_inputs_for_layer(
    model: Qwen3ForCausalLM,
    layer_idx: int,
) -> dict[str, torch.Tensor]:
    layer = f"model.layers.{layer_idx}"
    attn = f"{layer}.self_attn"
    mlp = f"{layer}.mlp"
    return {
        "input_norm_weight": model.w(f"{layer}.input_layernorm.weight")
        .flatten()
        .contiguous(),
        "W_q": model.w(f"{attn}.q_proj.weight").contiguous(),
        "W_k": model.w(f"{attn}.k_proj.weight").contiguous(),
        "W_v": model.w(f"{attn}.v_proj.weight").contiguous(),
        "W_o": model.w(f"{attn}.o_proj.weight").contiguous(),
        "W_q_norm": model.w(f"{attn}.q_norm.weight").flatten().contiguous(),
        "W_k_norm": model.w(f"{attn}.k_norm.weight").flatten().contiguous(),
        "post_norm_weight": model.w(f"{layer}.post_attention_layernorm.weight")
        .flatten()
        .contiguous(),
        "W_gate": model.w(f"{mlp}.gate_proj.weight").contiguous(),
        "W_up": model.w(f"{mlp}.up_proj.weight").contiguous(),
        "W_down": model.w(f"{mlp}.down_proj.weight").contiguous(),
    }


def build_full_layer_inputs_for_layer(
    model: Qwen3ForCausalLM,
    layer_idx: int,
    hidden: torch.Tensor,
    state,
) -> dict[str, torch.Tensor]:
    return {
        "hidden": hidden.flatten().contiguous(),
        **build_full_layer_weight_inputs_for_layer(model, layer_idx),
        "rope_angles": rope_lut_for_position(
            model.config.head_dim,
            model.config.rope_theta,
            state.position,
        ),
        "initial_cache": pack_layer_cache(state, layer_idx),
        "initial_keys_cache": state.keys[layer_idx].contiguous(),
        "initial_values_cache": state.values[layer_idx].contiguous(),
    }


def pack_full_layer_weights(inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [inputs[name].flatten() for name in FULL_LAYER_WEIGHT_ORDER]
    ).contiguous()


def pack_full_layer_weights_for_layer(
    model: Qwen3ForCausalLM,
    layer_idx: int,
) -> torch.Tensor:
    return pack_full_layer_weights(
        build_full_layer_weight_inputs_for_layer(model, layer_idx)
    )


def default_packed_weights_dir(model_dir: Path) -> Path:
    return Path(model_dir) / "qwen3_iron_packed"


def _config_dict(model: Qwen3ForCausalLM) -> dict[str, object]:
    config = model.config
    if is_dataclass(config):
        raw = asdict(config)
    else:
        raw = dict(vars(config))
    keys = (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "head_dim",
        "rms_norm_eps",
        "rope_theta",
        "vocab_size",
        "tie_word_embeddings",
        "torch_dtype",
    )
    return {key: raw[key] for key in keys if key in raw}


def _bf16_to_file(tensor: torch.Tensor, path: Path) -> None:
    if tensor.dtype != torch.bfloat16:
        tensor = tensor.to(torch.bfloat16)
    tensor = tensor.contiguous()
    raw = tensor.view(torch.uint16).cpu().numpy()
    raw.tofile(path)


def _bf16_from_file(path: Path, numel: int) -> torch.Tensor:
    raw = np.fromfile(path, dtype=np.uint16)
    if raw.size != numel:
        raise RuntimeError(
            f"packed weight file has {raw.size} bf16 elements, expected {numel}"
        )
    return torch.from_numpy(raw.copy()).view(torch.bfloat16)


def _byte_alignment(byte_offset: int) -> int | None:
    if byte_offset == 0:
        return None
    return byte_offset & -byte_offset


def build_full_layer_weight_manifest_for_layer(
    model: Qwen3ForCausalLM,
    layer_idx: int,
    layer_base_offset: int,
) -> tuple[list[dict[str, object]], int]:
    inputs = build_full_layer_weight_inputs_for_layer(model, layer_idx)
    offset = layer_base_offset
    segments: list[dict[str, object]] = []
    for name in FULL_LAYER_WEIGHT_ORDER:
        tensor = inputs[name]
        numel = tensor.numel()
        segments.append(
            {
                "name": name,
                "shape": list(tensor.shape),
                "dtype": "bfloat16",
                "element_offset": offset,
                "numel": numel,
                "byte_offset": offset * 2,
                "byte_length": numel * 2,
                "byte_alignment": _byte_alignment(offset * 2),
            }
        )
        offset += numel
    return segments, offset


def build_packed_weights_manifest(
    model: Qwen3ForCausalLM,
    per_layer_numel: int,
) -> dict[str, object]:
    layers = []
    offset = 0
    for layer_idx in range(model.config.num_hidden_layers):
        segments, end_offset = build_full_layer_weight_manifest_for_layer(
            model,
            layer_idx,
            offset,
        )
        actual_numel = end_offset - offset
        if actual_numel != per_layer_numel:
            raise RuntimeError(
                f"layer {layer_idx} packed weight size {actual_numel} "
                f"!= expected per-layer size {per_layer_numel}"
            )
        layers.append(
            {
                "id": layer_idx,
                "element_offset": offset,
                "numel": per_layer_numel,
                "byte_offset": offset * 2,
                "byte_length": per_layer_numel * 2,
                "byte_alignment": _byte_alignment(offset * 2),
                "segments": segments,
            }
        )
        offset = end_offset

    return {
        "format": PACKED_WEIGHTS_FORMAT,
        "dtype": "bfloat16",
        "element_size_bytes": 2,
        "weight_order": list(FULL_LAYER_WEIGHT_ORDER),
        "model_config": _config_dict(model),
        "num_layers": model.config.num_hidden_layers,
        "per_layer_numel": per_layer_numel,
        "total_numel": offset,
        "total_bytes": offset * 2,
        "data_file": PACKED_WEIGHTS_BIN,
        "layers": layers,
    }


def pack_all_full_layer_weights(
    model: Qwen3ForCausalLM,
    expected_per_layer_numel: int | None = None,
) -> tuple[torch.Tensor, dict[str, object]]:
    packed_layers = []
    for layer_idx in range(model.config.num_hidden_layers):
        packed = pack_full_layer_weights_for_layer(model, layer_idx).to(torch.bfloat16)
        if expected_per_layer_numel is not None and (
            packed.numel() != expected_per_layer_numel
        ):
            raise RuntimeError(
                f"layer {layer_idx} packed weight size {packed.numel()} "
                f"!= expected per-layer size {expected_per_layer_numel}"
            )
        packed_layers.append(packed)
    per_layer_numel = packed_layers[0].numel()
    weights = torch.cat(packed_layers).contiguous()
    manifest = build_packed_weights_manifest(model, per_layer_numel)
    return weights, manifest


def write_packed_weight_artifact(
    model: Qwen3ForCausalLM,
    output_dir: Path,
    expected_per_layer_numel: int | None = None,
) -> dict[str, object]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    weights, manifest = pack_all_full_layer_weights(
        model,
        expected_per_layer_numel=expected_per_layer_numel,
    )
    bin_path = output_dir / PACKED_WEIGHTS_BIN
    manifest_path = output_dir / PACKED_WEIGHTS_MANIFEST
    tmp_bin = output_dir / f"{PACKED_WEIGHTS_BIN}.tmp"
    tmp_manifest = output_dir / f"{PACKED_WEIGHTS_MANIFEST}.tmp"
    _bf16_to_file(weights, tmp_bin)
    tmp_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    tmp_bin.replace(bin_path)
    tmp_manifest.replace(manifest_path)
    validate_packed_weight_artifact(
        model,
        output_dir,
        expected_per_layer_numel=expected_per_layer_numel,
    )
    return manifest


def load_packed_weights_manifest(packed_dir: Path) -> dict[str, object]:
    manifest_path = Path(packed_dir) / PACKED_WEIGHTS_MANIFEST
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing packed weight manifest: {manifest_path}")
    return json.loads(manifest_path.read_text())


def validate_packed_weights_manifest(
    model: Qwen3ForCausalLM,
    manifest: dict[str, object],
    expected_per_layer_numel: int | None = None,
) -> None:
    if manifest.get("format") != PACKED_WEIGHTS_FORMAT:
        raise RuntimeError(
            f"unsupported packed weight format {manifest.get('format')!r}; "
            f"expected {PACKED_WEIGHTS_FORMAT}"
        )
    if manifest.get("dtype") != "bfloat16":
        raise RuntimeError(
            f"packed weight dtype must be bfloat16: {manifest.get('dtype')}"
        )
    if manifest.get("weight_order") != list(FULL_LAYER_WEIGHT_ORDER):
        raise RuntimeError("packed weight order does not match the runtime layout")
    config = _config_dict(model)
    manifest_config = manifest.get("model_config", {})
    for key, value in config.items():
        if key in manifest_config and manifest_config[key] != value:
            raise RuntimeError(
                f"packed weight config mismatch for {key}: "
                f"{manifest_config[key]!r} != {value!r}"
            )
    if int(manifest.get("num_layers", -1)) != model.config.num_hidden_layers:
        raise RuntimeError(
            f"packed weight layer count {manifest.get('num_layers')} "
            f"!= model layer count {model.config.num_hidden_layers}"
        )
    per_layer_numel = int(manifest.get("per_layer_numel", -1))
    if expected_per_layer_numel is not None and (
        per_layer_numel != expected_per_layer_numel
    ):
        raise RuntimeError(
            f"packed per-layer size {per_layer_numel} "
            f"!= expected {expected_per_layer_numel}"
        )
    expected_total = per_layer_numel * model.config.num_hidden_layers
    if int(manifest.get("total_numel", -1)) != expected_total:
        raise RuntimeError(
            f"packed total elements {manifest.get('total_numel')} "
            f"!= expected {expected_total}"
        )
    layers = manifest.get("layers")
    if not isinstance(layers, list) or len(layers) != model.config.num_hidden_layers:
        raise RuntimeError("packed manifest layer table is missing or incomplete")
    for layer_idx, layer in enumerate(layers):
        offset = layer_idx * per_layer_numel
        if int(layer["id"]) != layer_idx:
            raise RuntimeError(f"packed layer id mismatch at index {layer_idx}")
        if int(layer["element_offset"]) != offset:
            raise RuntimeError(f"packed layer {layer_idx} element offset mismatch")
        if int(layer["numel"]) != per_layer_numel:
            raise RuntimeError(f"packed layer {layer_idx} size mismatch")
        if int(layer["byte_offset"]) != offset * 2:
            raise RuntimeError(f"packed layer {layer_idx} byte offset mismatch")
        if int(layer["byte_offset"]) % 64 != 0:
            raise RuntimeError(
                f"packed layer {layer_idx} byte offset is not 64B aligned"
            )


def validate_packed_weight_artifact(
    model: Qwen3ForCausalLM,
    packed_dir: Path,
    expected_per_layer_numel: int | None = None,
) -> dict[str, object]:
    packed_dir = Path(packed_dir)
    manifest = load_packed_weights_manifest(packed_dir)
    validate_packed_weights_manifest(
        model,
        manifest,
        expected_per_layer_numel=expected_per_layer_numel,
    )
    bin_path = packed_dir / PACKED_WEIGHTS_BIN
    if not bin_path.exists():
        raise FileNotFoundError(f"missing packed weight data file: {bin_path}")
    expected_bytes = int(manifest["total_bytes"])
    actual_bytes = bin_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise RuntimeError(
            f"packed weight file size {actual_bytes} bytes != expected {expected_bytes}"
        )
    return manifest


def load_packed_weight_tensor(
    packed_dir: Path,
    manifest: dict[str, object] | None = None,
) -> torch.Tensor:
    packed_dir = Path(packed_dir)
    if manifest is None:
        manifest = load_packed_weights_manifest(packed_dir)
    return _bf16_from_file(
        packed_dir / PACKED_WEIGHTS_BIN,
        int(manifest["total_numel"]),
    )


def packed_weight_layer_slice(
    packed_weights: torch.Tensor,
    manifest: dict[str, object],
    layer_idx: int,
) -> torch.Tensor:
    layer = manifest["layers"][layer_idx]
    start = int(layer["element_offset"])
    end = start + int(layer["numel"])
    return packed_weights[start:end]
