#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from iron.applications.qwen3_0_6b.qwen3_cpu import Qwen3ForCausalLM, rms_norm
from iron.applications.qwen3_0_6b.persistent.checks import print_tensor_check
from iron.applications.qwen3_0_6b.persistent.layout import host_owned_tensor
from iron.applications.qwen3_0_6b.persistent.ops_core import (
    Qwen3PersistentInputRMSNormQKV,
)
from iron.applications.qwen3_0_6b.qwen3_preflight import (
    run_persistent_artifact_preflight,
)
from iron.common.context import AIEContext


def run_qkv_boundary_diagnostic(
    *,
    model: Qwen3ForCausalLM,
    qkv_op: Qwen3PersistentInputRMSNormQKV,
    qkv_op_func,
    layer_idx: int,
    inputs: dict[str, torch.Tensor],
    full_layer_v: torch.Tensor,
) -> dict[str, int]:
    hidden_buf = XRTTensor.from_torch(inputs["hidden"].contiguous())
    packed_weights = torch.cat(
        [
            inputs["input_norm_weight"].flatten(),
            inputs["W_q"].flatten(),
            inputs["W_k"].flatten(),
            inputs["W_v"].flatten(),
        ]
    ).contiguous()
    weights_buf = XRTTensor.from_torch(packed_weights)
    packed_outputs_buf = XRTTensor(
        (qkv_op.packed_outputs_size,),
        dtype=hidden_buf.dtype,
    )
    result = qkv_op_func(hidden_buf, weights_buf, packed_outputs_buf)
    packed_outputs_buf.device = "npu"
    packed_outputs = packed_outputs_buf.to_torch()

    x_norm = packed_outputs[: qkv_op.q_output_base]
    values = packed_outputs[qkv_op.v_output_base :]
    x_norm_expected = rms_norm(
        inputs["hidden"].view(1, 1, -1),
        inputs["input_norm_weight"],
        model.config.rms_norm_eps,
    ).flatten()
    values_local = F.linear(
        x_norm.view(1, 1, -1),
        inputs["W_v"],
    ).flatten()
    values_py_ref = F.linear(
        x_norm_expected.view(1, 1, -1),
        inputs["W_v"],
    ).flatten()

    print(f"layer_{layer_idx}_diag_qkv_npu_time_us: {result.npu_time / 1e3:.3f}")
    return {
        "x_norm": print_tensor_check(
            f"layer_{layer_idx}_diag_qkv_x_norm",
            x_norm,
            x_norm_expected,
            rel_tol=0.04,
            abs_tol=1e-6,
        ),
        "values_local": print_tensor_check(
            f"layer_{layer_idx}_diag_qkv_values_local",
            values,
            values_local,
            rel_tol=0.04,
            abs_tol=1e-6,
        ),
        "full_v_vs_qkv": print_tensor_check(
            f"layer_{layer_idx}_diag_full_v_vs_qkv_values",
            full_layer_v,
            values,
            rel_tol=0.05,
            abs_tol=0.025,
        ),
        "full_v_vs_py_ref": print_tensor_check(
            f"layer_{layer_idx}_diag_full_v_vs_py_ref_values",
            full_layer_v,
            values_py_ref,
            rel_tol=0.05,
            abs_tol=0.025,
        ),
    }


def write_qkv_boundary_diagnostic_bundle(
    *,
    build_dir: str,
    layer_idx: int,
    inputs: dict[str, torch.Tensor],
    full_layer_v: torch.Tensor,
) -> Path:
    def as_float32_array(tensor: torch.Tensor) -> np.ndarray:
        host_tensor = host_owned_tensor(tensor)
        return host_tensor.to(torch.float32).cpu().numpy().copy()

    diag_dir = Path(build_dir) / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)
    path = diag_dir / f"qkv_boundary_layer_{layer_idx}.npz"
    print(f"layer_{layer_idx}_qkv_diagnostic_bundle_begin: {path}")
    sys.stdout.flush()
    arrays = {"layer_idx": np.array(layer_idx, dtype=np.int64)}
    tensors = {
        "hidden": inputs["hidden"],
        "input_norm_weight": inputs["input_norm_weight"],
        "W_q": inputs["W_q"],
        "W_k": inputs["W_k"],
        "W_v": inputs["W_v"],
        "full_layer_v": full_layer_v,
    }
    for name, tensor in tensors.items():
        print(f"layer_{layer_idx}_qkv_diagnostic_bundle_tensor_begin: {name}")
        sys.stdout.flush()
        arrays[name] = as_float32_array(tensor)
        print(
            f"layer_{layer_idx}_qkv_diagnostic_bundle_tensor_done: "
            f"{name} shape={arrays[name].shape}"
        )
        sys.stdout.flush()
    np.savez(
        path,
        **arrays,
    )
    print(f"layer_{layer_idx}_qkv_diagnostic_bundle: {path}")
    sys.stdout.flush()
    return path


def run_qkv_diagnostic_bundle(
    qkv_diagnostic_bundle: Path, model: Qwen3ForCausalLM, context: AIEContext
) -> bool:
    def bf16_from_float32(array: np.ndarray, shape: tuple[int, ...]) -> torch.Tensor:
        return torch.from_numpy(array.copy()).reshape(shape).to(torch.bfloat16)

    bundle = np.load(qkv_diagnostic_bundle)
    qkv_op = Qwen3PersistentInputRMSNormQKV(
        hidden_size=model.config.hidden_size,
        q_size=model.config.num_attention_heads * model.config.head_dim,
        kv_size=model.config.num_key_value_heads * model.config.head_dim,
        epsilon=model.config.rms_norm_eps,
        context=context,
    )
    start = time.perf_counter()
    qkv_op.compile()
    print("stage: qkv-diagnostic-bundle")
    print(f"bundle: {qkv_diagnostic_bundle}")
    print(f"diagnostic_qkv_compile_s: {time.perf_counter() - start:.3f}")
    qkv_diag_preflight = run_persistent_artifact_preflight(
        mlir_path=Path(qkv_op.xclbin_artifact.mlir_input.filename),
        arg_specs=len(qkv_op.get_arg_spec()),
    )
    print(
        "diagnostic_qkv_preflight: ok "
        f"runtime_memrefs={qkv_diag_preflight.runtime_memrefs} "
        f"arg_specs={qkv_diag_preflight.arg_specs} "
        f"compute_cores={qkv_diag_preflight.compute_cores} "
        f"non_advancing_acquires={qkv_diag_preflight.non_advancing_acquires}"
    )
    qkv_op_func = qkv_op.get_callable()
    layer_idx = int(bundle["layer_idx"].item())
    inputs = {
        "hidden": bf16_from_float32(bundle["hidden"], (model.config.hidden_size,)),
        "input_norm_weight": bf16_from_float32(
            bundle["input_norm_weight"], (model.config.hidden_size,)
        ),
        "W_q": bf16_from_float32(
            bundle["W_q"],
            (
                model.config.num_attention_heads * model.config.head_dim,
                model.config.hidden_size,
            ),
        ),
        "W_k": bf16_from_float32(
            bundle["W_k"],
            (
                model.config.num_key_value_heads * model.config.head_dim,
                model.config.hidden_size,
            ),
        ),
        "W_v": bf16_from_float32(
            bundle["W_v"],
            (
                model.config.num_key_value_heads * model.config.head_dim,
                model.config.hidden_size,
            ),
        ),
    }
    full_layer_v = bf16_from_float32(
        bundle["full_layer_v"],
        (model.config.num_key_value_heads * model.config.head_dim,),
    )
    errors = run_qkv_boundary_diagnostic(
        model=model,
        qkv_op=qkv_op,
        qkv_op_func=qkv_op_func,
        layer_idx=layer_idx,
        inputs=inputs,
        full_layer_v=full_layer_v,
    )
    if errors["x_norm"]:
        print(f"qkv_diagnostic_result: layer_{layer_idx}_diag_qkv_x_norm")
    elif errors["values_local"]:
        print(f"qkv_diagnostic_result: layer_{layer_idx}_diag_qkv_values_local")
    elif errors["full_v_vs_qkv"]:
        print(f"qkv_diagnostic_result: layer_{layer_idx}_diag_full_v_vs_qkv_values")
    elif errors["full_v_vs_py_ref"]:
        print(f"qkv_diagnostic_result: layer_{layer_idx}_diag_py_ref_drift")
    else:
        print(f"qkv_diagnostic_result: layer_{layer_idx}_diag_qkv_passed")
    return bool(errors["x_norm"] or errors["values_local"] or errors["full_v_vs_qkv"])
