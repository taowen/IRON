#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import torch
import torch.nn.functional as F
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

from iron.applications.qwen3_0_6b.qwen3_cpu import Qwen3ForCausalLM, rms_norm
from iron.applications.qwen3_0_6b.qwen3_decode_reference import (
    Qwen3CachedReference,
    clone_decode_state,
)
from iron.applications.qwen3_0_6b.persistent.checks import print_tensor_check
from iron.applications.qwen3_0_6b.persistent.diagnostics import (
    write_qkv_boundary_diagnostic_bundle,
)
from iron.applications.qwen3_0_6b.persistent.layout import (
    build_full_layer_inputs_for_layer,
    host_owned_tensor,
    pack_full_layer_weights,
    pack_qk_rope_metadata_for_layers,
    pack_segment_major_full_layer_weights,
)
from iron.applications.qwen3_0_6b.persistent.ops_core import verification_tolerance
from iron.applications.qwen3_0_6b.persistent.refs import (
    build_reference_multi_layer_full_layer,
    build_reference_full_mlp,
    build_reference_input,
    build_reference_mlp_down_residual,
    build_reference_mlp_gate_up,
    build_reference_qkv,
    one_layer_reference_tensors,
)
from iron.applications.qwen3_0_6b.persistent.stages import N_LAYER_FINAL_ONLY_STAGE
from iron.applications.qwen3_0_6b.qwen3_preflight import (
    run_persistent_artifact_preflight,
)
from iron.common.test_utils import verify_buffer


def _pack_n_layer_runtime_buffers(op, initial_hidden, inputs_by_layer):
    if (
        op.layer_iterations == 1
        and op.num_aie_columns == 1
        and getattr(op, "attention_columns", 1) == 1
    ):
        packed_weights = pack_full_layer_weights(inputs_by_layer[0])
    else:
        packed_weights = pack_segment_major_full_layer_weights(
            inputs_by_layer,
            mlp_columns=op.num_aie_columns if op.num_aie_columns == 2 else 1,
            mlp_gate_up_columns=getattr(op, "effective_mlp_gate_up_columns", 0),
            attention_columns=getattr(op, "attention_columns", 1),
            mlp_gate_up_pair_rows=getattr(op, "mlp_gate_up_pair_rows", False),
            mlp_gate_up_row_group=getattr(op, "mlp_gate_up_row_group", 4),
        )
    if getattr(op, "attention_columns", 1) == 2:
        rope_runtime = pack_qk_rope_metadata_for_layers(
            inputs_by_layer,
            position=None,
        )
    else:
        rope_runtime = inputs_by_layer[0]["rope_angles"]
    if getattr(op, "runtime_hidden_size", op.hidden_size) != op.hidden_size:
        metadata_rows = rope_runtime.view(op.layer_iterations, -1)
        runtime_hidden = torch.cat(
            [
                torch.cat([initial_hidden.flatten(), metadata_rows[layer_idx]])
                for layer_idx in range(op.layer_iterations)
            ]
        ).contiguous()
    else:
        runtime_hidden = initial_hidden
    initial_cache = torch.cat(
        [inputs["initial_cache"].clone() for inputs in inputs_by_layer]
    ).contiguous()
    return packed_weights, rope_runtime, runtime_hidden, initial_cache


def _run_n_layer_prefix_hidden(
    *,
    base_op,
    prefix_layers: int,
    initial_hidden: torch.Tensor,
    initial_state,
    inputs_by_layer: list[dict[str, torch.Tensor]],
) -> torch.Tensor:
    prefix_inputs = inputs_by_layer[:prefix_layers]
    prefix_op = type(base_op)(
        hidden_size=base_op.hidden_size,
        q_size=base_op.q_size,
        kv_size=base_op.kv_size,
        head_dim=base_op.head_dim,
        max_seq_len=base_op.max_seq_len,
        position=base_op.position,
        intermediate_size=base_op.intermediate_size,
        layer_iterations=prefix_layers,
        num_aie_columns=base_op.num_aie_columns,
        attention_columns=getattr(base_op, "attention_columns", 1),
        mlp_gate_up_columns=getattr(base_op, "mlp_gate_up_columns", 0),
        mlp_gate_up_pair_rows=getattr(base_op, "mlp_gate_up_pair_rows", False),
        mlp_gate_up_direct_silu=getattr(base_op, "mlp_gate_up_direct_silu", False),
        mlp_gate_up_row_group=getattr(base_op, "mlp_gate_up_row_group", 4),
        attention_probe_only=False,
        tile_size_input=base_op.tile_size_input,
        tile_size_output=base_op.tile_size_output,
        epsilon=base_op.epsilon,
        context=base_op.context,
    )
    prefix_op.compile()
    preflight = run_persistent_artifact_preflight(
        mlir_path=Path(prefix_op.xclbin_artifact.mlir_input.filename),
        arg_specs=len(prefix_op.get_arg_spec()),
    )
    print(
        "nlayer_prefix_preflight: ok "
        f"layers={prefix_layers} "
        f"runtime_memrefs={preflight.runtime_memrefs} "
        f"compute_cores={preflight.compute_cores} "
        f"max_dma_tasks_per_fifo={preflight.max_dma_tasks_per_fifo} "
        f"max_tile_inputs={preflight.max_compute_tile_inputs} "
        f"max_tile_outputs={preflight.max_compute_tile_outputs}"
    )
    packed_weights, rope_runtime, runtime_hidden, initial_cache = (
        _pack_n_layer_runtime_buffers(prefix_op, initial_hidden, prefix_inputs)
    )
    hidden_buf = XRTTensor.from_torch(runtime_hidden)
    weights_buf = XRTTensor.from_torch(packed_weights)
    rope_buf = XRTTensor.from_torch(rope_runtime)
    output_buf = XRTTensor((prefix_op.packed_outputs_size,), dtype=hidden_buf.dtype)
    cache_buf = XRTTensor.from_torch(initial_cache.clone())
    result = prefix_op.get_callable()(
        hidden_buf,
        weights_buf,
        rope_buf,
        output_buf,
        cache_buf,
    )
    print(
        f"nlayer_prefix_npu_time_us: layers={prefix_layers} "
        f"{result.npu_time / 1e3:.3f}"
    )
    output_buf.device = "npu"
    return host_owned_tensor(output_buf.to_torch())


def _run_single_layer_from_boundary(
    *,
    base_op,
    layer_input: torch.Tensor,
    layer_inputs: dict[str, torch.Tensor],
) -> torch.Tensor:
    single_op = type(base_op)(
        hidden_size=base_op.hidden_size,
        q_size=base_op.q_size,
        kv_size=base_op.kv_size,
        head_dim=base_op.head_dim,
        max_seq_len=base_op.max_seq_len,
        position=base_op.position,
        intermediate_size=base_op.intermediate_size,
        layer_iterations=1,
        num_aie_columns=base_op.num_aie_columns,
        attention_columns=getattr(base_op, "attention_columns", 1),
        mlp_gate_up_columns=getattr(base_op, "mlp_gate_up_columns", 0),
        mlp_gate_up_pair_rows=getattr(base_op, "mlp_gate_up_pair_rows", False),
        mlp_gate_up_direct_silu=getattr(base_op, "mlp_gate_up_direct_silu", False),
        mlp_gate_up_row_group=getattr(base_op, "mlp_gate_up_row_group", 4),
        attention_probe_only=False,
        tile_size_input=base_op.tile_size_input,
        tile_size_output=base_op.tile_size_output,
        epsilon=base_op.epsilon,
        context=base_op.context,
    )
    single_op.compile()
    preflight = run_persistent_artifact_preflight(
        mlir_path=Path(single_op.xclbin_artifact.mlir_input.filename),
        arg_specs=len(single_op.get_arg_spec()),
    )
    print(
        "nlayer_single_boundary_preflight: ok "
        f"runtime_memrefs={preflight.runtime_memrefs} "
        f"compute_cores={preflight.compute_cores} "
        f"max_dma_tasks_per_fifo={preflight.max_dma_tasks_per_fifo} "
        f"max_tile_inputs={preflight.max_compute_tile_inputs} "
        f"max_tile_outputs={preflight.max_compute_tile_outputs}"
    )
    packed_weights, rope_runtime, runtime_hidden, initial_cache = (
        _pack_n_layer_runtime_buffers(single_op, layer_input, [layer_inputs])
    )
    hidden_buf = XRTTensor.from_torch(runtime_hidden)
    weights_buf = XRTTensor.from_torch(packed_weights)
    rope_buf = XRTTensor.from_torch(rope_runtime)
    output_buf = XRTTensor((single_op.packed_outputs_size,), dtype=hidden_buf.dtype)
    cache_buf = XRTTensor.from_torch(initial_cache.clone())
    result = single_op.get_callable()(
        hidden_buf,
        weights_buf,
        rope_buf,
        output_buf,
        cache_buf,
    )
    print(f"nlayer_single_boundary_npu_time_us: {result.npu_time / 1e3:.3f}")
    output_buf.device = "npu"
    return host_owned_tensor(output_buf.to_torch())


def _local_single_layer_reference(
    *,
    model: Qwen3ForCausalLM,
    args,
    layer_idx: int,
    layer_input: torch.Tensor,
    initial_state,
) -> torch.Tensor:
    ref = Qwen3CachedReference(
        model, args.max_seq_len, num_layers=max(1, layer_idx + 1)
    )
    state = clone_decode_state(initial_state)
    layer_prefix = f"model.layers.{layer_idx}"
    residual = layer_input.view(1, 1, -1)
    x_norm = rms_norm(
        residual,
        model.w(f"{layer_prefix}.input_layernorm.weight"),
        model.config.rms_norm_eps,
    )
    x = residual + ref._attention_decode(x_norm, layer_idx, state)
    residual = x
    x_norm = rms_norm(
        x,
        model.w(f"{layer_prefix}.post_attention_layernorm.weight"),
        model.config.rms_norm_eps,
    )
    x = residual + ref._mlp(x_norm, layer_idx)
    return x.flatten().contiguous()


def _run_n_layer_local_cache_diagnostic(
    *,
    args,
    model: Qwen3ForCausalLM,
    op,
    layer_idx: int,
    next_token: int,
    initial_hidden: torch.Tensor,
    initial_state,
    expected_state,
    inputs_by_layer: list[dict[str, torch.Tensor]],
    packed_cache_chunk: torch.Tensor,
    actual_hidden: torch.Tensor,
) -> bool:
    if not (0 <= layer_idx < op.layer_iterations):
        raise ValueError(
            f"diagnose layer must be in [0, {op.layer_iterations}), got {layer_idx}"
        )
    if layer_idx == 0:
        layer_input = initial_hidden.flatten().contiguous()
    else:
        layer_input = _run_n_layer_prefix_hidden(
            base_op=op,
            prefix_layers=layer_idx,
            initial_hidden=initial_hidden,
            initial_state=initial_state,
            inputs_by_layer=inputs_by_layer,
        )

    if layer_idx == 0:
        expected_layer_input = initial_hidden.view(1, 1, -1).contiguous()
    else:
        prefix_ref = Qwen3CachedReference(model, args.max_seq_len, num_layers=layer_idx)
        expected_layer_input, _ = prefix_ref.decode_hidden(
            next_token,
            clone_decode_state(initial_state),
        )
    layer_inputs = inputs_by_layer[layer_idx]
    layer_cache = packed_cache_chunk[
        layer_idx * op.packed_cache_size : (layer_idx + 1) * op.packed_cache_size
    ]
    values_cache = layer_cache[op.cache_half_size :].view(
        op.kv_heads, args.max_seq_len, op.head_dim
    )
    actual_v_current = values_cache[:, op.position, :].flatten().contiguous()
    actual_v_current = host_owned_tensor(actual_v_current)
    x_norm_local = rms_norm(
        layer_input.view(1, 1, -1),
        layer_inputs["input_norm_weight"],
        model.config.rms_norm_eps,
    ).flatten()
    values_local = F.linear(
        x_norm_local.view(1, 1, -1),
        layer_inputs["W_v"],
    ).flatten()
    print(f"nlayer_local_diag_layer: {layer_idx}")
    errors = {}
    errors["layer_input_full_ref"] = print_tensor_check(
        f"layer{layer_idx}_diag_layer_input_full_ref",
        layer_input,
        expected_layer_input.flatten().contiguous(),
        rel_tol=0.06,
        abs_tol=0.04 * max(1, layer_idx),
    )
    errors["values_cache_local"] = print_tensor_check(
        f"layer{layer_idx}_diag_values_cache_vs_local_input",
        actual_v_current,
        values_local,
        rel_tol=0.04,
        abs_tol=1e-6,
    )
    errors["values_cache_full_ref"] = print_tensor_check(
        f"layer{layer_idx}_diag_values_cache_vs_full_ref",
        actual_v_current,
        expected_state.values[layer_idx][:, op.position, :].flatten(),
        rel_tol=0.05,
        abs_tol=0.025 * (layer_idx + 1),
    )
    if errors["values_cache_local"]:
        bundle_inputs = {
            **layer_inputs,
            "hidden": layer_input,
        }
        write_qkv_boundary_diagnostic_bundle(
            build_dir=args.build_dir,
            layer_idx=layer_idx,
            inputs=bundle_inputs,
            full_layer_v=actual_v_current,
        )
        print(f"nlayer_local_diag_result: layer_{layer_idx}_values_cache_local_failed")
    elif errors["values_cache_full_ref"]:
        print(f"nlayer_local_diag_result: layer_{layer_idx}_full_ref_drift")
    else:
        print(f"nlayer_local_diag_result: layer_{layer_idx}_values_cache_local_passed")
    if layer_idx == op.layer_iterations - 1:
        single_hidden = _run_single_layer_from_boundary(
            base_op=op,
            layer_input=layer_input,
            layer_inputs=layer_inputs,
        )
        local_hidden = _local_single_layer_reference(
            model=model,
            args=args,
            layer_idx=layer_idx,
            layer_input=layer_input,
            initial_state=initial_state,
        )
        print_tensor_check(
            f"layer{layer_idx}_diag_main_vs_single_layer_hidden",
            actual_hidden,
            single_hidden,
            rel_tol=0.04,
            abs_tol=1e-6,
        )
        print_tensor_check(
            f"layer{layer_idx}_diag_single_layer_hidden_vs_local_ref",
            single_hidden,
            local_hidden,
            rel_tol=0.06,
            abs_tol=0.04,
        )
    return bool(errors["values_cache_local"])


def run_n_layer_final_only(
    args,
    model: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    op,
    op_func,
) -> bool:
    if args.layer_chunk_size != op.layer_iterations:
        raise RuntimeError(
            f"CLI layer_chunk_size {args.layer_chunk_size} != "
            f"op.layer_iterations {op.layer_iterations}"
        )
    (
        next_token,
        position,
        initial_hidden,
        initial_state,
        expected_hidden,
        expected_state,
    ) = build_reference_multi_layer_full_layer(
        model,
        input_ids,
        args.max_seq_len,
        num_layers=op.layer_iterations,
        prefill_num_layers=op.layer_iterations,
    )
    if position != op.position:
        raise RuntimeError(f"compiled position {op.position} != reference {position}")

    inputs_by_layer = [
        build_full_layer_inputs_for_layer(
            model,
            layer_idx,
            initial_hidden,
            initial_state,
        )
        for layer_idx in range(op.layer_iterations)
    ]
    packed_weights, rope_runtime, runtime_hidden, initial_cache = (
        _pack_n_layer_runtime_buffers(op, initial_hidden, inputs_by_layer)
    )
    weights_buf = XRTTensor.from_torch(packed_weights)
    hidden_buf = XRTTensor.from_torch(runtime_hidden)
    rope_angles_buf = XRTTensor.from_torch(rope_runtime)

    failed = False
    repeat_count = max(1, args.verify_repeat)
    for iteration in range(repeat_count):
        output_buf = XRTTensor((op.packed_outputs_size,), dtype=hidden_buf.dtype)
        cache_buf = XRTTensor.from_torch(initial_cache.clone())
        result = op_func(
            hidden_buf,
            weights_buf,
            rope_angles_buf,
            output_buf,
            cache_buf,
        )
        print(f"iteration: {iteration}")
        print(f"prompt_next_token: {next_token}")
        print(f"decode_position: {position}")
        print(f"npu_time_us: {result.npu_time / 1e3:.3f}")

        output_buf.device = "npu"
        cache_buf.device = "npu"
        actual_hidden = output_buf.to_torch()
        packed_cache_chunk = cache_buf.to_torch()

        if getattr(op, "attention_probe_only", False):
            reference_tensors = one_layer_reference_tensors(
                model,
                next_token,
                initial_state,
                args.max_seq_len,
            )
            checks = {
                "attention_probe_residual": (
                    actual_hidden,
                    reference_tensors["attn_residual"].flatten().contiguous(),
                    0.06,
                    0.04,
                ),
            }
        else:
            checks = {
                "chunk_hidden": (
                    actual_hidden,
                    expected_hidden,
                    0.06,
                    0.04 * op.layer_iterations,
                ),
            }
        for layer_idx in range(op.layer_iterations):
            layer_cache = packed_cache_chunk[
                layer_idx
                * op.packed_cache_size : (layer_idx + 1)
                * op.packed_cache_size
            ]
            keys_cache = layer_cache[: op.cache_half_size].view(
                op.kv_heads, args.max_seq_len, op.head_dim
            )
            values_cache = layer_cache[op.cache_half_size :].view(
                op.kv_heads, args.max_seq_len, op.head_dim
            )
            checks[f"layer{layer_idx}_keys_cache_current"] = (
                keys_cache[:, position, :].flatten(),
                expected_state.keys[layer_idx][:, position, :].flatten(),
                0.05,
                0.6,
            )
            checks[f"layer{layer_idx}_values_cache_current"] = (
                values_cache[:, position, :].flatten(),
                expected_state.values[layer_idx][:, position, :].flatten(),
                0.05,
                0.025 * (layer_idx + 1),
            )
        for name, (actual, expected, rel_tol, abs_tol) in checks.items():
            errors = print_tensor_check(name, actual, expected, rel_tol, abs_tol)
            failed = failed or bool(errors)
        if args.diagnose_nlayer_layer is not None:
            failed = (
                _run_n_layer_local_cache_diagnostic(
                    args=args,
                    model=model,
                    op=op,
                    layer_idx=args.diagnose_nlayer_layer,
                    next_token=next_token,
                    initial_hidden=initial_hidden,
                    initial_state=initial_state,
                    expected_state=expected_state,
                    inputs_by_layer=inputs_by_layer,
                    packed_cache_chunk=packed_cache_chunk,
                    actual_hidden=actual_hidden,
                )
                or failed
            )

    return failed


def print_stage_proof(args, op) -> None:
    for artifact in op.artifacts:
        print(f"artifact: {artifact.filename}")
    if args.stage == "input-rmsnorm":
        print("dispatch_shape: hidden[1024] + norm_weight[1024] -> x_norm[1024]")
        print(
            "worker_graph: hidden_fifo -> rmsnorm_worker -> weight_worker -> output_fifo"
        )
    elif args.stage == "input-rmsnorm-qkv":
        print(
            "dispatch_shape: hidden[1024] + norm_weight[1024] + "
            "Wq[2048,1024] + Wk[1024,1024] + Wv[1024,1024] -> "
            "x_norm[1024], queries_raw[2048], keys_raw[1024], values[1024]"
        )
        print(
            "runtime_bos: hidden[1024], "
            f"packed_weights[{op.packed_weights_size}], "
            f"packed_outputs[{op.packed_outputs_size}]"
        )
        print(f"qkv_columns: {op.num_aie_columns}")
        print(
            "worker_graph: hidden_fifo -> rmsnorm_worker -> weight_worker -> "
            "single xnorm broadcast FIFO -> Q/K/V matvec workers"
        )
    elif args.stage == "post-attn-rmsnorm-mlp-gate-up":
        print(
            "dispatch_shape: attn_residual[1024] + "
            "post_attention_norm_weight[1024] + W_gate[3072,1024] + "
            "W_up[3072,1024] -> mlp_x_norm, ffn_gate, ffn_up, "
            "ffn_gate_silu, ffn_hidden"
        )
        print(
            "runtime_bos: attn_residual[1024], "
            f"packed_weights[{op.packed_weights_size}], "
            f"packed_outputs[{op.packed_outputs_size}]"
        )
        print(f"mlp_columns: {op.num_aie_columns}")
        print(
            "worker_graph: attn_residual_fifo + post_norm_weight_fifo -> "
            "weighted_rmsnorm_worker -> xnorm broadcast -> gate/up matvec "
            "workers -> silu_worker + mul_worker"
        )
    elif args.stage == "post-attn-mlp-down-residual":
        print(
            "dispatch_shape: ffn_hidden[3072] + attn_residual[1024] + "
            "W_down[1024,3072] -> ffn_out[1024], layer_residual[1024]"
        )
        print(
            "runtime_bos: ffn_hidden[3072], attn_residual[1024], "
            f"packed_weights[{op.packed_weights_size}], "
            f"packed_outputs[{op.packed_outputs_size}]"
        )
        print(f"down_columns: {op.num_aie_columns}")
        print(
            "worker_graph: ffn_hidden broadcast + down_weight_fifo -> "
            "down_matvec_worker -> ffn_out broadcast -> residual_add_worker"
        )
    elif args.stage == "post-attn-rmsnorm-full-mlp":
        print(
            "dispatch_shape: attn_residual[1024] + "
            "post_attention_norm_weight[1024] + W_gate[3072,1024] + "
            "W_up[3072,1024] + W_down[1024,3072] -> mlp_x_norm, "
            "ffn_gate, ffn_up, ffn_gate_silu, ffn_hidden, ffn_out, "
            "layer_residual"
        )
        print(
            "runtime_bos: attn_residual[1024], "
            f"packed_weights[{op.packed_weights_size}], "
            f"packed_outputs[{op.packed_outputs_size}]"
        )
        print(f"full_mlp_columns: {op.num_aie_columns}")
        print(
            "worker_graph: attn_residual_fifo + post_norm_weight_fifo -> "
            "weighted_rmsnorm_worker -> xnorm broadcast -> gate/up matvec "
            "workers -> silu_worker + mul_worker -> down_matvec_worker -> "
            "residual_add_worker"
        )
    elif args.stage == N_LAYER_FINAL_ONLY_STAGE:
        print(
            "dispatch_shape: hidden[1024] through "
            f"{op.layer_iterations} full layer(s) in one persistent invocation"
        )
        print(
            "runtime_bos: hidden[1024], "
            f"weight_chunk[{op.packed_weight_chunk_size}], "
            "rope_angles[128], "
            f"final_hidden[{op.packed_outputs_size}], "
            f"cache_chunk[{op.packed_cache_chunk_size}]"
        )
        print(f"decode_position: {op.position}")
        print(
            "worker_graph: one transformer-layer worker graph loops over the chunk; "
            "intermediate residuals are routed back as the next layer hidden, "
            "and only the final residual is drained to host"
        )


def run_compiled_stage(
    args,
    model: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    op,
    op_func,
) -> bool:
    if args.stage == N_LAYER_FINAL_ONLY_STAGE:
        return run_n_layer_final_only(args, model, input_ids, op, op_func)

    if args.stage == "input-rmsnorm":
        next_token, hidden, weight, expected = build_reference_input(
            model, input_ids, args.max_seq_len
        )
        hidden_buf = XRTTensor.from_torch(hidden)
        weight_buf = XRTTensor.from_torch(weight)
        output_buf = XRTTensor((model.config.hidden_size,), dtype=hidden_buf.dtype)
        op_args = [hidden_buf, weight_buf, output_buf]
        output_buffers = {"input_rmsnorm": output_buf}
        expected_buffers = {"input_rmsnorm": expected}
        full_expected_buffers = expected_buffers
    elif args.stage == "input-rmsnorm-qkv":
        next_token, inputs, expected_buffers = build_reference_qkv(
            model, input_ids, args.max_seq_len
        )
        full_expected_buffers = expected_buffers
        hidden_buf = XRTTensor.from_torch(inputs["hidden"])
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
            (op.packed_outputs_size,),
            dtype=hidden_buf.dtype,
        )
        op_args = [hidden_buf, weights_buf, packed_outputs_buf]
        output_buffers = {"packed_outputs": packed_outputs_buf}
    elif args.stage == "post-attn-rmsnorm-mlp-gate-up":
        next_token, inputs, expected_buffers = build_reference_mlp_gate_up(
            model, input_ids, args.max_seq_len
        )
        full_expected_buffers = expected_buffers
        hidden_buf = XRTTensor.from_torch(inputs["attn_residual"])
        packed_weights = torch.cat(
            [
                inputs["post_norm_weight"].flatten(),
                inputs["W_gate"].flatten(),
                inputs["W_up"].flatten(),
            ]
        ).contiguous()
        weights_buf = XRTTensor.from_torch(packed_weights)
        packed_outputs_buf = XRTTensor(
            (op.packed_outputs_size,),
            dtype=hidden_buf.dtype,
        )
        op_args = [hidden_buf, weights_buf, packed_outputs_buf]
        output_buffers = {"packed_outputs": packed_outputs_buf}
    elif args.stage == "post-attn-mlp-down-residual":
        next_token, inputs, expected_buffers = build_reference_mlp_down_residual(
            model, input_ids, args.max_seq_len
        )
        full_expected_buffers = expected_buffers
        hidden_buf = XRTTensor.from_torch(inputs["ffn_hidden"])
        residual_buf = XRTTensor.from_torch(inputs["attn_residual"])
        weights_buf = XRTTensor.from_torch(inputs["W_down"].flatten().contiguous())
        packed_outputs_buf = XRTTensor(
            (op.packed_outputs_size,),
            dtype=hidden_buf.dtype,
        )
        op_args = [hidden_buf, residual_buf, weights_buf, packed_outputs_buf]
        output_buffers = {"packed_outputs": packed_outputs_buf}
    elif args.stage == "post-attn-rmsnorm-full-mlp":
        next_token, inputs, expected_buffers = build_reference_full_mlp(
            model, input_ids, args.max_seq_len
        )
        full_expected_buffers = expected_buffers
        hidden_buf = XRTTensor.from_torch(inputs["attn_residual"])
        packed_weights = torch.cat(
            [
                inputs["post_norm_weight"].flatten(),
                inputs["W_gate"].flatten(),
                inputs["W_up"].flatten(),
                inputs["W_down"].flatten(),
            ]
        ).contiguous()
        weights_buf = XRTTensor.from_torch(packed_weights)
        packed_outputs_buf = XRTTensor(
            (op.packed_outputs_size,),
            dtype=hidden_buf.dtype,
        )
        op_args = [hidden_buf, weights_buf, packed_outputs_buf]
        output_buffers = {"packed_outputs": packed_outputs_buf}
    else:
        raise ValueError(f"unsupported stage after attention cleanup: {args.stage}")

    failed = False
    repeat_count = max(1, args.verify_repeat)
    for iteration in range(repeat_count):
        result = op_func(*op_args)
        print(f"iteration: {iteration}")
        print(f"prompt_next_token: {next_token}")
        print(f"npu_time_us: {result.npu_time / 1e3:.3f}")
        if args.stage == "input-rmsnorm":
            actual_buffers = {}
            for name, buffer in output_buffers.items():
                buffer.device = "npu"
                actual_buffers[name] = buffer.to_torch()
            local_expected_buffers = expected_buffers
        elif args.stage == "input-rmsnorm-qkv":
            packed_outputs_buf.device = "npu"
            packed_outputs = packed_outputs_buf.to_torch()
            actual_buffers = {
                "x_norm": packed_outputs[: op.q_output_base],
                "queries_raw": packed_outputs[op.q_output_base : op.k_output_base],
                "keys_raw": packed_outputs[op.k_output_base : op.v_output_base],
                "values": packed_outputs[op.v_output_base :],
            }
            local_expected_buffers = {
                **full_expected_buffers,
                "queries_raw": F.linear(
                    actual_buffers["x_norm"].view(1, 1, -1), inputs["W_q"]
                )
                .flatten()
                .contiguous(),
                "keys_raw": F.linear(
                    actual_buffers["x_norm"].view(1, 1, -1), inputs["W_k"]
                )
                .flatten()
                .contiguous(),
                "values": F.linear(
                    actual_buffers["x_norm"].view(1, 1, -1), inputs["W_v"]
                )
                .flatten()
                .contiguous(),
            }
        elif args.stage == "post-attn-rmsnorm-mlp-gate-up":
            packed_outputs_buf.device = "npu"
            packed_outputs = packed_outputs_buf.to_torch()
            actual_buffers = {
                "mlp_x_norm": packed_outputs[
                    op.mlp_x_norm_output_base : op.ffn_gate_output_base
                ],
                "ffn_gate": packed_outputs[
                    op.ffn_gate_output_base : op.ffn_up_output_base
                ],
                "ffn_up": packed_outputs[
                    op.ffn_up_output_base : op.ffn_gate_silu_output_base
                ],
                "ffn_gate_silu": packed_outputs[
                    op.ffn_gate_silu_output_base : op.ffn_hidden_output_base
                ],
                "ffn_hidden": packed_outputs[op.ffn_hidden_output_base :],
            }
            gate_local = F.linear(
                actual_buffers["mlp_x_norm"].view(1, 1, -1), inputs["W_gate"]
            ).flatten()
            up_local = F.linear(
                actual_buffers["mlp_x_norm"].view(1, 1, -1), inputs["W_up"]
            ).flatten()
            gate_silu_local = F.silu(actual_buffers["ffn_gate"])
            hidden_local = actual_buffers["ffn_gate_silu"] * actual_buffers["ffn_up"]
            local_expected_buffers = {
                **expected_buffers,
                "ffn_gate": gate_local.contiguous(),
                "ffn_up": up_local.contiguous(),
                "ffn_gate_silu": gate_silu_local.contiguous(),
                "ffn_hidden": hidden_local.contiguous(),
            }
        elif args.stage == "post-attn-mlp-down-residual":
            packed_outputs_buf.device = "npu"
            packed_outputs = packed_outputs_buf.to_torch()
            actual_buffers = {
                "ffn_out": packed_outputs[
                    op.ffn_out_output_base : op.layer_residual_output_base
                ],
                "layer_residual": packed_outputs[op.layer_residual_output_base :],
            }
            ffn_out_local = F.linear(
                inputs["ffn_hidden"].view(1, 1, -1), inputs["W_down"]
            ).flatten()
            residual_local = inputs["attn_residual"] + actual_buffers["ffn_out"]
            local_expected_buffers = {
                **expected_buffers,
                "ffn_out": ffn_out_local.contiguous(),
                "layer_residual": residual_local.contiguous(),
            }
        elif args.stage == "post-attn-rmsnorm-full-mlp":
            packed_outputs_buf.device = "npu"
            packed_outputs = packed_outputs_buf.to_torch()
            actual_buffers = {
                "mlp_x_norm": packed_outputs[
                    op.mlp_x_norm_output_base : op.ffn_gate_output_base
                ],
                "ffn_gate": packed_outputs[
                    op.ffn_gate_output_base : op.ffn_up_output_base
                ],
                "ffn_up": packed_outputs[
                    op.ffn_up_output_base : op.ffn_gate_silu_output_base
                ],
                "ffn_gate_silu": packed_outputs[
                    op.ffn_gate_silu_output_base : op.ffn_hidden_output_base
                ],
                "ffn_hidden": packed_outputs[
                    op.ffn_hidden_output_base : op.ffn_out_output_base
                ],
                "ffn_out": packed_outputs[
                    op.ffn_out_output_base : op.layer_residual_output_base
                ],
                "layer_residual": packed_outputs[op.layer_residual_output_base :],
            }
            gate_local = F.linear(
                actual_buffers["mlp_x_norm"].view(1, 1, -1), inputs["W_gate"]
            ).flatten()
            up_local = F.linear(
                actual_buffers["mlp_x_norm"].view(1, 1, -1), inputs["W_up"]
            ).flatten()
            gate_silu_local = F.silu(actual_buffers["ffn_gate"])
            hidden_local = actual_buffers["ffn_gate_silu"] * actual_buffers["ffn_up"]
            ffn_out_local = F.linear(
                actual_buffers["ffn_hidden"].view(1, 1, -1), inputs["W_down"]
            ).flatten()
            residual_local = inputs["attn_residual"] + actual_buffers["ffn_out"]
            local_expected_buffers = {
                **expected_buffers,
                "ffn_gate": gate_local.contiguous(),
                "ffn_up": up_local.contiguous(),
                "ffn_gate_silu": gate_silu_local.contiguous(),
                "ffn_hidden": hidden_local.contiguous(),
                "ffn_out": ffn_out_local.contiguous(),
                "layer_residual": residual_local.contiguous(),
            }
        else:
            raise ValueError(f"unsupported stage after attention cleanup: {args.stage}")

        for name, output in actual_buffers.items():
            expected = local_expected_buffers[name]
            rel_tol, abs_tol = verification_tolerance(args.stage, name)
            errors = verify_buffer(
                output,
                name,
                expected,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
            diff = (output.to(torch.float32) - expected.to(torch.float32)).abs()
            print(f"{name}_max_abs: {float(diff.max()):.6f}")
            print(f"{name}_mean_abs: {float(diff.mean()):.6f}")
            full_ref_name = {
                "attn_o_proj": "attn_out",
            }.get(name, name)
            if (
                args.stage
                in {
                    "input-rmsnorm-qkv",
                    "post-attn-rmsnorm-mlp-gate-up",
                    "post-attn-mlp-down-residual",
                    "post-attn-rmsnorm-full-mlp",
                }
                and full_ref_name in full_expected_buffers
                and name not in {"x_norm", "mlp_x_norm"}
            ):
                full_ref = full_expected_buffers[full_ref_name]
                full_diff = (
                    output.to(torch.float32) - full_ref.to(torch.float32)
                ).abs()
                print(f"{name}_full_ref_max_abs: {float(full_diff.max()):.6f}")
                print(f"{name}_full_ref_mean_abs: {float(full_diff.mean()):.6f}")
            print(f"{name}_errors: {len(errors)}")
            failed = failed or bool(errors)

    return failed
