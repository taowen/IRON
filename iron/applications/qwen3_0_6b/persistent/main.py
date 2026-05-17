#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import gc
import math
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

repo_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo_root))

from iron.applications.qwen3_0_6b.qwen3_cpu import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    Qwen3ForCausalLM,
    apply_rope,
    encode_prompt,
    repeat_kv,
    resolve_model_dir,
    rms_norm,
)
from iron.applications.qwen3_0_6b.qwen3_decode_reference import (  # noqa: E402
    Qwen3CachedReference,
    clone_decode_state,
)
from iron.applications.qwen3_0_6b.persistent.checks import (  # noqa: E402
    full_layer_local_invariants,
    print_tensor_check,
)
from iron.applications.qwen3_0_6b.persistent.diagnostics import (  # noqa: E402
    run_qkv_diagnostic_bundle,
    write_qkv_boundary_diagnostic_bundle,
)
from iron.applications.qwen3_0_6b.persistent.layout import (  # noqa: E402
    build_full_layer_inputs_for_layer,
    default_packed_weights_dir,
    host_owned_tensor,
    pack_full_layer_weights,
    validate_packed_weight_artifact,
    write_packed_weight_artifact,
    unpack_full_layer_outputs,
)
from iron.applications.qwen3_0_6b.persistent.generate import (  # noqa: E402
    prepare_fast_generate_buffers,
    run_full_layer_decode_hidden_fast,
    run_full_layer_decode_hidden_fast_chunked,
)
from iron.applications.qwen3_0_6b.persistent.ops import (  # noqa: E402
    Qwen3PersistentInputRMSNorm,
    Qwen3PersistentInputRMSNormQKV,
    Qwen3PersistentInputRMSNormQKVRopeCache,
    Qwen3PersistentInputRMSNormQKVRopeCacheScoresSoftmax,
    Qwen3PersistentInputRMSNormQKVRopeCacheScoresSoftmaxContext,
    Qwen3PersistentInputRMSNormQKVRopeCacheScoresSoftmaxContextOProj,
    Qwen3PersistentInputRMSNormQKVRopeCacheScoresSoftmaxContextOProjFullMLP,
    Qwen3PersistentSingleLayerFinalOnly,
    Qwen3PersistentTwoLayerFullLayer,
    Qwen3PersistentPostAttnMLPDownResidual,
    Qwen3PersistentPostAttnRMSNormFullMLP,
    Qwen3PersistentPostAttnRMSNormMLPGateUp,
    verification_tolerance,
)
from iron.applications.qwen3_0_6b.persistent.refs import (  # noqa: E402
    build_reference_multi_layer_full_layer,
    build_qk_pair_reference,
    build_reference_input,
    build_reference_full_layer,
    build_reference_full_mlp,
    build_reference_mlp_down_residual,
    build_reference_mlp_gate_up,
    build_reference_qkv,
    build_reference_qkv_rope_cache,
    full_layer_reference_from_inputs,
    print_structured_attention_error,
)
from iron.applications.qwen3_0_6b.qwen3_preflight import (  # noqa: E402
    run_persistent_artifact_preflight,
)
from iron.common.context import AIEContext  # noqa: E402
from iron.common.test_utils import verify_buffer  # noqa: E402
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor  # noqa: E402

FULL_LAYER_STAGE = "input-rmsnorm-qkv-rope-cache-scores-softmax-context-o-proj-full-mlp"
MULTI_LAYER_FULL_LAYER_STAGE = "multi-layer-full-layer"
SINGLE_LAYER_FINAL_ONLY_STAGE = "single-layer-final-only"
TWO_LAYER_FULL_LAYER_STAGE = "two-layer-full-layer"
GENERATE_STAGE = "generate"


def assert_standard_runtime_available():
    import pyxrt  # noqa: F401


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen3-0.6B persistent decode megakernel bring-up"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="HF repo id or local dir"
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--raw-prompt", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--build-dir", default="build_qwen3_persistent")
    parser.add_argument("--clean-build", action="store_true")
    parser.add_argument(
        "--prepare-weights",
        action="store_true",
        help=(
            "Prepack HF Qwen3 weights into a persistent bf16 artifact and exit. "
            "The default output is <model_dir>/qwen3_iron_packed."
        ),
    )
    parser.add_argument(
        "--packed-weights-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing weights.bf16.bin and manifest.json, or the "
            "output directory for --prepare-weights."
        ),
    )
    parser.add_argument(
        "--require-packed-weights",
        action="store_true",
        help="Require --fast-generate to load prepacked weights instead of runtime packing.",
    )
    parser.add_argument(
        "--stage",
        choices=[
            "input-rmsnorm",
            "input-rmsnorm-qkv",
            "input-rmsnorm-qkv-rope-cache",
            "input-rmsnorm-qkv-rope-cache-scores-softmax",
            "input-rmsnorm-qkv-rope-cache-scores-softmax-context",
            "input-rmsnorm-qkv-rope-cache-scores-softmax-context-o-proj",
            FULL_LAYER_STAGE,
            MULTI_LAYER_FULL_LAYER_STAGE,
            SINGLE_LAYER_FINAL_ONLY_STAGE,
            TWO_LAYER_FULL_LAYER_STAGE,
            GENERATE_STAGE,
            "post-attn-rmsnorm-mlp-gate-up",
            "post-attn-mlp-down-residual",
            "post-attn-rmsnorm-full-mlp",
        ],
        default="input-rmsnorm",
        help="Persistent bring-up stage to compile/run",
    )
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--verify-generate",
        action="store_true",
        help="Compare generated tokens against the cached CPU decode reference",
    )
    parser.add_argument(
        "--fast-generate",
        action="store_true",
        help=(
            "Reuse packed weight/cache XRT buffers for --stage generate. "
            "The default generate path keeps the older host round-trip debug flow."
        ),
    )
    parser.add_argument("--verify-repeat", type=int, default=1)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2,
        help="Number of new tokens to produce for --stage generate",
    )
    parser.add_argument(
        "--layer-chunk-size",
        type=int,
        choices=[1, 2],
        default=1,
        help="Number of full layers per persistent fast-generate invocation",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of transformer layers to run for multi-layer-full-layer",
    )
    parser.add_argument(
        "--diagnose-depth",
        action="store_true",
        help="Print per-layer boundary diagnostics for multi-layer-full-layer",
    )
    parser.add_argument(
        "--reference-num-layers",
        type=int,
        default=None,
        help=(
            "Number of layers used for prefill/token/cache reference in "
            "multi-layer-full-layer; defaults to --num-layers"
        ),
    )
    parser.add_argument(
        "--qkv-diagnostic-bundle",
        type=Path,
        default=None,
        help="Run an isolated QKV diagnostic from a bundle emitted by --diagnose-depth",
    )
    parser.add_argument("--dump-proof", action="store_true")
    return parser.parse_args()


def run_multi_layer_full_layer(
    args,
    model: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    op,
    op_func,
) -> bool:
    if not (1 <= args.num_layers <= model.config.num_hidden_layers):
        raise ValueError(
            f"num_layers must be in [1, {model.config.num_hidden_layers}], "
            f"got {args.num_layers}"
        )
    reference_num_layers = (
        args.num_layers
        if args.reference_num_layers is None
        else args.reference_num_layers
    )
    if not (args.num_layers <= reference_num_layers <= model.config.num_hidden_layers):
        raise ValueError(
            "reference_num_layers must be in "
            f"[{args.num_layers}, {model.config.num_hidden_layers}], "
            f"got {reference_num_layers}"
        )

    (
        next_token,
        position,
        initial_hidden,
        initial_state,
        expected_hidden,
        _expected_state,
    ) = build_reference_multi_layer_full_layer(
        model,
        input_ids,
        args.max_seq_len,
        args.num_layers,
        prefill_num_layers=reference_num_layers,
    )
    if position != op.position:
        raise RuntimeError(f"compiled position {op.position} != reference {position}")

    failed = False
    first_failure = None
    first_full_ref_drift = None
    first_v_diagnostic = None
    repeat_count = max(1, args.verify_repeat)
    for iteration in range(repeat_count):
        current_hidden = initial_hidden.clone()
        total_npu_time = 0
        print(f"iteration: {iteration}")
        print(f"prompt_next_token: {next_token}")
        print(f"decode_position: {position}")
        print(f"reference_num_layers: {reference_num_layers}")
        for layer_idx in range(args.num_layers):
            inputs = build_full_layer_inputs_for_layer(
                model,
                layer_idx,
                current_hidden,
                initial_state,
            )
            local_expected = full_layer_reference_from_inputs(
                model,
                inputs,
                position,
                args.max_seq_len,
            )
            hidden_buf = XRTTensor.from_torch(inputs["hidden"])
            weights_buf = XRTTensor.from_torch(pack_full_layer_weights(inputs))
            rope_angles_buf = XRTTensor.from_torch(inputs["rope_angles"])
            packed_outputs_buf = XRTTensor(
                (op.packed_outputs_size,),
                dtype=hidden_buf.dtype,
            )
            packed_cache_buf = XRTTensor.from_torch(inputs["initial_cache"].clone())
            result = op_func(
                hidden_buf,
                weights_buf,
                rope_angles_buf,
                packed_outputs_buf,
                packed_cache_buf,
            )
            total_npu_time += result.npu_time
            packed_outputs_buf.device = "npu"
            packed_cache_buf.device = "npu"
            packed_outputs = packed_outputs_buf.to_torch()
            packed_cache = packed_cache_buf.to_torch()
            actual = unpack_full_layer_outputs(op, packed_outputs)
            pending_v_diagnostic = False

            keys_cache = packed_cache[: op.cache_half_size].view(
                op.kv_heads, args.max_seq_len, op.head_dim
            )
            values_cache = packed_cache[op.cache_half_size :].view(
                op.kv_heads, args.max_seq_len, op.head_dim
            )
            actual["keys_cache_current"] = keys_cache[:, position, :].flatten()
            actual["values_cache_current"] = values_cache[:, position, :].flatten()
            v_context_stream = actual["v_context_stream"].view(
                op.kv_heads, op.max_seq_len, op.head_dim
            )
            actual["v_context_stream_current"] = v_context_stream[
                :, position, :
            ].flatten()

            local_invariants = full_layer_local_invariants(inputs, actual)
            add_errors = print_tensor_check(
                f"layer_{layer_idx}_residual_add",
                actual["layer_residual"],
                local_invariants["layer_residual_local"],
                rel_tol=0.04,
                abs_tol=1e-6,
            )
            failed = failed or bool(add_errors)
            if add_errors and first_failure is None:
                first_failure = f"layer_{layer_idx}_residual_add"
            print(f"layer_{layer_idx}_npu_time_us: {result.npu_time / 1e3:.3f}")

            if args.diagnose_depth:
                full_ref_tolerances = {
                    "attn_residual": verification_tolerance(
                        FULL_LAYER_STAGE, "attn_residual"
                    ),
                    "ffn_hidden": verification_tolerance(
                        FULL_LAYER_STAGE, "ffn_hidden"
                    ),
                    "ffn_out": verification_tolerance(FULL_LAYER_STAGE, "ffn_out"),
                    "layer_residual": verification_tolerance(
                        FULL_LAYER_STAGE, "layer_residual"
                    ),
                }
                local_checks = {
                    "ffn_out_local": (
                        actual["ffn_out"],
                        local_invariants["ffn_out_local"],
                        0.04,
                        1e-6,
                    ),
                    "values_cache_vs_context_current": (
                        actual["values_cache_current"],
                        actual["v_context_stream_current"],
                        0.05,
                        0.025,
                    ),
                }
                for name, (output, expected, rel_tol, abs_tol) in local_checks.items():
                    errors = print_tensor_check(
                        f"layer_{layer_idx}_{name}",
                        output,
                        expected,
                        rel_tol,
                        abs_tol,
                    )
                    failed = failed or bool(errors)
                    if errors and first_failure is None:
                        first_failure = f"layer_{layer_idx}_{name}"
                for name, (rel_tol, abs_tol) in full_ref_tolerances.items():
                    errors = print_tensor_check(
                        f"layer_{layer_idx}_{name}_full_ref",
                        actual[name],
                        local_expected[name],
                        rel_tol,
                        abs_tol,
                    )
                    if errors and first_full_ref_drift is None:
                        first_full_ref_drift = f"layer_{layer_idx}_{name}_full_ref"

            for name, abs_tol in {
                "keys_cache_current": 0.5,
                "v_context_stream_current": 0.025,
                "values_cache_current": 0.025,
            }.items():
                expected = (
                    local_expected["values_cache_current"]
                    if name == "v_context_stream_current"
                    else local_expected[name]
                )
                errors = print_tensor_check(
                    f"layer_{layer_idx}_{name}",
                    actual[name],
                    expected,
                    rel_tol=0.05,
                    abs_tol=abs_tol,
                )
                failed = failed or bool(errors)
                if errors and first_failure is None:
                    first_failure = f"layer_{layer_idx}_{name}"
                if (
                    errors
                    and args.diagnose_depth
                    and name
                    in {
                        "v_context_stream_current",
                        "values_cache_current",
                    }
                    and first_v_diagnostic is None
                ):
                    pending_v_diagnostic = True

            next_hidden = host_owned_tensor(actual["layer_residual"])
            if pending_v_diagnostic:
                full_layer_v = host_owned_tensor(actual["v_context_stream_current"])
                bundle_path = write_qkv_boundary_diagnostic_bundle(
                    build_dir=args.build_dir,
                    layer_idx=layer_idx,
                    inputs=inputs,
                    full_layer_v=full_layer_v,
                )
                first_v_diagnostic = f"layer_{layer_idx}_qkv_bundle:{bundle_path}"

            current_hidden = next_hidden

        final_rel_tol = 0.06
        final_abs_tol = 0.04 * max(1, args.num_layers)
        final_errors = print_tensor_check(
            "hidden_after_layers",
            current_hidden,
            expected_hidden,
            rel_tol=final_rel_tol,
            abs_tol=final_abs_tol,
        )
        print(f"npu_time_us_total: {total_npu_time / 1e3:.3f}")
        failed = failed or bool(final_errors)
        if final_errors and first_failure is None:
            first_failure = "hidden_after_layers"

    if first_failure is None:
        print("first_failure: none")
    else:
        print(f"first_failure: {first_failure}")
    if first_full_ref_drift is None:
        print("first_full_ref_drift: none")
    else:
        print(f"first_full_ref_drift: {first_full_ref_drift}")
    if first_v_diagnostic is None:
        print("first_v_diagnostic: none")
    else:
        print(f"first_v_diagnostic: {first_v_diagnostic}")

    return failed


def run_two_layer_full_layer(
    args,
    model: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    op,
    op_func,
) -> bool:
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
        num_layers=2,
        prefill_num_layers=2,
    )
    if position != op.position:
        raise RuntimeError(f"compiled position {op.position} != reference {position}")

    inputs0 = build_full_layer_inputs_for_layer(
        model,
        0,
        initial_hidden,
        initial_state,
    )
    inputs1 = build_full_layer_inputs_for_layer(
        model,
        1,
        initial_hidden,
        initial_state,
    )
    hidden_buf = XRTTensor.from_torch(initial_hidden)
    packed_weight_pair = torch.cat(
        [
            pack_full_layer_weights(inputs0),
            pack_full_layer_weights(inputs1),
        ]
    ).contiguous()
    weights_buf = XRTTensor.from_torch(packed_weight_pair)
    rope_angles_buf = XRTTensor.from_torch(inputs0["rope_angles"])

    failed = False
    repeat_count = max(1, args.verify_repeat)
    for iteration in range(repeat_count):
        output_buf = XRTTensor((op.packed_outputs_size,), dtype=hidden_buf.dtype)
        packed_cache_pair = torch.cat(
            [
                inputs0["initial_cache"].clone(),
                inputs1["initial_cache"].clone(),
            ]
        ).contiguous()
        cache_buf = XRTTensor.from_torch(packed_cache_pair)
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
        cache_pair = cache_buf.to_torch()
        cache0 = cache_pair[: op.packed_cache_size]
        cache1 = cache_pair[op.packed_cache_size :]
        actual_keys0 = cache0[: op.cache_half_size].view(
            op.kv_heads, args.max_seq_len, op.head_dim
        )
        actual_values0 = cache0[op.cache_half_size :].view(
            op.kv_heads, args.max_seq_len, op.head_dim
        )
        actual_keys1 = cache1[: op.cache_half_size].view(
            op.kv_heads, args.max_seq_len, op.head_dim
        )
        actual_values1 = cache1[op.cache_half_size :].view(
            op.kv_heads, args.max_seq_len, op.head_dim
        )

        checks = {
            "two_layer_hidden": (actual_hidden, expected_hidden, 0.06, 0.08),
            "layer0_keys_cache_current": (
                actual_keys0[:, position, :].flatten(),
                expected_state.keys[0][:, position, :].flatten(),
                0.05,
                0.5,
            ),
            "layer0_values_cache_current": (
                actual_values0[:, position, :].flatten(),
                expected_state.values[0][:, position, :].flatten(),
                0.05,
                0.025,
            ),
            "layer1_keys_cache_current": (
                actual_keys1[:, position, :].flatten(),
                expected_state.keys[1][:, position, :].flatten(),
                0.05,
                0.5,
            ),
            "layer1_values_cache_current": (
                actual_values1[:, position, :].flatten(),
                expected_state.values[1][:, position, :].flatten(),
                0.05,
                0.025,
            ),
        }
        for name, (actual, expected, rel_tol, abs_tol) in checks.items():
            errors = print_tensor_check(name, actual, expected, rel_tol, abs_tol)
            failed = failed or bool(errors)

    return failed


def run_single_layer_final_only(
    args,
    model: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    op,
    op_func,
) -> bool:
    next_token, position, inputs, expected_buffers = build_reference_full_layer(
        model,
        input_ids,
        args.max_seq_len,
    )
    if position != op.position:
        raise RuntimeError(f"compiled position {op.position} != reference {position}")

    hidden_buf = XRTTensor.from_torch(inputs["hidden"])
    weights_buf = XRTTensor.from_torch(pack_full_layer_weights(inputs))
    rope_angles_buf = XRTTensor.from_torch(inputs["rope_angles"])
    initial_cache = inputs["initial_cache"].clone()

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
        packed_cache = cache_buf.to_torch()
        keys_cache = packed_cache[: op.cache_half_size].view(
            op.kv_heads, args.max_seq_len, op.head_dim
        )
        values_cache = packed_cache[op.cache_half_size :].view(
            op.kv_heads, args.max_seq_len, op.head_dim
        )

        checks = {
            "layer_residual": (
                actual_hidden,
                expected_buffers["layer_residual"],
                0.06,
                0.04,
            ),
            "keys_cache_current": (
                keys_cache[:, position, :].flatten(),
                expected_buffers["keys"],
                0.05,
                0.5,
            ),
            "values_cache_current": (
                values_cache[:, position, :].flatten(),
                expected_buffers["values"],
                0.05,
                0.025,
            ),
        }
        for name, (actual, expected, rel_tol, abs_tol) in checks.items():
            errors = print_tensor_check(name, actual, expected, rel_tol, abs_tol)
            failed = failed or bool(errors)

    return failed


def make_full_layer_op_for_position(
    args,
    model: Qwen3ForCausalLM,
    context: AIEContext,
    position: int,
):
    return Qwen3PersistentInputRMSNormQKVRopeCacheScoresSoftmaxContextOProjFullMLP(
        hidden_size=model.config.hidden_size,
        q_size=model.config.num_attention_heads * model.config.head_dim,
        kv_size=model.config.num_key_value_heads * model.config.head_dim,
        head_dim=model.config.head_dim,
        max_seq_len=args.max_seq_len,
        position=position,
        intermediate_size=model.config.intermediate_size,
        epsilon=model.config.rms_norm_eps,
        context=context,
    )


def make_single_layer_final_only_op_for_position(
    args,
    model: Qwen3ForCausalLM,
    context: AIEContext,
    position: int,
):
    return Qwen3PersistentSingleLayerFinalOnly(
        hidden_size=model.config.hidden_size,
        q_size=model.config.num_attention_heads * model.config.head_dim,
        kv_size=model.config.num_key_value_heads * model.config.head_dim,
        head_dim=model.config.head_dim,
        max_seq_len=args.max_seq_len,
        position=position,
        intermediate_size=model.config.intermediate_size,
        epsilon=model.config.rms_norm_eps,
        context=context,
    )


def make_two_layer_op_for_position(
    args,
    model: Qwen3ForCausalLM,
    context: AIEContext,
    position: int,
):
    return Qwen3PersistentTwoLayerFullLayer(
        hidden_size=model.config.hidden_size,
        q_size=model.config.num_attention_heads * model.config.head_dim,
        kv_size=model.config.num_key_value_heads * model.config.head_dim,
        head_dim=model.config.head_dim,
        max_seq_len=args.max_seq_len,
        position=position,
        intermediate_size=model.config.intermediate_size,
        epsilon=model.config.rms_norm_eps,
        context=context,
    )


def compile_full_layer_op_for_position(
    args,
    model: Qwen3ForCausalLM,
    context: AIEContext,
    position: int,
):
    op = make_full_layer_op_for_position(args, model, context, position)
    start = time.perf_counter()
    op.compile()
    compile_s = time.perf_counter() - start
    preflight = run_persistent_artifact_preflight(
        mlir_path=Path(op.xclbin_artifact.mlir_input.filename),
        arg_specs=len(op.get_arg_spec()),
    )
    print(
        f"generate_position_{position}_compile_s: {compile_s:.3f} "
        f"operator_name={op.name}"
    )
    print(
        f"generate_position_{position}_preflight: ok "
        f"runtime_memrefs={preflight.runtime_memrefs} "
        f"arg_specs={preflight.arg_specs} "
        f"metadata_host_bos={preflight.metadata_host_bos} "
        f"compute_cores={preflight.compute_cores} "
        f"max_fifo_buffered_bytes={preflight.max_fifo_buffered_bytes} "
        f"max_dma_tasks_per_fifo={preflight.max_dma_tasks_per_fifo} "
        f"max_tile_inputs={preflight.max_compute_tile_inputs} "
        f"max_tile_outputs={preflight.max_compute_tile_outputs} "
        f"non_advancing_acquires={preflight.non_advancing_acquires}"
    )
    return op, op.get_callable()


def compile_single_layer_final_only_op_for_position(
    args,
    model: Qwen3ForCausalLM,
    context: AIEContext,
    position: int,
):
    op = make_single_layer_final_only_op_for_position(args, model, context, position)
    start = time.perf_counter()
    op.compile()
    compile_s = time.perf_counter() - start
    preflight = run_persistent_artifact_preflight(
        mlir_path=Path(op.xclbin_artifact.mlir_input.filename),
        arg_specs=len(op.get_arg_spec()),
    )
    print(
        f"generate_position_{position}_single_final_compile_s: {compile_s:.3f} "
        f"operator_name={op.name}"
    )
    print(
        f"generate_position_{position}_single_final_preflight: ok "
        f"runtime_memrefs={preflight.runtime_memrefs} "
        f"arg_specs={preflight.arg_specs} "
        f"metadata_host_bos={preflight.metadata_host_bos} "
        f"compute_cores={preflight.compute_cores} "
        f"max_fifo_buffered_bytes={preflight.max_fifo_buffered_bytes} "
        f"max_dma_tasks_per_fifo={preflight.max_dma_tasks_per_fifo} "
        f"max_tile_inputs={preflight.max_compute_tile_inputs} "
        f"max_tile_outputs={preflight.max_compute_tile_outputs} "
        f"non_advancing_acquires={preflight.non_advancing_acquires}"
    )
    return op, op.get_callable()


def compile_two_layer_op_for_position(
    args,
    model: Qwen3ForCausalLM,
    context: AIEContext,
    position: int,
):
    op = make_two_layer_op_for_position(args, model, context, position)
    start = time.perf_counter()
    op.compile()
    compile_s = time.perf_counter() - start
    preflight = run_persistent_artifact_preflight(
        mlir_path=Path(op.xclbin_artifact.mlir_input.filename),
        arg_specs=len(op.get_arg_spec()),
    )
    print(
        f"generate_position_{position}_two_layer_compile_s: {compile_s:.3f} "
        f"operator_name={op.name}"
    )
    print(
        f"generate_position_{position}_two_layer_preflight: ok "
        f"runtime_memrefs={preflight.runtime_memrefs} "
        f"arg_specs={preflight.arg_specs} "
        f"metadata_host_bos={preflight.metadata_host_bos} "
        f"compute_cores={preflight.compute_cores} "
        f"max_fifo_buffered_bytes={preflight.max_fifo_buffered_bytes} "
        f"max_dma_tasks_per_fifo={preflight.max_dma_tasks_per_fifo} "
        f"max_tile_inputs={preflight.max_compute_tile_inputs} "
        f"max_tile_outputs={preflight.max_compute_tile_outputs} "
        f"non_advancing_acquires={preflight.non_advancing_acquires}"
    )
    return op, op.get_callable()


def final_logits_from_hidden(
    model: Qwen3ForCausalLM,
    hidden: torch.Tensor,
) -> torch.Tensor:
    x = rms_norm(
        hidden.view(1, 1, -1).to(dtype=model.dtype),
        model.w("model.norm.weight"),
        model.config.rms_norm_eps,
    )
    lm_head = (
        model.w("model.embed_tokens.weight")
        if model.config.tie_word_embeddings
        else model.w("lm_head.weight")
    )
    return F.linear(x, lm_head)


def run_full_layer_decode_hidden(
    args,
    model: Qwen3ForCausalLM,
    state,
    token_id: int,
    op,
    op_func,
) -> tuple[torch.Tensor, float]:
    if state.position != op.position:
        raise RuntimeError(
            f"compiled position {op.position} != decode state position {state.position}"
        )

    current_hidden = host_owned_tensor(
        model.embed(torch.tensor([[token_id]], dtype=torch.long)).flatten()
    )
    total_npu_time = 0.0
    for layer_idx in range(model.config.num_hidden_layers):
        inputs = build_full_layer_inputs_for_layer(
            model,
            layer_idx,
            current_hidden,
            state,
        )
        hidden_buf = XRTTensor.from_torch(inputs["hidden"])
        weights_buf = XRTTensor.from_torch(pack_full_layer_weights(inputs))
        rope_angles_buf = XRTTensor.from_torch(inputs["rope_angles"])
        packed_outputs_buf = XRTTensor(
            (op.packed_outputs_size,),
            dtype=hidden_buf.dtype,
        )
        packed_cache_buf = XRTTensor.from_torch(inputs["initial_cache"].clone())
        result = op_func(
            hidden_buf,
            weights_buf,
            rope_angles_buf,
            packed_outputs_buf,
            packed_cache_buf,
        )
        total_npu_time += result.npu_time
        packed_outputs_buf.device = "npu"
        packed_cache_buf.device = "npu"
        packed_outputs = packed_outputs_buf.to_torch()
        packed_cache = packed_cache_buf.to_torch()
        actual = unpack_full_layer_outputs(op, packed_outputs)

        keys_cache = packed_cache[: op.cache_half_size].view(
            op.kv_heads, args.max_seq_len, op.head_dim
        )
        values_cache = packed_cache[op.cache_half_size :].view(
            op.kv_heads, args.max_seq_len, op.head_dim
        )
        state.keys[layer_idx] = host_owned_tensor(keys_cache)
        state.values[layer_idx] = host_owned_tensor(values_cache)
        current_hidden = host_owned_tensor(actual["layer_residual"])

    state.position += 1
    return current_hidden, total_npu_time


def run_generate(
    args,
    model: Qwen3ForCausalLM,
    tokenizer,
    input_ids: torch.Tensor,
    context: AIEContext,
) -> bool:
    if args.layer_chunk_size != 1 and not args.fast_generate:
        raise ValueError("--layer-chunk-size 2 requires --fast-generate")
    if args.max_new_tokens < 1:
        raise ValueError(f"max_new_tokens must be positive, got {args.max_new_tokens}")
    if input_ids.shape[1] + args.max_new_tokens > args.max_seq_len:
        raise ValueError(
            f"prompt length {input_ids.shape[1]} + max_new_tokens "
            f"{args.max_new_tokens} exceeds max_seq_len {args.max_seq_len}"
        )

    ref = Qwen3CachedReference(
        model,
        args.max_seq_len,
        num_layers=model.config.num_hidden_layers,
    )
    prefill_logits, prefill_state = ref.prefill(input_ids)
    first_token = int(torch.argmax(prefill_logits[:, -1, :], dim=-1).item())
    first_text = tokenizer.decode([first_token], skip_special_tokens=True)
    print("stage: generate")
    if args.fast_generate:
        chunk_desc = (
            "single-layer final-only"
            if args.layer_chunk_size == 1
            else "two-layer final-only"
        )
        print(
            f"implementation: persistent {chunk_desc} decode with cached XRT "
            "weights/cache + CPU final norm/lm head"
        )
    else:
        print("implementation: persistent full-layer decode + CPU final norm/lm head")
    print(f"prompt_len: {input_ids.shape[1]}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"layer_chunk_size: {args.layer_chunk_size}")
    print(f"prompt_next_token: {first_token} text={first_text!r}")

    op_cache = {}
    single_final_op_cache = {}
    two_layer_op_cache = {}

    def get_position_op(position: int):
        if position not in op_cache:
            op_cache[position] = compile_full_layer_op_for_position(
                args,
                model,
                context,
                position,
            )
        return op_cache[position]

    def get_position_single_final_op(position: int):
        if position not in single_final_op_cache:
            single_final_op_cache[position] = (
                compile_single_layer_final_only_op_for_position(
                    args,
                    model,
                    context,
                    position,
                )
            )
        return single_final_op_cache[position]

    def get_position_two_layer_op(position: int):
        if position not in two_layer_op_cache:
            two_layer_op_cache[position] = compile_two_layer_op_for_position(
                args,
                model,
                context,
                position,
            )
        return two_layer_op_cache[position]

    if args.compile_only:
        if args.layer_chunk_size == 2:
            get_position_two_layer_op(prefill_state.position)
            if model.config.num_hidden_layers % 2:
                get_position_single_final_op(prefill_state.position)
        elif args.fast_generate:
            get_position_single_final_op(prefill_state.position)
        else:
            get_position_op(prefill_state.position)
        return False

    generated_tokens = [first_token]
    npu_state = clone_decode_state(prefill_state)
    ref_state = clone_decode_state(prefill_state)
    fast_buffers = None
    if args.fast_generate:
        if args.layer_chunk_size == 2:
            first_two_layer_op, _first_two_layer_op_func = get_position_two_layer_op(
                prefill_state.position
            )
            first_op = first_two_layer_op
            if model.config.num_hidden_layers % 2:
                first_op, _first_op_func = get_position_single_final_op(
                    prefill_state.position
                )
        else:
            first_op, _first_op_func = get_position_single_final_op(
                prefill_state.position
            )
            first_two_layer_op = None
        setup_start = time.perf_counter()
        fast_buffers = prepare_fast_generate_buffers(
            model,
            prefill_state,
            first_op,
            packed_weights_dir=args.packed_weights_dir,
            require_packed_weights=args.require_packed_weights,
            layer_chunk_size=args.layer_chunk_size,
            two_layer_op=first_two_layer_op,
        )
        setup_s = time.perf_counter() - setup_start
        setup_timing = fast_buffers.timing
        print(f"fast_generate_setup_s: {setup_s:.6f}")
        print(f"fast_generate_weight_source: {setup_timing.weight_source}")
        print(f"fast_generate_weight_pack_s: {setup_timing.weight_pack_s:.6f}")
        print(
            f"fast_generate_weight_disk_load_s: {setup_timing.weight_disk_load_s:.6f}"
        )
        print(f"fast_generate_weight_xrt_s: {setup_timing.weight_xrt_s:.6f}")
        print(f"fast_generate_cache_xrt_s: {setup_timing.cache_xrt_s:.6f}")
    failed = False

    for token_idx in range(1, args.max_new_tokens):
        current_token = generated_tokens[-1]
        position = npu_state.position
        op = None
        op_func = None
        two_layer_op = None
        two_layer_op_func = None
        if args.fast_generate and args.layer_chunk_size == 2:
            two_layer_op, two_layer_op_func = get_position_two_layer_op(position)
            if model.config.num_hidden_layers % 2:
                op, op_func = get_position_single_final_op(position)
        elif args.fast_generate:
            op, op_func = get_position_single_final_op(position)
        else:
            op, op_func = get_position_op(position)
        start = time.perf_counter()
        fast_timing = None
        if args.fast_generate:
            if args.layer_chunk_size == 2:
                npu_hidden, npu_time, fast_timing = (
                    run_full_layer_decode_hidden_fast_chunked(
                        model,
                        current_token,
                        position,
                        op,
                        op_func,
                        two_layer_op,
                        two_layer_op_func,
                        fast_buffers,
                    )
                )
            else:
                npu_hidden, npu_time, fast_timing = run_full_layer_decode_hidden_fast(
                    model,
                    current_token,
                    position,
                    op,
                    op_func,
                    fast_buffers,
                )
            npu_state.position += 1
        else:
            npu_hidden, npu_time = run_full_layer_decode_hidden(
                args,
                model,
                npu_state,
                current_token,
                op,
                op_func,
            )
        decode_s = time.perf_counter() - start
        final_start = time.perf_counter()
        npu_logits = final_logits_from_hidden(model, npu_hidden)
        final_s = time.perf_counter() - final_start
        npu_next = int(torch.argmax(npu_logits[:, -1, :], dim=-1).item())
        npu_text = tokenizer.decode([npu_next], skip_special_tokens=True)
        generated_tokens.append(npu_next)

        print(f"token_step: {token_idx}")
        print(f"decode_position: {position}")
        print(f"npu_layer_time_us_total: {npu_time / 1e3:.3f}")
        print(f"decode_s: {decode_s:.6f}")
        print(f"cpu_final_lm_head_s: {final_s:.6f}")
        if fast_timing is not None:
            print(f"fast_hidden_sync_s: {fast_timing.hidden_sync_s:.6f}")
            print(f"fast_rope_sync_s: {fast_timing.rope_sync_s:.6f}")
            print(f"fast_op_call_s: {fast_timing.op_call_s:.6f}")
            print(f"fast_output_drain_s: {fast_timing.output_drain_s:.6f}")
            print(
                "fast_layer_residual_clone_s: "
                f"{fast_timing.layer_residual_clone_s:.6f}"
            )
        print(f"npu_next_token: {npu_next} text={npu_text!r}")

        if args.verify_generate:
            ref_logits, ref_state = ref.decode(current_token, ref_state)
            ref_next = int(torch.argmax(ref_logits[:, -1, :], dim=-1).item())
            ref_text = tokenizer.decode([ref_next], skip_special_tokens=True)
            diff = (npu_logits.to(torch.float32) - ref_logits.to(torch.float32)).abs()
            token_match = npu_next == ref_next
            failed = failed or not token_match
            print(f"ref_next_token: {ref_next} text={ref_text!r}")
            print(f"token_match: {token_match}")
            print(f"logits_max_abs: {float(diff.max()):.6f}")
            print(f"logits_mean_abs: {float(diff.mean()):.6f}")
            if not token_match:
                top_npu = torch.topk(npu_logits[0, -1].to(torch.float32), k=5)
                top_ref = torch.topk(ref_logits[0, -1].to(torch.float32), k=5)
                print(f"npu_top5_ids: {top_npu.indices.tolist()}")
                print(f"npu_top5_values: {[float(v) for v in top_npu.values]}")
                print(f"ref_top5_ids: {top_ref.indices.tolist()}")
                print(f"ref_top5_values: {[float(v) for v in top_ref.values]}")

        if npu_next == model.config.eos_token_id:
            break

    generated = torch.cat(
        [
            input_ids,
            torch.tensor([generated_tokens], dtype=torch.long),
        ],
        dim=1,
    )
    new_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"generated_ids: {generated.tolist()[0]}")
    print(f"new_token_ids: {generated_tokens}")
    print(f"new_text: {new_text!r}")
    print(f"final_decode_position: {npu_state.position}")
    return failed


def main():
    args = parse_args()
    if args.clean_build:
        shutil.rmtree(args.build_dir, ignore_errors=True)

    model_dir = resolve_model_dir(args.model, args.revision)
    model = Qwen3ForCausalLM(model_dir)
    if model.config.hidden_size != 1024:
        raise ValueError(
            f"expected Qwen3-0.6B hidden_size=1024, got {model.config.hidden_size}"
        )
    if model.config.intermediate_size != 3072:
        raise ValueError(
            "expected Qwen3-0.6B intermediate_size=3072, "
            f"got {model.config.intermediate_size}"
        )
    if args.max_seq_len < 256:
        raise ValueError("max_seq_len must be at least 256")
    if args.packed_weights_dir is None:
        args.packed_weights_dir = default_packed_weights_dir(model_dir)

    if args.prepare_weights:
        op_for_layout = make_full_layer_op_for_position(
            args,
            model,
            AIEContext(build_dir=args.build_dir),
            position=0,
        )
        start = time.perf_counter()
        manifest = write_packed_weight_artifact(
            model,
            args.packed_weights_dir,
            expected_per_layer_numel=op_for_layout.packed_weights_size,
        )
        elapsed = time.perf_counter() - start
        print("stage: prepare-weights")
        print(f"packed_weights_dir: {args.packed_weights_dir}")
        print(f"packed_weights_file: {args.packed_weights_dir / 'weights.bf16.bin'}")
        print(f"packed_manifest_file: {args.packed_weights_dir / 'manifest.json'}")
        print(f"packed_layers: {manifest['num_layers']}")
        print(f"packed_per_layer_numel: {manifest['per_layer_numel']}")
        print(f"packed_total_bytes: {manifest['total_bytes']}")
        print(f"prepare_weights_s: {elapsed:.6f}")
        return

    if args.require_packed_weights:
        validate_packed_weight_artifact(
            model,
            args.packed_weights_dir,
            expected_per_layer_numel=make_full_layer_op_for_position(
                args,
                model,
                AIEContext(build_dir=args.build_dir),
                position=0,
            ).packed_weights_size,
        )
        if not args.fast_generate:
            raise ValueError("--require-packed-weights is only used by --fast-generate")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    input_ids = encode_prompt(
        tokenizer,
        args.prompt,
        raw_prompt=args.raw_prompt,
        enable_thinking=args.enable_thinking,
    )
    if not args.compile_only:
        assert_standard_runtime_available()

    context = AIEContext(build_dir=args.build_dir)
    if args.qkv_diagnostic_bundle is not None:
        failed = run_qkv_diagnostic_bundle(args.qkv_diagnostic_bundle, model, context)
        if failed:
            raise SystemExit(1)
        return
    if args.stage == GENERATE_STAGE:
        failed = run_generate(args, model, tokenizer, input_ids, context)
        if args.verify_generate and failed:
            raise SystemExit(1)
        gc.collect()
        return

    if args.stage == "input-rmsnorm":
        op = Qwen3PersistentInputRMSNorm(
            hidden_size=model.config.hidden_size,
            epsilon=model.config.rms_norm_eps,
            context=context,
        )
    elif args.stage == "input-rmsnorm-qkv":
        op = Qwen3PersistentInputRMSNormQKV(
            hidden_size=model.config.hidden_size,
            q_size=model.config.num_attention_heads * model.config.head_dim,
            kv_size=model.config.num_key_value_heads * model.config.head_dim,
            epsilon=model.config.rms_norm_eps,
            context=context,
        )
    elif args.stage == "input-rmsnorm-qkv-rope-cache":
        op = Qwen3PersistentInputRMSNormQKVRopeCache(
            hidden_size=model.config.hidden_size,
            q_size=model.config.num_attention_heads * model.config.head_dim,
            kv_size=model.config.num_key_value_heads * model.config.head_dim,
            head_dim=model.config.head_dim,
            max_seq_len=args.max_seq_len,
            position=input_ids.shape[1],
            epsilon=model.config.rms_norm_eps,
            context=context,
        )
    elif args.stage == "input-rmsnorm-qkv-rope-cache-scores-softmax":
        op = Qwen3PersistentInputRMSNormQKVRopeCacheScoresSoftmax(
            hidden_size=model.config.hidden_size,
            q_size=model.config.num_attention_heads * model.config.head_dim,
            kv_size=model.config.num_key_value_heads * model.config.head_dim,
            head_dim=model.config.head_dim,
            max_seq_len=args.max_seq_len,
            position=input_ids.shape[1],
            epsilon=model.config.rms_norm_eps,
            context=context,
        )
    elif args.stage == "input-rmsnorm-qkv-rope-cache-scores-softmax-context":
        op = Qwen3PersistentInputRMSNormQKVRopeCacheScoresSoftmaxContext(
            hidden_size=model.config.hidden_size,
            q_size=model.config.num_attention_heads * model.config.head_dim,
            kv_size=model.config.num_key_value_heads * model.config.head_dim,
            head_dim=model.config.head_dim,
            max_seq_len=args.max_seq_len,
            position=input_ids.shape[1],
            epsilon=model.config.rms_norm_eps,
            context=context,
        )
    elif args.stage == "post-attn-rmsnorm-mlp-gate-up":
        op = Qwen3PersistentPostAttnRMSNormMLPGateUp(
            hidden_size=model.config.hidden_size,
            intermediate_size=model.config.intermediate_size,
            epsilon=model.config.rms_norm_eps,
            context=context,
        )
    elif args.stage == "post-attn-mlp-down-residual":
        op = Qwen3PersistentPostAttnMLPDownResidual(
            hidden_size=model.config.hidden_size,
            intermediate_size=model.config.intermediate_size,
            context=context,
        )
    elif args.stage == "post-attn-rmsnorm-full-mlp":
        op = Qwen3PersistentPostAttnRMSNormFullMLP(
            hidden_size=model.config.hidden_size,
            intermediate_size=model.config.intermediate_size,
            epsilon=model.config.rms_norm_eps,
            context=context,
        )
    elif args.stage == TWO_LAYER_FULL_LAYER_STAGE:
        op = Qwen3PersistentTwoLayerFullLayer(
            hidden_size=model.config.hidden_size,
            q_size=model.config.num_attention_heads * model.config.head_dim,
            kv_size=model.config.num_key_value_heads * model.config.head_dim,
            head_dim=model.config.head_dim,
            max_seq_len=args.max_seq_len,
            position=input_ids.shape[1],
            intermediate_size=model.config.intermediate_size,
            epsilon=model.config.rms_norm_eps,
            context=context,
        )
    elif args.stage == SINGLE_LAYER_FINAL_ONLY_STAGE:
        op = Qwen3PersistentSingleLayerFinalOnly(
            hidden_size=model.config.hidden_size,
            q_size=model.config.num_attention_heads * model.config.head_dim,
            kv_size=model.config.num_key_value_heads * model.config.head_dim,
            head_dim=model.config.head_dim,
            max_seq_len=args.max_seq_len,
            position=input_ids.shape[1],
            intermediate_size=model.config.intermediate_size,
            epsilon=model.config.rms_norm_eps,
            context=context,
        )
    elif args.stage in {FULL_LAYER_STAGE, MULTI_LAYER_FULL_LAYER_STAGE}:
        op = Qwen3PersistentInputRMSNormQKVRopeCacheScoresSoftmaxContextOProjFullMLP(
            hidden_size=model.config.hidden_size,
            q_size=model.config.num_attention_heads * model.config.head_dim,
            kv_size=model.config.num_key_value_heads * model.config.head_dim,
            head_dim=model.config.head_dim,
            max_seq_len=args.max_seq_len,
            position=input_ids.shape[1],
            intermediate_size=model.config.intermediate_size,
            epsilon=model.config.rms_norm_eps,
            context=context,
        )
    else:
        op = Qwen3PersistentInputRMSNormQKVRopeCacheScoresSoftmaxContextOProj(
            hidden_size=model.config.hidden_size,
            q_size=model.config.num_attention_heads * model.config.head_dim,
            kv_size=model.config.num_key_value_heads * model.config.head_dim,
            head_dim=model.config.head_dim,
            max_seq_len=args.max_seq_len,
            position=input_ids.shape[1],
            epsilon=model.config.rms_norm_eps,
            context=context,
        )

    start = time.perf_counter()
    op.compile()
    compile_s = time.perf_counter() - start
    preflight = run_persistent_artifact_preflight(
        mlir_path=Path(op.xclbin_artifact.mlir_input.filename),
        arg_specs=len(op.get_arg_spec()),
    )
    print(f"stage: {args.stage}")
    print("implementation: hand-authored IRON Program/Worker/ObjectFifo")
    print(f"operator_name: {op.name}")
    print(f"compile_s: {compile_s:.3f}")
    print(
        "preflight: ok "
        f"runtime_memrefs={preflight.runtime_memrefs} "
        f"arg_specs={preflight.arg_specs} "
        f"metadata_host_bos={preflight.metadata_host_bos} "
        f"compute_cores={preflight.compute_cores} "
        f"max_fifo_buffered_bytes={preflight.max_fifo_buffered_bytes} "
        f"max_dma_tasks_per_fifo={preflight.max_dma_tasks_per_fifo} "
        f"max_tile_inputs={preflight.max_compute_tile_inputs} "
        f"max_tile_outputs={preflight.max_compute_tile_outputs} "
        f"non_advancing_acquires={preflight.non_advancing_acquires}"
    )
    if args.dump_proof:
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
        elif args.stage == "input-rmsnorm-qkv-rope-cache":
            print(
                "dispatch_shape: hidden[1024] + packed QKV/norm weights + "
                "rope_angles[128] -> x_norm, Q/K/V raw, Q/K norm, Q/K rope, KV cache"
            )
            print(
                "runtime_bos: hidden[1024], "
                f"packed_weights[{op.packed_weights_size}], "
                "rope_angles[128], "
                f"packed_outputs[{op.packed_outputs_size}], "
                f"packed_cache[{op.packed_cache_size}]"
            )
            print(f"decode_position: {op.position}")
            print(f"qkv_columns: {op.num_aie_columns}")
            print(
                "worker_graph: hidden_fifo -> rmsnorm_worker -> weight_worker -> "
                "xnorm broadcast -> Q/K/V matvec -> Q/K norm -> Q/K RoPE -> KV cache drains"
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
        elif args.stage == FULL_LAYER_STAGE:
            print(
                "dispatch_shape: hidden[1024] + packed attention/MLP weights + "
                "rope_angles[128] + K/V cache -> attention residual -> "
                "post RMSNorm -> gate/up -> SiLU/mul -> down/residual"
            )
            print(
                "runtime_bos: hidden[1024], "
                f"packed_weights[{op.packed_weights_size}], "
                "rope_angles[128], "
                f"packed_outputs[{op.packed_outputs_size}], "
                f"packed_cache[{op.packed_cache_size}]"
            )
            print(f"decode_position: {op.position}")
            print(
                "worker_graph: Q/K matvec -> packed metadata norm+RoPE, "
                "attention context/O-proj full residual, then postnorm+gate/up "
                "-> fused SiLU/mul -> down_proj -> residual add"
            )
        elif args.stage == MULTI_LAYER_FULL_LAYER_STAGE:
            print(
                "dispatch_shape: one-token decode hidden[1024] through "
                f"{args.num_layers} sequential full-layer invocations"
            )
            print(
                "runtime_bos_per_layer: hidden[1024], "
                f"packed_weights[{op.packed_weights_size}], "
                "rope_angles[128], "
                f"packed_outputs[{op.packed_outputs_size}], "
                f"packed_cache[{op.packed_cache_size}]"
            )
            print(f"decode_position: {op.position}")
            print(
                "worker_graph: reuse the accepted full-layer persistent graph; "
                "host switches layer weights and per-layer KV cache slices"
            )
        elif args.stage == SINGLE_LAYER_FINAL_ONLY_STAGE:
            print(
                "dispatch_shape: hidden[1024] through one full layer with only "
                "final hidden[1024] and KV cache current position drained"
            )
            print(
                "runtime_bos: hidden[1024], "
                f"packed_weights[{op.packed_weights_size}], "
                "rope_angles[128], "
                f"final_hidden[{op.packed_outputs_size}], "
                f"packed_cache[{op.packed_cache_size}]"
            )
            print(f"decode_position: {op.position}")
            print(
                "worker_graph: accepted full-layer graph with debug drains removed; "
                "attention/MLP internal FIFOs stay on chip and layer_residual is "
                "the only host output"
            )
        elif args.stage == TWO_LAYER_FULL_LAYER_STAGE:
            print(
                "dispatch_shape: hidden[1024] through layers 0 and 1 in one "
                "persistent invocation"
            )
            print(
                "runtime_bos: hidden[1024], "
                f"weight_pair[{op.packed_weight_pair_size}], "
                "rope_angles[128], "
                f"final_hidden[{op.packed_outputs_size}], "
                f"cache_pair[{op.packed_cache_pair_size}]"
            )
            print(f"decode_position: {op.position}")
            print(
                "worker_graph: one full-layer worker graph loops twice; "
                "layer0 residual is routed back as layer1 hidden, and only "
                "the final residual is drained to host"
            )
        else:
            print(
                "dispatch_shape: hidden[1024] + packed QKV/norm weights + "
                "rope_angles[128] + K/V cache -> x_norm, Q/K/V raw, Q/K norm, "
                "RoPE Q, qk_pair, attention scores, attention weights, "
                "optional attention context, optional O projection/residual, KV cache"
            )
            print(
                "runtime_bos: hidden[1024], "
                f"packed_weights[{op.packed_weights_size}], "
                "rope_angles[128], "
                f"packed_outputs[{op.packed_outputs_size}], "
                f"packed_cache[{op.packed_cache_size}]"
            )
            print(f"decode_position: {op.position}")
            print(f"qkv_columns: {op.num_aie_columns}")
            print(
                "worker_graph: hidden_fifo -> rmsnorm_worker -> weight_worker -> "
                "xnorm broadcast -> Q/K/V matvec -> Q/K norm -> Q/K RoPE -> "
                "qk_pair debug drain -> score worker -> softmax worker -> "
                "optional V merge/context worker -> optional O projection/residual, "
                "with KV cache drains"
            )
    if args.compile_only:
        return

    op_func = op.get_callable()
    if args.stage == MULTI_LAYER_FULL_LAYER_STAGE:
        failed = run_multi_layer_full_layer(args, model, input_ids, op, op_func)
        if args.verify and failed:
            raise SystemExit(1)
        gc.collect()
        return
    if args.stage == SINGLE_LAYER_FINAL_ONLY_STAGE:
        failed = run_single_layer_final_only(args, model, input_ids, op, op_func)
        if args.verify and failed:
            raise SystemExit(1)
        gc.collect()
        return
    if args.stage == TWO_LAYER_FULL_LAYER_STAGE:
        failed = run_two_layer_full_layer(args, model, input_ids, op, op_func)
        if args.verify and failed:
            raise SystemExit(1)
        gc.collect()
        return

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
        if (
            args.stage
            == "input-rmsnorm-qkv-rope-cache-scores-softmax-context-o-proj-full-mlp"
        ):
            next_token, position, inputs, expected_buffers = build_reference_full_layer(
                model, input_ids, args.max_seq_len
            )
        else:
            next_token, position, inputs, expected_buffers = (
                build_reference_qkv_rope_cache(model, input_ids, args.max_seq_len)
            )
        if position != op.position:
            raise RuntimeError(
                f"compiled position {op.position} != reference {position}"
            )
        full_expected_buffers = expected_buffers
        hidden_buf = XRTTensor.from_torch(inputs["hidden"])
        packed_weight_parts = [
            inputs["input_norm_weight"].flatten(),
            inputs["W_q"].flatten(),
            inputs["W_k"].flatten(),
            inputs["W_v"].flatten(),
            inputs["W_q_norm"].flatten(),
            inputs["W_k_norm"].flatten(),
        ]
        if args.stage in {
            "input-rmsnorm-qkv-rope-cache-scores-softmax-context-o-proj",
            "input-rmsnorm-qkv-rope-cache-scores-softmax-context-o-proj-full-mlp",
        }:
            packed_weight_parts.append(inputs["W_o"].flatten())
        if (
            args.stage
            == "input-rmsnorm-qkv-rope-cache-scores-softmax-context-o-proj-full-mlp"
        ):
            packed_weight_parts.extend(
                [
                    inputs["post_norm_weight"].flatten(),
                    inputs["W_gate"].flatten(),
                    inputs["W_up"].flatten(),
                    inputs["W_down"].flatten(),
                ]
            )
        packed_weights = torch.cat(packed_weight_parts).contiguous()
        weights_buf = XRTTensor.from_torch(packed_weights)
        rope_angles_buf = XRTTensor.from_torch(inputs["rope_angles"])
        packed_outputs_buf = XRTTensor(
            (op.packed_outputs_size,),
            dtype=hidden_buf.dtype,
        )
        packed_cache_buf = XRTTensor.from_torch(inputs["initial_cache"].clone())
        op_args = [
            hidden_buf,
            weights_buf,
            rope_angles_buf,
            packed_outputs_buf,
            packed_cache_buf,
        ]
        output_buffers = {
            "packed_outputs": packed_outputs_buf,
            "packed_cache": packed_cache_buf,
        }

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
        elif (
            args.stage
            == "input-rmsnorm-qkv-rope-cache-scores-softmax-context-o-proj-full-mlp"
        ):
            packed_outputs_buf.device = "npu"
            packed_cache_buf.device = "npu"
            packed_outputs = packed_outputs_buf.to_torch()
            actual_buffers = {
                "attn_residual": packed_outputs[
                    op.attn_residual_output_base : op.mlp_x_norm_output_base
                ],
                "ffn_hidden": packed_outputs[
                    op.ffn_hidden_output_base : op.ffn_out_output_base
                ],
                "ffn_out": packed_outputs[
                    op.ffn_out_output_base : op.layer_residual_output_base
                ],
                "layer_residual": packed_outputs[op.layer_residual_output_base :],
            }
            ffn_out_local = F.linear(
                actual_buffers["ffn_hidden"].view(1, 1, -1), inputs["W_down"]
            ).flatten()
            layer_residual_local = (
                actual_buffers["attn_residual"] + actual_buffers["ffn_out"]
            )
            local_expected_buffers = {
                **expected_buffers,
                "ffn_hidden": expected_buffers["ffn_hidden"].contiguous(),
                "ffn_out": ffn_out_local.contiguous(),
                "layer_residual": layer_residual_local.contiguous(),
            }
        else:
            packed_outputs_buf.device = "npu"
            packed_cache_buf.device = "npu"
            packed_outputs = packed_outputs_buf.to_torch()
            packed_cache = packed_cache_buf.to_torch()
            has_scores_softmax = args.stage in {
                "input-rmsnorm-qkv-rope-cache-scores-softmax",
                "input-rmsnorm-qkv-rope-cache-scores-softmax-context",
                "input-rmsnorm-qkv-rope-cache-scores-softmax-context-o-proj",
                "input-rmsnorm-qkv-rope-cache-scores-softmax-context-o-proj-full-mlp",
            }
            has_context = args.stage in {
                "input-rmsnorm-qkv-rope-cache-scores-softmax-context",
                "input-rmsnorm-qkv-rope-cache-scores-softmax-context-o-proj",
                "input-rmsnorm-qkv-rope-cache-scores-softmax-context-o-proj-full-mlp",
            }
            has_o_proj = args.stage in {
                "input-rmsnorm-qkv-rope-cache-scores-softmax-context-o-proj",
                "input-rmsnorm-qkv-rope-cache-scores-softmax-context-o-proj-full-mlp",
            }
            has_full_mlp = (
                args.stage
                == "input-rmsnorm-qkv-rope-cache-scores-softmax-context-o-proj-full-mlp"
            )
            q_rope_end = (
                op.qk_pair_output_base if has_scores_softmax else op.packed_outputs_size
            )
            keys_cache = packed_cache[: op.cache_half_size].view(
                op.kv_heads, args.max_seq_len, op.head_dim
            )
            values_cache = packed_cache[op.cache_half_size :].view(
                op.kv_heads, args.max_seq_len, op.head_dim
            )
            keys_cache_current = keys_cache[:, op.position, :].flatten()
            values_cache_current = values_cache[:, op.position, :].flatten()
            actual_buffers = {
                "x_norm": packed_outputs[: op.q_raw_output_base],
                "queries_raw": packed_outputs[
                    op.q_raw_output_base : op.k_raw_output_base
                ],
                "keys_raw": packed_outputs[
                    op.k_raw_output_base : op.q_norm_output_base
                ],
                "queries_norm": packed_outputs[
                    op.q_norm_output_base : op.k_norm_output_base
                ],
                "keys_norm": packed_outputs[
                    op.k_norm_output_base : op.q_rope_output_base
                ],
                "queries": packed_outputs[op.q_rope_output_base : q_rope_end],
                "keys": keys_cache_current,
                "values": values_cache_current,
                "keys_cache_current": keys_cache_current,
                "values_cache_current": values_cache_current,
                "keys_cache_prefix": keys_cache[:, : op.position, :].flatten(),
                "values_cache_prefix": values_cache[:, : op.position, :].flatten(),
            }
            if has_scores_softmax:
                actual_buffers["qk_pair"] = packed_outputs[
                    op.qk_pair_output_base : op.k_cache_stream_output_base
                ]
                if op.k_cache_debug_size:
                    k_cache_stream = packed_outputs[
                        op.k_cache_stream_output_base : op.attn_scores_output_base
                    ].view(op.kv_heads, op.max_seq_len, op.head_dim)
                    actual_buffers["k_cache_stream_prefix"] = k_cache_stream[
                        :, : op.position, :
                    ].flatten()
                actual_buffers["attn_scores"] = packed_outputs[
                    op.attn_scores_output_base : op.attn_weights_output_base
                ]
                attn_weight_end = (
                    op.v_context_stream_output_base
                    if has_context
                    else op.packed_outputs_size
                )
                actual_buffers["attn_weights"] = packed_outputs[
                    op.attn_weights_output_base : attn_weight_end
                ]
                if has_context:
                    v_context_stream = packed_outputs[
                        op.v_context_stream_output_base : op.attn_context_output_base
                    ].view(op.kv_heads, op.max_seq_len, op.head_dim)
                    actual_buffers["v_context_stream_prefix"] = v_context_stream[
                        :, : op.position, :
                    ].flatten()
                    actual_buffers["v_context_stream_current"] = v_context_stream[
                        :, op.position, :
                    ].flatten()
                    actual_buffers["attn_context"] = packed_outputs[
                        op.attn_context_output_base : (
                            op.attn_context_flat_output_base
                            if has_o_proj
                            else op.packed_outputs_size
                        )
                    ]
                    if has_o_proj:
                        actual_buffers["attn_context_flat"] = packed_outputs[
                            op.attn_context_flat_output_base : op.attn_o_proj_output_base
                        ]
                        actual_buffers["attn_o_proj"] = packed_outputs[
                            op.attn_o_proj_output_base : op.attn_residual_output_base
                        ]
                        attn_residual_end = (
                            op.mlp_x_norm_output_base
                            if has_full_mlp
                            else op.packed_outputs_size
                        )
                        actual_buffers["attn_residual"] = packed_outputs[
                            op.attn_residual_output_base : attn_residual_end
                        ]
                        if has_full_mlp:
                            actual_buffers["mlp_x_norm"] = packed_outputs[
                                op.mlp_x_norm_output_base : op.ffn_gate_output_base
                            ]
                            actual_buffers["ffn_hidden"] = packed_outputs[
                                op.ffn_hidden_output_base : op.ffn_out_output_base
                            ]
                            actual_buffers["ffn_out"] = packed_outputs[
                                op.ffn_out_output_base : op.layer_residual_output_base
                            ]
                            actual_buffers["layer_residual"] = packed_outputs[
                                op.layer_residual_output_base :
                            ]
            q_raw_local = F.linear(
                actual_buffers["x_norm"].view(1, 1, -1), inputs["W_q"]
            ).flatten()
            k_raw_local = F.linear(
                actual_buffers["x_norm"].view(1, 1, -1), inputs["W_k"]
            ).flatten()
            values_local = F.linear(
                actual_buffers["x_norm"].view(1, 1, -1), inputs["W_v"]
            ).flatten()
            q_norm_local = rms_norm(
                actual_buffers["queries_raw"].view(1, op.q_heads, 1, op.head_dim),
                inputs["W_q_norm"],
                model.config.rms_norm_eps,
            ).flatten()
            k_norm_local = rms_norm(
                actual_buffers["keys_raw"].view(1, op.kv_heads, 1, op.head_dim),
                inputs["W_k_norm"],
                model.config.rms_norm_eps,
            ).flatten()
            q_rope_local, k_rope_local = apply_rope(
                actual_buffers["queries_norm"].view(1, op.q_heads, 1, op.head_dim),
                actual_buffers["keys_norm"].view(1, op.kv_heads, 1, op.head_dim),
                torch.tensor([op.position]),
                op.head_dim,
                model.config.rope_theta,
            )
            score_expected_buffers = {}
            context_alt_expected_buffers = {}
            if has_scores_softmax:
                qk_pair = build_qk_pair_reference(
                    actual_buffers["queries"],
                    actual_buffers["keys"],
                    op.q_heads,
                    op.kv_heads,
                    op.head_dim,
                )
                q_for_score = actual_buffers["queries"].view(
                    1, op.q_heads, 1, op.head_dim
                )
                k_ctx = repeat_kv(
                    keys_cache[:, : op.position + 1, :].unsqueeze(0),
                    op.q_heads // op.kv_heads,
                )
                scores = torch.matmul(
                    q_for_score.to(torch.float32),
                    k_ctx.to(torch.float32).transpose(-2, -1),
                ) / math.sqrt(op.head_dim)
                padded_scores = torch.zeros(
                    (op.q_heads, op.max_seq_len),
                    dtype=packed_outputs.dtype,
                )
                padded_weights = torch.zeros_like(padded_scores)
                padded_scores[:, : op.position + 1] = scores.view(
                    op.q_heads, op.position + 1
                ).to(dtype=packed_outputs.dtype)
                weights = torch.softmax(
                    padded_scores[:, : op.position + 1].to(torch.float32),
                    dim=-1,
                ).to(dtype=packed_outputs.dtype)
                padded_weights[:, : op.position + 1] = weights.view(
                    op.q_heads, op.position + 1
                )
                context_expected_buffers = {}
                if has_context:
                    v_context = inputs["initial_values_cache"].clone()
                    v_context[:, op.position, :] = actual_buffers["values"].view(
                        op.kv_heads, op.head_dim
                    )
                    context = torch.zeros(
                        (op.q_heads, op.head_dim), dtype=packed_outputs.dtype
                    )
                    context_float = torch.zeros(
                        (op.q_heads, op.head_dim), dtype=torch.float32
                    )
                    weights_by_head = actual_buffers["attn_weights"].view(
                        op.q_heads, op.max_seq_len
                    )
                    q_per_kv = op.q_heads // op.kv_heads
                    for q_head in range(op.q_heads):
                        kv_head = q_head // q_per_kv
                        for block_start in range(0, op.max_seq_len, 64):
                            block_end = min(block_start + 64, op.position + 1)
                            if block_start >= block_end:
                                continue
                            block_accum = context[q_head].to(torch.float32)
                            for seq_pos in range(block_start, block_end):
                                term = weights_by_head[q_head, seq_pos].to(
                                    torch.float32
                                ) * v_context[kv_head, seq_pos, :].to(torch.float32)
                                block_accum += term
                                context_float[q_head] += term
                            context[q_head] = block_accum.to(dtype=packed_outputs.dtype)
                    context_alt_expected_buffers = {
                        "attn_context_float_accum": context_float.to(
                            dtype=packed_outputs.dtype
                        )
                        .flatten()
                        .contiguous()
                    }
                    context_expected_buffers = {
                        "v_context_stream_prefix": inputs["initial_values_cache"][
                            :, : op.position, :
                        ]
                        .flatten()
                        .contiguous(),
                        "v_context_stream_current": actual_buffers[
                            "values"
                        ].contiguous(),
                        "attn_context": context.flatten().contiguous(),
                    }
                    if has_o_proj:
                        o_proj_local = F.linear(
                            actual_buffers["attn_context_flat"].view(1, 1, -1),
                            inputs["W_o"],
                        ).flatten()
                        residual_local = (
                            inputs["hidden"] + actual_buffers["attn_o_proj"]
                        )
                        context_expected_buffers.update(
                            {
                                "attn_context_flat": actual_buffers[
                                    "attn_context"
                                ].contiguous(),
                                "attn_o_proj": o_proj_local.contiguous(),
                                "attn_residual": residual_local.contiguous(),
                            }
                        )
                        if has_full_mlp:
                            mlp_x_norm_local = rms_norm(
                                actual_buffers["attn_residual"].view(1, 1, -1),
                                inputs["post_norm_weight"],
                                model.config.rms_norm_eps,
                            ).flatten()
                            ffn_out_local = F.linear(
                                actual_buffers["ffn_hidden"].view(1, 1, -1),
                                inputs["W_down"],
                            ).flatten()
                            layer_residual_local = (
                                actual_buffers["attn_residual"]
                                + actual_buffers["ffn_out"]
                            )
                            context_expected_buffers.update(
                                {
                                    "mlp_x_norm": mlp_x_norm_local.contiguous(),
                                    "ffn_hidden": expected_buffers[
                                        "ffn_hidden"
                                    ].contiguous(),
                                    "ffn_out": ffn_out_local.contiguous(),
                                    "layer_residual": layer_residual_local.contiguous(),
                                }
                            )
                score_expected_buffers = {
                    "qk_pair": qk_pair,
                    "attn_scores": padded_scores.flatten().contiguous(),
                    "attn_weights": padded_weights.flatten().contiguous(),
                    **context_expected_buffers,
                }
                if op.k_cache_debug_size:
                    score_expected_buffers["k_cache_stream_prefix"] = (
                        inputs["initial_keys_cache"][:, : op.position, :]
                        .flatten()
                        .contiguous()
                    )
            local_expected_buffers = {
                **full_expected_buffers,
                "queries_raw": q_raw_local.contiguous(),
                "keys_raw": k_raw_local.contiguous(),
                "values": values_local.contiguous(),
                "queries_norm": q_norm_local.contiguous(),
                "keys_norm": k_norm_local.contiguous(),
                "queries": q_rope_local.flatten().contiguous(),
                "keys": k_rope_local.flatten().contiguous(),
                "keys_cache_current": actual_buffers["keys"].contiguous(),
                "values_cache_current": actual_buffers["values"].contiguous(),
                "keys_cache_prefix": inputs["initial_keys_cache"][:, : op.position, :]
                .flatten()
                .contiguous(),
                "values_cache_prefix": inputs["initial_values_cache"][
                    :, : op.position, :
                ]
                .flatten()
                .contiguous(),
                **score_expected_buffers,
            }
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
                    "input-rmsnorm-qkv-rope-cache",
                    "input-rmsnorm-qkv-rope-cache-scores-softmax",
                    "input-rmsnorm-qkv-rope-cache-scores-softmax-context",
                    "input-rmsnorm-qkv-rope-cache-scores-softmax-context-o-proj",
                    "input-rmsnorm-qkv-rope-cache-scores-softmax-context-o-proj-full-mlp",
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
            if name == "attn_context" and errors:
                for alt_name, alt_expected in context_alt_expected_buffers.items():
                    alt_diff = (
                        output.to(torch.float32) - alt_expected.to(torch.float32)
                    ).abs()
                    alt_errors = verify_buffer(
                        output,
                        alt_name,
                        alt_expected,
                        rel_tol=rel_tol,
                        abs_tol=abs_tol,
                    )
                    print(f"{alt_name}_max_abs: {float(alt_diff.max()):.6f}")
                    print(f"{alt_name}_mean_abs: {float(alt_diff.mean()):.6f}")
                    print(f"{alt_name}_errors: {len(alt_errors)}")
            print_structured_attention_error(name, errors, output, expected, op)
            failed = failed or bool(errors)

    if args.verify and failed:
        raise SystemExit(1)
    gc.collect()


if __name__ == "__main__":
    main()
