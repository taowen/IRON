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

repo_root = Path(__file__).resolve().parents[3]
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
from iron.applications.qwen3_0_6b.qwen3_persistent_ops import (  # noqa: E402
    Qwen3PersistentInputRMSNorm,
    Qwen3PersistentInputRMSNormQKV,
    Qwen3PersistentInputRMSNormQKVRopeCache,
    Qwen3PersistentInputRMSNormQKVRopeCacheScoresSoftmax,
    Qwen3PersistentInputRMSNormQKVRopeCacheScoresSoftmaxContext,
    Qwen3PersistentInputRMSNormQKVRopeCacheScoresSoftmaxContextOProj,
    Qwen3PersistentPostAttnRMSNormMLPGateUp,
    verification_tolerance,
)
from iron.applications.qwen3_0_6b.qwen3_persistent_refs import (  # noqa: E402
    build_qk_pair_reference,
    build_reference_input,
    build_reference_mlp_gate_up,
    build_reference_qkv,
    build_reference_qkv_rope_cache,
    print_structured_attention_error,
)
from iron.applications.qwen3_0_6b.qwen3_preflight import (  # noqa: E402
    run_persistent_artifact_preflight,
)
from iron.common.context import AIEContext  # noqa: E402
from iron.common.test_utils import verify_buffer  # noqa: E402
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor  # noqa: E402


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
        "--stage",
        choices=[
            "input-rmsnorm",
            "input-rmsnorm-qkv",
            "input-rmsnorm-qkv-rope-cache",
            "input-rmsnorm-qkv-rope-cache-scores-softmax",
            "input-rmsnorm-qkv-rope-cache-scores-softmax-context",
            "input-rmsnorm-qkv-rope-cache-scores-softmax-context-o-proj",
            "post-attn-rmsnorm-mlp-gate-up",
        ],
        default="input-rmsnorm",
        help="Persistent bring-up stage to compile/run",
    )
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verify-repeat", type=int, default=1)
    parser.add_argument("--dump-proof", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.clean_build:
        shutil.rmtree(args.build_dir, ignore_errors=True)

    model_dir = resolve_model_dir(args.model, args.revision)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    input_ids = encode_prompt(
        tokenizer,
        args.prompt,
        raw_prompt=args.raw_prompt,
        enable_thinking=args.enable_thinking,
    )
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
    if not args.compile_only:
        assert_standard_runtime_available()

    context = AIEContext(build_dir=args.build_dir)
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
    else:
        next_token, position, inputs, expected_buffers = build_reference_qkv_rope_cache(
            model, input_ids, args.max_seq_len
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
        if args.stage == "input-rmsnorm-qkv-rope-cache-scores-softmax-context-o-proj":
            packed_weight_parts.append(inputs["W_o"].flatten())
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
        else:
            packed_outputs_buf.device = "npu"
            packed_cache_buf.device = "npu"
            packed_outputs = packed_outputs_buf.to_torch()
            packed_cache = packed_cache_buf.to_torch()
            has_scores_softmax = args.stage in {
                "input-rmsnorm-qkv-rope-cache-scores-softmax",
                "input-rmsnorm-qkv-rope-cache-scores-softmax-context",
                "input-rmsnorm-qkv-rope-cache-scores-softmax-context-o-proj",
            }
            has_context = args.stage in {
                "input-rmsnorm-qkv-rope-cache-scores-softmax-context",
                "input-rmsnorm-qkv-rope-cache-scores-softmax-context-o-proj",
            }
            has_o_proj = (
                args.stage
                == "input-rmsnorm-qkv-rope-cache-scores-softmax-context-o-proj"
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
                        actual_buffers["attn_residual"] = packed_outputs[
                            op.attn_residual_output_base :
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
                    "post-attn-rmsnorm-mlp-gate-up",
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
