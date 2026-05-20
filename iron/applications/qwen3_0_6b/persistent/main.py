#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import gc
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
    encode_prompt,
    resolve_model_dir,
    rms_norm,
)
from iron.applications.qwen3_0_6b.qwen3_decode_reference import (  # noqa: E402
    Qwen3CachedReference,
    clone_decode_state,
)
from iron.applications.qwen3_0_6b.persistent.checks import (  # noqa: E402
    print_tensor_check,
)
from iron.applications.qwen3_0_6b.persistent.diagnostics import (  # noqa: E402
    run_qkv_diagnostic_bundle,
)
from iron.applications.qwen3_0_6b.persistent.layout import (  # noqa: E402
    build_full_layer_inputs_for_layer,
    default_packed_weights_dir,
    pack_full_layer_weights,
    validate_packed_weight_artifact,
    write_packed_weight_artifact,
)
from iron.applications.qwen3_0_6b.persistent.generate import (  # noqa: E402
    prepare_fast_generate_buffers,
    run_n_layer_decode_hidden_fast,
)
from iron.applications.qwen3_0_6b.persistent.ops import (  # noqa: E402
    Qwen3PersistentInputRMSNorm,
    Qwen3PersistentInputRMSNormQKV,
    Qwen3PersistentNLayerFinalOnly,
    Qwen3PersistentPostAttnMLPDownResidual,
    Qwen3PersistentPostAttnRMSNormFullMLP,
    Qwen3PersistentPostAttnRMSNormMLPGateUp,
    verification_tolerance,
)
from iron.applications.qwen3_0_6b.persistent.refs import (  # noqa: E402
    build_reference_multi_layer_full_layer,
    build_reference_input,
    build_reference_full_mlp,
    build_reference_mlp_down_residual,
    build_reference_mlp_gate_up,
    build_reference_qkv,
)
from iron.applications.qwen3_0_6b.qwen3_preflight import (  # noqa: E402
    run_persistent_artifact_preflight,
)
from iron.common.context import AIEContext  # noqa: E402
from iron.common.test_utils import verify_buffer  # noqa: E402
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor  # noqa: E402

N_LAYER_FINAL_ONLY_STAGE = "n-layer-final-only"
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
            N_LAYER_FINAL_ONLY_STAGE,
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
            "Use the n-layer final-only generate path with cached packed "
            "weight/cache XRT buffers."
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
        default=1,
        help="Number of full layers per persistent fast-generate invocation",
    )
    parser.add_argument(
        "--qkv-diagnostic-bundle",
        type=Path,
        default=None,
        help="Run an isolated QKV diagnostic bundle",
    )
    parser.add_argument("--dump-proof", action="store_true")
    return parser.parse_args()


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
    hidden_buf = XRTTensor.from_torch(initial_hidden)
    weights_buf = XRTTensor.from_torch(
        torch.cat(
            [pack_full_layer_weights(inputs) for inputs in inputs_by_layer]
        ).contiguous()
    )
    rope_angles_buf = XRTTensor.from_torch(inputs_by_layer[0]["rope_angles"])
    initial_cache = torch.cat(
        [inputs["initial_cache"].clone() for inputs in inputs_by_layer]
    ).contiguous()

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
                0.5,
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

    return failed


def make_single_layer_final_only_op_for_position(
    args,
    model: Qwen3ForCausalLM,
    context: AIEContext,
    position: int,
):
    return Qwen3PersistentNLayerFinalOnly(
        hidden_size=model.config.hidden_size,
        q_size=model.config.num_attention_heads * model.config.head_dim,
        kv_size=model.config.num_key_value_heads * model.config.head_dim,
        head_dim=model.config.head_dim,
        max_seq_len=args.max_seq_len,
        position=position,
        intermediate_size=model.config.intermediate_size,
        layer_iterations=1,
        epsilon=model.config.rms_norm_eps,
        context=context,
    )


def make_n_layer_final_only_op_for_position(
    args,
    model: Qwen3ForCausalLM,
    context: AIEContext,
    position: int,
    layer_iterations: int,
):
    return Qwen3PersistentNLayerFinalOnly(
        hidden_size=model.config.hidden_size,
        q_size=model.config.num_attention_heads * model.config.head_dim,
        kv_size=model.config.num_key_value_heads * model.config.head_dim,
        head_dim=model.config.head_dim,
        max_seq_len=args.max_seq_len,
        position=position,
        intermediate_size=model.config.intermediate_size,
        layer_iterations=layer_iterations,
        epsilon=model.config.rms_norm_eps,
        context=context,
    )


def compile_n_layer_final_only_op_for_position(
    args,
    model: Qwen3ForCausalLM,
    context: AIEContext,
    position: int,
    layer_iterations: int,
):
    op = make_n_layer_final_only_op_for_position(
        args,
        model,
        context,
        position,
        layer_iterations,
    )
    start = time.perf_counter()
    op.compile()
    compile_s = time.perf_counter() - start
    preflight = run_persistent_artifact_preflight(
        mlir_path=Path(op.xclbin_artifact.mlir_input.filename),
        arg_specs=len(op.get_arg_spec()),
    )
    print(
        f"generate_position_{position}_n_layer_{layer_iterations}_compile_s: "
        f"{compile_s:.3f} "
        f"operator_name={op.name}"
    )
    print(
        f"generate_position_{position}_n_layer_{layer_iterations}_preflight: ok "
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


def run_generate(
    args,
    model: Qwen3ForCausalLM,
    tokenizer,
    input_ids: torch.Tensor,
    context: AIEContext,
) -> bool:
    if args.layer_chunk_size < 1:
        raise ValueError(
            f"layer_chunk_size must be positive, got {args.layer_chunk_size}"
        )
    if args.layer_chunk_size != 1 and not args.fast_generate:
        raise ValueError("--layer-chunk-size > 1 requires --fast-generate")
    if not args.fast_generate:
        raise ValueError(
            "generate now uses the n-layer final-only path; pass --fast-generate"
        )
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
    print(
        f"implementation: persistent n-layer final-only decode "
        f"(chunk={args.layer_chunk_size}) with cached XRT "
        "weights/cache + CPU final norm/lm head"
    )
    print(f"prompt_len: {input_ids.shape[1]}")
    print(f"max_new_tokens: {args.max_new_tokens}")
    print(f"layer_chunk_size: {args.layer_chunk_size}")
    print(f"prompt_next_token: {first_token} text={first_text!r}")

    chunk_op_cache = {}

    def get_position_chunk_op(position: int, chunk_len: int):
        key = (position, chunk_len)
        if key not in chunk_op_cache:
            chunk_op_cache[key] = compile_n_layer_final_only_op_for_position(
                args,
                model,
                context,
                position,
                chunk_len,
            )
        return chunk_op_cache[key]

    def chunk_lengths_for_model():
        layer_idx = 0
        while layer_idx < model.config.num_hidden_layers:
            chunk_len = min(
                args.layer_chunk_size,
                model.config.num_hidden_layers - layer_idx,
            )
            yield chunk_len
            layer_idx += chunk_len

    def get_position_chunk_ops(position: int):
        return {
            chunk_len: get_position_chunk_op(position, chunk_len)
            for chunk_len in sorted(set(chunk_lengths_for_model()))
        }

    if args.compile_only:
        get_position_chunk_ops(prefill_state.position)
        return False

    generated_tokens = [first_token]
    npu_state = clone_decode_state(prefill_state)
    ref_state = clone_decode_state(prefill_state)
    fast_buffers = None
    first_chunk_ops = get_position_chunk_ops(prefill_state.position)
    first_op, _first_op_func = first_chunk_ops[
        min(args.layer_chunk_size, model.config.num_hidden_layers)
    ]
    setup_start = time.perf_counter()
    fast_buffers = prepare_fast_generate_buffers(
        model,
        prefill_state,
        first_op,
        packed_weights_dir=args.packed_weights_dir,
        require_packed_weights=args.require_packed_weights,
        layer_chunk_size=args.layer_chunk_size,
    )
    setup_s = time.perf_counter() - setup_start
    setup_timing = fast_buffers.timing
    print(f"fast_generate_setup_s: {setup_s:.6f}")
    print(f"fast_generate_weight_source: {setup_timing.weight_source}")
    print(f"fast_generate_weight_pack_s: {setup_timing.weight_pack_s:.6f}")
    print(f"fast_generate_weight_disk_load_s: {setup_timing.weight_disk_load_s:.6f}")
    print(f"fast_generate_weight_xrt_s: {setup_timing.weight_xrt_s:.6f}")
    print(f"fast_generate_cache_xrt_s: {setup_timing.cache_xrt_s:.6f}")
    failed = False

    for token_idx in range(1, args.max_new_tokens):
        current_token = generated_tokens[-1]
        position = npu_state.position
        chunk_ops = get_position_chunk_ops(position)
        start = time.perf_counter()
        fast_timing = None
        npu_hidden, npu_time, fast_timing = run_n_layer_decode_hidden_fast(
            model,
            current_token,
            position,
            args.layer_chunk_size,
            chunk_ops,
            fast_buffers,
        )
        npu_state.position += 1
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
        op_for_layout = make_single_layer_final_only_op_for_position(
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
            expected_per_layer_numel=make_single_layer_final_only_op_for_position(
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
    elif args.stage == N_LAYER_FINAL_ONLY_STAGE:
        op = Qwen3PersistentNLayerFinalOnly(
            hidden_size=model.config.hidden_size,
            q_size=model.config.num_attention_heads * model.config.head_dim,
            kv_size=model.config.num_key_value_heads * model.config.head_dim,
            head_dim=model.config.head_dim,
            max_seq_len=args.max_seq_len,
            position=input_ids.shape[1],
            intermediate_size=model.config.intermediate_size,
            layer_iterations=args.layer_chunk_size,
            epsilon=model.config.rms_norm_eps,
            context=context,
        )
    else:
        raise ValueError(f"unsupported stage after attention cleanup: {args.stage}")

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
    if args.compile_only:
        return

    op_func = op.get_callable()
    if args.stage == N_LAYER_FINAL_ONLY_STAGE:
        failed = run_n_layer_final_only(args, model, input_ids, op, op_func)
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

    if args.verify and failed:
        raise SystemExit(1)
    gc.collect()


if __name__ == "__main__":
    main()
