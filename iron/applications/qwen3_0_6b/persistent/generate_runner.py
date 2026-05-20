#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import time
from pathlib import Path

import torch
import torch.nn.functional as F

from iron.applications.qwen3_0_6b.qwen3_cpu import Qwen3ForCausalLM, rms_norm
from iron.applications.qwen3_0_6b.qwen3_decode_reference import (
    Qwen3CachedReference,
    clone_decode_state,
)
from iron.applications.qwen3_0_6b.persistent.generate import (
    prepare_fast_generate_buffers,
    run_n_layer_decode_hidden_fast,
)
from iron.applications.qwen3_0_6b.persistent.ops_nlayer import (
    Qwen3PersistentNLayerFinalOnly,
)
from iron.applications.qwen3_0_6b.qwen3_preflight import (
    run_persistent_artifact_preflight,
)
from iron.common.context import AIEContext


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
