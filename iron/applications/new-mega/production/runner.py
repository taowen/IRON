#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import torch
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from ml_dtypes import bfloat16
from transformers import AutoTokenizer

from iron.applications.qwen3_0_6b.persistent.checks import (
    print_tensor_check,
    tensor_error_stats,
)
from iron.applications.qwen3_0_6b.persistent.refs import one_layer_reference_tensors
from iron.applications.qwen3_0_6b.qwen3_cpu import (
    Qwen3ForCausalLM,
    encode_prompt,
    resolve_model_dir,
)
from iron.applications.qwen3_0_6b.qwen3_decode_reference import Qwen3CachedReference
from iron.applications.qwen3_0_6b.qwen3_preflight import (
    Qwen3PreflightError,
    PersistentPreflightResult,
    run_persistent_artifact_preflight,
)
from iron.common.context import AIEContext

from ops import NewMegaFixedCacheAttentionContext


@dataclass(frozen=True)
class FixedAttentionCase:
    next_token: int
    position: int
    prompt_tokens: int
    q: torch.Tensor
    packed_stream: torch.Tensor
    expected_context: torch.Tensor


@dataclass(frozen=True)
class FixedAttentionRunResult:
    op: NewMegaFixedCacheAttentionContext
    preflight: PersistentPreflightResult
    prompt_tokens: int
    position: int
    next_token: int
    npu_time_us: float | None
    max_abs: float | None
    mean_abs: float | None
    errors: int | None


def make_mask(max_seq_len: int, position: int) -> torch.Tensor:
    mask = torch.zeros(max_seq_len, dtype=torch.float32)
    mask[: position + 1] = 1.0
    return mask.to(torch.bfloat16).contiguous()


def pack_q_head_cache_stream(
    *,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    mask: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    chunk_size: int,
) -> torch.Tensor:
    max_seq_len = k_cache.shape[1]
    repeats = q_heads // kv_heads
    chunks: list[torch.Tensor] = []
    for q_head in range(q_heads):
        kv_head = q_head // repeats
        for start in range(0, max_seq_len, chunk_size):
            end = start + chunk_size
            chunks.append(
                torch.cat(
                    [
                        k_cache[kv_head, start:end].flatten(),
                        v_cache[kv_head, start:end].flatten(),
                        mask[start:end].flatten(),
                    ]
                )
            )
    return torch.cat(chunks).to(torch.bfloat16).contiguous()


def build_real_qwen3_attention_case(
    *,
    model: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    max_seq_len: int,
    chunk_size: int,
) -> FixedAttentionCase:
    ref = Qwen3CachedReference(model, max_seq_len, num_layers=1)
    prefill_logits, state = ref.prefill(input_ids)
    next_token = int(torch.argmax(prefill_logits[:, -1, :], dim=-1).item())
    position = state.position
    refs = one_layer_reference_tensors(model, next_token, state, max_seq_len)

    q = refs["queries"].flatten().contiguous()
    k_cache = state.keys[0].clone()
    v_cache = state.values[0].clone()
    k_cache[:, position, :] = refs["keys"].to(k_cache.dtype)
    v_cache[:, position, :] = refs["values"].to(v_cache.dtype)
    mask = make_mask(max_seq_len, position)
    packed_stream = pack_q_head_cache_stream(
        k_cache=k_cache,
        v_cache=v_cache,
        mask=mask,
        q_heads=model.config.num_attention_heads,
        kv_heads=model.config.num_key_value_heads,
        chunk_size=chunk_size,
    )
    return FixedAttentionCase(
        next_token=next_token,
        position=position,
        prompt_tokens=input_ids.shape[1],
        q=q,
        packed_stream=packed_stream,
        expected_context=refs["attn_context"].flatten().contiguous(),
    )


def load_model_and_prompt(
    *,
    model_name: str,
    revision: str | None,
    prompt: str,
    raw_prompt: bool,
    enable_thinking: bool,
) -> tuple[Path, Qwen3ForCausalLM, torch.Tensor]:
    model_dir = resolve_model_dir(model_name, revision)
    model = Qwen3ForCausalLM(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    input_ids = encode_prompt(
        tokenizer,
        prompt,
        raw_prompt=raw_prompt,
        enable_thinking=enable_thinking,
    )
    return model_dir, model, input_ids


def compile_fixed_attention_stage(
    *,
    model: Qwen3ForCausalLM,
    max_seq_len: int,
    chunk_size: int,
    build_dir: Path,
) -> tuple[NewMegaFixedCacheAttentionContext, PersistentPreflightResult]:
    if model.config.hidden_size != 1024:
        raise ValueError("new-mega production expects Qwen3-0.6B hidden_size=1024")
    if model.config.head_dim != 128:
        raise ValueError("new-mega production expects Qwen3-0.6B head_dim=128")

    context = AIEContext(build_dir=build_dir)
    op = NewMegaFixedCacheAttentionContext(
        max_seq_len=max_seq_len,
        q_heads=model.config.num_attention_heads,
        kv_heads=model.config.num_key_value_heads,
        head_dim=model.config.head_dim,
        chunk_size=chunk_size,
        context=context,
    )
    op.compile()
    try:
        preflight = run_persistent_artifact_preflight(
            mlir_path=Path(op.xclbin_artifact.mlir_input.filename),
            arg_specs=len(op.get_arg_spec()),
        )
    except Qwen3PreflightError:
        raise
    return op, preflight


def run_fixed_attention_stage(
    *,
    op: NewMegaFixedCacheAttentionContext,
    preflight: PersistentPreflightResult,
    case: FixedAttentionCase,
    rel_tol: float,
    abs_tol: float,
    compile_only: bool = False,
) -> FixedAttentionRunResult:
    if compile_only:
        return FixedAttentionRunResult(
            op=op,
            preflight=preflight,
            prompt_tokens=case.prompt_tokens,
            position=case.position,
            next_token=case.next_token,
            npu_time_us=None,
            max_abs=None,
            mean_abs=None,
            errors=None,
        )

    q_buf = XRTTensor.from_torch(case.q)
    packed_buf = XRTTensor.from_torch(case.packed_stream)
    out_buf = XRTTensor((op.q_size,), dtype=bfloat16)
    npu_result = op.get_callable()(q_buf, packed_buf, out_buf)
    out_buf.device = "npu"
    actual_context = out_buf.to_torch().detach().clone().flatten()
    errors, max_abs, mean_abs, *_ = tensor_error_stats(
        actual_context,
        case.expected_context,
        rel_tol,
        abs_tol,
    )
    return FixedAttentionRunResult(
        op=op,
        preflight=preflight,
        prompt_tokens=case.prompt_tokens,
        position=case.position,
        next_token=case.next_token,
        npu_time_us=npu_result.npu_time / 1000.0,
        max_abs=max_abs,
        mean_abs=mean_abs,
        errors=errors,
    )


def print_preflight(result: PersistentPreflightResult) -> None:
    print(f"preflight_runtime_memrefs: {result.runtime_memrefs}")
    print(f"preflight_arg_specs: {result.arg_specs}")
    print(f"preflight_metadata_host_bos: {result.metadata_host_bos}")
    print(f"preflight_compute_cores: {result.compute_cores}")
    print(f"preflight_max_fifo_buffered_bytes: {result.max_fifo_buffered_bytes}")
    print(f"preflight_total_dma_tasks: {result.total_dma_tasks}")
    print(f"preflight_max_dma_tasks_per_fifo: {result.max_dma_tasks_per_fifo}")
    print(f"preflight_max_compute_tile_inputs: {result.max_compute_tile_inputs}")
    print(f"preflight_max_compute_tile_outputs: {result.max_compute_tile_outputs}")
    print(f"preflight_non_advancing_acquires: {result.non_advancing_acquires}")


def print_fixed_attention_run(
    *,
    result: FixedAttentionRunResult,
    expected_context: torch.Tensor | None = None,
    actual_context: torch.Tensor | None = None,
    rel_tol: float | None = None,
    abs_tol: float | None = None,
) -> None:
    op = result.op
    print(f"xclbin: {op.xclbin_artifact.filename}")
    print(f"runtime_bin: {op.insts_artifact.filename}")
    print(f"mlir: {op.xclbin_artifact.mlir_input.filename}")
    print(f"max_seq_len: {op.max_seq_len}")
    print(f"chunk_size: {op.chunk_size}")
    print(f"num_chunks: {op.num_chunks}")
    print(f"q_heads: {op.q_heads}")
    print(f"kv_heads: {op.kv_heads}")
    print(f"head_dim: {op.head_dim}")
    print(f"packed_chunk_elements: {op.packed_chunk_elements}")
    print(f"packed_chunk_bytes: {op.packed_chunk_elements * 2}")
    print(f"packed_stream_elements: {op.packed_stream_elements}")
    print(f"host_kv_writeback_bytes: {2 * op.kv_heads * op.head_dim * 2}")
    print_preflight(result.preflight)
    print(f"prompt_tokens: {result.prompt_tokens}")
    print(f"decode_position: {result.position}")
    print(f"prompt_next_token: {result.next_token}")
    if result.npu_time_us is None:
        print("decision: compile-only")
        return
    print(f"npu_time_us: {result.npu_time_us:.3f}")
    if actual_context is not None and expected_context is not None:
        if rel_tol is None or abs_tol is None:
            raise ValueError("rel_tol and abs_tol are required for tensor printing")
        print_tensor_check(
            "new_mega_attention_context",
            actual_context,
            expected_context,
            rel_tol=rel_tol,
            abs_tol=abs_tol,
        )
    else:
        print(f"new_mega_attention_context_max_abs: {result.max_abs:.6f}")
        print(f"new_mega_attention_context_mean_abs: {result.mean_abs:.6f}")
        print(f"new_mega_attention_context_errors: {result.errors}")
