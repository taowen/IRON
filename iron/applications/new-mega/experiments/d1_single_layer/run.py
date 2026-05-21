#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import aie.utils as aie_utils
import torch
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from ml_dtypes import bfloat16
from transformers import AutoTokenizer

from iron.applications.qwen3_0_6b.persistent.checks import print_tensor_check
from iron.applications.qwen3_0_6b.persistent.refs import one_layer_reference_tensors
from iron.applications.qwen3_0_6b.qwen3_cpu import (
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    Qwen3ForCausalLM,
    encode_prompt,
    resolve_model_dir,
)
from iron.applications.qwen3_0_6b.qwen3_decode_reference import Qwen3CachedReference
from iron.applications.qwen3_0_6b.qwen3_preflight import (
    Qwen3PreflightError,
    run_persistent_artifact_preflight,
)
from iron.common import (
    AIERuntimeArgSpec,
    DesignGenerator,
    KernelObjectArtifact,
    MLIROperator,
    PythonGeneratedMLIRArtifact,
    SourceArtifact,
)
from iron.common.context import AIEContext


@dataclass
class D1FixedCacheAttentionContext(MLIROperator):
    max_seq_len: int = 256
    q_heads: int = 16
    kv_heads: int = 8
    head_dim: int = 128
    chunk_size: int = 64
    context: AIEContext | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.max_seq_len % self.chunk_size != 0:
            raise ValueError("max_seq_len must be divisible by chunk_size")
        if self.q_heads % self.kv_heads != 0:
            raise ValueError("q_heads must be divisible by kv_heads")
        if self.head_dim % 32 != 0:
            raise ValueError("head_dim must be a multiple of 32")
        super().__init__(context=self.context)

    @property
    def q_size(self) -> int:
        return self.q_heads * self.head_dim

    @property
    def num_chunks(self) -> int:
        return self.max_seq_len // self.chunk_size

    @property
    def packed_chunk_elements(self) -> int:
        return 2 * self.chunk_size * self.head_dim + self.chunk_size

    @property
    def packed_stream_elements(self) -> int:
        return self.q_heads * self.num_chunks * self.packed_chunk_elements

    def get_mlir_artifact(self) -> PythonGeneratedMLIRArtifact:
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "d1_fixed_cache_attention_context",
                (
                    aie_utils.get_current_device(),
                    self.max_seq_len,
                    self.q_heads,
                    self.kv_heads,
                    self.head_dim,
                    self.chunk_size,
                ),
            ),
        )

    def get_kernel_artifacts(self) -> list[KernelObjectArtifact]:
        return [
            KernelObjectArtifact(
                "d1_fixed_attention.o",
                dependencies=[
                    SourceArtifact(self.operator_dir / "d1_fixed_attention.cc")
                ],
                extra_flags=[
                    f"-DD1_HEAD_DIM={self.head_dim}",
                    f"-DD1_CHUNK_SIZE={self.chunk_size}",
                    f"-DD1_ATTN_SCALE={self.head_dim ** -0.5}f",
                ],
            )
        ]

    def get_arg_spec(self) -> list[AIERuntimeArgSpec]:
        return [
            AIERuntimeArgSpec("in", (self.q_size,)),
            AIERuntimeArgSpec("in", (self.packed_stream_elements,)),
            AIERuntimeArgSpec("out", (self.q_size,)),
        ]


def _make_mask(max_seq_len: int, position: int) -> torch.Tensor:
    mask = torch.zeros(max_seq_len, dtype=torch.float32)
    mask[: position + 1] = 1.0
    return mask.to(torch.bfloat16).contiguous()


def _pack_q_head_cache_stream(
    *,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    mask: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    chunk_size: int,
) -> torch.Tensor:
    max_seq_len = k_cache.shape[1]
    head_dim = k_cache.shape[2]
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


def _build_real_qwen3_attention_case(
    *,
    model: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    max_seq_len: int,
) -> tuple[int, int, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    expected_context = refs["attn_context"].flatten().contiguous()
    return next_token, position, q, k_cache, v_cache, expected_context


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="D1.0 real Qwen3 fixed-cache attention context."
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--raw-prompt", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--rel-tol", type=float, default=0.08)
    parser.add_argument("--abs-tol", type=float, default=0.08)
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument(
        "--build-dir",
        type=Path,
        default=Path("build_new_mega_d1_single_layer"),
    )
    return parser.parse_args()


def _print_preflight(result) -> None:
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


def main() -> None:
    args = parse_args()
    model_dir = resolve_model_dir(args.model, args.revision)
    model = Qwen3ForCausalLM(model_dir)
    if model.config.hidden_size != 1024:
        raise ValueError("D1.0 expects Qwen3-0.6B hidden_size=1024")
    if model.config.head_dim != 128:
        raise ValueError("D1.0 expects Qwen3-0.6B head_dim=128")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    input_ids = encode_prompt(
        tokenizer,
        args.prompt,
        raw_prompt=args.raw_prompt,
        enable_thinking=args.enable_thinking,
    )
    if input_ids.shape[1] >= args.max_seq_len:
        raise ValueError("prompt must fit inside max_seq_len")

    context = AIEContext(build_dir=args.build_dir)
    op = D1FixedCacheAttentionContext(
        max_seq_len=args.max_seq_len,
        q_heads=model.config.num_attention_heads,
        kv_heads=model.config.num_key_value_heads,
        head_dim=model.config.head_dim,
        chunk_size=args.chunk_size,
        context=context,
    )
    op.compile()
    mlir_path = Path(op.xclbin_artifact.mlir_input.filename)
    try:
        preflight = run_persistent_artifact_preflight(
            mlir_path=mlir_path,
            arg_specs=len(op.get_arg_spec()),
        )
    except Qwen3PreflightError as err:
        print("experiment: D1.0 real fixed-cache attention context")
        print(f"build_dir: {args.build_dir}")
        print(f"mlir: {mlir_path}")
        print("decision: rejected")
        print(f"root_cause: preflight failed: {err}")
        raise SystemExit(1) from err

    print("experiment: D1.0 real fixed-cache attention context")
    print(f"build_dir: {args.build_dir}")
    print(f"model_dir: {model_dir}")
    print(f"xclbin: {op.xclbin_artifact.filename}")
    print(f"runtime_bin: {op.insts_artifact.filename}")
    print(f"mlir: {mlir_path}")
    print(f"max_seq_len: {args.max_seq_len}")
    print(f"chunk_size: {args.chunk_size}")
    print(f"num_chunks: {op.num_chunks}")
    print(f"q_heads: {op.q_heads}")
    print(f"kv_heads: {op.kv_heads}")
    print(f"head_dim: {op.head_dim}")
    print(f"packed_chunk_elements: {op.packed_chunk_elements}")
    print(f"packed_chunk_bytes: {op.packed_chunk_elements * 2}")
    print(f"packed_stream_elements: {op.packed_stream_elements}")
    print(f"host_kv_writeback_bytes: {2 * op.kv_heads * op.head_dim * 2}")
    _print_preflight(preflight)

    if args.compile_only:
        print("decision: compile-only")
        return

    next_token, position, q, k_cache, v_cache, expected_context = (
        _build_real_qwen3_attention_case(
            model=model,
            input_ids=input_ids,
            max_seq_len=args.max_seq_len,
        )
    )
    mask = _make_mask(args.max_seq_len, position)
    packed_stream = _pack_q_head_cache_stream(
        k_cache=k_cache,
        v_cache=v_cache,
        mask=mask,
        q_heads=op.q_heads,
        kv_heads=op.kv_heads,
        chunk_size=args.chunk_size,
    )

    q_buf = XRTTensor.from_torch(q)
    packed_buf = XRTTensor.from_torch(packed_stream)
    out_buf = XRTTensor((op.q_size,), dtype=bfloat16)
    result = op.get_callable()(q_buf, packed_buf, out_buf)
    out_buf.device = "npu"
    actual_context = out_buf.to_torch().detach().clone().flatten()

    print(f"prompt_tokens: {input_ids.shape[1]}")
    print(f"decode_position: {position}")
    print(f"prompt_next_token: {next_token}")
    print(f"npu_time_us: {result.npu_time / 1000.0:.3f}")
    errors = print_tensor_check(
        "d1_attention_context",
        actual_context,
        expected_context,
        rel_tol=args.rel_tol,
        abs_tol=args.abs_tol,
    )

    if errors == 0:
        print("decision: accepted")
        print(
            "root_cause: real Qwen3 layer-0 attention context matches when the "
            "host owns current K/V writeback and the NPU reads a fixed full-cache "
            "stream masked by runtime data."
        )
    else:
        print("decision: rejected")
        print(
            "root_cause: fixed-cache real Qwen3 attention context did not match "
            "the PyTorch/reference tensor under the requested tolerance."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
