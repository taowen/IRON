#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass, field

import aie.utils as aie_utils

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
class NewMegaFixedCacheAttentionContext(MLIROperator):
    """Production fixed-cache Qwen3 decode-attention context stage.

    This is the first production new-mega stage. Inputs are real Qwen3
    q_norm+RoPE queries and a host-updated full K/V cache stream. The graph
    shape is fixed for max_seq_len; live position is represented by mask data
    inside the packed cache stream.
    """

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
                "fixed_cache_attention_context",
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
                "fixed_attention.o",
                dependencies=[SourceArtifact(self.operator_dir / "fixed_attention.cc")],
                extra_flags=[
                    f"-DNEW_MEGA_HEAD_DIM={self.head_dim}",
                    f"-DNEW_MEGA_CHUNK_SIZE={self.chunk_size}",
                    f"-DNEW_MEGA_ATTN_SCALE={self.head_dim ** -0.5}f",
                ],
            )
        ]

    def get_arg_spec(self) -> list[AIERuntimeArgSpec]:
        return [
            AIERuntimeArgSpec("in", (self.q_size,)),
            AIERuntimeArgSpec("in", (self.packed_stream_elements,)),
            AIERuntimeArgSpec("out", (self.q_size,)),
        ]
