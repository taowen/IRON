#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass, field
from typing import ClassVar

import aie.utils as aie_utils

from iron.common import (
    AIERuntimeArgSpec,
    DesignGenerator,
    KernelObjectArtifact,
    MLIROperator,
    PythonGeneratedMLIRArtifact,
    SourceArtifact,
)
from iron.common.device_utils import get_kernel_dir


def verification_tolerance(stage: str, name: str) -> tuple[float, float]:
    rope_stages = {
        "input-rmsnorm-qkv-rope-cache",
        "input-rmsnorm-qkv-rope-cache-scores-softmax",
        "input-rmsnorm-qkv-rope-cache-scores-softmax-context",
        "input-rmsnorm-qkv-rope-cache-scores-softmax-context-o-proj",
    }
    if stage in rope_stages and name in {"queries", "keys"}:
        return 0.05, 0.5
    if stage in {
        "input-rmsnorm-qkv-rope-cache-scores-softmax-context",
        "input-rmsnorm-qkv-rope-cache-scores-softmax-context-o-proj",
    } and name in {
        "attn_context",
        "attn_context_flat",
    }:
        return 0.05, 0.5
    if (
        stage
        in {
            "post-attn-rmsnorm-mlp-gate-up",
            "post-attn-rmsnorm-full-mlp",
            "input-rmsnorm-qkv-rope-cache-scores-softmax-context-o-proj-full-mlp",
        }
        and name == "ffn_gate_silu"
    ):
        return 0.04, 0.025
    if (
        stage == "input-rmsnorm-qkv-rope-cache-scores-softmax-context-o-proj-full-mlp"
        and name == "ffn_hidden"
    ):
        return 0.04, 0.025
    if (
        stage == "input-rmsnorm-qkv-rope-cache-scores-softmax-context-o-proj-full-mlp"
        and name == "attn_residual"
    ):
        return 0.04, 0.016
    return 0.04, 1e-6


@dataclass
class Qwen3PersistentInputRMSNorm(MLIROperator):
    """Single-token persistent Qwen3 input RMSNorm stage."""

    hidden_size: int = 1024
    epsilon: float = 1e-6
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[dict[str, str]] = {
        **MLIROperator._name_aliases,
        "hidden_size": "h",
    }

    def __post_init__(self):
        if self.hidden_size != 1024:
            raise ValueError(
                f"Qwen3-0.6B persistent input RMSNorm expects hidden_size=1024, got {self.hidden_size}"
            )
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")
        MLIROperator.__init__(self, context=self.context)

    @property
    def _epsilon_tag(self):
        return f"eps_{self.epsilon:.0e}".replace("-", "m")

    @property
    def _kernel_object(self):
        return f"qwen3_persistent_rms_norm_{self._epsilon_tag}.o"

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "qwen3_persistent_input_rmsnorm",
                (
                    aie_utils.get_current_device(),
                    self.hidden_size,
                    0,
                ),
                {"kernel_object": self._kernel_object},
            ),
        )

    def get_kernel_artifacts(self):
        arch_dir = get_kernel_dir()
        return [
            KernelObjectArtifact(
                self._kernel_object,
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / arch_dir / "rms_norm.cc"
                    )
                ],
                extra_flags=[f"-DRMS_NORM_EPSILON={self.epsilon}f"],
            ),
            KernelObjectArtifact(
                "mul.o",
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / "generic" / "mul.cc"
                    )
                ],
            ),
        ]

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (self.hidden_size,)),
            AIERuntimeArgSpec("in", (self.hidden_size,)),
            AIERuntimeArgSpec("out", (self.hidden_size,)),
        ]


@dataclass
class Qwen3PersistentInputRMSNormQKV(MLIROperator):
    """Single-token persistent Qwen3 input RMSNorm + QKV projection stage."""

    hidden_size: int = 1024
    q_size: int = 2048
    kv_size: int = 1024
    num_aie_columns: int = 1
    tile_size_input: int = 4
    tile_size_output: int = 64
    epsilon: float = 1e-6
    kernel_vector_size: int = field(default=64, repr=False)
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[dict[str, str]] = {
        **MLIROperator._name_aliases,
        "hidden_size": "h",
        "q_size": "q",
        "kv_size": "kv",
        "num_aie_columns": "col",
        "tile_size_input": "tsi",
        "tile_size_output": "tso",
    }

    def __post_init__(self):
        if self.hidden_size != 1024:
            raise ValueError(
                f"Qwen3-0.6B persistent QKV expects hidden_size=1024, got {self.hidden_size}"
            )
        if self.q_size != 2048 or self.kv_size != 1024:
            raise ValueError(
                f"Qwen3-0.6B persistent QKV expects q_size=2048 and kv_size=1024, "
                f"got q_size={self.q_size} kv_size={self.kv_size}"
            )
        if self.hidden_size % self.kernel_vector_size != 0:
            raise ValueError("hidden_size must be a multiple of kernel_vector_size")
        if self.q_size % self.num_aie_columns != 0:
            raise ValueError("q_size must be divisible by num_aie_columns")
        if self.kv_size % self.num_aie_columns != 0:
            raise ValueError("kv_size must be divisible by num_aie_columns")
        if self.tile_size_output % self.tile_size_input != 0:
            raise ValueError("tile_size_output must be a multiple of tile_size_input")
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")
        MLIROperator.__init__(self, context=self.context)

    @property
    def _epsilon_tag(self):
        return f"eps_{self.epsilon:.0e}".replace("-", "m")

    @property
    def _rms_kernel_object(self):
        return f"qwen3_persistent_rms_norm_{self._epsilon_tag}.o"

    @property
    def _gemv_kernel_object(self):
        return (
            f"qwen3_persistent_gemv_{self.hidden_size}k_{self.kernel_vector_size}vs.o"
        )

    @property
    def packed_weights_size(self):
        return (
            self.hidden_size
            + self.q_size * self.hidden_size
            + 2 * self.kv_size * self.hidden_size
        )

    @property
    def packed_outputs_size(self):
        return self.hidden_size + self.q_size + 2 * self.kv_size

    @property
    def q_output_base(self):
        return self.hidden_size

    @property
    def k_output_base(self):
        return self.q_output_base + self.q_size

    @property
    def v_output_base(self):
        return self.k_output_base + self.kv_size

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "qwen3_persistent_input_rmsnorm_qkv",
                (
                    aie_utils.get_current_device(),
                    self.hidden_size,
                    self.q_size,
                    self.kv_size,
                    self.num_aie_columns,
                    self.tile_size_input,
                    self.tile_size_output,
                    0,
                ),
                {
                    "rms_kernel_object": self._rms_kernel_object,
                    "gemv_kernel_object": self._gemv_kernel_object,
                },
            ),
        )

    def get_kernel_artifacts(self):
        arch_dir = get_kernel_dir()
        return [
            KernelObjectArtifact(
                self._rms_kernel_object,
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / arch_dir / "rms_norm.cc"
                    )
                ],
                extra_flags=[f"-DRMS_NORM_EPSILON={self.epsilon}f"],
            ),
            KernelObjectArtifact(
                "mul.o",
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / "generic" / "mul.cc"
                    )
                ],
            ),
            KernelObjectArtifact(
                self._gemv_kernel_object,
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / "generic" / "mv.cc"
                    )
                ],
                extra_flags=[
                    f"-DDIM_K={self.hidden_size}",
                    f"-DVEC_SIZE={self.kernel_vector_size}",
                ],
            ),
        ]

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (self.hidden_size,)),
            AIERuntimeArgSpec("in", (self.packed_weights_size,)),
            AIERuntimeArgSpec("out", (self.packed_outputs_size,)),
        ]


@dataclass
class Qwen3PersistentInputRMSNormQKVRopeCache(Qwen3PersistentInputRMSNormQKV):
    """Single-token persistent Qwen3 input RMSNorm + QKV + RoPE + KV write."""

    head_dim: int = 128
    max_seq_len: int = 256
    position: int = 0
    tile_size_output: int = 128

    _name_aliases: ClassVar[dict[str, str]] = {
        **Qwen3PersistentInputRMSNormQKV._name_aliases,
        "head_dim": "hd",
        "max_seq_len": "msl",
        "position": "pos",
    }

    def __post_init__(self):
        super().__post_init__()
        if self.head_dim != 128:
            raise ValueError(
                f"Qwen3-0.6B persistent RoPE cache expects head_dim=128, got {self.head_dim}"
            )
        if self.q_size % self.head_dim != 0 or self.kv_size % self.head_dim != 0:
            raise ValueError("q_size and kv_size must be divisible by head_dim")
        if self.tile_size_output != self.head_dim:
            raise ValueError(
                "tile_size_output must equal head_dim for RoPE cache stage"
            )
        if self.max_seq_len < 256:
            raise ValueError("max_seq_len must be at least 256")
        if not (0 <= self.position < self.max_seq_len):
            raise ValueError(
                f"position must be in [0, {self.max_seq_len}), got {self.position}"
            )

    @property
    def q_heads(self):
        return self.q_size // self.head_dim

    @property
    def kv_heads(self):
        return self.kv_size // self.head_dim

    @property
    def _rope_kernel_object(self):
        return "qwen3_persistent_rope_0.o"

    @property
    def packed_weights_size(self):
        return super().packed_weights_size + 2 * self.head_dim

    @property
    def packed_outputs_size(self):
        return (
            self.hidden_size
            + self.q_size
            + self.kv_size
            + self.q_size
            + self.kv_size
            + self.q_size
        )

    @property
    def q_norm_weight_base(self):
        return super().packed_weights_size

    @property
    def k_norm_weight_base(self):
        return self.q_norm_weight_base + self.head_dim

    @property
    def q_raw_output_base(self):
        return self.hidden_size

    @property
    def k_raw_output_base(self):
        return self.q_raw_output_base + self.q_size

    @property
    def q_norm_output_base(self):
        return self.k_raw_output_base + self.kv_size

    @property
    def k_norm_output_base(self):
        return self.q_norm_output_base + self.q_size

    @property
    def q_rope_output_base(self):
        return self.k_norm_output_base + self.kv_size

    @property
    def k_rope_output_base(self):
        return self.q_rope_output_base + self.q_size

    @property
    def cache_half_size(self):
        return self.kv_size * self.max_seq_len

    @property
    def packed_cache_size(self):
        return 2 * self.cache_half_size

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "attention_design.py",
                "qwen3_persistent_input_rmsnorm_qkv_rope_cache",
                (
                    aie_utils.get_current_device(),
                    self.hidden_size,
                    self.q_size,
                    self.kv_size,
                    self.head_dim,
                    self.max_seq_len,
                    self.position,
                    self.num_aie_columns,
                    self.tile_size_input,
                    self.tile_size_output,
                    0,
                ),
                {
                    "rms_kernel_object": self._rms_kernel_object,
                    "gemv_kernel_object": self._gemv_kernel_object,
                    "rope_kernel_object": self._rope_kernel_object,
                },
            ),
        )

    def get_kernel_artifacts(self):
        artifacts = super().get_kernel_artifacts()
        artifacts.append(
            KernelObjectArtifact(
                self._rope_kernel_object,
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / "generic" / "rope.cc"
                    )
                ],
                extra_flags=["-DTWO_HALVES"],
            )
        )
        return artifacts

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (self.hidden_size,)),
            AIERuntimeArgSpec("in", (self.packed_weights_size,)),
            AIERuntimeArgSpec("in", (self.head_dim,)),
            AIERuntimeArgSpec("out", (self.packed_outputs_size,)),
            AIERuntimeArgSpec("inout", (self.packed_cache_size,)),
        ]


@dataclass
class Qwen3PersistentInputRMSNormQKVRopeCacheScoresSoftmax(
    Qwen3PersistentInputRMSNormQKVRopeCache
):
    """Single-token persistent Qwen3 stage through attention scores and softmax."""

    def __post_init__(self):
        super().__post_init__()
        if self.num_aie_columns != 1:
            raise ValueError(
                "scores+softmax checkpoint is currently single-column only"
            )
        if self.max_seq_len % 64 != 0:
            raise ValueError(
                "softmax checkpoint requires max_seq_len to be a multiple of 64"
            )

    @property
    def _attention_kernel_object(self):
        return (
            f"qwen3_persistent_attention_scores_hd{self.head_dim}_"
            f"msl{self.max_seq_len}.o"
        )

    @property
    def _softmax_kernel_object(self):
        return "qwen3_persistent_softmax.o"

    @property
    def _passthrough_kernel_object(self):
        return "qwen3_persistent_passThrough_bf16.o"

    @property
    def score_size(self):
        return self.q_heads * self.max_seq_len

    @property
    def qk_pair_debug_size(self):
        return self.kv_heads * 3 * self.head_dim

    @property
    def k_cache_debug_size(self):
        return self.kv_heads * self.max_seq_len * self.head_dim

    @property
    def packed_outputs_size(self):
        return (
            super().packed_outputs_size
            + self.qk_pair_debug_size
            + self.k_cache_debug_size
            + 2 * self.score_size
        )

    @property
    def qk_pair_output_base(self):
        return super().packed_outputs_size

    @property
    def k_cache_stream_output_base(self):
        return self.qk_pair_output_base + self.qk_pair_debug_size

    @property
    def attn_scores_output_base(self):
        return self.k_cache_stream_output_base + self.k_cache_debug_size

    @property
    def attn_weights_output_base(self):
        return self.attn_scores_output_base + self.score_size

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "attention_design.py",
                "qwen3_persistent_input_rmsnorm_qkv_rope_cache_scores_softmax",
                (
                    aie_utils.get_current_device(),
                    self.hidden_size,
                    self.q_size,
                    self.kv_size,
                    self.head_dim,
                    self.max_seq_len,
                    self.position,
                    self.num_aie_columns,
                    self.tile_size_input,
                    self.tile_size_output,
                    0,
                ),
                {
                    "rms_kernel_object": self._rms_kernel_object,
                    "gemv_kernel_object": self._gemv_kernel_object,
                    "rope_kernel_object": self._rope_kernel_object,
                    "attention_kernel_object": self._attention_kernel_object,
                    "passthrough_kernel_object": self._passthrough_kernel_object,
                    "softmax_kernel_object": self._softmax_kernel_object,
                },
            ),
        )

    def get_kernel_artifacts(self):
        arch_dir = get_kernel_dir()
        if arch_dir != "aie2p":
            raise ValueError(
                "scores+softmax persistent checkpoint currently targets NPU2/aie2p"
            )
        artifacts = super().get_kernel_artifacts()
        artifacts.extend(
            [
                KernelObjectArtifact(
                    self._attention_kernel_object,
                    dependencies=[
                        SourceArtifact(
                            self.context.base_dir
                            / "aie_kernels"
                            / "generic"
                            / "qwen3_attention.cc"
                        )
                    ],
                    extra_flags=[
                        f"-DHEAD_DIM={self.head_dim}",
                        f"-DMAX_SEQ_LEN={self.max_seq_len}",
                        "-DCACHE_BLOCK=64",
                        f"-DHIDDEN_SIZE={self.hidden_size}",
                        f"-DRMS_NORM_EPSILON={self.epsilon}f",
                        f"-DATTN_SCALE={1.0 / math.sqrt(self.head_dim)}f",
                    ],
                ),
                KernelObjectArtifact(
                    self._softmax_kernel_object,
                    dependencies=[
                        SourceArtifact(
                            self.context.base_dir
                            / "aie_kernels"
                            / arch_dir
                            / "softmax.cc"
                        )
                    ],
                ),
                KernelObjectArtifact(
                    self._passthrough_kernel_object,
                    dependencies=[
                        SourceArtifact(
                            self.context.base_dir
                            / "aie_kernels"
                            / "generic"
                            / "passThrough.cc"
                        )
                    ],
                    extra_flags=["-DBIT_WIDTH=16"],
                ),
            ]
        )
        return artifacts


@dataclass
class Qwen3PersistentInputRMSNormQKVRopeCacheScoresSoftmaxContext(
    Qwen3PersistentInputRMSNormQKVRopeCacheScoresSoftmax
):
    """Single-token persistent Qwen3 stage through attention context."""

    @property
    def v_cache_debug_size(self):
        return self.kv_heads * self.max_seq_len * self.head_dim

    @property
    def context_size(self):
        return self.q_size

    @property
    def packed_outputs_size(self):
        return super().packed_outputs_size + self.v_cache_debug_size + self.context_size

    @property
    def v_context_stream_output_base(self):
        return self.attn_weights_output_base + self.score_size

    @property
    def attn_context_output_base(self):
        return self.v_context_stream_output_base + self.v_cache_debug_size

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "attention_design.py",
                "qwen3_persistent_input_rmsnorm_qkv_rope_cache_scores_softmax_context",
                (
                    aie_utils.get_current_device(),
                    self.hidden_size,
                    self.q_size,
                    self.kv_size,
                    self.head_dim,
                    self.max_seq_len,
                    self.position,
                    self.num_aie_columns,
                    self.tile_size_input,
                    self.tile_size_output,
                    0,
                ),
                {
                    "rms_kernel_object": self._rms_kernel_object,
                    "gemv_kernel_object": self._gemv_kernel_object,
                    "rope_kernel_object": self._rope_kernel_object,
                    "attention_kernel_object": self._attention_kernel_object,
                    "passthrough_kernel_object": self._passthrough_kernel_object,
                    "softmax_kernel_object": self._softmax_kernel_object,
                },
            ),
        )


@dataclass
class Qwen3PersistentInputRMSNormQKVRopeCacheScoresSoftmaxContextOProj(
    Qwen3PersistentInputRMSNormQKVRopeCacheScoresSoftmaxContext
):
    """Single-token persistent Qwen3 stage through attention O projection and residual add."""

    @property
    def k_cache_debug_size(self):
        return 0

    @property
    def context_flat_size(self):
        return self.q_size

    @property
    def o_proj_size(self):
        return self.hidden_size

    @property
    def residual_size(self):
        return self.hidden_size

    @property
    def _o_gemv_kernel_object(self):
        return (
            f"qwen3_persistent_gemv_{self.q_size}k_{self.kernel_vector_size}vs_o_proj.o"
        )

    @property
    def _o_gemv_vectorized_fn(self):
        return "qwen3_o_proj_matvec_vectorized_bf16_bf16"

    @property
    def _o_gemv_scalar_fn(self):
        return "qwen3_o_proj_matvec_scalar_bf16_bf16"

    @property
    def _add_kernel_object(self):
        return "qwen3_persistent_add.o"

    @property
    def o_weight_base(self):
        return super().packed_weights_size

    @property
    def packed_weights_size(self):
        return super().packed_weights_size + self.hidden_size * self.q_size

    @property
    def packed_outputs_size(self):
        return (
            super().packed_outputs_size
            + self.context_flat_size
            + self.o_proj_size
            + self.residual_size
        )

    @property
    def attn_context_flat_output_base(self):
        return super().packed_outputs_size

    @property
    def attn_o_proj_output_base(self):
        return self.attn_context_flat_output_base + self.context_flat_size

    @property
    def attn_residual_output_base(self):
        return self.attn_o_proj_output_base + self.o_proj_size

    def __post_init__(self):
        super().__post_init__()
        if self.q_size % self.kernel_vector_size != 0:
            raise ValueError("q_size must be a multiple of kernel_vector_size")
        if self._o_gemv_kernel_object == self._gemv_kernel_object:
            raise ValueError("O projection GEMV must use a distinct kernel object")

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "attention_design.py",
                "qwen3_persistent_input_rmsnorm_qkv_rope_cache_scores_softmax_context_o_proj",
                (
                    aie_utils.get_current_device(),
                    self.hidden_size,
                    self.q_size,
                    self.kv_size,
                    self.head_dim,
                    self.max_seq_len,
                    self.position,
                    self.num_aie_columns,
                    self.tile_size_input,
                    self.tile_size_output,
                    0,
                ),
                {
                    "rms_kernel_object": self._rms_kernel_object,
                    "gemv_kernel_object": self._gemv_kernel_object,
                    "rope_kernel_object": self._rope_kernel_object,
                    "attention_kernel_object": self._attention_kernel_object,
                    "passthrough_kernel_object": self._passthrough_kernel_object,
                    "softmax_kernel_object": self._softmax_kernel_object,
                    "o_gemv_kernel_object": self._o_gemv_kernel_object,
                    "add_kernel_object": self._add_kernel_object,
                },
            ),
        )

    def get_kernel_artifacts(self):
        artifacts = super().get_kernel_artifacts()
        artifacts.extend(
            [
                KernelObjectArtifact(
                    self._o_gemv_kernel_object,
                    dependencies=[
                        SourceArtifact(
                            self.context.base_dir / "aie_kernels" / "generic" / "mv.cc"
                        )
                    ],
                    extra_flags=[
                        f"-DDIM_K={self.q_size}",
                        f"-DVEC_SIZE={self.kernel_vector_size}",
                        f"-DMATVEC_SCALAR_FN={self._o_gemv_scalar_fn}",
                        f"-DMATVEC_VECTORIZED_FN={self._o_gemv_vectorized_fn}",
                    ],
                ),
                KernelObjectArtifact(
                    self._add_kernel_object,
                    dependencies=[
                        SourceArtifact(
                            self.context.base_dir / "aie_kernels" / "generic" / "add.cc"
                        )
                    ],
                ),
            ]
        )
        return artifacts


@dataclass
class Qwen3PersistentInputRMSNormQKVRopeCacheScoresSoftmaxContextOProjFullMLP(
    Qwen3PersistentInputRMSNormQKVRopeCacheScoresSoftmaxContextOProj
):
    """Single-token persistent Qwen3 stage through attention and full MLP."""

    intermediate_size: int = 3072

    _name_aliases: ClassVar[dict[str, str]] = {
        **Qwen3PersistentInputRMSNormQKVRopeCacheScoresSoftmaxContextOProj._name_aliases,
        "intermediate_size": "ffn",
    }

    def __post_init__(self):
        super().__post_init__()
        if self.intermediate_size != 3072:
            raise ValueError(
                "Qwen3-0.6B full-layer checkpoint expects intermediate_size=3072, "
                f"got {self.intermediate_size}"
            )
        if self.num_aie_columns != 1:
            raise ValueError("full-layer checkpoint is currently single-column only")
        if self.hidden_size % self.kernel_vector_size != 0:
            raise ValueError("hidden_size must be a multiple of kernel_vector_size")
        if self.intermediate_size % self.kernel_vector_size != 0:
            raise ValueError(
                "intermediate_size must be a multiple of kernel_vector_size"
            )

    @property
    def _mlp_gemv_kernel_object(self):
        return (
            f"qwen3_persistent_gemv_{self.hidden_size}k_"
            f"{self.kernel_vector_size}vs_mlp.o"
        )

    @property
    def _mlp_gemv_vectorized_fn(self):
        return "qwen3_mlp_matvec_vectorized_bf16_bf16"

    @property
    def _mlp_gemv_scalar_fn(self):
        return "qwen3_mlp_matvec_scalar_bf16_bf16"

    @property
    def _silu_kernel_object(self):
        return "qwen3_persistent_silu.o"

    @property
    def _mul_kernel_object(self):
        return "qwen3_persistent_mul.o"

    @property
    def _down_gemv_kernel_object(self):
        return (
            f"qwen3_persistent_gemv_{self.intermediate_size}k_"
            f"{self.kernel_vector_size}vs_down_proj.o"
        )

    @property
    def _down_gemv_vectorized_fn(self):
        return "qwen3_down_proj_matvec_vectorized_bf16_bf16"

    @property
    def _down_gemv_scalar_fn(self):
        return "qwen3_down_proj_matvec_scalar_bf16_bf16"

    @property
    def mlp_post_norm_weight_base(self):
        return super().packed_weights_size

    @property
    def mlp_gate_weight_base(self):
        return self.mlp_post_norm_weight_base + self.hidden_size

    @property
    def mlp_up_weight_base(self):
        return self.mlp_gate_weight_base + self.intermediate_size * self.hidden_size

    @property
    def mlp_down_weight_base(self):
        return self.mlp_up_weight_base + self.intermediate_size * self.hidden_size

    @property
    def packed_weights_size(self):
        return super().packed_weights_size + (
            self.hidden_size + 3 * self.intermediate_size * self.hidden_size
        )

    @property
    def mlp_outputs_size(self):
        return 3 * self.hidden_size + 4 * self.intermediate_size

    @property
    def packed_outputs_size(self):
        return super().packed_outputs_size + self.mlp_outputs_size

    @property
    def mlp_x_norm_output_base(self):
        return super().packed_outputs_size

    @property
    def ffn_gate_output_base(self):
        return self.mlp_x_norm_output_base + self.hidden_size

    @property
    def ffn_up_output_base(self):
        return self.ffn_gate_output_base + self.intermediate_size

    @property
    def ffn_gate_silu_output_base(self):
        return self.ffn_up_output_base + self.intermediate_size

    @property
    def ffn_hidden_output_base(self):
        return self.ffn_gate_silu_output_base + self.intermediate_size

    @property
    def ffn_out_output_base(self):
        return self.ffn_hidden_output_base + self.intermediate_size

    @property
    def layer_residual_output_base(self):
        return self.ffn_out_output_base + self.hidden_size

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "attention_design.py",
                "qwen3_persistent_input_rmsnorm_qkv_rope_cache_scores_softmax_context_o_proj_full_mlp",
                (
                    aie_utils.get_current_device(),
                    self.hidden_size,
                    self.q_size,
                    self.kv_size,
                    self.head_dim,
                    self.max_seq_len,
                    self.position,
                    self.intermediate_size,
                    self.num_aie_columns,
                    self.tile_size_input,
                    self.tile_size_output,
                    0,
                ),
                {
                    "rms_kernel_object": self._rms_kernel_object,
                    "gemv_kernel_object": self._gemv_kernel_object,
                    "rope_kernel_object": self._rope_kernel_object,
                    "attention_kernel_object": self._attention_kernel_object,
                    "passthrough_kernel_object": self._passthrough_kernel_object,
                    "softmax_kernel_object": self._softmax_kernel_object,
                    "o_gemv_kernel_object": self._o_gemv_kernel_object,
                    "add_kernel_object": self._add_kernel_object,
                    "mlp_gemv_kernel_object": self._mlp_gemv_kernel_object,
                    "silu_kernel_object": self._silu_kernel_object,
                    "mul_kernel_object": self._mul_kernel_object,
                    "down_gemv_kernel_object": self._down_gemv_kernel_object,
                },
            ),
        )

    def get_kernel_artifacts(self):
        arch_dir = get_kernel_dir()
        artifacts = super().get_kernel_artifacts()
        artifacts.extend(
            [
                KernelObjectArtifact(
                    self._mlp_gemv_kernel_object,
                    dependencies=[
                        SourceArtifact(
                            self.context.base_dir / "aie_kernels" / "generic" / "mv.cc"
                        )
                    ],
                    extra_flags=[
                        f"-DDIM_K={self.hidden_size}",
                        f"-DVEC_SIZE={self.kernel_vector_size}",
                        f"-DMATVEC_SCALAR_FN={self._mlp_gemv_scalar_fn}",
                        f"-DMATVEC_VECTORIZED_FN={self._mlp_gemv_vectorized_fn}",
                    ],
                ),
                KernelObjectArtifact(
                    self._silu_kernel_object,
                    dependencies=[
                        SourceArtifact(
                            self.context.base_dir / "aie_kernels" / arch_dir / "silu.cc"
                        )
                    ],
                ),
                KernelObjectArtifact(
                    self._mul_kernel_object,
                    dependencies=[
                        SourceArtifact(
                            self.context.base_dir / "aie_kernels" / "generic" / "mul.cc"
                        )
                    ],
                ),
                KernelObjectArtifact(
                    self._down_gemv_kernel_object,
                    dependencies=[
                        SourceArtifact(
                            self.context.base_dir / "aie_kernels" / "generic" / "mv.cc"
                        )
                    ],
                    extra_flags=[
                        f"-DDIM_K={self.intermediate_size}",
                        f"-DVEC_SIZE={self.kernel_vector_size}",
                        f"-DMATVEC_SCALAR_FN={self._down_gemv_scalar_fn}",
                        f"-DMATVEC_VECTORIZED_FN={self._down_gemv_vectorized_fn}",
                    ],
                ),
            ]
        )
        return artifacts


@dataclass
class Qwen3PersistentNLayerFinalOnly(
    Qwen3PersistentInputRMSNormQKVRopeCacheScoresSoftmaxContextOProjFullMLP
):
    """One or more sequential Qwen3 full layers using one persistent graph."""

    max_supported_layer_iterations: ClassVar[int] = 7
    layer_iterations: int = 1

    _name_aliases: ClassVar[dict[str, str]] = {
        **Qwen3PersistentInputRMSNormQKVRopeCacheScoresSoftmaxContextOProjFullMLP._name_aliases,
        "layer_iterations": "layers",
    }

    def __post_init__(self):
        super().__post_init__()
        if self.layer_iterations < 1:
            raise ValueError("layer_iterations must be positive")
        if self.layer_iterations > self.max_supported_layer_iterations:
            raise ValueError(
                "Qwen3 n-layer final-only currently supports at most "
                f"{self.max_supported_layer_iterations} layers per chunk; "
                f"got {self.layer_iterations}. Chunk=8 still exhausts NPU "
                "BD/L1 resources in the current writeback/TAP expression."
            )

    @property
    def packed_weight_chunk_size(self):
        return self.layer_iterations * self.packed_weights_size

    @property
    def packed_cache_chunk_size(self):
        return self.layer_iterations * self.packed_cache_size

    @property
    def packed_outputs_size(self):
        return self.hidden_size

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "attention_design.py",
                "qwen3_persistent_n_layer_final_only",
                (
                    aie_utils.get_current_device(),
                    self.hidden_size,
                    self.q_size,
                    self.kv_size,
                    self.head_dim,
                    self.max_seq_len,
                    self.position,
                    self.intermediate_size,
                    self.layer_iterations,
                    self.num_aie_columns,
                    self.tile_size_input,
                    self.tile_size_output,
                    0,
                ),
                {
                    "rms_kernel_object": self._rms_kernel_object,
                    "gemv_kernel_object": self._gemv_kernel_object,
                    "rope_kernel_object": self._rope_kernel_object,
                    "attention_kernel_object": self._attention_kernel_object,
                    "passthrough_kernel_object": self._passthrough_kernel_object,
                    "softmax_kernel_object": self._softmax_kernel_object,
                    "o_gemv_kernel_object": self._o_gemv_kernel_object,
                    "add_kernel_object": self._add_kernel_object,
                    "mlp_gemv_kernel_object": self._mlp_gemv_kernel_object,
                    "silu_kernel_object": self._silu_kernel_object,
                    "mul_kernel_object": self._mul_kernel_object,
                    "down_gemv_kernel_object": self._down_gemv_kernel_object,
                },
            ),
        )

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (self.hidden_size,)),
            AIERuntimeArgSpec("in", (self.packed_weight_chunk_size,)),
            AIERuntimeArgSpec("in", (self.head_dim,)),
            AIERuntimeArgSpec("out", (self.packed_outputs_size,)),
            AIERuntimeArgSpec("inout", (self.packed_cache_chunk_size,)),
        ]


@dataclass
class Qwen3PersistentPostAttnRMSNormMLPGateUp(MLIROperator):
    """Single-token persistent post-attention RMSNorm + MLP gate/up stage."""

    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_aie_columns: int = 1
    tile_size_input: int = 4
    tile_size_output: int = 384
    epsilon: float = 1e-6
    kernel_vector_size: int = field(default=64, repr=False)
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[dict[str, str]] = {
        **MLIROperator._name_aliases,
        "hidden_size": "h",
        "intermediate_size": "ffn",
        "num_aie_columns": "col",
        "tile_size_input": "tsi",
        "tile_size_output": "tso",
    }

    def __post_init__(self):
        if self.hidden_size != 1024:
            raise ValueError(
                f"Qwen3-0.6B persistent MLP expects hidden_size=1024, got {self.hidden_size}"
            )
        if self.intermediate_size != 3072:
            raise ValueError(
                "Qwen3-0.6B persistent MLP expects intermediate_size=3072, "
                f"got {self.intermediate_size}"
            )
        if self.hidden_size % self.kernel_vector_size != 0:
            raise ValueError("hidden_size must be a multiple of kernel_vector_size")
        if self.intermediate_size % self.num_aie_columns != 0:
            raise ValueError("intermediate_size must be divisible by num_aie_columns")
        if self.intermediate_size % self.tile_size_output != 0:
            raise ValueError("intermediate_size must be divisible by tile_size_output")
        if self.tile_size_output % self.tile_size_input != 0:
            raise ValueError("tile_size_output must be a multiple of tile_size_input")
        if self.tile_size_output % 16 != 0:
            raise ValueError("tile_size_output must be a multiple of 16")
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")
        MLIROperator.__init__(self, context=self.context)

    @property
    def _epsilon_tag(self):
        return f"eps_{self.epsilon:.0e}".replace("-", "m")

    @property
    def _rms_kernel_object(self):
        return f"qwen3_persistent_rms_norm_{self._epsilon_tag}.o"

    @property
    def _gemv_kernel_object(self):
        return (
            f"qwen3_persistent_gemv_{self.hidden_size}k_{self.kernel_vector_size}vs.o"
        )

    @property
    def _silu_kernel_object(self):
        return "qwen3_persistent_silu.o"

    @property
    def _mul_kernel_object(self):
        return "qwen3_persistent_mul.o"

    @property
    def packed_weights_size(self):
        return self.hidden_size + 2 * self.intermediate_size * self.hidden_size

    @property
    def packed_outputs_size(self):
        return self.hidden_size + 4 * self.intermediate_size

    @property
    def mlp_x_norm_output_base(self):
        return 0

    @property
    def ffn_gate_output_base(self):
        return self.hidden_size

    @property
    def ffn_up_output_base(self):
        return self.ffn_gate_output_base + self.intermediate_size

    @property
    def ffn_gate_silu_output_base(self):
        return self.ffn_up_output_base + self.intermediate_size

    @property
    def ffn_hidden_output_base(self):
        return self.ffn_gate_silu_output_base + self.intermediate_size

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "qwen3_persistent_post_attn_rmsnorm_mlp_gate_up",
                (
                    aie_utils.get_current_device(),
                    self.hidden_size,
                    self.intermediate_size,
                    self.num_aie_columns,
                    self.tile_size_input,
                    self.tile_size_output,
                    0,
                ),
                {
                    "rms_kernel_object": self._rms_kernel_object,
                    "gemv_kernel_object": self._gemv_kernel_object,
                    "silu_kernel_object": self._silu_kernel_object,
                    "mul_kernel_object": self._mul_kernel_object,
                },
            ),
        )

    def get_kernel_artifacts(self):
        arch_dir = get_kernel_dir()
        return [
            KernelObjectArtifact(
                self._rms_kernel_object,
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / arch_dir / "rms_norm.cc"
                    )
                ],
                extra_flags=[f"-DRMS_NORM_EPSILON={self.epsilon}f"],
            ),
            KernelObjectArtifact(
                self._gemv_kernel_object,
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / "generic" / "mv.cc"
                    )
                ],
                extra_flags=[
                    f"-DDIM_K={self.hidden_size}",
                    f"-DVEC_SIZE={self.kernel_vector_size}",
                ],
            ),
            KernelObjectArtifact(
                self._silu_kernel_object,
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / arch_dir / "silu.cc"
                    )
                ],
            ),
            KernelObjectArtifact(
                self._mul_kernel_object,
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / "generic" / "mul.cc"
                    )
                ],
            ),
        ]

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (self.hidden_size,)),
            AIERuntimeArgSpec("in", (self.packed_weights_size,)),
            AIERuntimeArgSpec("out", (self.packed_outputs_size,)),
        ]


@dataclass
class Qwen3PersistentPostAttnMLPDownResidual(MLIROperator):
    """Single-token persistent MLP down projection + layer residual stage."""

    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_aie_columns: int = 1
    tile_size_input: int = 4
    tile_size_output: int = 128
    kernel_vector_size: int = field(default=64, repr=False)
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[dict[str, str]] = {
        **MLIROperator._name_aliases,
        "hidden_size": "h",
        "intermediate_size": "ffn",
        "num_aie_columns": "col",
        "tile_size_input": "tsi",
        "tile_size_output": "tso",
    }

    def __post_init__(self):
        if self.hidden_size != 1024:
            raise ValueError(
                f"Qwen3-0.6B persistent MLP down expects hidden_size=1024, got {self.hidden_size}"
            )
        if self.intermediate_size != 3072:
            raise ValueError(
                "Qwen3-0.6B persistent MLP down expects intermediate_size=3072, "
                f"got {self.intermediate_size}"
            )
        if self.intermediate_size % self.kernel_vector_size != 0:
            raise ValueError(
                "intermediate_size must be a multiple of kernel_vector_size"
            )
        if self.hidden_size % self.num_aie_columns != 0:
            raise ValueError("hidden_size must be divisible by num_aie_columns")
        if self.hidden_size % self.tile_size_output != 0:
            raise ValueError("hidden_size must be divisible by tile_size_output")
        if self.tile_size_output % self.tile_size_input != 0:
            raise ValueError("tile_size_output must be a multiple of tile_size_input")
        if self.tile_size_output % 16 != 0:
            raise ValueError("tile_size_output must be a multiple of 16")
        MLIROperator.__init__(self, context=self.context)

    @property
    def _gemv_kernel_object(self):
        return (
            f"qwen3_persistent_gemv_{self.intermediate_size}k_"
            f"{self.kernel_vector_size}vs_down_proj.o"
        )

    @property
    def _gemv_vectorized_fn(self):
        return "qwen3_down_proj_matvec_vectorized_bf16_bf16"

    @property
    def _gemv_scalar_fn(self):
        return "qwen3_down_proj_matvec_scalar_bf16_bf16"

    @property
    def _add_kernel_object(self):
        return "qwen3_persistent_add.o"

    @property
    def packed_weights_size(self):
        return self.hidden_size * self.intermediate_size

    @property
    def packed_outputs_size(self):
        return 2 * self.hidden_size

    @property
    def ffn_out_output_base(self):
        return 0

    @property
    def layer_residual_output_base(self):
        return self.hidden_size

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "qwen3_persistent_post_attn_mlp_down_residual",
                (
                    aie_utils.get_current_device(),
                    self.hidden_size,
                    self.intermediate_size,
                    self.num_aie_columns,
                    self.tile_size_input,
                    self.tile_size_output,
                    0,
                ),
                {
                    "gemv_kernel_object": self._gemv_kernel_object,
                    "add_kernel_object": self._add_kernel_object,
                },
            ),
        )

    def get_kernel_artifacts(self):
        return [
            KernelObjectArtifact(
                self._gemv_kernel_object,
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / "generic" / "mv.cc"
                    )
                ],
                extra_flags=[
                    f"-DDIM_K={self.intermediate_size}",
                    f"-DVEC_SIZE={self.kernel_vector_size}",
                    f"-DMATVEC_SCALAR_FN={self._gemv_scalar_fn}",
                    f"-DMATVEC_VECTORIZED_FN={self._gemv_vectorized_fn}",
                ],
            ),
            KernelObjectArtifact(
                self._add_kernel_object,
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / "generic" / "add.cc"
                    )
                ],
            ),
        ]

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (self.intermediate_size,)),
            AIERuntimeArgSpec("in", (self.hidden_size,)),
            AIERuntimeArgSpec("in", (self.packed_weights_size,)),
            AIERuntimeArgSpec("out", (self.packed_outputs_size,)),
        ]


@dataclass
class Qwen3PersistentPostAttnRMSNormFullMLP(MLIROperator):
    """Single-token persistent post-attention RMSNorm + full MLP checkpoint."""

    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_aie_columns: int = 1
    tile_size_input: int = 4
    tile_size_output: int = 128
    epsilon: float = 1e-6
    kernel_vector_size: int = field(default=64, repr=False)
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[dict[str, str]] = {
        **MLIROperator._name_aliases,
        "hidden_size": "h",
        "intermediate_size": "ffn",
        "num_aie_columns": "col",
        "tile_size_input": "tsi",
        "tile_size_output": "tso",
    }

    def __post_init__(self):
        if self.hidden_size != 1024:
            raise ValueError(
                f"Qwen3-0.6B persistent full MLP expects hidden_size=1024, got {self.hidden_size}"
            )
        if self.intermediate_size != 3072:
            raise ValueError(
                "Qwen3-0.6B persistent full MLP expects intermediate_size=3072, "
                f"got {self.intermediate_size}"
            )
        if self.num_aie_columns != 1:
            raise ValueError("full MLP checkpoint is currently single-column only")
        if self.hidden_size % self.kernel_vector_size != 0:
            raise ValueError("hidden_size must be a multiple of kernel_vector_size")
        if self.intermediate_size % self.kernel_vector_size != 0:
            raise ValueError(
                "intermediate_size must be a multiple of kernel_vector_size"
            )
        if self.hidden_size % self.tile_size_output != 0:
            raise ValueError("hidden_size must be divisible by tile_size_output")
        if self.tile_size_output % self.tile_size_input != 0:
            raise ValueError("tile_size_output must be a multiple of tile_size_input")
        if self.tile_size_output % 16 != 0:
            raise ValueError("tile_size_output must be a multiple of 16")
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")
        MLIROperator.__init__(self, context=self.context)

    @property
    def _epsilon_tag(self):
        return f"eps_{self.epsilon:.0e}".replace("-", "m")

    @property
    def _rms_kernel_object(self):
        return f"qwen3_persistent_rms_norm_{self._epsilon_tag}.o"

    @property
    def _gemv_kernel_object(self):
        return (
            f"qwen3_persistent_gemv_{self.hidden_size}k_{self.kernel_vector_size}vs.o"
        )

    @property
    def _silu_kernel_object(self):
        return "qwen3_persistent_silu.o"

    @property
    def _mul_kernel_object(self):
        return "qwen3_persistent_mul.o"

    @property
    def _down_gemv_kernel_object(self):
        return (
            f"qwen3_persistent_gemv_{self.intermediate_size}k_"
            f"{self.kernel_vector_size}vs_down_proj.o"
        )

    @property
    def _down_gemv_vectorized_fn(self):
        return "qwen3_down_proj_matvec_vectorized_bf16_bf16"

    @property
    def _down_gemv_scalar_fn(self):
        return "qwen3_down_proj_matvec_scalar_bf16_bf16"

    @property
    def _add_kernel_object(self):
        return "qwen3_persistent_add.o"

    @property
    def packed_weights_size(self):
        return self.hidden_size + 3 * self.intermediate_size * self.hidden_size

    @property
    def packed_outputs_size(self):
        return 3 * self.hidden_size + 4 * self.intermediate_size

    @property
    def mlp_x_norm_output_base(self):
        return 0

    @property
    def mlp_xnorm_output_base(self):
        return self.mlp_x_norm_output_base

    @property
    def ffn_gate_output_base(self):
        return self.hidden_size

    @property
    def ffn_up_output_base(self):
        return self.ffn_gate_output_base + self.intermediate_size

    @property
    def ffn_gate_silu_output_base(self):
        return self.ffn_up_output_base + self.intermediate_size

    @property
    def ffn_hidden_output_base(self):
        return self.ffn_gate_silu_output_base + self.intermediate_size

    @property
    def ffn_out_output_base(self):
        return self.ffn_hidden_output_base + self.intermediate_size

    @property
    def layer_residual_output_base(self):
        return self.ffn_out_output_base + self.hidden_size

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "qwen3_persistent_post_attn_rmsnorm_full_mlp",
                (
                    aie_utils.get_current_device(),
                    self.hidden_size,
                    self.intermediate_size,
                    self.num_aie_columns,
                    self.tile_size_input,
                    self.tile_size_output,
                    0,
                ),
                {
                    "rms_kernel_object": self._rms_kernel_object,
                    "gemv_kernel_object": self._gemv_kernel_object,
                    "silu_kernel_object": self._silu_kernel_object,
                    "mul_kernel_object": self._mul_kernel_object,
                    "down_gemv_kernel_object": self._down_gemv_kernel_object,
                    "add_kernel_object": self._add_kernel_object,
                },
            ),
        )

    def get_kernel_artifacts(self):
        arch_dir = get_kernel_dir()
        return [
            KernelObjectArtifact(
                self._rms_kernel_object,
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / arch_dir / "rms_norm.cc"
                    )
                ],
                extra_flags=[f"-DRMS_NORM_EPSILON={self.epsilon}f"],
            ),
            KernelObjectArtifact(
                self._gemv_kernel_object,
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / "generic" / "mv.cc"
                    )
                ],
                extra_flags=[
                    f"-DDIM_K={self.hidden_size}",
                    f"-DVEC_SIZE={self.kernel_vector_size}",
                ],
            ),
            KernelObjectArtifact(
                self._silu_kernel_object,
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / arch_dir / "silu.cc"
                    )
                ],
            ),
            KernelObjectArtifact(
                self._mul_kernel_object,
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / "generic" / "mul.cc"
                    )
                ],
            ),
            KernelObjectArtifact(
                self._down_gemv_kernel_object,
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / "generic" / "mv.cc"
                    )
                ],
                extra_flags=[
                    f"-DDIM_K={self.intermediate_size}",
                    f"-DVEC_SIZE={self.kernel_vector_size}",
                    f"-DMATVEC_SCALAR_FN={self._down_gemv_scalar_fn}",
                    f"-DMATVEC_VECTORIZED_FN={self._down_gemv_vectorized_fn}",
                ],
            ),
            KernelObjectArtifact(
                self._add_kernel_object,
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / "generic" / "add.cc"
                    )
                ],
            ),
        ]

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (self.hidden_size,)),
            AIERuntimeArgSpec("in", (self.packed_weights_size,)),
            AIERuntimeArgSpec("out", (self.packed_outputs_size,)),
        ]
