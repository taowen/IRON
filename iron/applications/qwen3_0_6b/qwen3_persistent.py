#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import gc
import math
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import torch
import torch.nn.functional as F
import aie.utils as aie_utils
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
from iron.applications.qwen3_0_6b.qwen3_decode_reference import (  # noqa: E402
    Qwen3CachedReference,
)
from iron.applications.qwen3_0_6b.qwen3_megakernel_debug import (  # noqa: E402
    one_layer_reference_tensors,
)
from iron.applications.qwen3_0_6b.qwen3_preflight import (  # noqa: E402
    run_persistent_artifact_preflight,
)
from iron.common import (  # noqa: E402
    AIERuntimeArgSpec,
    DesignGenerator,
    KernelObjectArtifact,
    MLIROperator,
    PythonGeneratedMLIRArtifact,
    SourceArtifact,
)
from iron.common.context import AIEContext  # noqa: E402
from iron.common.device_utils import get_kernel_dir  # noqa: E402
from iron.common.test_utils import verify_buffer  # noqa: E402
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor  # noqa: E402


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
    if stage == "post-attn-rmsnorm-mlp-gate-up" and name == "ffn_gate_silu":
        return 0.04, 0.025
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
                self.operator_dir / "qwen3_persistent_design.py",
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
                self.operator_dir / "qwen3_persistent_design.py",
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
                self.operator_dir / "qwen3_persistent_design.py",
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
                self.operator_dir / "qwen3_persistent_design.py",
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
                self.operator_dir / "qwen3_persistent_design.py",
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
                self.operator_dir / "qwen3_persistent_design.py",
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
                self.operator_dir / "qwen3_persistent_design.py",
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


def assert_standard_runtime_available():
    import pyxrt  # noqa: F401


def build_reference_input(
    model: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    max_seq_len: int,
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    ref = Qwen3CachedReference(model, max_seq_len, num_layers=1)
    prefill_logits, _ = ref.prefill(input_ids)
    next_token = int(torch.argmax(prefill_logits[:, -1, :], dim=-1).item())
    hidden = model.embed(torch.tensor([[next_token]], dtype=torch.long)).flatten()
    weight = model.w("model.layers.0.input_layernorm.weight").flatten()
    expected = rms_norm(
        hidden.view(1, 1, -1),
        weight,
        model.config.rms_norm_eps,
    ).flatten()
    return next_token, hidden.contiguous(), weight.contiguous(), expected.contiguous()


def build_reference_qkv(
    model: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    max_seq_len: int,
):
    ref = Qwen3CachedReference(model, max_seq_len, num_layers=1)
    prefill_logits, state = ref.prefill(input_ids)
    next_token = int(torch.argmax(prefill_logits[:, -1, :], dim=-1).item())
    hidden = model.embed(torch.tensor([[next_token]], dtype=torch.long)).flatten()
    references = one_layer_reference_tensors(model, next_token, state, max_seq_len)
    attn = "model.layers.0.self_attn"
    return (
        next_token,
        {
            "hidden": hidden.contiguous(),
            "input_norm_weight": model.w("model.layers.0.input_layernorm.weight")
            .flatten()
            .contiguous(),
            "W_q": model.w(f"{attn}.q_proj.weight").contiguous(),
            "W_k": model.w(f"{attn}.k_proj.weight").contiguous(),
            "W_v": model.w(f"{attn}.v_proj.weight").contiguous(),
        },
        {
            "x_norm": references["x_norm"].flatten().contiguous(),
            "queries_raw": references["queries_raw"].flatten().contiguous(),
            "keys_raw": references["keys_raw"].flatten().contiguous(),
            "values": references["values"].flatten().contiguous(),
        },
    )


def rope_lut_for_position(
    head_dim: int, rope_theta: float, position: int
) -> torch.Tensor:
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    freqs = position * inv_freq
    lut = torch.empty(head_dim, dtype=torch.bfloat16)
    lut[::2] = freqs.cos().to(torch.bfloat16)
    lut[1::2] = freqs.sin().to(torch.bfloat16)
    return lut.contiguous()


def build_reference_qkv_rope_cache(
    model: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    max_seq_len: int,
):
    ref = Qwen3CachedReference(model, max_seq_len, num_layers=1)
    prefill_logits, state = ref.prefill(input_ids)
    next_token = int(torch.argmax(prefill_logits[:, -1, :], dim=-1).item())
    hidden = model.embed(torch.tensor([[next_token]], dtype=torch.long)).flatten()
    references = one_layer_reference_tensors(model, next_token, state, max_seq_len)
    attn = "model.layers.0.self_attn"
    initial_cache = torch.cat(
        [state.keys[0].flatten(), state.values[0].flatten()]
    ).contiguous()
    return (
        next_token,
        state.position,
        {
            "hidden": hidden.contiguous(),
            "input_norm_weight": model.w("model.layers.0.input_layernorm.weight")
            .flatten()
            .contiguous(),
            "W_q": model.w(f"{attn}.q_proj.weight").contiguous(),
            "W_k": model.w(f"{attn}.k_proj.weight").contiguous(),
            "W_v": model.w(f"{attn}.v_proj.weight").contiguous(),
            "W_o": model.w(f"{attn}.o_proj.weight").contiguous(),
            "W_q_norm": model.w(f"{attn}.q_norm.weight").flatten().contiguous(),
            "W_k_norm": model.w(f"{attn}.k_norm.weight").flatten().contiguous(),
            "rope_angles": rope_lut_for_position(
                model.config.head_dim, model.config.rope_theta, state.position
            ),
            "initial_cache": initial_cache,
            "initial_keys_cache": state.keys[0].contiguous(),
            "initial_values_cache": state.values[0].contiguous(),
        },
        {
            "x_norm": references["x_norm"].flatten().contiguous(),
            "queries_raw": references["queries_raw"].flatten().contiguous(),
            "keys_raw": references["keys_raw"].flatten().contiguous(),
            "values": references["values"].flatten().contiguous(),
            "queries_norm": references["queries_norm"].flatten().contiguous(),
            "keys_norm": references["keys_norm"].flatten().contiguous(),
            "queries": references["queries"].flatten().contiguous(),
            "keys": references["keys"].flatten().contiguous(),
            "attn_context": references["attn_context"].flatten().contiguous(),
            "attn_out": references["attn_out"].flatten().contiguous(),
            "attn_residual": references["attn_residual"].flatten().contiguous(),
        },
    )


def build_reference_mlp_gate_up(
    model: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    max_seq_len: int,
):
    ref = Qwen3CachedReference(model, max_seq_len, num_layers=1)
    prefill_logits, state = ref.prefill(input_ids)
    next_token = int(torch.argmax(prefill_logits[:, -1, :], dim=-1).item())
    references = one_layer_reference_tensors(model, next_token, state, max_seq_len)
    layer = "model.layers.0"
    mlp = f"{layer}.mlp"
    attn_residual = references["attn_residual"].flatten().contiguous()
    post_norm_weight = model.w(f"{layer}.post_attention_layernorm.weight").flatten()
    mlp_x_norm = rms_norm(
        attn_residual.view(1, 1, -1),
        post_norm_weight,
        model.config.rms_norm_eps,
    ).flatten()
    w_gate = model.w(f"{mlp}.gate_proj.weight").contiguous()
    w_up = model.w(f"{mlp}.up_proj.weight").contiguous()
    ffn_gate = F.linear(mlp_x_norm.view(1, 1, -1), w_gate).flatten()
    ffn_up = F.linear(mlp_x_norm.view(1, 1, -1), w_up).flatten()
    ffn_gate_silu = F.silu(ffn_gate)
    ffn_hidden = ffn_gate_silu * ffn_up
    return (
        next_token,
        {
            "attn_residual": attn_residual.contiguous(),
            "post_norm_weight": post_norm_weight.contiguous(),
            "W_gate": w_gate,
            "W_up": w_up,
        },
        {
            "mlp_x_norm": mlp_x_norm.contiguous(),
            "ffn_gate": ffn_gate.contiguous(),
            "ffn_up": ffn_up.contiguous(),
            "ffn_gate_silu": ffn_gate_silu.contiguous(),
            "ffn_hidden": ffn_hidden.contiguous(),
        },
    )


def build_qk_pair_reference(
    queries: torch.Tensor,
    current_keys: torch.Tensor,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    q_by_head = queries.view(q_heads, head_dim)
    k_by_head = current_keys.view(kv_heads, head_dim)
    q_per_kv = q_heads // kv_heads
    if q_per_kv != 2:
        raise ValueError(f"expected q_per_kv=2 for qk_pair debug, got {q_per_kv}")
    pairs = torch.empty((kv_heads, 3, head_dim), dtype=queries.dtype)
    for kv_head in range(kv_heads):
        pairs[kv_head, 0, :] = q_by_head[kv_head * q_per_kv]
        pairs[kv_head, 1, :] = q_by_head[kv_head * q_per_kv + 1]
        pairs[kv_head, 2, :] = k_by_head[kv_head]
    return pairs.flatten().contiguous()


def print_structured_attention_error(
    name: str,
    errors: list[int],
    output: torch.Tensor,
    expected: torch.Tensor,
    op,
) -> None:
    if not errors:
        return
    first = int(errors[0])
    out_flat = output.flatten().to(torch.float32)
    exp_flat = expected.flatten().to(torch.float32)
    if name == "qk_pair":
        slot_names = ("q0", "q1", "current_k")
        elems_per_pair = 3 * op.head_dim
        kv_head = first // elems_per_pair
        rem = first % elems_per_pair
        slot = rem // op.head_dim
        dim = rem % op.head_dim
        print(
            "qk_pair_first_error: "
            f"kv_head={kv_head} slot={slot_names[slot]} dim={dim} "
            f"expected={float(exp_flat[first]):.6f} got={float(out_flat[first]):.6f}"
        )
    elif name in {"attn_scores", "attn_weights"}:
        q_head = first // op.max_seq_len
        pos = first % op.max_seq_len
        region = "valid" if pos <= op.position else "future"
        heads = sorted({int(idx) // op.max_seq_len for idx in errors})
        head_counts = {
            head: sum(1 for idx in errors if int(idx) // op.max_seq_len == head)
            for head in heads
        }
        counts = ", ".join(f"h{head}:{count}" for head, count in head_counts.items())
        print(
            f"{name}_first_error: "
            f"q_head={q_head} pos={pos} region={region} "
            f"expected={float(exp_flat[first]):.6f} got={float(out_flat[first]):.6f}"
        )
        print(f"{name}_error_heads: {counts}")
        if name == "attn_scores":
            got = out_flat[first]
            matches = (exp_flat == got).nonzero(as_tuple=False).flatten().tolist()
            formatted = []
            for idx in matches[:8]:
                match_q = int(idx) // op.max_seq_len
                match_pos = int(idx) % op.max_seq_len
                formatted.append(f"h{match_q}:p{match_pos}")
            if formatted:
                print(
                    f"attn_scores_first_got_matches_expected_at: {', '.join(formatted)}"
                )
            else:
                print("attn_scores_first_got_matches_expected_at: none")
    elif name in {"attn_context", "attn_context_flat"}:
        q_head = first // op.head_dim
        dim = first % op.head_dim
        print(
            f"{name}_first_error: "
            f"q_head={q_head} dim={dim} "
            f"expected={float(exp_flat[first]):.6f} got={float(out_flat[first]):.6f}"
        )
        got = out_flat[first]
        matches = (exp_flat == got).nonzero(as_tuple=False).flatten().tolist()
        formatted = []
        for idx in matches[:8]:
            match_q = int(idx) // op.head_dim
            match_dim = int(idx) % op.head_dim
            formatted.append(f"h{match_q}:d{match_dim}")
        if formatted:
            print(f"{name}_first_got_matches_expected_at: {', '.join(formatted)}")
        else:
            print(f"{name}_first_got_matches_expected_at: none")
    elif name in {"attn_o_proj", "attn_residual"}:
        dim = first % op.hidden_size
        print(
            f"{name}_first_error: "
            f"dim={dim} expected={float(exp_flat[first]):.6f} "
            f"got={float(out_flat[first]):.6f}"
        )
    elif name in {
        "mlp_x_norm",
        "ffn_gate",
        "ffn_up",
        "ffn_gate_silu",
        "ffn_hidden",
    }:
        dim = first % output.numel()
        print(
            f"{name}_first_error: "
            f"dim={dim} expected={float(exp_flat[first]):.6f} "
            f"got={float(out_flat[first]):.6f}"
        )


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
