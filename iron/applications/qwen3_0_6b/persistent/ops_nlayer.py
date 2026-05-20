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


@dataclass
class Qwen3PersistentNLayerFinalOnly(MLIROperator):
    """One or more sequential Qwen3 full layers using one persistent graph."""

    hidden_size: int = 1024
    q_size: int = 2048
    kv_size: int = 1024
    head_dim: int = 128
    max_seq_len: int = 256
    position: int = 0
    intermediate_size: int = 3072
    num_aie_columns: int = 1
    attention_columns: int = 1
    mlp_gate_up_columns: int = 0
    mlp_gate_up_pair_rows: bool = False
    mlp_gate_up_direct_silu: bool = False
    mlp_gate_up_row_group: int = 4
    attention_probe_only: bool = False
    tile_size_input: int = 4
    tile_size_output: int = 128
    epsilon: float = 1e-6
    kernel_vector_size: int = field(default=64, repr=False)
    layer_iterations: int = 1
    context: object = field(default=None, repr=False)

    max_supported_layer_iterations: ClassVar[int] = 28

    _name_aliases: ClassVar[dict[str, str]] = {
        **MLIROperator._name_aliases,
        "hidden_size": "h",
        "q_size": "q",
        "kv_size": "kv",
        "head_dim": "hd",
        "max_seq_len": "msl",
        "position": "pos",
        "intermediate_size": "ffn",
        "num_aie_columns": "col",
        "attention_columns": "attncol",
        "mlp_gate_up_columns": "mlpgatecol",
        "mlp_gate_up_pair_rows": "mlpgatepair",
        "mlp_gate_up_direct_silu": "mlpgatedirect",
        "mlp_gate_up_row_group": "mlpgaterg",
        "attention_probe_only": "attnprobe",
        "tile_size_input": "tsi",
        "tile_size_output": "tso",
        "layer_iterations": "layers",
    }

    def __post_init__(self):
        if self.hidden_size != 1024:
            raise ValueError(
                "Qwen3-0.6B n-layer final-only expects hidden_size=1024, "
                f"got {self.hidden_size}"
            )
        if self.q_size != 2048 or self.kv_size != 1024:
            raise ValueError(
                "Qwen3-0.6B n-layer final-only expects q_size=2048 and "
                f"kv_size=1024, got q_size={self.q_size} kv_size={self.kv_size}"
            )
        if self.head_dim != 128:
            raise ValueError(
                "Qwen3-0.6B n-layer final-only expects head_dim=128, "
                f"got {self.head_dim}"
            )
        if self.intermediate_size != 3072:
            raise ValueError(
                "Qwen3-0.6B n-layer final-only expects intermediate_size=3072, "
                f"got {self.intermediate_size}"
            )
        if self.num_aie_columns < 1:
            raise ValueError("num_aie_columns must be positive")
        if self.num_aie_columns not in {1, 2, 4}:
            raise ValueError("num_aie_columns must be one of 1, 2, or 4")
        if self.attention_columns not in {1, 2}:
            raise ValueError("attention_columns must be one of 1 or 2")
        if self.attention_columns == 2 and self.num_aie_columns not in {1, 2}:
            raise ValueError(
                "attention_columns=2 currently supports num_aie_columns=1 for "
                "the attention probe or num_aie_columns=2 for the packed MLP2 "
                "path"
            )
        if self.mlp_gate_up_columns not in {0, 1, 2, 3}:
            raise ValueError("mlp_gate_up_columns must be one of 0, 1, 2, or 3")
        if self.effective_mlp_gate_up_columns in {2, 3} and self.num_aie_columns != 2:
            raise ValueError(
                "mlp_gate_up_columns=2 or 3 currently requires num_aie_columns=2"
            )
        if self.mlp_gate_up_pair_rows:
            if self.effective_mlp_gate_up_columns not in {2, 3}:
                raise ValueError(
                    "mlp_gate_up_pair_rows currently requires "
                    "mlp_gate_up_columns=2 or 3"
                )
            if self.tile_size_input != 4:
                raise ValueError("mlp_gate_up_pair_rows requires tile_size_input=4")
        if self.mlp_gate_up_direct_silu and not self.mlp_gate_up_pair_rows:
            raise ValueError("mlp_gate_up_direct_silu requires mlp_gate_up_pair_rows")
        if self.mlp_gate_up_row_group not in {4, 8}:
            raise ValueError("mlp_gate_up_row_group must be one of 4 or 8")
        if self.mlp_gate_up_row_group != 4 and not (
            self.mlp_gate_up_pair_rows and self.mlp_gate_up_direct_silu
        ):
            raise ValueError(
                "mlp_gate_up_row_group=8 currently requires "
                "mlp_gate_up_pair_rows and mlp_gate_up_direct_silu"
            )
        if self.mlp_gate_up_row_group != 4 and self.effective_mlp_gate_up_columns != 2:
            raise ValueError(
                "mlp_gate_up_row_group=8 currently supports only "
                "mlp_gate_up_columns=2"
            )
        if (
            self.intermediate_size
            % (self.effective_mlp_gate_up_columns * self.mlp_gate_up_row_group)
            != 0
        ):
            raise ValueError(
                "intermediate_size per gate/up column must be divisible by "
                "mlp_gate_up_row_group"
            )
        if self.num_aie_columns > 2 and self.layer_iterations != 1:
            raise ValueError(
                "n-layer final-only column scaling above 2 columns is currently "
                "limited to layer_iterations=1 while multi-stage joins are developed"
            )
        if self.hidden_size % (self.tile_size_output * self.num_aie_columns) != 0:
            raise ValueError(
                "hidden_size must be divisible by tile_size_output * num_aie_columns"
            )
        if self.q_size % self.head_dim != 0 or self.kv_size % self.head_dim != 0:
            raise ValueError("q_size and kv_size must be divisible by head_dim")
        if self.tile_size_output != self.head_dim:
            raise ValueError(
                "tile_size_output must equal head_dim for n-layer final-only"
            )
        if self.max_seq_len < 256 or self.max_seq_len % 64 != 0:
            raise ValueError("max_seq_len must be at least 256 and a multiple of 64")
        if not (0 <= self.position < self.max_seq_len):
            raise ValueError(
                f"position must be in [0, {self.max_seq_len}), got {self.position}"
            )
        if self.hidden_size % self.kernel_vector_size != 0:
            raise ValueError("hidden_size must be a multiple of kernel_vector_size")
        if self.q_size % self.kernel_vector_size != 0:
            raise ValueError("q_size must be a multiple of kernel_vector_size")
        if self.intermediate_size % self.kernel_vector_size != 0:
            raise ValueError(
                "intermediate_size must be a multiple of kernel_vector_size"
            )
        if self.tile_size_output % self.tile_size_input != 0:
            raise ValueError("tile_size_output must be a multiple of tile_size_input")
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")
        if self.layer_iterations < 1:
            raise ValueError("layer_iterations must be positive")
        if self.attention_probe_only and self.layer_iterations != 1:
            raise ValueError(
                "attention_probe_only is a single-layer boundary probe; "
                f"got layer_iterations={self.layer_iterations}"
            )
        if self.layer_iterations > self.max_supported_layer_iterations:
            raise ValueError(
                "Qwen3 n-layer final-only currently supports at most "
                f"{self.max_supported_layer_iterations} layers per chunk; "
                f"got {self.layer_iterations}. Larger chunks need a runtime "
                "state-machine design that reuses cache DMA descriptors instead "
                "of statically issuing more layer groups."
            )
        MLIROperator.__init__(self, context=self.context)

    @property
    def q_heads(self):
        return self.q_size // self.head_dim

    @property
    def kv_heads(self):
        return self.kv_size // self.head_dim

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
    def _rope_kernel_object(self):
        return "qwen3_persistent_rope_0.o"

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
    def _o_gemv_kernel_object(self):
        return (
            f"qwen3_persistent_gemv_{self.o_gemv_k_size}k_"
            f"{self.kernel_vector_size}vs_o_proj.o"
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
    def cache_half_size(self):
        return self.kv_size * self.max_seq_len

    @property
    def packed_cache_size(self):
        return 2 * self.cache_half_size

    @property
    def packed_cache_chunk_size(self):
        return self.layer_iterations * self.packed_cache_size

    @property
    def effective_mlp_gate_up_columns(self):
        if self.mlp_gate_up_columns != 0:
            return self.mlp_gate_up_columns
        return 2 if self.num_aie_columns == 2 else 1

    @property
    def qk_rope_metadata_size(self):
        if self.attention_columns == 2:
            return 3 * self.head_dim
        return self.head_dim

    @property
    def rope_runtime_size(self):
        if self.attention_columns == 2:
            return self.layer_iterations * self.qk_rope_metadata_size
        return self.head_dim

    @property
    def runtime_hidden_size(self):
        if (
            self.attention_columns == 2
            and self.num_aie_columns == 2
            and self.effective_mlp_gate_up_columns in {2, 3}
        ):
            return self.layer_iterations * (
                self.hidden_size + self.qk_rope_metadata_size
            )
        return self.hidden_size

    @property
    def o_gemv_k_size(self):
        return self.q_size // self.attention_columns

    @property
    def packed_weights_size(self):
        return (
            self.hidden_size
            + self.q_size * self.hidden_size
            + 2 * self.kv_size * self.hidden_size
            + 2 * self.head_dim
            + self.hidden_size * self.q_size
            + self.hidden_size
            + 3 * self.intermediate_size * self.hidden_size
        )

    @property
    def packed_weight_chunk_size(self):
        return self.layer_iterations * self.packed_weights_size

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
                    "attention_columns": self.attention_columns,
                    "mlp_gate_up_columns": self.effective_mlp_gate_up_columns,
                    "mlp_gate_up_pair_rows": self.mlp_gate_up_pair_rows,
                    "mlp_gate_up_direct_silu": self.mlp_gate_up_direct_silu,
                    "mlp_gate_up_row_group": self.mlp_gate_up_row_group,
                    "attention_probe_only": self.attention_probe_only,
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
        if arch_dir != "aie2p":
            raise ValueError(
                "n-layer final-only persistent path currently targets NPU2/aie2p"
            )
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
                self._rope_kernel_object,
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / "generic" / "rope.cc"
                    )
                ],
                extra_flags=["-DTWO_HALVES"],
            ),
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
                        self.context.base_dir / "aie_kernels" / arch_dir / "softmax.cc"
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
            KernelObjectArtifact(
                self._o_gemv_kernel_object,
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / "generic" / "mv.cc"
                    )
                ],
                extra_flags=[
                    f"-DDIM_K={self.o_gemv_k_size}",
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

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (self.runtime_hidden_size,)),
            AIERuntimeArgSpec("in", (self.packed_weight_chunk_size,)),
            AIERuntimeArgSpec("in", (self.rope_runtime_size,)),
            AIERuntimeArgSpec("out", (self.packed_outputs_size,)),
            AIERuntimeArgSpec("inout", (self.packed_cache_chunk_size,)),
        ]
