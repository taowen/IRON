#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
        if self.num_aie_columns < 1:
            raise ValueError("num_aie_columns must be positive")
        if self.hidden_size % self.kernel_vector_size != 0:
            raise ValueError("hidden_size must be a multiple of kernel_vector_size")
        if self.intermediate_size % self.kernel_vector_size != 0:
            raise ValueError(
                "intermediate_size must be a multiple of kernel_vector_size"
            )
        if self.hidden_size % self.tile_size_output != 0:
            raise ValueError("hidden_size must be divisible by tile_size_output")
        if self.hidden_size % (self.tile_size_output * self.num_aie_columns) != 0:
            raise ValueError(
                "hidden_size must be divisible by tile_size_output * num_aie_columns"
            )
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
