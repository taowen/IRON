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


def verification_tolerance(stage: str, name: str) -> tuple[float, float]:
    if (
        stage in {"post-attn-rmsnorm-mlp-gate-up", "post-attn-rmsnorm-full-mlp"}
        and name == "ffn_gate_silu"
    ):
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
