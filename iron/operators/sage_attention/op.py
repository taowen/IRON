# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import ClassVar, Dict

import numpy as np
from ml_dtypes import bfloat16

from iron.common import (
    MLIROperator,
    AIERuntimeArgSpec,
    KernelObjectArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
    DesignGenerator,
)
import aie.utils as aie_utils


@dataclass
class SageAttention(MLIROperator):
    """First SageAttention-B bring-up: int8 QK, bf16 online softmax/PV."""

    seq_len: int
    d: int
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "seq_len": "s",
    }

    def __post_init__(self):
        self.B_q = 64
        self.B_kv = 64
        self.num_pipelines = 2 if self.seq_len >= 128 else 1
        if self.d != 64:
            raise ValueError(f"Only d=64 is supported in this bring-up, got {self.d}")
        MLIROperator.__init__(self, context=self.context)

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "sage_attention",
                (),
                {
                    "dev": aie_utils.DefaultNPURuntime.device(),
                    "S_q": self.seq_len,
                    "S_kv": self.seq_len,
                    "d": self.d,
                    "B_q": self.B_q,
                    "B_kv": self.B_kv,
                    "num_pipelines": self.num_pipelines,
                    "trace_size": 0,
                    "verbose": False,
                },
            ),
        )

    def get_kernel_artifacts(self):
        sage_source = str(
            self.context.base_dir / "aie_kernels" / "aie2p" / "sage_attention.cc"
        )
        mha_source = str(self.context.base_dir / "aie_kernels" / "aie2p" / "mha.cc")
        mm_source = str(self.context.base_dir / "aie_kernels" / "aie2p" / "mm.cc")
        softmax_source = str(
            self.context.base_dir / "aie_kernels" / "aie2p" / "softmax.cc"
        )
        passthrough_source = str(
            self.context.base_dir / "aie_kernels" / "generic" / "passThrough.cc"
        )

        return [
            KernelObjectArtifact(
                "sage_attention.o",
                extra_flags=[
                    "-Dbf16_bf16_ONLY",
                    f"-DDIM_M={self.B_q}",
                    f"-DDIM_K={self.d}",
                    f"-DDIM_N={self.B_kv}",
                    "-DROUND_CONV_EVEN",
                    "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
                    "-DB_COL_MAJ",
                ],
                dependencies=[
                    SourceArtifact(sage_source),
                    SourceArtifact(mha_source),
                    SourceArtifact(mm_source),
                    SourceArtifact(softmax_source),
                ],
            ),
            KernelObjectArtifact(
                "sage_attention_passThrough.o",
                extra_flags=["-DBIT_WIDTH=16"],
                dependencies=[SourceArtifact(passthrough_source)],
            ),
        ]

    def get_artifacts(self):
        return super().get_artifacts(dynamic_obj_fifos=True)

    def get_arg_spec(self):
        q_group = self.B_q * self.num_pipelines
        seq_padding = ((self.seq_len + q_group - 1) // q_group) * q_group
        buffer_size = self.d * seq_padding
        num_blocks = seq_padding // 64
        return [
            AIERuntimeArgSpec("in", (buffer_size,), dtype=np.dtype(np.int8)),
            AIERuntimeArgSpec("in", (buffer_size,), dtype=np.dtype(np.int8)),
            AIERuntimeArgSpec("in", (buffer_size,), dtype=bfloat16),
            AIERuntimeArgSpec(
                "in", (num_blocks, num_blocks), dtype=np.dtype(np.float32)
            ),
            AIERuntimeArgSpec("out", (buffer_size,), dtype=bfloat16),
        ]
