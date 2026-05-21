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

ATTENTION_SCORE_PV_PHASES = (
    "attention_score_pv_0",
    "attention_score_pv_1",
    "attention_score_pv_2",
    "attention_score_pv_3",
)

ATTENTION_SCORE_PV_SECONDARY_PHASES = (
    "attention_score_pv_4",
    "attention_score_pv_5",
    "attention_score_pv_6",
    "attention_score_pv_7",
)

O_PROJECTION_CHUNK_ROWS = 32
O_PROJECTION_CHUNK_COUNT = 32
O_PROJECTION_PHASES = tuple(
    f"o_proj_chunk_{chunk}" for chunk in range(O_PROJECTION_CHUNK_COUNT)
)
FFN_REDUCE_CHUNK_ROWS = O_PROJECTION_CHUNK_ROWS
FFN_REDUCE_GROUP_COUNT = 8
FFN_REDUCE_PHASES = tuple(
    phase
    for chunk in range(FFN_REDUCE_GROUP_COUNT)
    for phase in (f"ffn_gate_chunk_{chunk}", f"down_partial_chunk_{chunk}")
)
FFN_GATE_PHASES = FFN_REDUCE_PHASES[0::2]
FFN_DOWN_PHASES = FFN_REDUCE_PHASES[1::2]

PHASE_LABELS = (
    "input_qkv",
    "attention_chunk_0",
    "attention_chunk_1",
    "attention_chunk_2",
    "attention_chunk_3",
    *ATTENTION_SCORE_PV_PHASES,
    *ATTENTION_SCORE_PV_SECONDARY_PHASES,
    *O_PROJECTION_PHASES,
    *FFN_REDUCE_PHASES,
    "next_layer_token",
)


@dataclass
class NewMegaPhaseOwnedDecode(MLIROperator):
    """Production direction: fixed lane workers consume packed phase streams.

    This class models the reusable phase-owned topology rather than a
    statically expanded single layer. Real Qwen3 kernels are added inside this
    topology, not as separate static stage workers.
    """

    num_lanes: int = 8
    num_layers: int = 28
    phase_packets_per_layer: int = len(PHASE_LABELS)
    hidden_size: int = 1024
    attention_size: int = 2048
    head_dim: int = 128
    intermediate_size: int = 3072
    max_seq_len: int = 256
    attention_chunk_size: int = 64
    q_rows_per_packet: int = 4
    fabric_group_size: int = 4
    packet_elements: int = 16576
    context: AIEContext | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.num_lanes <= 0:
            raise ValueError("num_lanes must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.phase_packets_per_layer <= 0:
            raise ValueError("phase_packets_per_layer must be positive")
        if self.phase_packets_per_layer != len(PHASE_LABELS):
            raise ValueError(
                f"phase_packets_per_layer must be {len(PHASE_LABELS)} for the "
                "production phase-owned body"
            )
        if self.packet_elements <= 0:
            raise ValueError("packet_elements must be positive")
        if self.packet_elements % 8 != 0:
            raise ValueError("packet_elements must be a multiple of 8 BF16 values")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.attention_size <= 0:
            raise ValueError("attention_size must be positive")
        if self.head_dim <= 0:
            raise ValueError("head_dim must be positive")
        if self.attention_size % self.head_dim != 0:
            raise ValueError("attention_size must be divisible by head_dim")
        if self.num_lanes > self.attention_head_count:
            raise ValueError(
                "production currently maps one attention context head per lane"
            )
        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if self.attention_chunk_size <= 0:
            raise ValueError("attention_chunk_size must be positive")
        if self.max_seq_len % self.attention_chunk_size != 0:
            raise ValueError("max_seq_len must be divisible by attention_chunk_size")
        if self.attention_chunk_count != len(ATTENTION_SCORE_PV_PHASES):
            raise ValueError(
                "production currently expects exactly four attention score/PV chunks"
            )
        if self.head_dim != 128:
            raise ValueError(
                "production attention score/PV kernel currently expects head_dim=128"
            )
        if self.q_rows_per_packet <= 0:
            raise ValueError("q_rows_per_packet must be positive")
        if self.q_rows_per_packet > 16:
            raise ValueError("q_rows_per_packet must be <= 16 for local FFN handoff")
        if self.num_lanes * self.q_rows_per_packet > self.head_dim:
            raise ValueError("current q/k norm+RoPE shard supports only the first head")
        if self.fabric_group_size <= 0:
            raise ValueError("fabric_group_size must be positive")
        if self.num_lanes % self.fabric_group_size != 0:
            raise ValueError("num_lanes must be divisible by fabric_group_size")
        if self.num_lanes // self.fabric_group_size != 2:
            raise ValueError(
                "production O cross-group partial reduce currently expects two "
                "fabric groups"
            )
        if self.o_projection_chunk_rows != O_PROJECTION_CHUNK_ROWS:
            raise ValueError(
                "production O projection phase labels currently expect 32 rows per chunk"
            )
        if self.o_projection_chunk_count != O_PROJECTION_CHUNK_COUNT:
            raise ValueError(
                "production O projection phase labels currently expect 32 chunks"
            )
        if self.ffn_reduce_chunk_rows != FFN_REDUCE_CHUNK_ROWS:
            raise ValueError(
                "production FFN reduce phase labels currently expect 32 rows per chunk"
            )
        q_phase_elements = (2 + self.q_rows_per_packet) * self.hidden_size
        gate_up_elements = 1 + (2 + 2 * self.q_rows_per_packet) * self.hidden_size
        o_elements = (
            2
            + self.o_projection_chunk_rows
            + self.o_projection_chunk_rows * self.context_output_values_per_lane
        )
        down_elements = (
            1
            + self.intermediate_size
            + self.q_rows_per_packet
            + self.q_rows_per_packet * self.intermediate_size
        )
        norm_rope_elements = 1 + 4 * self.head_dim
        attention_chunk_elements = (
            self.head_dim
            + 2 * self.attention_chunk_size * self.head_dim
            + self.attention_chunk_size
        )
        minimum_packet_elements = max(
            o_elements,
            q_phase_elements,
            gate_up_elements,
            down_elements,
            norm_rope_elements,
            attention_chunk_elements,
        )
        if self.packet_elements < minimum_packet_elements:
            raise ValueError(
                "packet_elements must hold o_proj, gate/up, down, q/k RoPE, "
                "and attention score/PV phase payloads"
            )
        super().__init__(context=self.context)

    @property
    def name(self) -> str:
        dev = aie_utils.get_current_device().resolve().name
        return (
            "NewMegaPhaseOwnedDecode"
            f"_l{self.num_lanes}"
            f"_d{self.num_layers}"
            f"_p{self.phase_packets_per_layer}"
            f"_e{self.packet_elements}"
            f"_h{self.hidden_size}"
            f"_a{self.attention_size}"
            f"_hd{self.head_dim}"
            f"_i{self.intermediate_size}"
            f"_m{self.max_seq_len}"
            f"_c{self.attention_chunk_size}"
            f"_q{self.q_rows_per_packet}"
            f"_g{self.fabric_group_size}"
            f"_{dev}"
        )

    @property
    def total_phase_packets(self) -> int:
        return self.num_layers * self.phase_packets_per_layer

    @property
    def input_elements(self) -> int:
        return self.num_lanes * self.total_phase_packets * self.packet_elements

    @property
    def shared_packet_elements(self) -> int:
        return 2 * self.hidden_size

    @property
    def shared_input_elements(self) -> int:
        return self.num_layers * self.shared_packet_elements

    @property
    def output_elements(self) -> int:
        return self.num_layers * self.output_values_per_layer

    @property
    def attention_chunk_count(self) -> int:
        return self.max_seq_len // self.attention_chunk_size

    @property
    def o_projection_chunk_rows(self) -> int:
        return self.num_lanes * self.q_rows_per_packet

    @property
    def o_projection_chunk_count(self) -> int:
        if self.hidden_size % self.o_projection_chunk_rows != 0:
            raise ValueError("hidden_size must be divisible by O projection chunk rows")
        return self.hidden_size // self.o_projection_chunk_rows

    @property
    def ffn_reduce_chunk_rows(self) -> int:
        return self.num_lanes * self.q_rows_per_packet

    @property
    def ffn_reduce_group_count(self) -> int:
        return FFN_REDUCE_GROUP_COUNT

    @property
    def ffn_npu_rows(self) -> int:
        return self.ffn_reduce_chunk_rows * self.ffn_reduce_group_count

    @property
    def attention_head_count(self) -> int:
        return self.attention_size // self.head_dim

    @property
    def q_output_values_per_lane(self) -> int:
        return ((self.q_rows_per_packet + 7) // 8) * 8

    @property
    def k_output_values_per_lane(self) -> int:
        return ((self.q_rows_per_packet + 7) // 8) * 8

    @property
    def v_output_values_per_lane(self) -> int:
        return ((self.q_rows_per_packet + 7) // 8) * 8

    @property
    def q_rope_output_values_per_lane(self) -> int:
        return ((self.q_rows_per_packet + 7) // 8) * 8

    @property
    def k_rope_output_values_per_lane(self) -> int:
        return ((self.q_rows_per_packet + 7) // 8) * 8

    @property
    def context_output_values_per_lane(self) -> int:
        return 2 * self.head_dim

    @property
    def attention_output_values_per_lane(self) -> int:
        return ((self.hidden_size + 7) // 8) * 8

    @property
    def gate_up_output_values_per_lane(self) -> int:
        return ((2 * self.q_rows_per_packet + 7) // 8) * 8

    @property
    def residual_output_values_per_lane(self) -> int:
        return ((self.q_rows_per_packet + 7) // 8) * 8

    @property
    def output_values_per_lane(self) -> int:
        return (
            self.q_output_values_per_lane
            + self.k_output_values_per_lane
            + self.v_output_values_per_lane
            + self.q_rope_output_values_per_lane
            + self.k_rope_output_values_per_lane
            + self.context_output_values_per_lane
            + self.attention_output_values_per_lane
            + self.gate_up_output_values_per_lane
            + self.residual_output_values_per_lane
        )

    @property
    def output_values_per_layer(self) -> int:
        return self.num_lanes * self.output_values_per_lane

    @property
    def packet_bytes(self) -> int:
        return self.packet_elements * 2

    @property
    def lane_stream_bytes(self) -> int:
        return self.total_phase_packets * self.packet_bytes

    def get_mlir_artifact(self) -> PythonGeneratedMLIRArtifact:
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "phase_owned_decode",
                (
                    aie_utils.get_current_device(),
                    self.num_lanes,
                    self.num_layers,
                    self.phase_packets_per_layer,
                    self.packet_elements,
                    self.hidden_size,
                    self.attention_size,
                    self.head_dim,
                    self.intermediate_size,
                    self.max_seq_len,
                    self.attention_chunk_size,
                    self.q_rows_per_packet,
                    self.fabric_group_size,
                ),
            ),
        )

    def get_kernel_artifacts(self) -> list[KernelObjectArtifact]:
        return [
            KernelObjectArtifact(
                "phase_owned_kernels.o",
                dependencies=[
                    SourceArtifact(self.operator_dir / "phase_owned_kernels.cc")
                ],
            )
        ]

    def get_arg_spec(self) -> list[AIERuntimeArgSpec]:
        return [
            AIERuntimeArgSpec("in", (self.shared_input_elements,)),
            AIERuntimeArgSpec("in", (self.input_elements,)),
            AIERuntimeArgSpec("out", (self.output_elements,)),
        ]
