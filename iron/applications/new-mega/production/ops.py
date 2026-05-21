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

PHASE_LABELS = (
    "input_qkv",
    "attention_chunk_0",
    "attention_chunk_1",
    "attention_chunk_2",
    "attention_chunk_3",
    "o_proj",
    "post_norm",
    "gate_up",
    "down_proj",
    "layer_residual",
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
    intermediate_size: int = 3072
    q_rows_per_packet: int = 4
    fabric_group_size: int = 4
    packet_elements: int = 15368
    context: AIEContext | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.num_lanes <= 0:
            raise ValueError("num_lanes must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")
        if self.phase_packets_per_layer <= 0:
            raise ValueError("phase_packets_per_layer must be positive")
        if self.phase_packets_per_layer < 2:
            raise ValueError(
                "phase_packets_per_layer must include a next_layer_token phase"
            )
        if self.phase_packets_per_layer < 9:
            raise ValueError("phase_packets_per_layer must include a gate_up phase")
        if self.phase_packets_per_layer < 10:
            raise ValueError("phase_packets_per_layer must include a down_proj phase")
        if self.packet_elements <= 0:
            raise ValueError("packet_elements must be positive")
        if self.packet_elements % 8 != 0:
            raise ValueError("packet_elements must be a multiple of 8 BF16 values")
        if self.hidden_size <= 0:
            raise ValueError("hidden_size must be positive")
        if self.attention_size <= 0:
            raise ValueError("attention_size must be positive")
        if self.intermediate_size <= 0:
            raise ValueError("intermediate_size must be positive")
        if self.q_rows_per_packet <= 0:
            raise ValueError("q_rows_per_packet must be positive")
        if self.fabric_group_size <= 0:
            raise ValueError("fabric_group_size must be positive")
        if self.num_lanes % self.fabric_group_size != 0:
            raise ValueError("num_lanes must be divisible by fabric_group_size")
        gate_up_elements = (2 + 2 * self.q_rows_per_packet) * self.hidden_size
        o_elements = (
            self.attention_size
            + self.q_rows_per_packet
            + self.q_rows_per_packet * self.attention_size
        )
        down_elements = (
            self.intermediate_size
            + self.q_rows_per_packet
            + self.q_rows_per_packet * self.intermediate_size
        )
        minimum_packet_elements = max(o_elements, gate_up_elements, down_elements)
        if self.packet_elements < minimum_packet_elements:
            raise ValueError(
                "packet_elements must hold o_proj, gate/up, and down phase payloads"
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
            f"_i{self.intermediate_size}"
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
    def q_output_values_per_lane(self) -> int:
        return ((self.q_rows_per_packet + 7) // 8) * 8

    @property
    def k_output_values_per_lane(self) -> int:
        return ((self.q_rows_per_packet + 7) // 8) * 8

    @property
    def v_output_values_per_lane(self) -> int:
        return ((self.q_rows_per_packet + 7) // 8) * 8

    @property
    def attention_output_values_per_lane(self) -> int:
        return ((self.q_rows_per_packet + 7) // 8) * 8

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
                    self.intermediate_size,
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
