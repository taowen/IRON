"""Executable physical channel ownership checks for qwen3-layer MLIR-AIE."""

from __future__ import annotations

import re

from contract import MAIN_COLUMNS, ROWS_PER_COLUMN

ROW1_COMPACT_S2MM = (0, 1, 2, 3)
ROW1_WEIGHT_S2MM = (4, 5)
ROW1_WEIGHT_MM2S = (0, 1, 2, 3)
MAIN_ACTIVATION_S2MM = 0
MAIN_WEIGHT_S2MM = 1
MAIN_RECORD_MM2S = 1
ROW1_COMPACT_MM2S = 4
C1R1_COMPACT_MM2S = 0
C1R1_PACKET_S2MM = 4
C1R1_PACKET_MM2S = 1


def _require_marker(scope: str, name: str, mlir: str, marker: str) -> list[str]:
    if marker in mlir:
        return []
    return [f"{scope}: missing physical marker for {name}: {marker}"]


def _require_marker_count(
    scope: str,
    name: str,
    mlir: str,
    marker: str,
    expected: int,
) -> list[str]:
    actual = mlir.count(marker)
    if actual == expected:
        return []
    return [f"{scope}: {name} marker count {actual} != {expected}: {marker}"]


def _require_any_marker_count(
    scope: str,
    name: str,
    mlir: str,
    markers: tuple[str, ...],
    expected: int,
) -> list[str]:
    actual = sum(mlir.count(marker) for marker in markers)
    if actual == expected:
        return []
    formatted = " | ".join(markers)
    return [f"{scope}: {name} marker count {actual} != {expected}: {formatted}"]


def _require_no_marker(scope: str, name: str, mlir: str, marker: str) -> list[str]:
    if marker not in mlir:
        return []
    return [f"{scope}: forbidden physical marker for {name}: {marker}"]


def _dma_start_channels(mlir: str, direction: str, label_pattern: str) -> tuple[int, ...]:
    pattern = rf"aie\.dma_start\({direction},\s*([0-9]+),\s*\^({label_pattern})"
    return tuple(int(match.group(1)) for match in re.finditer(pattern, mlir))


def validate_q4nx_down_dual_input_abi(scope: str, mlir: str) -> list[str]:
    errors: list[str] = []
    main_tiles = len(MAIN_COLUMNS) * ROWS_PER_COLUMN
    groups = len(MAIN_COLUMNS)
    errors.extend(
        _require_marker_count(
            scope,
            "main16 activation input DMA0",
            mlir,
            f"aie.dma_start(S2MM, {MAIN_ACTIVATION_S2MM}, ^act_ping, ^wt_start)",
            main_tiles,
        )
    )
    errors.extend(
        _require_marker_count(
            scope,
            "main16 Q4NX weight input DMA1",
            mlir,
            f"aie.dma_start(S2MM, {MAIN_WEIGHT_S2MM}, ^wt_ping, ^output_start)",
            main_tiles,
        )
    )
    errors.extend(
        _require_marker_count(
            scope,
            "main16 record/output stream",
            mlir,
            f"aie.dma_start(MM2S, {MAIN_RECORD_MM2S}, ^output_out, ^end)",
            main_tiles,
        )
    )
    errors.extend(
        _require_marker_count(
            scope,
            "row1 patch0 weight ingress S2MM4",
            mlir,
            f"aie.dma_start(S2MM, {ROW1_WEIGHT_S2MM[0]}, ^patch0_ping, ^patch1_start)",
            groups,
        )
    )
    errors.extend(
        _require_marker_count(
            scope,
            "row1 patch1 weight ingress S2MM5",
            mlir,
            f"aie.dma_start(S2MM, {ROW1_WEIGHT_S2MM[1]}, ^patch1_ping, ^row0_start)",
            groups,
        )
    )
    for row, channel in enumerate(ROW1_WEIGHT_MM2S):
        errors.extend(
            _require_marker_count(
                scope,
                f"row1 weight fanout MM2S{channel}",
                mlir,
                f"aie.dma_start(MM2S, {channel}, ^row{row}_ping",
                groups,
            )
        )
    for group in range(groups):
        errors.extend(
            _require_marker(
                scope,
                f"group{group} host patch0 to row1 S2MM4",
                mlir,
                f"aie.flow(%shim{group}, DMA : 0, %mt{group}, DMA : {ROW1_WEIGHT_S2MM[0]})",
            )
        )
        errors.extend(
            _require_marker(
                scope,
                f"group{group} host patch1 to row1 S2MM5",
                mlir,
                f"aie.flow(%shim{group}, DMA : 1, %mt{group}, DMA : {ROW1_WEIGHT_S2MM[1]})",
            )
        )
        errors.extend(
            _require_no_marker(
                scope,
                f"group{group} old patch0 ingress on compact S2MM0",
                mlir,
                f"aie.flow(%shim{group}, DMA : 0, %mt{group}, DMA : 0)",
            )
        )
        errors.extend(
            _require_no_marker(
                scope,
                f"group{group} old patch1 ingress on compact S2MM1",
                mlir,
                f"aie.flow(%shim{group}, DMA : 1, %mt{group}, DMA : 1)",
            )
        )
        for row in range(ROWS_PER_COLUMN):
            errors.extend(
                _require_marker(
                    scope,
                    f"group{group} row{row} weight to main16 DMA1",
                    mlir,
                    f"aie.flow(%mt{group}, DMA : {row}, %m{group}_{row}, DMA : {MAIN_WEIGHT_S2MM})",
                )
            )
    patch_channels = _dma_start_channels(mlir, "S2MM", r"patch[01]_ping")
    if sorted(set(patch_channels)) != list(ROW1_WEIGHT_S2MM):
        errors.append(
            f"{scope}: row1 weight patch ingress channels are {patch_channels}, "
            f"expected only {ROW1_WEIGHT_S2MM}"
        )
    if re.search(r"aie\.dma_start\(S2MM,\s*[01],\s*\^patch[01]_ping", mlir):
        errors.append(f"{scope}: row1 Q4NX patch ingress uses compact-owned S2MM0/1")
    if "func.call @q4nx_chunk_accum_slice" not in mlir:
        errors.append(f"{scope}: missing Q4NX chunk kernel call")
    return errors


def validate_compact_only_full_layer_ownership(scope: str, mlir: str) -> list[str]:
    errors: list[str] = []
    main_tiles = len(MAIN_COLUMNS) * ROWS_PER_COLUMN
    groups = len(MAIN_COLUMNS)
    for row in ROW1_COMPACT_S2MM:
        errors.extend(
            _require_marker_count(
                scope,
                f"row1 compact gather S2MM{row}",
                mlir,
                f"aie.dma_start(S2MM, {row}, ^compact_row{row}_ping",
                groups,
            )
        )
    errors.extend(
        _require_marker_count(
            scope,
            "main16 activation/replay input DMA0",
            mlir,
            f"aie.dma_start(S2MM, {MAIN_ACTIVATION_S2MM}, ^chunk_ping, ^record_start)",
            main_tiles,
        )
    )
    errors.extend(
        _require_marker_count(
            scope,
            "main16 compact record output",
            mlir,
            f"aie.dma_start(MM2S, {MAIN_RECORD_MM2S}, ^record_ping, ^end)",
            main_tiles,
        )
    )
    errors.extend(
        _require_marker_count(
            scope,
            "row1 compact output MM2S4",
            mlir,
            f"aie.dma_start(MM2S, {ROW1_COMPACT_MM2S}, ^compact_ping_out",
            groups,
        )
    )
    errors.extend(
        _require_marker_count(
            scope,
            "c1r1 compact output MM2S0",
            mlir,
            f"aie.dma_start(MM2S, {C1R1_COMPACT_MM2S}, ^compact_ping_out",
            1,
        )
    )
    errors.extend(
        _require_any_marker_count(
            scope,
            "c1r1 shared activation bridge ingress DMA4",
            mlir,
            (
                f"aie.dma_start(S2MM, {C1R1_PACKET_S2MM}, ^packet_in_ping, ^packet_out_start)",
                f"aie.dma_start(S2MM, {C1R1_PACKET_S2MM}, ^attention_in, ^packet_out_start)",
            ),
            1,
        )
    )
    errors.extend(
        _require_any_marker_count(
            scope,
            "c1r1 shared activation bridge egress DMA1",
            mlir,
            (
                f"aie.dma_start(MM2S, {C1R1_PACKET_MM2S}, ^packet_out_ping, ^end)",
                f"aie.dma_start(MM2S, {C1R1_PACKET_MM2S}, ^attention_out, ^end)",
            ),
            1,
        )
    )
    for channel in ROW1_WEIGHT_S2MM:
        errors.extend(
            _require_no_marker(
                scope,
                f"compact-only full layer row1 weight ingress S2MM{channel}",
                mlir,
                f"aie.dma_start(S2MM, {channel}, ^patch",
            )
        )
    if re.search(r"aie\.flow\(%shim[0-9]+,\s*DMA\s*:\s*[01],\s*%mt[0-9]+,\s*DMA\s*:\s*[01]\)", mlir):
        errors.append(
            f"{scope}: compact-only full layer contains old shim-to-row1 low-channel weight route"
        )
    return errors


def validate_q4nx_down_full_layer_ownership(scope: str, mlir: str) -> list[str]:
    errors: list[str] = []
    main_tiles = len(MAIN_COLUMNS) * ROWS_PER_COLUMN
    groups = len(MAIN_COLUMNS)
    for row in ROW1_COMPACT_S2MM:
        errors.extend(
            _require_marker_count(
                scope,
                f"row1 compact gather S2MM{row}",
                mlir,
                f"aie.dma_start(S2MM, {row}, ^compact_row{row}_ping",
                groups,
            )
        )
    errors.extend(
        _require_marker_count(
            scope,
            "main16 activation/replay input DMA0",
            mlir,
            f"aie.dma_start(S2MM, {MAIN_ACTIVATION_S2MM}, ^chunk_ping, ^wt_start)",
            main_tiles,
        )
    )
    errors.extend(
        _require_marker_count(
            scope,
            "main16 Q4NX weight input DMA1",
            mlir,
            f"aie.dma_start(S2MM, {MAIN_WEIGHT_S2MM}, ^wt_ping, ^record_start)",
            main_tiles,
        )
    )
    errors.extend(
        _require_marker_count(
            scope,
            "main16 compact record output",
            mlir,
            f"aie.dma_start(MM2S, {MAIN_RECORD_MM2S}, ^record_ping, ^end)",
            main_tiles,
        )
    )
    errors.extend(
        _require_marker_count(
            scope,
            "row1 patch0 weight ingress S2MM4",
            mlir,
            f"aie.dma_start(S2MM, {ROW1_WEIGHT_S2MM[0]}, ^patch0_q4nx_ping, ^patch1_start)",
            groups,
        )
    )
    errors.extend(
        _require_marker_count(
            scope,
            "row1 patch1 weight ingress S2MM5",
            mlir,
            f"aie.dma_start(S2MM, {ROW1_WEIGHT_S2MM[1]}, ^patch1_q4nx_ping, ^wt_row0_start)",
            groups,
        )
    )
    for row, channel in enumerate(ROW1_WEIGHT_MM2S):
        errors.extend(
            _require_marker_count(
                scope,
                f"row1 weight fanout MM2S{channel}",
                mlir,
                f"aie.dma_start(MM2S, {channel}, ^row{row}_ping",
                groups,
            )
        )
    errors.extend(
        _require_marker_count(
            scope,
            "row1 compact output MM2S4",
            mlir,
            f"aie.dma_start(MM2S, {ROW1_COMPACT_MM2S}, ^compact_ping_out",
            groups,
        )
    )
    errors.extend(
        _require_marker_count(
            scope,
            "c1r1 compact output MM2S0",
            mlir,
            f"aie.dma_start(MM2S, {C1R1_COMPACT_MM2S}, ^compact_ping_out",
            1,
        )
    )
    errors.extend(
        _require_any_marker_count(
            scope,
            "c1r1 shared activation bridge ingress DMA4",
            mlir,
            (
                f"aie.dma_start(S2MM, {C1R1_PACKET_S2MM}, ^packet_in_ping, ^packet_out_start)",
                f"aie.dma_start(S2MM, {C1R1_PACKET_S2MM}, ^attention_in, ^packet_out_start)",
            ),
            1,
        )
    )
    errors.extend(
        _require_any_marker_count(
            scope,
            "c1r1 shared activation bridge egress DMA1",
            mlir,
            (
                f"aie.dma_start(MM2S, {C1R1_PACKET_MM2S}, ^packet_out_ping, ^end)",
                f"aie.dma_start(MM2S, {C1R1_PACKET_MM2S}, ^attention_out, ^end)",
            ),
            1,
        )
    )
    for group in range(groups):
        errors.extend(
            _require_marker(
                scope,
                f"group{group} host patch0 to row1 S2MM4",
                mlir,
                f"aie.flow(%shim{group}, DMA : 0, %mt{group}, DMA : {ROW1_WEIGHT_S2MM[0]})",
            )
        )
        errors.extend(
            _require_marker(
                scope,
                f"group{group} host patch1 to row1 S2MM5",
                mlir,
                f"aie.flow(%shim{group}, DMA : 1, %mt{group}, DMA : {ROW1_WEIGHT_S2MM[1]})",
            )
        )
        for row in range(ROWS_PER_COLUMN):
            errors.extend(
                _require_marker(
                    scope,
                    f"group{group} row{row} weight to main16 DMA1",
                    mlir,
                    f"aie.flow(%mt{group}, DMA : {row}, %m{group}_{row}, DMA : {MAIN_WEIGHT_S2MM})",
                )
            )
        errors.extend(
            _require_no_marker(
                scope,
                f"group{group} old patch0 ingress on compact S2MM0",
                mlir,
                f"aie.flow(%shim{group}, DMA : 0, %mt{group}, DMA : 0)",
            )
        )
        errors.extend(
            _require_no_marker(
                scope,
                f"group{group} old patch1 ingress on compact S2MM1",
                mlir,
                f"aie.flow(%shim{group}, DMA : 1, %mt{group}, DMA : 1)",
            )
        )
    patch_channels = _dma_start_channels(mlir, "S2MM", r"patch[01]_q4nx_ping")
    if sorted(set(patch_channels)) != list(ROW1_WEIGHT_S2MM):
        errors.append(
            f"{scope}: full-layer row1 weight patch ingress channels are {patch_channels}, "
            f"expected only {ROW1_WEIGHT_S2MM}"
        )
    if (
        "func.call @q4nx_chunk_accum_slice_i32_fast" not in mlir
        and "func.call @q4nx_main16_layer_scheduler" not in mlir
    ):
        errors.append(f"{scope}: missing integrated fast Q4NX main16 call")
    role_objects = (
        "main_projection_q4nx_fast.o",
        "postprocess_qkv.o",
        "full_vector_station.o",
        "swiglu.o",
        "edge_attention.o",
    )
    for role_object in role_objects:
        if role_object not in mlir:
            errors.append(f"{scope}: full-layer must link the {role_object} role object")
    if "qwen3_layer.o" in mlir:
        errors.append(f"{scope}: old mixed qwen3_layer object is not a valid full-layer role boundary")
    if "qwen3_bridge.o" in mlir:
        errors.append(f"{scope}: old mixed qwen3_bridge object is not a valid full-layer role boundary")
    if "debug_contract.o" in mlir:
        errors.append(f"{scope}: production full-layer path must not link debug_contract.o")
    return errors
