"""Reusable row1 Q4NX weight-stream MLIR-AIE pieces."""

from __future__ import annotations

from dataclasses import dataclass

from contract import CHUNK_BF16, PATCHES_PER_COLUMN, ROWS_PER_COLUMN, ROWS_PER_PATCH

DEFAULT_PATCH_INPUT_BDS = (
    (0, 1),
    (28, 29),
)
DEFAULT_ROW_BDS = (
    (2, 3),
    (24, 25),
    (4, 5),
    (26, 27),
)
PATCH_BUFFER_BF16 = ROWS_PER_PATCH * CHUNK_BF16


@dataclass(frozen=True)
class WeightStreamConfig:
    group: int
    input_channels: tuple[int, ...]
    patch_input_bds: tuple[tuple[int, int], ...]
    row_bds: tuple[tuple[int, int], ...]
    row_block_prefix: str
    row_terminal_label: str
    input_block_prefix: str


def slot_lock(prefix: str, patch: int, slot: str, suffix: str) -> str:
    return f"%{prefix}_patch{patch}_{slot}_{suffix}"


def weight_stream_buffers(tile: str) -> str:
    lines = []
    for patch in range(PATCHES_PER_COLUMN):
        lines.extend(
            (
                f'    %{tile}_patch{patch}_ping = aie.buffer(%{tile}) {{sym_name = "{tile}_patch{patch}_ping"}} : memref<{PATCH_BUFFER_BF16}xbf16>',
                f'    %{tile}_patch{patch}_pong = aie.buffer(%{tile}) {{sym_name = "{tile}_patch{patch}_pong"}} : memref<{PATCH_BUFFER_BF16}xbf16>',
            )
        )
    return "\n".join(lines)


def weight_stream_lock_defs(tile: str, lock_base: int) -> str:
    lines = []
    for patch in range(PATCHES_PER_COLUMN):
        base = lock_base + patch * 4
        lines.extend(
            (
                f'    %{tile}_patch{patch}_ping_empty = aie.lock(%{tile}, {base}) {{init = {ROWS_PER_PATCH} : i32, sym_name = "{tile}_patch{patch}_ping_empty"}}',
                f'    %{tile}_patch{patch}_ping_full = aie.lock(%{tile}, {base + 1}) {{init = 0 : i32, sym_name = "{tile}_patch{patch}_ping_full"}}',
                f'    %{tile}_patch{patch}_pong_empty = aie.lock(%{tile}, {base + 2}) {{init = {ROWS_PER_PATCH} : i32, sym_name = "{tile}_patch{patch}_pong_empty"}}',
                f'    %{tile}_patch{patch}_pong_full = aie.lock(%{tile}, {base + 3}) {{init = 0 : i32, sym_name = "{tile}_patch{patch}_pong_full"}}',
            )
        )
    return "\n".join(lines)


def _patch_block_label(config: WeightStreamConfig, patch: int, slot: str) -> str:
    return f"patch{patch}_{config.input_block_prefix}{slot}"


def _row_start_label(config: WeightStreamConfig, row: int) -> str:
    return f"{config.row_block_prefix}{row}_start"


def _split_input_ring(config: WeightStreamConfig, patch: int) -> str:
    prefix = f"mt{config.group}"
    channel = config.input_channels[patch]
    ping_bd, pong_bd = config.patch_input_bds[patch]
    ping_label = _patch_block_label(config, patch, "ping")
    pong_label = _patch_block_label(config, patch, "pong")
    next_label = f"^{_row_start_label(config, 0)}" if patch + 1 == PATCHES_PER_COLUMN else f"^patch{patch + 1}_start"
    return f"""    ^patch{patch}_start:
      %patch{patch}_dma = aie.dma_start(S2MM, {channel}, ^{ping_label}, {next_label})
    ^{ping_label}:
      aie.use_lock({slot_lock(prefix, patch, "ping", "empty")}, AcquireGreaterEqual, {ROWS_PER_PATCH})
      aie.dma_bd(%{prefix}_patch{patch}_ping : memref<{PATCH_BUFFER_BF16}xbf16>, 0, {PATCH_BUFFER_BF16}) {{bd_id = {ping_bd} : i32, next_bd_id = {pong_bd} : i32}}
      aie.use_lock({slot_lock(prefix, patch, "ping", "full")}, Release, {ROWS_PER_PATCH})
      aie.next_bd ^{pong_label}
    ^{pong_label}:
      aie.use_lock({slot_lock(prefix, patch, "pong", "empty")}, AcquireGreaterEqual, {ROWS_PER_PATCH})
      aie.dma_bd(%{prefix}_patch{patch}_pong : memref<{PATCH_BUFFER_BF16}xbf16>, 0, {PATCH_BUFFER_BF16}) {{bd_id = {pong_bd} : i32, next_bd_id = {ping_bd} : i32}}
      aie.use_lock({slot_lock(prefix, patch, "pong", "full")}, Release, {ROWS_PER_PATCH})
      aie.next_bd ^{ping_label}"""


def weight_stream_input_rings(config: WeightStreamConfig) -> str:
    if len(config.input_channels) != PATCHES_PER_COLUMN:
        raise ValueError(f"bad weight input channel count: {config.input_channels}")
    if len(config.patch_input_bds) != PATCHES_PER_COLUMN:
        raise ValueError(f"bad weight input BD count: {config.patch_input_bds}")
    return "\n".join(_split_input_ring(config, patch) for patch in range(PATCHES_PER_COLUMN))


def _row_stream(config: WeightStreamConfig, row: int) -> str:
    prefix = f"mt{config.group}"
    patch = row // ROWS_PER_PATCH
    row_in_patch = row % ROWS_PER_PATCH
    offset = row_in_patch * CHUNK_BF16
    ping_bd, pong_bd = config.row_bds[row]
    next_label = f"^{_row_start_label(config, row + 1)}" if row + 1 < ROWS_PER_COLUMN else config.row_terminal_label
    return f"""    ^{_row_start_label(config, row)}:
      %{config.row_block_prefix}{row}_dma = aie.dma_start(MM2S, {row}, ^row{row}_ping, {next_label})
    ^row{row}_ping:
      aie.use_lock({slot_lock(prefix, patch, "ping", "full")}, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_patch{patch}_ping : memref<{PATCH_BUFFER_BF16}xbf16>, {offset}, {CHUNK_BF16}) {{bd_id = {ping_bd} : i32, next_bd_id = {pong_bd} : i32}}
      aie.use_lock({slot_lock(prefix, patch, "ping", "empty")}, Release, 1)
      aie.next_bd ^row{row}_pong
    ^row{row}_pong:
      aie.use_lock({slot_lock(prefix, patch, "pong", "full")}, AcquireGreaterEqual, 1)
      aie.dma_bd(%{prefix}_patch{patch}_pong : memref<{PATCH_BUFFER_BF16}xbf16>, {offset}, {CHUNK_BF16}) {{bd_id = {pong_bd} : i32, next_bd_id = {ping_bd} : i32}}
      aie.use_lock({slot_lock(prefix, patch, "pong", "empty")}, Release, 1)
      aie.next_bd ^row{row}_ping"""


def weight_stream_row_streams(config: WeightStreamConfig) -> str:
    if len(config.row_bds) != ROWS_PER_COLUMN:
        raise ValueError(f"bad row stream BD count: {config.row_bds}")
    return "\n".join(_row_stream(config, row) for row in range(ROWS_PER_COLUMN))
