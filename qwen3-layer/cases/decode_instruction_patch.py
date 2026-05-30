"""Patch qwen3 decode schedule fields in generated NPU instructions."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from cases.decode_cache_reference import CACHE_BLOCK_DWORDS, DecodeSchedule

RTP_WRITE_TRAILER = (0x18, 0, 0)
PUSH_QUEUE_REPEAT_FLAG = 0x80000000
RTP_WRITE_COUNT = 13
RTP_POST_CURRENT_TOKEN = 0
RTP_SHAPE_BLOCKS = (1, 2, 4, 5, 7, 8, 10, 11)
RTP_SHAPE_A_TAIL_TOKENS = (3, 6, 9, 12)


def _rtp_positions(words: np.ndarray) -> list[int]:
    positions: list[int] = []
    for index in range(2, words.size - 3):
        if words[index - 1] == 0 and words[index + 1] == RTP_WRITE_TRAILER[0]:
            positions.append(index)
            if len(positions) == RTP_WRITE_COUNT:
                return positions
    return positions


def _replace_positions(
    words: np.ndarray,
    positions: list[int],
    old_value: int,
    new_value: int,
    expected_count: int,
    label: str,
) -> str:
    if len(positions) != expected_count:
        raise ValueError(f"{label}: expected {expected_count} patch sites, found {len(positions)}")
    old_u32 = np.uint32(old_value)
    new_u32 = np.uint32(new_value)
    for position in positions:
        if words[position] != old_u32:
            raise ValueError(
                f"{label}: word {position} is 0x{int(words[position]):08x}, "
                f"expected 0x{old_value & 0xFFFFFFFF:08x}"
            )
        words[position] = new_u32
    return f"{label}: {expected_count} word(s) 0x{old_value & 0xFFFFFFFF:08x}->0x{new_value & 0xFFFFFFFF:08x}"


def _replace_exact(
    words: np.ndarray,
    old_value: int,
    new_value: int,
    expected_count: int,
    label: str,
) -> str:
    positions = [int(position) for position in np.flatnonzero(words == np.uint32(old_value))]
    return _replace_positions(words, positions, old_value, new_value, expected_count, label)


def _scan_iteration_word(blocks: int) -> int:
    return (blocks << 20) | (CACHE_BLOCK_DWORDS - 1)


def _push_queue_repeat_word(blocks: int, channel: int) -> int:
    return PUSH_QUEUE_REPEAT_FLAG | ((blocks - 1) << 16) | channel


def patch_instruction_stream(
    source_path: Path,
    target_path: Path,
    base_schedule: DecodeSchedule,
    target_schedule: DecodeSchedule,
) -> list[str]:
    if target_schedule.kv_blocks > base_schedule.kv_blocks:
        raise ValueError(
            f"target token{target_schedule.current_token} needs {target_schedule.kv_blocks} blocks, "
            f"but base token{base_schedule.current_token} only reserves {base_schedule.kv_blocks}"
        )

    words = np.fromfile(source_path, dtype=np.uint32)
    if words.size == 0:
        raise ValueError(f"{source_path} is empty")
    rtp_or_lock_positions = _rtp_positions(words)
    if len(rtp_or_lock_positions) < RTP_WRITE_COUNT:
        raise ValueError(f"expected at least {RTP_WRITE_COUNT} RTP writes, found {len(rtp_or_lock_positions)}")
    rtp_positions = rtp_or_lock_positions[:RTP_WRITE_COUNT]

    changes = [
        _replace_positions(
            words,
            [rtp_positions[RTP_POST_CURRENT_TOKEN]],
            base_schedule.current_token,
            target_schedule.current_token,
            1,
            "post current-token RTP",
        ),
        _replace_positions(
            words,
            [rtp_positions[index] for index in RTP_SHAPE_BLOCKS],
            base_schedule.kv_blocks,
            target_schedule.kv_blocks,
            8,
            "Shape block-count RTP",
        ),
        _replace_positions(
            words,
            [rtp_positions[index] for index in RTP_SHAPE_A_TAIL_TOKENS],
            base_schedule.tail_tokens,
            target_schedule.tail_tokens,
            4,
            "Shape-A tail-token RTP",
        ),
        _replace_exact(
            words,
            base_schedule.current_write_byte_offset,
            target_schedule.current_write_byte_offset,
            4,
            "current K/V even write offset",
        ),
        _replace_exact(
            words,
            base_schedule.current_write_byte_offset + 4,
            target_schedule.current_write_byte_offset + 4,
            4,
            "current K/V odd write offset",
        ),
        _replace_exact(
            words,
            _scan_iteration_word(base_schedule.kv_blocks),
            _scan_iteration_word(target_schedule.kv_blocks),
            4,
            "scan BD iteration",
        ),
        _replace_exact(
            words,
            _push_queue_repeat_word(base_schedule.kv_blocks, 0),
            _push_queue_repeat_word(target_schedule.kv_blocks, 0),
            2,
            "K scan queue repeat",
        ),
        _replace_exact(
            words,
            _push_queue_repeat_word(base_schedule.kv_blocks, 1),
            _push_queue_repeat_word(target_schedule.kv_blocks, 1),
            2,
            "V scan queue repeat",
        ),
    ]

    target_path.parent.mkdir(parents=True, exist_ok=True)
    words.tofile(target_path)
    return changes
