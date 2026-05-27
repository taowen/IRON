#!/usr/bin/env python3
"""MyLM edge/KV BD-ring contract draft.

This file intentionally stays below full MLIR generation.  It captures the
static rows we must reproduce before another fused-layer experiment is useful.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import re

PLANE_BASES = (
    ("k03", 0x000000),
    ("v03", 0x400000),
    ("k47", 0x800000),
    ("v47", 0xC00000),
)

CURRENT_PLANES = ("k03", "k47", "v03", "v47")
HISTORY_PLANES = ("k03", "v03", "k47", "v47")


@dataclass(frozen=True)
class BdRow:
    tile: str
    bd: int
    base: int
    length_dwords: int
    next_bd: int
    acquire_lock: int
    acquire_value: int
    release_lock: int
    release_value: int
    packet_id: int | None = None
    d1_step: int | None = None
    d1_wrap: int | None = None
    d2_step: int | None = None
    d2_wrap: int | None = None


@dataclass(frozen=True)
class RuntimeKvDescriptor:
    role: str
    plane: str
    offset: int
    length_dwords: int


def kv_descriptors(context_len: int) -> tuple[RuntimeKvDescriptor, ...]:
    current_inside = (context_len - 1) * 0x400
    history_len = ((context_len + 15) // 16) * 0x1000
    plane_bases = dict(PLANE_BASES)
    current = tuple(
        RuntimeKvDescriptor(
            "current", plane, plane_bases[plane] + current_inside, 0x100
        )
        for plane in CURRENT_PLANES
    )
    history = tuple(
        RuntimeKvDescriptor("history", plane, plane_bases[plane], history_len)
        for plane in HISTORY_PLANES
    )
    return current + history


def edge_rows(tile: str) -> tuple[BdRow, ...]:
    return (
        BdRow(tile, 0, 0x20000, 4096, 1, 64, -1, 65, 1),
        BdRow(tile, 1, 0x24000, 4096, 0, 64, -1, 65, 1),
        BdRow(
            tile,
            2,
            0x20000,
            2048,
            3,
            65,
            -1,
            66,
            1,
            d1_step=255,
            d1_wrap=16,
            d2_step=3,
            d2_wrap=32,
        ),
        BdRow(
            tile,
            3,
            0x24000,
            2048,
            2,
            65,
            -1,
            66,
            1,
            d1_step=255,
            d1_wrap=16,
            d2_step=3,
            d2_wrap=32,
        ),
        BdRow(
            tile,
            24,
            0x20080,
            2048,
            25,
            66,
            -1,
            64,
            1,
            d1_step=255,
            d1_wrap=16,
            d2_step=3,
            d2_wrap=32,
        ),
        BdRow(
            tile,
            25,
            0x24080,
            2048,
            24,
            66,
            -1,
            64,
            1,
            d1_step=255,
            d1_wrap=16,
            d2_step=3,
            d2_wrap=32,
        ),
        BdRow(tile, 26, 0x28000, 4096, 27, 67, -1, 68, 1),
        BdRow(tile, 27, 0x2C000, 4096, 26, 67, -1, 68, 1),
        BdRow(
            tile,
            4,
            0x28000,
            2048,
            5,
            68,
            -1,
            69,
            1,
            d1_step=255,
            d1_wrap=8,
            d2_step=3,
            d2_wrap=32,
        ),
        BdRow(
            tile,
            5,
            0x2C000,
            2048,
            4,
            68,
            -1,
            69,
            1,
            d1_step=255,
            d1_wrap=8,
            d2_step=3,
            d2_wrap=32,
        ),
        BdRow(
            tile,
            28,
            0x28080,
            2048,
            29,
            69,
            -1,
            67,
            1,
            d1_step=255,
            d1_wrap=8,
            d2_step=3,
            d2_wrap=32,
        ),
        BdRow(
            tile,
            29,
            0x2C080,
            2048,
            28,
            69,
            -1,
            67,
            1,
            d1_step=255,
            d1_wrap=8,
            d2_step=3,
            d2_wrap=32,
        ),
    )


def c1r1_rows() -> tuple[BdRow, ...]:
    tile = "c1r1"
    return (
        BdRow(tile, 0, 0x20000, 65, 1, 64, -1, 65, 1),
        BdRow(tile, 1, 0x24000, 65, 0, 64, -1, 65, 1),
        BdRow(tile, 24, 0x20041, 64, 25, 65, -1, 66, 1),
        BdRow(tile, 25, 0x24041, 64, 24, 65, -1, 66, 1),
        BdRow(tile, 2, 0x20081, 64, 3, 66, -1, 67, 1),
        BdRow(tile, 3, 0x24081, 64, 2, 66, -1, 67, 1),
        BdRow(tile, 26, 0x200C1, 64, 27, 67, -1, 68, 1),
        BdRow(tile, 27, 0x240C1, 64, 26, 67, -1, 68, 1),
        BdRow(tile, 4, 0x20000, 257, 5, 68, -1, 64, 1),
        BdRow(tile, 5, 0x24000, 257, 4, 68, -1, 64, 1),
        BdRow(tile, 6, 0x28000, 256, 7, 69, -1, 70, 1),
        BdRow(tile, 7, 0x2C000, 256, 6, 69, -1, 70, 1),
        BdRow(tile, 28, 0x28000, 256, 29, 70, -1, 69, 1),
        BdRow(tile, 29, 0x2C000, 256, 28, 70, -1, 69, 1),
    )


def c6r1_rows() -> tuple[BdRow, ...]:
    tile = "c6r1"
    return (
        BdRow(tile, 0, 0x20000, 6144, 0, 64, -8, 65, 8),
        BdRow(tile, 1, 0x20000, 6144, 1, 65, -1, 64, 1, packet_id=0),
        BdRow(tile, 24, 0x24000, 2048, 24, 66, -1, 67, 1),
        BdRow(
            tile,
            25,
            0x24000,
            512,
            25,
            67,
            -1,
            68,
            1,
            d1_step=63,
            d1_wrap=8,
            d2_step=3,
            d2_wrap=16,
        ),
        BdRow(
            tile,
            2,
            0x24200,
            512,
            2,
            68,
            -1,
            69,
            1,
            d1_step=63,
            d1_wrap=8,
            d2_step=3,
            d2_wrap=16,
        ),
        BdRow(
            tile,
            26,
            0x24400,
            512,
            26,
            69,
            -1,
            70,
            1,
            d1_step=63,
            d1_wrap=8,
            d2_step=3,
            d2_wrap=16,
        ),
        BdRow(
            tile,
            3,
            0x24600,
            512,
            3,
            70,
            -1,
            66,
            1,
            d1_step=63,
            d1_wrap=8,
            d2_step=3,
            d2_wrap=16,
        ),
        BdRow(tile, 4, 0x28000, 512, 4, 71, -8, 72, 1),
        BdRow(tile, 27, 0x28200, 512, 27, 72, -1, 73, 1),
        BdRow(tile, 5, 0x28400, 512, 5, 73, -1, 74, 1),
        BdRow(tile, 28, 0x28600, 512, 28, 74, -1, 75, 8),
        BdRow(tile, 29, 0x28000, 2048, 29, 75, -1, 71, 1, packet_id=2),
    )


def packet_rows() -> tuple[BdRow, ...]:
    return (
        BdRow("c1r3", 2, 0x8000000, 256, 3, 5, -1, 4, 1, packet_id=14),
        BdRow("c1r3", 3, 0xC000000, 256, 4, 7, -1, 6, 1, packet_id=15),
        BdRow("c1r3", 4, 0x8400000, 256, 5, 5, -1, 4, 1, packet_id=14),
        BdRow("c1r3", 5, 0xC400000, 256, 2, 7, -1, 6, 1, packet_id=15),
        BdRow("c6r2", 0, 0x400000, 512, 1, 0, -1, 1, 1),
        BdRow("c6r2", 1, 0x4000000, 512, 0, 0, -1, 1, 1),
        BdRow("c6r2", 2, 0x8000000, 256, 3, 3, -1, 2, 1),
        BdRow("c6r2", 3, 0xC000000, 256, 2, 3, -1, 2, 1),
    )


def all_rows() -> tuple[BdRow, ...]:
    return (
        edge_rows("c0r1")
        + edge_rows("c7r1")
        + c1r1_rows()
        + c6r1_rows()
        + packet_rows()
    )


def validate(rows: tuple[BdRow, ...]) -> list[str]:
    errors: list[str] = []
    by_tile_bd = {(row.tile, row.bd): row for row in rows}
    for row in rows:
        if row.length_dwords <= 0:
            errors.append(f"{row.tile} bd{row.bd}: non-positive length")
        if (row.tile, row.next_bd) not in by_tile_bd:
            errors.append(f"{row.tile} bd{row.bd}: next bd{row.next_bd} missing")
        if row.acquire_lock < 0 or row.release_lock < 0:
            errors.append(f"{row.tile} bd{row.bd}: negative lock id")
    return errors


def parse_int(text: str) -> int:
    return int(text, 0)


def parse_fields(text: str) -> dict[str, str]:
    return dict(re.findall(r"(\w+)=([^ ]+)", text))


def read_bd_csv(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    with path.open() as f:
        csv_rows = list(csv.DictReader(f))
    rows: dict[tuple[str, int], dict[str, str]] = {}
    for row in csv_rows:
        rows[(row["tile"], int(row["bd"]))] = parse_fields(row["fields"])
    return rows


def compare_to_bd_csv(rows: tuple[BdRow, ...], bd_csv: Path) -> list[str]:
    actual = read_bd_csv(bd_csv)
    errors: list[str] = []
    for row in rows:
        key = (row.tile, row.bd)
        fields = actual.get(key)
        if fields is None:
            errors.append(f"{row.tile} bd{row.bd}: missing in {bd_csv}")
            continue

        expected_values = {
            "base": row.base,
            "len": row.length_dwords,
            "next_bd": row.next_bd,
            "acq_id": row.acquire_lock,
            "acq_val": row.acquire_value,
            "rel_id": row.release_lock,
            "rel_val": row.release_value,
        }
        for field, expected in expected_values.items():
            actual_value = parse_int(fields[field])
            if actual_value != expected:
                errors.append(
                    f"{row.tile} bd{row.bd}: {field} expected {expected}, got {actual_value}"
                )

        packet_en = parse_int(fields["packet_en"])
        packet_id = parse_int(fields["packet_id"])
        if row.packet_id is None:
            if packet_en != 0:
                errors.append(f"{row.tile} bd{row.bd}: expected packet disabled")
        elif packet_en != 1 or packet_id != row.packet_id:
            errors.append(
                f"{row.tile} bd{row.bd}: packet expected {row.packet_id}, got en={packet_en} id={packet_id}"
            )

        if row.d1_step is not None:
            dim_values = {
                "d1_step": row.d1_step,
                "d1_wrap": row.d1_wrap,
                "d2_step": row.d2_step,
                "d2_wrap": row.d2_wrap,
            }
            for field, expected in dim_values.items():
                actual_value = parse_int(fields[field])
                if actual_value != expected:
                    errors.append(
                        f"{row.tile} bd{row.bd}: {field} expected {expected}, got {actual_value}"
                    )
    return errors


def stride_text(row: BdRow) -> str:
    if row.d1_step is None:
        return ""
    return f" d1={row.d1_step}x{row.d1_wrap} d2={row.d2_step}x{row.d2_wrap}"


def print_contract(context_len: int) -> None:
    current_inside = (context_len - 1) * 0x400
    history_len = ((context_len + 15) // 16) * 0x1000
    print(f"L={context_len}")
    print(f"current_inside=0x{current_inside:x}")
    print(f"history_len=0x{history_len:x} dwords")
    print()
    print("runtime KV descriptors:")
    for desc in kv_descriptors(context_len):
        print(
            f"  {desc.role:7} {desc.plane:3} "
            f"offset=0x{desc.offset:07x} len=0x{desc.length_dwords:x}"
        )

    rows = all_rows()
    errors = validate(rows)
    print()
    print("static BD rows:")
    for row in rows:
        packet = "-" if row.packet_id is None else str(row.packet_id)
        print(
            f"  {row.tile} bd{row.bd:02d} base=0x{row.base:x} "
            f"len={row.length_dwords} next={row.next_bd} "
            f"lock {row.acquire_lock}:{row.acquire_value}->{row.release_lock}:{row.release_value} "
            f"pkt={packet}{stride_text(row)}"
        )

    print()
    if errors:
        print("validation: FAIL")
        for error in errors:
            print(f"  {error}")
    else:
        print("validation: PASS")


def lock_op(value: int) -> tuple[str, int]:
    if value < 0:
        return "AcquireGreaterEqual", -value
    return "Release", value


def mlir_fragment(rows: tuple[BdRow, ...]) -> str:
    buffer_lengths: dict[tuple[str, int], int] = {}
    for row in rows:
        key = (row.tile, row.base)
        buffer_lengths[key] = max(buffer_lengths.get(key, 0), row.length_dwords)

    lines = [
        "// Generated from bd_ring_spec.py.",
        "// This is a checked BD-ring fragment, not a complete aie.device module.",
        "// Buffers are named by their MyLM absolute base for auditability.",
    ]
    for row in rows:
        acquire_op, acquire_value = lock_op(row.acquire_value)
        release_op, release_value = lock_op(row.release_value)
        packet = ""
        if row.packet_id is not None:
            packet = (
                f", packet = #aie.packet_info<pkt_type = 0, pkt_id = {row.packet_id}>"
            )
        if row.d1_step is None:
            dim_comment = ""
        else:
            dim_comment = (
                f" // d1={row.d1_step}x{row.d1_wrap} " f"d2={row.d2_step}x{row.d2_wrap}"
            )
        lines.extend(
            [
                f"// {row.tile} bd{row.bd} base=0x{row.base:x}{dim_comment}",
                f"^{row.tile}_bd{row.bd:02d}:",
                f"  aie.use_lock(%{row.tile}_lock{row.acquire_lock}, {acquire_op}, {acquire_value})",
                (
                    f"  aie.dma_bd(%{row.tile}_buf_{row.base:x} : "
                    f"memref<{buffer_lengths[(row.tile, row.base)]}xi32>, "
                    f"0, {row.length_dwords}) "
                    f"{{bd_id = {row.bd} : i32, next_bd_id = {row.next_bd} : i32{packet}}}"
                ),
                f"  aie.use_lock(%{row.tile}_lock{row.release_lock}, {release_op}, {release_value})",
                f"  aie.next_bd ^{row.tile}_bd{row.next_bd:02d}",
                "",
            ]
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--l", type=int, default=31, help="context length")
    parser.add_argument("--compare-bd-csv", type=Path)
    parser.add_argument("--emit-mlir-fragment", action="store_true")
    args = parser.parse_args()
    rows = all_rows()
    if args.emit_mlir_fragment:
        print(mlir_fragment(rows))
    else:
        print_contract(args.l)

    errors = validate(rows)
    if args.compare_bd_csv:
        errors.extend(compare_to_bd_csv(rows, args.compare_bd_csv))
        if errors:
            print()
            print("compare: FAIL")
            for error in errors:
                print(f"  {error}")
        else:
            print()
            print("compare: PASS")
    if errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
