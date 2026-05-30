#!/usr/bin/env python3
"""Compare the active IRON main16 ABI with the MyLM raw-main16 ABI."""

from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass
from pathlib import Path


QWEN3_LAYER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(QWEN3_LAYER_DIR))

from cases import full_layer_engine_generate


DEFAULT_MYLM_BD_CSV = Path("/tmp/mylm_solidify_L31/layer_bd.csv")
DEFAULT_TILE = "c2r2"
DEFAULT_IRON_TILE = "m0_0"
LOCAL_MEMORY_BASE = 0x70000
INT_RE = re.compile(r"^-?(?:0x[0-9a-fA-F]+|\d+)$")
BUFFER_RE = re.compile(
    r"%(?P<name>m\d+_\d+_\w+) = aie\.buffer\(%(?P<tile>m\d+_\d+)\) "
    r"\{address = (?P<address>\d+) : i32,"
)
LOCK_RE = re.compile(
    r"%(?P<name>m\d+_\d+_\w+) = aie\.lock\(%(?P<tile>m\d+_\d+), (?P<lock_id>\d+)\)"
)
DMA_BD_RE = re.compile(
    r"aie\.dma_bd\(%(?P<buffer>m\d+_\d+_\w+) : memref<(?P<memref>[^>]+)>, "
    r"(?P<offset>\d+), (?P<length>\d+).*"
    r"\{bd_id = (?P<bd_id>\d+) : i32(?:, next_bd_id = (?P<next_bd>\d+) : i32)?"
)
USE_LOCK_RE = re.compile(
    r"aie\.use_lock\(%(?P<lock>m\d+_\d+_\w+), (?P<mode>\w+), (?P<value>-?\d+)\)"
)


@dataclass(frozen=True)
class Endpoint:
    role: str
    bd_id: int
    length: int
    local_base: int
    next_bd: int
    acq_lock: int
    acq_value: int
    rel_lock: int
    rel_value: int


@dataclass(frozen=True)
class RoleExpectation:
    role: str
    bd_ids: tuple[int, ...]
    lengths: tuple[int, ...]
    local_bases: tuple[int, ...]
    acq_lock: int
    rel_lock: int


@dataclass(frozen=True)
class ComparisonResult:
    role: str
    mylm: RoleExpectation
    iron: RoleExpectation
    mismatches: tuple[str, ...]

    @property
    def matches(self) -> bool:
        return not self.mismatches


MYLM_ROLE_BY_BD = {
    0: "activation",
    1: "activation",
    2: "weight",
    3: "weight",
    4: "record",
    5: "record",
}
IRON_ROLE_BY_BUFFER = {
    "chunk_ping": "activation",
    "chunk_pong": "activation",
    "wt_ping": "weight",
    "wt_pong": "weight",
    "record_ping": "record",
    "record_pong": "record",
}


def _parse_int(text: str) -> int:
    if not INT_RE.match(text):
        raise ValueError(f"bad integer field: {text}")
    return int(text, 0)


def _parse_fields(raw_fields: str) -> dict[str, int | str]:
    fields: dict[str, int | str] = {}
    for item in raw_fields.split():
        key, sep, raw_value = item.partition("=")
        if not sep:
            raise ValueError(f"bad BD field item: {item}")
        fields[key] = _parse_int(raw_value) if INT_RE.match(raw_value) else raw_value
    return fields


def _local_base(global_base: int) -> int:
    if global_base < LOCAL_MEMORY_BASE:
        return global_base
    return global_base - LOCAL_MEMORY_BASE


def _role_expectation(role: str, endpoints: tuple[Endpoint, ...]) -> RoleExpectation:
    role_endpoints = tuple(endpoint for endpoint in endpoints if endpoint.role == role)
    if not role_endpoints:
        return RoleExpectation(role, (), (), (), -1, -1)
    acq_locks = tuple(sorted({endpoint.acq_lock for endpoint in role_endpoints}))
    rel_locks = tuple(sorted({endpoint.rel_lock for endpoint in role_endpoints}))
    return RoleExpectation(
        role=role,
        bd_ids=tuple(endpoint.bd_id for endpoint in sorted(role_endpoints, key=lambda endpoint: endpoint.bd_id)),
        lengths=tuple(endpoint.length for endpoint in sorted(role_endpoints, key=lambda endpoint: endpoint.bd_id)),
        local_bases=tuple(endpoint.local_base for endpoint in sorted(role_endpoints, key=lambda endpoint: endpoint.bd_id)),
        acq_lock=acq_locks[0] if len(acq_locks) == 1 else -1,
        rel_lock=rel_locks[0] if len(rel_locks) == 1 else -1,
    )


def _load_mylm_endpoints(path: Path, tile: str) -> tuple[Endpoint, ...]:
    endpoints: list[Endpoint] = []
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["tile"] != tile:
                continue
            bd_id = int(row["bd"])
            if bd_id not in MYLM_ROLE_BY_BD:
                continue
            fields = _parse_fields(row["fields"])
            endpoints.append(
                Endpoint(
                    role=MYLM_ROLE_BY_BD[bd_id],
                    bd_id=bd_id,
                    length=int(fields["len"]),
                    local_base=_local_base(int(fields["base"])),
                    next_bd=int(fields["next_bd"]),
                    acq_lock=int(fields["acq_id"]),
                    acq_value=int(fields["acq_val"]),
                    rel_lock=int(fields["rel_id"]),
                    rel_value=int(fields["rel_val"]),
                )
            )
    return tuple(sorted(endpoints, key=lambda endpoint: endpoint.bd_id))


def _buffer_short_name(buffer: str, tile: str) -> str:
    prefix = f"{tile}_"
    if not buffer.startswith(prefix):
        raise ValueError(f"buffer {buffer} does not belong to {tile}")
    return buffer[len(prefix) :]


def _parse_lock_use(line: str) -> tuple[str, int]:
    match = USE_LOCK_RE.search(line)
    if match is None:
        raise ValueError(f"missing lock use near DMA BD: {line}")
    return match.group("lock"), int(match.group("value"))


def _dword_length(memref: str, element_count: int) -> int:
    if memref.endswith("xbf16"):
        if element_count % 2 != 0:
            raise ValueError(f"bf16 DMA length is not dword-aligned: memref<{memref}> len={element_count}")
        return element_count // 2
    return element_count


def _load_iron_endpoints(mlir: str, tile: str) -> tuple[Endpoint, ...]:
    buffers: dict[str, int] = {}
    locks: dict[str, int] = {}
    lines = mlir.splitlines()

    for line in lines:
        buffer_match = BUFFER_RE.search(line)
        if buffer_match is not None and buffer_match.group("tile") == tile:
            buffers[buffer_match.group("name")] = int(buffer_match.group("address"))
        lock_match = LOCK_RE.search(line)
        if lock_match is not None and lock_match.group("tile") == tile:
            locks[lock_match.group("name")] = int(lock_match.group("lock_id"))

    endpoints: list[Endpoint] = []
    for index, line in enumerate(lines):
        match = DMA_BD_RE.search(line)
        if match is None:
            continue
        buffer = match.group("buffer")
        if buffer not in buffers:
            continue
        short_name = _buffer_short_name(buffer, tile)
        if short_name in IRON_ROLE_BY_BUFFER:
            role = IRON_ROLE_BY_BUFFER[short_name]
        elif short_name.endswith("records") or short_name == "records":
            role = "record"
        else:
            continue
        acq_lock_name, acq_value = _parse_lock_use(lines[index - 1])
        rel_lock_name, rel_value = _parse_lock_use(lines[index + 1])
        offset_dwords = int(match.group("offset"))
        endpoints.append(
            Endpoint(
                role=role,
                bd_id=int(match.group("bd_id")),
                length=_dword_length(match.group("memref"), int(match.group("length"))),
                local_base=buffers[buffer] + offset_dwords * 4,
                next_bd=int(match.group("next_bd")) if match.group("next_bd") is not None else -1,
                acq_lock=locks[acq_lock_name],
                acq_value=acq_value,
                rel_lock=locks[rel_lock_name],
                rel_value=rel_value,
            )
        )
    return tuple(sorted(endpoints, key=lambda endpoint: (endpoint.role, endpoint.bd_id)))


def _compare_role(mylm: RoleExpectation, iron: RoleExpectation) -> ComparisonResult:
    mismatches: list[str] = []
    if mylm.bd_ids != iron.bd_ids:
        mismatches.append(f"bd_ids mylm={mylm.bd_ids} iron={iron.bd_ids}")
    if mylm.lengths != iron.lengths:
        mismatches.append(f"lengths mylm={mylm.lengths} iron={iron.lengths}")
    if mylm.local_bases != iron.local_bases:
        mismatches.append(
            "local_bases "
            f"mylm={tuple(hex(value) for value in mylm.local_bases)} "
            f"iron={tuple(hex(value) for value in iron.local_bases)}"
        )
    if mylm.acq_lock != iron.acq_lock:
        mismatches.append(f"acq_lock mylm=L{mylm.acq_lock} iron=L{iron.acq_lock}")
    if mylm.rel_lock != iron.rel_lock:
        mismatches.append(f"rel_lock mylm=L{mylm.rel_lock} iron=L{iron.rel_lock}")
    return ComparisonResult(mylm.role, mylm, iron, tuple(mismatches))


def compare_main16_raw_abi(mylm_bd_csv: Path, mylm_tile: str, iron_tile: str) -> tuple[ComparisonResult, ...]:
    mylm_endpoints = _load_mylm_endpoints(mylm_bd_csv, mylm_tile)
    iron_endpoints = _load_iron_endpoints(full_layer_engine_generate.generate_mlir(), iron_tile)
    results: list[ComparisonResult] = []
    for role in ("activation", "weight", "record"):
        results.append(
            _compare_role(
                _role_expectation(role, mylm_endpoints),
                _role_expectation(role, iron_endpoints),
            )
        )
    return tuple(results)


def report(mylm_bd_csv: Path, mylm_tile: str, iron_tile: str) -> str:
    results = compare_main16_raw_abi(mylm_bd_csv, mylm_tile, iron_tile)
    ready = all(result.matches for result in results)
    lines = [
        "main16_raw_abi_check:",
        f"  mylm_tile={mylm_tile}",
        f"  iron_tile={iron_tile}",
        f"  raw_main16_abi_ready={str(ready).lower()}",
    ]
    for result in results:
        status = "MATCH" if result.matches else "DIFF"
        lines.append(f"  {result.role}: {status}")
        lines.append(
            f"    mylm bd={result.mylm.bd_ids} len={result.mylm.lengths} "
            f"base={tuple(hex(value) for value in result.mylm.local_bases)} "
            f"locks=L{result.mylm.acq_lock}->L{result.mylm.rel_lock}"
        )
        lines.append(
            f"    iron bd={result.iron.bd_ids} len={result.iron.lengths} "
            f"base={tuple(hex(value) for value in result.iron.local_bases)} "
            f"locks=L{result.iron.acq_lock}->L{result.iron.rel_lock}"
        )
        for mismatch in result.mismatches:
            lines.append(f"    mismatch: {mismatch}")
    if not ready:
        mismatched_roles = tuple(result.role for result in results if not result.matches)
        if mismatched_roles == ("record",):
            next_step = (
                "    replace IRON main16 multi-record residency with MyLM 17-dword record ping/pong,\n"
                "    then update row1 compact gather to consume one compact record at a time."
            )
        else:
            next_step = (
                "    migrate IRON main16 activation/weight/record BD and lock ownership to MyLM,\n"
                "    or generate a raw scheduled main16 core for the current IRON ABI."
            )
        lines.extend(
            [
                "  next_step:",
                next_step,
            ]
        )
    return "\n".join(lines) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mylm-bd-csv", type=Path, default=DEFAULT_MYLM_BD_CSV)
    parser.add_argument("--mylm-tile", default=DEFAULT_TILE)
    parser.add_argument("--iron-tile", default=DEFAULT_IRON_TILE)
    parser.add_argument("--strict", action="store_true")
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    text = report(args.mylm_bd_csv, args.mylm_tile, args.iron_tile)
    print(text, end="")
    if args.strict and "raw_main16_abi_ready=false" in text:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
