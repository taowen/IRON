"""Tile-local resource ownership manifests for qwen3-layer generators."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

QKV_PHASES = frozenset(("Q", "K", "V"))
AIE_BUFFER_ALLOCATION_GRANULARITY = 0x200
COMPUTE_BANK_SIZE = 0x4000
COMPUTE_BANKS = 4
COMPUTE_STACK_SIZE = 0x400


@dataclass(frozen=True)
class BufferResource:
    name: str
    mlir_name: str
    address: int
    size_bytes: int
    phases: tuple[str, ...]
    semantic: str


@dataclass(frozen=True)
class LockResource:
    name: str
    mlir_name: str
    lock_id: int
    init: int
    phases: tuple[str, ...]
    semantic: str


@dataclass(frozen=True)
class BDResource:
    name: str
    bd_id: int
    channel: int
    direction: str
    length: int
    buffer: str
    phases: tuple[str, ...]
    semantic: str
    next_bd_id: int | None = None
    allow_phase_reuse: bool = False


@dataclass(frozen=True)
class TileResource:
    name: str
    coord: tuple[int, int]
    role: str
    buffers: tuple[BufferResource, ...]
    locks: tuple[LockResource, ...]
    bds: tuple[BDResource, ...]


@dataclass(frozen=True)
class ResourceManifest:
    schema: str
    case: str
    tiles: tuple[TileResource, ...]


def write_resource_manifest(path: Path, manifest: ResourceManifest) -> None:
    path.write_text(json.dumps(asdict(manifest), indent=2, sort_keys=True) + "\n")


def validate_resource_manifest(scope: str, manifest: ResourceManifest) -> list[str]:
    errors: list[str] = []
    if manifest.schema != "qwen3-layer-resource-manifest-v1":
        errors.append(f"{scope}: unknown resource manifest schema {manifest.schema}")
    tile_names = tuple(tile.name for tile in manifest.tiles)
    if len(set(tile_names)) != len(tile_names):
        errors.append(f"{scope}: duplicate tile resource names in manifest")
    for tile in manifest.tiles:
        errors.extend(_validate_tile_buffers(scope, tile))
        errors.extend(_validate_tile_locks(scope, tile))
        errors.extend(_validate_tile_bds(scope, tile))
    return errors


def validate_manifest_matches_mlir(scope: str, manifest: ResourceManifest, mlir: str) -> list[str]:
    errors: list[str] = []
    for tile in manifest.tiles:
        for buffer in tile.buffers:
            marker = (
                f"%{buffer.mlir_name} = aie.buffer(%{tile.name}) "
                f'{{address = {buffer.address} : i32, sym_name = "{buffer.mlir_name}"}}'
            )
            if marker not in mlir:
                errors.append(
                    f"{scope}: manifest buffer {tile.name}.{buffer.name} missing from MLIR "
                    f"at 0x{buffer.address:04x}"
                )
        for lock in tile.locks:
            marker = (
                f"%{lock.mlir_name} = aie.lock(%{tile.name}, {lock.lock_id}) "
                f'{{init = {lock.init} : i32, sym_name = "{lock.mlir_name}"}}'
            )
            if marker not in mlir:
                errors.append(f"{scope}: manifest lock {tile.name}.{lock.name} missing from MLIR")
        for bd in tile.bds:
            if f"aie.dma_start({bd.direction}, {bd.channel}," not in mlir:
                errors.append(
                    f"{scope}: manifest BD {tile.name}.{bd.name} missing DMA start "
                    f"{bd.direction}{bd.channel}"
                )
            if not _bd_marker_present(mlir, bd):
                errors.append(
                    f"{scope}: manifest BD {tile.name}.{bd.name} missing buffer={bd.buffer} "
                    f"bd_id={bd.bd_id} length={bd.length}"
                )
        if tile.role == "main16":
            errors.extend(_validate_compute_bank_cursor(scope, tile, mlir))
    return errors


def compare_main16_qkv_contracts(
    scope: str,
    baseline: ResourceManifest,
    candidate: ResourceManifest,
) -> list[str]:
    baseline_qkv = _main16_qkv_signature(baseline)
    candidate_qkv = _main16_qkv_signature(candidate)
    if baseline_qkv == candidate_qkv:
        return []
    return [
        f"{scope}: main16 QKV resource contract changed between "
        f"{baseline.case} and {candidate.case}"
    ]


def _validate_tile_buffers(scope: str, tile: TileResource) -> list[str]:
    errors: list[str] = []
    names = tuple(buffer.name for buffer in tile.buffers)
    if len(set(names)) != len(names):
        errors.append(f"{scope}: {tile.name} has duplicate buffer names")
    for left_idx, left in enumerate(tile.buffers):
        if left.size_bytes <= 0:
            errors.append(f"{scope}: {tile.name}.{left.name} has non-positive size")
        for right in tile.buffers[left_idx + 1 :]:
            if _range_overlap(
                left.address,
                _round_allocation_size(left.size_bytes),
                right.address,
                _round_allocation_size(right.size_bytes),
            ):
                errors.append(
                    f"{scope}: {tile.name} allocator-rounded buffers overlap: "
                    f"{left.name}[0x{left.address:04x},"
                    f"0x{left.address + _round_allocation_size(left.size_bytes):04x}) and "
                    f"{right.name}[0x{right.address:04x},"
                    f"0x{right.address + _round_allocation_size(right.size_bytes):04x})"
                )
            if _phase_overlap(left.phases, right.phases) and _range_overlap(
                left.address,
                left.size_bytes,
                right.address,
                right.size_bytes,
            ):
                errors.append(
                    f"{scope}: {tile.name} overlapping buffers with shared lifetime: "
                    f"{left.name}[0x{left.address:04x},0x{left.address + left.size_bytes:04x}) and "
                    f"{right.name}[0x{right.address:04x},0x{right.address + right.size_bytes:04x})"
                )
    return errors


def _validate_compute_bank_cursor(scope: str, tile: TileResource, mlir: str) -> list[str]:
    errors: list[str] = []
    ordered_buffers = sorted(tile.buffers, key=lambda buffer: _buffer_declaration_position(tile, buffer, mlir))
    bank_next = [
        COMPUTE_STACK_SIZE,
        COMPUTE_BANK_SIZE,
        COMPUTE_BANK_SIZE * 2,
        COMPUTE_BANK_SIZE * 3,
    ]
    for buffer in ordered_buffers:
        bank = buffer.address // COMPUTE_BANK_SIZE
        if bank >= COMPUTE_BANKS:
            errors.append(
                f"{scope}: {tile.name}.{buffer.name} address 0x{buffer.address:04x} is outside compute banks"
            )
            continue
        bank_end = (bank + 1) * COMPUTE_BANK_SIZE
        if buffer.address < bank_next[bank]:
            errors.append(
                f"{scope}: {tile.name}.{buffer.name} declared after a later address in bank {bank}: "
                f"0x{buffer.address:04x} < cursor 0x{bank_next[bank]:04x}"
            )
        if buffer.address + buffer.size_bytes > bank_end:
            errors.append(
                f"{scope}: {tile.name}.{buffer.name} crosses bank {bank} boundary: "
                f"0x{buffer.address:04x}+0x{buffer.size_bytes:x} > 0x{bank_end:04x}"
            )
        bank_next[bank] = max(bank_next[bank], buffer.address + buffer.size_bytes)
    return errors


def _buffer_declaration_position(tile: TileResource, buffer: BufferResource, mlir: str) -> int:
    marker = f"%{buffer.mlir_name} = aie.buffer(%{tile.name}) "
    position = mlir.find(marker)
    if position < 0:
        return len(mlir)
    return position


def _validate_tile_locks(scope: str, tile: TileResource) -> list[str]:
    errors: list[str] = []
    by_id: dict[int, list[LockResource]] = {}
    for lock in tile.locks:
        by_id.setdefault(lock.lock_id, []).append(lock)
    for lock_id, locks in by_id.items():
        first = locks[0]
        for lock in locks[1:]:
            if (
                first.mlir_name != lock.mlir_name
                or first.init != lock.init
                or first.semantic != lock.semantic
                or _phase_overlap(first.phases, lock.phases)
            ):
                errors.append(f"{scope}: {tile.name} lock id {lock_id} has conflicting ownership")
    return errors


def _validate_tile_bds(scope: str, tile: TileResource) -> list[str]:
    errors: list[str] = []
    by_id: dict[int, list[BDResource]] = {}
    for bd in tile.bds:
        if bd.direction not in ("S2MM", "MM2S"):
            errors.append(f"{scope}: {tile.name}.{bd.name} has invalid direction {bd.direction}")
        by_id.setdefault(bd.bd_id, []).append(bd)
    for bd_id, bds in by_id.items():
        first = bds[0]
        for bd in bds[1:]:
            identical = (
                first.channel == bd.channel
                and first.direction == bd.direction
                and first.length == bd.length
                and first.buffer == bd.buffer
                and first.semantic == bd.semantic
                and first.next_bd_id == bd.next_bd_id
            )
            explicit_reuse = first.allow_phase_reuse and bd.allow_phase_reuse and not _phase_overlap(first.phases, bd.phases)
            if not identical and not explicit_reuse:
                errors.append(f"{scope}: {tile.name} BD id {bd_id} has conflicting ownership")
    return errors


def _bd_marker_present(mlir: str, bd: BDResource) -> bool:
    next_marker = "" if bd.next_bd_id is None else rf"(?=[^}}]*next_bd_id = {bd.next_bd_id} : i32)"
    pattern = (
        rf"aie\.dma_bd\(%{re.escape(bd.buffer)} : memref<[^>]+>, "
        rf"[^,]+, {bd.length}(?:, \[[^\]]+\])?\) "
        rf"\{{(?=[^}}]*bd_id = {bd.bd_id} : i32){next_marker}[^}}]*\}}"
    )
    return re.search(pattern, mlir) is not None


def _main16_qkv_signature(manifest: ResourceManifest) -> tuple[tuple[str, tuple, tuple, tuple], ...]:
    rows = []
    for tile in manifest.tiles:
        if tile.role != "main16":
            continue
        buffers = tuple(
            sorted(
                (
                    buffer.name,
                    buffer.mlir_name,
                    buffer.address,
                    buffer.size_bytes,
                    buffer.phases,
                    buffer.semantic,
                )
                for buffer in tile.buffers
                if _phase_intersects_qkv(buffer.phases)
            )
        )
        locks = tuple(
            sorted(
                (
                    lock.name,
                    lock.mlir_name,
                    lock.lock_id,
                    lock.init,
                    lock.phases,
                    lock.semantic,
                )
                for lock in tile.locks
                if _phase_intersects_qkv(lock.phases)
            )
        )
        bds = tuple(
            sorted(
                (
                    bd.name,
                    bd.bd_id,
                    bd.channel,
                    bd.direction,
                    bd.length,
                    bd.buffer,
                    bd.phases,
                    bd.semantic,
                )
                for bd in tile.bds
                if _phase_intersects_qkv(bd.phases)
            )
        )
        rows.append((tile.name, buffers, locks, bds))
    return tuple(sorted(rows))


def _phase_intersects_qkv(phases: tuple[str, ...]) -> bool:
    return bool(QKV_PHASES.intersection(phases))


def _phase_overlap(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    return bool(set(left).intersection(right))


def _range_overlap(left_address: int, left_size: int, right_address: int, right_size: int) -> bool:
    return left_address < right_address + right_size and right_address < left_address + left_size


def _round_allocation_size(size_bytes: int) -> int:
    return ((size_bytes + AIE_BUFFER_ALLOCATION_GRANULARITY - 1) // AIE_BUFFER_ALLOCATION_GRANULARITY) * (
        AIE_BUFFER_ALLOCATION_GRANULARITY
    )
