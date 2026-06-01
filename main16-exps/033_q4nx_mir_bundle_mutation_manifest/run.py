#!/usr/bin/env python3
"""Build a bundle-level mutation manifest for the MyLM Q4NX hot loop."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import traceback
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
EXP006_RUN = REPO_ROOT / "main16-exps/006_q4nx_alias_lifetime_graph/run.py"
EXP029_RUN = REPO_ROOT / "main16-exps/029_q4nx_mir_prebundled_exact_replay/run.py"
MANIFEST = EXPERIMENT_DIR / "q4nx_mir_bundle_mutation_manifest.json"
REPORT = EXPERIMENT_DIR / "q4nx_mir_bundle_mutation_manifest.md"

HOT_START = 0x260
HOT_END = 0x1850
TIMING_CRITICAL_PADDING = {0x2BA}
BYTE_DIFF_CANARY_PADDING = {0x1820, 0x183C}
KNOWN_FAILED_SOURCE_MUTATIONS = {
    0x44E: "failed experiment 034 direct-QKV numeric gate after vmov-to-nopm replacement",
    0x700: "failed experiment 038 zero/offset direct-QKV numeric gate after vmov-to-nopm replacement",
    0x9B4: "failed experiment 038 zero/offset direct-QKV numeric gate after vmov-to-nopm replacement",
    0xC68: "failed experiment 038 zero/offset direct-QKV numeric gate after vmov-to-nopm replacement",
    0xF1C: "failed experiment 038 zero/offset direct-QKV numeric gate after vmov-to-nopm replacement",
    0x11D0: "failed experiment 038 zero/offset direct-QKV numeric gate after vmov-to-nopm replacement",
    0x1484: "failed experiment 038 zero/offset direct-QKV numeric gate after vmov-to-nopm replacement",
    0x173E: "failed experiment 038 zero/offset direct-QKV numeric gate after vmov-to-nopm replacement",
}
SOURCE_CANDIDATE_SEMANTICS = {"register_move"}
MUTATION_BLOCKING_SEMANTICS = {
    "activation_lane_broadcast",
    "activation_load",
    "accum_to_bf16_vector",
    "dequant_add",
    "dequant_sub",
    "group_sum_load",
    "local_scratch_load",
    "mac",
    "mul_seed",
    "q4_unpack",
    "scalar_pointer_setup",
    "ups_vector_to_accum",
    "weight_or_state_load",
}


@dataclass(frozen=True)
class CellUse:
    cell: str
    action: str
    next_event_index: int | None
    next_address: str | None
    next_op: str | None


@dataclass(frozen=True)
class EventRow:
    index: int
    address: str
    group: int
    op: str
    semantic: str
    fragment: str
    def_cells: tuple[str, ...]
    use_cells: tuple[str, ...]
    def_next: tuple[CellUse, ...]


@dataclass(frozen=True)
class BundleRow:
    kind: str
    address: str
    length: int
    bytes_hex: str
    group: int | None
    semantics: tuple[str, ...]
    ops: tuple[str, ...]
    def_cells: tuple[str, ...]
    use_cells: tuple[str, ...]
    mutation_class: str
    reason: str
    events: tuple[EventRow, ...]


@dataclass(frozen=True)
class Summary:
    status: str
    total_bundles: int
    source_bundles: int
    padding_bundles: int
    hard_safe_source_candidates: int
    weak_dead_def_source_candidates: int
    timing_critical_padding: int
    byte_diff_canary_padding: int


@dataclass(frozen=True)
class ManifestRoot:
    experiment: str
    summary: Summary
    weak_source_candidates: tuple[BundleRow, ...]
    known_failed_source_mutations: tuple[BundleRow, ...]
    padding_canaries: tuple[BundleRow, ...]
    timing_critical_padding: tuple[BundleRow, ...]
    bundles: tuple[BundleRow, ...]


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXP006 = load_module("mylm_exp006_for_bundle_manifest", EXP006_RUN)
EXP029 = load_module("mylm_exp029_for_bundle_manifest", EXP029_RUN)


def dedup(values: list[str]) -> tuple[str, ...]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return tuple(out)


def register_cells(register: str) -> tuple[str, ...]:
    match = re.fullmatch(r"x(\d+)", register)
    if match is not None:
        index = match.group(1)
        return (f"vec{index}.lo", f"vec{index}.hi")
    match = re.fullmatch(r"wl(\d+)", register)
    if match is not None:
        return (f"vec{match.group(1)}.lo",)
    match = re.fullmatch(r"wh(\d+)", register)
    if match is not None:
        return (f"vec{match.group(1)}.hi",)
    match = re.fullmatch(r"dm(\d+)", register)
    if match is not None:
        index = match.group(1)
        return (
            f"acc{index}.bmll",
            f"acc{index}.bmlh",
            f"acc{index}.bmhl",
            f"acc{index}.bmhh",
        )
    match = re.fullmatch(r"cml(\d+)", register)
    if match is not None:
        index = match.group(1)
        return (f"acc{index}.bmll", f"acc{index}.bmlh")
    match = re.fullmatch(r"cmh(\d+)", register)
    if match is not None:
        index = match.group(1)
        return (f"acc{index}.bmhl", f"acc{index}.bmhh")
    match = re.fullmatch(r"bm(ll|lh|hl|hh)(\d+)", register)
    if match is not None:
        return (f"acc{match.group(2)}.bm{match.group(1)}",)
    return (f"scalar.{register}",)


def event_def_cells(event) -> tuple[str, ...]:
    cells: list[str] = []
    for register in event.defs:
        cells.extend(register_cells(register))
    return dedup(cells)


def event_use_cells(event) -> tuple[str, ...]:
    cells: list[str] = []
    for register in event.uses:
        cells.extend(register_cells(register))
    return dedup(cells)


def next_cell_uses(events: tuple) -> dict[tuple[int, str], CellUse]:
    table: dict[tuple[int, str], CellUse] = {}
    event_defs = [event_def_cells(event) for event in events]
    event_uses = [event_use_cells(event) for event in events]
    for event in events:
        for cell in event_def_cells(event):
            action = "terminal_dead"
            next_event_index: int | None = None
            next_address: str | None = None
            next_op: str | None = None
            for later, defs, uses in zip(events[event.index + 1 :], event_defs[event.index + 1 :], event_uses[event.index + 1 :]):
                if cell in uses:
                    action = "used"
                    next_event_index = later.index
                    next_address = hex(later.address)
                    next_op = later.op
                    break
                if cell in defs:
                    action = "overwritten"
                    next_event_index = later.index
                    next_address = hex(later.address)
                    next_op = later.op
                    break
            if action == "terminal_dead" and cell.startswith("scalar.lfh"):
                action = "external_live_out"
            table[(event.index, cell)] = CellUse(
                cell=cell,
                action=action,
                next_event_index=next_event_index,
                next_address=next_address,
                next_op=next_op,
            )
    return table


def source_bundle_lengths() -> tuple[tuple, tuple[int, ...]]:
    bundles = EXP029.source_bundles()
    original = EXP029.original_hot_bytes()
    EXP029.BUILD_DIR.mkdir(parents=True, exist_ok=True)
    unpadded = EXP029.compile_mir(
        "mylm_hotloop_manifest_unpadded",
        EXP029.unpadded_mir(bundles),
        original,
    )
    lengths = EXP029.encoded_bundle_lengths(unpadded, len(bundles))
    return bundles, lengths


def events_by_address(events: tuple) -> dict[int, tuple]:
    grouped: dict[int, list] = {}
    for event in events:
        grouped.setdefault(event.address, []).append(event)
    return {address: tuple(items) for address, items in grouped.items()}


def event_row(event, def_next: dict[tuple[int, str], CellUse]) -> EventRow:
    def_cells = event_def_cells(event)
    return EventRow(
        index=event.index,
        address=hex(event.address),
        group=event.group,
        op=event.op,
        semantic=event.semantic,
        fragment=event.fragment,
        def_cells=def_cells,
        use_cells=event_use_cells(event),
        def_next=tuple(def_next[(event.index, cell)] for cell in def_cells),
    )


def classify_source_bundle(address: int, rows: tuple[EventRow, ...]) -> tuple[str, str]:
    if address in KNOWN_FAILED_SOURCE_MUTATIONS:
        return "known_failed_source_mutation", KNOWN_FAILED_SOURCE_MUTATIONS[address]
    semantics = {row.semantic for row in rows}
    if semantics & MUTATION_BLOCKING_SEMANTICS:
        return "blocked", "contains load/MAC/dequant/convert/pointer semantic"
    if not semantics <= SOURCE_CANDIDATE_SEMANTICS:
        return "blocked", "contains an unsupported non-memory semantic"
    if not rows:
        return "blocked", "empty source bundle"
    def_next = [item for row in rows for item in row.def_next]
    if not def_next:
        return "blocked", "has no destination cell to prove dead"
    if all(item.action in {"overwritten", "terminal_dead"} for item in def_next):
        return "weak_dead_def_candidate", "all destination cells are overwritten or terminal-dead before use"
    if any(item.action == "external_live_out" for item in def_next):
        return "blocked", "destination cell is live outside the modeled hot-loop range"
    return "blocked", "at least one destination cell is used before overwrite"


def classify_padding(address: int) -> tuple[str, str]:
    if address in TIMING_CRITICAL_PADDING:
        return "timing_critical_padding", "failed the experiment 032 self-move numeric gate"
    if address in BYTE_DIFF_CANARY_PADDING:
        return "byte_diff_canary_padding", "passed the experiment 032 self-move numeric gate"
    return "padding_unknown", "not covered by the experiment 032 NPU sweep"


def bundle_row(
    kind: str,
    address: int,
    length: int,
    raw: bytes,
    rows: tuple[EventRow, ...],
    mutation_class: str,
    reason: str,
) -> BundleRow:
    semantics = dedup([row.semantic for row in rows])
    ops = dedup([row.op for row in rows])
    def_cells: list[str] = []
    use_cells: list[str] = []
    for row in rows:
        def_cells.extend(row.def_cells)
        use_cells.extend(row.use_cells)
    groups = {row.group for row in rows}
    group = groups.pop() if len(groups) == 1 else None
    return BundleRow(
        kind=kind,
        address=hex(address),
        length=length,
        bytes_hex=raw[address - HOT_START : address - HOT_START + length].hex(),
        group=group,
        semantics=semantics,
        ops=ops,
        def_cells=dedup(def_cells),
        use_cells=dedup(use_cells),
        mutation_class=mutation_class,
        reason=reason,
        events=rows,
    )


def build_bundles() -> tuple[BundleRow, ...]:
    raw = EXP029.original_hot_bytes()
    events = EXP006.build_events()
    def_next = next_cell_uses(events)
    grouped = events_by_address(events)
    source_bundles, lengths = source_bundle_lengths()
    rows: list[BundleRow] = []
    cursor = 0
    for source_bundle, length in zip(source_bundles, lengths):
        target = source_bundle.address - HOT_START
        if target > cursor:
            address = HOT_START + cursor
            mutation_class, reason = classify_padding(address)
            rows.append(bundle_row("padding", address, target - cursor, raw, (), mutation_class, reason))
        event_rows = tuple(event_row(event, def_next) for event in grouped.get(source_bundle.address, ()))
        mutation_class, reason = classify_source_bundle(source_bundle.address, event_rows)
        rows.append(bundle_row("source", source_bundle.address, length, raw, event_rows, mutation_class, reason))
        cursor = target + length
    if cursor < HOT_END - HOT_START:
        address = HOT_START + cursor
        mutation_class, reason = classify_padding(address)
        rows.append(bundle_row("padding", address, HOT_END - HOT_START - cursor, raw, (), mutation_class, reason))
    return tuple(rows)


def build_manifest() -> ManifestRoot:
    bundles = build_bundles()
    counts = Counter(row.mutation_class for row in bundles)
    source_count = sum(1 for row in bundles if row.kind == "source")
    padding_count = sum(1 for row in bundles if row.kind == "padding")
    weak_source_candidates = tuple(row for row in bundles if row.mutation_class == "weak_dead_def_candidate")
    known_failed_source = tuple(row for row in bundles if row.mutation_class == "known_failed_source_mutation")
    padding_canaries = tuple(row for row in bundles if row.mutation_class == "byte_diff_canary_padding")
    timing_critical = tuple(row for row in bundles if row.mutation_class == "timing_critical_padding")
    hard_safe_source = tuple(row for row in bundles if row.mutation_class == "hard_safe_source_candidate")
    status = "passed" if len(padding_canaries) == 2 and len(timing_critical) == 1 else "failed"
    return ManifestRoot(
        experiment="033_q4nx_mir_bundle_mutation_manifest",
        summary=Summary(
            status=status,
            total_bundles=len(bundles),
            source_bundles=source_count,
            padding_bundles=padding_count,
            hard_safe_source_candidates=len(hard_safe_source),
            weak_dead_def_source_candidates=len(weak_source_candidates),
            timing_critical_padding=counts["timing_critical_padding"],
            byte_diff_canary_padding=counts["byte_diff_canary_padding"],
        ),
        weak_source_candidates=weak_source_candidates[:20],
        known_failed_source_mutations=known_failed_source,
        padding_canaries=padding_canaries,
        timing_critical_padding=timing_critical,
        bundles=bundles,
    )


def render_bundle_table(rows: tuple[BundleRow, ...]) -> list[str]:
    lines = ["| address | kind | length | class | ops | reason |", "| --- | --- | ---: | --- | --- | --- |"]
    for row in rows:
        lines.append(
            f"| `{row.address}` | `{row.kind}` | `{row.length}` | `{row.mutation_class}` | "
            f"`{','.join(row.ops)}` | {row.reason} |"
        )
    return lines


def render_report(manifest: ManifestRoot) -> str:
    summary = manifest.summary
    lines = [
        "# Q4NX MIR Bundle Mutation Manifest",
        "",
        f"Status: `{summary.status}`",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| total bundles | {summary.total_bundles} |",
        f"| source bundles | {summary.source_bundles} |",
        f"| padding bundles | {summary.padding_bundles} |",
        f"| hard-safe source candidates | {summary.hard_safe_source_candidates} |",
        f"| weak dead-def source candidates | {summary.weak_dead_def_source_candidates} |",
        f"| timing-critical padding | {summary.timing_critical_padding} |",
        f"| byte-diff canary padding | {summary.byte_diff_canary_padding} |",
        "",
        "## Padding Results",
        "",
    ]
    lines.extend(render_bundle_table(manifest.timing_critical_padding + manifest.padding_canaries))
    lines.extend(
        [
            "",
            "## Weak Source Candidates",
            "",
            "These are not ready for mutation. They only pass a conservative cell-level dead-destination check.",
            "",
        ]
    )
    lines.extend(render_bundle_table(manifest.weak_source_candidates))
    lines.extend(
        [
            "",
            "## Known Failed Source Mutation Gates",
            "",
            "These bundles may look removable under a narrow def/overwrite model, but real NPU numeric gates already failed.",
            "",
        ]
    )
    lines.extend(render_bundle_table(manifest.known_failed_source_mutations))
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The manifest confirms there are no hard-safe source bundle mutations yet.",
            "- `vups.4x` now uses the measured experiment 036/037 transfer model: `dm` updates all quadrants, `cml` updates the low half, and `cmh` updates the high half.",
            "- Experiments 034 and 038's failed source mutations are tracked as known failed gates instead of being hidden behind a broad implicit-destination-read rule.",
            "- `lfh*` writes are treated as external live-out because the hot-loop range ends before the phase body record emit is fully modeled.",
            "- Padding canary sites are useful for packaging tests, but not for performance work.",
            "- A real optimization now needs a stricter cell-level proof plus a cycle-preserving replacement bundle.",
            "- The next real source mutation needs a fuller instruction semantics model, not another candidate from the old weak list.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    try:
        manifest = build_manifest()
        MANIFEST.write_text(json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8")
        REPORT.write_text(render_report(manifest), encoding="utf-8")
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
