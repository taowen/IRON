#!/usr/bin/env python3
"""Compare MyLM and IRON main16 Q4NX disassembly shape."""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


DEFAULT_MYLM_DISASM = Path("/tmp/mylm_solidify_L31/disasm/c2r2.s")
DEFAULT_MYLM_BD_CSV = Path("/tmp/mylm_solidify_L31/layer_bd.csv")
DEFAULT_MYLM_PROGRAM_SEGMENTS = Path("/tmp/mylm_solidify_L31/programs/program_segments.tsv")
DEFAULT_MYLM_PROGRAM_IMAGES = Path("/tmp/mylm_solidify_L31/programs/program_images.tsv")
DEFAULT_IRON_BASELINE_OBJECT = Path("qwen3-layer/main_projection_q4nx_fast.o")
DEFAULT_IRON_FULL_DISASM = Path("qwen3-layer/build/main_core_2_2.after-direct-emit.s")
DEFAULT_LLVM_OBJDUMP = Path(".venv/lib/python3.12/site-packages/llvm-aie/bin/llvm-objdump")
DEFAULT_LLVM_SIZE = Path(".venv/lib/python3.12/site-packages/llvm-aie/bin/llvm-size")

ADDRESS_RE = re.compile(r"^\s*([0-9a-fA-F]+):\s+(.*)$")
LABEL_RE = re.compile(r"^\s*([0-9a-fA-F]+)\s+<([^>]+)>:$")
SECTION_RE = re.compile(r"^Disassembly of section (?P<section>[^:]+):$")
INT_RE = re.compile(r"^-?(?:0x[0-9a-fA-F]+|\d+)$")
JL_TARGET_RE = re.compile(r"\bjl\s+#0x([0-9a-fA-F]+)\b")

MAIN16_BD_ROLES = {
    0: "activation_ping",
    1: "activation_pong",
    2: "weight_ping",
    3: "weight_pong",
    4: "record_ping",
    5: "record_pong",
}

EVIDENCE_ADDRESSES = (
    0x1F0,
    0x238,
    0x23E,
    0x260,
    0x1D70,
    0x1D76,
    0x1D7E,
    0x1D90,
    0x1D9C,
    0x1DB0,
    0x1DB6,
    0x1DFA,
    0x1DFE,
    0x1E02,
)


@dataclass(frozen=True)
class RangeSpec:
    name: str
    start: int
    end: int


@dataclass(frozen=True)
class SectionSpec:
    name: str
    section: str


@dataclass(frozen=True)
class OpStats:
    name: str
    instruction_lines: int
    op_slots: int
    counts: Counter[str]


@dataclass(frozen=True)
class ObjectSize:
    name: str
    path: Path
    file_bytes: int
    text_bytes: int


@dataclass(frozen=True)
class DisasmLine:
    address: int
    text: str


@dataclass(frozen=True)
class BdEntry:
    bd_id: int
    role: str
    length: int
    base: int
    next_bd: int
    acq_id: int
    acq_val: int
    rel_id: int
    rel_val: int


@dataclass(frozen=True)
class CallSite:
    call_line: DisasmLine
    setup_lines: tuple[DisasmLine, ...]


@dataclass(frozen=True)
class ProgramSegment:
    col: int
    row: int
    offset: int
    bytes: int
    source: str


@dataclass(frozen=True)
class ProgramImage:
    col: int
    row: int
    bytes: int
    path: str


@dataclass(frozen=True)
class PhaseBodySpec:
    name: str
    start: int
    end: int
    records: int


@dataclass(frozen=True)
class PhaseBodySummary:
    name: str
    start: int
    end: int
    bytes: int
    records: int
    q4_calls: int
    jl: int
    acq: int
    rel: int
    jnz: int
    loop_register_lines: int
    instruction_lines: int
    op_slots: int


@dataclass(frozen=True)
class FullCoreSummary:
    name: str
    path: Path
    q4_calls: int
    jl: int
    acq: int
    rel: int
    jnz: int
    loop_register_lines: int
    instruction_lines: int
    op_slots: int


MYLM_RANGES = (
    RangeSpec("mylm_q4_microkernel", 0x1F0, 0x1850),
    RangeSpec("mylm_q4_hot_loop", 0x260, 0x1850),
    RangeSpec("mylm_qkv_body", 0x1870, 0x1E80),
)

MYLM_PHASE_BODIES = (
    PhaseBodySpec("Q/K/V", 0x1870, 0x1E80, 12),
    PhaseBodySpec("O", 0x1E80, 0x2490, 8),
    PhaseBodySpec("up/gate", 0x2490, 0x2AA0, 48),
    PhaseBodySpec("down", 0x2AA0, 0x30C0, 8),
    PhaseBodySpec("alternate", 0x30C0, 0x36D0, 304),
)

IRON_FAST_SECTIONS = (
    SectionSpec("iron_fast_q4_asm_zol", ".text.q4nx_chunk_accum_asm_zol"),
    SectionSpec("iron_fast_q4_function", ".text.q4nx_chunk_accum_slice_i32_fast"),
    SectionSpec("iron_fast_perf_fill", ".text.q4nx_fill_perf_inputs"),
)


def _instruction_text(payload: str) -> str:
    _before, sep, after = payload.partition("\t")
    if not sep:
        return ""
    return after.strip()


def _op_name(fragment: str) -> str:
    fragment = fragment.strip()
    if not fragment:
        return ""
    return fragment.split(None, 1)[0]


def _count_ops(name: str, lines: tuple[DisasmLine, ...]) -> OpStats:
    counts: Counter[str] = Counter()
    instruction_lines = 0
    op_slots = 0
    for line in lines:
        match = ADDRESS_RE.match(line.text)
        text = _instruction_text(match.group(2)) if match is not None else ""
        if not text:
            continue
        instruction_lines += 1
        for fragment in text.split(";"):
            op = _op_name(fragment)
            if op:
                counts[op] += 1
                op_slots += 1
    return OpStats(name=name, instruction_lines=instruction_lines, op_slots=op_slots, counts=counts)


def _disasm_lines(disasm: str) -> tuple[DisasmLine, ...]:
    lines: list[DisasmLine] = []
    for line in disasm.splitlines():
        match = ADDRESS_RE.match(line)
        if match is None:
            continue
        address = int(match.group(1), 16)
        lines.append(DisasmLine(address=address, text=line.strip()))
    return tuple(lines)


def _symbol_addresses(disasm: str) -> dict[str, int]:
    symbols: dict[str, int] = {}
    for line in disasm.splitlines():
        match = LABEL_RE.match(line)
        if match is None:
            continue
        symbols[match.group(2)] = int(match.group(1), 16)
    return symbols


def _mylm_range_lines(lines: tuple[DisasmLine, ...], spec: RangeSpec) -> tuple[DisasmLine, ...]:
    return tuple(line for line in lines if spec.start <= line.address < spec.end)


def _section_lines(disasm: str, spec: SectionSpec) -> tuple[DisasmLine, ...]:
    selected: list[DisasmLine] = []
    in_section = False
    for line in disasm.splitlines():
        section_match = SECTION_RE.match(line)
        if section_match is not None:
            in_section = section_match.group("section") == spec.section
            continue
        if not in_section:
            continue
        match = ADDRESS_RE.match(line)
        if match is not None:
            selected.append(DisasmLine(address=int(match.group(1), 16), text=line.strip()))
    return tuple(selected)


def _parse_int(text: str) -> int:
    if not INT_RE.match(text):
        raise ValueError(f"bad integer field: {text}")
    return int(text, 0)


def _parse_fields(raw_fields: str) -> dict[str, int | str]:
    parsed: dict[str, int | str] = {}
    for item in raw_fields.split():
        key, sep, raw_value = item.partition("=")
        if not sep:
            raise ValueError(f"bad BD field item: {item}")
        parsed[key] = _parse_int(raw_value) if INT_RE.match(raw_value) else raw_value
    return parsed


def _load_bd_entries(path: Path, tile: str) -> tuple[BdEntry, ...]:
    entries: list[BdEntry] = []
    with path.open(newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            if row["tile"] != tile:
                continue
            bd_id = int(row["bd"])
            if bd_id not in MAIN16_BD_ROLES:
                continue
            fields = _parse_fields(row["fields"])
            entries.append(
                BdEntry(
                    bd_id=bd_id,
                    role=MAIN16_BD_ROLES[bd_id],
                    length=int(fields["len"]),
                    base=int(fields["base"]),
                    next_bd=int(fields["next_bd"]),
                    acq_id=int(fields["acq_id"]),
                    acq_val=int(fields["acq_val"]),
                    rel_id=int(fields["rel_id"]),
                    rel_val=int(fields["rel_val"]),
                )
            )
    return tuple(sorted(entries, key=lambda entry: entry.bd_id))


def _load_program_segments(path: Path, col: int, row: int) -> tuple[ProgramSegment, ...]:
    segments: list[ProgramSegment] = []
    with path.open(newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for item in reader:
            item_col = int(item["col"])
            item_row = int(item["row"])
            if item_col != col or item_row != row:
                continue
            segments.append(
                ProgramSegment(
                    col=item_col,
                    row=item_row,
                    offset=int(item["program_offset"], 0),
                    bytes=int(item["bytes"]),
                    source=item["source"],
                )
            )
    return tuple(sorted(segments, key=lambda segment: segment.offset))


def _load_program_images(path: Path) -> tuple[ProgramImage, ...]:
    images: list[ProgramImage] = []
    with path.open(newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for item in reader:
            images.append(
                ProgramImage(
                    col=int(item["col"]),
                    row=int(item["row"]),
                    bytes=int(item["bytes"]),
                    path=item["path"],
                )
            )
    return tuple(images)


def _main16_images(images: tuple[ProgramImage, ...]) -> tuple[ProgramImage, ...]:
    return tuple(
        sorted(
            (image for image in images if 2 <= image.col <= 5 and 2 <= image.row <= 5),
            key=lambda image: (image.col, image.row),
        )
    )


def _call_sites(lines: tuple[DisasmLine, ...]) -> tuple[CallSite, ...]:
    sites: list[CallSite] = []
    for idx, line in enumerate(lines):
        if "jl\t#0x1f0" in line.text:
            sites.append(CallSite(call_line=line, setup_lines=lines[idx + 1 : idx + 3]))
    return tuple(sites)


def _jump_targets(lines: tuple[DisasmLine, ...]) -> tuple[int, ...]:
    targets: list[int] = []
    for line in lines:
        match = JL_TARGET_RE.search(line.text)
        if match is not None:
            targets.append(int(match.group(1), 16))
    return tuple(targets)


def _loop_register_line_count(lines: tuple[DisasmLine, ...]) -> int:
    return sum(
        1
        for line in lines
        if re.search(r"\b(?:lc|ls|le)\b", line.text) is not None
    )


def _phase_body_summaries(
    lines: tuple[DisasmLine, ...],
    specs: tuple[PhaseBodySpec, ...],
    q4_targets: tuple[int, ...],
) -> tuple[PhaseBodySummary, ...]:
    summaries: list[PhaseBodySummary] = []
    q4_target_set = set(q4_targets)
    for spec in specs:
        body_lines = tuple(line for line in lines if spec.start <= line.address < spec.end)
        stats = _count_ops(spec.name, body_lines)
        q4_calls = sum(1 for target in _jump_targets(body_lines) if target in q4_target_set)
        summaries.append(
            PhaseBodySummary(
                name=spec.name,
                start=spec.start,
                end=spec.end,
                bytes=spec.end - spec.start,
                records=spec.records,
                q4_calls=q4_calls,
                jl=stats.counts["jl"],
                acq=stats.counts["acq"],
                rel=stats.counts["rel"],
                jnz=stats.counts["jnz"],
                loop_register_lines=_loop_register_line_count(body_lines),
                instruction_lines=stats.instruction_lines,
                op_slots=stats.op_slots,
            )
        )
    return tuple(summaries)


def _q4_symbol_targets(symbols: dict[str, int]) -> tuple[int, ...]:
    return tuple(
        sorted(
            address
            for name, address in symbols.items()
            if "q4nx_chunk_accum" in name
        )
    )


def _full_core_summary(name: str, path: Path, disasm: str) -> FullCoreSummary:
    lines = _disasm_lines(disasm)
    stats = _count_ops(name, lines)
    q4_targets = set(_q4_symbol_targets(_symbol_addresses(disasm)))
    q4_calls = sum(1 for target in _jump_targets(lines) if target in q4_targets)
    return FullCoreSummary(
        name=name,
        path=path,
        q4_calls=q4_calls,
        jl=stats.counts["jl"],
        acq=stats.counts["acq"],
        rel=stats.counts["rel"],
        jnz=stats.counts["jnz"],
        loop_register_lines=_loop_register_line_count(lines),
        instruction_lines=stats.instruction_lines,
        op_slots=stats.op_slots,
    )


def _line_map(lines: tuple[DisasmLine, ...]) -> dict[int, DisasmLine]:
    return {line.address: line for line in lines}


def _exact_line(lines_by_addr: dict[int, DisasmLine], address: int) -> str:
    line = lines_by_addr.get(address)
    if line is None:
        return f"0x{address:x}: missing"
    return line.text


def _load_iron_disasm(llvm_objdump: Path, iron_object: Path) -> str:
    result = subprocess.run(
        (str(llvm_objdump), "-d", "--demangle", str(iron_object)),
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _load_object_size(llvm_size: Path, name: str, path: Path) -> ObjectSize:
    result = subprocess.run(
        (str(llvm_size), "-A", str(path)),
        check=True,
        capture_output=True,
        text=True,
    )
    text_bytes = 0
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith(".text"):
            text_bytes += int(parts[1], 0)
    return ObjectSize(
        name=name,
        path=path,
        file_bytes=path.stat().st_size,
        text_bytes=text_bytes,
    )


def _print_stats(stats: OpStats, top: int) -> None:
    if stats.instruction_lines == 0:
        raise ValueError(f"empty disassembly range: {stats.name}")
    print(f"[{stats.name}]")
    print(f"  instruction_lines={stats.instruction_lines} op_slots={stats.op_slots}")
    for op, count in stats.counts.most_common(top):
        print(f"  {op}={count}")


def _markdown_table(headers: tuple[str, ...], rows: tuple[tuple[str, ...], ...]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _render_markdown(
    mylm_disasm: Path,
    mylm_bd_csv: Path,
    mylm_program_segments: Path,
    mylm_program_images: Path,
    object_sizes: tuple[ObjectSize, ...],
    bd_entries: tuple[BdEntry, ...],
    program_segments: tuple[ProgramSegment, ...],
    main16_images: tuple[ProgramImage, ...],
    phase_summaries: tuple[PhaseBodySummary, ...],
    full_core_summary: FullCoreSummary,
    call_sites: tuple[CallSite, ...],
    evidence: tuple[str, ...],
    stats: tuple[OpStats, ...],
    top: int,
) -> str:
    bd_rows = tuple(
        (
            entry.role,
            f"bd{entry.bd_id}",
            str(entry.length),
            f"0x{entry.base:x}",
            f"bd{entry.next_bd}",
            f"L{entry.acq_id}:{entry.acq_val}",
            f"L{entry.rel_id}:{entry.rel_val}",
        )
        for entry in bd_entries
    )
    segment_rows = tuple(
        (
            f"0x{segment.offset:x}",
            str(segment.bytes),
            Path(segment.source).name,
        )
        for segment in program_segments
    )
    main16_image_bytes = tuple(sorted({image.bytes for image in main16_images}))
    image_summary = (
        f"{len(main16_images)} main16 images, bytes={main16_image_bytes[0]}"
        if len(main16_image_bytes) == 1
        else f"{len(main16_images)} main16 images, byte_sizes={main16_image_bytes}"
    )
    call_rows = tuple(
        (
            f"0x{site.call_line.address:x}",
            "<br>".join(line.text for line in site.setup_lines),
        )
        for site in call_sites
    )
    phase_rows = tuple(
        (
            item.name,
            f"0x{item.start:x}-0x{item.end:x}",
            str(item.bytes),
            str(item.records),
            str(item.q4_calls),
            str(item.jl),
            str(item.acq),
            str(item.rel),
            str(item.jnz),
            str(item.loop_register_lines),
        )
        for item in phase_summaries
    )
    full_core_rows = (
        (
            full_core_summary.name,
            str(full_core_summary.q4_calls),
            str(full_core_summary.jl),
            str(full_core_summary.acq),
            str(full_core_summary.rel),
            str(full_core_summary.jnz),
            str(full_core_summary.loop_register_lines),
            str(full_core_summary.instruction_lines),
            str(full_core_summary.op_slots),
            str(full_core_summary.path),
        ),
    )
    stat_rows = tuple(
        (
            item.name,
            str(item.instruction_lines),
            str(item.op_slots),
            ", ".join(f"{op}={count}" for op, count in item.counts.most_common(top)),
        )
        for item in stats
    )
    size_rows = tuple(
        (
            item.name,
            str(item.file_bytes),
            str(item.text_bytes),
            str(item.path),
        )
        for item in object_sizes
    )
    lines = [
        "# Main16 Q4NX MyLM Secret Report",
        "",
        "This report is generated from the local MyLM disassembly and the selected IRON role objects.",
        "",
        "## Inputs",
        "",
        f"- MyLM disassembly: `{mylm_disasm}`",
        f"- MyLM BD CSV: `{mylm_bd_csv}`",
        f"- MyLM program segments: `{mylm_program_segments}`",
        f"- MyLM program images: `{mylm_program_images}`",
        "",
        "## MyLM Raw Program Layout",
        "",
        f"- {image_summary}",
        "",
    ]
    lines.extend(_markdown_table(("offset", "bytes", "source"), segment_rows))
    lines.extend(
        [
            "",
            "The c2r2 program is a raw segmented core program. The Q4NX microkernel is loaded once at `0x1f0`; the visible fused phase bodies call into it instead of embedding separate C++-style hot loops per phase.",
            "",
            "## MyLM Phase Body Shape",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            ("phase", "range", "bytes", "records", "q4 calls", "jl", "acq", "rel", "jnz", "lc/ls/le lines"),
            phase_rows,
        )
    )
    lines.extend(
        [
            "",
            "Each normal phase body has one scheduled `jl #0x1f0` into the shared Q4NX microkernel. The compact-record replay count is encoded by the body entry setup, not by cloning the per-chunk lock choreography.",
            "",
            "## MyLM Main16 BD Contract",
            "",
        ]
    )
    lines.extend(_markdown_table(("role", "bd", "len", "base", "next", "acquire", "release"), bd_rows))
    lines.extend(
        [
            "",
            "The outer ABI matches the active IRON design: DMA0 activation, DMA1 Q4NX weight, and a 17-dword compact record output. The performance gap is inside the core program consuming that ABI.",
            "",
            "## Q4NX Call-Site Evidence",
            "",
        ]
    )
    lines.extend(_markdown_table(("call", "branch-slot setup"), call_rows))
    lines.extend(
        [
            "",
            "The arguments are prepared in the branch-slot window after `jl #0x1f0`; MyLM is using a raw scheduled core body, not a normal C++ call boundary.",
            "",
            "## IRON Full Main Core Shape",
            "",
        ]
    )
    lines.extend(
        _markdown_table(
            (
                "core",
                "q4 calls",
                "jl",
                "acq",
                "rel",
                "jnz",
                "lc/ls/le lines",
                "instruction lines",
                "op slots",
                "path",
            ),
            full_core_rows,
        )
    )
    lines.extend(
        [
            "",
            "This is the active full-layer main core shape produced by MLIR-AIE. It is the gate that must collapse toward the MyLM phase-body shape before a 5x main16 improvement is credible.",
            "",
            "## Fixed Address Evidence",
            "",
            "```text",
        ]
    )
    lines.extend(evidence)
    lines.extend(
        [
            "```",
            "",
            "## Opcode Shape",
            "",
        ]
    )
    lines.extend(_markdown_table(("range", "instruction lines", "op slots", "top ops"), stat_rows))
    lines.extend(
        [
            "",
            "## IRON Object Size",
            "",
        ]
    )
    lines.extend(_markdown_table(("object", "file bytes", "text bytes", "path"), size_rows))
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            "MyLM's main16 advantage is a raw zero-overhead Q4NX loop with scheduled vector dequant/MAC/output packing and caller-side branch-slot setup. The active IRON kernel is numerically correct, but the full main core still exposes per-chunk lock/control structure that the compiler does not collapse into the same phase-body schedule.",
            "",
            "The next performance step should generate a fixed-schedule main16 core body for the existing DMA0/DMA1/record ABI while preserving the verified Q4NX numerical contract `int4 * scale + offset`. Small Python generator cleanup cannot close this gap by itself.",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mylm-disasm", type=Path, default=DEFAULT_MYLM_DISASM)
    parser.add_argument("--mylm-bd-csv", type=Path, default=DEFAULT_MYLM_BD_CSV)
    parser.add_argument("--mylm-program-segments", type=Path, default=DEFAULT_MYLM_PROGRAM_SEGMENTS)
    parser.add_argument("--mylm-program-images", type=Path, default=DEFAULT_MYLM_PROGRAM_IMAGES)
    parser.add_argument("--iron-baseline-object", type=Path, default=DEFAULT_IRON_BASELINE_OBJECT)
    parser.add_argument("--iron-full-disasm", type=Path, default=DEFAULT_IRON_FULL_DISASM)
    parser.add_argument("--llvm-objdump", type=Path, default=DEFAULT_LLVM_OBJDUMP)
    parser.add_argument("--llvm-size", type=Path, default=DEFAULT_LLVM_SIZE)
    parser.add_argument("--markdown", type=Path, default=None)
    parser.add_argument("--top", type=int, default=16)
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    if args.top <= 0:
        raise ValueError("--top must be positive")

    mylm_disasm = args.mylm_disasm.read_text()
    mylm_lines = _disasm_lines(mylm_disasm)
    mylm_by_addr = _line_map(mylm_lines)
    phase_summaries = _phase_body_summaries(mylm_lines, MYLM_PHASE_BODIES, (0x1F0,))
    iron_full_disasm = args.iron_full_disasm.read_text()
    full_core_summary = _full_core_summary("iron_full_main_core", args.iron_full_disasm, iron_full_disasm)
    bd_entries = _load_bd_entries(args.mylm_bd_csv, "c2r2")
    program_segments = _load_program_segments(args.mylm_program_segments, 2, 2)
    main16_images = _main16_images(_load_program_images(args.mylm_program_images))
    call_sites = _call_sites(mylm_lines)
    iron_objects = (
        ("baseline", args.iron_baseline_object),
    )
    iron_disasms = tuple(
        (name, path, _load_iron_disasm(args.llvm_objdump, path))
        for name, path in iron_objects
    )
    object_sizes = tuple(
        _load_object_size(args.llvm_size, name, path)
        for name, path in iron_objects
    )

    print("main16 Q4NX disassembly comparison")
    print(f"mylm_disasm={args.mylm_disasm}")
    print(f"mylm_bd_csv={args.mylm_bd_csv}")
    print(f"mylm_program_segments={args.mylm_program_segments}")
    print(f"mylm_program_images={args.mylm_program_images}")
    for name, path in iron_objects:
        print(f"iron_{name}_object={path}")
    print(f"iron_full_disasm={args.iron_full_disasm}")
    print()

    image_bytes = tuple(sorted({image.bytes for image in main16_images}))
    print("[mylm_main16_raw_program_layout]")
    print(f"  main16_images={len(main16_images)} byte_sizes={image_bytes}")
    for segment in program_segments:
        print(f"  offset=0x{segment.offset:x} bytes={segment.bytes} source={Path(segment.source).name}")
    print()

    print("[mylm_phase_body_shape]")
    for item in phase_summaries:
        print(
            f"  {item.name}: range=0x{item.start:x}-0x{item.end:x} bytes={item.bytes} "
            f"records={item.records} q4_calls={item.q4_calls} jl={item.jl} "
            f"acq={item.acq} rel={item.rel} jnz={item.jnz} "
            f"lc_ls_le_lines={item.loop_register_lines} "
            f"instruction_lines={item.instruction_lines} op_slots={item.op_slots}"
        )
    print()

    print("[iron_full_main_core_shape]")
    print(
        f"  {full_core_summary.name}: q4_calls={full_core_summary.q4_calls} "
        f"jl={full_core_summary.jl} acq={full_core_summary.acq} "
        f"rel={full_core_summary.rel} jnz={full_core_summary.jnz} "
        f"lc_ls_le_lines={full_core_summary.loop_register_lines} "
        f"instruction_lines={full_core_summary.instruction_lines} "
        f"op_slots={full_core_summary.op_slots} path={full_core_summary.path}"
    )
    print()

    print("[mylm_main16_bd_contract]")
    for entry in bd_entries:
        print(
            f"  {entry.role}: bd={entry.bd_id} len={entry.length} base=0x{entry.base:x} "
            f"next={entry.next_bd} acq=L{entry.acq_id}:{entry.acq_val} rel=L{entry.rel_id}:{entry.rel_val}"
        )
    print()

    print("[mylm_q4_call_sites]")
    for site in call_sites:
        setup = " | ".join(line.text for line in site.setup_lines)
        print(f"  {site.call_line.text} -> {setup}")
    print()

    evidence = tuple(_exact_line(mylm_by_addr, address) for address in EVIDENCE_ADDRESSES)
    print("[mylm_fixed_address_evidence]")
    for line in evidence:
        print(f"  {line}")
    print()

    all_stats: list[OpStats] = []
    for spec in MYLM_RANGES:
        stats = _count_ops(spec.name, _mylm_range_lines(mylm_lines, spec))
        all_stats.append(stats)
        _print_stats(stats, args.top)
        print()
    for name, _path, disasm in iron_disasms:
        for spec in IRON_FAST_SECTIONS:
            stats = _count_ops(f"iron_{name}_{spec.name}", _section_lines(disasm, spec))
            all_stats.append(stats)
            _print_stats(stats, args.top)
            print()

    print("[iron_object_size]")
    for size in object_sizes:
        print(
            f"  {size.name}: file_bytes={size.file_bytes} "
            f"text_bytes={size.text_bytes} path={size.path}"
        )
    print()

    if args.markdown is not None:
        args.markdown.write_text(
            _render_markdown(
                args.mylm_disasm,
                args.mylm_bd_csv,
                args.mylm_program_segments,
                args.mylm_program_images,
                object_sizes,
                bd_entries,
                program_segments,
                main16_images,
                phase_summaries,
                full_core_summary,
                call_sites,
                evidence,
                tuple(all_stats),
                args.top,
            )
        )
        print(f"wrote markdown={args.markdown}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
