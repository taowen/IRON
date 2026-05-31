#!/usr/bin/env python3
"""Summarize the reverse-engineered MyLM main16 Q4NX kernel shape."""

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DISASM = Path("/tmp/mylm_qwen3_layer_L31_deep/disasm/c2r2.s")
ADDRESS_RE = re.compile(r"^\s*([0-9a-fA-F]+):\s+(.*)$")
LC_RE = re.compile(r"\bmova\s+lc,\s*#0x([0-9a-fA-F]+)\b")
OP_RE = re.compile(r"\b[a-z][a-z0-9]*(?:\.[a-z0-9]+)*\b")
ACTIVATION_LOAD_RE = re.compile(r"\bvldb\s+x11,\s*\[p1\],\s*#0x40\b")
ACTIVATION_REWIND_RE = re.compile(r"\bpaddb\s+\[p1\],\s*#-0x200\b")
SCRATCH_LOAD_RE = re.compile(r"\blda\.s16\s+r7,\s*\[p3\],\s*#0x2\b")
SCRATCH_REWIND_RE = re.compile(r"\badd\.nc\s+p3,\s*r21,\s*#-0x10\b")
X11_EXTRACT_RE = re.compile(r"\bvextbcst\.16\s+\w+,\s*x11,\s*#0x([0-9a-fA-F]+)\b")
PHASE_ACTIVATION_LOAD_RE = re.compile(r"\bvlda\.conv\.fp32\.bf16\s+\w+,\s*\[p3(?:,?\s*#0x[0-9a-fA-F]+)?\]")
PHASE_SCRATCH_STORE_RE = re.compile(r"\bst\.s16\s+\w+,\s*\[p2(?:\]|\s*,)")
PHASE_SCALAR_EXTRACT_RE = re.compile(r"\bvextract\.16\s+\w+,\s*x\d+,\s*#0x0\b")
HOT_START = 0x260
HOT_END = 0x1850
PHASE_BODY_START = 0x1870
PHASE_BODY_END = 0x1E80
Q4_CALL_SLOT = (0x1D70, 0x1D76, 0x1D7E)


Q4_ROWS = 32
Q4_COLS = 256
Q4_GROUPS = 8
Q4_GROUP_COLS = 32
VECTOR_ROWS = 16
CORRECTION_MACS_PER_PASS = Q4_GROUPS


@dataclass(frozen=True)
class DisasmLine:
    address: int
    text: str


@dataclass(frozen=True)
class ActivationLaneGroup:
    load_address: int
    lanes: tuple[int, ...]

    @property
    def complete(self) -> bool:
        return tuple(sorted(self.lanes)) == tuple(range(Q4_GROUP_COLS))


@dataclass(frozen=True)
class PhaseScratchSummary:
    activation_loads: tuple[int, ...]
    scratch_stores: tuple[int, ...]
    scalar_extracts: tuple[int, ...]
    q4_call_slot_lines: tuple[DisasmLine, ...]
    vadd: int
    vshift: int


@dataclass(frozen=True)
class HotGroupSummary:
    index: int
    start: int
    end: int
    lanes: tuple[int, ...]
    vmac: int
    vextbcst16: int
    vbcst16: int
    lda_s16: int
    vunpack: int
    vups: int
    vconv_bf16_fp32: int
    vst: int


@dataclass(frozen=True)
class KernelSummary:
    loop_count: int
    static_ops: Counter[str]
    phase_scratch: PhaseScratchSummary
    activation_loads: tuple[int, ...]
    activation_rewinds: tuple[int, ...]
    activation_lane_groups: tuple[ActivationLaneGroup, ...]
    hot_groups: tuple[HotGroupSummary, ...]
    scratch_loads: tuple[int, ...]
    scratch_rewinds: tuple[int, ...]


def parse_disasm(path: Path) -> tuple[DisasmLine, ...]:
    lines: list[DisasmLine] = []
    for raw_line in path.read_text().splitlines():
        match = ADDRESS_RE.match(raw_line)
        if match is None:
            continue
        lines.append(
            DisasmLine(
                address=int(match.group(1), 16),
                text=raw_line.strip(),
            )
        )
    return tuple(lines)


def ops_for_line(line: DisasmLine) -> tuple[str, ...]:
    payload = line.text.split("\t", 1)[1] if "\t" in line.text else ""
    return tuple(match.group(0) for match in OP_RE.finditer(payload))


def hot_lines(lines: tuple[DisasmLine, ...]) -> tuple[DisasmLine, ...]:
    return tuple(line for line in lines if HOT_START <= line.address < HOT_END)


def phase_body_lines(lines: tuple[DisasmLine, ...]) -> tuple[DisasmLine, ...]:
    return tuple(line for line in lines if PHASE_BODY_START <= line.address < PHASE_BODY_END)


def loop_count(lines: tuple[DisasmLine, ...]) -> int:
    for line in lines:
        if line.address != 0x1F0:
            continue
        match = LC_RE.search(line.text)
        if match is None:
            continue
        return int(match.group(1), 16)
    raise ValueError("failed to find lc setup at 0x1f0")


def activation_lane_groups(lines: tuple[DisasmLine, ...]) -> tuple[ActivationLaneGroup, ...]:
    groups: list[tuple[int, list[int]]] = []
    current_lanes: list[int] | None = None
    for line in hot_lines(lines):
        if ACTIVATION_LOAD_RE.search(line.text):
            current_lanes = []
            groups.append((line.address, current_lanes))
        if current_lanes is None:
            continue
        for match in X11_EXTRACT_RE.finditer(line.text):
            current_lanes.append(int(match.group(1), 16))
    return tuple(
        ActivationLaneGroup(load_address=address, lanes=tuple(lanes))
        for address, lanes in groups
    )


def phase_scratch_summary(lines: tuple[DisasmLine, ...]) -> PhaseScratchSummary:
    body = phase_body_lines(lines)
    by_address = {line.address: line for line in body}
    op_counts = Counter()
    for line in body:
        op_counts.update(ops_for_line(line))
    return PhaseScratchSummary(
        activation_loads=tuple(
            line.address for line in body if PHASE_ACTIVATION_LOAD_RE.search(line.text)
        ),
        scratch_stores=tuple(
            line.address for line in body if PHASE_SCRATCH_STORE_RE.search(line.text)
        ),
        scalar_extracts=tuple(
            line.address for line in body if PHASE_SCALAR_EXTRACT_RE.search(line.text)
        ),
        q4_call_slot_lines=tuple(
            by_address[address] for address in Q4_CALL_SLOT if address in by_address
        ),
        vadd=op_counts["vadd.f"],
        vshift=op_counts["vshift"],
    )


def hot_group_summaries(
    lines: tuple[DisasmLine, ...],
    lane_groups: tuple[ActivationLaneGroup, ...],
) -> tuple[HotGroupSummary, ...]:
    starts = tuple(group.load_address for group in lane_groups)
    groups: list[HotGroupSummary] = []
    for index, group in enumerate(lane_groups):
        end = starts[index + 1] if index + 1 < len(starts) else HOT_END
        group_lines = tuple(
            line for line in lines if group.load_address <= line.address < end
        )
        ops = Counter()
        for line in group_lines:
            ops.update(ops_for_line(line))
        groups.append(
            HotGroupSummary(
                index=index,
                start=group.load_address,
                end=end,
                lanes=group.lanes,
                vmac=ops["vmac.f"],
                vextbcst16=ops["vextbcst.16"],
                vbcst16=ops["vbcst.16"],
                lda_s16=ops["lda.s16"],
                vunpack=ops["vunpack"],
                vups=ops["vups.4x"],
                vconv_bf16_fp32=ops["vconv.bf16.fp32"],
                vst=ops["vst"],
            )
        )
    return tuple(groups)


def summarize(lines: tuple[DisasmLine, ...]) -> KernelSummary:
    body = hot_lines(lines)
    static_ops: Counter[str] = Counter()
    for line in body:
        static_ops.update(ops_for_line(line))
    lane_groups = activation_lane_groups(lines)

    return KernelSummary(
        loop_count=loop_count(lines),
        static_ops=static_ops,
        phase_scratch=phase_scratch_summary(lines),
        activation_loads=tuple(
            line.address for line in body if ACTIVATION_LOAD_RE.search(line.text)
        ),
        activation_rewinds=tuple(
            line.address for line in body if ACTIVATION_REWIND_RE.search(line.text)
        ),
        activation_lane_groups=lane_groups,
        hot_groups=hot_group_summaries(body, lane_groups),
        scratch_loads=tuple(
            line.address for line in body if SCRATCH_LOAD_RE.search(line.text)
        ),
        scratch_rewinds=tuple(
            line.address for line in body if SCRATCH_REWIND_RE.search(line.text)
        ),
    )


def format_addresses(addresses: tuple[int, ...]) -> str:
    return ", ".join(f"0x{address:x}" for address in addresses)


def render(summary: KernelSummary) -> str:
    loop_count = summary.loop_count
    static_vmac = summary.static_ops["vmac.f"]
    static_vext = summary.static_ops["vextbcst.16"]
    static_scratch_loads = len(summary.scratch_loads)
    static_correction_broadcasts = summary.static_ops["vbcst.16"]

    lane_group_count = len(summary.activation_lane_groups)
    complete_lane_groups = sum(group.complete for group in summary.activation_lane_groups)
    expected_static_extracts = Q4_GROUPS * Q4_GROUP_COLS
    expected_main_macs = (Q4_ROWS // VECTOR_ROWS) * Q4_GROUPS * Q4_GROUP_COLS
    expected_correction_macs = (Q4_ROWS // VECTOR_ROWS) * CORRECTION_MACS_PER_PASS
    expected_total_macs = expected_main_macs + expected_correction_macs
    phase = summary.phase_scratch

    lines = [
        "mylm_main16_q4nx_kernel:",
        f"  hot_loop=0x{HOT_START:x}..0x{HOT_END:x}",
        f"  lc={loop_count}",
        "  semantic_shape:",
        f"    q4_chunk={Q4_ROWS}x{Q4_COLS}",
        f"    output_lane_passes={Q4_ROWS // VECTOR_ROWS}",
        f"    vector_rows={VECTOR_ROWS}",
        f"    activation_groups={Q4_GROUPS}",
        f"    activation_lanes_per_group={Q4_GROUP_COLS}",
        "  static_hot_loop:",
        f"    vmac.f={static_vmac}",
        f"    vextbcst.16={static_vext}",
        f"    vbcst.16={static_correction_broadcasts}",
        f"    lda.s16={static_scratch_loads}",
        f"    vunpack={summary.static_ops['vunpack']}",
        f"    vups={summary.static_ops['vups.4x']}",
        f"    vst={summary.static_ops['vst']}",
        "  dynamic_q4nx_chunk:",
        f"    vmac.f={static_vmac * loop_count}",
        f"    vextbcst.16={static_vext * loop_count}",
        f"    lda.s16={static_scratch_loads * loop_count}",
        f"    vbcst.16={static_correction_broadcasts * loop_count}",
        "  first_principles_check:",
        f"    main_macs={expected_main_macs}",
        f"    correction_macs={expected_correction_macs}",
        f"    total_macs={expected_total_macs}",
        f"    matches_dynamic_vmac={expected_total_macs == static_vmac * loop_count}",
        f"    expected_static_activation_extracts={expected_static_extracts}",
        f"    matches_static_vextbcst={expected_static_extracts == static_vext}",
        "  activation_stream:",
        f"    x11_loads={format_addresses(summary.activation_loads)}",
        f"    p1_rewinds={format_addresses(summary.activation_rewinds)}",
        f"    lane_groups={lane_group_count}",
        f"    complete_lane_groups={complete_lane_groups}",
        "  correction_scratch:",
        f"    p3_loads={format_addresses(summary.scratch_loads)}",
        f"    p3_rewinds={format_addresses(summary.scratch_rewinds)}",
        "  phase_body_group_sum_producer:",
        f"    activation_vector_loads={len(phase.activation_loads)} @ {format_addresses(phase.activation_loads)}",
        f"    scalar_extracts={len(phase.scalar_extracts)} @ {format_addresses(phase.scalar_extracts)}",
        f"    scratch_stores={len(phase.scratch_stores)} @ {format_addresses(phase.scratch_stores)}",
        f"    vadd.f={phase.vadd}",
        f"    vshift={phase.vshift}",
        f"    produces_exactly_8_group_sums={len(phase.scratch_stores) == Q4_GROUPS}",
        "    q4_call_slot:",
    ]
    lines.extend(f"      {line.text}" for line in phase.q4_call_slot_lines)
    for group_index, group in enumerate(summary.activation_lane_groups):
        prefix = f"  activation_group_{group_index}:"
        lanes = ",".join(str(lane) for lane in group.lanes)
        lines.extend(
            [
                prefix,
                f"    load=0x{group.load_address:x}",
                f"    lanes={lanes}",
                f"    complete_0_31={group.complete}",
            ]
        )
    lines.append("  hot_group_instruction_shape:")
    for group in summary.hot_groups:
        lines.append(
            "    "
            f"group={group.index} range=0x{group.start:x}..0x{group.end:x} "
            f"lanes={len(group.lanes)} "
            f"vmac.f={group.vmac} "
            f"vextbcst.16={group.vextbcst16} "
            f"vbcst.16={group.vbcst16} "
            f"lda.s16={group.lda_s16} "
            f"vunpack={group.vunpack} "
            f"vups={group.vups} "
            f"vconv.bf16.fp32={group.vconv_bf16_fp32} "
            f"vst={group.vst}"
        )
    return "\n".join(lines) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disasm", type=Path, default=DEFAULT_DISASM)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    print(render(summarize(parse_disasm(args.disasm))), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
