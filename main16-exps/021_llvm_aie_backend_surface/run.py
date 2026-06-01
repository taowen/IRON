#!/usr/bin/env python3
"""Inventory useful llvm-aie backend surfaces for the main16 Q4NX route."""

from __future__ import annotations

import json
import re
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
LLVM_AIE_ROOT = Path("/var/home/taowen/projects/llvm-aie")
LLVM_TARGET_AIE = LLVM_AIE_ROOT / "llvm/lib/Target/AIE"
MANIFEST = EXPERIMENT_DIR / "llvm_aie_backend_surface.json"
REPORT = EXPERIMENT_DIR / "llvm_aie_backend_surface.md"


@dataclass(frozen=True)
class SourceNeedle:
    name: str
    relative_path: str
    pattern: str
    reason: str


@dataclass(frozen=True)
class SourceHit:
    name: str
    path: str
    line: int
    matches: int
    pattern: str
    reason: str
    excerpt: list[str]


@dataclass(frozen=True)
class FileMetric:
    name: str
    path: str
    metrics: dict[str, int]
    reason: str


@dataclass(frozen=True)
class BackendSurface:
    name: str
    paths: list[str]
    use: str
    risk: str


NEEDLES = (
    SourceNeedle(
        name="aie2p vextbcst.16 instruction",
        relative_path="llvm/lib/Target/AIE/aie2p/AIE2PGenInstrInfo.td",
        pattern=r"def VEXTBCST_16_vec_extract_broadcast_imm",
        reason="Defines the exact AIE2P machine instruction spelling and operands for vextbcst.16.",
    ),
    SourceNeedle(
        name="aie2p vmac.f bf16 x/x instruction",
        relative_path="llvm/lib/Target/AIE/aie2p/AIE2PGenInstrInfo.td",
        pattern=r"def VMAC_f_vmac_bf_vmul_bf_core_X_X",
        reason="Defines one of the bf16 vmac.f forms used by Q4NX-style MAC bodies.",
    ),
    SourceNeedle(
        name="aie2p vups.4x x-to-d instruction",
        relative_path="llvm/lib/Target/AIE/aie2p/AIE2PGenInstrInfo.td",
        pattern=r"def VUPS_4x_mv_ups_x2d_upsSign0",
        reason="Defines the vups.4x form closest to the MyLM dequant pipeline.",
    ),
    SourceNeedle(
        name="aie2p vups.4x pattern",
        relative_path="llvm/lib/Target/AIE/aie2p/AIE2PInstrPatterns.td",
        pattern=r"VUPS_4x_mv_ups_x2d_upsSign0",
        reason="Shows the intrinsic/global-isel pattern that can select vups.4x.",
    ),
    SourceNeedle(
        name="aie2p vextbcst.16 pattern",
        relative_path="llvm/lib/Target/AIE/aie2p/AIE2PInstrPatterns.td",
        pattern=r"ExtBcstPair<VEXTBCST_16_vec_extract_broadcast_imm",
        reason="Shows that .16 extract-broadcast is explicitly modeled, not an accidental asm spelling.",
    ),
    SourceNeedle(
        name="aie2p vextbcst.16 schedule",
        relative_path="llvm/lib/Target/AIE/aie2p/AIE2PGenSchedule.td",
        pattern=r"InstrItinData<II_VEXTBCST_16_vec_extract_broadcast_imm",
        reason="Provides latency/bypass metadata for a scheduler or static checker.",
    ),
    SourceNeedle(
        name="aie2p vmac.f schedule",
        relative_path="llvm/lib/Target/AIE/aie2p/AIE2PGenSchedule.td",
        pattern=r"InstrItinData<II_VMAC_f_vmac_bf_vmul_bf_core_X_X",
        reason="Provides latency/bypass metadata for bf16 vmac.f.",
    ),
    SourceNeedle(
        name="aie2p vups.4x schedule",
        relative_path="llvm/lib/Target/AIE/aie2p/AIE2PGenSchedule.td",
        pattern=r"InstrItinData<II_VUPS_4x_mv_ups_x2d_upsSign0",
        reason="Provides latency/bypass metadata for vups.4x x-to-d.",
    ),
    SourceNeedle(
        name="postpipeliner spill guidance",
        relative_path="PERF_OPT_GUIDE.md",
        pattern=r"The pipelined loop has spill code in the loop body",
        reason="Explains why our C++ route fails when register pressure creates spills.",
    ),
    SourceNeedle(
        name="postpipeliner target ii guidance",
        relative_path="PERF_OPT_GUIDE.md",
        pattern=r"AIE_TRY_INITIATION_INTERVAL",
        reason="Documents the compiler-level control surface for software-pipelined loops.",
    ),
    SourceNeedle(
        name="bundle count guidance",
        relative_path="PERF_OPT_GUIDE.md",
        pattern=r"BundleCount.*Number of VLIW bundles",
        reason="Gives a compiler-reported cycle proxy we can gate in experiments.",
    ),
)


MIR_FILES = (
    (
        "aie2p vups binary MIR",
        "llvm/test/CodeGen/AIE/aie2p/BinaryOutput/vups.mir",
        "Contains concrete machine-instruction names for vups.2x and vups.4x.",
    ),
    (
        "aie2p vextbcst binary MIR",
        "llvm/test/CodeGen/AIE/aie2p/BinaryOutput/vextbcst.mir",
        "Contains concrete machine-instruction names for vextbcst.16.",
    ),
    (
        "aie2p vmac binary MIR",
        "llvm/test/CodeGen/AIE/aie2p/BinaryOutput/vmac.mir",
        "Contains concrete machine-instruction names for vmac.f forms.",
    ),
    (
        "aie2p postpipelined gemm MIR",
        "llvm/test/CodeGen/AIE/aie2p/schedule/postpipeliner/gemm-bfp16-v10.mir",
        "Shows the MIR-level route into postpipeliner, remarks, ZOL, liveins, and fixed registers.",
    ),
)


BACKEND_SURFACES = (
    BackendSurface(
        name="instruction database",
        paths=[
            "llvm/lib/Target/AIE/aie2p/AIE2PGenInstrInfo.td",
            "llvm/lib/Target/AIE/aie2p/AIE2PInstrPatterns.td",
        ],
        use="Parse or reference exact instruction names, operands, implicit regs, and selectable patterns for the Q4NX generator.",
        risk="Generated TD files are backend-internal contracts; pin the llvm-aie revision before depending on names.",
    ),
    BackendSurface(
        name="schedule database",
        paths=["llvm/lib/Target/AIE/aie2p/AIE2PGenSchedule.td"],
        use="Extract instruction latency/bypass facts for a nop-free source-asm checker or for MIR schedule expectations.",
        risk="It is not enough to count opcodes; resource conflicts and VLIW packetization still need llc/llvm-mc validation.",
    ),
    BackendSurface(
        name="MIR examples",
        paths=[
            "llvm/test/CodeGen/AIE/aie2p/BinaryOutput/vups.mir",
            "llvm/test/CodeGen/AIE/aie2p/BinaryOutput/vextbcst.mir",
            "llvm/test/CodeGen/AIE/aie2p/BinaryOutput/vmac.mir",
            "llvm/test/CodeGen/AIE/aie2p/schedule/postpipeliner/gemm-bfp16-v10.mir",
        ],
        use="Use as templates for a tiny Q4NX MIR/codegen experiment with explicit registers and postpipeliner remarks.",
        risk="MIR is lower-level and brittle, but it gives more control than C++ and less manual scheduling than raw source asm.",
    ),
    BackendSurface(
        name="external tools",
        paths=[
            ".venv/lib/python3.12/site-packages/llvm-aie/bin/llc",
            ".venv/lib/python3.12/site-packages/llvm-aie/bin/llvm-mc",
            ".venv/lib/python3.12/site-packages/llvm-aie/bin/llvm-objdump",
        ],
        use="Keep assembler, disassembler, object writer, and scheduler as external services invoked by experiments.",
        risk="Do not copy these internals into IRON; use command outputs as gates.",
    ),
    BackendSurface(
        name="backend implementation reference",
        paths=[
            "llvm/lib/Target/AIE/AIEPostPipeliner.cpp",
            "llvm/lib/Target/AIE/AIEHazardRecognizer.cpp",
            "llvm/lib/Target/AIE/AIEFinalizeBundle.cpp",
            "llvm/lib/Target/AIE/MCTargetDesc/aie2p/AIE2PMCCodeEmitter.cpp",
            "llvm/lib/Target/AIE/AsmParser/AIE2PAsmParser.cpp",
            "llvm/lib/Target/AIE/Disassembler/AIE2PDisassembler.cpp",
        ],
        use="Read-only reference for why schedules fail, how bundles are finalized, and how source/MIR becomes bytes.",
        risk="These files are not small reusable libraries; direct extraction would import too much LLVM state.",
    ),
)


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    return path.read_text(encoding="utf-8").splitlines()


def find_hit(needle: SourceNeedle) -> SourceHit:
    path = LLVM_AIE_ROOT / needle.relative_path
    lines = read_lines(path)
    regexp = re.compile(needle.pattern)
    matched = [index for index, line in enumerate(lines) if regexp.search(line)]
    if not matched:
        raise RuntimeError(f"pattern not found: {needle.pattern} in {path}")
    line_index = matched[0]
    start = max(0, line_index - 2)
    end = min(len(lines), line_index + 3)
    excerpt = [f"{line_no + 1}: {lines[line_no]}" for line_no in range(start, end)]
    return SourceHit(
        name=needle.name,
        path=str(path),
        line=line_index + 1,
        matches=len(matched),
        pattern=needle.pattern,
        reason=needle.reason,
        excerpt=excerpt,
    )


def file_metric(name: str, relative_path: str, reason: str) -> FileMetric:
    path = LLVM_AIE_ROOT / relative_path
    text = path.read_text(encoding="utf-8")
    metrics = {
        "vups_4x": len(re.findall(r"VUPS_4x|vups\.4x", text)),
        "vups_2x": len(re.findall(r"VUPS_2x|vups\.2x", text)),
        "vextbcst_16": len(re.findall(r"VEXTBCST_16|vextbcst\.16", text)),
        "vextbcst_32": len(re.findall(r"VEXTBCST_32|vextbcst\.32", text)),
        "vmac_f": len(re.findall(r"VMAC_f|vmac\.f", text)),
        "postpipeliner": len(re.findall(r"postpipeliner", text)),
        "schedule_found": len(re.findall(r"Schedule found|!Passed", text)),
        "hardware_loop": len(re.findall(r"\$lc|\$ls|\$le|PseudoLoopEnd", text)),
    }
    return FileMetric(name=name, path=str(path), metrics=metrics, reason=reason)


def render_report(hits: list[SourceHit], metrics: list[FileMetric]) -> str:
    lines = [
        "# LLVM-AIE Backend Surface",
        "",
        "Status: `route-selected`",
        "",
        "C++ remains the wrong control surface for the MyLM-style main16 Q4NX hot body. ",
        "The useful `llvm-aie` pieces are lower-level: AIE2P instruction definitions, ",
        "schedule metadata, MIR examples, and the Peano command-line tools.",
        "",
        "## Reusable Surfaces",
        "",
    ]
    for surface in BACKEND_SURFACES:
        lines.append(f"### {surface.name}")
        lines.append("")
        lines.append(f"- Use: {surface.use}")
        lines.append(f"- Risk: {surface.risk}")
        lines.append("- Paths:")
        for path in surface.paths:
            lines.append(f"  - `{path}`")
        lines.append("")

    lines.extend(
        [
            "## Source Evidence",
            "",
            "| Surface | Line | Matches | Why |",
            "| --- | ---: | ---: | --- |",
        ]
    )
    for hit in hits:
        relative = Path(hit.path).relative_to(LLVM_AIE_ROOT)
        lines.append(f"| `{relative}` | {hit.line} | {hit.matches} | {hit.reason} |")

    lines.extend(["", "## MIR Metrics", ""])
    for metric in metrics:
        relative = Path(metric.path).relative_to(LLVM_AIE_ROOT)
        lines.append(f"### `{relative}`")
        lines.append("")
        lines.append(metric.reason)
        lines.append("")
        for key, value in metric.metrics.items():
            lines.append(f"- `{key}`: `{value}`")
        lines.append("")

    lines.extend(
        [
            "## Decision",
            "",
            "- Do not continue trying to coerce high-level C++ into MyLM-like assembly.",
            "- Do not copy LLVM backend C++ into IRON; it is too coupled to LLVM pass state.",
            "- Use `llvm-aie` as a compiler service and as machine-contract metadata.",
            "- Next implementation route: generate a tiny Q4NX MIR/basic-block template with explicit AIE2P machine instructions and physical register intent, then run Peano `llc`/`llvm-mc` and gate on assembly counts, postpipeliner remarks, bundle count, and NPU numeric equality.",
            "- Fallback route if MIR is too brittle: keep source assembly, but generate it from TD-derived latency/bypass metadata instead of hand-inserting nops.",
            "",
            "## Next Experiment",
            "",
            "`022_q4nx_mir_schedule_probe`: start from the AIE2P MIR test style, build a tiny Q4NX group body using `VUPS_4x`, `VEXTBCST_16`, and `VMAC_f`, and verify whether `llc --start-before=postmisched` can produce a scheduled AIE2P object without C++-induced spills.",
            "",
        ]
    )
    return "\n".join(lines)


def run() -> None:
    if not LLVM_AIE_ROOT.exists():
        raise FileNotFoundError(LLVM_AIE_ROOT)
    hits = [find_hit(needle) for needle in NEEDLES]
    metrics = [file_metric(name, path, reason) for name, path, reason in MIR_FILES]
    manifest = {
        "experiment": "021_llvm_aie_backend_surface",
        "llvm_aie_root": str(LLVM_AIE_ROOT),
        "status": "route-selected",
        "source_hits": [asdict(hit) for hit in hits],
        "mir_metrics": [asdict(metric) for metric in metrics],
        "backend_surfaces": [asdict(surface) for surface in BACKEND_SURFACES],
        "decision": [
            "C++ is not the right control surface for MyLM-style main16 Q4NX.",
            "Use llvm-aie TD/MIR/schedule data and tools, not copied backend C++.",
            "Try a tiny MIR schedule probe next.",
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(render_report(hits, metrics), encoding="utf-8")


def main() -> int:
    try:
        run()
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

