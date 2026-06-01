#!/usr/bin/env python3
"""Summarize the compiler-visible VUPS.4x contract and compile minimal MIR forms."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
LLVM_AIE = Path.home() / "projects/llvm-aie"
PEANO_BIN = REPO_ROOT / ".venv/lib/python3.12/site-packages/llvm-aie/bin"
LLC = PEANO_BIN / "llc"
OBJDUMP = PEANO_BIN / "llvm-objdump"
INSTR_INFO = LLVM_AIE / "llvm/lib/Target/AIE/aie2p/AIE2PGenInstrInfo.td"
SCHEDULE = LLVM_AIE / "llvm/lib/Target/AIE/aie2p/AIE2PGenSchedule.td"
UPS_HEADER = LLVM_AIE / "clang/lib/Headers/aie2p/aie2p_ups.h"
BUILTINS = LLVM_AIE / "clang/include/clang/Basic/BuiltinsAIE2P.def"
BUILD_DIR = EXPERIMENT_DIR / "build"
MANIFEST = EXPERIMENT_DIR / "vups4x_semantics_surface.json"
REPORT = EXPERIMENT_DIR / "vups4x_semantics_surface.md"

VUPS_VARIANTS = (
    "VUPS_4x_mv_ups_w2c_upsSign0",
    "VUPS_4x_mv_ups_w2c_upsSign1",
    "VUPS_4x_mv_ups_x2d_upsSign0",
    "VUPS_4x_mv_ups_x2d_upsSign1",
)


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class InstructionFact:
    opcode: str
    asm_name: str
    dst_class: str
    src_class: str
    implicit_defs: tuple[str, ...]
    implicit_uses: tuple[str, ...]
    encoding_id: str


@dataclass(frozen=True)
class ScheduleFact:
    opcode: str
    resources: tuple[str, ...]
    latencies: tuple[str, ...]


@dataclass(frozen=True)
class IntrinsicFact:
    source: str
    function: str
    return_type: str
    argument_type: str
    builtin: str | None


@dataclass(frozen=True)
class MirBuild:
    variant: str
    mir: str
    object: str
    objdump: str
    llc: CommandResult
    objdump_cmd: CommandResult
    disassembly_line: str
    status: str


def run_command(args: tuple[str, ...]) -> CommandResult:
    completed = subprocess.run(
        args,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def clean_group(text: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in text.split(",") if part.strip())


def instruction_facts() -> tuple[InstructionFact, ...]:
    text = INSTR_INFO.read_text(encoding="utf-8")
    facts: list[InstructionFact] = []
    pattern = re.compile(
        r"let Itinerary = II_(?P<opcode>VUPS_4x_[^,]+), Defs = \[(?P<defs>[^\]]*)\], "
        r"Uses = \[(?P<uses>[^\]]*)\] in\n"
        r"def (?P=opcode) : [^<]+<\(outs (?P<dst>[^:]+):\$dst\), "
        r"\(ins (?P<src>[^:]+):\$src, eS:\$su\), \"(?P<asm>[^\"]+)\""
        r".*?\n// id: (?P<id>[^\n]+)",
        re.DOTALL,
    )
    for match in pattern.finditer(text):
        opcode = match.group("opcode")
        if opcode in VUPS_VARIANTS:
            facts.append(
                InstructionFact(
                    opcode=opcode,
                    asm_name=match.group("asm"),
                    dst_class=match.group("dst"),
                    src_class=match.group("src"),
                    implicit_defs=clean_group(match.group("defs")),
                    implicit_uses=clean_group(match.group("uses")),
                    encoding_id=match.group("id"),
                )
            )
    return tuple(facts)


def split_top_level_commas(text: str) -> tuple[str, ...]:
    parts: list[str] = []
    start = 0
    depth = 0
    for index, char in enumerate(text):
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
        elif char == "," and depth == 0:
            parts.append(text[start:index].strip())
            start = index + 1
    parts.append(text[start:].strip())
    return tuple(part for part in parts if part)


def schedule_facts() -> tuple[ScheduleFact, ...]:
    text = SCHEDULE.read_text(encoding="utf-8")
    facts: list[ScheduleFact] = []
    pattern = re.compile(
        r"InstrItinData<II_(?P<opcode>VUPS_4x_[^,]+), "
        r"\[(?P<resources>[^\]]*)\], \[(?P<latencies>[^\]]*)\]>",
        re.DOTALL,
    )
    for match in pattern.finditer(text):
        opcode = match.group("opcode")
        if opcode in VUPS_VARIANTS:
            facts.append(
                ScheduleFact(
                    opcode=opcode,
                    resources=split_top_level_commas(match.group("resources")),
                    latencies=split_top_level_commas(match.group("latencies")),
                )
            )
    return tuple(facts)


def intrinsic_facts() -> tuple[IntrinsicFact, ...]:
    header = UPS_HEADER.read_text(encoding="utf-8")
    builtins = BUILTINS.read_text(encoding="utf-8")
    rows = (
        ("aie2p_ups.h", "sups", "v16acc32", "v16int16", "__builtin_aie2p_acc32_v16_I256_ups"),
        ("aie2p_ups.h", "sups", "v32acc32", "v32int16", "__builtin_aie2p_acc32_v32_I512_ups"),
        ("aie2p_ups.h", "ups", "v16accfloat", "v16bfloat16", "__builtin_aie2p_v16bf16_to_v16accfloat"),
        ("aie2p_ups.h", "ups", "v32accfloat", "v32bfloat16", "__builtin_aie2p_v32bf16_to_v32accfloat"),
    )
    facts: list[IntrinsicFact] = []
    for source, function, return_type, argument_type, builtin in rows:
        signature = f"INTRINSIC({return_type}) {function}({argument_type} a,"
        if signature in header or builtin in builtins:
            facts.append(
                IntrinsicFact(
                    source=source,
                    function=function,
                    return_type=return_type,
                    argument_type=argument_type,
                    builtin=builtin if builtin in builtins else None,
                )
            )
    return tuple(facts)


def mir_body(variant: str) -> str:
    if "_w2c_" in variant:
        dst = "$cml0"
        src = "$wl0"
        liveins = "$wl0, $s0, $lr"
    else:
        dst = "$dm0"
        src = "$x0"
        liveins = "$x0, $s0, $lr"
    sign = "$upssign1" if variant.endswith("upsSign1") else "$upssign0"
    return f"""---
name:            {variant.lower()}
alignment:       16
tracksRegLiveness: true
body:             |
  bb.0.entry (align 16):
    liveins: {liveins}

    BUNDLE {{
      {dst} = {variant} {src}, $s0, implicit-def $srups_of, implicit $crsat, implicit $crupsmode, implicit {sign}
    }}
    BUNDLE {{
      RET implicit $lr
    }}
    DelayedSchedBarrier

...
"""


def compile_variant(variant: str) -> MirBuild:
    name = variant.lower()
    mir_path = BUILD_DIR / f"{name}.mir"
    object_path = BUILD_DIR / f"{name}.o"
    objdump_path = BUILD_DIR / f"{name}.objdump"
    mir = mir_body(variant)
    mir_path.write_text(mir, encoding="utf-8")
    llc = run_command(
        (
            str(LLC),
            "--mtriple=aie2p",
            "--start-after=postmisched",
            "--skip-machine-alignment",
            "--filetype=obj",
            str(mir_path),
            "-o",
            str(object_path),
        )
    )
    objdump = CommandResult(1, "", "llc failed")
    disassembly_line = ""
    status = "compile_failed"
    if llc.returncode == 0:
        objdump = run_command(
            (
                str(OBJDUMP),
                "--triple=aie2p",
                "-dr",
                "--no-print-imm-hex",
                "--disassemble-zeroes",
                str(object_path),
            )
        )
        objdump_path.write_text(objdump.stdout + objdump.stderr, encoding="utf-8")
        for line in (objdump.stdout + objdump.stderr).splitlines():
            if "vups.4x" in line:
                disassembly_line = line.strip()
                break
        status = "passed" if objdump.returncode == 0 and disassembly_line else "missing_vups4x"
    return MirBuild(
        variant=variant,
        mir=str(mir_path.relative_to(REPO_ROOT)),
        object=str(object_path.relative_to(REPO_ROOT)),
        objdump=str(objdump_path.relative_to(REPO_ROOT)),
        llc=llc,
        objdump_cmd=objdump,
        disassembly_line=disassembly_line,
        status=status,
    )


def build_manifest() -> dict[str, object]:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    instr = instruction_facts()
    sched = schedule_facts()
    intrinsics = intrinsic_facts()
    builds = tuple(compile_variant(variant) for variant in VUPS_VARIANTS)
    status = "passed" if len(instr) == 4 and len(sched) == 4 and all(build.status == "passed" for build in builds) else "failed"
    return {
        "status": status,
        "instruction_facts": [asdict(row) for row in instr],
        "schedule_facts": [asdict(row) for row in sched],
        "intrinsic_facts": [asdict(row) for row in intrinsics],
        "mir_builds": [asdict(row) for row in builds],
        "interpretation": {
            "compiler_visible": (
                "llvm-aie exposes VUPS.4x opcode forms, operand classes, implicit "
                "control/status registers, and schedule resources."
            ),
            "still_unknown": (
                "The compiler metadata does not describe physical accumulator "
                "sub-cell preservation/read-modify-write behavior. Experiment 034 "
                "already falsified the simple pure-destination-overwrite model."
            ),
            "next_probe": (
                "Run isolated NPU readback probes for w2c/x2d, upssign0/1, "
                "crupsmode, and initialized accumulator quadrants."
            ),
        },
    }


def render_report(manifest: dict[str, object]) -> str:
    lines = [
        "# VUPS.4x Semantics Surface",
        "",
        f"Status: `{manifest['status']}`",
        "",
        "## What llvm-aie Teaches",
        "",
        "| opcode | asm | dst | src | implicit uses | implicit defs |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for row in manifest["instruction_facts"]:
        uses = ", ".join(row["implicit_uses"])
        defs = ", ".join(row["implicit_defs"])
        lines.append(
            f"| `{row['opcode']}` | `{row['asm_name']}` | `{row['dst_class']}` | "
            f"`{row['src_class']}` | `{uses}` | `{defs}` |"
        )
    lines.extend(
        [
            "",
            "## Schedule Surface",
            "",
            "| opcode | resources | latencies |",
            "| --- | --- | --- |",
        ]
    )
    for row in manifest["schedule_facts"]:
        lines.append(
            f"| `{row['opcode']}` | `{', '.join(row['resources'])}` | "
            f"`{', '.join(row['latencies'])}` |"
        )
    lines.extend(
        [
            "",
            "## Minimal MIR Builds",
            "",
            "| variant | status | disassembly |",
            "| --- | --- | --- |",
        ]
    )
    for row in manifest["mir_builds"]:
        lines.append(f"| `{row['variant']}` | `{row['status']}` | `{row['disassembly_line']}` |")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- MIR/llvm-aie is enough to encode all four `vups.4x` forms reproducibly.",
            "- The available compiler metadata is not a value-level semantics document.",
            "- In particular, it does not prove whether `dm/cml/cmh/bm*` sub-cells are fully overwritten, preserved, or implicitly read.",
            "- Experiment 034 already showed that treating a nearby accumulator-cell move as dead is unsound for the zero/offset path.",
            "- The next useful experiment is not another static mutation. It is an isolated NPU readback probe that initializes accumulator quadrants, executes one `vups.4x`, and records which cells changed.",
            "",
            "## Sources",
            "",
            f"- `{INSTR_INFO.relative_to(Path.home())}`",
            f"- `{SCHEDULE.relative_to(Path.home())}`",
            f"- `{UPS_HEADER.relative_to(Path.home())}`",
            f"- `{BUILTINS.relative_to(Path.home())}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    try:
        manifest = build_manifest()
        MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        REPORT.write_text(render_report(manifest), encoding="utf-8")
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
