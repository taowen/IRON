#!/usr/bin/env python3
"""Compare naive and MyLM-projected Q4NX one-group MIR ordering."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
EXP006_RUN = REPO_ROOT / "main16-exps/006_q4nx_alias_lifetime_graph/run.py"
PEANO_BIN = REPO_ROOT / ".venv/lib/python3.12/site-packages/llvm-aie/bin"
LLC = PEANO_BIN / "llc"
OBJDUMP = PEANO_BIN / "llvm-objdump"
BUILD_DIR = EXPERIMENT_DIR / "build"
MANIFEST = EXPERIMENT_DIR / "q4nx_mir_lifetime_order_probe.json"
REPORT = EXPERIMENT_DIR / "q4nx_mir_lifetime_order_probe.md"


@dataclass(frozen=True)
class Candidate:
    name: str
    goal: str
    mir: str
    expect_schedule: bool


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class CandidateResult:
    name: str
    goal: str
    status: str
    llc: CommandResult
    objdump: CommandResult
    counts: dict[str, int]
    files: dict[str, str]
    reasons: list[str]


def load_exp006() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mylm_exp006_for_mir_order", EXP006_RUN)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load exp006 helper: {EXP006_RUN}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXP006 = load_exp006()


def run_command(args: tuple[str, ...]) -> CommandResult:
    completed = subprocess.run(
        args,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def count_text(text: str) -> dict[str, int]:
    bundle_matches = re.findall(r"BundleCount:\s+'(\d+)'", text)
    loop_bundle = int(bundle_matches[1]) if len(bundle_matches) >= 2 else -1
    return {
        "vups_4x": len(re.findall(r"\bvups\.4x\b", text)),
        "vups_2x": len(re.findall(r"\bvups\.2x\b", text)),
        "vextbcst_16": len(re.findall(r"\bvextbcst\.16\b", text)),
        "vextbcst_32": len(re.findall(r"\bvextbcst\.32\b", text)),
        "vmac_f": len(re.findall(r"\bvmac\.f\b", text)),
        "vst": len(re.findall(r"\bvst(?:\.|\b)", text)),
        "vlda": len(re.findall(r"\bvlda(?:\.|\b)", text)),
        "schedule_found": len(re.findall(r"Schedule found", text)),
        "schedule_missed": len(re.findall(r"No schedule found|Longest circuit does not fit II", text)),
        "bundle_count_remarks": len(bundle_matches),
        "loop_bundle_count": loop_bundle,
    }


def exp006_group1_events() -> list:
    return [
        event
        for event in EXP006.build_events()
        if event.group == 1 and event.op in {"vups.4x", "vextbcst.16", "vmac.f"}
    ]


def liveins() -> str:
    regs = [
        "$x0",
        "$x1",
        "$x2",
        "$x3",
        "$x4",
        "$x5",
        "$x6",
        "$x7",
        "$x8",
        "$x9",
        "$x10",
        "$x11",
        "$dm0",
        "$dm1",
        "$dm2",
        "$dm3",
        "$dm4",
        "$r4",
        "$s0",
        "$lr",
    ]
    return ", ".join(regs)


def vups_line(defs: tuple[str, ...], uses: tuple[str, ...]) -> str:
    if len(defs) != 1 or len(uses) < 2:
        raise RuntimeError(f"unexpected vups operands: defs={defs} uses={uses}")
    return (
        f"    ${defs[0]} = VUPS_4x_mv_ups_x2d_upsSign0 "
        f"${uses[0]}, ${uses[1]}, implicit-def $srups_of, implicit $crsat, "
        "implicit $crupsmode, implicit $upssign0"
    )


def vext_line(defs: tuple[str, ...], uses: tuple[str, ...], fragment: str) -> str:
    if len(defs) != 1 or len(uses) < 1:
        raise RuntimeError(f"unexpected vext operands: defs={defs} uses={uses}")
    match = re.search(r"#0x([0-9a-fA-F]+)", fragment)
    if match is None:
        raise RuntimeError(f"missing vext lane: {fragment}")
    lane = int(match.group(1), 16)
    return f"    ${defs[0]} = VEXTBCST_16_vec_extract_broadcast_imm ${uses[0]}, {lane}"


def vmac_line(defs: tuple[str, ...], uses: tuple[str, ...]) -> str:
    if len(defs) != 1 or len(uses) < 4:
        raise RuntimeError(f"unexpected vmac operands: defs={defs} uses={uses}")
    return (
        f"    ${defs[0]} = VMAC_f_vmac_bf_vmul_bf_core_X_X "
        f"${uses[0]}, ${uses[1]}, ${uses[2]}, ${uses[3]}, "
        "implicit-def $srfpflags, implicit $crfpmask"
    )


def mylm_projected_lines() -> list[str]:
    lines: list[str] = []
    for event in exp006_group1_events():
        if event.op == "vups.4x":
            lines.append(vups_line(event.defs, event.uses))
        elif event.op == "vextbcst.16":
            lines.append(vext_line(event.defs, event.uses, event.fragment))
        elif event.op == "vmac.f":
            lines.append(vmac_line(event.defs, event.uses))
    return lines


def naive_lines() -> list[str]:
    lines: list[str] = []
    for index in range(8):
        lines.append(
            f"    $dm{index % 5} = VUPS_4x_mv_ups_x2d_upsSign0 "
            f"$x{index % 8}, $s0, implicit-def $srups_of, implicit $crsat, "
            "implicit $crupsmode, implicit $upssign0"
        )
    for index in range(32):
        lines.append(f"    $x{index % 8} = VEXTBCST_16_vec_extract_broadcast_imm $x11, {index}")
    for index in range(33):
        lines.append(
            f"    $dm{index % 5} = VMAC_f_vmac_bf_vmul_bf_core_X_X "
            f"$dm{(index + 1) % 5}, $x{index % 8}, $x{(index + 3) % 8}, $r4, "
            "implicit-def $srfpflags, implicit $crfpmask"
        )
    return lines


def mir_for(name: str, body_lines: list[str]) -> str:
    body = "\n".join(body_lines)
    return f"""--- |
  define dso_local void @{name}(i32 noundef %n) local_unnamed_addr {{
  entry:
    call void @llvm.set.loop.iterations.i32(i32 %n)
    br label %loop

  loop:
    %0 = call i1 @llvm.loop.decrement.i32(i32 1)
    br i1 %0, label %loop, label %exit, !llvm.loop !0

  exit:
    ret void
  }}

  declare void @llvm.set.loop.iterations.i32(i32)
  declare i1 @llvm.loop.decrement.i32(i32)

  !0 = distinct !{{!0, !1, !2}}
  !1 = !{{!"llvm.loop.mustprogress"}}
  !2 = !{{!"llvm.loop.itercount.range", i64 4}}

...
---
name:            {name}
alignment:       16
tracksRegLiveness: true
body:             |
  bb.0.entry (align 16):
    successors: %bb.1
    liveins: {liveins()}

    $lc = ADD_NC_mv_add_ri $r4, 0
    $ls = MOVXM %bb.1
    $le = MOVXM <mcsymbol .L_LEnd0>

  bb.1.loop (align 16):
    successors: %bb.1, %bb.2
    liveins: {liveins()}

{body}
    PseudoLoopEnd <mcsymbol .L_LEnd0>, %bb.1

  bb.2.exit (align 16):
    RET implicit $lr
    DelayedSchedBarrier

...
"""


def candidates() -> tuple[Candidate, ...]:
    return (
        Candidate(
            name="naive_x2d_order",
            goal="Naive x2d one-group order with the same macro counts as MyLM group1.",
            mir=mir_for("naive_x2d_order", naive_lines()),
            expect_schedule=False,
        ),
        Candidate(
            name="mylm_group1_projected_order",
            goal="MyLM group1 projection preserving actual order and registers for vups/vext/vmac.",
            mir=mir_for("mylm_group1_projected_order", mylm_projected_lines()),
            expect_schedule=False,
        ),
    )


def run_candidate(candidate: Candidate) -> CandidateResult:
    mir_path = BUILD_DIR / f"{candidate.name}.mir"
    object_path = BUILD_DIR / f"{candidate.name}.o"
    asm_path = BUILD_DIR / f"{candidate.name}.s"
    objdump_path = BUILD_DIR / f"{candidate.name}.objdump"
    mir_path.write_text(candidate.mir, encoding="utf-8")
    common_args = (
        "-verify-machineinstrs",
        "--mtriple=aie2p",
        "-O2",
        "--start-before=postmisched",
        "-pass-remarks-output=-",
        "-pass-remarks-filter=pipeliner|aie-asm-printer",
    )
    llc = run_command((str(LLC), *common_args, "--filetype=obj", str(mir_path), "-o", str(object_path)))
    asm = CommandResult(1, "", "object build did not complete")
    objdump = CommandResult(1, "", "object build did not complete")
    if llc.returncode == 0:
        asm = run_command((str(LLC), *common_args, str(mir_path), "-o", str(asm_path)))
        objdump = run_command(
            (
                str(OBJDUMP),
                "--triple=aie2p",
                "-dr",
                "--no-print-imm-hex",
                str(object_path),
            )
        )
        objdump_path.write_text(objdump.stdout + objdump.stderr, encoding="utf-8")
    combined = "\n".join([llc.stdout, llc.stderr, asm.stdout, asm.stderr, objdump.stdout, objdump.stderr])
    counts = count_text(combined)
    reasons: list[str] = []
    if llc.returncode != 0:
        reasons.append(f"llc failed with return code {llc.returncode}")
    if objdump.returncode != 0:
        reasons.append(f"llvm-objdump failed with return code {objdump.returncode}")
    if counts["vups_4x"] != 8:
        reasons.append(f"expected 8 vups.4x, got {counts['vups_4x']}")
    if counts["vextbcst_16"] != 32:
        reasons.append(f"expected 32 vextbcst.16, got {counts['vextbcst_16']}")
    if counts["vmac_f"] != 33:
        reasons.append(f"expected 33 vmac.f, got {counts['vmac_f']}")
    if counts["vextbcst_32"] != 0:
        reasons.append("unexpected vextbcst.32")
    if counts["vst"] != 0:
        reasons.append("unexpected vector store")
    status = "pass" if not reasons else "fail"
    return CandidateResult(
        name=candidate.name,
        goal=candidate.goal,
        status=status,
        llc=llc,
        objdump=objdump,
        counts=counts,
        files={
            "mir": str(mir_path),
            "object": str(object_path),
            "asm": str(asm_path),
            "objdump": str(objdump_path),
        },
        reasons=reasons,
    )


def render_report(results: list[CandidateResult]) -> str:
    lines = [
        "# Q4NX MIR Lifetime Order Probe",
        "",
        "Status: `passed`" if all(result.status == "pass" for result in results) else "Status: `failed`",
        "",
        "This experiment compares naive one-group MIR ordering with a projection of MyLM group1's actual",
        "`vups.4x`, `vextbcst.16`, and `vmac.f` order/registers.",
        "",
        "## Results",
        "",
    ]
    for result in results:
        lines.append(f"### `{result.name}`")
        lines.append("")
        lines.append(f"Status: `{result.status}`")
        lines.append("")
        lines.append(result.goal)
        lines.append("")
        lines.append(f"- `llc_returncode`: `{result.llc.returncode}`")
        lines.append(f"- `objdump_returncode`: `{result.objdump.returncode}`")
        for key, value in result.counts.items():
            lines.append(f"- `{key}`: `{value}`")
        if result.reasons:
            lines.append("")
            lines.append("Reasons:")
            for reason in result.reasons:
                lines.append(f"- {reason}")
        lines.append("")
    lines.extend(
        [
            "## Interpretation",
            "",
            "- Direct MIR can preserve the MyLM instruction vocabulary and register names.",
            "- The key comparison is loop bundle count and postpipeliner status, not only opcode counts.",
            "- If the projected order still does not schedule, the missing part is the full interleaved dependency graph including `vlda/vldb/vunpack/vadd/vsub/vconv/vmov`, not only the three hot op classes.",
            "",
            "## Next Step",
            "",
            "Generate a full group1 MIR projection with the extra producer instructions that feed the mixed-half operands in experiment 007, then rerun the same schedule gate.",
            "",
        ]
    )
    return "\n".join(lines)


def run() -> None:
    if not LLC.exists():
        raise FileNotFoundError(LLC)
    if not OBJDUMP.exists():
        raise FileNotFoundError(OBJDUMP)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    results = [run_candidate(candidate) for candidate in candidates()]
    manifest = {
        "experiment": "023_q4nx_mir_lifetime_order_probe",
        "status": "passed" if all(result.status == "pass" for result in results) else "failed",
        "source": str(EXP006.EXP004.DEFAULT_ELF.relative_to(REPO_ROOT)),
        "group1_projected_events": len(exp006_group1_events()),
        "results": [asdict(result) for result in results],
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(render_report(results), encoding="utf-8")


def main() -> int:
    try:
        run()
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

