#!/usr/bin/env python3
"""Probe direct AIE2P MIR as a Q4NX kernel generation route."""

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
PEANO_BIN = REPO_ROOT / ".venv/lib/python3.12/site-packages/llvm-aie/bin"
LLC = PEANO_BIN / "llc"
OBJDUMP = PEANO_BIN / "llvm-objdump"
BUILD_DIR = EXPERIMENT_DIR / "build"
MANIFEST = EXPERIMENT_DIR / "q4nx_mir_schedule_probe.json"
REPORT = EXPERIMENT_DIR / "q4nx_mir_schedule_probe.md"


@dataclass(frozen=True)
class Candidate:
    name: str
    goal: str
    mir: str
    compile_args: tuple[str, ...]
    expect_postpipeline: bool


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


BASIC_BLOCK_MIR = """--- |
  define dso_local void @q4nx_mir_basic_block() local_unnamed_addr {
  entry:
    ret void
  }

...
---
name:            q4nx_mir_basic_block
alignment:       16
tracksRegLiveness: true
body:             |
  bb.0.entry (align 16):
    liveins: $wl0, $x0, $x1, $dm0, $r0, $s0, $lr

    $cml0 = VUPS_4x_mv_ups_w2c_upsSign0 $wl0, $s0, implicit-def $srups_of, implicit $crsat, implicit $crupsmode, implicit $upssign0
    $x2 = VEXTBCST_16_vec_extract_broadcast_imm $x0, 0
    $dm0 = VMAC_f_vmac_bf_vmul_bf_core_X_X $dm0, $x1, $x2, $r0, implicit-def $srfpflags, implicit $crfpmask
    RET implicit $lr
    DelayedSchedBarrier

...
"""


LOOP_BODY_MIR = """--- |
  define dso_local void @q4nx_mir_loop_probe(i32 noundef %n) local_unnamed_addr {
  entry:
    call void @llvm.set.loop.iterations.i32(i32 %n)
    br label %loop

  loop:
    %0 = call i1 @llvm.loop.decrement.i32(i32 1)
    br i1 %0, label %loop, label %exit, !llvm.loop !0

  exit:
    ret void
  }

  declare void @llvm.set.loop.iterations.i32(i32)
  declare i1 @llvm.loop.decrement.i32(i32)

  !0 = distinct !{!0, !1, !2}
  !1 = !{!"llvm.loop.mustprogress"}
  !2 = !{!"llvm.loop.itercount.range", i64 4}

...
---
name:            q4nx_mir_loop_probe
alignment:       16
tracksRegLiveness: true
body:             |
  bb.0.entry (align 16):
    successors: %bb.1
    liveins: $wl0, $x0, $x1, $dm0, $r0, $s0, $lr

    $lc = ADD_NC_mv_add_ri $r0, 0
    $ls = MOVXM %bb.1
    $le = MOVXM <mcsymbol .L_LEnd0>

  bb.1.loop (align 16):
    successors: %bb.1, %bb.2
    liveins: $wl0, $x0, $x1, $dm0, $r0, $s0, $lr

    $cml0 = VUPS_4x_mv_ups_w2c_upsSign0 $wl0, $s0, implicit-def $srups_of, implicit $crsat, implicit $crupsmode, implicit $upssign0
    $x2 = VEXTBCST_16_vec_extract_broadcast_imm $x0, 0
    $dm0 = VMAC_f_vmac_bf_vmul_bf_core_X_X $dm0, $x1, $x2, $r0, implicit-def $srfpflags, implicit $crfpmask
    PseudoLoopEnd <mcsymbol .L_LEnd0>, %bb.1

  bb.2.exit (align 16):
    RET implicit $lr
    DelayedSchedBarrier

...
"""


def one_group_shape_mir() -> str:
    vups_dest = ("cml0", "cmh0", "cml1", "cmh1", "cml2", "cmh2", "cml3", "cmh3")
    vups_lines = [
        f"    ${dest} = VUPS_4x_mv_ups_w2c_upsSign0 $wl{index}, $s{index % 4}, implicit-def $srups_of, implicit $crsat, implicit $crupsmode, implicit $upssign0"
        for index, dest in enumerate(vups_dest)
    ]
    vext_lines = [
        f"    $x{2 + (index % 8)} = VEXTBCST_16_vec_extract_broadcast_imm $x{index % 2}, {index}"
        for index in range(32)
    ]
    vmac_lines = [
        f"    $dm{index % 5} = VMAC_f_vmac_bf_vmul_bf_core_X_X $dm{(index + 1) % 5}, $x{2 + (index % 8)}, $x{2 + ((index + 3) % 8)}, $r{index % 4}, implicit-def $srfpflags, implicit $crfpmask"
        for index in range(33)
    ]
    body_lines = "\n".join([*vups_lines, *vext_lines, *vmac_lines])
    return f"""--- |
  define dso_local void @q4nx_mir_one_group_shape(i32 noundef %n) local_unnamed_addr {{
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
name:            q4nx_mir_one_group_shape
alignment:       16
tracksRegLiveness: true
body:             |
  bb.0.entry (align 16):
    successors: %bb.1
    liveins: $wl0, $wl1, $wl2, $wl3, $wl4, $wl5, $wl6, $wl7, $x0, $x1, $dm0, $dm1, $dm2, $dm3, $dm4, $r0, $r1, $r2, $r3, $s0, $s1, $s2, $s3, $lr

    $lc = ADD_NC_mv_add_ri $r0, 0
    $ls = MOVXM %bb.1
    $le = MOVXM <mcsymbol .L_LEnd0>

  bb.1.loop (align 16):
    successors: %bb.1, %bb.2
    liveins: $wl0, $wl1, $wl2, $wl3, $wl4, $wl5, $wl6, $wl7, $x0, $x1, $dm0, $dm1, $dm2, $dm3, $dm4, $r0, $r1, $r2, $r3, $s0, $s1, $s2, $s3, $lr

{body_lines}
    PseudoLoopEnd <mcsymbol .L_LEnd0>, %bb.1

  bb.2.exit (align 16):
    RET implicit $lr
    DelayedSchedBarrier

...
"""


def candidate_list() -> tuple[Candidate, ...]:
    return (
        Candidate(
        name="basic_machine_block",
        goal="Prove direct AIE2P machine MIR can assemble the three key Q4NX instructions.",
        mir=BASIC_BLOCK_MIR,
        compile_args=(
            "-verify-machineinstrs",
            "--mtriple=aie2p",
            "-O2",
            "--start-before=postmisched",
            "--filetype=obj",
        ),
        expect_postpipeline=False,
        ),
        Candidate(
        name="loop_postpipeline_probe",
        goal="Probe whether the same direct machine-instruction shape can enter the postpipeliner/ZOL route.",
        mir=LOOP_BODY_MIR,
        compile_args=(
            "-verify-machineinstrs",
            "--mtriple=aie2p",
            "-O2",
            "--start-before=postmisched",
            "--filetype=obj",
            "-pass-remarks-output=-",
            "-pass-remarks-filter=pipeliner|aie-asm-printer",
        ),
        expect_postpipeline=True,
        ),
        Candidate(
            name="one_group_shape_probe",
            goal="Probe the MyLM-style one-group macro shape: 8 vups.4x, 32 vextbcst.16, and 33 vmac.f.",
            mir=one_group_shape_mir(),
            compile_args=(
                "-verify-machineinstrs",
                "--mtriple=aie2p",
                "-O2",
                "--start-before=postmisched",
                "--filetype=obj",
                "-pass-remarks-output=-",
                "-pass-remarks-filter=pipeliner|aie-asm-printer",
            ),
            expect_postpipeline=True,
        ),
    )


def run_command(args: tuple[str, ...]) -> CommandResult:
    completed = subprocess.run(
        args,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return CommandResult(
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def count_text(text: str) -> dict[str, int]:
    return {
        "vups_4x": len(re.findall(r"\bvups\.4x\b", text)),
        "vups_2x": len(re.findall(r"\bvups\.2x\b", text)),
        "vextbcst_16": len(re.findall(r"\bvextbcst\.16\b", text)),
        "vextbcst_32": len(re.findall(r"\bvextbcst\.32\b", text)),
        "vmac_f": len(re.findall(r"\bvmac\.f\b", text)),
        "vst": len(re.findall(r"\bvst(?:\.|\b)", text)),
        "vlda": len(re.findall(r"\bvlda(?:\.|\b)", text)),
        "schedule_found": len(re.findall(r"Schedule found", text)),
        "schedule_missed": len(re.findall(r"No schedule found", text)),
        "bundle_count": len(re.findall(r"BundleCount", text)),
        "ret": len(re.findall(r"\bret\b", text)),
    }


def classify(
    candidate: Candidate,
    llc: CommandResult,
    objdump: CommandResult,
    counts: dict[str, int],
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if llc.returncode != 0:
        reasons.append(f"llc failed with return code {llc.returncode}")
        return "fail", reasons
    if objdump.returncode != 0:
        reasons.append(f"llvm-objdump failed with return code {objdump.returncode}")
        return "fail", reasons
    if counts["vups_4x"] < 1:
        reasons.append("missing vups.4x in objdump")
    if counts["vextbcst_16"] < 1:
        reasons.append("missing vextbcst.16 in objdump")
    if counts["vmac_f"] < 1:
        reasons.append("missing vmac.f in objdump")
    if counts["vextbcst_32"] != 0:
        reasons.append("unexpected vextbcst.32 in objdump")
    if counts["vst"] != 0:
        reasons.append("unexpected vector store in objdump")
    if candidate.expect_postpipeline and counts["schedule_found"] < 1:
        reasons.append("postpipeliner did not report Schedule found")
    status = "pass" if not reasons else "partial"
    return status, reasons


def run_candidate(candidate: Candidate) -> CandidateResult:
    mir_path = BUILD_DIR / f"{candidate.name}.mir"
    obj_path = BUILD_DIR / f"{candidate.name}.o"
    objdump_path = BUILD_DIR / f"{candidate.name}.objdump"
    asm_path = BUILD_DIR / f"{candidate.name}.s"
    mir_path.write_text(candidate.mir, encoding="utf-8")

    llc_args = (str(LLC), *candidate.compile_args, str(mir_path), "-o", str(obj_path))
    llc = run_command(llc_args)

    asm = CommandResult(returncode=1, stdout="", stderr="llc object build did not complete")
    objdump = CommandResult(returncode=1, stdout="", stderr="llc object build did not complete")
    if llc.returncode == 0:
        asm_args = (
            str(LLC),
            "-verify-machineinstrs",
            "--mtriple=aie2p",
            "-O2",
            "--start-before=postmisched",
            str(mir_path),
            "-o",
            str(asm_path),
        )
        asm = run_command(asm_args)
        objdump_args = (
            str(OBJDUMP),
            "--triple=aie2p",
            "-dr",
            "--no-print-imm-hex",
            str(obj_path),
        )
        objdump = run_command(objdump_args)
        objdump_path.write_text(objdump.stdout + objdump.stderr, encoding="utf-8")

    combined = "\n".join([llc.stdout, llc.stderr, asm.stdout, asm.stderr, objdump.stdout, objdump.stderr])
    counts = count_text(combined)
    status, reasons = classify(candidate, llc, objdump, counts)
    return CandidateResult(
        name=candidate.name,
        goal=candidate.goal,
        status=status,
        llc=llc,
        objdump=objdump,
        counts=counts,
        files={
            "mir": str(mir_path),
            "object": str(obj_path),
            "asm": str(asm_path),
            "objdump": str(objdump_path),
        },
        reasons=reasons,
    )


def render_report(results: list[CandidateResult]) -> str:
    pass_count = sum(1 for result in results if result.status == "pass")
    overall = "pass" if pass_count == len(results) else "partial"
    lines = [
        "# Q4NX MIR Schedule Probe",
        "",
        f"Status: `{overall}`",
        "",
        "This experiment bypasses C++ and emits AIE2P machine MIR directly.",
        "It tests whether Peano can assemble the Q4NX-critical instruction trio:",
        "`vups.4x`, `vextbcst.16`, and `vmac.f`.",
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
            "## Conclusion",
            "",
            "- Direct machine MIR is a viable route if the basic block passes: it avoids C++ lowering and gives exact instruction selection.",
            "- The next meaningful gate is not C++ tuning; it is MIR/codegen scheduling with explicit register lifetimes.",
            "- If the loop/postpipeline probe remains partial, we should still use MIR for object generation and use TD-derived latency checks for a source-asm/codegen scheduler.",
            "",
            "## Next Step",
            "",
            "Extend this from a three-instruction probe to one MyLM-style Q4NX activation group:",
            "",
            "- generate 8 `vups.4x`, 32 `vextbcst.16`, and 33 `vmac.f` shape for one group;",
            "- keep registers explicit enough to prevent C++-style spills;",
            "- compare objdump counts and then run an isolated NPU numeric gate against MyLM direct `0x1870`.",
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
    results = [run_candidate(candidate) for candidate in candidate_list()]
    manifest = {
        "experiment": "022_q4nx_mir_schedule_probe",
        "status": "pass" if all(result.status == "pass" for result in results) else "partial",
        "llc": str(LLC),
        "llvm_objdump": str(OBJDUMP),
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
