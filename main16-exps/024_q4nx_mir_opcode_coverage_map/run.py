#!/usr/bin/env python3
"""Map MyLM group1 Q4NX events to AIE2P machine MIR opcodes."""

from __future__ import annotations

import importlib.util
import json
import re
import subprocess
import sys
import traceback
from collections import Counter
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
MANIFEST = EXPERIMENT_DIR / "q4nx_mir_opcode_coverage_map.json"
REPORT = EXPERIMENT_DIR / "q4nx_mir_opcode_coverage_map.md"


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class Candidate:
    name: str
    goal: str
    mir: str
    expected_counts: dict[str, int]
    require_schedule: bool


@dataclass(frozen=True)
class CandidateResult:
    name: str
    goal: str
    status: str
    llc: CommandResult
    asm: CommandResult
    objdump: CommandResult
    counts: dict[str, int]
    expected_counts: dict[str, int]
    files: dict[str, str]
    reasons: list[str]


def load_exp006() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mylm_exp006_for_mir_coverage", EXP006_RUN)
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


def group1_events() -> list:
    return [event for event in EXP006.build_events() if event.group == 1]


def parse_lane(fragment: str) -> int:
    match = re.search(r"#0x([0-9a-fA-F]+)", fragment)
    if match is None:
        raise RuntimeError(f"missing immediate lane in fragment: {fragment}")
    return int(match.group(1), 16)


def parse_signed_hex_imm(fragment: str) -> int:
    match = re.search(r"#(-?)0x([0-9a-fA-F]+)", fragment)
    if match is None:
        raise RuntimeError(f"missing signed hex immediate in fragment: {fragment}")
    value = int(match.group(2), 16)
    return -value if match.group(1) == "-" else value


def reg_class(reg: str) -> str:
    if re.fullmatch(r"x\d+", reg):
        return "x"
    if re.fullmatch(r"wl\d+|wh\d+", reg):
        return "w"
    if re.fullmatch(r"dm\d+", reg):
        return "dm"
    if re.fullmatch(r"cml\d+|cmh\d+", reg):
        return "cm"
    if re.fullmatch(r"bmll\d+|bmlh\d+|bmhl\d+|bmhh\d+|lfh\d+|lfl\d+", reg):
        return "x_alias"
    if re.fullmatch(r"p\d+", reg):
        return "p"
    if re.fullmatch(r"r\d+", reg):
        return "r"
    if re.fullmatch(r"dj\d+", reg):
        return "dj"
    if re.fullmatch(r"s\d+", reg):
        return "s"
    raise RuntimeError(f"unknown register class: {reg}")


def mir_for_event(event) -> str | None:
    defs = event.defs
    uses = event.uses
    if event.op in {"nop", "nopa", "nopb"}:
        return None
    if event.op == "lshl":
        return f"    ${defs[0]} = LSHL ${uses[0]}, ${uses[1]}"
    if event.op == "add.nc":
        if len(uses) == 2:
            return f"    ${defs[0]} = ADD_NC_mv_add_rr ${uses[0]}, ${uses[1]}"
        return f"    ${defs[0]} = ADD_NC_mv_add_ri ${uses[0]}, {parse_signed_hex_imm(event.fragment)}"
    if event.op == "movx":
        return f"    ${defs[0]} = MOVX_alu_cg {parse_signed_hex_imm(event.fragment)}"
    if event.op == "mov":
        return f"    ${defs[0]} = MOV_alu_mv_mv_mv_scl ${uses[0]}"
    if event.op == "paddb":
        return f"    ${uses[0]} = PADDB_pstm_nrm_imm ${uses[0]}, {parse_signed_hex_imm(event.fragment)}"
    if event.op == "vldb":
        imm = 64
        dst = defs[0]
        ptr = uses[0]
        if reg_class(dst) == "x":
            return f"    ${dst}, ${ptr} = VLDB_dmx_ldb_x_pstm_nrm_imm ${ptr}, {imm}"
        if reg_class(dst) == "w":
            return f"    ${dst}, ${ptr} = VLDB_dmw_ldb_pstm_nrm_imm ${ptr}, {imm}"
    if event.op == "vlda":
        imm = 64
        dst = defs[0]
        ptr = uses[0]
        if len(uses) == 2:
            return f"    ${dst} = VLDA_dmx_lda_fifohl_idx ${ptr}, ${uses[1]}"
        if reg_class(dst) == "x":
            return f"    ${dst}, ${ptr} = VLDA_dmx_lda_x_pstm_nrm_imm ${ptr}, {imm}"
        if reg_class(dst) == "w":
            return f"    ${dst}, ${ptr} = VLDA_dmw_lda_w_pstm_nrm_imm ${ptr}, {imm}"
    if event.op == "lda.s16":
        return f"    ${defs[0]}, ${uses[0]} = LDA_s16_pstm_nrm_imm ${uses[0]}, 2"
    if event.op == "vunpack":
        dst = defs[0]
        src = uses[0]
        if reg_class(dst) == "x" and reg_class(src) == "w":
            return (
                f"    ${dst} = VUNPACK_mv_unpack_w_unpackSign0 ${src}, "
                "implicit $crunpacksize, implicit $unpacksign0"
            )
    if event.op == "vups.4x":
        dst = defs[0]
        src = uses[0]
        shift = uses[1]
        if reg_class(dst) == "dm" and reg_class(src) == "x":
            return (
                f"    ${dst} = VUPS_4x_mv_ups_x2d_upsSign0 ${src}, ${shift}, "
                "implicit-def $srups_of, implicit $crsat, implicit $crupsmode, "
                "implicit $upssign0"
            )
        if reg_class(dst) == "cm" and reg_class(src) == "w":
            return (
                f"    ${dst} = VUPS_4x_mv_ups_w2c_upsSign0 ${src}, ${shift}, "
                "implicit-def $srups_of, implicit $crsat, implicit $crupsmode, "
                "implicit $upssign0"
            )
    if event.op == "vadd":
        return f"    ${defs[0]} = VADD_vmac_cm2_add_reg ${uses[0]}, ${uses[1]}, ${uses[2]}"
    if event.op == "vsub.f":
        return (
            f"    ${defs[0]} = VSUB_f_vmac_cm2_add_reg ${uses[0]}, ${uses[1]}, ${uses[2]}, "
            "implicit-def $srfpflags, implicit $crfpmask"
        )
    if event.op == "vconv.bf16.fp32":
        dst = defs[0]
        src = uses[0]
        if reg_class(dst) == "w":
            return (
                f"    ${dst} = VCONV_bf16_fp32_mv_w_srs_bf ${src}, "
                "implicit-def $srf2fflags, implicit $crf2fmask, implicit $crrnd"
            )
        if reg_class(dst) == "x":
            return (
                f"    ${dst} = VCONV_bf16_fp32_mv_x_srs_bf ${src}, "
                "implicit-def $srf2fflags, implicit $crf2fmask, implicit $crrnd"
            )
    if event.op == "vextbcst.16":
        return f"    ${defs[0]} = VEXTBCST_16_vec_extract_broadcast_imm ${uses[0]}, {parse_lane(event.fragment)}"
    if event.op == "vmul.f":
        return (
            f"    ${defs[0]} = VMUL_f_vmul_bf_vmul_bf_core_X_X ${uses[0]}, ${uses[1]}, ${uses[2]}, "
            "implicit-def $srfpflags, implicit $crfpmask"
        )
    if event.op == "vmac.f":
        return (
            f"    ${defs[0]} = VMAC_f_vmac_bf_vmul_bf_core_X_X "
            f"${uses[0]}, ${uses[1]}, ${uses[2]}, ${uses[3]}, "
            "implicit-def $srfpflags, implicit $crfpmask"
        )
    if event.op == "vmov.d":
        return f"    ${defs[0]} = VMOV_D ${uses[0]}"
    if event.op == "vmov":
        dst = defs[0]
        src = uses[0]
        dst_class = reg_class(dst)
        src_class = reg_class(src)
        if dst_class == "w" and src_class == "w":
            return f"    ${dst} = VMOV_alu_mv_mv_w ${src}"
        if dst_class == "cm" and src_class == "cm":
            return f"    ${dst} = VMOV_alu_mv_mv_cm ${src}"
        if dst_class in {"x", "x_alias"} and src_class in {"x", "x_alias"}:
            return f"    ${dst} = VMOV_alu_mv_mv_x ${src}"
    if event.op == "vbcst.16":
        return f"    ${defs[0]} = VBCST_16 ${uses[0]}"
    raise RuntimeError(f"cannot map event: {event}")


def liveins() -> str:
    regs = [
        *(f"$p{index}" for index in range(8)),
        *(f"$r{index}" for index in range(32)),
        *(f"$dj{index}" for index in range(8)),
        "$s0",
        "$lr",
        *(f"$x{index}" for index in range(12)),
        *(f"$wl{index}" for index in range(12)),
        *(f"$wh{index}" for index in range(12)),
        *(f"$dm{index}" for index in range(5)),
        *(f"$cml{index}" for index in range(5)),
        *(f"$cmh{index}" for index in range(5)),
        *(f"$bmll{index}" for index in range(5)),
        *(f"$bmlh{index}" for index in range(5)),
        *(f"$bmhl{index}" for index in range(5)),
        *(f"$bmhh{index}" for index in range(5)),
        "$lfh0",
    ]
    return ", ".join(regs)


def mir_function(name: str, body_lines: list[str], loop: bool) -> str:
    body = "\n".join(body_lines)
    if not loop:
        return f"""--- |
  define dso_local void @{name}() local_unnamed_addr {{
  entry:
    ret void
  }}

...
---
name:            {name}
alignment:       16
tracksRegLiveness: true
body:             |
  bb.0.entry (align 16):
    liveins: {liveins()}

{body}
    RET implicit $lr
    DelayedSchedBarrier

...
"""
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


def group1_projected_lines() -> list[str]:
    lines: list[str] = []
    for event in group1_events():
        line = mir_for_event(event)
        if line is not None:
            lines.append(line)
    return lines


def coverage_lines() -> list[str]:
    chosen_ops: set[str] = set()
    lines: list[str] = []
    for event in group1_events():
        if event.op == "nop":
            continue
        key = event.op
        if key == "vmov":
            key = f"vmov:{reg_class(event.defs[0])}->{reg_class(event.uses[0])}"
        if key in chosen_ops:
            continue
        chosen_ops.add(key)
        line = mir_for_event(event)
        if line is not None:
            lines.append(line)
    return lines


def expected_counts_for(lines: list[str]) -> dict[str, int]:
    text = "\n".join(lines)
    return {
        "lda_s16": len(re.findall(r"\bLDA_s16_", text)),
        "vlda": len(re.findall(r"\bVLDA_", text)),
        "vldb": len(re.findall(r"\bVLDB_", text)),
        "vunpack": len(re.findall(r"\bVUNPACK_", text)),
        "vups_4x": len(re.findall(r"\bVUPS_4x_", text)),
        "vadd": len(re.findall(r"\bVADD_", text)),
        "vsub_f": len(re.findall(r"\bVSUB_f_", text)),
        "vconv_bf16_fp32": len(re.findall(r"\bVCONV_bf16_fp32", text)),
        "vextbcst_16": len(re.findall(r"\bVEXTBCST_16", text)),
        "vbcst_16": len(re.findall(r"\bVBCST_16", text)),
        "vmul_f": len(re.findall(r"\bVMUL_f_", text)),
        "vmac_f": len(re.findall(r"\bVMAC_f_", text)),
        "vmov": len(re.findall(r"\bVMOV_alu_mv_mv_", text)),
        "vmov_d": len(re.findall(r"\bVMOV_D\b", text)),
    }


def candidates() -> tuple[Candidate, ...]:
    coverage = coverage_lines()
    group1 = group1_projected_lines()
    return (
        Candidate(
            name="opcode_coverage_block",
            goal="Compile one representative MIR instruction for every MyLM group1 opcode/register-class form.",
            mir=mir_function("opcode_coverage_block", coverage, loop=False),
            expected_counts=expected_counts_for(coverage),
            require_schedule=False,
        ),
        Candidate(
            name="mylm_group1_full_projection",
            goal="Compile the full MyLM group1 event order, excluding only explicit nops.",
            mir=mir_function("mylm_group1_full_projection", group1, loop=True),
            expected_counts=expected_counts_for(group1),
            require_schedule=False,
        ),
    )


def count_objdump(text: str) -> dict[str, int]:
    return {
        "lshl": len(re.findall(r"\blshl\b", text)),
        "add_nc": len(re.findall(r"\badd\.nc\b", text)),
        "movx": len(re.findall(r"\bmovx\b", text)),
        "mov_scl": len(re.findall(r"\bmov(?:\s|\t)", text)),
        "paddb": len(re.findall(r"\bpaddb\b", text)),
        "lda_s16": len(re.findall(r"\blda\.s16\b", text)),
        "vlda": len(re.findall(r"\bvlda(?:\s|\t)", text)),
        "vldb": len(re.findall(r"\bvldb(?:\s|\t)", text)),
        "vunpack": len(re.findall(r"\bvunpack\b", text)),
        "vups_4x": len(re.findall(r"\bvups\.4x\b", text)),
        "vadd": len(re.findall(r"\bvadd\b", text)),
        "vsub_f": len(re.findall(r"\bvsub\.f\b", text)),
        "vconv_bf16_fp32": len(re.findall(r"\bvconv\.bf16\.fp32\b", text)),
        "vextbcst_16": len(re.findall(r"\bvextbcst\.16\b", text)),
        "vextbcst_32": len(re.findall(r"\bvextbcst\.32\b", text)),
        "vbcst_16": len(re.findall(r"\bvbcst\.16\b", text)),
        "vmul_f": len(re.findall(r"\bvmul\.f\b", text)),
        "vmac_f": len(re.findall(r"\bvmac\.f\b", text)),
        "vmov": len(re.findall(r"\bvmov(?:\s|\t)", text)),
        "vmov_d": len(re.findall(r"\bvmov\.d\b", text)),
        "vst": len(re.findall(r"\bvst(?:\.|\b)", text)),
    }


def schedule_counts(text: str) -> dict[str, int]:
    bundle_matches = [int(value) for value in re.findall(r"BundleCount:\s+'(\d+)'", text)]
    loop_match = re.search(r"BasicBlock:\s+loop\s*\n\s+-\s+BundleCount:\s+'(\d+)'", text)
    return {
        "schedule_found": len(re.findall(r"Schedule found", text)),
        "schedule_missed": len(re.findall(r"No schedule found|Longest circuit does not fit II", text)),
        "bundle_count_entries": len(bundle_matches),
        "loop_bundle_count": int(loop_match.group(1)) if loop_match is not None else -1,
    }


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
    counts = count_objdump(objdump.stdout + objdump.stderr) | schedule_counts(llc.stdout + llc.stderr)
    reasons: list[str] = []
    if llc.returncode != 0:
        reasons.append(f"llc failed with return code {llc.returncode}")
    if llc.returncode == 0 and objdump.returncode != 0:
        reasons.append(f"llvm-objdump failed with return code {objdump.returncode}")
    for key, expected in candidate.expected_counts.items():
        got = counts.get(key, 0)
        if got != expected:
            reasons.append(f"expected {key}={expected}, got {got}")
    if counts["vextbcst_32"] != 0:
        reasons.append("unexpected vextbcst.32")
    if counts["vst"] != 0:
        reasons.append("unexpected vector store")
    if candidate.require_schedule and counts["schedule_found"] == 0:
        reasons.append("postpipeliner did not find a schedule")
    status = "pass" if not reasons else "fail"
    return CandidateResult(
        name=candidate.name,
        goal=candidate.goal,
        status=status,
        llc=llc,
        asm=asm,
        objdump=objdump,
        counts=counts,
        expected_counts=candidate.expected_counts,
        files={
            "mir": str(mir_path),
            "object": str(object_path),
            "asm": str(asm_path),
            "objdump": str(objdump_path),
        },
        reasons=reasons,
    )


def op_counts() -> dict[str, int]:
    return dict(Counter(event.op for event in group1_events()))


def render_report(results: list[CandidateResult]) -> str:
    lines = [
        "# Q4NX MIR Opcode Coverage Map",
        "",
        "Status: `passed`" if all(result.status == "pass" for result in results) else "Status: `failed`",
        "",
        "This experiment maps MyLM group1 Q4NX events to explicit AIE2P machine MIR.",
        "It checks backend opcode coverage before attempting a full numeric replacement kernel.",
        "",
        "## MyLM Group1 Op Counts",
        "",
    ]
    for op, count in sorted(op_counts().items()):
        lines.append(f"- `{op}`: `{count}`")
    lines.extend(["", "## Results", ""])
    for result in results:
        lines.append(f"### `{result.name}`")
        lines.append("")
        lines.append(f"Status: `{result.status}`")
        lines.append("")
        lines.append(result.goal)
        lines.append("")
        lines.append(f"- `llc_returncode`: `{result.llc.returncode}`")
        lines.append(f"- `asm_returncode`: `{result.asm.returncode}`")
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
            "- The useful part of `llvm-aie` is now concrete: it can compile a direct MIR form for the full MyLM group1 opcode vocabulary.",
            "- This avoids the C++ instruction-selection drift that produced `vups.2x`/extra `vmul.f` in experiment 020.",
            "- Passing this gate does not mean the replacement kernel is ready. The remaining work is a schedulable fill/steady/drain MIR graph with the same data dependencies and a numeric NPU comparison against MyLM direct `0x1870`.",
            "",
            "## Next Step",
            "",
            "Generate a full steady-state group pair, not one isolated group, so the postpipeliner sees the same cross-group producers and consumers that MyLM uses to hide latency.",
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
        "experiment": "024_q4nx_mir_opcode_coverage_map",
        "status": "passed" if all(result.status == "pass" for result in results) else "failed",
        "source": str(EXP006.EXP004.DEFAULT_ELF.relative_to(REPO_ROOT)),
        "group1_op_counts": op_counts(),
        "translated_group1_events": len(group1_projected_lines()),
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
