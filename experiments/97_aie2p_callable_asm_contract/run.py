#!/usr/bin/env python3
"""Build callable AIE2P source-assembly probes and report ABI evidence."""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_PEANO = REPO_ROOT / ".venv/lib/python3.12/site-packages/llvm-aie"
DEFAULT_MLIR_AIE_INSTALL = REPO_ROOT / ".venv/lib/python3.12/site-packages/mlir_aie"
DEFAULT_LLVM_AIE_SOURCE = REPO_ROOT.parent / "llvm-aie"
DEFAULT_OUTPUT_DIR = EXPERIMENT_DIR / "build"
CPP_SOURCE = EXPERIMENT_DIR / "abi_probe.cc"
ASM_SOURCE = EXPERIMENT_DIR / "abi_probe.s"
CALLING_CONV_SOURCE = Path("llvm/lib/Target/AIE/aie2p/AIE2PCallingConv.td")
ISEL_SOURCE = Path("llvm/lib/Target/AIE/aie2p/AIE2PISelLowering.cpp")
SYMBOLS = (
    "cpp_call_asm_scalar_store",
    "cpp_call_asm_vmac_bf16_store",
    "cpp_call_asm_vmac_float_store",
    "cpp_call_asm_vector_clobber_probe",
    "cpp_accum_float_store",
    "cpp_accum_bf16_store",
    "asm_scalar_store",
    "asm_vmac_bf16_store",
    "asm_vmac_float_store",
    "asm_vector_clobber_probe",
)


@dataclass(frozen=True)
class InstructionPattern:
    label: str
    regex: str


PATTERNS = (
    InstructionPattern("tail_j", r"\bj\s+#0"),
    InstructionPattern("call_jl", r"\bjl\s+#0"),
    InstructionPattern("reloc_r_aie_1", r"R_AIE_1"),
    InstructionPattern("reloc_r_aie_2", r"R_AIE_2"),
    InstructionPattern("ret_lr", r"\bret\s+lr"),
    InstructionPattern("vlda", r"\bvlda"),
    InstructionPattern("vldb", r"\bvldb"),
    InstructionPattern("vmac_f", r"\bvmac\.f"),
    InstructionPattern("vbcst_16", r"\bvbcst\.16"),
    InstructionPattern("vextbcst_16", r"\bvextbcst\.16"),
    InstructionPattern("vst_conv_bf16_fp32", r"\bvst\.conv\.bf16\.fp32"),
    InstructionPattern("vst_accumulator", r"\bvst\s+b[mcd]"),
    InstructionPattern("vst_vector", r"\bvst\s+x"),
    InstructionPattern("scalar_store", r"\bst\s+"),
)


@dataclass(frozen=True)
class Toolchain:
    clang: Path
    clangxx: Path
    linker: Path
    nm: Path
    objdump: Path


@dataclass(frozen=True)
class BuildArtifacts:
    cpp_object: Path
    asm_object: Path
    combined_object: Path
    disasm: Path
    report: Path


@dataclass(frozen=True)
class AbiEvidence:
    callee_saved_comment: str
    csr_aie2p: str
    csr_aie2p_vec: str
    preserve_all_vec_sites: int


def run_command(cmd: tuple[str, ...]) -> str:
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "command failed:\n"
            + " ".join(cmd)
            + "\nstdout:\n"
            + completed.stdout
            + "\nstderr:\n"
            + completed.stderr
        )
    return completed.stdout


def toolchain(peano: Path) -> Toolchain:
    return Toolchain(
        clang=peano / "bin/clang",
        clangxx=peano / "bin/clang++",
        linker=peano / "bin/ld.lld",
        nm=peano / "bin/llvm-nm",
        objdump=peano / "bin/llvm-objdump",
    )


def compile_and_link(
    tools: Toolchain,
    mlir_aie_install: Path,
    output_dir: Path,
) -> BuildArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    cpp_object = output_dir / "abi_probe.cc.o"
    asm_object = output_dir / "abi_probe.s.o"
    combined_object = output_dir / "abi_probe.combined.o"
    disasm = output_dir / "abi_probe.combined.s"
    report = output_dir / "report.md"
    include_dir = mlir_aie_install / "include"
    runtime_include_dir = mlir_aie_install / "aie_runtime_lib/AIE2P"
    run_command(
        (
            str(tools.clangxx),
            "-O2",
            "-std=c++20",
            "--target=aie2p-none-unknown-elf",
            "-ffunction-sections",
            "-fdata-sections",
            f"-I{include_dir}",
            f"-I{runtime_include_dir}",
            "-c",
            str(CPP_SOURCE),
            "-o",
            str(cpp_object),
        )
    )
    run_command(
        (
            str(tools.clang),
            "--target=aie2p-none-unknown-elf",
            "-c",
            str(ASM_SOURCE),
            "-o",
            str(asm_object),
        )
    )
    run_command(
        (
            str(tools.linker),
            "-r",
            str(cpp_object),
            str(asm_object),
            "-o",
            str(combined_object),
        )
    )
    disasm.write_text(
        run_command(
            (
                str(tools.objdump),
                "--triple=aie2p",
                "-dr",
                "--no-print-imm-hex",
                str(combined_object),
            )
        )
    )
    return BuildArtifacts(
        cpp_object=cpp_object,
        asm_object=asm_object,
        combined_object=combined_object,
        disasm=disasm,
        report=report,
    )


def compact_block(text: str, pattern: str) -> str:
    match = re.search(pattern, text, re.MULTILINE | re.DOTALL)
    if match is None:
        raise ValueError(f"required source pattern not found: {pattern}")
    return " ".join(match.group(0).split())


def read_abi_evidence(llvm_aie_source: Path) -> AbiEvidence:
    calling_conv = (llvm_aie_source / CALLING_CONV_SOURCE).read_text()
    isel = (llvm_aie_source / ISEL_SOURCE).read_text()
    callee_saved = compact_block(
        calling_conv,
        r"callee_saved\s*:\s*r8, r9, r10, r11, r12, r13, r14, r15, p6, p7;",
    )
    csr_aie2p = compact_block(
        calling_conv,
        r"def CSR_AIE2P\s*: CalleeSavedRegs<\(add lr, r8, r9, r10, r11, r12, r13, r14, r15, p6, p7\)>;",
    )
    csr_aie2p_vec = compact_block(
        calling_conv,
        r"def CSR_AIE2P_Vec\s*: CalleeSavedRegs<\(add lr, r8, r9, r10, r11, r12, r13, r14, r15, p6, p7,.*?bmhh4\)>;",
    )
    return AbiEvidence(
        callee_saved_comment=callee_saved,
        csr_aie2p=csr_aie2p,
        csr_aie2p_vec=csr_aie2p_vec,
        preserve_all_vec_sites=isel.count("CallingConv::AIE_PreserveAll_Vec"),
    )


def symbol_body(disasm_text: str, symbol: str) -> str:
    start = re.search(rf"^[0-9a-fA-F]+ <{re.escape(symbol)}>:\n", disasm_text, re.MULTILINE)
    if start is None:
        raise ValueError(f"symbol not found in disassembly: {symbol}")
    next_func = re.search(
        r"^[0-9a-fA-F]+ <(?!\.)[^>]+>:\n",
        disasm_text[start.end():],
        re.MULTILINE,
    )
    if next_func is None:
        return disasm_text[start.end():]
    return disasm_text[start.end(): start.end() + next_func.start()]


def symbol_counts(disasm_text: str, symbol: str) -> list[str]:
    body = symbol_body(disasm_text, symbol)
    return [f"{pattern.label}={len(re.findall(pattern.regex, body))}" for pattern in PATTERNS]


def symbol_definition_count(nm_text: str, symbol: str) -> int:
    count = 0
    for line in nm_text.splitlines():
        fields = line.split()
        if len(fields) == 3 and fields[1] in ("T", "t") and fields[2] == symbol:
            count += 1
    return count


def unresolved_symbol_count(nm_text: str) -> int:
    count = 0
    for line in nm_text.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0] == "U":
            count += 1
    return count


def render_report(artifacts: BuildArtifacts, tools: Toolchain, abi: AbiEvidence) -> str:
    nm_text = run_command((str(tools.nm), str(artifacts.combined_object)))
    disasm_text = artifacts.disasm.read_text()
    lines = [
        "# AIE2P Callable Assembly Contract Report",
        "",
        "## Artifacts",
        "",
        f"- cpp object: `{artifacts.cpp_object}`",
        f"- asm object: `{artifacts.asm_object}`",
        f"- combined object: `{artifacts.combined_object}`",
        f"- disassembly: `{artifacts.disasm}`",
        "",
        "## ABI Source Evidence",
        "",
        f"- normal callee-saved comment: `{abi.callee_saved_comment}`",
        f"- `CSR_AIE2P`: `{abi.csr_aie2p}`",
        f"- `CSR_AIE2P_Vec` includes vector/accumulator registers: `{abi.csr_aie2p_vec}`",
        f"- `AIE_PreserveAll_Vec` sites in lowering: `{abi.preserve_all_vec_sites}`",
        "",
        "## Linked Symbols",
        "",
        f"- unresolved symbols: `{unresolved_symbol_count(nm_text)}`",
    ]
    for symbol in SYMBOLS:
        lines.append(f"- `{symbol}` definitions: `{symbol_definition_count(nm_text, symbol)}`")
    lines.extend(["", "## Per-Symbol Instruction Counts", ""])
    for symbol in SYMBOLS:
        lines.append(f"### `{symbol}`")
        lines.append("")
        for count in symbol_counts(disasm_text, symbol):
            lines.append(f"- `{count}`")
        lines.append("")
    lines.extend(
        [
            "## Conclusions",
            "",
            "- Normal AIE2P C ABI preserves scalar `r8..r15,p6,p7`; vector preserve-all is a separate convention used for selected libcalls, not a free guarantee for ordinary source-assembly calls.",
            "- C++ wrappers can carry `R_AIE_1` relocations to source-assembly symbols in a combined relocatable object, which is the viable production integration shape for one complete hot body.",
            "- A wrapper with a live vector value around an external call spills that value to local stack, so a source-assembly callee must not assume caller vector state is preserved unless the call contract explicitly says so.",
            "- BF16 accumulator output and float accumulator output lower to different storeback shapes; replacing one with the other changes both scheduling pressure and required latency padding.",
            "- AIE2P disassembly must be read with branch and return delay slots in mind; useful stores may appear after `ret lr` in objdump order.",
            "- A production Q4NX assembly body must own its vector/accumulator clobber contract and storeback latency explicitly; matching `vextbcst.16 + vmac.f` alone is not enough.",
            "",
        ]
    )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--peano", type=Path, default=DEFAULT_PEANO)
    parser.add_argument("--mlir-aie-install", type=Path, default=DEFAULT_MLIR_AIE_INSTALL)
    parser.add_argument("--llvm-aie-source", type=Path, default=DEFAULT_LLVM_AIE_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    tools = toolchain(args.peano.resolve())
    artifacts = compile_and_link(
        tools=tools,
        mlir_aie_install=args.mlir_aie_install.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    abi = read_abi_evidence(args.llvm_aie_source.resolve())
    report = render_report(artifacts, tools, abi)
    artifacts.report.write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
