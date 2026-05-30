#!/usr/bin/env python3
"""Probe the aiecc core-codegen pipeline for one generated AIE core."""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


DEFAULT_PROJECT_DIR = Path("design.mlir.prj")
DEFAULT_PEANO = Path(".venv/lib/python3.12/site-packages/llvm-aie")
DEFAULT_CORE = "main_core_2_2"
Q4_CALL_SYMBOLS = (
    "q4nx_chunk_accum_slice_i32_fast",
    "q4nx_chunk_accum_block_slice_i32_fast",
    "q4nx_chunk_accum_slice_i32",
    "q4nx_chunk_accum_block_slice_i32",
)
OPT_FLAGS = ("--passes=default<O1>", "-inline-threshold=10")
DISASM_PATTERNS = {
    "jl": re.compile(r"\bjl\b"),
    "jnz": re.compile(r"\bjnz\b"),
    "acq": re.compile(r"\bacq\b"),
    "rel": re.compile(r"\brel\b"),
    "lc_ls_le": re.compile(r"\b(?:lc|ls|le)\b"),
}


@dataclass(frozen=True)
class Toolchain:
    opt: Path
    llc: Path
    clang: Path
    ld_lld: Path
    llvm_objdump: Path
    llvm_size: Path


@dataclass(frozen=True)
class CoreArtifacts:
    project_dir: Path
    core_name: str
    llvm_ir: Path
    peanohack_ir: Path
    opt_ir: Path
    object_file: Path
    elf: Path
    linker_script: Path


@dataclass(frozen=True)
class CoreMetrics:
    label: str
    llvm_q4_refs: int
    opt_q4_refs: int
    text_bytes: int
    jl: int
    jnz: int
    acq: int
    rel: int
    lc_ls_le: int


def _run(cmd: tuple[str, ...]) -> str:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout


def _toolchain(peano: Path) -> Toolchain:
    bin_dir = peano / "bin"
    return Toolchain(
        opt=bin_dir / "opt",
        llc=bin_dir / "llc",
        clang=bin_dir / "clang",
        ld_lld=bin_dir / "ld.lld",
        llvm_objdump=bin_dir / "llvm-objdump",
        llvm_size=bin_dir / "llvm-size",
    )


def _artifacts(project_dir: Path, core_name: str) -> CoreArtifacts:
    return CoreArtifacts(
        project_dir=project_dir,
        core_name=core_name,
        llvm_ir=project_dir / f"{core_name}.ll",
        peanohack_ir=project_dir / f"{core_name}.peanohack.ll",
        opt_ir=project_dir / f"{core_name}.opt.ll",
        object_file=project_dir / f"{core_name}.o",
        elf=project_dir / f"{core_name}.elf",
        linker_script=project_dir / f"{core_name}.ld.script",
    )


def _require_artifacts(artifacts: CoreArtifacts) -> None:
    for path in (
        artifacts.llvm_ir,
        artifacts.peanohack_ir,
        artifacts.opt_ir,
        artifacts.elf,
        artifacts.linker_script,
    ):
        if not path.exists():
            raise FileNotFoundError(f"missing generated core artifact: {path}")


def _count(text: str, pattern: str) -> int:
    return len(re.findall(re.escape(pattern), text))


def _count_q4_refs(text: str) -> int:
    return sum(_count(text, symbol) for symbol in Q4_CALL_SYMBOLS)


def _text_size(tools: Toolchain, elf: Path) -> int:
    output = _run((str(tools.llvm_size), "-A", str(elf)))
    for line in output.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] == ".text":
            return int(fields[1], 0)
    raise ValueError(f"missing .text size in {elf}")


def _disassemble(tools: Toolchain, elf: Path) -> str:
    return _run((str(tools.llvm_objdump), "-d", "--no-show-raw-insn", str(elf)))


def _metrics(
    label: str,
    tools: Toolchain,
    llvm_ir: Path,
    opt_ir: Path,
    elf: Path,
) -> CoreMetrics:
    llvm_text = llvm_ir.read_text()
    opt_text = opt_ir.read_text()
    disasm = _disassemble(tools, elf)
    counts = {name: len(pattern.findall(disasm)) for name, pattern in DISASM_PATTERNS.items()}
    return CoreMetrics(
        label=label,
        llvm_q4_refs=_count_q4_refs(llvm_text),
        opt_q4_refs=_count_q4_refs(opt_text),
        text_bytes=_text_size(tools, elf),
        jl=counts["jl"],
        jnz=counts["jnz"],
        acq=counts["acq"],
        rel=counts["rel"],
        lc_ls_le=counts["lc_ls_le"],
    )


def _compile_manual_disable_unroll(
    artifacts: CoreArtifacts,
    tools: Toolchain,
    output_dir: Path,
) -> CoreMetrics:
    output_dir.mkdir(parents=True, exist_ok=True)
    opt_ir = output_dir / f"{artifacts.core_name}.disable_unroll.opt.ll"
    object_file = output_dir / f"{artifacts.core_name}.disable_unroll.o"
    elf = output_dir / f"{artifacts.core_name}.disable_unroll.elf"
    _run(
        (
            str(tools.opt),
            *OPT_FLAGS,
            "-disable-loop-unrolling",
            "-S",
            str(artifacts.peanohack_ir),
            "-o",
            str(opt_ir),
        )
    )
    _run(
        (
            str(tools.llc),
            str(opt_ir),
            "-O2",
            "--march=aie2p",
            "--function-sections",
            "--filetype=obj",
            "-o",
            str(object_file),
        )
    )
    _run(
        (
            str(tools.clang),
            "-O2",
            "--target=aie2p-none-unknown-elf",
            f"-fuse-ld={tools.ld_lld.resolve()}",
            str(object_file),
            "-Wl,--gc-sections",
            "-Wl,--orphan-handling=error",
            f"-Wl,-T,{artifacts.linker_script.resolve()}",
            "-o",
            str(elf),
        )
    )
    return _metrics("manual_disable_loop_unrolling", tools, artifacts.llvm_ir, opt_ir, elf)


def _format_metrics(metrics: CoreMetrics) -> str:
    return (
        f"  {metrics.label}:\n"
        f"    llvm_q4_refs={metrics.llvm_q4_refs}\n"
        f"    opt_q4_refs={metrics.opt_q4_refs}\n"
        f"    text_bytes={metrics.text_bytes}\n"
        f"    disasm_jl={metrics.jl}\n"
        f"    disasm_jnz={metrics.jnz}\n"
        f"    disasm_acq={metrics.acq}\n"
        f"    disasm_rel={metrics.rel}\n"
        f"    disasm_lc_ls_le={metrics.lc_ls_le}"
    )


def probe(project_dir: Path, core_name: str, peano: Path, output_dir: Path | None) -> str:
    tools = _toolchain(peano)
    artifacts = _artifacts(project_dir, core_name)
    _require_artifacts(artifacts)
    metrics = [_metrics("aiecc_default", tools, artifacts.llvm_ir, artifacts.opt_ir, artifacts.elf)]
    if output_dir is not None:
        metrics.append(_compile_manual_disable_unroll(artifacts, tools, output_dir))

    lines = [
        "aiecc_core_codegen_probe:",
        f"  project_dir={project_dir}",
        f"  core={core_name}",
        "  observed_aiecc_core_pipeline:",
        "    llvm_lowering -> peanohack.ll",
        "    opt --passes=default<O1> -inline-threshold=10",
        "    llc -O2 --march=aie2p --function-sections",
        "  metrics:",
    ]
    lines.extend(_format_metrics(item) for item in metrics)
    if output_dir is not None:
        lines.append(f"  manual_probe_dir={output_dir}")
    lines.extend(
        (
            "  conclusion:",
            "    default aiecc unrolls the generated constant-trip phase loops before llc sees them.",
            "    manual opt -disable-loop-unrolling keeps the outer calls compact, but this is outside the stock aiecc core compile path.",
        )
    )
    return "\n".join(lines) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", type=Path, default=DEFAULT_PROJECT_DIR)
    parser.add_argument("--core", default=DEFAULT_CORE)
    parser.add_argument("--peano", type=Path, default=DEFAULT_PEANO)
    parser.add_argument("--manual-output-dir", type=Path, default=None)
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    print(probe(args.project_dir, args.core, args.peano, args.manual_output_dir), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
