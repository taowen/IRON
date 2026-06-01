#!/usr/bin/env python3
"""Probe the raw whole-core AIE2P program/codegen route."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
LLVM_AIE_BIN = REPO_ROOT / ".venv/lib/python3.12/site-packages/llvm-aie/bin"
MLIR_AIE_BIN = REPO_ROOT / ".venv/lib/python3.12/site-packages/mlir_aie/bin"
TOOLS_DIR = REPO_ROOT / "qwen3-layer/tools"
EXP127_OBJECT = REPO_ROOT / "experiments/127_main16_whole_program_scaffold/main16_whole_program_scaffold.o"
DEFAULT_MYLM_RAW = Path("/tmp/mylm_qwen3_8b_layer_redump/programs/c2r2_program.bin")

CLANG = LLVM_AIE_BIN / "clang"
LD_LLD = LLVM_AIE_BIN / "ld.lld"
LLVM_READELF = LLVM_AIE_BIN / "llvm-readelf"
LLVM_SIZE = LLVM_AIE_BIN / "llvm-size"
LLVM_OBJDUMP = LLVM_AIE_BIN / "llvm-objdump"
AIE_OPT = MLIR_AIE_BIN / "aie-opt"
AIECC = MLIR_AIE_BIN / "aiecc"
XRT_BIN = Path("/var/opt/xilinx/xrt/bin")

REPORT = EXPERIMENT_DIR / "raw_core_program_codegen_feasibility.md"
MANIFEST = EXPERIMENT_DIR / "raw_core_program_codegen_feasibility.json"

RAW_ASM = EXPERIMENT_DIR / "raw_codegen_probe.s"
RAW_LD = EXPERIMENT_DIR / "raw_codegen_probe.ld"
RAW_OBJECT = EXPERIMENT_DIR / "raw_codegen_probe.o"
RAW_ELF = EXPERIMENT_DIR / "raw_codegen_probe.elf"
RAW_MLIR = EXPERIMENT_DIR / "raw_codegen_probe.mlir"
RAW_TXN = EXPERIMENT_DIR / "raw_codegen_probe.txn.mlir"
REL_MLIR = EXPERIMENT_DIR / "raw_codegen_probe_rel.mlir"
REL_TXN = EXPERIMENT_DIR / "raw_codegen_probe_rel.txn.mlir"

SCAFFOLD_LD = EXPERIMENT_DIR / "main16_scaffold_whole_core.ld"
SCAFFOLD_ELF = EXPERIMENT_DIR / "main16_scaffold_whole_core.elf"
SCAFFOLD_MLIR = EXPERIMENT_DIR / "main16_scaffold_whole_core.mlir"
SCAFFOLD_TXN = EXPERIMENT_DIR / "main16_scaffold_whole_core.txn.mlir"

MYLM_RAW_COPY = EXPERIMENT_DIR / "mylm_c2r2_program.bin"
MYLM_ELF = EXPERIMENT_DIR / "mylm_c2r2_exec.elf"
MYLM_MLIR = EXPERIMENT_DIR / "mylm_c2r2_exec.mlir"
MYLM_TXN = EXPERIMENT_DIR / "mylm_c2r2_exec.txn.mlir"

AIECC_PROJECT_DIR = EXPERIMENT_DIR / "aiecc_raw_probe_prj"
AIECC_TXN = EXPERIMENT_DIR / "raw_codegen_probe.aiecc.txn.mlir"
AIECC_XCLBIN = EXPERIMENT_DIR / "raw_codegen_probe.xclbin"


@dataclass(frozen=True)
class CommandResult:
    stdout: str
    stderr: str


def run(
    cmd: tuple[str, ...],
    cwd: Path = EXPERIMENT_DIR,
    env: dict[str, str] | None = None,
) -> CommandResult:
    completed = subprocess.run(cmd, cwd=cwd, env=env, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "command failed:\n"
            + " ".join(cmd)
            + "\nstdout:\n"
            + completed.stdout
            + "\nstderr:\n"
            + completed.stderr
        )
    return CommandResult(stdout=completed.stdout, stderr=completed.stderr)


def write_raw_probe_sources() -> None:
    RAW_ASM.write_text(
        "\n".join(
            [
                '  .section .text,"ax",@progbits',
                "  .globl __start",
                "  .type __start,@function",
                "  .p2align 4",
                "__start:",
                "  mova r0, #7",
                "  mova r1, #35",
                ".Lspin:",
                "  add r0, r0, #1",
                "  j #.Lspin",
                "  nop",
                "  nop",
                "  nop",
                "  nop",
                "  nop",
                "  .size __start, .-__start",
                "",
            ]
        )
    )
    RAW_LD.write_text(
        "\n".join(
            [
                "MEMORY { program (RX) : ORIGIN = 0, LENGTH = 0x20000 }",
                "ENTRY(__start)",
                "SECTIONS { . = 0x0; .text : { *(.text*) } > program }",
                "",
            ]
        )
    )
    RAW_MLIR.write_text(elf_core_mlir("raw_codegen_probe.elf"))
    REL_MLIR.write_text(elf_core_mlir("raw_codegen_probe.o"))


def write_scaffold_sources() -> None:
    SCAFFOLD_LD.write_text(
        "\n".join(
            [
                "MEMORY {",
                "  program (RX) : ORIGIN = 0, LENGTH = 0x20000",
                "  data (!RX) : ORIGIN = 0x7C200, LENGTH = 0x3E00",
                "}",
                "ENTRY(q4nx_main16_whole_program_entry)",
                "SECTIONS {",
                "  . = 0x0;",
                "  .text : {",
                "    *(.text.q4nx_main16_whole_program_entry)",
                "    *(.text*)",
                "  } > program",
                "  .data : { *(.data*) *(.rodata*) } > data",
                "  .bss : { *(.bss*) } > data",
                "}",
                "",
            ]
        )
    )
    SCAFFOLD_MLIR.write_text(elf_core_mlir("main16_scaffold_whole_core.elf"))


def elf_core_mlir(elf_name: str) -> str:
    return "\n".join(
        [
            "module @raw_core_codegen_probe {",
            "  aie.device(npu2) {",
            "    %t22 = aie.tile(2, 2)",
            "    %c22 = aie.core(%t22) {",
            "      aie.end",
            f'    }} {{elf_file = "{elf_name}"}}',
            "  }",
            "}",
            "",
        ]
    )


def compile_raw_probe() -> None:
    run((str(CLANG), "--target=aie2p-none-unknown-elf", "-c", str(RAW_ASM), "-o", str(RAW_OBJECT)))
    run((str(LD_LLD), "-T", str(RAW_LD), str(RAW_OBJECT), "-o", str(RAW_ELF)))


def link_scaffold() -> bool:
    if not EXP127_OBJECT.exists():
        return False
    run((str(LD_LLD), "-T", str(SCAFFOLD_LD), str(EXP127_OBJECT), "-o", str(SCAFFOLD_ELF)))
    return True


def convert_to_transaction(mlir: Path, txn: Path) -> None:
    result = run(
        (
            str(AIE_OPT),
            f"--convert-aie-to-transaction=elf-dir={EXPERIMENT_DIR}",
            str(mlir),
        )
    )
    txn.write_text(result.stdout)


def text_size(elf: Path) -> int:
    output = run((str(LLVM_SIZE), "-A", str(elf))).stdout
    for line in output.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0] == ".text":
            return int(fields[1], 0)
    raise ValueError(f"missing .text section in {elf}")


def section_sizes(elf: Path) -> dict[str, int]:
    output = run((str(LLVM_SIZE), "-A", str(elf))).stdout
    sizes: dict[str, int] = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0].startswith("."):
            sizes[fields[0]] = int(fields[1], 0)
    return sizes


def elf_header(elf: Path) -> dict[str, Any]:
    output = run((str(LLVM_READELF), "-h", "-l", "-S", str(elf))).stdout
    elf_type = re.search(r"Type:\s+(\S+)", output)
    entry = re.search(r"Entry point address:\s+(0x[0-9a-fA-F]+)", output)
    program_headers = re.search(r"Number of program headers:\s+(\d+)", output)
    load_segments = len(re.findall(r"^\s+LOAD\s", output, re.MULTILINE))
    return {
        "type": elf_type.group(1) if elf_type else None,
        "entry": entry.group(1) if entry else None,
        "program_headers": int(program_headers.group(1)) if program_headers else None,
        "load_segments": load_segments,
    }


def txn_payload_sizes(txn: Path) -> list[int]:
    text = txn.read_text()
    return [
        int(words) * 4
        for words in re.findall(
            r"memref\.global\s+\"private\"\s+constant\s+@config_blockwrite_data_\d+\s*:\s*memref<(\d+)xi32>",
            text,
        )
    ]


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def disasm_head(elf: Path, max_lines: int = 20) -> list[str]:
    output = run((str(LLVM_OBJDUMP), "-d", "--no-show-raw-insn", str(elf))).stdout
    return output.splitlines()[:max_lines]


def wrap_mylm_raw(raw_path: Path) -> bool:
    if not raw_path.exists():
        return False
    sys.path.insert(0, str(TOOLS_DIR))
    from wrap_raw_aie_program import wrap_raw_program

    shutil.copy2(raw_path, MYLM_RAW_COPY)
    MYLM_ELF.write_bytes(wrap_raw_program(MYLM_RAW_COPY.read_bytes(), text_addr=0))
    MYLM_MLIR.write_text(elf_core_mlir("mylm_c2r2_exec.elf"))
    convert_to_transaction(MYLM_MLIR, MYLM_TXN)
    return True


def package_tiny_with_aiecc() -> None:
    if AIECC_PROJECT_DIR.exists():
        shutil.rmtree(AIECC_PROJECT_DIR)
    AIECC_PROJECT_DIR.mkdir()
    shutil.copy2(RAW_ELF, AIECC_PROJECT_DIR / RAW_ELF.name)
    env = os.environ.copy()
    env["PATH"] = f"{XRT_BIN}:{env.get('PATH', '')}"
    run(
        (
            str(AIECC),
            "-j1",
            "--no-compile",
            "--no-compile-host",
            "--no-xchesscc",
            "--no-xbridge",
            "--alloc-scheme=basic-sequential",
            "--peano",
            str(REPO_ROOT / ".venv/lib/python3.12/site-packages/llvm-aie"),
            "--aie-generate-xclbin",
            f"--xclbin-name={AIECC_XCLBIN}",
            "--xclbin-kernel-name=MLIR_AIE",
            "--aie-generate-txn",
            f"--txn-name={AIECC_TXN}",
            f"--tmpdir={AIECC_PROJECT_DIR}",
            str(RAW_MLIR),
        ),
        env=env,
    )


def build_manifest(mylm_raw: Path) -> dict[str, Any]:
    write_raw_probe_sources()
    write_scaffold_sources()
    compile_raw_probe()
    convert_to_transaction(RAW_MLIR, RAW_TXN)
    convert_to_transaction(REL_MLIR, REL_TXN)
    package_tiny_with_aiecc()
    scaffold_linked = link_scaffold()
    if scaffold_linked:
        convert_to_transaction(SCAFFOLD_MLIR, SCAFFOLD_TXN)
    mylm_wrapped = wrap_mylm_raw(mylm_raw)

    manifest: dict[str, Any] = {
        "status": "pass",
        "tools": {
            "clang": str(CLANG.relative_to(REPO_ROOT)),
            "ld_lld": str(LD_LLD.relative_to(REPO_ROOT)),
            "aie_opt": str(AIE_OPT.relative_to(REPO_ROOT)),
            "aiecc": str(AIECC.relative_to(REPO_ROOT)),
            "wrap_raw_aie_program": str((TOOLS_DIR / "wrap_raw_aie_program.py").relative_to(REPO_ROOT)),
        },
        "tiny_source_asm_exec": {
            "asm": str(RAW_ASM.relative_to(REPO_ROOT)),
            "object": str(RAW_OBJECT.relative_to(REPO_ROOT)),
            "elf": str(RAW_ELF.relative_to(REPO_ROOT)),
            "elf_header": elf_header(RAW_ELF),
            "sections": section_sizes(RAW_ELF),
            "txn": str(RAW_TXN.relative_to(REPO_ROOT)),
            "txn_payload_bytes": txn_payload_sizes(RAW_TXN),
            "disasm_head": disasm_head(RAW_ELF),
        },
        "et_rel_negative_control": {
            "object": str(RAW_OBJECT.relative_to(REPO_ROOT)),
            "mlir": str(REL_MLIR.relative_to(REPO_ROOT)),
            "txn": str(REL_TXN.relative_to(REPO_ROOT)),
            "txn_payload_bytes": txn_payload_sizes(REL_TXN),
            "expected": "no config_blockwrite payloads from ET_REL",
        },
        "aiecc_no_compile_package": {
            "project_dir": str(AIECC_PROJECT_DIR.relative_to(REPO_ROOT)),
            "txn": str(AIECC_TXN.relative_to(REPO_ROOT)),
            "xclbin": str(AIECC_XCLBIN.relative_to(REPO_ROOT)),
            "xclbin_bytes": AIECC_XCLBIN.stat().st_size if AIECC_XCLBIN.exists() else 0,
            "txn_payload_bytes": txn_payload_sizes(AIECC_TXN) if AIECC_TXN.exists() else [],
        },
        "main16_scaffold_exec": {
            "available": scaffold_linked,
        },
        "mylm_raw_exec": {
            "available": mylm_wrapped,
            "raw_input": str(mylm_raw),
        },
        "verdict": {
            "raw_asm_to_exec_elf": elf_header(RAW_ELF)["type"] == "EXEC" and text_size(RAW_ELF) > 0,
            "exec_elf_to_transaction_payload": txn_payload_sizes(RAW_TXN) == [align_up(text_size(RAW_ELF), 4)],
            "aiecc_no_compile_xclbin_package": (
                AIECC_XCLBIN.exists()
                and AIECC_XCLBIN.stat().st_size > 0
                and txn_payload_sizes(AIECC_TXN) == [align_up(text_size(RAW_ELF), 4)]
            ),
            "relocatable_object_is_not_enough": txn_payload_sizes(REL_TXN) == [],
            "main16_scaffold_can_be_whole_core_elf": scaffold_linked and bool(txn_payload_sizes(SCAFFOLD_TXN)),
            "mylm_raw_bytes_can_be_wrapped_as_exec": mylm_wrapped and bool(txn_payload_sizes(MYLM_TXN)),
        },
    }
    if scaffold_linked:
        manifest["main16_scaffold_exec"].update(
            {
                "source_object": str(EXP127_OBJECT.relative_to(REPO_ROOT)),
                "elf": str(SCAFFOLD_ELF.relative_to(REPO_ROOT)),
                "elf_header": elf_header(SCAFFOLD_ELF),
                "sections": section_sizes(SCAFFOLD_ELF),
                "txn": str(SCAFFOLD_TXN.relative_to(REPO_ROOT)),
                "txn_payload_bytes": txn_payload_sizes(SCAFFOLD_TXN),
            }
        )
    if mylm_wrapped:
        manifest["mylm_raw_exec"].update(
            {
                "raw_copy": str(MYLM_RAW_COPY.relative_to(REPO_ROOT)),
                "elf": str(MYLM_ELF.relative_to(REPO_ROOT)),
                "elf_header": elf_header(MYLM_ELF),
                "sections": section_sizes(MYLM_ELF),
                "txn": str(MYLM_TXN.relative_to(REPO_ROOT)),
                "txn_payload_bytes": txn_payload_sizes(MYLM_TXN),
            }
        )
    if not all(manifest["verdict"].values()):
        optional_keys = {"mylm_raw_bytes_can_be_wrapped_as_exec"}
        hard_failures = [
            key
            for key, value in manifest["verdict"].items()
            if not value and key not in optional_keys
        ]
        if hard_failures:
            manifest["status"] = "fail"
            manifest["hard_failures"] = hard_failures
        elif not mylm_wrapped:
            manifest["status"] = "pass_mylm_raw_skipped"
    return manifest


def render_report(manifest: dict[str, Any]) -> str:
    tiny = manifest["tiny_source_asm_exec"]
    rel = manifest["et_rel_negative_control"]
    package = manifest["aiecc_no_compile_package"]
    scaffold = manifest["main16_scaffold_exec"]
    mylm = manifest["mylm_raw_exec"]
    verdict = manifest["verdict"]
    lines = [
        "# Raw Core Program Codegen Feasibility",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Tiny asm ELF type: `{tiny['elf_header']['type']}`",
        f"- Tiny asm `.text`: `{tiny['sections'].get('.text', 0)}` bytes",
        f"- Tiny asm transaction payloads: `{tiny['txn_payload_bytes']}`",
        f"- Tiny asm aiecc xclbin bytes: `{package['xclbin_bytes']}`",
        f"- ET_REL negative-control payloads: `{rel['txn_payload_bytes']}`",
        f"- Main16 scaffold available: `{scaffold['available']}`",
        f"- MyLM raw available: `{mylm['available']}`",
        "",
        "## Verdict",
        "",
        "| Check | Pass |",
        "| --- | --- |",
    ]
    for key, value in verdict.items():
        lines.append(f"| `{key}` | `{value}` |")
    lines.extend(
        [
            "",
            "## Toolchain Route",
            "",
            "The viable route is:",
            "",
            "```text",
            "Python/codegen AIE2P asm or raw bytes",
            "  -> Peano clang/ld.lld ET_EXEC whole-core ELF",
            "  -> aie.core { aie.end } { elf_file = ... }",
            "  -> aie-opt/aiecc --no-compile transaction/xclbin packaging",
            "```",
            "",
            "The negative control matters: a relocatable object can pass through the",
            "MLIR verifier, but it does not generate a core-program blockwrite payload.",
            "So production raw main16 must hand packaging an executable ELF with loadable",
            "segments, not just a normal callable object.",
            "",
            "The `aiecc --no-compile` probe produced an xclbin from the tiny external",
            f"ELF without recompiling it: `{package['xclbin']}`, {package['xclbin_bytes']} bytes.",
        ]
    )
    if scaffold["available"]:
        lines.extend(
            [
                "",
                "## Main16 Scaffold",
                "",
                f"- ELF: `{scaffold['elf']}`",
                f"- Sections: `{scaffold['sections']}`",
                f"- Transaction payloads: `{scaffold['txn_payload_bytes']}`",
                "",
                "This proves the current exp127 generated source asm can be promoted from",
                "ET_REL object to a whole-core executable. Its BSS accumulator also becomes",
                "an explicit zero-fill blockwrite, so fixed local-memory symbols must be",
                "managed deliberately in the linker script.",
            ]
        )
    if mylm["available"]:
        lines.extend(
            [
                "",
                "## MyLM Raw Bytes",
                "",
                f"- Raw input: `{mylm['raw_input']}`",
                f"- Wrapped ELF: `{mylm['elf']}`",
                f"- Sections: `{mylm['sections']}`",
                f"- Transaction payloads: `{mylm['txn_payload_bytes']}`",
                "",
                "This checks the fallback route where our generator emits raw AIE2P bytes",
                "directly and only uses an ELF wrapper for MLIR-AIE/aiebu packaging.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## MyLM Raw Bytes",
                "",
                f"Skipped because `{mylm['raw_input']}` was not present.",
            ]
        )
    return "\n".join(lines) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mylm-raw", type=Path, default=DEFAULT_MYLM_RAW)
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    manifest = build_manifest(args.mylm_raw)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    REPORT.write_text(render_report(manifest))
    print(f"wrote {REPORT}")
    print(f"wrote {MANIFEST}")
    return 0 if manifest["status"].startswith("pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
