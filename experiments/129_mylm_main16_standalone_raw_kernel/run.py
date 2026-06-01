#!/usr/bin/env python3
"""Package and runtime-probe the MyLM main16 raw whole-core program."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
LLVM_AIE_BIN = REPO_ROOT / ".venv/lib/python3.12/site-packages/llvm-aie/bin"
MLIR_AIE_BIN = REPO_ROOT / ".venv/lib/python3.12/site-packages/mlir_aie/bin"
AIECC = MLIR_AIE_BIN / "aiecc"
LLVM_SIZE = LLVM_AIE_BIN / "llvm-size"
LLVM_READELF = LLVM_AIE_BIN / "llvm-readelf"
LLVM_OBJDUMP = LLVM_AIE_BIN / "llvm-objdump"
XRT_BIN = Path("/var/opt/xilinx/xrt/bin")

DEFAULT_MYLM_RAW = Path("/tmp/mylm_qwen3_8b_layer_redump/programs/c2r2_program.bin")
DEFAULT_MYLM_DISASM = Path("/tmp/mylm_qwen3_8b_layer_redump/disasm/c2r2.s")
DEFAULT_STATIC_73C80 = Path("/tmp/mylm_qwen3_8b_layer_redump/dma/dma_0027_addr_4203c80.bin")
DEFAULT_STATIC_73D00 = Path("/tmp/mylm_qwen3_8b_layer_redump/dma/dma_0028_addr_4203d00.bin")
KNOWN_XCLBIN = REPO_ROOT / "qwen3-layer/build/main16-q4nx-compute-perf/design.xclbin"
KNOWN_INSTS = REPO_ROOT / "qwen3-layer/build/main16-q4nx-compute-perf/design.bin"

RAW_COPY = EXPERIMENT_DIR / "mylm_c2r2_program.bin"
STATIC_73C80_COPY = EXPERIMENT_DIR / "mylm_static_73c80.bin"
STATIC_73D00_COPY = EXPERIMENT_DIR / "mylm_static_73d00.bin"
ELF = EXPERIMENT_DIR / "mylm_c2r2_main16_exec.elf"
MLIR = EXPERIMENT_DIR / "design.mlir"
TXN = EXPERIMENT_DIR / "design.txn.mlir"
XCLBIN = EXPERIMENT_DIR / "design.xclbin"
INSTS = EXPERIMENT_DIR / "design.bin"
PRJ_DIR = EXPERIMENT_DIR / "prj"
MANIFEST = EXPERIMENT_DIR / "mylm_main16_standalone_raw_kernel.json"
REPORT = EXPERIMENT_DIR / "mylm_main16_standalone_raw_kernel.md"

ELF32_EHDR_SIZE = 52
ELF32_PHDR_SIZE = 32
ELF32_SHDR_SIZE = 40
EM_AIE = 0x108
EF_AIE_AIE2P = 0x3


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class Segment:
    name: str
    addr: int
    data: bytes
    sh_flags: int
    p_flags: int


def run_cmd(
    cmd: tuple[str, ...],
    cwd: Path = EXPERIMENT_DIR,
    env: dict[str, str] | None = None,
    timeout: int | None = None,
    check: bool = True,
) -> CommandResult:
    completed = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        timeout=timeout,
        capture_output=True,
        text=True,
    )
    result = CommandResult(completed.returncode, completed.stdout, completed.stderr)
    if check and completed.returncode != 0:
        raise RuntimeError(
            "command failed:\n"
            + " ".join(cmd)
            + f"\nreturncode: {completed.returncode}"
            + "\nstdout:\n"
            + completed.stdout
            + "\nstderr:\n"
            + completed.stderr
        )
    return result


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) & ~(alignment - 1)


def phdr(
    p_type: int,
    offset: int,
    vaddr: int,
    paddr: int,
    filesz: int,
    memsz: int,
    flags: int,
    align: int,
) -> bytes:
    return struct.pack("<IIIIIIII", p_type, offset, vaddr, paddr, filesz, memsz, flags, align)


def shdr(
    name: int,
    sh_type: int,
    flags: int,
    addr: int,
    offset: int,
    size: int,
    link: int,
    info: int,
    addralign: int,
    entsize: int,
) -> bytes:
    return struct.pack(
        "<IIIIIIIIII",
        name,
        sh_type,
        flags,
        addr,
        offset,
        size,
        link,
        info,
        addralign,
        entsize,
    )


def wrap_segments_as_aie_exec(segments: list[Segment], entry: int = 0) -> bytes:
    shstr = b"\x00" + b"".join(seg.name.encode("ascii") + b"\x00" for seg in segments) + b".shstrtab\x00"
    sh_name_offsets: dict[str, int] = {}
    cursor = 1
    for seg in segments:
        sh_name_offsets[seg.name] = cursor
        cursor += len(seg.name) + 1
    shstr_name_offset = cursor

    phoff = ELF32_EHDR_SIZE
    file_cursor = align_up(ELF32_EHDR_SIZE + ELF32_PHDR_SIZE * len(segments), 16)
    offsets: list[int] = []
    for seg in segments:
        file_cursor = align_up(file_cursor, 16)
        offsets.append(file_cursor)
        file_cursor += len(seg.data)
    shstr_off = file_cursor
    shoff = align_up(shstr_off + len(shstr), 4)

    out = bytearray()
    out.extend(b"\x7fELF")
    out.extend(bytes([1, 1, 1, 0]))
    out.extend(b"\x00" * 8)
    out.extend(
        struct.pack(
            "<HHIIIIIHHHHHH",
            2,
            EM_AIE,
            1,
            entry,
            phoff,
            shoff,
            EF_AIE_AIE2P,
            ELF32_EHDR_SIZE,
            ELF32_PHDR_SIZE,
            len(segments),
            ELF32_SHDR_SIZE,
            len(segments) + 2,
            len(segments) + 1,
        )
    )
    for seg, offset in zip(segments, offsets, strict=True):
        out.extend(phdr(1, offset, seg.addr, seg.addr, len(seg.data), len(seg.data), seg.p_flags, 16))
    for seg, offset in zip(segments, offsets, strict=True):
        out.extend(b"\x00" * (offset - len(out)))
        out.extend(seg.data)
    out.extend(shstr)
    out.extend(b"\x00" * (shoff - len(out)))
    out.extend(shdr(0, 0, 0, 0, 0, 0, 0, 0, 0, 0))
    for seg, offset in zip(segments, offsets, strict=True):
        out.extend(shdr(sh_name_offsets[seg.name], 1, seg.sh_flags, seg.addr, offset, len(seg.data), 0, 0, 16, 0))
    out.extend(shdr(shstr_name_offset, 3, 0, 0, shstr_off, len(shstr), 0, 0, 1, 0))
    return bytes(out)


def copy_required(src: Path, dst: Path) -> bytes:
    if not src.exists():
        raise FileNotFoundError(src)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return dst.read_bytes()


def write_elf(raw_path: Path, static_73c80: Path, static_73d00: Path) -> dict[str, Any]:
    raw = copy_required(raw_path, RAW_COPY)
    data_73c80 = copy_required(static_73c80, STATIC_73C80_COPY)
    data_73d00 = copy_required(static_73d00, STATIC_73D00_COPY)
    segments = [
        Segment(".text", 0x0, raw, 0x6, 0x5),
        Segment(".mylm_static_73c80", 0x73C80, data_73c80, 0x3, 0x6),
        Segment(".mylm_static_73d00", 0x73D00, data_73d00, 0x3, 0x6),
    ]
    ELF.write_bytes(wrap_segments_as_aie_exec(segments, entry=0))
    return {
        "raw_bytes": len(raw),
        "static_segments": [
            {"name": seg.name, "addr": hex(seg.addr), "bytes": len(seg.data)}
            for seg in segments[1:]
        ],
    }


def write_mlir() -> None:
    MLIR.write_text(
        "\n".join(
            [
                "module @mylm_main16_standalone_raw_kernel {",
                "  aie.device(npu2) {",
                "    %t22 = aie.tile(2, 2)",
                '    %activation_empty = aie.lock(%t22, 0) {init = 2 : i32, sym_name = "activation_empty"}',
                '    %activation_full = aie.lock(%t22, 1) {init = 0 : i32, sym_name = "activation_full"}',
                '    %weight_empty = aie.lock(%t22, 2) {init = 2 : i32, sym_name = "weight_empty"}',
                '    %weight_full = aie.lock(%t22, 3) {init = 0 : i32, sym_name = "weight_full"}',
                '    %record_empty = aie.lock(%t22, 4) {init = 2 : i32, sym_name = "record_empty"}',
                '    %record_full = aie.lock(%t22, 5) {init = 0 : i32, sym_name = "record_full"}',
                '    %start_lock = aie.lock(%t22, 6) {init = 0 : i32, sym_name = "start_lock"}',
                "    %c22 = aie.core(%t22) {",
                "      aie.end",
                f'    }} {{elf_file = "{ELF.name}"}}',
                "    aie.runtime_sequence() {",
                "      aiex.set_lock(%start_lock, 1)",
                "    }",
                "  }",
                "}",
                "",
            ]
        )
    )


def package_with_aiecc() -> None:
    if PRJ_DIR.exists():
        shutil.rmtree(PRJ_DIR)
    PRJ_DIR.mkdir()
    shutil.copy2(ELF, PRJ_DIR / ELF.name)
    env = os.environ.copy()
    env["PATH"] = f"{XRT_BIN}:{env.get('PATH', '')}"
    run_cmd(
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
            f"--xclbin-name={XCLBIN}",
            "--xclbin-kernel-name=MLIR_AIE",
            "--aie-generate-txn",
            f"--txn-name={TXN}",
            "--aie-generate-npu-insts",
            f"--npu-insts-name={INSTS}",
            f"--tmpdir={PRJ_DIR}",
            str(MLIR),
        ),
        env=env,
    )


def section_sizes(elf: Path) -> dict[str, int]:
    output = run_cmd((str(LLVM_SIZE), "-A", str(elf))).stdout
    sizes: dict[str, int] = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0].startswith("."):
            sizes[fields[0]] = int(fields[1], 0)
    return sizes


def elf_header(elf: Path) -> dict[str, Any]:
    output = run_cmd((str(LLVM_READELF), "-h", "-l", "-S", str(elf))).stdout
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
    text = txn.read_text() if txn.exists() else ""
    return [
        int(words) * 4
        for words in re.findall(
            r"memref\.global\s+\"private\"\s+constant\s+@config_blockwrite_data_\d+\s*:\s*memref<(\d+)xi32>",
            text,
        )
    ]


def txn_blockwrite_addresses(txn: Path) -> list[int]:
    text = txn.read_text() if txn.exists() else ""
    return [int(address) for address in re.findall(r"aiex\.npu\.blockwrite\(%\d+\) \{address = (\d+) : ui32\}", text)]


def disasm_head(elf: Path, max_lines: int = 20) -> list[str]:
    result = run_cmd(
        (str(LLVM_OBJDUMP), "-d", "--no-show-raw-insn", str(elf)),
        check=False,
    )
    lines = result.stdout.splitlines()[:max_lines]
    if result.returncode != 0:
        lines.append(f"<llvm-objdump exited {result.returncode}; stdout above is partial>")
        lines.extend(result.stderr.splitlines()[:4])
    return lines


def parse_mylm_disasm(disasm: Path) -> dict[str, Any]:
    if not disasm.exists():
        return {"available": False, "path": str(disasm)}
    text = disasm.read_text(errors="replace")
    lock_counts: dict[str, int] = {}
    for op, number in re.findall(r"\b(acq|rel)\s+#(0x[0-9a-fA-F]+|\d+)", text):
        key = f"{op} #{int(number, 0)}"
        lock_counts[key] = lock_counts.get(key, 0) + 1
    fixed_address_counts: dict[str, int] = {}
    for address in ("0x73c60", "0x73c62", "0x73c64", "0x73c68", "0x73c80", "0x73d00"):
        fixed_address_counts[address] = text.count(address)
    return {
        "available": True,
        "path": str(disasm),
        "lock_counts": lock_counts,
        "fixed_address_counts": fixed_address_counts,
    }


def try_load_run(xclbin: Path, insts: Path, timeout_s: int) -> dict[str, Any]:
    if not xclbin.exists() or not insts.exists():
        return {
            "status": "missing_artifact",
            "xclbin": str(xclbin),
            "insts": str(insts),
        }
    env = os.environ.copy()
    env["PATH"] = f"{XRT_BIN}:{env.get('PATH', '')}"
    code = f"""
import os, time, traceback
os.environ['PATH'] = {env["PATH"]!r}
import aie.utils as aie_utils
from aie.utils.npukernel import NPUKernel
try:
    kernel = NPUKernel(xclbin_path={str(xclbin)!r}, kernel_name='MLIR_AIE', insts_path={str(insts)!r})
    print('load_begin')
    handle = aie_utils.DefaultNPURuntime.load(kernel)
    print('load_ok')
    start = time.time()
    result = aie_utils.DefaultNPURuntime.run(handle, [])
    elapsed = time.time() - start
    print('run_ok')
    print('elapsed=' + str(elapsed))
    print('result_type=' + type(result).__name__)
    print('npu_time=' + str(getattr(result, 'npu_time', None)))
except Exception:
    print('exception_begin')
    traceback.print_exc()
    raise
finally:
    try:
        aie_utils.DefaultNPURuntime.cleanup()
    except Exception:
        traceback.print_exc()
"""
    started = time.time()
    try:
        result = run_cmd(
            (str(REPO_ROOT / ".venv/bin/python"), "-c", code),
            env=env,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "timeout_s": timeout_s,
            "elapsed_s": time.time() - started,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }
    output = result.stdout + result.stderr
    if result.returncode == 0 and "run_ok" in result.stdout:
        status = "run_pass"
    elif "DRM_IOCTL_AMDXDNA_GET_INFO IOCTL failed" in output:
        status = "xrt_get_info_failed"
    elif "load_ok" in result.stdout:
        status = "loaded_run_failed_or_blocked"
    else:
        status = "load_failed"
    return {
        "status": status,
        "returncode": result.returncode,
        "elapsed_s": time.time() - started,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def xrt_validate(timeout_s: int) -> dict[str, Any]:
    env = os.environ.copy()
    env["PATH"] = f"{XRT_BIN}:{env.get('PATH', '')}"
    try:
        result = run_cmd(("xrt-smi", "validate"), env=env, timeout=timeout_s, check=False)
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout",
            "timeout_s": timeout_s,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }
    output = result.stdout + result.stderr
    status = "pass" if result.returncode == 0 else "fail"
    if "DRM_IOCTL_AMDXDNA_GET_INFO IOCTL failed" in output:
        status = "xrt_get_info_failed"
    return {
        "status": status,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def classify_status(runtime: dict[str, Any], known_runtime: dict[str, Any], validate: dict[str, Any]) -> str:
    if runtime["status"] == "run_pass":
        return "run_pass"
    if (
        runtime["status"] == "xrt_get_info_failed"
        and (known_runtime["status"] == "xrt_get_info_failed" or validate["status"] == "xrt_get_info_failed")
    ):
        return "packaged_pass_runtime_blocked_by_xrt"
    if runtime["status"] == "timeout":
        return "packaged_pass_kernel_started_or_blocked"
    return "packaged_pass_runtime_failed"


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    source_info = write_elf(args.mylm_raw, args.static_73c80, args.static_73d00)
    write_mlir()
    package_with_aiecc()

    runtime = try_load_run(XCLBIN, INSTS, args.runtime_timeout)
    known_runtime = try_load_run(KNOWN_XCLBIN, KNOWN_INSTS, args.runtime_timeout)
    validate = xrt_validate(args.xrt_timeout)
    manifest: dict[str, Any] = {
        "status": "",
        "source": {
            "mylm_raw": str(args.mylm_raw),
            "static_73c80": str(args.static_73c80),
            "static_73d00": str(args.static_73d00),
            **source_info,
        },
        "artifacts": {
            "raw_copy": str(RAW_COPY.relative_to(REPO_ROOT)),
            "static_73c80_copy": str(STATIC_73C80_COPY.relative_to(REPO_ROOT)),
            "static_73d00_copy": str(STATIC_73D00_COPY.relative_to(REPO_ROOT)),
            "elf": str(ELF.relative_to(REPO_ROOT)),
            "mlir": str(MLIR.relative_to(REPO_ROOT)),
            "txn": str(TXN.relative_to(REPO_ROOT)),
            "xclbin": str(XCLBIN.relative_to(REPO_ROOT)),
            "insts": str(INSTS.relative_to(REPO_ROOT)),
            "project_dir": str(PRJ_DIR.relative_to(REPO_ROOT)),
        },
        "elf": {
            "header": elf_header(ELF),
            "sections": section_sizes(ELF),
            "disasm_head": disasm_head(ELF),
        },
        "transaction": {
            "payload_bytes": txn_payload_sizes(TXN),
            "blockwrite_addresses": [hex(address) for address in txn_blockwrite_addresses(TXN)],
            "has_start_lock_release": "aiex.set_lock(%start_lock, 1)" in MLIR.read_text()
            or "maskwrite32" in TXN.read_text(),
        },
        "xclbin": {
            "bytes": XCLBIN.stat().st_size if XCLBIN.exists() else 0,
        },
        "insts": {
            "bytes": INSTS.stat().st_size if INSTS.exists() else 0,
        },
        "mylm_disasm": parse_mylm_disasm(args.mylm_disasm),
        "runtime_probe": runtime,
        "known_kernel_runtime_probe": known_runtime,
        "xrt_validate": validate,
    }
    manifest["status"] = classify_status(runtime, known_runtime, validate)
    return manifest


def _short_output(value: str, max_chars: int = 1800) -> str:
    value = value.strip()
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "\n... truncated ..."


def render_report(manifest: dict[str, Any]) -> str:
    artifacts = manifest["artifacts"]
    txn = manifest["transaction"]
    runtime = manifest["runtime_probe"]
    known = manifest["known_kernel_runtime_probe"]
    validate = manifest["xrt_validate"]
    disasm = manifest["mylm_disasm"]
    lines = [
        "# MyLM Main16 Standalone Raw Kernel",
        "",
        f"- Status: `{manifest['status']}`",
        f"- ELF type: `{manifest['elf']['header']['type']}`",
        f"- ELF load segments: `{manifest['elf']['header']['load_segments']}`",
        f"- Sections: `{manifest['elf']['sections']}`",
        f"- Transaction payload bytes: `{txn['payload_bytes']}`",
        f"- Transaction blockwrite addresses: `{txn['blockwrite_addresses']}`",
        f"- XCLBIN bytes: `{manifest['xclbin']['bytes']}`",
        f"- Insts bytes: `{manifest['insts']['bytes']}`",
        f"- Runtime probe: `{runtime['status']}`",
        f"- Known-kernel probe: `{known['status']}`",
        f"- `xrt-smi validate`: `{validate['status']}`",
        "",
        "## What This Tests",
        "",
        "This experiment uses the raw-core route:",
        "",
        "```text",
        "MyLM c2r2 raw program bytes + fixed local static data",
        "  -> synthetic AIE2P ET_EXEC ELF with three PT_LOAD segments",
        "  -> aie.core(...){ aie.end } {elf_file = ...}",
        "  -> aiecc --no-compile xclbin/txn/insts",
        "  -> NPUKernel load/run with no host buffers",
        "```",
        "",
        "The MLIR harness only declares tile `(2,2)`, lock ids `0..6`, and releases",
        "lock id `6` at runtime. The MyLM core program uses the hardware lock numbers",
        "`48..54`; MLIR lock id `6` maps to core lock `54`, which is the start gate",
        "seen in the MyLM dispatcher.",
        "",
        "## Artifacts",
        "",
        f"- MLIR: `{artifacts['mlir']}`",
        f"- ELF: `{artifacts['elf']}`",
        f"- XCLBIN: `{artifacts['xclbin']}`",
        f"- Insts: `{artifacts['insts']}`",
        f"- TXN: `{artifacts['txn']}`",
    ]
    if disasm.get("available"):
        lines.extend(
            [
                "",
                "## Static Contract Seen In MyLM Disasm",
                "",
                f"- Lock counts: `{disasm['lock_counts']}`",
                f"- Fixed-address counts: `{disasm['fixed_address_counts']}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Runtime Result",
            "",
            f"Standalone MyLM raw kernel status: `{runtime['status']}`.",
        ]
    )
    if runtime.get("stdout") or runtime.get("stderr"):
        lines.extend(
            [
                "",
                "```text",
                _short_output((runtime.get("stdout") or "") + (runtime.get("stderr") or "")),
                "```",
            ]
        )
    lines.extend(
        [
            "",
            "Known existing xclbin control status:",
            "",
            "```text",
            _short_output((known.get("stdout") or "") + (known.get("stderr") or "")),
            "```",
            "",
            "`xrt-smi validate` status:",
            "",
            "```text",
            _short_output((validate.get("stdout") or "") + (validate.get("stderr") or "")),
            "```",
            "",
            "## Interpretation",
            "",
        ]
    )
    if manifest["status"] == "packaged_pass_runtime_blocked_by_xrt":
        lines.extend(
            [
                "The raw MyLM kernel was packaged successfully, but the NPU runtime could not",
                "load it because the current XRT/driver stack fails `GET_INFO`. The same",
                "error appears on the known existing xclbin and in `xrt-smi validate`, so this",
                "run did not reach a kernel-specific failure.",
            ]
        )
    elif manifest["status"] == "run_pass":
        lines.append("The standalone raw MyLM kernel loaded and the no-buffer run returned.")
    elif manifest["status"] == "packaged_pass_kernel_started_or_blocked":
        lines.extend(
            [
                "The package loaded far enough that the no-buffer run timed out. That is",
                "consistent with the core starting and then waiting on activation/weight",
                "locks, because this standalone harness does not feed the DMA rings.",
            ]
        )
    else:
        lines.append("Packaging succeeded, but runtime failed for a reason not classified as global XRT failure.")
    return "\n".join(lines) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mylm-raw", type=Path, default=DEFAULT_MYLM_RAW)
    parser.add_argument("--mylm-disasm", type=Path, default=DEFAULT_MYLM_DISASM)
    parser.add_argument("--static-73c80", type=Path, default=DEFAULT_STATIC_73C80)
    parser.add_argument("--static-73d00", type=Path, default=DEFAULT_STATIC_73D00)
    parser.add_argument("--runtime-timeout", type=int, default=20)
    parser.add_argument("--xrt-timeout", type=int, default=20)
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    try:
        manifest = build_manifest(args)
    except Exception:
        failure = {
            "status": "experiment_failed",
            "traceback": traceback.format_exc(),
        }
        MANIFEST.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        REPORT.write_text("# MyLM Main16 Standalone Raw Kernel\n\nExperiment failed.\n\n```text\n" + failure["traceback"] + "```\n")
        print(f"wrote {REPORT}")
        print(f"wrote {MANIFEST}")
        raise
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    REPORT.write_text(render_report(manifest))
    print(f"wrote {REPORT}")
    print(f"wrote {MANIFEST}")
    print(f"status: {manifest['status']}")
    return 0 if manifest["status"].startswith("packaged_pass") or manifest["status"] == "run_pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
