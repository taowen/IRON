#!/usr/bin/env python3
"""Replace MyLM entry with a caller stub that drives dispatcher control."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import struct
import subprocess
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
EXP130_RUN = REPO_ROOT / "experiments/130_mylm_main16_record_observable_harness/run.py"
DEFAULT_RAW = REPO_ROOT / "experiments/129_mylm_main16_standalone_raw_kernel/mylm_c2r2_program.bin"
DEFAULT_73C80 = REPO_ROOT / "experiments/129_mylm_main16_standalone_raw_kernel/mylm_static_73c80.bin"
DEFAULT_73D00 = REPO_ROOT / "experiments/129_mylm_main16_standalone_raw_kernel/mylm_static_73d00.bin"
LLVM_AIE_BIN = REPO_ROOT / ".venv/lib/python3.12/site-packages/llvm-aie/bin"
CLANG = LLVM_AIE_BIN / "clang"
LD_LLD = LLVM_AIE_BIN / "ld.lld"
LLVM_OBJDUMP = LLVM_AIE_BIN / "llvm-objdump"
XRT_SMI = Path("/var/opt/xilinx/xrt/bin/xrt-smi")
BUILD_DIR = EXPERIMENT_DIR / "build"
MANIFEST = EXPERIMENT_DIR / "mylm_main16_dispatcher_stub_probe.json"
REPORT = EXPERIMENT_DIR / "mylm_main16_dispatcher_stub_probe.md"

RECORD_DWORDS_PER_RECORD = 17


@dataclass(frozen=True)
class Variant:
    name: str
    control_value: int


@dataclass(frozen=True)
class Topology:
    status: str
    topology: str | None
    stdout: str
    stderr: str


def load_exp130() -> Any:
    spec = importlib.util.spec_from_file_location("mylm_exp130", EXP130_RUN)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load exp130 harness: {EXP130_RUN}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXP130 = load_exp130()


def default_variants() -> tuple[Variant, ...]:
    return (
        Variant("dispatcher_stub_control_0", 0),
        Variant("dispatcher_stub_control_1", 1),
        Variant("dispatcher_stub_control_8", 8),
    )


def configure_exp130_paths(build: Path) -> None:
    EXP130.RAW_COPY = build / "mylm_c2r2_program.bin"
    EXP130.STATIC_73C80_COPY = build / "mylm_static_73c80.bin"
    EXP130.STATIC_73D00_COPY = build / "mylm_static_73d00.bin"
    EXP130.ELF = build / "mylm_c2r2_dispatcher_stub_probe.elf"
    EXP130.MLIR = build / "design.mlir"
    EXP130.TXN = build / "design.txn.mlir"
    EXP130.XCLBIN = build / "design.xclbin"
    EXP130.INSTS = build / "design.bin"
    EXP130.PRJ_DIR = build / "prj"


def stub_asm(control_value: int) -> str:
    return "\n".join(
        [
            '  .section .text,"ax",@progbits',
            "  .globl __start",
            "  .type __start,@function",
            "  .p2align 4",
            "__start:",
            "  movxm sp, #0x70000",
            "  paddxm [sp], #0x80",
            "  movxm r16, #0x78200",
            "  st r16, [sp, #-4]",
            "  mov p0, r16",
            f"  mova r17, #{control_value}",
            "  st r17, [p0, #0]",
            "  movxm r12, #0x73c00",
            "  movxm r10, #0x78000",
            "  movxm r9, #0x75400",
            "  movxm r8, #0x78200",
            "  movxm p1, #0x72800",
            "  movxm p7, #0x74000",
            "  movxm p6, #0x7c000",
            "  mov p0, r12",
            "  mov p2, r10",
            "  mov p3, r9",
            "  mov p4, p7",
            "  mov p5, p6",
            "  mova r13, #0",
            "  mova r14, #0",
            "  movx r15, #-0x2",
            "  mova r11, #0",
            "  mov r0, r14",
            "  mov r1, r15",
            "  jl #0x36d0",
            "  nop",
            "  nop",
            "  nop",
            "  nop",
            "  nop",
            "  done",
            ".Lspin:",
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


def linker_script() -> str:
    return "\n".join(
        [
            "MEMORY { program (RX) : ORIGIN = 0, LENGTH = 0x200 }",
            "ENTRY(__start)",
            "SECTIONS { . = 0x0; .text : { *(.text*) } > program }",
            "",
        ]
    )


def run_cmd(cmd: tuple[str, ...], cwd: Path) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(
            "command failed:\n"
            + " ".join(cmd)
            + "\nstdout:\n"
            + result.stdout
            + "\nstderr:\n"
            + result.stderr
        )
    return result


def extract_text_section(elf: Path) -> bytes:
    data = elf.read_bytes()
    if data[:4] != b"\x7fELF" or data[4] != 1:
        raise ValueError(f"not an ELF32 file: {elf}")
    shoff = struct.unpack_from("<I", data, 32)[0]
    shentsize = struct.unpack_from("<H", data, 46)[0]
    shnum = struct.unpack_from("<H", data, 48)[0]
    shstrndx = struct.unpack_from("<H", data, 50)[0]
    shstr_header = shoff + shstrndx * shentsize
    shstr_off = struct.unpack_from("<I", data, shstr_header + 16)[0]
    shstr_size = struct.unpack_from("<I", data, shstr_header + 20)[0]
    shstr = data[shstr_off : shstr_off + shstr_size]
    for index in range(shnum):
        header = shoff + index * shentsize
        name_off = struct.unpack_from("<I", data, header)[0]
        name = shstr[name_off : shstr.find(b"\x00", name_off)].decode("ascii")
        if name == ".text":
            offset = struct.unpack_from("<I", data, header + 16)[0]
            size = struct.unpack_from("<I", data, header + 20)[0]
            return data[offset : offset + size]
    raise ValueError(f"missing .text section: {elf}")


def assemble_stub(build: Path, control_value: int) -> bytes:
    asm = build / "dispatcher_stub.s"
    obj = build / "dispatcher_stub.o"
    ld = build / "dispatcher_stub.ld"
    elf = build / "dispatcher_stub.elf"
    asm.write_text(stub_asm(control_value))
    ld.write_text(linker_script())
    run_cmd((str(CLANG), "--target=aie2p-none-unknown-elf", "-c", str(asm), "-o", str(obj)), build)
    run_cmd((str(LD_LLD), "-T", str(ld), str(obj), "-o", str(elf)), build)
    return extract_text_section(elf)


def patched_raw(raw: bytes, stub: bytes) -> bytes:
    if len(stub) >= 0x1F0:
        raise ValueError(f"stub is too large: {len(stub)} bytes")
    out = bytearray(raw)
    out[: len(stub)] = stub
    return bytes(out)


def build_variant(variant: Variant, records: int) -> dict[str, Any]:
    build = BUILD_DIR / variant.name
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    stub = assemble_stub(build, variant.control_value)
    raw = patched_raw(DEFAULT_RAW.read_bytes(), stub)
    raw_path = build / "mylm_c2r2_program.stubbed.bin"
    raw_path.write_bytes(raw)
    configure_exp130_paths(build)
    source = EXP130.write_elf(
        raw_path,
        DEFAULT_73C80,
        DEFAULT_73D00,
        preserve_dispatcher_p5=False,
    )
    EXP130.write_mlir(records)
    EXP130.package_with_aiecc()
    disasm = run_cmd(
        (
            str(LLVM_OBJDUMP),
            "-d",
            "--start-address=0x0",
            "--stop-address=0xc0",
            str(EXP130.ELF),
        ),
        build,
    ).stdout
    return {
        "build_dir": str(build.relative_to(REPO_ROOT)),
        "stub_bytes": len(stub),
        "source": source,
        "artifacts": {
            "elf": str(EXP130.ELF.relative_to(REPO_ROOT)),
            "mlir": str(EXP130.MLIR.relative_to(REPO_ROOT)),
            "txn": str(EXP130.TXN.relative_to(REPO_ROOT)),
            "xclbin": str(EXP130.XCLBIN.relative_to(REPO_ROOT)),
            "insts": str(EXP130.INSTS.relative_to(REPO_ROOT)),
        },
        "stub_disasm": EXP130._short_output(disasm, max_chars=1800),
        "transaction": {
            "payload_bytes": EXP130.txn_payload_sizes(EXP130.TXN),
            "blockwrite_addresses": [
                hex(address) for address in EXP130.txn_blockwrite_addresses(EXP130.TXN)
            ],
        },
    }


def examine_topology(timeout_s: int) -> Topology:
    result = subprocess.run(
        (str(XRT_SMI), "examine"),
        cwd=EXPERIMENT_DIR,
        timeout=timeout_s,
        capture_output=True,
        text=True,
        check=False,
    )
    text = result.stdout + result.stderr
    match = re.search(r"Topology\s*:\s*([^\n]+)", text)
    if match:
        topology = match.group(1).strip()
    else:
        table_match = re.search(r"\|\[[^\n]+\]\s*\|[^|]+\|[^|]+\|\s*([^|\s]+)\s*\|", text)
        topology = table_match.group(1).strip() if table_match else None
    if result.returncode != 0:
        status = "xrt_smi_failed"
    elif topology == "6x8":
        status = "ok"
    else:
        status = "unexpected_topology"
    return Topology(status, topology, result.stdout, result.stderr)


def run_variant(variant: Variant, records: int, runtime_timeout: int, xrt_timeout: int) -> dict[str, Any]:
    before = examine_topology(xrt_timeout)
    if before.status != "ok":
        return {
            "name": variant.name,
            "status": "skipped_topology_not_ok",
            "control_value": variant.control_value,
            "topology_before": before.__dict__,
        }
    build_info = build_variant(variant, records)
    runtime = EXP130.try_run_observable(runtime_timeout, records, activation_pong_flag=0)
    after = examine_topology(xrt_timeout)
    record_words = runtime.get("record_words") or []
    headers = record_words[0::RECORD_DWORDS_PER_RECORD]
    return {
        "name": variant.name,
        "status": runtime["status"],
        "control_value": variant.control_value,
        "topology_before": before.__dict__,
        "topology_after": after.__dict__,
        **build_info,
        "runtime_probe": {
            key: value
            for key, value in runtime.items()
            if key not in {"stdout", "stderr"}
        },
        "runtime_output": EXP130._short_output(
            (runtime.get("stdout") or "") + (runtime.get("stderr") or ""),
            max_chars=1400,
        ),
        "record_headers": headers,
        "unique_record_headers": sorted(set(headers)),
    }


def render_report(manifest: dict[str, Any]) -> str:
    lines = [
        "# MyLM Main16 Dispatcher Stub Probe",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Records per variant: `{manifest['records']}`",
        f"- Variants attempted: `{len(manifest['variants'])}`",
        "",
        "## Results",
        "",
        "| variant | control | status | unique headers | topology after |",
        "| --- | ---: | --- | --- | --- |",
    ]
    for result in manifest["variants"]:
        topology = result.get("topology_after", {}).get("topology")
        lines.append(
            f"| `{result['name']}` | `{result['control_value']}` | "
            f"`{result['status']}` | `{result.get('unique_record_headers', [])}` | `{topology}` |"
        )
    lines.extend(["", "## Interpretation", ""])
    headers = {
        result["name"]: tuple(result.get("unique_record_headers") or ())
        for result in manifest["variants"]
    }
    if headers.get("dispatcher_stub_control_1") == ("0x1",):
        lines.append(
            "The caller-side control slot successfully selected the Q/K/V dispatcher path. "
            "This gives us an observable way to enter MyLM's `0x1870` body without guessing "
            "static data words."
        )
        if headers.get("dispatcher_stub_control_8") == ("0x4",):
            lines.append(
                "`control=8` still emitted `0x4`, so this control point is not a general "
                "phase-id selector. It behaves as a QKV-vs-alternate gate: value `1` enters "
                "the `0x1870` Q/K/V body, while non-`1` values take the alternate `0x4` path."
            )
    elif headers:
        lines.append(
            "The stub did not switch the observed header to `0x1`. That means the dispatcher "
            "gate depends on more caller state than `[sp-4] -> 0x78200` plus the first control "
            "word, or the first emitted record still comes from the alternate `0x4` path before "
            "the selected phase becomes visible."
        )
    if manifest.get("stopped_early"):
        lines.append("The sweep stopped early after a timeout, XRT failure, or topology change.")
    lines.extend(["", "## Runtime Output Preview", ""])
    for result in manifest["variants"]:
        lines.extend(
            [
                f"### {result['name']}",
                "",
                "```text",
                result.get("runtime_output", ""),
                "```",
                "",
            ]
        )
    return "\n".join(lines) + "\n"


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    variants = default_variants()
    if args.max_variants > 0:
        variants = variants[: args.max_variants]
    results: list[dict[str, Any]] = []
    stopped_early = False
    for variant in variants:
        result = run_variant(variant, args.records, args.runtime_timeout, args.xrt_timeout)
        results.append(result)
        if result["status"] in {"timeout_waiting_for_record", "xrt_get_info_failed"}:
            stopped_early = True
            break
        topology_after = result.get("topology_after", {})
        if topology_after.get("status") != "ok":
            stopped_early = True
            break
    return {
        "status": "stopped_early" if stopped_early else "completed",
        "records": args.records,
        "runtime_timeout_s": args.runtime_timeout,
        "variants": results,
        "stopped_early": stopped_early,
        "source": {
            "harness": str(EXP130_RUN.relative_to(REPO_ROOT)),
            "raw": str(DEFAULT_RAW.relative_to(REPO_ROOT)),
            "static_73c80": str(DEFAULT_73C80.relative_to(REPO_ROOT)),
            "static_73d00": str(DEFAULT_73D00.relative_to(REPO_ROOT)),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=int, default=1)
    parser.add_argument("--runtime-timeout", type=int, default=8)
    parser.add_argument("--xrt-timeout", type=int, default=15)
    parser.add_argument("--max-variants", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = build_manifest(args)
    except Exception:
        failure = {
            "status": "experiment_failed",
            "traceback": traceback.format_exc(),
        }
        MANIFEST.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        REPORT.write_text(
            "# MyLM Main16 Dispatcher Stub Probe\n\nExperiment failed.\n\n```text\n"
            + failure["traceback"]
            + "```\n"
        )
        print(f"wrote {REPORT}")
        print(f"wrote {MANIFEST}")
        raise
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    REPORT.write_text(render_report(manifest))
    print(f"wrote {REPORT}")
    print(f"wrote {MANIFEST}")
    print(f"status: {manifest['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
