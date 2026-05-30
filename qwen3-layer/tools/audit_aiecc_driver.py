#!/usr/bin/env python3
"""Audit which aiecc flags actually reach the Peano core compiler."""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_AIECC = REPO_ROOT / ".venv/lib/python3.12/site-packages/mlir_aie/bin/aiecc"
DEFAULT_PEANO = REPO_ROOT / ".venv/lib/python3.12/site-packages/llvm-aie"
DEFAULT_MLIR = REPO_ROOT / "qwen3-layer/build/main16-q4nx-compute-perf/design.mlir"
FAST_FAIL_MLIR = Path("/tmp/iron-aiecc-audit-fastfail.mlir")
MISSING_OBJECT = Path("/tmp/iron-aiecc-audit-missing.o")
LINK_WITH_RE = re.compile(r'link_with = "[^"]+\.o"')


@dataclass(frozen=True)
class DriverCase:
    label: str
    extra_args: tuple[str, ...]


@dataclass(frozen=True)
class DriverResult:
    label: str
    opt_command: str
    llc_command: str
    returncode: int


DRIVER_CASES = (
    DriverCase("default", ()),
    DriverCase("disable_loop_unrolling", ("--disable-loop-unrolling",)),
    DriverCase("opt_disable_loop_unroll", ("--opt-disable=loop-unroll",)),
    DriverCase(
        "llc_loop_scheduler",
        ("--aie-loop-aware", "--aie-loop-epilogue-analysis", "--aie-loop-sched-heuristics"),
    ),
    DriverCase(
        "force_hardware_loops",
        ("--aie-force-hl-gen", "--aie-hardware-loops-minitercount=1"),
    ),
    DriverCase("o0", ("-O", "0")),
    DriverCase("o3", ("-O", "3")),
)


def _run(cmd: tuple[str, ...]) -> tuple[int, str]:
    result = subprocess.run(cmd, cwd="/tmp", capture_output=True, text=True, timeout=45)
    return result.returncode, result.stdout + result.stderr


def _first_tool_command(output: str, tool_name: str) -> str:
    marker = f"/bin/{tool_name} "
    for line in output.splitlines():
        if marker in line:
            return line.strip()
    raise ValueError(f"aiecc dry-run did not print a {tool_name} command")


def _driver_command(aiecc: Path, peano: Path, mlir: Path, extra_args: tuple[str, ...]) -> tuple[str, ...]:
    return (
        str(aiecc),
        "-n",
        "-v",
        "-j1",
        *extra_args,
        "--no-compile-host",
        "--no-xchesscc",
        "--no-xbridge",
        "--alloc-scheme=basic-sequential",
        "--peano",
        str(peano),
        "--aie-generate-xclbin",
        "--xclbin-name=/tmp/iron-aiecc-audit.xclbin",
        "--xclbin-kernel-name=MLIR_AIE",
        "--aie-generate-npu-insts",
        "--npu-insts-name=/tmp/iron-aiecc-audit.txt",
        str(mlir),
    )


def _driver_version(aiecc: Path) -> str:
    returncode, output = _run((str(aiecc), "--aie-version"))
    if returncode != 0:
        raise RuntimeError(output.strip())
    return output.strip()


def _write_fast_fail_mlir(mlir: Path) -> Path:
    source = mlir.read_text()
    rewritten, replacements = LINK_WITH_RE.subn(f'link_with = "{MISSING_OBJECT}"', source)
    if replacements == 0:
        raise ValueError(f"MLIR has no link_with object to fast-fail after core command extraction: {mlir}")
    FAST_FAIL_MLIR.write_text(rewritten)
    return FAST_FAIL_MLIR


def audit_aiecc_driver(aiecc: Path, peano: Path, mlir: Path) -> str:
    if not mlir.exists():
        raise FileNotFoundError(f"missing MLIR input for dry-run audit: {mlir}")
    audit_mlir = _write_fast_fail_mlir(mlir)
    results: list[DriverResult] = []
    for case in DRIVER_CASES:
        returncode, output = _run(_driver_command(aiecc, peano, audit_mlir, case.extra_args))
        results.append(
            DriverResult(
                label=case.label,
                opt_command=_first_tool_command(output, "opt"),
                llc_command=_first_tool_command(output, "llc"),
                returncode=returncode,
            )
        )

    default = results[0]
    disable_loop = results[1]
    opt_disable = results[2]
    llc_loop_scheduler = results[3]
    force_hardware_loops = results[4]
    o0 = results[5]
    o3 = results[6]

    lines = [
        "aiecc_driver_audit:",
        f"  version={_driver_version(aiecc)}",
        f"  aiecc={aiecc}",
        f"  peano={peano}",
        f"  mlir={mlir}",
        f"  dry_run_mlir={audit_mlir}",
        "  returncode_note=1 is expected: the audit MLIR deliberately points link_with at a missing object after aiecc prints the child core commands.",
        "  dry_run_commands:",
    ]
    for result in results:
        lines.extend(
            (
                f"    {result.label}:",
                f"      returncode={result.returncode}",
                f"      opt={result.opt_command}",
                f"      llc={result.llc_command}",
            )
        )

    lines.extend(
        (
            "  flag_effects:",
            f"    disable_loop_unrolling_reaches_opt={str(disable_loop.opt_command != default.opt_command).lower()}",
            f"    opt_disable_loop_unroll_reaches_opt={str(opt_disable.opt_command != default.opt_command).lower()}",
            f"    llc_loop_scheduler_reaches_llc={str(llc_loop_scheduler.llc_command != default.llc_command).lower()}",
            f"    force_hardware_loops_reaches_llc={str(force_hardware_loops.llc_command != default.llc_command).lower()}",
            f"    o0_changes_child_pipeline={str(o0.opt_command != default.opt_command or o0.llc_command != default.llc_command).lower()}",
            f"    o3_changes_child_pipeline={str(o3.opt_command != default.opt_command or o3.llc_command != default.llc_command).lower()}",
            "  conclusion:",
            "    aiecc exposes -O to choose the child opt/llc optimization level.",
            "    aiecc does not expose a usable pass-through for disabling child opt loop unrolling.",
            "    AIE loop-scheduler and hardware-loop flags are visible in Peano llc, but this aiecc driver does not forward them to the child llc command.",
            "    production raw-main16 work should use externalized replacement ELFs or a custom aiecc build.",
        )
    )
    return "\n".join(lines) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aiecc", type=Path, default=DEFAULT_AIECC)
    parser.add_argument("--peano", type=Path, default=DEFAULT_PEANO)
    parser.add_argument("--mlir", type=Path, default=DEFAULT_MLIR)
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    print(audit_aiecc_driver(args.aiecc, args.peano, args.mlir), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
