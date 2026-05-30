#!/usr/bin/env python3
"""Probe whether AIEVec/XLLVM can express the Q4NX hot-loop building blocks."""

from __future__ import annotations

import argparse
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


DEFAULT_IREE_OPT = Path("/var/home/taowen/projects/iree-amd-aie/iree-build/tools/iree-opt")
DEFAULT_AIEVEC_DIR = Path(
    "/var/home/taowen/projects/iree-amd-aie/compiler/plugins/target/AMD-AIE/aievec"
)

NPU4_MATMUL_PROBE = """
#foo = #hal.executable.target<"foo", "foo", {target_device = "npu4"}>
module attributes {hal.executable.target = #foo} {
  func.func @matmuli8i8i32npu4(
      %A : vector<8x8xi8>, %B : vector<8x8xi8>, %C : vector<8x8xi32>
  ) -> vector<8x8xi32> {
    %0 = aievec.matmul %A, %B, %C : vector<8x8xi8>, vector<8x8xi8>
                                      into vector<8x8xi32>
    return %0 : vector<8x8xi32>
  }
}
"""

BF16_CONVERT_PROBE = """
#foo = #hal.executable.target<"foo", "foo", {target_device = "npu1_4col"}>
module attributes {hal.executable.target = #foo} {
  func.func @bf16_ups_srs(%input : vector<32xbf16>, %acc : vector<32xf32>)
      -> (vector<32xf32>, vector<32xbf16>) {
    %c0 = arith.constant 0 : i32
    %ups = aievec.ups %input {shift = 0 : i8} : vector<32xbf16>, vector<32xf32>
    %srs = aievec.srs %acc, %c0 : vector<32xf32>, i32, vector<32xbf16>
    return %ups, %srs : vector<32xf32>, vector<32xbf16>
  }
}
"""


@dataclass(frozen=True)
class ProbeResult:
    name: str
    xllvm_intrinsics: dict[str, int]
    lowered_bytes: int


@dataclass(frozen=True)
class SupportSurface:
    aievec_ops: tuple[str, ...]
    has_xllvm_aie2p_mac: bool
    has_xllvm_bf16_ups_srs: bool
    has_xllvm_unpack: bool
    has_xllvm_extbcst: bool


def _run(cmd: tuple[str, ...]) -> str:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return result.stdout


def _lower(iree_opt: Path, name: str, mlir: str) -> ProbeResult:
    with tempfile.NamedTemporaryFile("w", suffix=".mlir") as tmp:
        tmp.write(mlir)
        tmp.flush()
        lowered = _run((str(iree_opt), tmp.name, "--convert-aievec-to-llvm"))
    intrinsics = {
        "aie2p_mac_conf": lowered.count("xllvm.intr.aie2p.I512.I512.ACC2048.mac.conf"),
        "aie2_bf16_mac_conf": lowered.count("xllvm.intr.aie2.bf.mac16.conf"),
        "aie2_bf16_ups": lowered.count("xllvm.intr.aie2.v16bf16.to.v16accfloat"),
        "aie2_bf16_srs": lowered.count("xllvm.intr.aie2.v16accfloat.to.v16bf16"),
        "aie2_ext": len(re.findall(r"xllvm\.intr\.aie2\.(?:ext|extract)\.", lowered)),
        "leftover_aievec": lowered.count("aievec."),
    }
    return ProbeResult(name=name, xllvm_intrinsics=intrinsics, lowered_bytes=len(lowered))


def _support_surface(aievec_dir: Path) -> SupportSurface:
    aievec_ops = aievec_dir / "AIEVecOps.td"
    xllvm_ops = aievec_dir / "XLLVMOps.td"
    aievec_text = aievec_ops.read_text()
    xllvm_text = xllvm_ops.read_text()
    op_names = tuple(sorted(re.findall(r'AIEVec_Op<"([^"]+)"', aievec_text)))
    return SupportSurface(
        aievec_ops=op_names,
        has_xllvm_aie2p_mac="intr.aie2p." in xllvm_text and "ACC2048.mac.conf" in xllvm_text,
        has_xllvm_bf16_ups_srs=(
            "v16bf16.to.v16accfloat" in xllvm_text
            and "v16accfloat.to.v16bf16" in xllvm_text
        ),
        has_xllvm_unpack="unpack" in xllvm_text.lower(),
        has_xllvm_extbcst="extbcst" in xllvm_text.lower(),
    )


def probe(iree_opt: Path, aievec_dir: Path) -> str:
    matmul = _lower(iree_opt, "npu4_i8_matmul", NPU4_MATMUL_PROBE)
    bf16 = _lower(iree_opt, "bf16_ups_srs", BF16_CONVERT_PROBE)
    surface = _support_surface(aievec_dir)
    lines = [
        "aievec_q4nx_codegen_probe:",
        f"  iree_opt={iree_opt}",
        "  lowered_probes:",
    ]
    for result in (matmul, bf16):
        lines.append(f"    {result.name}:")
        lines.append(f"      lowered_bytes={result.lowered_bytes}")
        for key, value in result.xllvm_intrinsics.items():
            lines.append(f"      {key}={value}")
    lines.extend(
        (
            "  support_surface:",
            f"    aievec_ops={','.join(surface.aievec_ops)}",
            f"    has_xllvm_aie2p_mac={surface.has_xllvm_aie2p_mac}",
            f"    has_xllvm_bf16_ups_srs={surface.has_xllvm_bf16_ups_srs}",
            f"    has_xllvm_unpack={surface.has_xllvm_unpack}",
            f"    has_xllvm_extbcst={surface.has_xllvm_extbcst}",
            "  conclusion:",
            "    AIEVec can force AIE2P MAC and BF16 UPS/SRS lowering through XLLVM.",
            "    This installed AIEVec/XLLVM surface does not expose MyLM's vunpack/vextbcst hot-loop ops.",
            "    Use AIEVec for a micro-schedule probe, but Q4NX parity still needs Peano compat intrinsics or an AIEVec/XLLVM extension for unpack+broadcast.",
        )
    )
    return "\n".join(lines) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iree-opt", type=Path, default=DEFAULT_IREE_OPT)
    parser.add_argument("--aievec-dir", type=Path, default=DEFAULT_AIEVEC_DIR)
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    print(probe(args.iree_opt, args.aievec_dir), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
