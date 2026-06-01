#!/usr/bin/env python3
"""Check whether vups.4x can be advanced without delaying the high-half copy."""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
EXP029_RUN = REPO_ROOT / "main16-exps/029_q4nx_mir_prebundled_exact_replay/run.py"
BUILD_DIR = EXPERIMENT_DIR / "build"
MANIFEST = EXPERIMENT_DIR / "q4nx_vups_preserve_copy_encoding_probe.json"
REPORT = EXPERIMENT_DIR / "q4nx_vups_preserve_copy_encoding_probe.md"

VUPS4X = "      $dm1 = VUPS_4x_mv_ups_x2d_upsSign0 $x6, $s0, implicit-def $srups_of, implicit $crsat, implicit $crupsmode, implicit $upssign0"
BMHH_COPY = "      $bmhh1 = VMOV_alu_mv_mv_x $bmhh2"
VADD_DM2 = "      $dm2 = VADD_vmac_cm2_add_reg $dm1, $dm0, $r0"
VEXT20 = "      $x7 = VEXTBCST_16_vec_extract_broadcast_imm $x11, 20"


@dataclass(frozen=True)
class Candidate:
    name: str
    replacements: dict[int, tuple[tuple[str, ...], tuple[str, ...]]]


@dataclass(frozen=True)
class CandidateResult:
    name: str
    status: str
    unpadded_returncode: int
    padded_returncode: int | None
    unpadded_text_bytes: int
    padded_text_bytes: int | None
    first_diff: str | None
    error_tail: str


@dataclass(frozen=True)
class EncodingProbeManifest:
    status: str
    results: tuple[CandidateResult, ...]


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXP029 = load_module("mylm_exp029_for_vups_preserve_copy", EXP029_RUN)


CANDIDATES = (
    Candidate(
        "copy_and_vups_at_700",
        {
            0x700: ((BMHH_COPY,), (BMHH_COPY, VUPS4X)),
            0x704: ((VUPS4X, VADD_DM2), (VADD_DM2,)),
        },
    ),
    Candidate(
        "vext_and_vups_at_6f2",
        {
            0x6F2: ((VEXT20,), (VEXT20, VUPS4X)),
            0x704: ((VUPS4X, VADD_DM2), (VADD_DM2,)),
        },
    ),
)


def candidate_bundles(candidate: Candidate) -> tuple:
    rows = []
    seen: set[int] = set()
    for bundle in EXP029.source_bundles():
        if bundle.address in candidate.replacements:
            original, variant = candidate.replacements[bundle.address]
            if bundle.lines != original:
                raise RuntimeError(f"{candidate.name}: unexpected bundle at {hex(bundle.address)}: {bundle.lines}")
            rows.append(EXP029.SourceBundle(bundle.address, variant))
            seen.add(bundle.address)
        else:
            rows.append(bundle)
    missing = tuple(sorted(set(candidate.replacements) - seen))
    if missing:
        raise RuntimeError(f"{candidate.name}: missing bundles: {', '.join(hex(address) for address in missing)}")
    return tuple(rows)


def run_candidate(candidate: Candidate) -> CandidateResult:
    build = BUILD_DIR / candidate.name
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    EXP029.BUILD_DIR = build
    original = EXP029.original_hot_bytes()
    bundles = candidate_bundles(candidate)
    unpadded = EXP029.compile_mir(f"{candidate.name}_unpadded", EXP029.unpadded_mir(bundles), original)
    if unpadded.llc.returncode != 0:
        return CandidateResult(
            candidate.name,
            "llc_unpadded_failed",
            unpadded.llc.returncode,
            None,
            unpadded.text_bytes,
            None,
            unpadded.first_diff,
            unpadded.llc.stderr[-1200:],
        )
    lengths = EXP029.encoded_bundle_lengths(unpadded, len(bundles))
    padded_text, _ = EXP029.padded_mir(bundles, lengths)
    padded = EXP029.compile_mir(f"{candidate.name}_padded", padded_text, original)
    status = "encoded" if padded.llc.returncode == 0 else "llc_padded_failed"
    return CandidateResult(
        candidate.name,
        status,
        unpadded.llc.returncode,
        padded.llc.returncode,
        unpadded.text_bytes,
        padded.text_bytes,
        padded.first_diff,
        padded.llc.stderr[-1200:],
    )


def build_manifest() -> EncodingProbeManifest:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    results = tuple(run_candidate(candidate) for candidate in CANDIDATES)
    status = "passed" if all(result.status != "encoded" for result in results) else "failed"
    return EncodingProbeManifest(status, results)


def render_report(manifest: EncodingProbeManifest) -> str:
    lines = [
        "# Q4NX VUPS Preserve-Copy Encoding Probe",
        "",
        f"Status: `{manifest.status}`",
        "",
        "| candidate | status | unpadded rc | padded rc | first diff |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for result in manifest.results:
        lines.append(
            f"| `{result.name}` | `{result.status}` | `{result.unpadded_returncode}` | "
            f"`{result.padded_returncode}` | `{result.first_diff}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `copy_and_vups_at_700` asks whether the current MIR route can issue the high-half copy and `vups.4x` in the same bundle.",
            "- `vext_and_vups_at_6f2` asks whether `vups.4x` can be paired with the earlier lane broadcast while leaving the high-half copy at `0x700`.",
            "- If both fail in the unpadded encoder, the local schedule is constrained before NPU numeric testing: preserving copy timing leaves no currently encodable early `vups.4x` slot in this small window.",
            "",
            "## Error Tails",
            "",
        ]
    )
    for result in manifest.results:
        lines.append(f"### {result.name}")
        lines.append("")
        lines.append("```text")
        lines.append(result.error_tail.strip())
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    try:
        manifest = build_manifest()
        MANIFEST.write_text(json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8")
        REPORT.write_text(render_report(manifest), encoding="utf-8")
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
