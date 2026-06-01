#!/usr/bin/env python3
"""Generate MyLM Q4NX MIR through a tiny schedule DSL and resource manifest."""

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
MANIFEST = EXPERIMENT_DIR / "q4nx_schedule_dsl_resource_manifest.json"
REPORT = EXPERIMENT_DIR / "q4nx_schedule_dsl_resource_manifest.md"

VUPS4X = "      $dm1 = VUPS_4x_mv_ups_x2d_upsSign0 $x6, $s0, implicit-def $srups_of, implicit $crsat, implicit $crupsmode, implicit $upssign0"
BMHL_COPY = "      $bmhl1 = VMOV_alu_mv_mv_x $bmhl2"
BMHH_COPY = "      $bmhh1 = VMOV_alu_mv_mv_x $bmhh2"
VADD_DM2 = "      $dm2 = VADD_vmac_cm2_add_reg $dm1, $dm0, $r0"
VEXT20 = "      $x7 = VEXTBCST_16_vec_extract_broadcast_imm $x11, 20"


@dataclass(frozen=True)
class Operation:
    text: str
    kind: str


@dataclass(frozen=True)
class Bundle:
    address: int
    operations: tuple[Operation, ...]

    @property
    def signature(self) -> tuple[str, ...]:
        return tuple(operation.kind for operation in self.operations)

    @property
    def lines(self) -> tuple[str, ...]:
        return tuple(operation.text for operation in self.operations)


@dataclass(frozen=True)
class Program:
    bundles: tuple[Bundle, ...]

    def with_replacements(self, replacements: dict[int, tuple[tuple[str, ...], tuple[str, ...]]]) -> "Program":
        rows: list[Bundle] = []
        seen: set[int] = set()
        for bundle in self.bundles:
            if bundle.address in replacements:
                original, variant = replacements[bundle.address]
                if bundle.lines != original:
                    raise RuntimeError(f"unexpected bundle at {hex(bundle.address)}: {bundle.lines}")
                rows.append(Bundle(bundle.address, tuple(operation(line) for line in variant)))
                seen.add(bundle.address)
            else:
                rows.append(bundle)
        missing = tuple(sorted(set(replacements) - seen))
        if missing:
            raise RuntimeError(f"missing replacement bundles: {', '.join(hex(address) for address in missing)}")
        return Program(tuple(rows))

    def source_bundles(self) -> tuple:
        return tuple(EXP029.SourceBundle(bundle.address, bundle.lines) for bundle in self.bundles)


@dataclass(frozen=True)
class ManifestViolation:
    address: str
    signature: tuple[str, ...]
    lines: tuple[str, ...]


@dataclass(frozen=True)
class ResourceManifest:
    observed_signatures: tuple[tuple[str, ...], ...]

    def validate(self, program: Program, original: Program) -> tuple[ManifestViolation, ...]:
        observed = set(self.observed_signatures)
        violations: list[ManifestViolation] = []
        original_by_address = {bundle.address: bundle for bundle in original.bundles}
        for bundle in program.bundles:
            original_bundle = original_by_address[bundle.address]
            if bundle.lines == original_bundle.lines:
                continue
            if bundle.signature not in observed:
                violations.append(ManifestViolation(hex(bundle.address), bundle.signature, bundle.lines))
        return tuple(violations)


@dataclass(frozen=True)
class ByteDiff:
    address: str
    original: str
    generated: str


@dataclass(frozen=True)
class Candidate:
    name: str
    replacements: dict[int, tuple[tuple[str, ...], tuple[str, ...]]]
    expected_status: str


@dataclass(frozen=True)
class CandidateResult:
    name: str
    expected_status: str
    status: str
    manifest_violations: tuple[ManifestViolation, ...]
    object: str | None
    text_bytes: int | None
    matches_original: bool | None
    diff_count: int | None
    first_diffs: tuple[ByteDiff, ...]
    error_tail: str


@dataclass(frozen=True)
class ScheduleDslManifest:
    status: str
    bundle_count: int
    observed_signature_count: int
    exact_candidate: CandidateResult
    candidates: tuple[CandidateResult, ...]


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXP029 = load_module("mylm_exp029_for_schedule_dsl", EXP029_RUN)


def opcode_kind(line: str) -> str:
    stripped = line.strip()
    if stripped in {"NOP", "NOPA", "NOPB"}:
        return stripped.lower()
    rhs = stripped.split(" = ", 1)[1] if " = " in stripped else stripped
    opcode = rhs.split()[0]
    if opcode.startswith("VMOV_alu_mv_mv_x"):
        return "vmov_x"
    if opcode.startswith("VMOV_alu_mv_mv_w"):
        return "vmov_w"
    if opcode.startswith("VUPS_4x"):
        return "vups4x"
    if opcode.startswith("VUPS_2x"):
        return "vups2x"
    if opcode.startswith("VADD_"):
        return "vadd"
    if opcode.startswith("VSUB_"):
        return "vsub"
    if opcode.startswith("VMAC_"):
        return "vmac"
    if opcode.startswith("VMUL_"):
        return "vmul"
    if opcode.startswith("VCONV_"):
        return "vconv"
    if opcode.startswith("VEXTBCST_16"):
        return "vextbcst16"
    if opcode.startswith("VUNPACK"):
        return "vunpack"
    if opcode.startswith("VLDB"):
        return "vldb"
    if opcode.startswith("PADDB_"):
        return "paddb"
    return opcode


def operation(line: str) -> Operation:
    return Operation(line, opcode_kind(line))


def original_program() -> Program:
    return Program(tuple(Bundle(bundle.address, tuple(operation(line) for line in bundle.lines)) for bundle in EXP029.source_bundles()))


def resource_manifest(program: Program) -> ResourceManifest:
    return ResourceManifest(tuple(sorted(set(bundle.signature for bundle in program.bundles))))


def byte_diffs(original: bytes, generated: bytes) -> tuple[ByteDiff, ...]:
    diffs: list[ByteDiff] = []
    for offset, (left, right) in enumerate(zip(original, generated)):
        if left != right:
            diffs.append(ByteDiff(hex(EXP029.HOT_START + offset), f"{left:02x}", f"{right:02x}"))
    if len(original) != len(generated):
        diffs.append(ByteDiff("length", str(len(original)), str(len(generated))))
    return tuple(diffs)


def compile_program(name: str, program: Program) -> tuple[str, bytes, str]:
    build = BUILD_DIR / name
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    EXP029.BUILD_DIR = build
    original = EXP029.original_hot_bytes()
    source = program.source_bundles()
    unpadded = EXP029.compile_mir(f"{name}_unpadded", EXP029.unpadded_mir(source), original)
    if unpadded.llc.returncode != 0:
        raise RuntimeError(unpadded.llc.stderr)
    lengths = EXP029.encoded_bundle_lengths(unpadded, len(source))
    padded_text, _ = EXP029.padded_mir(source, lengths)
    padded = EXP029.compile_mir(f"{name}_padded", padded_text, original)
    if padded.llc.returncode != 0:
        raise RuntimeError(padded.llc.stderr)
    generated = EXP029.read_elf_section(build / f"{name}_padded.o", ".text")
    return padded.object, generated, padded.llc.stderr


def run_candidate(candidate: Candidate, original: Program, manifest: ResourceManifest) -> CandidateResult:
    program = original.with_replacements(candidate.replacements)
    violations = manifest.validate(program, original)
    if violations:
        return CandidateResult(
            candidate.name,
            candidate.expected_status,
            "rejected_by_manifest",
            violations,
            None,
            None,
            None,
            None,
            (),
            "",
        )
    try:
        object_path, generated, _ = compile_program(candidate.name, program)
    except RuntimeError as error:
        return CandidateResult(
            candidate.name,
            candidate.expected_status,
            "llc_failed",
            (),
            None,
            None,
            None,
            None,
            (),
            str(error)[-1200:],
        )
    original_bytes = EXP029.original_hot_bytes()
    diffs = byte_diffs(original_bytes, generated)
    return CandidateResult(
        candidate.name,
        candidate.expected_status,
        "encoded",
        (),
        object_path,
        len(generated),
        generated == original_bytes,
        len(diffs),
        diffs[:16],
        "",
    )


def exact_candidate(original: Program, manifest: ResourceManifest) -> CandidateResult:
    return run_candidate(Candidate("exact_replay", {}, "encoded_byte_exact"), original, manifest)


CANDIDATES = (
    Candidate(
        "highhalf_swap_045",
        {
            0x6F6: (
                (
                    "      $x2 = VCONV_bf16_fp32_mv_x_srs_bf $cml1, implicit-def $srf2fflags, implicit $crf2fmask, implicit $crrnd",
                    "      $bmhl1 = VMOV_alu_mv_mv_x $bmhl2",
                    "      $dm3 = VMAC_f_vmac_bf_vmul_bf_core_X_X $dm3, $x2, $x7, $r4, implicit-def $srfpflags, implicit $crfpmask",
                ),
                (
                    "      $x2 = VCONV_bf16_fp32_mv_x_srs_bf $cml1, implicit-def $srf2fflags, implicit $crf2fmask, implicit $crrnd",
                    "      $bmhh1 = VMOV_alu_mv_mv_x $bmhh2",
                    "      $dm3 = VMAC_f_vmac_bf_vmul_bf_core_X_X $dm3, $x2, $x7, $r4, implicit-def $srfpflags, implicit $crfpmask",
                ),
            ),
            0x700: ((BMHH_COPY,), (BMHL_COPY,)),
        },
        "encoded",
    ),
    Candidate(
        "vups_advance_046",
        {
            0x700: ((BMHH_COPY,), (VUPS4X,)),
            0x704: ((VUPS4X, VADD_DM2), (BMHH_COPY, VADD_DM2)),
        },
        "rejected_by_manifest",
    ),
    Candidate(
        "copy_and_vups_at_700_047",
        {
            0x700: ((BMHH_COPY,), (BMHH_COPY, VUPS4X)),
            0x704: ((VUPS4X, VADD_DM2), (VADD_DM2,)),
        },
        "rejected_by_manifest",
    ),
    Candidate(
        "vext_and_vups_at_6f2_047",
        {
            0x6F2: ((VEXT20,), (VEXT20, VUPS4X)),
            0x704: ((VUPS4X, VADD_DM2), (VADD_DM2,)),
        },
        "rejected_by_manifest",
    ),
)


def build_manifest() -> ScheduleDslManifest:
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    original = original_program()
    resources = resource_manifest(original)
    exact = exact_candidate(original, resources)
    results = tuple(run_candidate(candidate, original, resources) for candidate in CANDIDATES)
    checks = [
        exact.status == "encoded",
        exact.matches_original is True,
        *[result.status == result.expected_status for result in results],
    ]
    status = "passed" if all(checks) else "failed"
    return ScheduleDslManifest(status, len(original.bundles), len(resources.observed_signatures), exact, results)


def render_diffs(diffs: tuple[ByteDiff, ...]) -> str:
    if not diffs:
        return ""
    return ", ".join(f"{diff.address}:{diff.original}->{diff.generated}" for diff in diffs[:6])


def render_violations(violations: tuple[ManifestViolation, ...]) -> str:
    if not violations:
        return ""
    return "<br>".join(f"{violation.address} {violation.signature}" for violation in violations)


def render_report(manifest: ScheduleDslManifest) -> str:
    lines = [
        "# Q4NX Schedule DSL Resource Manifest",
        "",
        f"Status: `{manifest.status}`",
        "",
        f"- bundle count: `{manifest.bundle_count}`",
        f"- observed bundle signatures: `{manifest.observed_signature_count}`",
        "",
        "## Exact Replay",
        "",
        "| candidate | status | matches original | text bytes | diffs |",
        "| --- | --- | --- | ---: | --- |",
        (
            f"| `{manifest.exact_candidate.name}` | `{manifest.exact_candidate.status}` | "
            f"`{manifest.exact_candidate.matches_original}` | `{manifest.exact_candidate.text_bytes}` | "
            f"`{render_diffs(manifest.exact_candidate.first_diffs)}` |"
        ),
        "",
        "## Candidate Gates",
        "",
        "| candidate | expected | status | violations | diff count | first diffs |",
        "| --- | --- | --- | --- | ---: | --- |",
    ]
    for result in manifest.candidates:
        lines.append(
            f"| `{result.name}` | `{result.expected_status}` | `{result.status}` | "
            f"`{render_violations(result.manifest_violations)}` | `{result.diff_count}` | "
            f"`{render_diffs(result.first_diffs)}` |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The DSL round-trips the original MyLM hot loop through pre-bundled MIR and `llc` byte-exactly.",
            "- The safe high-half schedule swap from experiment 045 is accepted because its opcode-kind bundle signatures match observed MyLM signatures.",
            "- The `vups.4x` advance schedules from experiments 046/047 are rejected before encoding because they create unobserved bundle signatures.",
            "- This is a conservative first resource manifest. It should be expanded with measured legal combinations, but it already prevents blind schedule guesses from reaching NPU gates.",
            "",
        ]
    )
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
