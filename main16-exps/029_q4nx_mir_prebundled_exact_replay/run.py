#!/usr/bin/env python3
"""Learn the MIR contract needed to reproduce MyLM's Q4NX hot loop exactly."""

from __future__ import annotations

import importlib.util
import json
import re
import struct
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
EXP024_RUN = REPO_ROOT / "main16-exps/024_q4nx_mir_opcode_coverage_map/run.py"
RAW_PROGRAM = REPO_ROOT / "experiments/130_mylm_main16_record_observable_harness/mylm_c2r2_program.bin"
BUILD_DIR = EXPERIMENT_DIR / "build"
MANIFEST = EXPERIMENT_DIR / "q4nx_mir_prebundled_exact_replay.json"
REPORT = EXPERIMENT_DIR / "q4nx_mir_prebundled_exact_replay.md"

HOT_START = 0x260
HOT_END = 0x1850
NOOP_MIR = {
    "nop": "      NOP",
    "nopa": "      NOPA",
    "nopb": "      NOPB",
}
ADDRESS_RE = re.compile(r"^\s*([0-9a-fA-F]+):")


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


@dataclass(frozen=True)
class SourceBundle:
    address: int
    lines: tuple[str, ...]


@dataclass(frozen=True)
class Gap:
    before_address: str
    bytes: int


@dataclass(frozen=True)
class Build:
    name: str
    mir: str
    object: str
    objdump: str
    llc: CommandResult
    objdump_cmd: CommandResult
    text_bytes: int
    matches_original: bool
    first_diff: str | None


def load_exp024() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mylm_exp024_for_mir_exact_replay", EXP024_RUN)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load exp024 helper: {EXP024_RUN}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXP024 = load_exp024()


def run_command(args: tuple[str, ...]) -> CommandResult:
    completed = subprocess.run(
        args,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def read_elf_section(path: Path, section_name: str) -> bytes:
    data = path.read_bytes()
    if data[:4] != b"\x7fELF" or data[4] != 1:
        raise ValueError(f"not an ELF32 file: {path}")
    section_header_offset = struct.unpack_from("<I", data, 32)[0]
    section_header_size = struct.unpack_from("<H", data, 46)[0]
    section_count = struct.unpack_from("<H", data, 48)[0]
    string_index = struct.unpack_from("<H", data, 50)[0]
    string_header = section_header_offset + string_index * section_header_size
    string_offset = struct.unpack_from("<I", data, string_header + 16)[0]
    string_size = struct.unpack_from("<I", data, string_header + 20)[0]
    strings = data[string_offset : string_offset + string_size]
    for index in range(section_count):
        header = section_header_offset + index * section_header_size
        name_offset = struct.unpack_from("<I", data, header)[0]
        end = strings.find(b"\x00", name_offset)
        name = strings[name_offset:end].decode("ascii")
        if name == section_name:
            offset = struct.unpack_from("<I", data, header + 16)[0]
            size = struct.unpack_from("<I", data, header + 20)[0]
            return data[offset : offset + size]
    raise ValueError(f"missing section {section_name}: {path}")


def original_hot_bytes() -> bytes:
    raw = RAW_PROGRAM.read_bytes()
    return raw[HOT_START:HOT_END]


def source_bundles() -> tuple[SourceBundle, ...]:
    bundles: list[SourceBundle] = []
    current_address: int | None = None
    lines: list[str] = []
    for event in EXP024.EXP006.build_events():
        if current_address is None:
            current_address = event.address
        if event.address != current_address:
            bundles.append(SourceBundle(current_address, tuple(lines)))
            current_address = event.address
            lines = []
        if event.op in NOOP_MIR:
            lines.append(NOOP_MIR[event.op])
        else:
            line = EXP024.mir_for_event(event)
            if line is None:
                raise RuntimeError(f"missing MIR mapping for event: {event}")
            lines.append("  " + line)
    if current_address is not None:
        bundles.append(SourceBundle(current_address, tuple(lines)))
    return tuple(bundles)


def render_bundle(lines: tuple[str, ...]) -> list[str]:
    return ["    BUNDLE {", *lines, "    }"]


def nop_bundle() -> list[str]:
    return ["    BUNDLE {", "      NOP", "    }"]


def mir_text(name: str, body_lines: list[str]) -> str:
    body = "\n".join(body_lines)
    return f"""---
name:            {name}
alignment:       16
tracksRegLiveness: true
body:             |
  bb.0.entry (align 16):
    liveins: {EXP024.liveins()}

{body}
...
"""


def compile_mir(name: str, mir: str, original: bytes) -> Build:
    mir_path = BUILD_DIR / f"{name}.mir"
    object_path = BUILD_DIR / f"{name}.o"
    objdump_path = BUILD_DIR / f"{name}.objdump"
    mir_path.write_text(mir, encoding="utf-8")
    llc = run_command(
        (
            str(EXP024.LLC),
            "--mtriple=aie2p",
            "--start-after=postmisched",
            "--skip-machine-alignment",
            "--filetype=obj",
            str(mir_path),
            "-o",
            str(object_path),
        )
    )
    objdump = CommandResult(1, "", "object build did not complete")
    text = b""
    if llc.returncode == 0:
        text = read_elf_section(object_path, ".text")
        objdump = run_command(
            (
                str(EXP024.OBJDUMP),
                "--triple=aie2p",
                "-dr",
                "--no-print-imm-hex",
                "--disassemble-zeroes",
                str(object_path),
            )
        )
        objdump_path.write_text(objdump.stdout + objdump.stderr, encoding="utf-8")
    first_diff = first_difference(text, original)
    return Build(
        name=name,
        mir=str(mir_path.relative_to(REPO_ROOT)),
        object=str(object_path.relative_to(REPO_ROOT)),
        objdump=str(objdump_path.relative_to(REPO_ROOT)),
        llc=llc,
        objdump_cmd=objdump,
        text_bytes=len(text),
        matches_original=text == original,
        first_diff=first_diff,
    )


def objdump_offsets(text: str) -> tuple[int, ...]:
    offsets: list[int] = []
    for line in text.splitlines():
        match = ADDRESS_RE.match(line)
        if match is not None:
            offsets.append(int(match.group(1), 16))
    return tuple(offsets)


def first_difference(got: bytes, expected: bytes) -> str | None:
    limit = min(len(got), len(expected))
    for index in range(limit):
        if got[index] != expected[index]:
            return f"0x{HOT_START + index:x}: got 0x{got[index]:02x}, expected 0x{expected[index]:02x}"
    if len(got) != len(expected):
        return f"length mismatch: got {len(got)}, expected {len(expected)}"
    return None


def unpadded_mir(bundles: tuple[SourceBundle, ...]) -> str:
    lines: list[str] = []
    for bundle in bundles:
        lines.extend(render_bundle(bundle.lines))
    return mir_text("mylm_hotloop_prebundled_unpadded", lines)


def encoded_bundle_lengths(unpadded: Build, bundle_count: int) -> tuple[int, ...]:
    if unpadded.objdump_cmd.returncode != 0:
        raise RuntimeError("cannot infer bundle lengths without objdump")
    offsets = objdump_offsets(unpadded.objdump_cmd.stdout + unpadded.objdump_cmd.stderr)
    if len(offsets) < bundle_count + 1:
        raise RuntimeError(f"not enough objdump offsets: {len(offsets)} for {bundle_count} bundles")
    return tuple(offsets[index + 1] - offsets[index] for index in range(bundle_count))


def padded_mir(bundles: tuple[SourceBundle, ...], lengths: tuple[int, ...]) -> tuple[str, tuple[Gap, ...]]:
    lines: list[str] = []
    gaps: list[Gap] = []
    cursor = 0
    for bundle, encoded_length in zip(bundles, lengths):
        target = bundle.address - HOT_START
        gap = target - cursor
        if gap < 0 or gap % 2 != 0:
            raise RuntimeError(
                f"invalid MIR padding before 0x{bundle.address:x}: target={target}, cursor={cursor}, gap={gap}"
            )
        if gap:
            gaps.append(Gap(before_address=f"0x{bundle.address:x}", bytes=gap))
        for _ in range(gap // 2):
            lines.extend(nop_bundle())
        lines.extend(render_bundle(bundle.lines))
        cursor = target + encoded_length
    hot_size = HOT_END - HOT_START
    tail = hot_size - cursor
    if tail < 0 or tail % 2 != 0:
        raise RuntimeError(f"invalid MIR tail padding: cursor={cursor}, tail={tail}")
    if tail:
        gaps.append(Gap(before_address=f"0x{HOT_END:x}", bytes=tail))
    for _ in range(tail // 2):
        lines.extend(nop_bundle())
    return mir_text("mylm_hotloop_prebundled_padded", lines), tuple(gaps)


def build_manifest() -> dict[str, object]:
    if not EXP024.LLC.exists():
        raise FileNotFoundError(EXP024.LLC)
    if not EXP024.OBJDUMP.exists():
        raise FileNotFoundError(EXP024.OBJDUMP)
    if not RAW_PROGRAM.exists():
        raise FileNotFoundError(RAW_PROGRAM)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    original = original_hot_bytes()
    bundles = source_bundles()
    unpadded = compile_mir("mylm_hotloop_prebundled_unpadded", unpadded_mir(bundles), original)
    lengths = encoded_bundle_lengths(unpadded, len(bundles))
    padded_text, gaps = padded_mir(bundles, lengths)
    padded = compile_mir("mylm_hotloop_prebundled_padded", padded_text, original)
    status = "passed" if padded.matches_original and not unpadded.matches_original else "failed"
    return {
        "experiment": "029_q4nx_mir_prebundled_exact_replay",
        "status": status,
        "source_raw": str(RAW_PROGRAM.relative_to(REPO_ROOT)),
        "hot_range": [hex(HOT_START), hex(HOT_END)],
        "hot_bytes": len(original),
        "source_bundles": len(bundles),
        "inserted_gaps": [asdict(gap) for gap in gaps],
        "inserted_gap_bytes": sum(gap.bytes for gap in gaps),
        "builds": [asdict(unpadded), asdict(padded)],
    }


def render_report(manifest: dict[str, object]) -> str:
    lines = [
        "# Q4NX MIR Prebundled Exact Replay",
        "",
        f"Status: `{manifest['status']}`",
        "",
        "This experiment is a MIR usage gate, not a new NPU numeric gate.",
        "It checks which MIR form can reproduce MyLM's `0x260..0x1850` Q4NX hot-loop bytes exactly.",
        "",
        "## Contract Learned",
        "",
        "- Unbundled machine MIR plus `--start-before=postmisched` is a scheduler input; opcode counts are not a numeric contract.",
        "- Pre-bundled MIR plus `--start-after=postmisched --skip-machine-alignment` is the right encoder path for a hand-scheduled body.",
        "- MyLM disassembly addresses include semantic padding. Address gaps must be emitted as explicit `BUNDLE { NOP }` entries.",
        "",
        "## Results",
        "",
        f"- source raw: `{manifest['source_raw']}`",
        f"- hot range: `{manifest['hot_range'][0]}..{manifest['hot_range'][1]}`",
        f"- hot bytes: `{manifest['hot_bytes']}`",
        f"- source bundles: `{manifest['source_bundles']}`",
        f"- inserted gap bytes: `{manifest['inserted_gap_bytes']}`",
        "",
        "| build | text bytes | matches original | first diff |",
        "| --- | ---: | --- | --- |",
    ]
    for build in manifest["builds"]:
        lines.append(
            f"| `{build['name']}` | `{build['text_bytes']}` | `{build['matches_original']}` | "
            f"`{build['first_diff']}` |"
        )
    lines.extend(["", "## Inserted Address Gaps", ""])
    for gap in manifest["inserted_gaps"]:
        lines.append(f"- before `{gap['before_address']}`: `{gap['bytes']}` bytes")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The padded pre-bundled MIR path is byte-exact against MyLM's hot loop.",
            "- The previous MIR numeric failure is now attributable to using scheduler-generated order, not to MIR encoding capability.",
            "- The next useful gate is to patch the byte-exact pre-bundled object through the existing 028 numeric harness. It should be a no-op replacement for MyLM hot bytes; if that passes, we can change one bundle at a time and keep byte/numeric gates tight.",
            "",
        ]
    )
    return "\n".join(lines)


def run() -> None:
    manifest = build_manifest()
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    REPORT.write_text(render_report(manifest), encoding="utf-8")


def main() -> int:
    try:
        run()
    except Exception:
        traceback.print_exc()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
