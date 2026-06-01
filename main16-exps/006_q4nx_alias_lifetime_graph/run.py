#!/usr/bin/env python3
"""Build an alias-aware lifetime graph for MyLM's main16 Q4NX hot loop."""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
EXP004_RUN = REPO_ROOT / "main16-exps/004_q4nx_hot_body_schedule/run.py"
MANIFEST = EXPERIMENT_DIR / "q4nx_alias_lifetime_graph.json"
REPORT = EXPERIMENT_DIR / "q4nx_alias_lifetime_graph.md"

REGISTER_RE = re.compile(
    r"\b(?:r\d+|p\d+|x\d+|y\d+|wl\d+|wh\d+|dm\d+|cml\d+|cmh\d+|"
    r"bmll\d+|bmlh\d+|bmhl\d+|bmhh\d+|lfh\d+|m\d+|dj\d+|lr|lc|ls|le|"
    r"el0|crrnd|crupsmode|s0|vaddsign0)\b"
)
FIRST_OPERAND_RE = re.compile(r"^\s*[a-z][a-z0-9]*(?:\.[a-z0-9]+)*\s+([^,\s]+)")
LANE_RE = re.compile(r"\bvextbcst\.16\s+\S+,\s*x11,\s*#0x([0-9a-fA-F]+)\b")


@dataclass(frozen=True)
class Event:
    index: int
    address: int
    group: int
    op: str
    fragment: str
    defs: tuple[str, ...]
    uses: tuple[str, ...]
    def_families: tuple[str, ...]
    use_families: tuple[str, ...]
    write_aliases: tuple[str, ...]
    semantic: str
    lane: int | None


@dataclass(frozen=True)
class BoundaryLive:
    boundary: str
    family: str
    producer: Event
    consumer: Event


def load_exp004() -> ModuleType:
    spec = importlib.util.spec_from_file_location("mylm_exp004_for_lifetime", EXP004_RUN)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load exp004 helper: {EXP004_RUN}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


EXP004 = load_exp004()


def op_name(fragment: str) -> str:
    return EXP004.op_name(fragment)


def direct_registers(fragment: str) -> tuple[str, ...]:
    return tuple(REGISTER_RE.findall(fragment))


def first_operand(fragment: str) -> str | None:
    match = FIRST_OPERAND_RE.match(fragment)
    return match.group(1) if match is not None else None


def is_write_op(op: str) -> bool:
    if not op or op == "nop":
        return False
    if op.startswith(("st", "vst", "j", "ret", "rel", "acq")):
        return False
    return True


def def_use(fragment: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    op = op_name(fragment)
    registers = direct_registers(fragment)
    first = first_operand(fragment)
    if is_write_op(op) and first in registers:
        defs = (first,)
        skipped = False
        uses: list[str] = []
        for register in registers:
            if not skipped and register == first:
                skipped = True
            else:
                uses.append(register)
        return defs, tuple(uses)
    return (), registers


def vector_family(register: str) -> str | None:
    match = re.fullmatch(r"(?:wl|wh|x)(\d+)", register)
    if match is not None:
        return f"vec{match.group(1)}"
    match = re.fullmatch(r"y(\d+)", register)
    if match is not None:
        return f"vec_y{match.group(1)}"
    return None


def accum_family(register: str) -> str | None:
    match = re.fullmatch(r"(?:bmll|bmlh|bmhl|bmhh|cml|cmh|dm)(\d+)", register)
    if match is not None:
        return f"acc{match.group(1)}"
    return None


def family(register: str) -> str:
    return vector_family(register) or accum_family(register) or register


def dedup(values: list[str]) -> tuple[str, ...]:
    out: list[str] = []
    for value in values:
        if value not in out:
            out.append(value)
    return tuple(out)


def alias_write_set(register: str) -> tuple[str, ...]:
    vector = vector_family(register)
    if vector is not None:
        index = int(vector.removeprefix("vec").removeprefix("_y")) if vector.startswith("vec") and not vector.startswith("vec_y") else None
        if index is None:
            return (register, vector)
        return dedup([register, f"wl{index}", f"wh{index}", f"x{index}", f"y{index // 2}", vector])

    match = re.fullmatch(r"(bmll|bmlh|bmhl|bmhh|cml|cmh|dm)(\d+)", register)
    if match is None:
        return (register,)
    prefix = match.group(1)
    index = int(match.group(2))
    all_children = [f"bmll{index}", f"bmlh{index}", f"bmhl{index}", f"bmhh{index}"]
    if prefix == "dm":
        return dedup([register, f"cml{index}", f"cmh{index}", *all_children, f"acc{index}"])
    if prefix == "cml":
        return dedup([register, f"bmll{index}", f"bmlh{index}", f"dm{index}", f"acc{index}"])
    if prefix == "cmh":
        return dedup([register, f"bmhl{index}", f"bmhh{index}", f"dm{index}", f"acc{index}"])
    parent = f"cml{index}" if prefix in {"bmll", "bmlh"} else f"cmh{index}"
    return dedup([register, parent, f"dm{index}", f"acc{index}"])


def semantic_class(op: str, fragment: str, defs: tuple[str, ...], uses: tuple[str, ...]) -> str:
    if op == "vldb" and defs == ("x11",):
        return "activation_load"
    if op in {"vlda", "vldb"} and "p0" in uses:
        return "weight_or_state_load"
    if op == "vlda" and defs and defs[0].startswith("lfh"):
        return "local_scratch_load"
    if op == "lda.s16" and "p3" in uses:
        return "group_sum_load"
    if op == "vunpack":
        return "q4_unpack"
    if op == "vextbcst.16" and "x11" in uses:
        return "activation_lane_broadcast"
    if op == "vups.4x":
        return "ups_vector_to_accum"
    if op == "vadd":
        return "dequant_add"
    if op == "vsub.f":
        return "dequant_sub"
    if op == "vconv.bf16.fp32":
        return "accum_to_bf16_vector"
    if op == "vmac.f":
        return "mac"
    if op == "vmul.f":
        return "mul_seed"
    if op.startswith("vmov"):
        return "register_move"
    if op.startswith("vst"):
        return "store"
    if op.startswith("add") or op.startswith("lshl") or op.startswith("mov"):
        return "scalar_pointer_setup"
    return op or "unknown"


def group_for_address(address: int, groups: tuple) -> int:
    for group in groups:
        if group.start <= address < group.end:
            return group.index
    return -1


def build_events() -> tuple[Event, ...]:
    instructions = EXP004.disassemble()
    hot = EXP004.range_instructions(instructions, EXP004.HOT_START, EXP004.HOT_END)
    groups = EXP004.lane_groups(hot)
    events: list[Event] = []
    for instruction in hot:
        group = group_for_address(instruction.address, groups)
        for fragment in instruction.fragments:
            op = op_name(fragment)
            defs, uses = def_use(fragment)
            lane_match = LANE_RE.search(fragment)
            lane = int(lane_match.group(1), 16) if lane_match is not None else None
            write_aliases: list[str] = []
            for register in defs:
                write_aliases.extend(alias_write_set(register))
            events.append(
                Event(
                    index=len(events),
                    address=instruction.address,
                    group=group,
                    op=op,
                    fragment=fragment,
                    defs=defs,
                    uses=uses,
                    def_families=dedup([family(register) for register in defs]),
                    use_families=dedup([family(register) for register in uses]),
                    write_aliases=dedup(write_aliases),
                    semantic=semantic_class(op, fragment, defs, uses),
                    lane=lane,
                )
            )
    return tuple(events)


def is_vector_or_accum_family(value: str) -> bool:
    return value.startswith("vec") or value.startswith("acc")


def group_family_summary(events: tuple[Event, ...]) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for group in range(8):
        group_events = [event for event in events if event.group == group]
        def_first: dict[str, int] = {}
        use_first: dict[str, int] = {}
        def_last: dict[str, int] = {}
        use_last: dict[str, int] = {}
        for event in group_events:
            for item in event.def_families:
                def_first.setdefault(item, event.index)
                def_last[item] = event.index
            for item in event.use_families:
                use_first.setdefault(item, event.index)
                use_last[item] = event.index
        families = sorted(set(def_first) | set(use_first))
        live_in = [
            item
            for item in families
            if is_vector_or_accum_family(item)
            and item in use_first
            and (item not in def_first or use_first[item] <= def_first[item])
        ]
        written = [item for item in families if is_vector_or_accum_family(item) and item in def_first]
        summaries.append(
            {
                "group": group,
                "events": len(group_events),
                "live_in": live_in,
                "written": written,
                "last_use_only": [
                    item
                    for item in families
                    if is_vector_or_accum_family(item) and item in use_last and item not in def_last
                ],
            }
        )
    return summaries


def boundary_lives(events: tuple[Event, ...]) -> tuple[BoundaryLive, ...]:
    out: list[BoundaryLive] = []
    for boundary_group in range(7):
        before = [event for event in events if event.group <= boundary_group]
        after = [event for event in events if event.group > boundary_group]
        families = sorted(
            {
                item
                for event in events
                for item in (*event.def_families, *event.use_families)
                if is_vector_or_accum_family(item)
            }
        )
        for item in families:
            producer = next(
                (
                    event
                    for event in reversed(before)
                    if item in event.def_families or item in event.write_aliases
                ),
                None,
            )
            if producer is None:
                continue
            next_access = next(
                (
                    event
                    for event in after
                    if item in event.def_families
                    or item in event.write_aliases
                    or item in event.use_families
                ),
                None,
            )
            if next_access is not None and item in next_access.use_families:
                out.append(BoundaryLive(f"g{boundary_group}->g{boundary_group + 1}", item, producer, next_access))
    return tuple(out)


def boundary_signatures(boundaries: tuple[BoundaryLive, ...]) -> list[dict[str, object]]:
    by_boundary: dict[str, list[str]] = {}
    for item in boundaries:
        by_boundary.setdefault(item.boundary, []).append(item.family)
    return [
        {
            "boundary": boundary,
            "families": sorted(families),
        }
        for boundary, families in sorted(by_boundary.items())
    ]


def event_row(event: Event) -> dict[str, object]:
    return {
        "index": event.index,
        "address": hex(event.address),
        "group": event.group,
        "op": event.op,
        "semantic": event.semantic,
        "lane": event.lane,
        "defs": list(event.defs),
        "uses": list(event.uses),
        "def_families": list(event.def_families),
        "use_families": list(event.use_families),
        "write_aliases": list(event.write_aliases),
        "fragment": event.fragment,
    }


def boundary_row(item: BoundaryLive) -> dict[str, object]:
    return {
        "boundary": item.boundary,
        "family": item.family,
        "producer": event_row(item.producer),
        "consumer": event_row(item.consumer),
    }


def build_manifest() -> dict[str, object]:
    events = build_events()
    boundaries = boundary_lives(events)
    semantic_counts = Counter(event.semantic for event in events)
    key_events = [
        event
        for event in events
        if event.semantic
        in {
            "activation_load",
            "activation_lane_broadcast",
            "q4_unpack",
            "ups_vector_to_accum",
            "dequant_add",
            "dequant_sub",
            "accum_to_bf16_vector",
            "mac",
            "mul_seed",
            "group_sum_load",
        }
    ]
    return {
        "status": "passed",
        "source": str(EXP004.DEFAULT_ELF.relative_to(REPO_ROOT)),
        "hot_range": [hex(EXP004.HOT_START), hex(EXP004.HOT_END)],
        "event_count": len(events),
        "semantic_counts": dict(sorted(semantic_counts.items())),
        "group_family_summary": group_family_summary(events),
        "boundary_live_count": len(boundaries),
        "boundary_signatures": boundary_signatures(boundaries),
        "boundary_lives": [boundary_row(item) for item in boundaries],
        "first_160_key_events": [event_row(event) for event in key_events[:160]],
    }


def render_event_ref(event: dict[str, object]) -> str:
    return (
        f"`{event['address']}` `{event['op']}` `{event['semantic']}` "
        f"defs={event['defs']} uses={event['uses']}"
    )


def render_report(manifest: dict[str, object]) -> str:
    lines = [
        "# Q4NX Alias Lifetime Graph",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Source: `{manifest['source']}`",
        f"- Hot range: `{manifest['hot_range'][0]}..{manifest['hot_range'][1]}`",
        f"- Event fragments: `{manifest['event_count']}`",
        f"- Boundary live-through edges: `{manifest['boundary_live_count']}`",
        "",
        "## Semantic Counts",
        "",
        "| semantic | count |",
        "| --- | ---: |",
    ]
    for name, count in manifest["semantic_counts"].items():
        lines.append(f"| `{name}` | `{count}` |")

    lines.extend(
        [
            "",
            "## Group Family Summary",
            "",
            "| group | events | live-in vector/acc families | written vector/acc families |",
            "| ---: | ---: | --- | --- |",
        ]
    )
    for summary in manifest["group_family_summary"]:
        lines.append(
            f"| `{summary['group']}` | `{summary['events']}` | "
            f"`{summary['live_in']}` | `{summary['written']}` |"
        )

    lines.extend(
        [
            "",
            "## Boundary Signature",
            "",
            "| boundary | live-through families |",
            "| --- | --- |",
        ]
    )
    for signature in manifest["boundary_signatures"]:
        lines.append(f"| `{signature['boundary']}` | `{signature['families']}` |")

    lines.extend(
        [
            "",
            "## Boundary Live-Through Families",
            "",
            "| boundary | family | producer | consumer |",
            "| --- | --- | --- | --- |",
        ]
    )
    for item in manifest["boundary_lives"][:80]:
        lines.append(
            f"| `{item['boundary']}` | `{item['family']}` | "
            f"{render_event_ref(item['producer'])} | {render_event_ref(item['consumer'])} |"
        )

    lines.extend(
        [
            "",
            "## Key Event Prefix",
            "",
            "| index | group | address | semantic | defs | uses | fragment |",
            "| ---: | ---: | --- | --- | --- | --- | --- |",
        ]
    )
    for event in manifest["first_160_key_events"][:80]:
        fragment = str(event["fragment"]).replace("|", "\\|")
        lines.append(
            f"| `{event['index']}` | `{event['group']}` | `{event['address']}` | "
            f"`{event['semantic']}` | `{event['defs']}` | `{event['uses']}` | `{fragment}` |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This experiment moves past register-name vocabulary. It treats vector and "
            "accumulator aliases as shared families and exposes which families are "
            "live across activation-group boundaries. Those live-through edges are "
            "the concrete shape of MyLM's cross-group software pipeline.",
            "",
            "The table is still conservative: `dm/cml/cmh/bm*` partial writes are "
            "reported at family granularity. That is enough to identify which "
            "families need manual lane/view-level decoding next.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    try:
        manifest = build_manifest()
    except Exception:
        failure = {"status": "experiment_failed", "traceback": traceback.format_exc()}
        MANIFEST.write_text(json.dumps(failure, indent=2, sort_keys=True) + "\n")
        REPORT.write_text(
            "# Q4NX Alias Lifetime Graph\n\nExperiment failed.\n\n```text\n"
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
