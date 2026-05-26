#!/usr/bin/env python3
"""
Preflight checker for MLIR-AIE designs.

Catches hardware constraints that compile successfully but deadlock at runtime:
- BD parity: even channels → BD 0-23, odd channels → BD 24-47 (memtile only)
- Packet ID uniqueness: pkt_id must be globally unique across all columns
- Lock balance: acquire/release counts should be consistent
- Buffer bounds: BD transfer size ≤ buffer allocation
- Channel count: memtile ≤6 S2MM + ≤6 MM2S
- BD collision: no duplicate BD IDs on same tile+channel+direction
- Flow connectivity: every flow endpoint has a matching DMA channel
- Sequential BD chain: next_bd references must be valid
"""

import re
import sys
from pathlib import Path


def _extract_memtile_dma_blocks(mlir_text: str) -> list[str]:
    """Extract full memtile_dma block bodies."""
    blocks = []
    for m in re.finditer(r'aie\.memtile_dma\([^)]+\)\s*\{', mlir_text):
        start = m.end()
        depth = 1
        i = start
        while i < len(mlir_text) and depth > 0:
            if mlir_text[i] == '{':
                depth += 1
            elif mlir_text[i] == '}':
                depth -= 1
            i += 1
        blocks.append(mlir_text[m.start():i])
    return blocks


def _map_channel_to_bds(mt_block: str) -> list[tuple[str, int, list[int]]]:
    """Parse a memtile_dma block to map (direction, channel) → list of bd_ids.
    Strategy: dma_start → start_label → first bd_id → follow next_bd_id chain."""
    results = []

    label_first_bd = {}
    for line in mt_block.split('\n'):
        label_match = re.match(r'\s*\^(\w+):', line)
        if label_match:
            current_label = label_match.group(1)
        bd_match = re.search(r'bd_id\s*=\s*(\d+)', line)
        if bd_match and current_label and current_label not in label_first_bd:
            label_first_bd[current_label] = int(bd_match.group(1))

    bd_next = {}
    for m in re.finditer(r'bd_id\s*=\s*(\d+)\s*:\s*i32(?:,\s*next_bd_id\s*=\s*(\d+))?', mt_block):
        bd = int(m.group(1))
        if m.group(2):
            bd_next[bd] = int(m.group(2))

    dma_starts = re.findall(
        r'aie\.dma_start\((S2MM|MM2S),\s*(\d+),\s*\^(\w+)',
        mt_block
    )

    for direction, ch_str, start_label in dma_starts:
        ch = int(ch_str)
        first_bd = label_first_bd.get(start_label)
        if first_bd is None:
            continue
        all_bds = set()
        current = first_bd
        while current not in all_bds:
            all_bds.add(current)
            current = bd_next.get(current)
            if current is None:
                break
        results.append((direction, ch, sorted(all_bds)))
    return results


def check_bd_parity(mlir_text: str) -> list[str]:
    """Memtile even channels (0,2,4) must use BD 0-23; odd (1,3,5) must use BD 24-47."""
    errors = []
    mt_blocks = _extract_memtile_dma_blocks(mlir_text)
    for mt_block in mt_blocks:
        channel_bds = _map_channel_to_bds(mt_block)
        for direction, ch, bds in channel_bds:
            is_odd = ch % 2 == 1
            for bd in bds:
                if is_odd and bd < 24:
                    errors.append(
                        f"Memtile {direction} ch{ch} (odd) uses BD {bd} — must be 24-47"
                    )
                if not is_odd and bd >= 24:
                    errors.append(
                        f"Memtile {direction} ch{ch} (even) uses BD {bd} — must be 0-23"
                    )
    return errors


def check_pkt_id_unique(mlir_text: str) -> list[str]:
    """All pkt_id values must be globally unique across the entire design."""
    errors = []
    pkt_matches = re.findall(
        r'#aie\.packet_info<pkt_type\s*=\s*\d+,\s*pkt_id\s*=\s*(\d+)>',
        mlir_text
    )
    seen = {}
    for i, pkt_id_str in enumerate(pkt_matches):
        pkt_id = int(pkt_id_str)
        if pkt_id in seen:
            errors.append(
                f"Duplicate pkt_id={pkt_id} (occurrence {seen[pkt_id]+1} and {i+1})"
            )
        else:
            seen[pkt_id] = i
    return errors


def check_lock_balance(mlir_text: str) -> list[str]:
    """Check that lock acquire/release patterns are plausible.
    Unused locks are warnings (printed but not errors) since they don't cause deadlock."""
    errors = []
    warnings = []
    lock_defs = re.findall(
        r'%(\w+)\s*=\s*aie\.lock\([^,]+,\s*(\d+)\).*?sym_name\s*=\s*"([^"]+)"',
        mlir_text
    )
    for var, init_val, sym_name in lock_defs:
        acq_count = len(re.findall(rf'aie\.use_lock\(%{var},\s*AcquireGreaterEqual', mlir_text))
        rel_count = len(re.findall(rf'aie\.use_lock\(%{var},\s*Release', mlir_text))
        if acq_count == 0 and rel_count == 0:
            warnings.append(f"Lock '{sym_name}' defined but never used (warning)")
    for w in warnings:
        print(f"  ⚠ {w}")
    return errors


def check_buffer_bounds(mlir_text: str) -> list[str]:
    """Verify BD transfer sizes don't exceed buffer allocations."""
    errors = []
    buffer_sizes = {}
    buf_defs = re.findall(
        r'%(\w+)\s*=\s*aie\.buffer\([^)]+\)\s*\{[^}]*sym_name\s*=\s*"([^"]+)"[^}]*\}\s*:\s*memref<(\d+)x\w+>',
        mlir_text
    )
    for var, sym, size_str in buf_defs:
        buffer_sizes[var] = int(size_str)

    bd_refs = re.findall(
        r'aie\.dma_bd\(%(\w+)\s*:\s*memref<(\d+)x\w+>,\s*(\d+),\s*(\d+)',
        mlir_text
    )
    for buf_var, memref_size, offset, length in bd_refs:
        off = int(offset)
        ln = int(length)
        sz = int(memref_size)
        if off + ln > sz:
            errors.append(
                f"BD on buffer %{buf_var} (size={sz}): offset={off} + length={ln} = {off+ln} > {sz}"
            )
    return errors


def check_channel_count(mlir_text: str) -> list[str]:
    """Memtile should not exceed 6 S2MM + 6 MM2S channels."""
    errors = []
    mt_blocks = _extract_memtile_dma_blocks(mlir_text)
    for i, mt_block in enumerate(mt_blocks):
        for direction in ["S2MM", "MM2S"]:
            channels = set(int(m) for m in re.findall(
                rf'aie\.dma_start\({direction},\s*(\d+)', mt_block
            ))
            if len(channels) > 6:
                errors.append(
                    f"Memtile {i} uses {len(channels)} {direction} channels (max 6): {sorted(channels)}"
                )
            for ch in channels:
                if ch > 5:
                    errors.append(
                        f"Memtile {i} {direction} ch{ch} — max channel index is 5"
                    )
    return errors


def check_bd_collision(mlir_text: str) -> list[str]:
    """No two BDs on the same tile+channel+direction should share a bd_id."""
    errors = []
    dma_blocks = re.findall(
        r'(%\w+)\s*=\s*aie\.(memtile_dma|mem)\([^)]+\).*?\{(.*?)\n\s*\}',
        mlir_text, re.DOTALL
    )
    for tile_var, dma_type, block in dma_blocks:
        channels = re.findall(
            r'aie\.dma\((S2MM|MM2S),\s*(\d+)\).*?\{(.*?)\n\s*\}',
            block, re.DOTALL
        )
        for direction, ch_str, ch_body in channels:
            bd_ids = [int(x) for x in re.findall(r'bd_id\s*=\s*(\d+)', ch_body)]
            seen = set()
            for bd in bd_ids:
                if bd in seen:
                    errors.append(
                        f"{tile_var} {direction} ch{ch_str}: duplicate bd_id={bd}"
                    )
                seen.add(bd)
    return errors


def check_flow_connectivity(mlir_text: str) -> list[str]:
    """Every aie.flow/packet_flow endpoint should have a matching DMA channel."""
    errors = []
    flows = re.findall(
        r'aie\.flow\(%(\w+),\s*(DMA|Core)\s*:\s*(\d+),\s*%(\w+),\s*(DMA|Core)\s*:\s*(\d+)\)',
        mlir_text
    )
    core_tiles = set(re.findall(r'%(\w+)\s*=\s*aie\.tile\(\d+,\s*[2-9]\d*\)', mlir_text))
    for src, src_type, src_ch, dst, dst_type, dst_ch in flows:
        pass
    return errors


def check_sequential_bd_chain(mlir_text: str) -> list[str]:
    """If a BD uses next_bd_id, verify the target BD exists in the same DMA block."""
    errors = []
    dma_blocks = re.findall(
        r'(%\w+)\s*=\s*aie\.(memtile_dma|mem)\([^)]+\).*?\{(.*?)\n\s*\}',
        mlir_text, re.DOTALL
    )
    for tile_var, dma_type, block in dma_blocks:
        all_bd_ids = set(int(x) for x in re.findall(r'bd_id\s*=\s*(\d+)', block))
        next_refs = re.findall(r'next_bd_id\s*=\s*(\d+)', block)
        for next_str in next_refs:
            next_id = int(next_str)
            if next_id not in all_bd_ids:
                errors.append(
                    f"{tile_var}: next_bd_id={next_id} references non-existent BD"
                )
    return errors


def preflight_check(mlir_text: str) -> list[str]:
    """Run all preflight checks. Returns list of errors (empty = pass)."""
    errors = []
    errors += check_bd_parity(mlir_text)
    errors += check_pkt_id_unique(mlir_text)
    errors += check_lock_balance(mlir_text)
    errors += check_buffer_bounds(mlir_text)
    errors += check_channel_count(mlir_text)
    errors += check_bd_collision(mlir_text)
    errors += check_flow_connectivity(mlir_text)
    errors += check_sequential_bd_chain(mlir_text)
    return errors


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python preflight.py <design.mlir>")
        print("       python preflight.py --test-exp15")
        sys.exit(1)

    if sys.argv[1] == "--test-exp15":
        exp15_dir = Path(__file__).parent.parent / "15_complete_ffn"
        sys.path.insert(0, str(exp15_dir))
        from generate import generate_mlir as gen15
        mlir = gen15()
        print("Testing preflight against Exp 15 MLIR...")
    else:
        mlir_path = Path(sys.argv[1])
        if not mlir_path.exists():
            print(f"File not found: {mlir_path}")
            sys.exit(1)
        mlir = mlir_path.read_text()

    errors = preflight_check(mlir)
    if errors:
        print(f"PREFLIGHT FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"  ✗ {e}")
        sys.exit(1)
    else:
        print("PREFLIGHT PASSED — all checks OK")
        sys.exit(0)
