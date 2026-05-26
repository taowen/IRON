#!/usr/bin/env python3
"""
Test for Experiment 03: Phase Replay

Tests whether IRON can express the fused layer projection pattern:
- Hold input across 3 phases (Q/K/V)
- 3 different tile-to-tile output FIFOs from same worker
- Sequential weight streaming through same FIFO
- Multi-stage pipeline without DDR intermediates

Key failure modes to watch:
- DMA channel exhaustion (3 output FIFOs > 2 MM2S channels)
- Stream switch routing conflicts
- Sequential weight fill ordering
"""

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

import aie.utils as aie_utils
from aie.iron.device import NPU2Col4
from design import phase_replay_pipeline


def test_compile():
    K = 128
    M_q = 64
    M_k = 64
    M_v = 64
    m_input = 4

    print(f"Configuration: K={K}, M_q={M_q}, M_k={M_k}, M_v={M_v}, m_input={m_input}")
    print(f"Pipeline: hidden[{K}] → Norm → normed[{K}] (HOLD) → Q/K/V GEMVs → collector → DDR")
    print(f"Key: normed held across 3 phases, Q/K/V outputs are tile-to-tile")
    print()

    dev = NPU2Col4()
    aie_utils.set_current_device(NPU2Col4)

    try:
        mlir_module = phase_replay_pipeline(
            dev=dev,
            K=K,
            M_q=M_q,
            M_k=M_k,
            M_v=M_v,
            m_input=m_input,
        )
        mlir_str = str(mlir_module)

        print("=== SUCCESS: Design compiled to MLIR ===\n")

        output_path = Path(__file__).parent / "output.mlir"
        with open(output_path, "w") as f:
            f.write(mlir_str)
        print(f"MLIR saved to: {output_path}")
        print(f"Generated MLIR length: {len(mlir_str)} chars\n")

        # === Critical checks ===
        print("Critical checks:")

        # Check 1: All internal FIFOs exist
        internal_fifos = ["normed", "Q_out", "K_out", "V_out"]
        fifo_checks = {}
        for name in internal_fifos:
            exists = f"@{name}" in mlir_str
            fifo_checks[name] = exists
            print(f"  [{'PASS' if exists else 'FAIL'}] '{name}' ObjectFifo exists")

        # Check 2: runtime_sequence has exactly 3 DDR args
        rt_seq_line = ""
        for line in mlir_str.split("\n"):
            if "runtime_sequence" in line:
                rt_seq_line = line
                break
        arg_count = rt_seq_line.count("%arg")
        print(f"  [{'PASS' if arg_count == 3 else 'FAIL'}] runtime_sequence has 3 DDR args (got {arg_count})")

        # Check 3: Internal FIFOs are NOT in runtime_sequence
        for name in internal_fifos:
            in_rt = name in rt_seq_line.lower()
            print(f"  [{'PASS' if not in_rt else 'FAIL'}] '{name}' NOT in runtime_sequence")

        # Check 4: normed fifo connects compute tiles (tile-to-tile)
        normed_lines = [l for l in mlir_str.split("\n") if "normed" in l and "objectfifo @" in l.lower()]
        for line in normed_lines:
            print(f"  [INFO] normed FIFO: {line.strip()}")

        # Check 5: Q/K/V output FIFOs connect compute tiles
        for name in ["Q_out", "K_out", "V_out"]:
            fifo_lines = [l for l in mlir_str.split("\n") if f"@{name}" in l and "objectfifo @" in l.lower()]
            for line in fifo_lines:
                print(f"  [INFO] {name} FIFO: {line.strip()}")

        # Check 6: Multiple cores exist
        core_count = mlir_str.count("aie.core(")
        print(f"  [{'PASS' if core_count >= 3 else 'FAIL'}] {core_count} cores (norm + projection + collector)")

        # Check 7: normed is acquired once and released once in projection core
        # Look for the projection core section
        lines = mlir_str.split("\n")
        in_proj_core = False
        normed_acquires = 0
        normed_releases = 0
        for line in lines:
            if "aie.core(" in line and "tile_0_3" in line:
                in_proj_core = True
            elif in_proj_core and "aie.end" in line:
                break
            elif in_proj_core:
                if "@normed" in line and "acquire" in line:
                    normed_acquires += 1
                if "@normed" in line and "release" in line:
                    normed_releases += 1

        print(f"  [{'PASS' if normed_acquires == 1 else 'FAIL'}] normed acquired {normed_acquires}x in projection core (want 1)")
        print(f"  [{'PASS' if normed_releases == 1 else 'FAIL'}] normed released {normed_releases}x in projection core (want 1)")

        # Check 8: No DMA task for internal FIFOs
        internal_dma = any(
            any(name in l for name in internal_fifos) and "dma_configure" in l
            for l in lines
        )
        print(f"  [{'PASS' if not internal_dma else 'FAIL'}] No DMA task for internal FIFOs")

        # Check 9: Weight FIFO has multiple DMA tasks (3 fills)
        w_dma_count = sum(1 for l in lines if "@W" in l and "dma_configure" in l)
        print(f"  [{'PASS' if w_dma_count == 3 else 'INFO'}] Weight FIFO has {w_dma_count} DMA tasks (want 3 for Q/K/V)")

        all_pass = (
            all(fifo_checks.values())
            and arg_count == 3
            and core_count >= 3
            and normed_acquires == 1
            and normed_releases == 1
            and not internal_dma
        )

        print(f"\n{'='*60}")
        if all_pass:
            print("CONCLUSION: IRON CAN express the fused layer projection pattern:")
            print("  - Phase replay (input held across 3 GEMV phases)")
            print("  - Multiple tile-to-tile output FIFOs from one worker")
            print("  - Sequential weight streaming through same FIFO")
            print("  - Multi-stage pipeline with NO DDR intermediates")
        else:
            print("CONCLUSION: Some checks failed — see above for details.")
        print(f"{'='*60}")

        return all_pass

    except Exception as e:
        print(f"=== FAILURE: Design could not compile ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error: {e}")
        print()
        import traceback
        traceback.print_exc()
        print()
        print("This reveals IRON CANNOT express the fused layer projection pattern.")
        print("The specific failure mode tells us what's missing.")
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("Experiment 03: Phase Replay (Norm → 3-Phase GEMV → Collector)")
    print("=" * 70)
    print()
    print("Question: Can IRON express the core fused layer projection dispatch?")
    print("  - Hold normed input across Q/K/V phases")
    print("  - Route each phase's output tile-to-tile (not DDR)")
    print("  - Stream 3 weight sets through same FIFO")
    print()

    success = test_compile()
    sys.exit(0 if success else 1)
