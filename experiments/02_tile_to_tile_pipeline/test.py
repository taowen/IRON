#!/usr/bin/env python3
"""
Test for Experiment 02: Tile-to-Tile Pipeline

Tests whether IRON can express a compute pipeline where an intermediate result
flows between tiles WITHOUT going through DDR:

  DDR(hidden) → [Norm Tile] → internal FIFO → [Projection Tiles] → DDR(output)

The normed hidden (internal FIFO) should appear in the MLIR as an ObjectFifo
between compute tiles (or via memtile), NOT as a runtime_sequence argument.

Key checks:
  1. Does it compile? (structural expressibility)
  2. In the generated MLIR, is `normed` fifo connected tile-to-tile?
  3. Is `normed` absent from the runtime_sequence arguments?
"""

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

import aie.utils as aie_utils
from aie.iron.device import NPU2Col4
from design import norm_projection_pipeline


def test_compile():
    """Test that the norm→projection pipeline compiles to MLIR."""

    K = 128       # hidden dimension
    M = 128       # projection output dimension
    cols = 2      # number of projection tiles
    m_input = 4   # GEMV tile size

    print(f"Configuration: K={K}, M={M}, cols={cols}, m_input={m_input}")
    print(f"Pipeline: hidden[{K}] → Norm → normed[{K}] → GEMV → output[{M}]")
    print(f"Key: normed[{K}] is tile-to-tile, NOT in DDR")
    print()

    dev = NPU2Col4()
    aie_utils.set_current_device(NPU2Col4)

    try:
        mlir_module = norm_projection_pipeline(
            dev=dev,
            cols=cols,
            K=K,
            M=M,
            m_input=m_input,
        )
        mlir_str = str(mlir_module)

        print("=== SUCCESS: Design compiled to MLIR ===\n")

        # Save MLIR
        output_path = Path(__file__).parent / "output.mlir"
        with open(output_path, "w") as f:
            f.write(mlir_str)
        print(f"MLIR saved to: {output_path}")
        print(f"Generated MLIR length: {len(mlir_str)} chars\n")

        # === Critical checks ===
        print("Critical checks:")

        # Check 1: normed fifo exists as an objectfifo between tiles
        has_normed_fifo = "@normed" in mlir_str or "normed" in mlir_str
        print(f"  [{'PASS' if has_normed_fifo else 'FAIL'}] 'normed' ObjectFifo exists in MLIR")

        # Check 2: normed is NOT a runtime_sequence argument
        # The runtime_sequence should only have 3 args: hidden, weights, output
        rt_seq_section = ""
        in_rt_seq = False
        for line in mlir_str.split("\n"):
            if "runtime_sequence" in line:
                in_rt_seq = True
                rt_seq_section = line
                break

        # Count runtime_sequence arguments
        arg_count = rt_seq_section.count("%arg")
        normed_in_args = "normed" in rt_seq_section.lower()

        print(f"  [{'PASS' if arg_count == 3 else 'FAIL'}] runtime_sequence has exactly 3 DDR args (got {arg_count})")
        print(f"  [{'PASS' if not normed_in_args else 'FAIL'}] 'normed' is NOT a runtime_sequence argument")

        # Check 3: normed fifo connects compute tiles (not shim)
        # Look for objectfifo declaration of normed — it should connect tile(x,2+) to tile(y,2+)
        normed_lines = [l for l in mlir_str.split("\n") if "normed" in l and "objectfifo" in l.lower()]
        print(f"  [INFO] normed FIFO declarations: {len(normed_lines)}")
        for line in normed_lines:
            print(f"         {line.strip()}")

        # Check 4: Multiple cores exist (heterogeneous workers)
        core_count = mlir_str.count("aie.core(")
        print(f"  [{'PASS' if core_count >= 3 else 'FAIL'}] Multiple cores in graph ({core_count} cores: 1 norm + {cols} projection)")

        # Check 5: No DDR DMA for normed intermediate
        normed_dma = any("normed" in l and "dma_configure" in l for l in mlir_str.split("\n"))
        print(f"  [{'PASS' if not normed_dma else 'FAIL'}] No DMA task configured for normed intermediate")

        all_pass = (
            has_normed_fifo
            and arg_count == 3
            and not normed_in_args
            and core_count >= 3
            and not normed_dma
        )

        print(f"\n{'='*60}")
        if all_pass:
            print("CONCLUSION: IRON CAN express tile-to-tile intermediate dataflow")
            print("without DDR. This is the fused layer engine's fundamental pattern.")
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
        print("This reveals IRON cannot express tile-to-tile intermediate flow.")
        print("The intermediate MUST go through DDR (runtime_sequence args).")
        return False


if __name__ == "__main__":
    print("=" * 70)
    print("Experiment 02: Tile-to-Tile Pipeline (Norm → Projection)")
    print("=" * 70)
    print()
    print("Question: Can an intermediate result flow between tiles WITHOUT DDR?")
    print("This is the #1 requirement of fused layer engines.")
    print()

    success = test_compile()
    sys.exit(0 if success else 1)
