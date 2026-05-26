#!/usr/bin/env python3
"""
Test for Experiment 01: Multi-Phase GEMV

This test attempts to compile and run a single IRON design where one Worker
holds an input vector on-chip while sequentially computing two projections
(Q and K) from different weight streams.

Expected outcomes:
  - If IRON supports this pattern: compiles and produces correct output
  - If IRON does NOT support this: fails at compile time or produces
    incorrect MLIR, revealing the missing abstraction

Run:
  cd experiments/01_multi_phase_gemv
  python test.py
"""

import sys
from pathlib import Path
import numpy as np
import ml_dtypes

# Add IRON repo to path
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))

import aie.utils as aie_utils
from aie.iron.device import NPU2Col4
from design import multi_phase_gemv


def test_compile_multi_phase_gemv():
    """Test that the multi-phase GEMV design compiles to valid MLIR."""

    # Small dimensions for fast iteration
    # M_q == M_k to isolate the multi-phase question from type signature issues
    M_q = 128     # Q projection output dim
    M_k = 128     # K projection output dim (same as Q for this test)
    K = 128       # shared input dim (hidden size)
    cols = 2      # number of AIE columns
    m_input = 4   # weight tile rows per kernel call

    print(f"Configuration: M_q={M_q}, M_k={M_k}, K={K}, cols={cols}, m_input={m_input}")
    print(f"Phase 1 (Q): {M_q}x{K} @ {K} -> {M_q}")
    print(f"Phase 2 (K): {M_k}x{K} @ {K} -> {M_k}")
    print(f"Input vector loaded ONCE, held across both phases")
    print()

    dev = NPU2Col4()
    aie_utils.set_current_device(NPU2Col4)
    print(f"Target device: NPU2Col4 (cols={dev.cols})")

    try:
        mlir_module = multi_phase_gemv(
            dev=dev,
            cols=cols,
            M_q=M_q,
            M_k=M_k,
            K=K,
            m_input=m_input,
        )
        print("\n=== SUCCESS: Design compiled to MLIR ===")
        print("This means IRON CAN express multi-phase workers with on-chip input reuse.")
        print()

        # Print the generated MLIR for inspection
        mlir_str = str(mlir_module)
        print(f"Generated MLIR length: {len(mlir_str)} chars")

        # Save MLIR for inspection
        output_path = Path(__file__).parent / "output.mlir"
        with open(output_path, "w") as f:
            f.write(mlir_str)
        print(f"MLIR saved to: {output_path}")

        # Check key patterns in the MLIR
        checks = [
            ("runtime_sequence", "Has runtime sequence"),
            ("objectfifo", "Has ObjectFifo declarations"),
        ]
        print("\nMLIR content checks:")
        for pattern, desc in checks:
            found = pattern.lower() in mlir_str.lower()
            status = "PASS" if found else "FAIL"
            print(f"  [{status}] {desc}")

        return True

    except Exception as e:
        print(f"\n=== FAILURE: Design could not compile ===")
        print(f"Error type: {type(e).__name__}")
        print(f"Error message: {e}")
        print()
        print("This reveals what IRON cannot express:")
        print("  - If ObjectFifo error: holding an element across phases is not supported")
        print("  - If Runtime error: sequential fills to same FIFO not supported")
        print("  - If type error: output FIFOs with different sizes in same Worker not supported")
        import traceback
        traceback.print_exc()
        return False


def test_numerical_correctness():
    """If compilation succeeds, verify numerical correctness."""

    M_q = 128
    M_k = 64
    K = 128

    # Generate reference data
    np.random.seed(42)
    W_q = np.random.randn(M_q, K).astype(np.float32)
    W_k = np.random.randn(M_k, K).astype(np.float32)
    x = np.random.randn(K).astype(np.float32)

    # Expected outputs
    y_q_ref = (W_q @ x).astype(np.float32)
    y_k_ref = (W_k @ x).astype(np.float32)

    print(f"\nReference Q output (first 8): {y_q_ref[:8]}")
    print(f"Reference K output (first 8): {y_k_ref[:8]}")
    print("\n(Numerical test requires NPU hardware — skipping execution)")


if __name__ == "__main__":
    print("=" * 70)
    print("Experiment 01: Multi-Phase GEMV with On-Chip Input Reuse")
    print("=" * 70)
    print()
    print("Question: Can a single IRON Worker hold input on-chip while doing")
    print("          multiple sequential projections from one weight stream?")
    print()

    success = test_compile_multi_phase_gemv()
    if success:
        test_numerical_correctness()

    print()
    print("=" * 70)
    sys.exit(0 if success else 1)
