#!/usr/bin/env python3
"""Emit the qwen3-dataflow MLIR-AIE physical skeleton."""

from __future__ import annotations

import traceback
from pathlib import Path

from contract import summary_lines, validate_contract
from dataflow import validate_dataflow
from generate import generate_mlir, validate_generated_mlir

OUTPUT = Path(__file__).parent / "build" / "qwen3_dataflow.mlir"


def main() -> int:
    mlir = generate_mlir()
    errors = validate_contract()
    errors.extend(validate_dataflow())
    errors.extend(validate_generated_mlir(mlir))
    if errors:
        raise RuntimeError("\n".join(errors))

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(mlir)
    print("\n".join(summary_lines()))
    print(f"wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        raise SystemExit(1)
