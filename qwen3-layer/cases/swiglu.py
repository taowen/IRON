"""Case wrapper for main16 -> row1/c1r1 -> c6r2 integration."""

from __future__ import annotations

import swiglu_runner
from swiglu_reference import CASE_NAME


def check_only() -> bool:
    return swiglu_runner.check_only()


def build_only() -> bool:
    return swiglu_runner.build_only()


def run() -> bool:
    return swiglu_runner.run()
