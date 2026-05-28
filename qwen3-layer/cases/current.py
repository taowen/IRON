"""Case wrapper for the current full-schedule backend."""

from __future__ import annotations

import current_runner

CASE_NAME = "current"


def check_only() -> bool:
    return current_runner.check_backend_structure()


def build_only() -> bool:
    return current_runner.build_only()


def run() -> bool:
    return current_runner.run_on_npu()
