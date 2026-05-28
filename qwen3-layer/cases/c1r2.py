"""Case wrapper for c1r2 O compact replay integration."""

from __future__ import annotations

import c1r2_runner
from c1r2_reference import CASE_NAME


def check_only() -> bool:
    return c1r2_runner.check_only()


def build_only() -> bool:
    return c1r2_runner.build_only()


def run() -> bool:
    return c1r2_runner.run()
