"""Case wrapper for Shape-A/B attention-to-O integration."""

from __future__ import annotations

import shape_runner
from shape_reference import CASE_NAME


def check_only() -> bool:
    return shape_runner.check_only()


def build_only() -> bool:
    return shape_runner.build_only()


def run() -> bool:
    return shape_runner.run()
