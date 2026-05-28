"""Case wrapper for shared c1r1 activation bridge smoke cases."""

from __future__ import annotations

import bridge_runner
from bridge_reference import BRIDGE_CASES

CASE_NAMES = tuple(case.name for case in BRIDGE_CASES)


def check_only(case_name: str) -> bool:
    return bridge_runner.check_only(case_name)


def build_only(case_name: str) -> bool:
    return bridge_runner.build_only(case_name)


def run(case_name: str) -> bool:
    return bridge_runner.run(case_name)
