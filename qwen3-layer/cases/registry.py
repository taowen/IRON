"""Dispatch table for runnable qwen3-layer NPU cases."""

from __future__ import annotations

import npu_build
from cases import (
    attention_kv16_runner,
    bridge,
    c1r2,
    current,
    full_layer_contract_runner,
    kvscan_attention_kv16_runner,
    mainq_kvscan_attention_kv16_runner,
    qkv_shape_o_c1r2_runner,
    shape,
    swiglu,
)

CASE_NAMES = (
    current.CASE_NAME,
    c1r2.CASE_NAME,
    attention_kv16_runner.CASE_NAME,
    kvscan_attention_kv16_runner.CASE_NAME,
    mainq_kvscan_attention_kv16_runner.CASE_NAME,
    qkv_shape_o_c1r2_runner.CASE_NAME,
    full_layer_contract_runner.CASE_NAME,
    shape.CASE_NAME,
    swiglu.CASE_NAME,
) + bridge.CASE_NAMES


def check_only(case_name: str) -> bool:
    if case_name == current.CASE_NAME:
        return current.check_only()
    if case_name == c1r2.CASE_NAME:
        return c1r2.check_only()
    if case_name == attention_kv16_runner.CASE_NAME:
        return attention_kv16_runner.check_only()
    if case_name == kvscan_attention_kv16_runner.CASE_NAME:
        return kvscan_attention_kv16_runner.check_only()
    if case_name == mainq_kvscan_attention_kv16_runner.CASE_NAME:
        return mainq_kvscan_attention_kv16_runner.check_only()
    if case_name == qkv_shape_o_c1r2_runner.CASE_NAME:
        return qkv_shape_o_c1r2_runner.check_only()
    if case_name == full_layer_contract_runner.CASE_NAME:
        return full_layer_contract_runner.check_only()
    if case_name == shape.CASE_NAME:
        return shape.check_only()
    if case_name == swiglu.CASE_NAME:
        return swiglu.check_only()
    if case_name in bridge.CASE_NAMES:
        return bridge.check_only(case_name)
    raise ValueError(f"unknown case {case_name}")


def build_only(case_name: str) -> bool:
    if case_name == current.CASE_NAME:
        return current.build_only()
    if case_name == c1r2.CASE_NAME:
        return c1r2.build_only()
    if case_name == attention_kv16_runner.CASE_NAME:
        return attention_kv16_runner.build_only()
    if case_name == kvscan_attention_kv16_runner.CASE_NAME:
        return kvscan_attention_kv16_runner.build_only()
    if case_name == mainq_kvscan_attention_kv16_runner.CASE_NAME:
        return mainq_kvscan_attention_kv16_runner.build_only()
    if case_name == qkv_shape_o_c1r2_runner.CASE_NAME:
        return qkv_shape_o_c1r2_runner.build_only()
    if case_name == full_layer_contract_runner.CASE_NAME:
        return full_layer_contract_runner.build_only()
    if case_name == shape.CASE_NAME:
        return shape.build_only()
    if case_name == swiglu.CASE_NAME:
        return swiglu.build_only()
    if case_name in bridge.CASE_NAMES:
        return bridge.build_only(case_name)
    raise ValueError(f"unknown case {case_name}")


def run(case_name: str) -> bool:
    print(f"NPU device: {npu_build.device()}")
    if case_name == current.CASE_NAME:
        return current.run()
    if case_name == c1r2.CASE_NAME:
        return c1r2.run()
    if case_name == attention_kv16_runner.CASE_NAME:
        return attention_kv16_runner.run()
    if case_name == kvscan_attention_kv16_runner.CASE_NAME:
        return kvscan_attention_kv16_runner.run()
    if case_name == mainq_kvscan_attention_kv16_runner.CASE_NAME:
        return mainq_kvscan_attention_kv16_runner.run()
    if case_name == qkv_shape_o_c1r2_runner.CASE_NAME:
        return qkv_shape_o_c1r2_runner.run()
    if case_name == full_layer_contract_runner.CASE_NAME:
        return full_layer_contract_runner.run()
    if case_name == shape.CASE_NAME:
        return shape.run()
    if case_name == swiglu.CASE_NAME:
        return swiglu.run()
    if case_name in bridge.CASE_NAMES:
        return bridge.run(case_name)
    raise ValueError(f"unknown case {case_name}")
