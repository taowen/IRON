#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import aie.utils as aie_utils
import torch

from iron.common import AIEContext
from iron.common.test_utils import run_test
from iron.operators.mha.op import MHA
from iron.operators.mha.reference import generate_golden_reference as generate_mha_ref
from iron.operators.sage_attention.op import SageAttention
from iron.operators.sage_attention.reference import generate_golden_reference


def _attention_error_metrics(output, reference):
    output = output[: reference.shape[0], : reference.shape[1]].to(torch.float32)
    reference = reference.to(torch.float32)
    diff = (output - reference).flatten()
    ref_flat = reference.flatten()
    cosine = torch.nn.functional.cosine_similarity(output.flatten(), ref_flat, dim=0)
    rel_l1 = diff.abs().sum() / ref_flat.abs().sum()
    rmse = torch.sqrt(torch.mean(diff * diff))
    return {
        "cosine": float(cosine.item()),
        "rel_l1": float(rel_l1.item()),
        "rmse": float(rmse.item()),
        "max_abs": float(diff.abs().max().item()),
    }


@pytest.mark.supported_devices("npu2")
@pytest.mark.metrics(
    SageLatency=r"SageAttention Latency \(us\): (?P<value>[\d\.]+)",
    MHALatency=r"MHA Latency \(us\): (?P<value>[\d\.]+)",
)
@pytest.mark.parametrize("seq_len,dim", [pytest.param(256, 64)])
def test_sage_attention_faster_than_mha(seq_len, dim, aie_context):
    aie_utils.DefaultNPURuntime.cleanup()

    sage_ref = generate_golden_reference(S_q=seq_len, S_kv=seq_len, d=dim)
    num_pipelines = 2 if seq_len >= 128 else 1
    seq_padding = ((seq_len + 64 * num_pipelines - 1) // (64 * num_pipelines)) * (
        64 * num_pipelines
    )
    num_blocks = seq_padding // 64
    smooth_metrics = _attention_error_metrics(
        sage_ref["O_smooth"][:seq_len], sage_ref["O_full"][:seq_len]
    )
    quant_metrics = _attention_error_metrics(
        sage_ref["O"][:seq_len], sage_ref["O_full"][:seq_len]
    )

    print(
        "\nSageAttention Quantization Accuracy: "
        f"cos={quant_metrics['cosine']:.8f}, "
        f"rel_l1={quant_metrics['rel_l1']:.6f}, "
        f"rmse={quant_metrics['rmse']:.6f}, "
        f"max_abs={quant_metrics['max_abs']:.6f}"
    )
    print(
        "K-smoothing Invariance: "
        f"cos={smooth_metrics['cosine']:.8f}, "
        f"rel_l1={smooth_metrics['rel_l1']:.6f}, "
        f"rmse={smooth_metrics['rmse']:.6f}, "
        f"max_abs={smooth_metrics['max_abs']:.6f}"
    )

    assert smooth_metrics["cosine"] > 0.9999
    assert smooth_metrics["rel_l1"] < 0.002
    assert quant_metrics["cosine"] > 0.999
    assert quant_metrics["rel_l1"] < 0.01
    assert quant_metrics["rmse"] < 0.03

    sage_op = SageAttention(
        seq_len=seq_len,
        d=dim,
        context=aie_context,
    )

    sage_errors, sage_latency_us, sage_bandwidth_gbps = run_test(
        sage_op,
        {
            "Q_i8": sage_ref["Q_i8"].flatten(),
            "K_i8": sage_ref["K_i8"].flatten(),
            "V": sage_ref["V"].flatten(),
            "dequant_scales": sage_ref["dequant_scales"].reshape(
                num_blocks, num_blocks
            ),
        },
        {"O": sage_ref["O"].flatten()},
        rel_tol=0.08,
        abs_tol=0.20,
        max_error_rate=0.005,
        warmup_iters=5,
        timed_iters=30,
    )

    aie_utils.DefaultNPURuntime.cleanup()

    # Use a separate context so the baseline build artifacts and xclbin load are
    # isolated from the SageAttention operator under test.
    mha_context = AIEContext(build_dir=aie_context.build_dir / "mha_baseline")
    mha_ref = generate_mha_ref(
        S_q=seq_len,
        S_kv=seq_len,
        d=dim,
        heads=1,
        num_kv_heads=0,
        num_pipeline=1,
    )
    mha_op = MHA(
        num_heads=1,
        seq_len=seq_len,
        d=dim,
        num_KV_heads=0,
        num_of_pipelines=1,
        context=mha_context,
    )
    mha_errors, mha_latency_us, mha_bandwidth_gbps = run_test(
        mha_op,
        {
            "Q": mha_ref["Q"].flatten(),
            "K": mha_ref["K"].flatten(),
            "V": mha_ref["V"].flatten(),
        },
        {"O": mha_ref["O"].flatten()},
        rel_tol=4.0e-2,
        abs_tol=1.5e-1,
        max_error_rate=0.005,
        warmup_iters=5,
        timed_iters=30,
    )

    print(f"\nSageAttention Latency (us): {sage_latency_us:.1f}")
    print(f"SageAttention Effective Bandwidth: {sage_bandwidth_gbps:.6e} GB/s")
    print(f"MHA Latency (us): {mha_latency_us:.1f}")
    print(f"MHA Effective Bandwidth: {mha_bandwidth_gbps:.6e} GB/s")
    print(f"Speedup: {mha_latency_us / sage_latency_us:.3f}x\n")

    aie_utils.DefaultNPURuntime.cleanup()

    assert not sage_errors, f"SageAttention failed with errors: {sage_errors}"
    assert not mha_errors, f"MHA baseline failed with errors: {mha_errors}"
    assert sage_latency_us < mha_latency_us, (
        f"Expected SageAttention to be faster than MHA, got "
        f"{sage_latency_us:.1f} us vs {mha_latency_us:.1f} us"
    )
