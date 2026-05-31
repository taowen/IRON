#!/usr/bin/env python3
"""Test exact Q4NX as a MyLM-style centered-dequant plus group-sum form."""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
from ml_dtypes import bfloat16

REPO_ROOT = Path(__file__).resolve().parents[2]
QWEN3_LAYER = REPO_ROOT / "qwen3-layer"
sys.path.insert(0, str(QWEN3_LAYER))

from cases.qwen3_8b_decode_layer_reference import (  # noqa: E402
    DOWN_PROJECTION,
    GATE_PROJECTION,
    K_PROJECTION,
    LayerInputs,
    LayerReferenceResult,
    O_PROJECTION,
    Q_PROJECTION,
    Qwen3LayerReference,
    UP_PROJECTION,
    V_PROJECTION,
    _apply_rope,
    _attention,
    _head_rms_norm,
    _rms_norm,
    _silu,
    make_reference_inputs,
)
from contract import HEAD_DIM, HIDDEN_DIM, NUM_KV_HEADS  # noqa: E402
from q4nx_reference import (  # noqa: E402
    ACT_SLICE_BF16,
    CHUNK_BYTES,
    GROUPS_PER_CHUNK,
    GROUP_SIZE,
    M_PER_TILE,
    Q4NX_DATA_BYTES_PER_LANE,
    Q4NX_DATA_OFFSET,
    Q4NX_LANES,
    Q4NX_ROWS_PER_LANE,
    Q4NX_SCALE_BYTES,
)
from qwen3_model import DEFAULT_QWEN3_8B_MODEL_PATH, ProjectionTensor, Qwen3Q4NXModel  # noqa: E402

EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = EXPERIMENT_DIR / "qwen3_q4nx_centered_dequant_contract.md"
Variant = Literal[
    "scaled_bf16_sum",
    "scaled_fp32_sum",
    "centered_bf16_sum",
    "centered_fp32_sum",
]
VARIANTS: tuple[Variant, ...] = (
    "scaled_bf16_sum",
    "scaled_fp32_sum",
    "centered_bf16_sum",
    "centered_fp32_sum",
)


@dataclass(frozen=True)
class DiffStats:
    name: str
    variant: str
    shape: tuple[int, ...]
    max_abs: float
    mean_abs: float
    p99_abs: float
    mismatches_1e_3: int
    mismatches_1e_2: int
    mismatches_5e_2: int


@dataclass(frozen=True)
class CoefficientStats:
    name: str
    centered_delta_values: int
    centered_delta_nonzero: int
    centered_delta_max_abs: float
    centered_delta_mean_abs: float
    zero_sum_round_values: int
    zero_sum_round_nonzero: int
    zero_sum_round_max_abs: float
    zero_sum_round_mean_abs: float


@dataclass
class RunningAbsStats:
    count: int = 0
    nonzero: int = 0
    sum_abs: float = 0.0
    max_abs: float = 0.0

    def update(self, values: np.ndarray) -> None:
        abs_values = np.abs(values.astype(np.float32)).reshape(-1)
        if abs_values.size == 0:
            return
        self.count += int(abs_values.size)
        self.nonzero += int(np.count_nonzero(abs_values))
        self.sum_abs += float(np.sum(abs_values, dtype=np.float64))
        self.max_abs = max(self.max_abs, float(np.max(abs_values)))

    @property
    def mean_abs(self) -> float:
        return self.sum_abs / self.count if self.count else 0.0


def bf16_round(values: np.ndarray) -> np.ndarray:
    return values.astype(bfloat16).astype(np.float32)


def diff_stats(name: str, variant: str, got: np.ndarray, expected: np.ndarray) -> DiffStats:
    got_values = got.astype(bfloat16).astype(np.float32).reshape(-1)
    expected_values = expected.astype(bfloat16).astype(np.float32).reshape(-1)
    if got_values.shape != expected_values.shape:
        raise ValueError(f"{name}/{variant} shape mismatch: {got_values.shape} != {expected_values.shape}")
    diff = np.abs(got_values - expected_values)
    return DiffStats(
        name=name,
        variant=variant,
        shape=expected.shape,
        max_abs=float(np.max(diff)),
        mean_abs=float(np.mean(diff)),
        p99_abs=float(np.percentile(diff, 99.0)),
        mismatches_1e_3=int(np.count_nonzero(diff > 1.0e-3)),
        mismatches_1e_2=int(np.count_nonzero(diff > 1.0e-2)),
        mismatches_5e_2=int(np.count_nonzero(diff > 5.0e-2)),
    )


def unpack_chunk_batch(packed_chunks: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if packed_chunks.ndim != 2 or packed_chunks.shape[1] != CHUNK_BYTES:
        raise ValueError(f"bad Q4NX chunk array shape: {packed_chunks.shape}")
    count = packed_chunks.shape[0]
    chunks = np.ascontiguousarray(packed_chunks)
    scales_raw = chunks[:, :Q4NX_SCALE_BYTES].view(bfloat16).astype(np.float32)
    zeros_raw = chunks[:, Q4NX_SCALE_BYTES:Q4NX_DATA_OFFSET].view(bfloat16).astype(np.float32)
    scales = scales_raw.reshape(count, GROUPS_PER_CHUNK, M_PER_TILE).transpose(0, 2, 1)
    zeros = zeros_raw.reshape(count, GROUPS_PER_CHUNK, M_PER_TILE).transpose(0, 2, 1)

    packed_u8 = chunks[:, Q4NX_DATA_OFFSET:]
    weights_u4 = np.empty((count, M_PER_TILE, ACT_SLICE_BF16), dtype=np.float32)
    for lane in range(Q4NX_LANES):
        lane_base = lane * Q4NX_DATA_BYTES_PER_LANE
        row_base = lane * Q4NX_ROWS_PER_LANE
        lane_bytes = packed_u8[
            :, lane_base : lane_base + Q4NX_DATA_BYTES_PER_LANE
        ].reshape(count, ACT_SLICE_BF16, Q4NX_ROWS_PER_LANE // 2)
        for byte_idx in range(Q4NX_ROWS_PER_LANE // 2):
            row0 = row_base + byte_idx * 2
            row1 = row0 + 1
            byte_values = lane_bytes[:, :, byte_idx]
            weights_u4[:, row0, :] = (byte_values & 0x0F).astype(np.float32)
            weights_u4[:, row1, :] = (byte_values >> 4).astype(np.float32)
    return scales, zeros, weights_u4


def q4nx_variant_matvec_from_chunks(
    packed_chunks: np.ndarray,
    activation_slice: np.ndarray,
    variant: Variant,
    batch_chunks: int = 256,
) -> np.ndarray:
    if activation_slice.shape != (ACT_SLICE_BF16,):
        raise ValueError(f"activation slice shape mismatch: {activation_slice.shape} != {(ACT_SLICE_BF16,)}")
    grouped_act = activation_slice.astype(np.float32).reshape(GROUPS_PER_CHUNK, GROUP_SIZE)
    group_sums_fp32 = np.sum(grouped_act, axis=1, dtype=np.float32)
    group_sums_bf16 = bf16_round(group_sums_fp32)
    group_sums = group_sums_bf16 if variant.endswith("bf16_sum") else group_sums_fp32
    output = np.empty((packed_chunks.shape[0], M_PER_TILE), dtype=np.float32)
    for start in range(0, packed_chunks.shape[0], batch_chunks):
        end = min(start + batch_chunks, packed_chunks.shape[0])
        scales, zeros, weights_u4 = unpack_chunk_batch(packed_chunks[start:end])
        grouped_weights = weights_u4.reshape(end - start, M_PER_TILE, GROUPS_PER_CHUNK, GROUP_SIZE)
        scaled = bf16_round(grouped_weights * scales[..., None])
        if variant.startswith("centered"):
            dequant = bf16_round(scaled + zeros[..., None])
            coeff = dequant - zeros[..., None]
        else:
            coeff = scaled
        mac = np.einsum("cmgd,gd->cm", coeff, grouped_act, optimize=True)
        correction = np.einsum("cmg,g->cm", zeros, group_sums, optimize=True)
        output[start:end] = mac + correction
    return output


def project_q4nx_variant_bf16(
    projection: ProjectionTensor,
    chunks: np.ndarray,
    activation: np.ndarray,
    variant: Variant,
) -> np.ndarray:
    if activation.shape != (projection.input_dim,):
        raise ValueError(f"{projection.phase} activation shape mismatch: {activation.shape}")
    output_chunks = projection.output_chunks
    accum = np.zeros((output_chunks, M_PER_TILE), dtype=np.float32)
    output_indices = np.arange(output_chunks, dtype=np.int32) * projection.chunks
    for input_chunk in range(projection.chunks):
        source = output_indices + input_chunk
        start = input_chunk * ACT_SLICE_BF16
        accum += q4nx_variant_matvec_from_chunks(
            chunks[source],
            activation[start : start + ACT_SLICE_BF16],
            variant,
        )
    return accum.reshape(projection.output_dim).astype(bfloat16)


class Qwen3VariantLayerReference:
    def __init__(self, base: Qwen3LayerReference, variant: Variant) -> None:
        self.base = base
        self.variant = variant

    def project(self, phase: str, activation: np.ndarray) -> np.ndarray:
        return project_q4nx_variant_bf16(
            self.base.projections[phase],
            self.base.chunks[phase],
            activation,
            self.variant,
        )

    def forward(self, inputs: LayerInputs) -> LayerReferenceResult:
        if inputs.hidden.shape != (HIDDEN_DIM,):
            raise ValueError(f"hidden shape mismatch: {inputs.hidden.shape}")
        expected_cache_shape = (inputs.current_token + 1, NUM_KV_HEADS, HEAD_DIM)
        if inputs.k_cache.shape != expected_cache_shape or inputs.v_cache.shape != expected_cache_shape:
            raise ValueError(f"KV cache shape mismatch: {inputs.k_cache.shape}/{inputs.v_cache.shape}")

        norm_hidden = self.base.input_norm_activation(inputs.hidden)
        q_raw = self.project(Q_PROJECTION, norm_hidden)
        k_raw = self.project(K_PROJECTION, norm_hidden)
        v = self.project(V_PROJECTION, norm_hidden)

        q = _apply_rope(
            _head_rms_norm(q_raw, self.base.q_norm_weight, self.base.model.config.rms_norm_eps),
            inputs.current_token,
            self.base.model.config.rope_theta,
        )
        k = _apply_rope(
            _head_rms_norm(k_raw, self.base.k_norm_weight, self.base.model.config.rms_norm_eps),
            inputs.current_token,
            self.base.model.config.rope_theta,
        )

        k_cache = inputs.k_cache.copy()
        v_cache = inputs.v_cache.copy()
        k_cache[inputs.current_token, :, :] = k.reshape(NUM_KV_HEADS, HEAD_DIM)
        v_cache[inputs.current_token, :, :] = v.reshape(NUM_KV_HEADS, HEAD_DIM)

        attention = _attention(q, k_cache, v_cache, inputs.current_token)
        o = self.project(O_PROJECTION, attention)
        post_attention = (inputs.hidden.astype(np.float32) + o.astype(np.float32)).astype(bfloat16)
        ffn_input = _rms_norm(
            post_attention,
            self.base.post_attention_norm_weight,
            self.base.model.config.rms_norm_eps,
        )
        up = self.project(UP_PROJECTION, ffn_input)
        gate = self.project(GATE_PROJECTION, ffn_input)
        swiglu = (_silu(gate) * up.astype(np.float32)).astype(bfloat16)
        down = self.project(DOWN_PROJECTION, swiglu)
        hidden_out = (post_attention.astype(np.float32) + down.astype(np.float32)).astype(bfloat16)
        return LayerReferenceResult(
            input_norm=norm_hidden,
            q_raw=q_raw,
            k_raw=k_raw,
            v_raw=v,
            q=q,
            k=k,
            v=v,
            hidden_out=hidden_out,
            k_cache=k_cache,
            v_cache=v_cache,
            attention=attention,
            o=o,
            post_attention=post_attention,
            ffn_input=ffn_input,
            up=up,
            gate=gate,
            swiglu=swiglu,
            down=down,
        )


def projection_cases(exact: LayerReferenceResult) -> tuple[tuple[str, np.ndarray, np.ndarray], ...]:
    return (
        (Q_PROJECTION, exact.input_norm, exact.q_raw),
        (K_PROJECTION, exact.input_norm, exact.k_raw),
        (V_PROJECTION, exact.input_norm, exact.v_raw),
        (O_PROJECTION, exact.attention, exact.o),
        (UP_PROJECTION, exact.ffn_input, exact.up),
        (GATE_PROJECTION, exact.ffn_input, exact.gate),
        (DOWN_PROJECTION, exact.swiglu, exact.down),
    )


def isolated_projection_stats(reference: Qwen3LayerReference, exact: LayerReferenceResult) -> tuple[DiffStats, ...]:
    stats: list[DiffStats] = []
    for phase, activation, expected in projection_cases(exact):
        projection = reference.projections[phase]
        chunks = reference.chunks[phase]
        for variant in VARIANTS:
            got = project_q4nx_variant_bf16(projection, chunks, activation, variant)
            stats.append(diff_stats(f"isolated_{phase}", variant, got, expected))
    return tuple(stats)


def full_forward_stats(reference: Qwen3LayerReference, inputs: LayerInputs, exact: LayerReferenceResult) -> tuple[DiffStats, ...]:
    stats: list[DiffStats] = []
    for variant in ("centered_bf16_sum", "centered_fp32_sum"):
        result = Qwen3VariantLayerReference(reference, variant).forward(inputs)
        stats.extend(
            (
                diff_stats("full_q_raw", variant, result.q_raw, exact.q_raw),
                diff_stats("full_k_raw", variant, result.k_raw, exact.k_raw),
                diff_stats("full_v_raw", variant, result.v_raw, exact.v_raw),
                diff_stats("full_attention", variant, result.attention, exact.attention),
                diff_stats("full_o", variant, result.o, exact.o),
                diff_stats("full_up", variant, result.up, exact.up),
                diff_stats("full_gate", variant, result.gate, exact.gate),
                diff_stats("full_swiglu", variant, result.swiglu, exact.swiglu),
                diff_stats("full_down", variant, result.down, exact.down),
                diff_stats("full_hidden_out", variant, result.hidden_out, exact.hidden_out),
            )
        )
    return tuple(stats)


def projection_coefficient_stats(
    projection: ProjectionTensor,
    chunks: np.ndarray,
    activation: np.ndarray,
    batch_chunks: int = 128,
) -> CoefficientStats:
    centered_delta = RunningAbsStats()
    zero_round = RunningAbsStats()
    output_indices = np.arange(projection.output_chunks, dtype=np.int32) * projection.chunks
    for input_chunk in range(projection.chunks):
        source = output_indices + input_chunk
        start = input_chunk * ACT_SLICE_BF16
        grouped_act = activation[start : start + ACT_SLICE_BF16].astype(np.float32).reshape(
            GROUPS_PER_CHUNK,
            GROUP_SIZE,
        )
        group_sums_fp32 = np.sum(grouped_act, axis=1, dtype=np.float32)
        group_sums_bf16 = bf16_round(group_sums_fp32)
        group_sum_delta = group_sums_fp32 - group_sums_bf16
        source_chunks = chunks[source]
        for batch_start in range(0, source_chunks.shape[0], batch_chunks):
            batch_end = min(batch_start + batch_chunks, source_chunks.shape[0])
            scales, zeros, weights_u4 = unpack_chunk_batch(source_chunks[batch_start:batch_end])
            grouped_weights = weights_u4.reshape(batch_end - batch_start, M_PER_TILE, GROUPS_PER_CHUNK, GROUP_SIZE)
            scaled = bf16_round(grouped_weights * scales[..., None])
            dequant = bf16_round(scaled + zeros[..., None])
            centered = dequant - zeros[..., None]
            centered_delta.update(centered - scaled)
            zero_round.update(zeros * group_sum_delta[None, None, :])
    return CoefficientStats(
        name=projection.phase,
        centered_delta_values=centered_delta.count,
        centered_delta_nonzero=centered_delta.nonzero,
        centered_delta_max_abs=centered_delta.max_abs,
        centered_delta_mean_abs=centered_delta.mean_abs,
        zero_sum_round_values=zero_round.count,
        zero_sum_round_nonzero=zero_round.nonzero,
        zero_sum_round_max_abs=zero_round.max_abs,
        zero_sum_round_mean_abs=zero_round.mean_abs,
    )


def coefficient_stats(reference: Qwen3LayerReference, exact: LayerReferenceResult) -> tuple[CoefficientStats, ...]:
    stats: list[CoefficientStats] = []
    for phase, activation, _ in projection_cases(exact):
        stats.append(
            projection_coefficient_stats(
                reference.projections[phase],
                reference.chunks[phase],
                activation,
            )
        )
    return tuple(stats)


def render_diff_table(stats: tuple[DiffStats, ...]) -> list[str]:
    lines = [
        "| Stage | Variant | Shape | Max Abs | P99 Abs | Mean Abs | >1e-3 | >1e-2 | >5e-2 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stat in stats:
        lines.append(
            f"| `{stat.name}` | `{stat.variant}` | `{stat.shape}` | {stat.max_abs:.9f} | "
            f"{stat.p99_abs:.9f} | {stat.mean_abs:.9f} | {stat.mismatches_1e_3} | "
            f"{stat.mismatches_1e_2} | {stat.mismatches_5e_2} |"
        )
    return lines


def render_coefficient_table(stats: tuple[CoefficientStats, ...]) -> list[str]:
    lines = [
        "| Phase | Center Delta Values | Center Delta Nonzero | Center Delta Max | "
        "Center Delta Mean | Zero-Sum-Round Values | Zero-Sum-Round Nonzero | "
        "Zero-Sum-Round Max | Zero-Sum-Round Mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for stat in stats:
        lines.append(
            f"| `{stat.name}` | {stat.centered_delta_values} | {stat.centered_delta_nonzero} | "
            f"{stat.centered_delta_max_abs:.9f} | {stat.centered_delta_mean_abs:.9f} | "
            f"{stat.zero_sum_round_values} | {stat.zero_sum_round_nonzero} | "
            f"{stat.zero_sum_round_max_abs:.9f} | {stat.zero_sum_round_mean_abs:.9f} |"
        )
    return lines


def render_report(
    layer: int,
    token: int,
    hidden_source: str,
    elapsed_seconds: float,
    isolated: tuple[DiffStats, ...],
    full: tuple[DiffStats, ...],
    coeffs: tuple[CoefficientStats, ...],
) -> str:
    centered_fp32_hidden = next(
        stat
        for stat in full
        if stat.name == "full_hidden_out" and stat.variant == "centered_fp32_sum"
    )
    if centered_fp32_hidden.mismatches_1e_2 == 0:
        decision = "centered_fp32_sum stays within the current full-layer 1e-2 gate"
    else:
        decision = (
            "centered_fp32_sum removes the systematic scaled-only drift but still has "
            "a small contracted-sum residual against the current exact order"
        )
    lines = [
        "# Qwen3 Q4NX Centered-Dequant Contract",
        "",
        f"- Layer: `{layer}`",
        f"- Current token: `{token}`",
        f"- Hidden source: `{hidden_source}`",
        f"- Elapsed seconds: `{elapsed_seconds:.3f}`",
        "",
        "## Formula",
        "",
        "```text",
        "exact current reference:",
        "  sum(bf16(bf16(q * scale) + zero) * activation)",
        "",
        "weak group sum from exp108:",
        "  sum(bf16(q * scale) * activation) + zero * sum(activation_group)",
        "",
        "centered-dequant group sum:",
        "  centered = bf16(bf16(q * scale) + zero) - zero",
        "  exact    = sum(centered * activation) + zero * sum(activation_group)",
        "```",
        "",
        "The last line is algebraically exact over real numbers. On NPU fp32 it is",
        "not necessarily bit-identical to the current direct order, because one",
        "`zero * sum(activation_group)` correction contracts 32 separate",
        "`zero * activation_dim` products into a different floating-point order.",
        "",
        "## Isolated Projection Delta",
        "",
        *render_diff_table(isolated),
        "",
        "## Full Layer Delta",
        "",
        *render_diff_table(full),
        "",
        "## Coefficient Shape",
        "",
        *render_coefficient_table(coeffs),
        "",
        "## Decision",
        "",
        f"`{decision}`.",
        "",
        "This means the next assembly target should not be the scaled-only",
        "group-sum body. It should use a centered-dequant coefficient path. If the",
        "production gate requires tighter than this residual, the zero correction",
        "must also preserve the direct per-dim accumulation order instead of a",
        "single contracted group-sum MAC.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_QWEN3_8B_MODEL_PATH)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--token", type=int, default=31)
    parser.add_argument("--hidden-bf16", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_hidden(path: Path) -> np.ndarray:
    values = np.fromfile(path, dtype=bfloat16)
    if values.shape != (HIDDEN_DIM,):
        raise ValueError(f"{path} hidden shape mismatch: {values.shape} != {(HIDDEN_DIM,)}")
    return values.copy()


def main() -> int:
    args = parse_args()
    started = time.perf_counter()
    model = Qwen3Q4NXModel(args.model_path)
    reference = Qwen3LayerReference(model, args.layer)
    inputs = make_reference_inputs(args.token)
    hidden_source = "synthetic make_reference_inputs"
    if args.hidden_bf16 is not None:
        inputs = LayerInputs(
            hidden=load_hidden(args.hidden_bf16),
            k_cache=inputs.k_cache,
            v_cache=inputs.v_cache,
            current_token=inputs.current_token,
        )
        hidden_source = str(args.hidden_bf16)
    exact = reference.forward(inputs)
    isolated = isolated_projection_stats(reference, exact)
    full = full_forward_stats(reference, inputs, exact)
    coeffs = coefficient_stats(reference, exact)
    elapsed = time.perf_counter() - started
    args.output.write_text(
        render_report(args.layer, args.token, hidden_source, elapsed, isolated, full, coeffs)
    )
    print(f"wrote {args.output}")
    print(f"elapsed_seconds={elapsed:.3f}")
    for stat in isolated:
        if stat.variant in ("scaled_bf16_sum", "centered_fp32_sum"):
            print(
                f"{stat.name}/{stat.variant}: "
                f"max_abs={stat.max_abs:.9f} >1e-2={stat.mismatches_1e_2}"
            )
    for stat in full:
        print(
            f"{stat.name}/{stat.variant}: "
            f"max_abs={stat.max_abs:.9f} >1e-2={stat.mismatches_1e_2}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
