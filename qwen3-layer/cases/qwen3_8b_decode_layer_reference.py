"""Numerical Qwen3-8B single-layer reference over MyLM Q4NX weights."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ml_dtypes import bfloat16

from contract import HEAD_DIM, HIDDEN_DIM, INTERMEDIATE_DIM, NUM_KV_HEADS, NUM_Q_HEADS
from q4nx_reference import q4nx_matvec_from_chunks
from qwen3_model import ProjectionTensor, Qwen3Q4NXModel, layer_projection_tensors

CASE_NAME = "qwen3-8b-decode-layer"
DEFAULT_LAYER = 0
DEFAULT_CURRENT_TOKEN = 31
Q_PROJECTION = "Q"
K_PROJECTION = "K"
V_PROJECTION = "V"
O_PROJECTION = "O"
UP_PROJECTION = "UP"
GATE_PROJECTION = "GATE"
DOWN_PROJECTION = "DOWN"
ATTENTION_CONTEXT = 16
ATTENTION_HEADS_PER_WINDOW = 8
ATTENTION_KV_HEADS_PER_WINDOW = 2
ATTENTION_RSQRT_HEAD_DIM = np.float32(0.0883883476)
SIGMOID_TABLE_SCALE = 8.0
SIGMOID_TABLE = np.array(
    [
        0.5000000000, 0.5312093734, 0.5621765009, 0.5926666000, 0.6224593312,
        0.6513548647, 0.6791786992, 0.7057850278, 0.7310585786, 0.7549149869,
        0.7772998612, 0.7981867777, 0.8175744762, 0.8354835371, 0.8519528020,
        0.8670357598, 0.8807970780, 0.8933094061, 0.9046505351, 0.9149009550,
        0.9241418200, 0.9324533089, 0.9399133498, 0.9465966702, 0.9525741268,
        0.9579122721, 0.9626731127, 0.9669140216, 0.9706877692, 0.9740426428,
        0.9770226301, 0.9796676467, 0.9820137900, 0.9840936083, 0.9859363730,
        0.9875683491, 0.9890130574, 0.9902915235, 0.9914225146, 0.9924227587,
        0.9933071491, 0.9940889311, 0.9947798743, 0.9953904278, 0.9959298623,
        0.9964063974, 0.9968273172, 0.9971990730, 0.9975273768, 0.9978172836,
        0.9980732653, 0.9982992776, 0.9984988177, 0.9986749776, 0.9988304897,
        0.9989677690, 0.9990889488, 0.9991959141, 0.9992903296, 0.9993736658,
        0.9994472214, 0.9995121429, 0.9995694429, 0.9996200155, 0.9996646499,
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class LayerInputs:
    hidden: np.ndarray
    k_cache: np.ndarray
    v_cache: np.ndarray
    current_token: int


@dataclass(frozen=True)
class LayerReferenceResult:
    input_norm: np.ndarray
    q_raw: np.ndarray
    k_raw: np.ndarray
    v_raw: np.ndarray
    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    hidden_out: np.ndarray
    k_cache: np.ndarray
    v_cache: np.ndarray
    attention: np.ndarray
    o: np.ndarray
    post_attention: np.ndarray
    ffn_input: np.ndarray
    up: np.ndarray
    gate: np.ndarray
    swiglu: np.ndarray
    down: np.ndarray

    @property
    def hidden_out_i32(self) -> np.ndarray:
        return pack_bf16_i32(self.hidden_out)


@dataclass(frozen=True)
class Bf16CompareStats:
    stage: str
    max_abs: float
    mean_abs: float
    mismatch_count: int
    abs_tol: float
    rel_tol: float


def pack_bf16_i32(values: np.ndarray) -> np.ndarray:
    packed = values.astype(bfloat16).tobytes()
    return np.frombuffer(packed, dtype=np.int32).copy()


def bf16_values(values: np.ndarray) -> np.ndarray:
    if values.dtype == np.int32:
        return np.frombuffer(values.tobytes(), dtype=bfloat16).astype(np.float32)
    return values.astype(bfloat16).astype(np.float32).reshape(-1)


def bf16_compare_stats(
    stage: str,
    expected: np.ndarray,
    got: np.ndarray,
    abs_tol: float,
    rel_tol: float,
) -> Bf16CompareStats:
    expected_values = bf16_values(expected)
    got_values = bf16_values(got)
    if expected_values.shape != got_values.shape:
        raise ValueError(f"{stage} shape mismatch: {got_values.shape} != {expected_values.shape}")
    abs_err = np.abs(expected_values - got_values)
    limit = np.maximum(abs_tol, rel_tol * np.abs(expected_values))
    finite = np.isfinite(expected_values) & np.isfinite(got_values)
    mismatch = np.flatnonzero((abs_err > limit) & finite)
    return Bf16CompareStats(
        stage=stage,
        max_abs=float(np.max(abs_err)),
        mean_abs=float(np.mean(abs_err)),
        mismatch_count=int(mismatch.size),
        abs_tol=abs_tol,
        rel_tol=rel_tol,
    )


def format_bf16_compare_stats(stats: Bf16CompareStats) -> str:
    return (
        f"{stats.stage}: max_abs={stats.max_abs:.9f} "
        f"mean_abs={stats.mean_abs:.9f} mismatches={stats.mismatch_count} "
        f"abs_tol={stats.abs_tol:.9f} rel_tol={stats.rel_tol:.6f}"
    )


def make_hidden_bf16() -> np.ndarray:
    lanes = np.arange(HIDDEN_DIM, dtype=np.int32)
    raw = ((lanes * 7 + (lanes >> 5) * 13) % 127) - 63
    return (raw.astype(np.float32) / 64.0).astype(bfloat16)


def _history_value(token: int, head: int, dim: int, is_v: bool) -> float:
    if is_v:
        raw = ((head + 5) * 7 + token * 11 + dim * 2) % 127 - 63
    else:
        raw = ((head + 3) * 9 + token * 5 + dim * 3) % 127 - 63
    return raw / 256.0


def make_kv_cache_bf16(current_token: int, is_v: bool) -> np.ndarray:
    if current_token < 0:
        raise ValueError("current_token must be non-negative")
    cache = np.empty((current_token + 1, NUM_KV_HEADS, HEAD_DIM), dtype=bfloat16)
    for token in range(current_token + 1):
        for head in range(NUM_KV_HEADS):
            for dim in range(HEAD_DIM):
                cache[token, head, dim] = bfloat16(_history_value(token, head, dim, is_v))
    return cache


def make_reference_inputs(current_token: int = DEFAULT_CURRENT_TOKEN) -> LayerInputs:
    return LayerInputs(
        hidden=make_hidden_bf16(),
        k_cache=make_kv_cache_bf16(current_token, False),
        v_cache=make_kv_cache_bf16(current_token, True),
        current_token=current_token,
    )


def _rms_norm(values: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    x = values.astype(np.float32)
    w = weight.astype(np.float32)
    scale = np.float32(1.0 / np.sqrt(float(np.mean(x * x)) + eps))
    return (x * scale * w).astype(bfloat16)


def _head_rms_norm(values: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    heads = values.reshape(-1, HEAD_DIM)
    normalized = np.empty_like(heads)
    for head in range(heads.shape[0]):
        normalized[head] = _rms_norm(heads[head], weight, eps)
    return normalized.reshape(values.shape).astype(bfloat16)


def _apply_rope(values: np.ndarray, position: int, rope_theta: float) -> np.ndarray:
    heads = values.astype(np.float32).reshape(-1, HEAD_DIM)
    dims = np.arange(0, HEAD_DIM, 2, dtype=np.float32)
    inv_freq = np.power(np.float32(rope_theta), -dims / np.float32(HEAD_DIM))
    angles = np.float32(position) * inv_freq
    cos = np.cos(angles).astype(bfloat16).astype(np.float32)
    sin = np.sin(angles).astype(bfloat16).astype(np.float32)
    output = np.empty_like(heads)
    even = heads[:, 0::2]
    odd = heads[:, 1::2]
    output[:, 0::2] = even * cos - odd * sin
    output[:, 1::2] = even * sin + odd * cos
    return output.reshape(values.shape).astype(bfloat16)


def _silu(values: np.ndarray) -> np.ndarray:
    x = values.astype(np.float32)
    abs_x = np.abs(x)
    scaled = abs_x * np.float32(SIGMOID_TABLE_SCALE)
    index = scaled.astype(np.int32)
    clamped = np.minimum(index, SIGMOID_TABLE.shape[0] - 2)
    fraction = scaled - clamped.astype(np.float32)
    low = SIGMOID_TABLE[clamped]
    high = SIGMOID_TABLE[clamped + 1]
    sigmoid = low + (high - low) * fraction
    edge = SIGMOID_TABLE[-1]
    sigmoid = np.where(index >= SIGMOID_TABLE.shape[0] - 1, edge, sigmoid)
    sigmoid = np.where(abs_x > 8.0, 1.0, sigmoid)
    sigmoid = np.where(x < 0.0, 1.0 - sigmoid, sigmoid)
    sigmoid = np.where(x < -8.0, 0.0, sigmoid)
    return x * sigmoid


class Qwen3LayerReference:
    def __init__(self, model: Qwen3Q4NXModel, layer: int) -> None:
        self.model = model
        self.layer = layer
        self.projections = {projection.phase: projection for projection in layer_projection_tensors(layer)}
        self.chunks = {
            projection.phase: model.projection_chunks(projection)
            for projection in self.projections.values()
        }
        (
            self.input_norm_weight,
            self.post_attention_norm_weight,
            self.q_norm_weight,
            self.k_norm_weight,
        ) = model.layer_norm_weights(layer)

    def project(self, phase: str, activation: np.ndarray) -> np.ndarray:
        projection = self.projections[phase]
        return _project_q4nx(projection, self.chunks[phase], activation)

    def input_norm_activation(self, hidden: np.ndarray) -> np.ndarray:
        return _rms_norm(hidden, self.input_norm_weight, self.model.config.rms_norm_eps)

    def forward(self, inputs: LayerInputs) -> LayerReferenceResult:
        if inputs.hidden.shape != (HIDDEN_DIM,):
            raise ValueError(f"hidden shape mismatch: {inputs.hidden.shape}")
        expected_cache_shape = (inputs.current_token + 1, NUM_KV_HEADS, HEAD_DIM)
        if inputs.k_cache.shape != expected_cache_shape or inputs.v_cache.shape != expected_cache_shape:
            raise ValueError(f"KV cache shape mismatch: {inputs.k_cache.shape}/{inputs.v_cache.shape}")

        norm_hidden = self.input_norm_activation(inputs.hidden)
        q_raw = self.project(Q_PROJECTION, norm_hidden)
        k_raw = self.project(K_PROJECTION, norm_hidden)
        v = self.project(V_PROJECTION, norm_hidden)

        q = _apply_rope(
            _head_rms_norm(q_raw, self.q_norm_weight, self.model.config.rms_norm_eps),
            inputs.current_token,
            self.model.config.rope_theta,
        )
        k = _apply_rope(
            _head_rms_norm(k_raw, self.k_norm_weight, self.model.config.rms_norm_eps),
            inputs.current_token,
            self.model.config.rope_theta,
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
            self.post_attention_norm_weight,
            self.model.config.rms_norm_eps,
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


def _project_q4nx(
    projection: ProjectionTensor,
    chunks: np.ndarray,
    activation: np.ndarray,
) -> np.ndarray:
    if activation.shape != (projection.input_dim,):
        raise ValueError(f"{projection.phase} activation shape mismatch: {activation.shape}")
    output_chunks = projection.output_chunks
    accum = np.zeros((output_chunks, 32), dtype=np.float32)
    output_indices = np.arange(output_chunks, dtype=np.int32) * projection.chunks
    for input_chunk in range(projection.chunks):
        source = output_indices + input_chunk
        start = input_chunk * 256
        accum += q4nx_matvec_from_chunks(chunks[source], activation[start : start + 256])
    return accum.reshape(projection.output_dim).astype(bfloat16)


def project_q4nx_bf16(
    projection: ProjectionTensor,
    chunks: np.ndarray,
    activation: np.ndarray,
) -> np.ndarray:
    return _project_q4nx(projection, chunks, activation)


def rms_norm_bf16(values: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    return _rms_norm(values, weight, eps)


def _attention(
    q: np.ndarray,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    current_token: int,
) -> np.ndarray:
    q_heads = q.astype(np.float32).reshape(NUM_Q_HEADS, HEAD_DIM)
    k_values = k_cache[: current_token + 1].astype(np.float32)
    v_values = v_cache[: current_token + 1].astype(np.float32)
    output = np.empty((NUM_Q_HEADS, HEAD_DIM), dtype=bfloat16)
    scale = np.float32(1.0 / np.sqrt(HEAD_DIM))
    for q_head in range(NUM_Q_HEADS):
        kv_head = q_head // (NUM_Q_HEADS // NUM_KV_HEADS)
        scores = np.einsum("d,td->t", q_heads[q_head], k_values[:, kv_head, :]) * scale
        shifted = scores - np.max(scores)
        weights = np.exp(shifted)
        weights /= np.sum(weights)
        output[q_head, :] = np.einsum("t,td->d", weights, v_values[:, kv_head, :]).astype(bfloat16)
    return output.reshape(HIDDEN_DIM).astype(bfloat16)


def _floor_i32(value: np.float32) -> int:
    truncated = int(value)
    return truncated - 1 if np.float32(truncated) > value else truncated


def _pow2_i32(exponent: int) -> np.float32:
    if exponent < -126:
        return np.float32(0.0)
    if exponent > 127:
        exponent = 127
    return np.float32(np.ldexp(np.float32(1.0), exponent))


def _fast_exp_npu(value: np.float32) -> np.float32:
    if value <= np.float32(-20.0):
        return np.float32(0.0)
    if value >= np.float32(0.0):
        return np.float32(1.0)
    inv_ln2 = np.float32(1.4426950409)
    ln2 = np.float32(0.6931471806)
    exponent = _floor_i32(np.float32(value * inv_ln2))
    reduced = np.float32(value - np.float32(exponent) * ln2)
    r2 = np.float32(reduced * reduced)
    r3 = np.float32(r2 * reduced)
    r4 = np.float32(r3 * reduced)
    polynomial = np.float32(
        np.float32(1.0)
        + reduced
        + np.float32(0.5) * r2
        + np.float32(0.16666667) * r3
        + np.float32(0.04166667) * r4
    )
    return np.float32(_pow2_i32(exponent) * polynomial)


def _attention_score_bf16_npu(
    q_window: np.ndarray,
    k_block: np.ndarray,
    q_head: int,
    token: int,
) -> np.float32:
    kv_head = q_head // (ATTENTION_HEADS_PER_WINDOW // ATTENTION_KV_HEADS_PER_WINDOW)
    dot = np.float32(0.0)
    for dim in range(HEAD_DIM):
        dot = np.float32(dot + np.float32(q_window[q_head, dim] * k_block[token, kv_head, dim]))
    return np.float32(dot * ATTENTION_RSQRT_HEAD_DIM)


def attention_bf16_npu(
    q: np.ndarray,
    k_cache: np.ndarray,
    v_cache: np.ndarray,
    current_token: int,
) -> np.ndarray:
    q_heads = q.astype(np.float32).reshape(NUM_Q_HEADS, HEAD_DIM)
    k_values = k_cache[: current_token + 1].astype(np.float32)
    v_values = v_cache[: current_token + 1].astype(np.float32)
    output = np.empty((NUM_Q_HEADS, HEAD_DIM), dtype=bfloat16)
    blocks = current_token // ATTENTION_CONTEXT + 1
    for window in range(NUM_Q_HEADS // ATTENTION_HEADS_PER_WINDOW):
        q_start = window * ATTENTION_HEADS_PER_WINDOW
        kv_start = window * ATTENTION_KV_HEADS_PER_WINDOW
        q_window = q_heads[q_start : q_start + ATTENTION_HEADS_PER_WINDOW]
        k_window = k_values[:, kv_start : kv_start + ATTENTION_KV_HEADS_PER_WINDOW, :]
        v_window = v_values[:, kv_start : kv_start + ATTENTION_KV_HEADS_PER_WINDOW, :]
        accum = np.zeros((ATTENTION_HEADS_PER_WINDOW, HEAD_DIM), dtype=np.float32)
        state_max = np.zeros((ATTENTION_HEADS_PER_WINDOW,), dtype=np.float32)
        state_sum = np.zeros((ATTENTION_HEADS_PER_WINDOW,), dtype=np.float32)
        for block in range(blocks):
            block_start = block * ATTENTION_CONTEXT
            valid_tokens = ATTENTION_CONTEXT
            if block + 1 == blocks:
                valid_tokens = current_token % ATTENTION_CONTEXT + 1
            k_block = k_window[block_start : block_start + valid_tokens]
            v_block = v_window[block_start : block_start + valid_tokens]
            weights = np.zeros((ATTENTION_HEADS_PER_WINDOW, ATTENTION_CONTEXT), dtype=np.float32)
            block_max = np.zeros((ATTENTION_HEADS_PER_WINDOW,), dtype=np.float32)
            block_sum = np.zeros((ATTENTION_HEADS_PER_WINDOW,), dtype=np.float32)
            for q_head in range(ATTENTION_HEADS_PER_WINDOW):
                running_max = _attention_score_bf16_npu(q_window, k_block, q_head, 0)
                scores = np.zeros((ATTENTION_CONTEXT,), dtype=np.float32)
                scores[0] = running_max
                for token in range(1, valid_tokens):
                    scores[token] = _attention_score_bf16_npu(q_window, k_block, q_head, token)
                    if scores[token] > running_max:
                        running_max = scores[token]
                weight_sum = np.float32(0.0)
                for token in range(ATTENTION_CONTEXT):
                    weight = np.float32(0.0)
                    if token < valid_tokens:
                        weight = _fast_exp_npu(np.float32(scores[token] - running_max))
                        weight_sum = np.float32(weight_sum + weight)
                    weights[q_head, token] = np.array(weight, dtype=bfloat16).astype(np.float32)
                block_max[q_head] = running_max
                block_sum[q_head] = weight_sum
            for q_head in range(ATTENTION_HEADS_PER_WINDOW):
                old_sum = state_sum[q_head]
                new_max = block_max[q_head]
                old_scale = np.float32(0.0)
                block_scale = np.float32(1.0)
                if old_sum != np.float32(0.0):
                    old_max = state_max[q_head]
                    new_max = old_max if old_max > block_max[q_head] else block_max[q_head]
                    old_scale = _fast_exp_npu(np.float32(old_max - new_max))
                    block_scale = _fast_exp_npu(np.float32(block_max[q_head] - new_max))
                kv_head = q_head // (ATTENTION_HEADS_PER_WINDOW // ATTENTION_KV_HEADS_PER_WINDOW)
                for dim in range(HEAD_DIM):
                    block_total = np.float32(0.0)
                    for token in range(valid_tokens):
                        block_total = np.float32(
                            block_total + np.float32(weights[q_head, token] * v_block[token, kv_head, dim])
                        )
                    accum[q_head, dim] = np.float32(
                        np.float32(accum[q_head, dim] * old_scale) + np.float32(block_total * block_scale)
                    )
                state_max[q_head] = new_max
                state_sum[q_head] = np.float32(
                    np.float32(old_sum * old_scale) + np.float32(block_sum[q_head] * block_scale)
                )
        for q_head in range(ATTENTION_HEADS_PER_WINDOW):
            for dim in range(HEAD_DIM):
                output[q_start + q_head, dim] = np.array(
                    np.float32(accum[q_head, dim] / state_sum[q_head]),
                    dtype=bfloat16,
                )
    return output.reshape(HIDDEN_DIM).astype(bfloat16)


def validate_model_assets(model: Qwen3Q4NXModel, layer: int) -> list[str]:
    errors: list[str] = []
    for projection in layer_projection_tensors(layer):
        model.projection_chunks(projection)
    input_norm, post_norm, q_norm, k_norm = model.layer_norm_weights(layer)
    if input_norm.shape != (HIDDEN_DIM,):
        errors.append(f"input RMSNorm shape mismatch: {input_norm.shape}")
    if post_norm.shape != (HIDDEN_DIM,):
        errors.append(f"post RMSNorm shape mismatch: {post_norm.shape}")
    if q_norm.shape != (HEAD_DIM,):
        errors.append(f"q_norm shape mismatch: {q_norm.shape}")
    if k_norm.shape != (HEAD_DIM,):
        errors.append(f"k_norm shape mismatch: {k_norm.shape}")
    return errors


def reference_summary(result: LayerReferenceResult) -> list[str]:
    hidden = result.hidden_out.astype(np.float32)
    return [
        f"hidden_out_shape={result.hidden_out.shape}",
        f"hidden_out_min={float(np.min(hidden)):.6f}",
        f"hidden_out_max={float(np.max(hidden)):.6f}",
        f"hidden_out_mean={float(np.mean(hidden)):.6f}",
        f"hidden_out_i32_head={result.hidden_out_i32[:8].tolist()}",
    ]
