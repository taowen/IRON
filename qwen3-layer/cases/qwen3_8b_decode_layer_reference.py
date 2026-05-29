"""Numerical Qwen3-8B single-layer reference over MyLM Q4NX weights."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ml_dtypes import bfloat16

from contract import HEAD_DIM, HIDDEN_DIM, INTERMEDIATE_DIM, NUM_KV_HEADS, NUM_Q_HEADS
from q4nx_reference import q4nx_matvec_from_chunk
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


@dataclass(frozen=True)
class LayerInputs:
    hidden: np.ndarray
    k_cache: np.ndarray
    v_cache: np.ndarray
    current_token: int


@dataclass(frozen=True)
class LayerReferenceResult:
    hidden_out: np.ndarray
    k_cache: np.ndarray
    v_cache: np.ndarray
    q: np.ndarray
    k: np.ndarray
    v: np.ndarray
    attention: np.ndarray

    @property
    def hidden_out_i32(self) -> np.ndarray:
        return pack_bf16_i32(self.hidden_out)


def pack_bf16_i32(values: np.ndarray) -> np.ndarray:
    packed = values.astype(bfloat16).tobytes()
    return np.frombuffer(packed, dtype=np.int32).copy()


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
    cos = np.cos(angles)
    sin = np.sin(angles)
    output = np.empty_like(heads)
    even = heads[:, 0::2]
    odd = heads[:, 1::2]
    output[:, 0::2] = even * cos - odd * sin
    output[:, 1::2] = even * sin + odd * cos
    return output.reshape(values.shape).astype(bfloat16)


def _silu(values: np.ndarray) -> np.ndarray:
    x = values.astype(np.float32)
    return x / (1.0 + np.exp(-x))


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
        q = self.project(Q_PROJECTION, norm_hidden)
        k = self.project(K_PROJECTION, norm_hidden)
        v = self.project(V_PROJECTION, norm_hidden)

        q = _apply_rope(
            _head_rms_norm(q, self.q_norm_weight, self.model.config.rms_norm_eps),
            inputs.current_token,
            self.model.config.rope_theta,
        )
        k = _apply_rope(
            _head_rms_norm(k, self.k_norm_weight, self.model.config.rms_norm_eps),
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
            hidden_out=hidden_out,
            k_cache=k_cache,
            v_cache=v_cache,
            q=q,
            k=k,
            v=v,
            attention=attention,
        )


def _project_q4nx(
    projection: ProjectionTensor,
    chunks: np.ndarray,
    activation: np.ndarray,
) -> np.ndarray:
    if activation.shape != (projection.input_dim,):
        raise ValueError(f"{projection.phase} activation shape mismatch: {activation.shape}")
    output = np.empty(projection.output_dim, dtype=bfloat16)
    for block in range(projection.blocks):
        for row_chunk_in_block in range(16):
            output_chunk = block * 16 + row_chunk_in_block
            accum = np.zeros(32, dtype=np.float32)
            for input_chunk in range(projection.chunks):
                source = output_chunk * projection.chunks + input_chunk
                start = input_chunk * 256
                accum += q4nx_matvec_from_chunk(chunks[source], activation[start : start + 256])
            start_row = output_chunk * 32
            output[start_row : start_row + 32] = accum.astype(bfloat16)
    return output


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
