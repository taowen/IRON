"""Qwen3 Q4NX model assets for the fused decode layer."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

from contract import (
    HEAD_DIM,
    HIDDEN_DIM,
    INTERMEDIATE_DIM,
    NUM_KV_HEADS,
    NUM_Q_HEADS,
    PHASE_SPECS,
)
from q4nx_reference import CHUNK_BYTES

DEFAULT_QWEN3_8B_MODEL_PATH = Path("/var/home/taowen/flm/models/Qwen3-8B-NPU2")
MODEL_FILES = ("config.json", "model.q4nx", "tokenizer.json", "tokenizer_config.json")
QWEN3_8B_BASE_URL = "https://huggingface.co/FastFlowLM/Qwen3-8B-NPU2/resolve/v0.9.22-faster-q4-1"


@dataclass(frozen=True)
class TensorMeta:
    name: str
    dtype: str
    shape: tuple[int, ...]
    offsets: tuple[int, int]

    @property
    def byte_size(self) -> int:
        return self.offsets[1] - self.offsets[0]


@dataclass(frozen=True)
class Qwen3Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_hidden_layers: int
    rms_norm_eps: float
    rope_theta: float


@dataclass(frozen=True)
class ProjectionTensor:
    phase: str
    tensor_name: str
    input_dim: int
    output_dim: int
    blocks: int
    chunks: int

    @property
    def output_chunks(self) -> int:
        return self.output_dim // 32

    @property
    def chunk_count(self) -> int:
        return self.output_chunks * self.chunks


PROJECTION_SUFFIXES = {
    "Q": "self_attn.q_proj.weight",
    "K": "self_attn.k_proj.weight",
    "V": "self_attn.v_proj.weight",
    "O": "self_attn.o_proj.weight",
    "UP": "mlp.up_proj.weight",
    "GATE": "mlp.gate_proj.weight",
    "DOWN": "mlp.down_proj.weight",
}
QKV_BODY_PHASES = ("Q", "K", "V")


def model_file_url(filename: str) -> str:
    if filename not in MODEL_FILES:
        raise ValueError(f"unknown Qwen3-8B model file: {filename}")
    return f"{QWEN3_8B_BASE_URL}/{filename}?download=true"


def missing_model_files(model_path: Path) -> tuple[Path, ...]:
    return tuple(model_path / filename for filename in MODEL_FILES if not (model_path / filename).exists())


def _read_config(model_path: Path) -> Qwen3Config:
    config_path = model_path / "config.json"
    with config_path.open(encoding="utf-8") as f:
        raw = json.load(f)
    return Qwen3Config(
        vocab_size=int(raw["vocab_size"]),
        hidden_size=int(raw["hidden_size"]),
        intermediate_size=int(raw["intermediate_size"]),
        num_attention_heads=int(raw["num_attention_heads"]),
        num_key_value_heads=int(raw["num_key_value_heads"]),
        head_dim=int(raw["head_dim"]),
        num_hidden_layers=int(raw["num_hidden_layers"]),
        rms_norm_eps=float(raw["rms_norm_eps"]),
        rope_theta=float(raw["rope_theta"]),
    )


def _require_qwen3_8b_config(config: Qwen3Config) -> None:
    expected = Qwen3Config(
        vocab_size=151936,
        hidden_size=HIDDEN_DIM,
        intermediate_size=INTERMEDIATE_DIM,
        num_attention_heads=NUM_Q_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        num_hidden_layers=36,
        rms_norm_eps=1.0e-6,
        rope_theta=1_000_000.0,
    )
    mismatches: list[str] = []
    if config.vocab_size != expected.vocab_size:
        mismatches.append(f"vocab_size={config.vocab_size}")
    if config.hidden_size != expected.hidden_size:
        mismatches.append(f"hidden_size={config.hidden_size}")
    if config.intermediate_size != expected.intermediate_size:
        mismatches.append(f"intermediate_size={config.intermediate_size}")
    if config.num_attention_heads != expected.num_attention_heads:
        mismatches.append(f"num_attention_heads={config.num_attention_heads}")
    if config.num_key_value_heads != expected.num_key_value_heads:
        mismatches.append(f"num_key_value_heads={config.num_key_value_heads}")
    if config.head_dim != expected.head_dim:
        mismatches.append(f"head_dim={config.head_dim}")
    if config.num_hidden_layers != expected.num_hidden_layers:
        mismatches.append(f"num_hidden_layers={config.num_hidden_layers}")
    if mismatches:
        raise ValueError("model is not Qwen3-8B-NPU2: " + ", ".join(mismatches))


def _load_q4nx_header(q4nx_path: Path) -> tuple[int, dict[str, TensorMeta]]:
    with q4nx_path.open("rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_size))
    metas: dict[str, TensorMeta] = {}
    for name, raw in header.items():
        if name == "__metadata__":
            continue
        metas[name] = TensorMeta(
            name=name,
            dtype=str(raw["dtype"]),
            shape=tuple(int(value) for value in raw["shape"]),
            offsets=(int(raw["data_offsets"][0]), int(raw["data_offsets"][1])),
        )
    return 8 + header_size, metas


def layer_projection_tensors(layer: int) -> tuple[ProjectionTensor, ...]:
    tensors: list[ProjectionTensor] = []
    for spec in PHASE_SPECS:
        suffix = PROJECTION_SUFFIXES[spec.name]
        tensors.append(
            ProjectionTensor(
                phase=spec.name,
                tensor_name=f"model.layers.{layer}.{suffix}",
                input_dim=spec.input_dim,
                output_dim=spec.output_dim,
                blocks=spec.blocks,
                chunks=spec.chunks,
            )
        )
    return tuple(tensors)


class Qwen3Q4NXModel:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        missing = missing_model_files(model_path)
        if missing:
            formatted = ", ".join(str(path) for path in missing)
            raise FileNotFoundError(f"missing Qwen3-8B model files: {formatted}")
        self.config = _read_config(model_path)
        _require_qwen3_8b_config(self.config)
        self.q4nx_path = model_path / "model.q4nx"
        self.data_start, self.tensors = _load_q4nx_header(self.q4nx_path)

    def tensor_meta(self, name: str) -> TensorMeta:
        try:
            return self.tensors[name]
        except KeyError as exc:
            raise KeyError(f"missing tensor in model.q4nx: {name}") from exc

    def tensor_bytes(self, name: str) -> np.ndarray:
        meta = self.tensor_meta(name)
        with self.q4nx_path.open("rb") as f:
            f.seek(self.data_start + meta.offsets[0])
            payload = f.read(meta.byte_size)
        if len(payload) != meta.byte_size:
            raise IOError(f"short read for {name}: {len(payload)} != {meta.byte_size}")
        return np.frombuffer(payload, dtype=np.uint8).copy()

    def tensor_bf16(self, name: str) -> np.ndarray:
        meta = self.tensor_meta(name)
        if meta.dtype != "BF16":
            raise TypeError(f"{name} dtype is {meta.dtype}, expected BF16")
        values = np.frombuffer(self.tensor_bytes(name).tobytes(), dtype=bfloat16).copy()
        expected = int(np.prod(meta.shape))
        if values.shape != (expected,):
            raise ValueError(f"{name} shape mismatch: {values.shape} != {(expected,)}")
        return values

    def token_embedding(self, token_id: int) -> np.ndarray:
        meta = self.tensor_meta("model.embed_tokens.weight")
        expected_shape = (self.config.vocab_size, HIDDEN_DIM)
        if meta.dtype != "BF16" or meta.shape != expected_shape:
            raise ValueError(
                f"model.embed_tokens.weight shape mismatch: dtype={meta.dtype} shape={meta.shape}, "
                f"expected BF16 {expected_shape}"
            )
        if token_id < 0 or token_id >= self.config.vocab_size:
            raise ValueError(f"token id out of range: {token_id}")

        row_bytes = HIDDEN_DIM * 2
        with self.q4nx_path.open("rb") as f:
            f.seek(self.data_start + meta.offsets[0] + token_id * row_bytes)
            payload = f.read(row_bytes)
        if len(payload) != row_bytes:
            raise IOError(f"short read for token embedding {token_id}: {len(payload)} != {row_bytes}")
        return np.frombuffer(payload, dtype=bfloat16).copy()

    def final_norm_weight(self) -> np.ndarray:
        weight = self.tensor_bf16("model.norm.weight")
        if weight.shape != (HIDDEN_DIM,):
            raise ValueError(f"model.norm.weight shape mismatch: {weight.shape}")
        return weight

    def lm_head_projection(self) -> ProjectionTensor:
        return ProjectionTensor(
            phase="LM_HEAD",
            tensor_name="lm_head.weight",
            input_dim=HIDDEN_DIM,
            output_dim=self.config.vocab_size,
            blocks=0,
            chunks=HIDDEN_DIM // 256,
        )

    def lm_head_chunks(self) -> np.ndarray:
        projection = self.lm_head_projection()
        meta = self.tensor_meta(projection.tensor_name)
        expected_shape = (projection.chunk_count, CHUNK_BYTES)
        if meta.dtype != "I8" or meta.shape != expected_shape:
            raise ValueError(
                f"lm_head.weight shape mismatch: dtype={meta.dtype} shape={meta.shape}, "
                f"expected I8 {expected_shape}"
            )
        return self.tensor_bytes(projection.tensor_name).reshape(expected_shape)

    def projection_chunks(self, tensor: ProjectionTensor) -> np.ndarray:
        meta = self.tensor_meta(tensor.tensor_name)
        expected_shape = (tensor.chunk_count, CHUNK_BYTES)
        if meta.dtype != "I8" or meta.shape != expected_shape:
            raise ValueError(
                f"{tensor.tensor_name} shape mismatch: dtype={meta.dtype} shape={meta.shape}, "
                f"expected I8 {expected_shape}"
            )
        return self.tensor_bytes(tensor.tensor_name).reshape(expected_shape)

    def layer_norm_weights(self, layer: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        prefix = f"model.layers.{layer}"
        input_norm = self.tensor_bf16(f"{prefix}.input_layernorm.weight")
        post_norm = self.tensor_bf16(f"{prefix}.post_attention_layernorm.weight")
        q_norm = self.tensor_bf16(f"{prefix}.self_attn.q_norm.weight")
        k_norm = self.tensor_bf16(f"{prefix}.self_attn.k_norm.weight")
        return input_norm, post_norm, q_norm, k_norm

    def _projection_stream_from_schedule(
        self,
        chunk_by_phase: dict[str, np.ndarray],
        schedule: list[tuple[ProjectionTensor, int, int]],
    ) -> np.ndarray:
        parts: list[np.ndarray] = []
        for group in range(4):
            for patch in range(2):
                for projection, block, input_chunk in schedule:
                    chunks = chunk_by_phase[projection.phase]
                    for row_in_patch in range(2):
                        row_chunk = block * 16 + group * 4 + patch * 2 + row_in_patch
                        source = row_chunk * projection.chunks + input_chunk
                        parts.append(chunks[source])
        return np.concatenate(parts).astype(np.uint8)

    def _layer_projection_stream(self, layer: int, phases: tuple[str, ...]) -> np.ndarray:
        projection_by_phase = {projection.phase: projection for projection in layer_projection_tensors(layer)}
        projections = tuple(projection_by_phase[phase] for phase in phases)
        chunk_by_phase = {projection.phase: self.projection_chunks(projection) for projection in projections}
        schedule: list[tuple[ProjectionTensor, int, int]] = []
        for projection in projections:
            for block in range(projection.blocks):
                for input_chunk in range(projection.chunks):
                    schedule.append((projection, block, input_chunk))
        return self._projection_stream_from_schedule(chunk_by_phase, schedule)

    def layer_weight_stream(self, layer: int) -> np.ndarray:
        projection_by_phase = {projection.phase: projection for projection in layer_projection_tensors(layer)}
        chunk_by_phase = {
            projection.phase: self.projection_chunks(projection)
            for projection in projection_by_phase.values()
        }
        schedule: list[tuple[ProjectionTensor, int, int]] = []

        for phase in QKV_BODY_PHASES:
            projection = projection_by_phase[phase]
            for block in range(projection.blocks):
                for input_chunk in range(projection.chunks):
                    schedule.append((projection, block, input_chunk))

        projection = projection_by_phase["O"]
        for block in range(projection.blocks):
            for input_chunk in range(projection.chunks):
                schedule.append((projection, block, input_chunk))

        up = projection_by_phase["UP"]
        gate = projection_by_phase["GATE"]
        if up.blocks != gate.blocks or up.chunks != gate.chunks:
            raise ValueError("UP/GATE stream requires matching block and chunk counts")
        for block in range(up.blocks):
            for projection in (up, gate):
                for input_chunk in range(projection.chunks):
                    schedule.append((projection, block, input_chunk))

        projection = projection_by_phase["DOWN"]
        for block in range(projection.blocks):
            for input_chunk in range(projection.chunks):
                schedule.append((projection, block, input_chunk))

        return self._projection_stream_from_schedule(chunk_by_phase, schedule)

    def layer_qkv_body_weight_stream(self, layer: int) -> np.ndarray:
        return self._layer_projection_stream(layer, QKV_BODY_PHASES)
