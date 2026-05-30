"""CPU Qwen3-8B reference decode over the full MyLM Q4NX model."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16
from tokenizers import Tokenizer

from cases.qwen3_8b_decode_layer_reference import (
    LayerInputs,
    LayerReferenceResult,
    Qwen3LayerReference,
    project_q4nx_bf16,
    rms_norm_bf16,
)
from contract import HEAD_DIM, HIDDEN_DIM, NUM_KV_HEADS
from qwen3_model import DEFAULT_QWEN3_8B_MODEL_PATH, Qwen3Q4NXModel


@dataclass(frozen=True)
class GreedyStep:
    position: int
    token_id: int
    decoded: str
    top_tokens: tuple[tuple[int, str, float], ...]


@dataclass(frozen=True)
class DecodeResult:
    input_token_ids: tuple[int, ...]
    generated: tuple[GreedyStep, ...]
    elapsed_seconds: float

    @property
    def generated_token_ids(self) -> tuple[int, ...]:
        return tuple(step.token_id for step in self.generated)


class Qwen3FullReference:
    def __init__(self, model: Qwen3Q4NXModel, max_positions: int, stop_layer: int) -> None:
        self.model = model
        if stop_layer <= 0 or stop_layer > model.config.num_hidden_layers:
            raise ValueError(
                f"stop_layer must be in 1..{model.config.num_hidden_layers}, got {stop_layer}"
            )
        self.stop_layer = stop_layer
        self.layers = tuple(
            Qwen3LayerReference(model, layer)
            for layer in range(stop_layer)
        )
        self.final_norm_weight = model.final_norm_weight()
        self.lm_head_projection = model.lm_head_projection()
        self.lm_head_chunks = model.lm_head_chunks()
        self.k_cache = [
            np.zeros((max_positions, NUM_KV_HEADS, HEAD_DIM), dtype=bfloat16)
            for _ in self.layers
        ]
        self.v_cache = [
            np.zeros((max_positions, NUM_KV_HEADS, HEAD_DIM), dtype=bfloat16)
            for _ in self.layers
        ]

    def forward_token(
        self,
        token_id: int,
        position: int,
        dump_dir: Path | None,
        dump_layers: tuple[int, ...],
    ) -> np.ndarray:
        hidden = self.model.token_embedding(token_id)
        for layer_index, layer in enumerate(self.layers):
            hidden_in = hidden
            inputs = LayerInputs(
                hidden=hidden,
                k_cache=self.k_cache[layer_index][: position + 1],
                v_cache=self.v_cache[layer_index][: position + 1],
                current_token=position,
            )
            result = layer.forward(inputs)
            self.k_cache[layer_index][position, :, :] = result.k_cache[position, :, :]
            self.v_cache[layer_index][position, :, :] = result.v_cache[position, :, :]
            if layer_index in dump_layers:
                _dump_layer(position, layer_index, hidden_in, result, dump_dir)
            hidden = result.hidden_out

        final_hidden = rms_norm_bf16(
            hidden,
            self.final_norm_weight,
            self.model.config.rms_norm_eps,
        )
        return project_q4nx_bf16(
            self.lm_head_projection,
            self.lm_head_chunks,
            final_hidden,
        ).astype(np.float32)


def _parse_token_ids(raw: str) -> tuple[int, ...]:
    stripped = raw.strip()
    if not stripped:
        return ()
    return tuple(int(part.strip()) for part in stripped.split(","))


def _parse_dump_layers(raw: str, stop_layer: int) -> tuple[int, ...]:
    stripped = raw.strip()
    if not stripped:
        return ()
    if stripped == "all":
        return tuple(range(stop_layer))
    layers = tuple(int(part.strip()) for part in stripped.split(","))
    for layer in layers:
        if layer < 0 or layer >= stop_layer:
            raise ValueError(f"dump layer {layer} outside executed layer range 0..{stop_layer - 1}")
    return layers


def _decode_token(tokenizer: Tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], skip_special_tokens=False)


def _top_tokens(tokenizer: Tokenizer, logits: np.ndarray, count: int) -> tuple[tuple[int, str, float], ...]:
    if count <= 0:
        raise ValueError("top token count must be positive")
    if count > logits.shape[0]:
        raise ValueError(f"top token count {count} exceeds vocab size {logits.shape[0]}")
    unsorted = np.argpartition(logits, -count)[-count:]
    sorted_ids = unsorted[np.argsort(logits[unsorted])[::-1]]
    return tuple(
        (int(token_id), _decode_token(tokenizer, int(token_id)), float(logits[token_id]))
        for token_id in sorted_ids
    )


def run_decode(
    model_path: Path,
    prompt: str,
    max_new_tokens: int,
    top_count: int,
    stop_layer: int,
    dump_dir: Path | None,
    dump_layers_raw: str,
) -> DecodeResult:
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")

    tokenizer = Tokenizer.from_file(str(model_path / "tokenizer.json"))
    input_token_ids = tuple(tokenizer.encode(prompt).ids)
    if not input_token_ids:
        raise ValueError("prompt produced no tokens")

    model = Qwen3Q4NXModel(model_path)
    reference = Qwen3FullReference(model, len(input_token_ids) + max_new_tokens, stop_layer)
    dump_layers = _parse_dump_layers(dump_layers_raw, reference.stop_layer)
    if dump_layers and dump_dir is None:
        raise ValueError("--dump-layers requires --dump-dir")
    if dump_dir is not None:
        dump_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    logits: np.ndarray | None = None
    for position, token_id in enumerate(input_token_ids):
        print(f"reference_decode: prefill position={position} token={token_id}", flush=True)
        logits = reference.forward_token(token_id, position, dump_dir, dump_layers)
        _dump_logits(position, logits, dump_dir)

    generated: list[GreedyStep] = []
    current_position = len(input_token_ids)
    for step_index in range(max_new_tokens):
        if logits is None:
            raise RuntimeError("decode has no logits after prefill")
        token_id = int(np.argmax(logits))
        generated.append(
            GreedyStep(
                position=current_position,
                token_id=token_id,
                decoded=_decode_token(tokenizer, token_id),
                top_tokens=_top_tokens(tokenizer, logits, top_count),
            )
        )
        if step_index + 1 < max_new_tokens:
            print(f"reference_decode: decode position={current_position} token={token_id}", flush=True)
            logits = reference.forward_token(token_id, current_position, dump_dir, dump_layers)
            _dump_logits(current_position, logits, dump_dir)
            current_position += 1

    return DecodeResult(
        input_token_ids=input_token_ids,
        generated=tuple(generated),
        elapsed_seconds=time.perf_counter() - started,
    )


def _format_token(token_id: int, decoded: str, logit: float | None = None) -> str:
    if logit is None:
        return f"{token_id}:{decoded!r}"
    return f"{token_id}:{decoded!r}:{logit:.6f}"


def _print_result(prompt: str, result: DecodeResult) -> None:
    print(f"prompt={prompt!r}")
    print(f"input_token_ids={list(result.input_token_ids)}")
    print(f"generated_token_ids={list(result.generated_token_ids)}")
    print("generated_text=" + "".join(step.decoded for step in result.generated))
    for step in result.generated:
        top = ", ".join(_format_token(token_id, decoded, logit) for token_id, decoded, logit in step.top_tokens)
        print(
            f"step position={step.position} token={_format_token(step.token_id, step.decoded)} "
            f"top={top}"
        )
    print(f"elapsed_seconds={result.elapsed_seconds:.3f}")


def _dump_bf16(path: Path, values: np.ndarray) -> None:
    values.astype(bfloat16).tofile(path)


def _dump_layer(
    position: int,
    layer: int,
    hidden_in: np.ndarray,
    result: LayerReferenceResult,
    dump_dir: Path | None,
) -> None:
    if dump_dir is None:
        return
    prefix = f"pos{position:04d}.layer{layer:02d}"
    for name, values in _layer_dump_items(hidden_in, result):
        _dump_bf16(dump_dir / f"{prefix}.{name}.bf16", values)


def _layer_dump_items(
    hidden_in: np.ndarray,
    result: LayerReferenceResult,
) -> tuple[tuple[str, np.ndarray], ...]:
    return (
        ("hidden_in", hidden_in),
        ("input_norm", result.input_norm),
        ("q_raw", result.q_raw),
        ("k_raw", result.k_raw),
        ("v_raw", result.v_raw),
        ("q", result.q),
        ("k", result.k),
        ("v", result.v),
        ("attention", result.attention),
        ("o", result.o),
        ("post_attention", result.post_attention),
        ("ffn_input", result.ffn_input),
        ("up", result.up),
        ("gate", result.gate),
        ("swiglu", result.swiglu),
        ("down", result.down),
        ("hidden_out", result.hidden_out),
    )


def _dump_logits(position: int, logits: np.ndarray, dump_dir: Path | None) -> None:
    if dump_dir is None:
        return
    _dump_bf16(dump_dir / f"pos{position:04d}.logits.bf16", logits)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_QWEN3_8B_MODEL_PATH)
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--expect-token-ids", default="")
    parser.add_argument("--stop-layer", type=int, default=36)
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument("--dump-layers", default="")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    expected = _parse_token_ids(args.expect_token_ids)
    result = run_decode(
        args.model_path,
        args.prompt,
        args.max_new_tokens,
        args.top_k,
        args.stop_layer,
        args.dump_dir,
        args.dump_layers,
    )
    _print_result(args.prompt, result)
    if expected and result.generated_token_ids != expected:
        raise SystemExit(
            "generated token mismatch: "
            f"got {list(result.generated_token_ids)} expected {list(expected)}"
        )


if __name__ == "__main__":
    main()
