#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import gc
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen3-0.6B"
DEFAULT_PROMPT = "What is the capital of France? Answer with only the city name."
LOCAL_QWEN3_0_6B_SNAPSHOT = (
    Path.home()
    / ".cache"
    / "huggingface"
    / "hub"
    / "models--Qwen--Qwen3-0.6B"
    / "snapshots"
    / "c1899de289a04d12100db370d81485cdf75e47ca"
)
QWEN3_MODEL_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
)
QWEN3_SAFETENSOR_PATTERNS = ("model.safetensors", "model-*.safetensors")


@dataclass(frozen=True)
class Qwen3Config:
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    vocab_size: int
    bos_token_id: int
    eos_token_id: int
    tie_word_embeddings: bool
    torch_dtype: str

    @classmethod
    def from_json(cls, path: Path):
        data = json.loads(path.read_text())
        return cls(
            hidden_size=data["hidden_size"],
            intermediate_size=data["intermediate_size"],
            num_hidden_layers=data["num_hidden_layers"],
            num_attention_heads=data["num_attention_heads"],
            num_key_value_heads=data["num_key_value_heads"],
            head_dim=data.get(
                "head_dim", data["hidden_size"] // data["num_attention_heads"]
            ),
            rms_norm_eps=float(data["rms_norm_eps"]),
            rope_theta=float(data["rope_theta"]),
            vocab_size=data["vocab_size"],
            bos_token_id=data["bos_token_id"],
            eos_token_id=data["eos_token_id"],
            tie_word_embeddings=bool(data.get("tie_word_embeddings", False)),
            torch_dtype=data.get("torch_dtype", "bfloat16"),
        )


def _repo_cache_name(repo_id: str) -> str:
    return f"models--{repo_id.replace('/', '--')}"


def _hf_hub_cache_roots() -> list[Path]:
    roots: list[Path] = []
    hf_hub_cache = os.environ.get("HF_HUB_CACHE")
    if hf_hub_cache:
        roots.append(Path(hf_hub_cache).expanduser())

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        roots.append(Path(hf_home).expanduser() / "hub")

    roots.append(Path.home() / ".cache" / "huggingface" / "hub")

    deduped: list[Path] = []
    seen: set[Path] = set()
    for root in roots:
        resolved = root.resolve() if root.exists() else root
        if resolved not in seen:
            deduped.append(root)
            seen.add(resolved)
    return deduped


def _has_qwen3_model_files(path: Path) -> bool:
    if not all((path / name).exists() for name in QWEN3_MODEL_FILES):
        return False
    return any(path.glob(pattern) for pattern in QWEN3_SAFETENSOR_PATTERNS)


def _local_hf_snapshot_dir(repo_id: str, revision: str | None = None) -> Path | None:
    """Return an existing HF cache snapshot before falling back to network."""

    for cache_root in _hf_hub_cache_roots():
        repo_cache = cache_root / _repo_cache_name(repo_id)
        snapshots_dir = repo_cache / "snapshots"
        candidates: list[Path] = []

        if revision:
            ref_file = repo_cache / "refs" / revision
            if ref_file.exists():
                candidates.append(snapshots_dir / ref_file.read_text().strip())
            candidates.append(snapshots_dir / revision)
        else:
            main_ref = repo_cache / "refs" / "main"
            if main_ref.exists():
                candidates.append(snapshots_dir / main_ref.read_text().strip())

        if snapshots_dir.exists():
            candidates.extend(
                sorted(
                    (path for path in snapshots_dir.iterdir() if path.is_dir()),
                    key=lambda path: path.stat().st_mtime,
                    reverse=True,
                )
            )

        for candidate in candidates:
            if candidate.exists() and _has_qwen3_model_files(candidate):
                return candidate
    return None


def resolve_model_dir(model: str, revision: str | None = None) -> Path:
    model_path = Path(model).expanduser()
    if model_path.exists():
        return model_path

    if (
        model == DEFAULT_MODEL
        and revision is None
        and not os.environ.get("HF_HUB_CACHE")
        and not os.environ.get("HF_HOME")
        and _has_qwen3_model_files(LOCAL_QWEN3_0_6B_SNAPSHOT)
    ):
        return LOCAL_QWEN3_0_6B_SNAPSHOT

    local_model_dir = _local_hf_snapshot_dir(model, revision)
    if local_model_dir is not None:
        return local_model_dir

    local_dir = snapshot_download(
        repo_id=model,
        revision=revision,
        allow_patterns=[
            "config.json",
            "generation_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "model.safetensors",
            "model-*.safetensors",
            "model.safetensors.index.json",
        ],
    )
    return Path(local_dir)


def load_safetensors(model_dir: Path) -> dict[str, torch.Tensor]:
    tensor_files = sorted(model_dir.glob("*.safetensors"))
    if not tensor_files:
        raise FileNotFoundError(f"No safetensors files found in {model_dir}")

    tensors: dict[str, torch.Tensor] = {}
    for tensor_file in tensor_files:
        tensors.update(load_file(tensor_file, device="cpu"))
    return tensors


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = x.dtype
    variance = x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps).to(dtype=input_dtype)
    return x * weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    position_ids: torch.Tensor,
    head_dim: int,
    rope_theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (
        rope_theta
        ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32, device=q.device)
            / head_dim
        )
    )
    freqs = torch.outer(position_ids.to(torch.float32), inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos = emb.cos().to(dtype=q.dtype).unsqueeze(0).unsqueeze(0)
    sin = emb.sin().to(dtype=q.dtype).unsqueeze(0).unsqueeze(0)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


def repeat_kv(hidden_states: torch.Tensor, repeats: int) -> torch.Tensor:
    if repeats == 1:
        return hidden_states
    batch, num_kv_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_kv_heads, repeats, seq_len, head_dim
    )
    return hidden_states.reshape(batch, num_kv_heads * repeats, seq_len, head_dim)


class Qwen3ForCausalLM:
    def __init__(self, model_dir: Path, dtype: torch.dtype = torch.bfloat16):
        self.model_dir = model_dir
        self.config = Qwen3Config.from_json(model_dir / "config.json")
        self.weights = load_safetensors(model_dir)
        self.dtype = dtype

        for name, tensor in list(self.weights.items()):
            if tensor.is_floating_point():
                self.weights[name] = tensor.to(dtype=dtype)

    def w(self, name: str) -> torch.Tensor:
        return self.weights[name]

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(input_ids, self.w("model.embed_tokens.weight"))

    def attention(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        cfg = self.config
        prefix = f"model.layers.{layer_idx}.self_attn"
        batch, seq_len, _ = x.shape

        q = F.linear(x, self.w(f"{prefix}.q_proj.weight"))
        k = F.linear(x, self.w(f"{prefix}.k_proj.weight"))
        v = F.linear(x, self.w(f"{prefix}.v_proj.weight"))

        q = q.view(batch, seq_len, cfg.num_attention_heads, cfg.head_dim).transpose(
            1, 2
        )
        k = k.view(batch, seq_len, cfg.num_key_value_heads, cfg.head_dim).transpose(
            1, 2
        )
        v = v.view(batch, seq_len, cfg.num_key_value_heads, cfg.head_dim).transpose(
            1, 2
        )

        q = rms_norm(q, self.w(f"{prefix}.q_norm.weight"), cfg.rms_norm_eps)
        k = rms_norm(k, self.w(f"{prefix}.k_norm.weight"), cfg.rms_norm_eps)

        position_ids = torch.arange(seq_len, device=x.device)
        q, k = apply_rope(q, k, position_ids, cfg.head_dim, cfg.rope_theta)

        repeats = cfg.num_attention_heads // cfg.num_key_value_heads
        k = repeat_kv(k, repeats)
        v = repeat_kv(v, repeats)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(cfg.head_dim)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores.to(torch.float32), dim=-1).to(dtype=x.dtype)
        context = torch.matmul(probs, v)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return F.linear(context, self.w(f"{prefix}.o_proj.weight"))

    def mlp(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        prefix = f"model.layers.{layer_idx}.mlp"
        gate = F.linear(x, self.w(f"{prefix}.gate_proj.weight"))
        up = F.linear(x, self.w(f"{prefix}.up_proj.weight"))
        hidden = F.silu(gate) * up
        return F.linear(hidden, self.w(f"{prefix}.down_proj.weight"))

    @torch.inference_mode()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        x = self.embed(input_ids).to(dtype=self.dtype)

        for layer_idx in range(cfg.num_hidden_layers):
            layer_prefix = f"model.layers.{layer_idx}"
            residual = x
            x = rms_norm(
                x, self.w(f"{layer_prefix}.input_layernorm.weight"), cfg.rms_norm_eps
            )
            x = residual + self.attention(x, layer_idx)

            residual = x
            x = rms_norm(
                x,
                self.w(f"{layer_prefix}.post_attention_layernorm.weight"),
                cfg.rms_norm_eps,
            )
            x = residual + self.mlp(x, layer_idx)

        x = rms_norm(x, self.w("model.norm.weight"), cfg.rms_norm_eps)
        if cfg.tie_word_embeddings:
            lm_head = self.w("model.embed_tokens.weight")
        else:
            lm_head = self.w("lm_head.weight")
        return F.linear(x, lm_head)

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        generated = input_ids.clone()
        eos_token_id = (
            self.config.eos_token_id if eos_token_id is None else eos_token_id
        )

        for _ in range(max_new_tokens):
            logits = self.forward(generated)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=-1)
            if int(next_token.item()) == eos_token_id:
                break
        return generated


def encode_prompt(tokenizer, prompt: str, raw_prompt: bool, enable_thinking: bool):
    if raw_prompt:
        encoded = tokenizer(prompt, return_tensors="pt")
        return encoded.input_ids

    messages = [{"role": "user", "content": prompt}]
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
            return_tensors="pt",
        )
    except TypeError:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    if isinstance(encoded, torch.Tensor):
        return encoded
    if hasattr(encoded, "input_ids"):
        return encoded.input_ids
    return torch.tensor([encoded], dtype=torch.long)


def verify_with_hf(
    model_dir: Path,
    input_ids: torch.Tensor,
    generated_ids: torch.Tensor,
    max_new_tokens: int,
) -> tuple[bool, torch.Tensor]:
    from transformers import AutoModelForCausalLM

    try:
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=torch.bfloat16, device_map=None
        )
    except TypeError:
        hf_model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, device_map=None
        )
    hf_model.eval()

    with torch.inference_mode():
        hf_generated = hf_model.generate(
            input_ids,
            attention_mask=torch.ones_like(input_ids),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=hf_model.config.eos_token_id,
        )
    return torch.equal(generated_ids, hf_generated), hf_generated


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-0.6B CPU inference")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="HF repo id or local dir"
    )
    parser.add_argument("--revision", default=None, help="Optional HF revision")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--raw-prompt", action="store_true")
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="Pass enable_thinking=True to the Qwen chat template when supported",
    )
    parser.add_argument(
        "--verify-hf",
        action="store_true",
        help="Compare greedy output token ids against Transformers reference",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = resolve_model_dir(args.model, args.revision)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    input_ids = encode_prompt(
        tokenizer,
        prompt=args.prompt,
        raw_prompt=args.raw_prompt,
        enable_thinking=args.enable_thinking,
    )

    model = Qwen3ForCausalLM(model_dir)
    generated = model.generate(input_ids, args.max_new_tokens)
    new_tokens = generated[:, input_ids.shape[1] :]
    text = tokenizer.decode(new_tokens[0], skip_special_tokens=True)

    print(f"model_dir: {model_dir}")
    print(f"input_ids: {input_ids.tolist()[0]}")
    print(f"generated_ids: {generated.tolist()[0]}")
    print(f"new_text: {text!r}")

    if args.verify_hf:
        del model
        gc.collect()
        matches, hf_generated = verify_with_hf(
            model_dir, input_ids, generated, args.max_new_tokens
        )
        print(f"hf_generated_ids: {hf_generated.tolist()[0]}")
        print(f"hf_match: {matches}")
        if not matches:
            hf_new_text = tokenizer.decode(
                hf_generated[0, input_ids.shape[1] :], skip_special_tokens=True
            )
            print(f"hf_new_text: {hf_new_text!r}")
            raise SystemExit(1)


if __name__ == "__main__":
    main()
