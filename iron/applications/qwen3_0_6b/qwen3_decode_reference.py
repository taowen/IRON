#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from iron.applications.qwen3_0_6b.qwen3_cpu import (
    Qwen3ForCausalLM,
    apply_rope,
    repeat_kv,
    rms_norm,
)


@dataclass
class Qwen3DecodeState:
    keys: list[torch.Tensor]
    values: list[torch.Tensor]
    position: int


def clone_decode_state(state: Qwen3DecodeState) -> Qwen3DecodeState:
    return Qwen3DecodeState(
        keys=[tensor.clone() for tensor in state.keys],
        values=[tensor.clone() for tensor in state.values],
        position=state.position,
    )


class Qwen3CachedReference:
    def __init__(self, model: Qwen3ForCausalLM, max_seq_len: int, num_layers: int):
        self.model = model
        self.config = model.config
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        if not (1 <= num_layers <= self.config.num_hidden_layers):
            raise ValueError(
                f"num_layers must be in [1, {self.config.num_hidden_layers}], got {num_layers}"
            )

    def _attention_prefill(
        self,
        x: torch.Tensor,
        layer_idx: int,
        state: Qwen3DecodeState,
    ) -> torch.Tensor:
        cfg = self.config
        prefix = f"model.layers.{layer_idx}.self_attn"
        batch, seq_len, _ = x.shape

        q = F.linear(x, self.model.w(f"{prefix}.q_proj.weight"))
        k = F.linear(x, self.model.w(f"{prefix}.k_proj.weight"))
        v = F.linear(x, self.model.w(f"{prefix}.v_proj.weight"))

        q = q.view(batch, seq_len, cfg.num_attention_heads, cfg.head_dim).transpose(
            1, 2
        )
        k = k.view(batch, seq_len, cfg.num_key_value_heads, cfg.head_dim).transpose(
            1, 2
        )
        v = v.view(batch, seq_len, cfg.num_key_value_heads, cfg.head_dim).transpose(
            1, 2
        )

        q = rms_norm(q, self.model.w(f"{prefix}.q_norm.weight"), cfg.rms_norm_eps)
        k = rms_norm(k, self.model.w(f"{prefix}.k_norm.weight"), cfg.rms_norm_eps)
        position_ids = torch.arange(seq_len)
        q, k = apply_rope(q, k, position_ids, cfg.head_dim, cfg.rope_theta)

        state.keys[layer_idx][:, :seq_len, :] = k.squeeze(0)
        state.values[layer_idx][:, :seq_len, :] = v.squeeze(0)

        repeats = cfg.num_attention_heads // cfg.num_key_value_heads
        k_rep = repeat_kv(k, repeats)
        v_rep = repeat_kv(v, repeats)
        scores = torch.matmul(q, k_rep.transpose(-2, -1)) / math.sqrt(cfg.head_dim)
        mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores.to(torch.float32), dim=-1).to(dtype=x.dtype)
        context = torch.matmul(probs, v_rep)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return F.linear(context, self.model.w(f"{prefix}.o_proj.weight"))

    def _attention_decode(
        self,
        x: torch.Tensor,
        layer_idx: int,
        state: Qwen3DecodeState,
    ) -> torch.Tensor:
        cfg = self.config
        prefix = f"model.layers.{layer_idx}.self_attn"
        batch, seq_len, _ = x.shape
        assert batch == 1 and seq_len == 1

        q = F.linear(x, self.model.w(f"{prefix}.q_proj.weight"))
        k = F.linear(x, self.model.w(f"{prefix}.k_proj.weight"))
        v = F.linear(x, self.model.w(f"{prefix}.v_proj.weight"))

        q = q.view(batch, seq_len, cfg.num_attention_heads, cfg.head_dim).transpose(
            1, 2
        )
        k = k.view(batch, seq_len, cfg.num_key_value_heads, cfg.head_dim).transpose(
            1, 2
        )
        v = v.view(batch, seq_len, cfg.num_key_value_heads, cfg.head_dim).transpose(
            1, 2
        )

        q = rms_norm(q, self.model.w(f"{prefix}.q_norm.weight"), cfg.rms_norm_eps)
        k = rms_norm(k, self.model.w(f"{prefix}.k_norm.weight"), cfg.rms_norm_eps)
        position_ids = torch.tensor([state.position])
        q, k = apply_rope(q, k, position_ids, cfg.head_dim, cfg.rope_theta)

        state.keys[layer_idx][:, state.position : state.position + 1, :] = k.squeeze(0)
        state.values[layer_idx][:, state.position : state.position + 1, :] = v.squeeze(
            0
        )

        k_ctx = state.keys[layer_idx][:, : state.position + 1, :].unsqueeze(0)
        v_ctx = state.values[layer_idx][:, : state.position + 1, :].unsqueeze(0)
        repeats = cfg.num_attention_heads // cfg.num_key_value_heads
        k_ctx = repeat_kv(k_ctx, repeats)
        v_ctx = repeat_kv(v_ctx, repeats)
        scores = torch.matmul(q, k_ctx.transpose(-2, -1)) / math.sqrt(cfg.head_dim)
        probs = torch.softmax(scores.to(torch.float32), dim=-1).to(dtype=x.dtype)
        context = torch.matmul(probs, v_ctx)
        context = context.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return F.linear(context, self.model.w(f"{prefix}.o_proj.weight"))

    def _mlp(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        prefix = f"model.layers.{layer_idx}.mlp"
        gate = F.linear(x, self.model.w(f"{prefix}.gate_proj.weight"))
        up = F.linear(x, self.model.w(f"{prefix}.up_proj.weight"))
        hidden = F.silu(gate) * up
        return F.linear(hidden, self.model.w(f"{prefix}.down_proj.weight"))

    def new_state(self) -> Qwen3DecodeState:
        cfg = self.config
        keys = [
            torch.zeros(
                cfg.num_key_value_heads,
                self.max_seq_len,
                cfg.head_dim,
                dtype=self.model.dtype,
            )
            for _ in range(self.num_layers)
        ]
        values = [torch.zeros_like(k) for k in keys]
        return Qwen3DecodeState(keys=keys, values=values, position=0)

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, Qwen3DecodeState]:
        cfg = self.config
        state = self.new_state()
        seq_len = input_ids.shape[1]
        if seq_len >= self.max_seq_len:
            raise ValueError(
                f"prompt length {seq_len} must be smaller than max_seq_len {self.max_seq_len}"
            )

        x = self.model.embed(input_ids).to(dtype=self.model.dtype)
        for layer_idx in range(self.num_layers):
            layer_prefix = f"model.layers.{layer_idx}"
            residual = x
            x_norm = rms_norm(
                x,
                self.model.w(f"{layer_prefix}.input_layernorm.weight"),
                cfg.rms_norm_eps,
            )
            x = residual + self._attention_prefill(x_norm, layer_idx, state)

            residual = x
            x_norm = rms_norm(
                x,
                self.model.w(f"{layer_prefix}.post_attention_layernorm.weight"),
                cfg.rms_norm_eps,
            )
            x = residual + self._mlp(x_norm, layer_idx)

        x = rms_norm(x, self.model.w("model.norm.weight"), cfg.rms_norm_eps)
        logits = F.linear(x, self.model.w("model.embed_tokens.weight"))
        state.position = seq_len
        return logits, state

    @torch.inference_mode()
    def decode_hidden(
        self, token_id: int, state: Qwen3DecodeState
    ) -> tuple[torch.Tensor, Qwen3DecodeState]:
        cfg = self.config
        x = self.model.embed(torch.tensor([[token_id]], dtype=torch.long)).to(
            dtype=self.model.dtype
        )
        for layer_idx in range(self.num_layers):
            layer_prefix = f"model.layers.{layer_idx}"
            residual = x
            x_norm = rms_norm(
                x,
                self.model.w(f"{layer_prefix}.input_layernorm.weight"),
                cfg.rms_norm_eps,
            )
            x = residual + self._attention_decode(x_norm, layer_idx, state)

            residual = x
            x_norm = rms_norm(
                x,
                self.model.w(f"{layer_prefix}.post_attention_layernorm.weight"),
                cfg.rms_norm_eps,
            )
            x = residual + self._mlp(x_norm, layer_idx)

        state.position += 1
        return x, state

    @torch.inference_mode()
    def decode(
        self, token_id: int, state: Qwen3DecodeState
    ) -> tuple[torch.Tensor, Qwen3DecodeState]:
        x, state = self.decode_hidden(token_id, state)
        cfg = self.config
        x = rms_norm(x, self.model.w("model.norm.weight"), cfg.rms_norm_eps)
        logits = F.linear(x, self.model.w("model.embed_tokens.weight"))
        return logits, state
