#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import gc
import math
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

repo_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo_root))

from iron.applications.qwen3_0_6b.qwen3_cpu import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    Qwen3ForCausalLM,
    encode_prompt,
    resolve_model_dir,
)
from iron.applications.qwen3_0_6b.qwen3_decode_reference import (  # noqa: E402
    Qwen3CachedReference,
    Qwen3DecodeState,
    clone_decode_state,
)
from iron.applications.qwen3_0_6b.full_elf.debug import (  # noqa: E402
    DEBUG_STAGE_OUTPUTS,
    local_reference_tensors,
    one_layer_reference_tensors,
    parse_debug_outputs,
    print_tensor_diff,
)
from iron.common.context import AIEContext  # noqa: E402
from iron.common.fusion import FusedFullELFCallable, FusedMLIROperator  # noqa: E402
from iron.common.fusion import load_elf, patch_elf  # noqa: E402
from iron.operators import (  # noqa: E402
    ElementwiseAdd,
    ElementwiseMul,
    GEMV,
    RMSNorm,
    Repeat,
    RoPE,
    SiLU,
    Softmax,
    StridedCopy,
    Transpose,
)


def rope_lut_for_position(
    head_dim: int, rope_theta: float, position: int
) -> torch.Tensor:
    inv_freq = 1.0 / (
        rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )
    freqs = position * inv_freq
    lut = torch.empty(head_dim, dtype=torch.bfloat16)
    lut[::2] = freqs.cos().to(torch.bfloat16)
    lut[1::2] = freqs.sin().to(torch.bfloat16)
    return lut.contiguous()


class Qwen3DecodeMegakernel:
    def __init__(
        self,
        model: Qwen3ForCausalLM,
        max_seq_len: int = 256,
        num_layers: int | None = None,
        build_dir: str = "build_qwen3_megakernel",
        create_runtime: bool = True,
        debug_outputs: tuple[str, ...] = (),
    ):
        self.model = model
        self.config = model.config
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers or self.config.num_hidden_layers
        self.context = AIEContext(build_dir=build_dir)
        self.create_runtime = create_runtime
        self.debug_outputs = debug_outputs
        self.debug_output_names = {name: f"debug_{name}" for name in self.debug_outputs}
        self._validate_shapes()
        self._validate_debug_outputs()
        self._build_fused_decode()
        if self.create_runtime:
            self._load_static_buffers()

    def _validate_shapes(self):
        cfg = self.config
        if self.max_seq_len < 256 or self.max_seq_len % 256 != 0:
            raise ValueError(
                "max_seq_len must be a multiple of 256 for transpose tiles"
            )
        if cfg.hidden_size != 1024:
            raise ValueError(
                f"expected Qwen3-0.6B hidden_size=1024, got {cfg.hidden_size}"
            )
        if cfg.head_dim != 128:
            raise ValueError(f"expected Qwen3-0.6B head_dim=128, got {cfg.head_dim}")
        if cfg.num_attention_heads != 16 or cfg.num_key_value_heads != 8:
            raise ValueError(
                "expected Qwen3-0.6B GQA layout: 16 Q heads and 8 KV heads"
            )
        if cfg.intermediate_size != 3072:
            raise ValueError(
                f"expected Qwen3-0.6B intermediate_size=3072, got {cfg.intermediate_size}"
            )
        if not (1 <= self.num_layers <= cfg.num_hidden_layers):
            raise ValueError(
                f"num_layers must be in [1, {cfg.num_hidden_layers}], got {self.num_layers}"
            )

    def _validate_debug_outputs(self):
        unknown = set(self.debug_outputs) - set(DEBUG_STAGE_OUTPUTS["all"])
        if unknown:
            raise ValueError(f"Unknown debug output buffer(s): {sorted(unknown)}")
        if self.debug_outputs and self.num_layers != 1:
            raise ValueError("debug outputs currently support exactly one layer")

    def _build_fused_decode(self):
        cfg = self.config
        ctx = self.context
        hidden = cfg.hidden_size
        head_dim = cfg.head_dim
        q_size = cfg.num_attention_heads * cfg.head_dim
        kv_size = cfg.num_key_value_heads * cfg.head_dim
        ffn = cfg.intermediate_size

        gemv_q = GEMV(
            M=q_size,
            K=hidden,
            num_aie_columns=8,
            tile_size_input=4,
            tile_size_output=64,
            context=ctx,
        )
        gemv_kv = GEMV(
            M=kv_size,
            K=hidden,
            num_aie_columns=8,
            tile_size_input=4,
            tile_size_output=64,
            context=ctx,
        )
        hidden_norm = RMSNorm(
            size=hidden,
            num_aie_columns=1,
            num_channels=1,
            tile_size=hidden,
            weighted=True,
            epsilon=cfg.rms_norm_eps,
            context=ctx,
        )
        q_norm = RMSNorm(
            size=q_size,
            num_aie_columns=8,
            num_channels=1,
            tile_size=head_dim,
            weighted=True,
            epsilon=cfg.rms_norm_eps,
            context=ctx,
        )
        k_norm = RMSNorm(
            size=kv_size,
            num_aie_columns=8,
            num_channels=1,
            tile_size=head_dim,
            weighted=True,
            epsilon=cfg.rms_norm_eps,
            context=ctx,
        )
        rope_q = RoPE(
            rows=cfg.num_attention_heads,
            cols=head_dim,
            angle_rows=1,
            method_type=0,
            context=ctx,
        )
        rope_k = RoPE(
            rows=cfg.num_key_value_heads,
            cols=head_dim,
            angle_rows=1,
            method_type=0,
            context=ctx,
        )

        cache_patch_magic = 0xA13D0DEC
        cache_write = StridedCopy(
            input_sizes=(cfg.num_key_value_heads, head_dim),
            input_strides=(head_dim, 1),
            input_offset=0,
            output_sizes=(1, cfg.num_key_value_heads, head_dim),
            output_strides=(0, self.max_seq_len * head_dim, 1),
            output_offset=7 * head_dim * 2,
            input_buffer_size=kv_size,
            output_buffer_size=cfg.num_key_value_heads * self.max_seq_len * head_dim,
            num_aie_channels=1,
            kwargs={"output_offset_patch_marker": cache_patch_magic},
            context=ctx,
        )
        repeat_kv_op = Repeat(
            rows=cfg.num_key_value_heads,
            cols=self.max_seq_len * head_dim,
            repeat=cfg.num_attention_heads // cfg.num_key_value_heads,
            transfer_size=head_dim,
            context=ctx,
        )
        attn_scores = GEMV(
            M=self.max_seq_len,
            K=head_dim,
            num_aie_columns=8,
            tile_size_input=4,
            tile_size_output=self.max_seq_len // 8,
            num_batches=cfg.num_attention_heads,
            context=ctx,
        )
        attn_scale = ElementwiseMul(
            size=cfg.num_attention_heads * self.max_seq_len,
            tile_size=self.max_seq_len // 8,
            num_aie_columns=8,
            context=ctx,
        )
        softmax_magic = 0xBA5EBA11
        softmax = Softmax(
            rows=cfg.num_attention_heads,
            cols=self.max_seq_len,
            num_aie_columns=1,
            num_channels=1,
            rtp_vector_size=self.max_seq_len,
            mask_patch_value=softmax_magic,
            context=ctx,
        )
        transpose_values = Transpose(
            M=self.max_seq_len,
            N=head_dim,
            num_aie_columns=2,
            num_channels=1,
            m=256,
            n=32,
            s=8,
            context=ctx,
        )
        attn_context = GEMV(
            M=head_dim,
            K=self.max_seq_len,
            num_aie_columns=8,
            tile_size_input=4,
            tile_size_output=4,
            num_batches=cfg.num_attention_heads,
            context=ctx,
        )
        attn_output = GEMV(
            M=hidden,
            K=q_size,
            num_aie_columns=8,
            tile_size_input=4,
            tile_size_output=hidden // 8,
            context=ctx,
        )
        ffn_up_gate = GEMV(
            M=ffn,
            K=hidden,
            num_aie_columns=8,
            tile_size_input=4,
            tile_size_output=ffn // 8,
            context=ctx,
        )
        ffn_down = GEMV(
            M=hidden,
            K=ffn,
            num_aie_columns=8,
            tile_size_input=1,
            tile_size_output=hidden // 8,
            context=ctx,
        )
        silu = SiLU(
            size=ffn,
            num_aie_columns=8,
            tile_size=ffn // 8,
            context=ctx,
        )
        eltwise_mul = ElementwiseMul(
            size=ffn,
            tile_size=ffn // 8,
            num_aie_columns=8,
            context=ctx,
        )
        residual_add = ElementwiseAdd(
            size=hidden,
            tile_size=hidden // 8,
            num_aie_columns=8,
            context=ctx,
        )
        lm_head = GEMV(
            M=cfg.vocab_size,
            K=hidden,
            num_aie_columns=8,
            tile_size_input=4,
            tile_size_output=16,
            context=ctx,
        )

        kv_cache_bytes = cfg.num_key_value_heads * self.max_seq_len * head_dim * 2
        values_per_head_bytes = self.max_seq_len * head_dim * 2
        values_repeated_bytes = cfg.num_attention_heads * values_per_head_bytes

        debug_sizes = {
            "x": hidden,
            "x_norm": hidden,
            "queries_raw": q_size,
            "queries_norm": q_size,
            "queries": q_size,
            "keys_raw": kv_size,
            "keys_norm": kv_size,
            "keys": kv_size,
            "values": kv_size,
            "attn_scores": cfg.num_attention_heads * self.max_seq_len,
            "attn_weights": cfg.num_attention_heads * self.max_seq_len,
            "attn_context": q_size,
            "attn_out": hidden,
            "ffn_gate": ffn,
            "ffn_up": ffn,
            "ffn_hidden": ffn,
            "ffn_out": hidden,
            "logits": cfg.vocab_size,
        }
        debug_copy_ops = {}

        runlist = []

        def append_debug_copy(buffer_name: str):
            if buffer_name not in self.debug_outputs:
                return
            numel = debug_sizes[buffer_name]
            if numel not in debug_copy_ops:
                debug_copy_ops[numel] = StridedCopy(
                    input_sizes=(numel,),
                    input_strides=(1,),
                    input_offset=0,
                    output_sizes=(numel,),
                    output_strides=(1,),
                    output_offset=0,
                    input_buffer_size=numel,
                    output_buffer_size=numel,
                    num_aie_channels=1,
                    context=ctx,
                )
            runlist.append(
                (
                    debug_copy_ops[numel],
                    buffer_name,
                    self.debug_output_names[buffer_name],
                )
            )

        for layer_idx in range(self.num_layers):
            runlist.extend(
                [
                    (hidden_norm, "x", f"W_input_norm_{layer_idx}", "x_norm"),
                    (gemv_q, f"W_q_proj_{layer_idx}", "x_norm", "queries_raw"),
                    (gemv_kv, f"W_k_proj_{layer_idx}", "x_norm", "keys_raw"),
                    (gemv_kv, f"W_v_proj_{layer_idx}", "x_norm", "values"),
                    (q_norm, "queries_raw", f"W_q_norm_{layer_idx}", "queries_norm"),
                    (k_norm, "keys_raw", f"W_k_norm_{layer_idx}", "keys_norm"),
                    (rope_q, "queries_norm", "rope_angles", "queries"),
                    (rope_k, "keys_norm", "rope_angles", "keys"),
                    (cache_write, "keys", f"keys_cache_{layer_idx}"),
                    (cache_write, "values", f"values_cache_{layer_idx}"),
                    (repeat_kv_op, f"keys_cache_{layer_idx}", "attn_keys_repeated"),
                    (repeat_kv_op, f"values_cache_{layer_idx}", "attn_values_repeated"),
                    (attn_scores, "attn_keys_repeated", "queries", "attn_scores"),
                    (attn_scale, "attn_scores", "attn_scale_factor", "attn_scores"),
                    (softmax, "attn_scores", "attn_weights"),
                ]
            )
            append_debug_copy("x_norm")
            append_debug_copy("queries_raw")
            append_debug_copy("keys_raw")
            append_debug_copy("values")
            append_debug_copy("queries_norm")
            append_debug_copy("keys_norm")
            append_debug_copy("queries")
            append_debug_copy("keys")
            append_debug_copy("attn_scores")
            append_debug_copy("attn_weights")
            runlist.extend(
                [
                    (
                        transpose_values,
                        f"attn_values_repeated[{h * values_per_head_bytes}:{(h + 1) * values_per_head_bytes}]",
                        f"attn_values_transposed[{h * values_per_head_bytes}:{(h + 1) * values_per_head_bytes}]",
                    )
                    for h in range(cfg.num_attention_heads)
                ]
            )
            runlist.extend(
                [
                    (
                        attn_context,
                        "attn_values_transposed",
                        "attn_weights",
                        "attn_context",
                    ),
                    (attn_output, f"W_o_proj_{layer_idx}", "attn_context", "attn_out"),
                    (residual_add, "x", "attn_out", "x"),
                    (hidden_norm, "x", f"W_post_norm_{layer_idx}", "x_norm"),
                    (ffn_up_gate, f"W_gate_proj_{layer_idx}", "x_norm", "ffn_gate"),
                    (ffn_up_gate, f"W_up_proj_{layer_idx}", "x_norm", "ffn_up"),
                ]
            )
            append_debug_copy("attn_context")
            append_debug_copy("attn_out")
            append_debug_copy("ffn_gate")
            append_debug_copy("ffn_up")
            runlist.extend(
                [
                    (silu, "ffn_gate", "ffn_gate"),
                    (eltwise_mul, "ffn_gate", "ffn_up", "ffn_hidden"),
                    (ffn_down, f"W_down_proj_{layer_idx}", "ffn_hidden", "ffn_out"),
                    (residual_add, "x", "ffn_out", "x"),
                ]
            )
            append_debug_copy("ffn_hidden")
            append_debug_copy("ffn_out")

        runlist.extend(
            [
                (hidden_norm, "x", "W_final_norm", "x"),
                (lm_head, "W_lm_head", "x", "logits"),
            ]
        )
        append_debug_copy("x")
        append_debug_copy("logits")

        self.cache_patch_magic = cache_patch_magic
        self.softmax_magic = softmax_magic
        input_args = ["x", "rope_angles"]
        output_args = list(dict.fromkeys(["logits", *self.debug_output_names.values()]))

        self.fused_op = FusedMLIROperator(
            "qwen3_decode_megakernel",
            runlist,
            input_args=input_args,
            output_args=output_args,
            buffer_sizes={
                **{
                    f"keys_cache_{layer_idx}": kv_cache_bytes
                    for layer_idx in range(self.num_layers)
                },
                **{
                    f"values_cache_{layer_idx}": kv_cache_bytes
                    for layer_idx in range(self.num_layers)
                },
                "attn_values_repeated": values_repeated_bytes,
                "attn_values_transposed": values_repeated_bytes,
            },
            context=ctx,
        ).compile()

        self.fused_elf_data = load_elf(self.fused_op)
        self.cache_patch_sites = self._find_cache_patch_sites()
        self.cache_patch_locations = {
            site["loc"]: site["base"] for site in self.cache_patch_sites
        }
        self.softmax_patch_offsets = self._get_patch_locs(self.softmax_magic)
        self._validate_patch_sites()
        self.fused = None
        if self.create_runtime:
            self.fused = FusedFullELFCallable(
                self.fused_op, elf_data=self.fused_elf_data
            )

    def _get_patch_locs(self, magic: int):
        magic = magic & 0xFFFFFFFF
        return np.where(self.fused_elf_data == magic)[0]

    def _find_cache_patch_sites(self):
        patch_sites = []
        for layer_idx in range(self.num_layers):
            _, keys_offset, _ = self.fused_op.get_layout_for_buffer(
                f"keys_cache_{layer_idx}"
            )
            _, values_offset, _ = self.fused_op.get_layout_for_buffer(
                f"values_cache_{layer_idx}"
            )
            key_locs = self._get_patch_locs(keys_offset + self.cache_patch_magic * 2)
            value_locs = self._get_patch_locs(
                values_offset + self.cache_patch_magic * 2
            )
            for loc in key_locs:
                patch_sites.append(
                    {
                        "kind": "keys_cache",
                        "layer": layer_idx,
                        "loc": int(loc),
                        "base": keys_offset,
                    }
                )
            for loc in value_locs:
                patch_sites.append(
                    {
                        "kind": "values_cache",
                        "layer": layer_idx,
                        "loc": int(loc),
                        "base": values_offset,
                    }
                )

        # The shared StridedCopy design also emits two marker locations before
        # fusion adds the parent cache buffer offset. These are intentionally
        # tracked as zero-base cache sites, with an exact count check below, so
        # this does not silently patch arbitrary marker-like constants.
        for loc in self._get_patch_locs(self.cache_patch_magic * 2):
            patch_sites.append(
                {"kind": "cache_no_base", "layer": None, "loc": int(loc), "base": 0}
            )
        if not patch_sites:
            raise RuntimeError("No cache patch locations found in fused ELF")
        return patch_sites

    def _validate_patch_sites(self):
        by_kind = {}
        for site in self.cache_patch_sites:
            by_kind.setdefault(site["kind"], []).append(site)

        patch_locs = [site["loc"] for site in self.cache_patch_sites]
        duplicate_locs = sorted(
            loc for loc in set(patch_locs) if patch_locs.count(loc) > 1
        )
        if duplicate_locs:
            raise RuntimeError(f"Duplicate cache patch locations: {duplicate_locs}")

        expected_per_layer = 2
        expected_no_base = 2
        for layer_idx in range(self.num_layers):
            key_sites = [
                site
                for site in by_kind.get("keys_cache", [])
                if site["layer"] == layer_idx
            ]
            value_sites = [
                site
                for site in by_kind.get("values_cache", [])
                if site["layer"] == layer_idx
            ]
            if len(key_sites) != expected_per_layer:
                raise RuntimeError(
                    f"Expected {expected_per_layer} key cache patch sites for "
                    f"layer {layer_idx}, found {len(key_sites)}"
                )
            if len(value_sites) != expected_per_layer:
                raise RuntimeError(
                    f"Expected {expected_per_layer} value cache patch sites for "
                    f"layer {layer_idx}, found {len(value_sites)}"
                )

        no_base_sites = by_kind.get("cache_no_base", [])
        if len(no_base_sites) != expected_no_base:
            raise RuntimeError(
                f"Expected {expected_no_base} zero-base cache patch sites, "
                f"found {len(no_base_sites)}"
            )

        expected_softmax_sites = self.num_layers + 1
        if len(self.softmax_patch_offsets) != expected_softmax_sites:
            raise RuntimeError(
                f"Expected {expected_softmax_sites} softmax patch sites, "
                f"found {len(self.softmax_patch_offsets)}"
            )

    def dump_layout(self):
        for name, layout in sorted(self.fused_op.subbuffer_layout.items()):
            print(f"layout {name}: {layout}")
        for name, slice_info in sorted(self.fused_op.slice_info.items()):
            print(f"slice {name}: {slice_info}")

    def dump_patches(self):
        for site in sorted(self.cache_patch_sites, key=lambda item: item["loc"]):
            print(
                "cache_patch "
                f"loc={site['loc']} kind={site['kind']} layer={site['layer']} "
                f"base={site['base']}"
            )
        for loc in self.softmax_patch_offsets:
            print(f"softmax_patch loc={int(loc)}")

    def _load_static_buffers(self):
        cfg = self.config
        if self.fused is None:
            raise RuntimeError("Runtime callable was not created")
        fused = self.fused
        for layer_idx in range(self.num_layers):
            layer = f"model.layers.{layer_idx}"
            attn = f"{layer}.self_attn"
            mlp = f"{layer}.mlp"
            static_weights = {
                f"W_input_norm_{layer_idx}": self.model.w(
                    f"{layer}.input_layernorm.weight"
                ),
                f"W_q_proj_{layer_idx}": self.model.w(f"{attn}.q_proj.weight"),
                f"W_k_proj_{layer_idx}": self.model.w(f"{attn}.k_proj.weight"),
                f"W_v_proj_{layer_idx}": self.model.w(f"{attn}.v_proj.weight"),
                f"W_q_norm_{layer_idx}": self.model.w(f"{attn}.q_norm.weight"),
                f"W_k_norm_{layer_idx}": self.model.w(f"{attn}.k_norm.weight"),
                f"W_o_proj_{layer_idx}": self.model.w(f"{attn}.o_proj.weight"),
                f"W_post_norm_{layer_idx}": self.model.w(
                    f"{layer}.post_attention_layernorm.weight"
                ),
                f"W_gate_proj_{layer_idx}": self.model.w(f"{mlp}.gate_proj.weight"),
                f"W_up_proj_{layer_idx}": self.model.w(f"{mlp}.up_proj.weight"),
                f"W_down_proj_{layer_idx}": self.model.w(f"{mlp}.down_proj.weight"),
            }
            for buf_name, tensor in static_weights.items():
                fused.get_buffer(buf_name).torch_view()[:] = tensor.flatten()

        fused.get_buffer("attn_scale_factor").fill_(1.0 / math.sqrt(cfg.head_dim))
        fused.get_buffer("W_final_norm").torch_view()[:] = self.model.w(
            "model.norm.weight"
        ).flatten()
        fused.get_buffer("W_lm_head").torch_view()[:] = self.model.w(
            "model.embed_tokens.weight"
        ).flatten()
        fused.input_buffer.to("npu")
        fused.scratch_buffer.to("npu")
        fused.output_buffer.to("npu")

    def load_kv_cache(self, state: Qwen3DecodeState):
        if self.fused is None:
            raise RuntimeError("Runtime callable was not created")
        self._validate_decode_state(state)
        for layer_idx in range(self.num_layers):
            self.fused.get_buffer(f"keys_cache_{layer_idx}").torch_view()[:] = (
                state.keys[layer_idx].flatten()
            )
            self.fused.get_buffer(f"values_cache_{layer_idx}").torch_view()[:] = (
                state.values[layer_idx].flatten()
            )
        self.fused.scratch_buffer.to("npu")

    def _patch_for_position(self, position: int):
        if self.fused is None:
            raise RuntimeError("Runtime callable was not created")
        if not (0 <= position < self.max_seq_len):
            raise ValueError(
                f"decode position must be in [0, {self.max_seq_len}), got {position}"
            )
        context_len = position + 1
        byte_offset = position * self.config.head_dim * 2
        patches = {
            loc: (base + byte_offset, 0xFFFFFFFF)
            for loc, base in self.cache_patch_locations.items()
        }
        patches.update(
            {int(loc): (context_len, 0xFFFFFFFF) for loc in self.softmax_patch_offsets}
        )
        elf_data = self.fused_elf_data.copy()
        patch_elf(elf_data, patches)
        self.fused.reload_elf(elf_data)

    def _validate_decode_state(self, state: Qwen3DecodeState):
        cfg = self.config
        if not (0 <= state.position < self.max_seq_len):
            raise ValueError(
                f"state.position must be in [0, {self.max_seq_len}), got {state.position}"
            )
        expected_shape = (cfg.num_key_value_heads, self.max_seq_len, cfg.head_dim)
        if len(state.keys) < self.num_layers or len(state.values) < self.num_layers:
            raise ValueError(
                f"decode state has {len(state.keys)} key layers and "
                f"{len(state.values)} value layers, expected at least {self.num_layers}"
            )
        for layer_idx in range(self.num_layers):
            key = state.keys[layer_idx]
            value = state.values[layer_idx]
            if tuple(key.shape) != expected_shape:
                raise ValueError(
                    f"state.keys[{layer_idx}] shape {tuple(key.shape)} "
                    f"does not match {expected_shape}"
                )
            if tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"state.values[{layer_idx}] shape {tuple(value.shape)} "
                    f"does not match {expected_shape}"
                )
            if key.dtype != self.model.dtype or value.dtype != self.model.dtype:
                raise ValueError(
                    f"state layer {layer_idx} dtype mismatch: "
                    f"keys={key.dtype} values={value.dtype} expected={self.model.dtype}"
                )

    def run_decode(self, token_id: int, position: int) -> torch.Tensor:
        if self.fused is None:
            raise RuntimeError("Runtime callable was not created")
        self._patch_for_position(position)
        x = self.model.embed(torch.tensor([[token_id]], dtype=torch.long)).squeeze(0)
        self.fused.get_buffer("x").torch_view()[:] = x.flatten()
        self.fused.get_buffer("rope_angles").torch_view()[:] = rope_lut_for_position(
            self.config.head_dim, self.config.rope_theta, position
        )
        self.fused.input_buffer.to("npu")
        self.fused()
        return (
            self.fused.get_buffer("logits")
            .to_torch()
            .view(1, 1, self.config.vocab_size)
        )


def run_linter(model: Qwen3ForCausalLM, max_seq_len: int, num_layers: int):
    cfg = model.config
    checks = {
        "hidden_size": cfg.hidden_size == 1024,
        "head_dim": cfg.head_dim == 128,
        "q_heads": cfg.num_attention_heads == 16,
        "kv_heads": cfg.num_key_value_heads == 8,
        "intermediate_size": cfg.intermediate_size == 3072,
        "max_seq_len_tile": max_seq_len >= 256 and max_seq_len % 256 == 0,
        "num_layers": 1 <= num_layers <= cfg.num_hidden_layers,
        "lm_head_tile": (cfg.vocab_size // 8) % 16 == 0,
    }
    for name, ok in checks.items():
        print(f"lint {name}: {'ok' if ok else 'FAIL'}")
    if not all(checks.values()):
        raise SystemExit(2)


def assert_full_elf_runtime_available():
    import pyxrt

    missing = [name for name in ("elf", "ext") if not hasattr(pyxrt, name)]
    if missing:
        raise RuntimeError(
            "Full-ELF runtime is unavailable in this pyxrt build; missing "
            f"{', '.join(missing)}. Use --compile-only to validate MLIR/aiecc "
            "generation, or install an XRT/pyxrt build with full-ELF support."
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen3-0.6B fused decode megakernel")
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="HF repo id or local dir"
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=28)
    parser.add_argument(
        "--build-dir",
        default="build_qwen3_megakernel",
        help="Directory for generated MLIR/ELF artifacts",
    )
    parser.add_argument(
        "--clean-build",
        action="store_true",
        help="Remove --build-dir before compiling to avoid stale fused artifacts",
    )
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--lint-only", action="store_true")
    parser.add_argument("--verify-one-step", action="store_true")
    parser.add_argument(
        "--verify-repeat",
        type=int,
        default=1,
        help="Run the same one-step decode this many times after one compile",
    )
    parser.add_argument(
        "--debug-stage",
        choices=sorted(DEBUG_STAGE_OUTPUTS),
        default=None,
        help="Enable a predefined set of non-intrusive debug drains",
    )
    parser.add_argument(
        "--debug-outputs",
        default=None,
        help="Comma-separated extra buffer names to copy out for debug diff",
    )
    parser.add_argument("--dump-layout", action="store_true")
    parser.add_argument("--dump-patches", action="store_true")
    parser.add_argument("--raw-prompt", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = resolve_model_dir(args.model, args.revision)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    input_ids = encode_prompt(
        tokenizer,
        args.prompt,
        raw_prompt=args.raw_prompt,
        enable_thinking=args.enable_thinking,
    )
    model = Qwen3ForCausalLM(model_dir)
    run_linter(model, args.max_seq_len, args.num_layers)
    if args.lint_only:
        return
    if not args.compile_only:
        assert_full_elf_runtime_available()

    debug_outputs = parse_debug_outputs(args.debug_stage, args.debug_outputs)
    if args.clean_build:
        shutil.rmtree(args.build_dir, ignore_errors=True)
    start = time.perf_counter()
    megakernel = Qwen3DecodeMegakernel(
        model,
        max_seq_len=args.max_seq_len,
        num_layers=args.num_layers,
        build_dir=args.build_dir,
        create_runtime=not args.compile_only,
        debug_outputs=debug_outputs,
    )
    print(f"compile_and_load_s: {time.perf_counter() - start:.3f}")
    if args.dump_layout:
        megakernel.dump_layout()
    if args.dump_patches:
        megakernel.dump_patches()
    if args.compile_only:
        return

    ref = Qwen3CachedReference(model, args.max_seq_len, args.num_layers)
    prefill_logits, state = ref.prefill(input_ids)
    next_token = int(torch.argmax(prefill_logits[:, -1, :], dim=-1).item())

    ref_logits, _ = ref.decode(next_token, clone_decode_state(state))
    ref_next = int(torch.argmax(ref_logits[:, -1, :], dim=-1).item())
    ref_text = tokenizer.decode([ref_next], skip_special_tokens=True)
    debug_refs = {}
    if debug_outputs:
        if args.num_layers != 1:
            raise ValueError("debug diff currently supports --num-layers 1 only")
        debug_refs = one_layer_reference_tensors(
            model, next_token, clone_decode_state(state), args.max_seq_len
        )

    failed = False
    repeat_count = max(1, args.verify_repeat)
    for iteration in range(repeat_count):
        megakernel.load_kv_cache(clone_decode_state(state))
        start = time.perf_counter()
        npu_logits = megakernel.run_decode(next_token, state.position)
        npu_s = time.perf_counter() - start
        npu_next = int(torch.argmax(npu_logits[:, -1, :], dim=-1).item())
        npu_text = tokenizer.decode([npu_next], skip_special_tokens=True)

        diff = (npu_logits.to(torch.float32) - ref_logits.to(torch.float32)).abs()
        print(f"iteration: {iteration}")
        print(f"decode_s: {npu_s:.6f}")
        print(f"prompt_next_token: {next_token}")
        print(f"npu_next_token: {npu_next} text={npu_text!r}")
        print(f"ref_next_token: {ref_next} text={ref_text!r}")
        print(f"logits_max_abs: {float(diff.max()):.6f}")
        print(f"logits_mean_abs: {float(diff.mean()):.6f}")

        if debug_outputs:
            npu_debug = {}
            for name in debug_outputs:
                buffer_name = megakernel.debug_output_names[name]
                npu_debug[name] = megakernel.fused.get_buffer(buffer_name).to_torch()
                if name in debug_refs:
                    print_tensor_diff(name, npu_debug[name], debug_refs[name], "full")
            local_refs = local_reference_tensors(model, npu_debug, state.position)
            for name, local_ref in local_refs.items():
                if name in npu_debug:
                    print_tensor_diff(name, npu_debug[name], local_ref, "local")

        failed = failed or npu_next != ref_next

    if args.verify_one_step and failed:
        raise SystemExit(1)
    del megakernel
    gc.collect()


if __name__ == "__main__":
    main()
