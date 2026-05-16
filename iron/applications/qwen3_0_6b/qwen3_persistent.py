#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import gc
import math
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar

import torch
import torch.nn.functional as F
import aie.utils as aie_utils
from transformers import AutoTokenizer

repo_root = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(repo_root))

from iron.applications.qwen3_0_6b.qwen3_cpu import (  # noqa: E402
    DEFAULT_MODEL,
    DEFAULT_PROMPT,
    Qwen3ForCausalLM,
    apply_rope,
    encode_prompt,
    repeat_kv,
    resolve_model_dir,
    rms_norm,
)
from iron.applications.qwen3_0_6b.qwen3_decode_reference import (  # noqa: E402
    Qwen3CachedReference,
)
from iron.applications.qwen3_0_6b.qwen3_megakernel_debug import (  # noqa: E402
    one_layer_reference_tensors,
)
from iron.applications.qwen3_0_6b.qwen3_preflight import (  # noqa: E402
    run_persistent_artifact_preflight,
)
from iron.common import (  # noqa: E402
    AIERuntimeArgSpec,
    DesignGenerator,
    KernelObjectArtifact,
    MLIROperator,
    PythonGeneratedMLIRArtifact,
    SourceArtifact,
)
from iron.common.context import AIEContext  # noqa: E402
from iron.common.device_utils import get_kernel_dir  # noqa: E402
from iron.common.test_utils import verify_buffer  # noqa: E402
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor  # noqa: E402


def verification_tolerance(stage: str, name: str) -> tuple[float, float]:
    rope_stages = {
        "input-rmsnorm-qkv-rope-cache",
        "input-rmsnorm-qkv-rope-cache-scores-softmax",
    }
    if stage in rope_stages and name in {"queries", "keys"}:
        return 0.05, 0.5
    return 0.04, 1e-6


@dataclass
class Qwen3PersistentInputRMSNorm(MLIROperator):
    """Single-token persistent Qwen3 input RMSNorm stage."""

    hidden_size: int = 1024
    epsilon: float = 1e-6
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[dict[str, str]] = {
        **MLIROperator._name_aliases,
        "hidden_size": "h",
    }

    def __post_init__(self):
        if self.hidden_size != 1024:
            raise ValueError(
                f"Qwen3-0.6B persistent input RMSNorm expects hidden_size=1024, got {self.hidden_size}"
            )
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")
        MLIROperator.__init__(self, context=self.context)

    @property
    def _epsilon_tag(self):
        return f"eps_{self.epsilon:.0e}".replace("-", "m")

    @property
    def _kernel_object(self):
        return f"qwen3_persistent_rms_norm_{self._epsilon_tag}.o"

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "qwen3_persistent_design.py",
                "qwen3_persistent_input_rmsnorm",
                (
                    aie_utils.get_current_device(),
                    self.hidden_size,
                    0,
                ),
                {"kernel_object": self._kernel_object},
            ),
        )

    def get_kernel_artifacts(self):
        arch_dir = get_kernel_dir()
        return [
            KernelObjectArtifact(
                self._kernel_object,
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / arch_dir / "rms_norm.cc"
                    )
                ],
                extra_flags=[f"-DRMS_NORM_EPSILON={self.epsilon}f"],
            ),
            KernelObjectArtifact(
                "mul.o",
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / "generic" / "mul.cc"
                    )
                ],
            ),
        ]

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (self.hidden_size,)),
            AIERuntimeArgSpec("in", (self.hidden_size,)),
            AIERuntimeArgSpec("out", (self.hidden_size,)),
        ]


@dataclass
class Qwen3PersistentInputRMSNormQKV(MLIROperator):
    """Single-token persistent Qwen3 input RMSNorm + QKV projection stage."""

    hidden_size: int = 1024
    q_size: int = 2048
    kv_size: int = 1024
    num_aie_columns: int = 1
    tile_size_input: int = 4
    tile_size_output: int = 64
    epsilon: float = 1e-6
    kernel_vector_size: int = field(default=64, repr=False)
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[dict[str, str]] = {
        **MLIROperator._name_aliases,
        "hidden_size": "h",
        "q_size": "q",
        "kv_size": "kv",
        "num_aie_columns": "col",
        "tile_size_input": "tsi",
        "tile_size_output": "tso",
    }

    def __post_init__(self):
        if self.hidden_size != 1024:
            raise ValueError(
                f"Qwen3-0.6B persistent QKV expects hidden_size=1024, got {self.hidden_size}"
            )
        if self.q_size != 2048 or self.kv_size != 1024:
            raise ValueError(
                f"Qwen3-0.6B persistent QKV expects q_size=2048 and kv_size=1024, "
                f"got q_size={self.q_size} kv_size={self.kv_size}"
            )
        if self.hidden_size % self.kernel_vector_size != 0:
            raise ValueError("hidden_size must be a multiple of kernel_vector_size")
        if self.q_size % self.num_aie_columns != 0:
            raise ValueError("q_size must be divisible by num_aie_columns")
        if self.kv_size % self.num_aie_columns != 0:
            raise ValueError("kv_size must be divisible by num_aie_columns")
        if self.tile_size_output % self.tile_size_input != 0:
            raise ValueError("tile_size_output must be a multiple of tile_size_input")
        if self.epsilon <= 0:
            raise ValueError(f"epsilon must be positive, got {self.epsilon}")
        MLIROperator.__init__(self, context=self.context)

    @property
    def _epsilon_tag(self):
        return f"eps_{self.epsilon:.0e}".replace("-", "m")

    @property
    def _rms_kernel_object(self):
        return f"qwen3_persistent_rms_norm_{self._epsilon_tag}.o"

    @property
    def _gemv_kernel_object(self):
        return (
            f"qwen3_persistent_gemv_{self.hidden_size}k_{self.kernel_vector_size}vs.o"
        )

    @property
    def packed_weights_size(self):
        return (
            self.hidden_size
            + self.q_size * self.hidden_size
            + 2 * self.kv_size * self.hidden_size
        )

    @property
    def packed_outputs_size(self):
        return self.hidden_size + self.q_size + 2 * self.kv_size

    @property
    def q_output_base(self):
        return self.hidden_size

    @property
    def k_output_base(self):
        return self.q_output_base + self.q_size

    @property
    def v_output_base(self):
        return self.k_output_base + self.kv_size

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "qwen3_persistent_design.py",
                "qwen3_persistent_input_rmsnorm_qkv",
                (
                    aie_utils.get_current_device(),
                    self.hidden_size,
                    self.q_size,
                    self.kv_size,
                    self.num_aie_columns,
                    self.tile_size_input,
                    self.tile_size_output,
                    0,
                ),
                {
                    "rms_kernel_object": self._rms_kernel_object,
                    "gemv_kernel_object": self._gemv_kernel_object,
                },
            ),
        )

    def get_kernel_artifacts(self):
        arch_dir = get_kernel_dir()
        return [
            KernelObjectArtifact(
                self._rms_kernel_object,
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / arch_dir / "rms_norm.cc"
                    )
                ],
                extra_flags=[f"-DRMS_NORM_EPSILON={self.epsilon}f"],
            ),
            KernelObjectArtifact(
                "mul.o",
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / "generic" / "mul.cc"
                    )
                ],
            ),
            KernelObjectArtifact(
                self._gemv_kernel_object,
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / "generic" / "mv.cc"
                    )
                ],
                extra_flags=[
                    f"-DDIM_K={self.hidden_size}",
                    f"-DVEC_SIZE={self.kernel_vector_size}",
                ],
            ),
        ]

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (self.hidden_size,)),
            AIERuntimeArgSpec("in", (self.packed_weights_size,)),
            AIERuntimeArgSpec("out", (self.packed_outputs_size,)),
        ]


@dataclass
class Qwen3PersistentInputRMSNormQKVRopeCache(Qwen3PersistentInputRMSNormQKV):
    """Single-token persistent Qwen3 input RMSNorm + QKV + RoPE + KV write."""

    head_dim: int = 128
    max_seq_len: int = 256
    position: int = 0
    tile_size_output: int = 128

    _name_aliases: ClassVar[dict[str, str]] = {
        **Qwen3PersistentInputRMSNormQKV._name_aliases,
        "head_dim": "hd",
        "max_seq_len": "msl",
        "position": "pos",
    }

    def __post_init__(self):
        super().__post_init__()
        if self.head_dim != 128:
            raise ValueError(
                f"Qwen3-0.6B persistent RoPE cache expects head_dim=128, got {self.head_dim}"
            )
        if self.q_size % self.head_dim != 0 or self.kv_size % self.head_dim != 0:
            raise ValueError("q_size and kv_size must be divisible by head_dim")
        if self.tile_size_output != self.head_dim:
            raise ValueError(
                "tile_size_output must equal head_dim for RoPE cache stage"
            )
        if self.max_seq_len < 256:
            raise ValueError("max_seq_len must be at least 256")
        if not (0 <= self.position < self.max_seq_len):
            raise ValueError(
                f"position must be in [0, {self.max_seq_len}), got {self.position}"
            )

    @property
    def q_heads(self):
        return self.q_size // self.head_dim

    @property
    def kv_heads(self):
        return self.kv_size // self.head_dim

    @property
    def _rope_kernel_object(self):
        return "qwen3_persistent_rope_0.o"

    @property
    def packed_weights_size(self):
        return super().packed_weights_size + 2 * self.head_dim

    @property
    def packed_outputs_size(self):
        return (
            self.hidden_size
            + self.q_size
            + self.kv_size
            + self.q_size
            + self.kv_size
            + self.q_size
        )

    @property
    def q_norm_weight_base(self):
        return super().packed_weights_size

    @property
    def k_norm_weight_base(self):
        return self.q_norm_weight_base + self.head_dim

    @property
    def q_raw_output_base(self):
        return self.hidden_size

    @property
    def k_raw_output_base(self):
        return self.q_raw_output_base + self.q_size

    @property
    def q_norm_output_base(self):
        return self.k_raw_output_base + self.kv_size

    @property
    def k_norm_output_base(self):
        return self.q_norm_output_base + self.q_size

    @property
    def q_rope_output_base(self):
        return self.k_norm_output_base + self.kv_size

    @property
    def k_rope_output_base(self):
        return self.q_rope_output_base + self.q_size

    @property
    def cache_half_size(self):
        return self.kv_size * self.max_seq_len

    @property
    def packed_cache_size(self):
        return 2 * self.cache_half_size

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "qwen3_persistent_design.py",
                "qwen3_persistent_input_rmsnorm_qkv_rope_cache",
                (
                    aie_utils.get_current_device(),
                    self.hidden_size,
                    self.q_size,
                    self.kv_size,
                    self.head_dim,
                    self.max_seq_len,
                    self.position,
                    self.num_aie_columns,
                    self.tile_size_input,
                    self.tile_size_output,
                    0,
                ),
                {
                    "rms_kernel_object": self._rms_kernel_object,
                    "gemv_kernel_object": self._gemv_kernel_object,
                    "rope_kernel_object": self._rope_kernel_object,
                },
            ),
        )

    def get_kernel_artifacts(self):
        artifacts = super().get_kernel_artifacts()
        artifacts.append(
            KernelObjectArtifact(
                self._rope_kernel_object,
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / "generic" / "rope.cc"
                    )
                ],
                extra_flags=["-DTWO_HALVES"],
            )
        )
        return artifacts

    def get_arg_spec(self):
        return [
            AIERuntimeArgSpec("in", (self.hidden_size,)),
            AIERuntimeArgSpec("in", (self.packed_weights_size,)),
            AIERuntimeArgSpec("in", (self.head_dim,)),
            AIERuntimeArgSpec("out", (self.packed_outputs_size,)),
            AIERuntimeArgSpec("inout", (self.packed_cache_size,)),
        ]


@dataclass
class Qwen3PersistentInputRMSNormQKVRopeCacheScoresSoftmax(
    Qwen3PersistentInputRMSNormQKVRopeCache
):
    """Single-token persistent Qwen3 stage through attention scores and softmax."""

    def __post_init__(self):
        super().__post_init__()
        if self.num_aie_columns != 1:
            raise ValueError(
                "scores+softmax checkpoint is currently single-column only"
            )
        if self.max_seq_len % 64 != 0:
            raise ValueError(
                "softmax checkpoint requires max_seq_len to be a multiple of 64"
            )

    @property
    def _attention_kernel_object(self):
        return (
            f"qwen3_persistent_attention_scores_hd{self.head_dim}_"
            f"msl{self.max_seq_len}.o"
        )

    @property
    def _softmax_kernel_object(self):
        return "qwen3_persistent_softmax.o"

    @property
    def score_size(self):
        return self.q_heads * self.max_seq_len

    @property
    def packed_outputs_size(self):
        return super().packed_outputs_size + 2 * self.score_size

    @property
    def attn_scores_output_base(self):
        return super().packed_outputs_size

    @property
    def attn_weights_output_base(self):
        return self.attn_scores_output_base + self.score_size

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "qwen3_persistent_design.py",
                "qwen3_persistent_input_rmsnorm_qkv_rope_cache_scores_softmax",
                (
                    aie_utils.get_current_device(),
                    self.hidden_size,
                    self.q_size,
                    self.kv_size,
                    self.head_dim,
                    self.max_seq_len,
                    self.position,
                    self.num_aie_columns,
                    self.tile_size_input,
                    self.tile_size_output,
                    0,
                ),
                {
                    "rms_kernel_object": self._rms_kernel_object,
                    "gemv_kernel_object": self._gemv_kernel_object,
                    "rope_kernel_object": self._rope_kernel_object,
                    "attention_kernel_object": self._attention_kernel_object,
                    "softmax_kernel_object": self._softmax_kernel_object,
                },
            ),
        )

    def get_kernel_artifacts(self):
        arch_dir = get_kernel_dir()
        if arch_dir != "aie2p":
            raise ValueError(
                "scores+softmax persistent checkpoint currently targets NPU2/aie2p"
            )
        artifacts = super().get_kernel_artifacts()
        artifacts.extend(
            [
                KernelObjectArtifact(
                    self._attention_kernel_object,
                    dependencies=[
                        SourceArtifact(
                            self.context.base_dir
                            / "aie_kernels"
                            / "generic"
                            / "qwen3_attention.cc"
                        )
                    ],
                    extra_flags=[
                        f"-DHEAD_DIM={self.head_dim}",
                        f"-DMAX_SEQ_LEN={self.max_seq_len}",
                        "-DCACHE_BLOCK=64",
                        f"-DATTN_SCALE={1.0 / math.sqrt(self.head_dim)}f",
                    ],
                ),
                KernelObjectArtifact(
                    self._softmax_kernel_object,
                    dependencies=[
                        SourceArtifact(
                            self.context.base_dir
                            / "aie_kernels"
                            / arch_dir
                            / "softmax.cc"
                        )
                    ],
                ),
            ]
        )
        return artifacts


def assert_standard_runtime_available():
    import pyxrt  # noqa: F401


def build_reference_input(
    model: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    max_seq_len: int,
) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
    ref = Qwen3CachedReference(model, max_seq_len, num_layers=1)
    prefill_logits, _ = ref.prefill(input_ids)
    next_token = int(torch.argmax(prefill_logits[:, -1, :], dim=-1).item())
    hidden = model.embed(torch.tensor([[next_token]], dtype=torch.long)).flatten()
    weight = model.w("model.layers.0.input_layernorm.weight").flatten()
    expected = rms_norm(
        hidden.view(1, 1, -1),
        weight,
        model.config.rms_norm_eps,
    ).flatten()
    return next_token, hidden.contiguous(), weight.contiguous(), expected.contiguous()


def build_reference_qkv(
    model: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    max_seq_len: int,
):
    ref = Qwen3CachedReference(model, max_seq_len, num_layers=1)
    prefill_logits, state = ref.prefill(input_ids)
    next_token = int(torch.argmax(prefill_logits[:, -1, :], dim=-1).item())
    hidden = model.embed(torch.tensor([[next_token]], dtype=torch.long)).flatten()
    references = one_layer_reference_tensors(model, next_token, state, max_seq_len)
    attn = "model.layers.0.self_attn"
    return (
        next_token,
        {
            "hidden": hidden.contiguous(),
            "input_norm_weight": model.w("model.layers.0.input_layernorm.weight")
            .flatten()
            .contiguous(),
            "W_q": model.w(f"{attn}.q_proj.weight").contiguous(),
            "W_k": model.w(f"{attn}.k_proj.weight").contiguous(),
            "W_v": model.w(f"{attn}.v_proj.weight").contiguous(),
        },
        {
            "x_norm": references["x_norm"].flatten().contiguous(),
            "queries_raw": references["queries_raw"].flatten().contiguous(),
            "keys_raw": references["keys_raw"].flatten().contiguous(),
            "values": references["values"].flatten().contiguous(),
        },
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


def build_reference_qkv_rope_cache(
    model: Qwen3ForCausalLM,
    input_ids: torch.Tensor,
    max_seq_len: int,
):
    ref = Qwen3CachedReference(model, max_seq_len, num_layers=1)
    prefill_logits, state = ref.prefill(input_ids)
    next_token = int(torch.argmax(prefill_logits[:, -1, :], dim=-1).item())
    hidden = model.embed(torch.tensor([[next_token]], dtype=torch.long)).flatten()
    references = one_layer_reference_tensors(model, next_token, state, max_seq_len)
    attn = "model.layers.0.self_attn"
    initial_cache = torch.cat(
        [state.keys[0].flatten(), state.values[0].flatten()]
    ).contiguous()
    return (
        next_token,
        state.position,
        {
            "hidden": hidden.contiguous(),
            "input_norm_weight": model.w("model.layers.0.input_layernorm.weight")
            .flatten()
            .contiguous(),
            "W_q": model.w(f"{attn}.q_proj.weight").contiguous(),
            "W_k": model.w(f"{attn}.k_proj.weight").contiguous(),
            "W_v": model.w(f"{attn}.v_proj.weight").contiguous(),
            "W_q_norm": model.w(f"{attn}.q_norm.weight").flatten().contiguous(),
            "W_k_norm": model.w(f"{attn}.k_norm.weight").flatten().contiguous(),
            "rope_angles": rope_lut_for_position(
                model.config.head_dim, model.config.rope_theta, state.position
            ),
            "initial_cache": initial_cache,
            "initial_keys_cache": state.keys[0].contiguous(),
            "initial_values_cache": state.values[0].contiguous(),
        },
        {
            "x_norm": references["x_norm"].flatten().contiguous(),
            "queries_raw": references["queries_raw"].flatten().contiguous(),
            "keys_raw": references["keys_raw"].flatten().contiguous(),
            "values": references["values"].flatten().contiguous(),
            "queries_norm": references["queries_norm"].flatten().contiguous(),
            "keys_norm": references["keys_norm"].flatten().contiguous(),
            "queries": references["queries"].flatten().contiguous(),
            "keys": references["keys"].flatten().contiguous(),
        },
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Qwen3-0.6B persistent decode megakernel bring-up"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL, help="HF repo id or local dir"
    )
    parser.add_argument("--revision", default=None)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--raw-prompt", action="store_true")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--max-seq-len", type=int, default=256)
    parser.add_argument("--build-dir", default="build_qwen3_persistent")
    parser.add_argument("--clean-build", action="store_true")
    parser.add_argument(
        "--stage",
        choices=[
            "input-rmsnorm",
            "input-rmsnorm-qkv",
            "input-rmsnorm-qkv-rope-cache",
            "input-rmsnorm-qkv-rope-cache-scores-softmax",
        ],
        default="input-rmsnorm",
        help="Persistent bring-up stage to compile/run",
    )
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--verify-repeat", type=int, default=1)
    parser.add_argument("--dump-proof", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.clean_build:
        shutil.rmtree(args.build_dir, ignore_errors=True)

    model_dir = resolve_model_dir(args.model, args.revision)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    input_ids = encode_prompt(
        tokenizer,
        args.prompt,
        raw_prompt=args.raw_prompt,
        enable_thinking=args.enable_thinking,
    )
    model = Qwen3ForCausalLM(model_dir)
    if model.config.hidden_size != 1024:
        raise ValueError(
            f"expected Qwen3-0.6B hidden_size=1024, got {model.config.hidden_size}"
        )
    if args.max_seq_len < 256:
        raise ValueError("max_seq_len must be at least 256")
    if not args.compile_only:
        assert_standard_runtime_available()

    context = AIEContext(build_dir=args.build_dir)
    if args.stage == "input-rmsnorm":
        op = Qwen3PersistentInputRMSNorm(
            hidden_size=model.config.hidden_size,
            epsilon=model.config.rms_norm_eps,
            context=context,
        )
    elif args.stage == "input-rmsnorm-qkv":
        op = Qwen3PersistentInputRMSNormQKV(
            hidden_size=model.config.hidden_size,
            q_size=model.config.num_attention_heads * model.config.head_dim,
            kv_size=model.config.num_key_value_heads * model.config.head_dim,
            epsilon=model.config.rms_norm_eps,
            context=context,
        )
    elif args.stage == "input-rmsnorm-qkv-rope-cache":
        op = Qwen3PersistentInputRMSNormQKVRopeCache(
            hidden_size=model.config.hidden_size,
            q_size=model.config.num_attention_heads * model.config.head_dim,
            kv_size=model.config.num_key_value_heads * model.config.head_dim,
            head_dim=model.config.head_dim,
            max_seq_len=args.max_seq_len,
            position=input_ids.shape[1],
            epsilon=model.config.rms_norm_eps,
            context=context,
        )
    else:
        op = Qwen3PersistentInputRMSNormQKVRopeCacheScoresSoftmax(
            hidden_size=model.config.hidden_size,
            q_size=model.config.num_attention_heads * model.config.head_dim,
            kv_size=model.config.num_key_value_heads * model.config.head_dim,
            head_dim=model.config.head_dim,
            max_seq_len=args.max_seq_len,
            position=input_ids.shape[1],
            epsilon=model.config.rms_norm_eps,
            context=context,
        )

    start = time.perf_counter()
    op.compile()
    compile_s = time.perf_counter() - start
    preflight = run_persistent_artifact_preflight(
        mlir_path=Path(op.xclbin_artifact.mlir_input.filename),
        arg_specs=len(op.get_arg_spec()),
    )
    print(f"stage: {args.stage}")
    print("implementation: hand-authored IRON Program/Worker/ObjectFifo")
    print(f"operator_name: {op.name}")
    print(f"compile_s: {compile_s:.3f}")
    print(
        "preflight: ok "
        f"runtime_memrefs={preflight.runtime_memrefs} "
        f"arg_specs={preflight.arg_specs} "
        f"metadata_host_bos={preflight.metadata_host_bos} "
        f"max_fifo_buffered_bytes={preflight.max_fifo_buffered_bytes} "
        f"max_dma_tasks_per_fifo={preflight.max_dma_tasks_per_fifo} "
        f"max_tile_inputs={preflight.max_compute_tile_inputs} "
        f"max_tile_outputs={preflight.max_compute_tile_outputs}"
    )
    if args.dump_proof:
        for artifact in op.artifacts:
            print(f"artifact: {artifact.filename}")
        if args.stage == "input-rmsnorm":
            print("dispatch_shape: hidden[1024] + norm_weight[1024] -> x_norm[1024]")
            print(
                "worker_graph: hidden_fifo -> rmsnorm_worker -> weight_worker -> output_fifo"
            )
        elif args.stage == "input-rmsnorm-qkv":
            print(
                "dispatch_shape: hidden[1024] + norm_weight[1024] + "
                "Wq[2048,1024] + Wk[1024,1024] + Wv[1024,1024] -> "
                "x_norm[1024], queries_raw[2048], keys_raw[1024], values[1024]"
            )
            print(
                "runtime_bos: hidden[1024], "
                f"packed_weights[{op.packed_weights_size}], "
                f"packed_outputs[{op.packed_outputs_size}]"
            )
            print(f"qkv_columns: {op.num_aie_columns}")
            print(
                "worker_graph: hidden_fifo -> rmsnorm_worker -> weight_worker -> "
                "single xnorm broadcast FIFO -> Q/K/V matvec workers"
            )
        elif args.stage == "input-rmsnorm-qkv-rope-cache":
            print(
                "dispatch_shape: hidden[1024] + packed QKV/norm weights + "
                "rope_angles[128] -> x_norm, Q/K/V raw, Q/K norm, Q/K rope, KV cache"
            )
            print(
                "runtime_bos: hidden[1024], "
                f"packed_weights[{op.packed_weights_size}], "
                "rope_angles[128], "
                f"packed_outputs[{op.packed_outputs_size}], "
                f"packed_cache[{op.packed_cache_size}]"
            )
            print(f"decode_position: {op.position}")
            print(f"qkv_columns: {op.num_aie_columns}")
            print(
                "worker_graph: hidden_fifo -> rmsnorm_worker -> weight_worker -> "
                "xnorm broadcast -> Q/K/V matvec -> Q/K norm -> Q/K RoPE -> KV cache drains"
            )
        else:
            print(
                "dispatch_shape: hidden[1024] + packed QKV/norm weights + "
                "rope_angles[128] + K cache -> x_norm, Q/K/V raw, Q/K norm, "
                "RoPE Q, attention scores, attention weights, KV cache"
            )
            print(
                "runtime_bos: hidden[1024], "
                f"packed_weights[{op.packed_weights_size}], "
                "rope_angles[128], "
                f"packed_outputs[{op.packed_outputs_size}], "
                f"packed_cache[{op.packed_cache_size}]"
            )
            print(f"decode_position: {op.position}")
            print(f"qkv_columns: {op.num_aie_columns}")
            print(
                "worker_graph: hidden_fifo -> rmsnorm_worker -> weight_worker -> "
                "xnorm broadcast -> Q/K/V matvec -> Q/K norm -> Q/K RoPE -> "
                "score worker -> softmax worker, with KV cache drains"
            )
    if args.compile_only:
        return

    op_func = op.get_callable()
    if args.stage == "input-rmsnorm":
        next_token, hidden, weight, expected = build_reference_input(
            model, input_ids, args.max_seq_len
        )
        hidden_buf = XRTTensor.from_torch(hidden)
        weight_buf = XRTTensor.from_torch(weight)
        output_buf = XRTTensor((model.config.hidden_size,), dtype=hidden_buf.dtype)
        op_args = [hidden_buf, weight_buf, output_buf]
        output_buffers = {"input_rmsnorm": output_buf}
        expected_buffers = {"input_rmsnorm": expected}
        full_expected_buffers = expected_buffers
    elif args.stage == "input-rmsnorm-qkv":
        next_token, inputs, expected_buffers = build_reference_qkv(
            model, input_ids, args.max_seq_len
        )
        full_expected_buffers = expected_buffers
        hidden_buf = XRTTensor.from_torch(inputs["hidden"])
        packed_weights = torch.cat(
            [
                inputs["input_norm_weight"].flatten(),
                inputs["W_q"].flatten(),
                inputs["W_k"].flatten(),
                inputs["W_v"].flatten(),
            ]
        ).contiguous()
        weights_buf = XRTTensor.from_torch(packed_weights)
        packed_outputs_buf = XRTTensor(
            (op.packed_outputs_size,),
            dtype=hidden_buf.dtype,
        )
        op_args = [hidden_buf, weights_buf, packed_outputs_buf]
        output_buffers = {"packed_outputs": packed_outputs_buf}
    else:
        next_token, position, inputs, expected_buffers = build_reference_qkv_rope_cache(
            model, input_ids, args.max_seq_len
        )
        if position != op.position:
            raise RuntimeError(
                f"compiled position {op.position} != reference {position}"
            )
        full_expected_buffers = expected_buffers
        hidden_buf = XRTTensor.from_torch(inputs["hidden"])
        packed_weights = torch.cat(
            [
                inputs["input_norm_weight"].flatten(),
                inputs["W_q"].flatten(),
                inputs["W_k"].flatten(),
                inputs["W_v"].flatten(),
                inputs["W_q_norm"].flatten(),
                inputs["W_k_norm"].flatten(),
            ]
        ).contiguous()
        weights_buf = XRTTensor.from_torch(packed_weights)
        rope_angles_buf = XRTTensor.from_torch(inputs["rope_angles"])
        packed_outputs_buf = XRTTensor(
            (op.packed_outputs_size,),
            dtype=hidden_buf.dtype,
        )
        packed_cache_buf = XRTTensor.from_torch(inputs["initial_cache"].clone())
        op_args = [
            hidden_buf,
            weights_buf,
            rope_angles_buf,
            packed_outputs_buf,
            packed_cache_buf,
        ]
        output_buffers = {
            "packed_outputs": packed_outputs_buf,
            "packed_cache": packed_cache_buf,
        }

    failed = False
    repeat_count = max(1, args.verify_repeat)
    for iteration in range(repeat_count):
        result = op_func(*op_args)
        print(f"iteration: {iteration}")
        print(f"prompt_next_token: {next_token}")
        print(f"npu_time_us: {result.npu_time / 1e3:.3f}")
        if args.stage == "input-rmsnorm":
            actual_buffers = {}
            for name, buffer in output_buffers.items():
                buffer.device = "npu"
                actual_buffers[name] = buffer.to_torch()
            local_expected_buffers = expected_buffers
        elif args.stage == "input-rmsnorm-qkv":
            packed_outputs_buf.device = "npu"
            packed_outputs = packed_outputs_buf.to_torch()
            actual_buffers = {
                "x_norm": packed_outputs[: op.q_output_base],
                "queries_raw": packed_outputs[op.q_output_base : op.k_output_base],
                "keys_raw": packed_outputs[op.k_output_base : op.v_output_base],
                "values": packed_outputs[op.v_output_base :],
            }
            local_expected_buffers = {
                **full_expected_buffers,
                "queries_raw": F.linear(
                    actual_buffers["x_norm"].view(1, 1, -1), inputs["W_q"]
                )
                .flatten()
                .contiguous(),
                "keys_raw": F.linear(
                    actual_buffers["x_norm"].view(1, 1, -1), inputs["W_k"]
                )
                .flatten()
                .contiguous(),
                "values": F.linear(
                    actual_buffers["x_norm"].view(1, 1, -1), inputs["W_v"]
                )
                .flatten()
                .contiguous(),
            }
        else:
            packed_outputs_buf.device = "npu"
            packed_cache_buf.device = "npu"
            packed_outputs = packed_outputs_buf.to_torch()
            packed_cache = packed_cache_buf.to_torch()
            has_scores_softmax = (
                args.stage == "input-rmsnorm-qkv-rope-cache-scores-softmax"
            )
            q_rope_end = (
                op.attn_scores_output_base
                if has_scores_softmax
                else op.packed_outputs_size
            )
            keys_cache = packed_cache[: op.cache_half_size].view(
                op.kv_heads, args.max_seq_len, op.head_dim
            )
            values_cache = packed_cache[op.cache_half_size :].view(
                op.kv_heads, args.max_seq_len, op.head_dim
            )
            keys_cache_current = keys_cache[:, op.position, :].flatten()
            values_cache_current = values_cache[:, op.position, :].flatten()
            actual_buffers = {
                "x_norm": packed_outputs[: op.q_raw_output_base],
                "queries_raw": packed_outputs[
                    op.q_raw_output_base : op.k_raw_output_base
                ],
                "keys_raw": packed_outputs[
                    op.k_raw_output_base : op.q_norm_output_base
                ],
                "queries_norm": packed_outputs[
                    op.q_norm_output_base : op.k_norm_output_base
                ],
                "keys_norm": packed_outputs[
                    op.k_norm_output_base : op.q_rope_output_base
                ],
                "queries": packed_outputs[op.q_rope_output_base : q_rope_end],
                "keys": keys_cache_current,
                "values": values_cache_current,
                "keys_cache_current": keys_cache_current,
                "values_cache_current": values_cache_current,
                "keys_cache_prefix": keys_cache[:, : op.position, :].flatten(),
                "values_cache_prefix": values_cache[:, : op.position, :].flatten(),
            }
            if has_scores_softmax:
                actual_buffers["attn_scores"] = packed_outputs[
                    op.attn_scores_output_base : op.attn_weights_output_base
                ]
                actual_buffers["attn_weights"] = packed_outputs[
                    op.attn_weights_output_base :
                ]
            q_raw_local = F.linear(
                actual_buffers["x_norm"].view(1, 1, -1), inputs["W_q"]
            ).flatten()
            k_raw_local = F.linear(
                actual_buffers["x_norm"].view(1, 1, -1), inputs["W_k"]
            ).flatten()
            values_local = F.linear(
                actual_buffers["x_norm"].view(1, 1, -1), inputs["W_v"]
            ).flatten()
            q_norm_local = rms_norm(
                actual_buffers["queries_raw"].view(1, op.q_heads, 1, op.head_dim),
                inputs["W_q_norm"],
                model.config.rms_norm_eps,
            ).flatten()
            k_norm_local = rms_norm(
                actual_buffers["keys_raw"].view(1, op.kv_heads, 1, op.head_dim),
                inputs["W_k_norm"],
                model.config.rms_norm_eps,
            ).flatten()
            q_rope_local, k_rope_local = apply_rope(
                actual_buffers["queries_norm"].view(1, op.q_heads, 1, op.head_dim),
                actual_buffers["keys_norm"].view(1, op.kv_heads, 1, op.head_dim),
                torch.tensor([op.position]),
                op.head_dim,
                model.config.rope_theta,
            )
            score_expected_buffers = {}
            if has_scores_softmax:
                q_for_score = actual_buffers["queries"].view(
                    1, op.q_heads, 1, op.head_dim
                )
                k_ctx = repeat_kv(
                    keys_cache[:, : op.position + 1, :].unsqueeze(0),
                    op.q_heads // op.kv_heads,
                )
                scores = torch.matmul(
                    q_for_score.to(torch.float32),
                    k_ctx.to(torch.float32).transpose(-2, -1),
                ) / math.sqrt(op.head_dim)
                weights = torch.softmax(scores, dim=-1).to(dtype=packed_outputs.dtype)
                padded_scores = torch.zeros(
                    (op.q_heads, op.max_seq_len),
                    dtype=packed_outputs.dtype,
                )
                padded_weights = torch.zeros_like(padded_scores)
                padded_scores[:, : op.position + 1] = scores.view(
                    op.q_heads, op.position + 1
                ).to(dtype=packed_outputs.dtype)
                padded_weights[:, : op.position + 1] = weights.view(
                    op.q_heads, op.position + 1
                )
                score_expected_buffers = {
                    "attn_scores": padded_scores.flatten().contiguous(),
                    "attn_weights": padded_weights.flatten().contiguous(),
                }
            local_expected_buffers = {
                **full_expected_buffers,
                "queries_raw": q_raw_local.contiguous(),
                "keys_raw": k_raw_local.contiguous(),
                "values": values_local.contiguous(),
                "queries_norm": q_norm_local.contiguous(),
                "keys_norm": k_norm_local.contiguous(),
                "queries": q_rope_local.flatten().contiguous(),
                "keys": k_rope_local.flatten().contiguous(),
                "keys_cache_current": actual_buffers["keys"].contiguous(),
                "values_cache_current": actual_buffers["values"].contiguous(),
                "keys_cache_prefix": inputs["initial_keys_cache"][:, : op.position, :]
                .flatten()
                .contiguous(),
                "values_cache_prefix": inputs["initial_values_cache"][
                    :, : op.position, :
                ]
                .flatten()
                .contiguous(),
                **score_expected_buffers,
            }
        for name, output in actual_buffers.items():
            expected = local_expected_buffers[name]
            rel_tol, abs_tol = verification_tolerance(args.stage, name)
            errors = verify_buffer(
                output,
                name,
                expected,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
            diff = (output.to(torch.float32) - expected.to(torch.float32)).abs()
            print(f"{name}_max_abs: {float(diff.max()):.6f}")
            print(f"{name}_mean_abs: {float(diff.mean()):.6f}")
            if (
                args.stage
                in {
                    "input-rmsnorm-qkv",
                    "input-rmsnorm-qkv-rope-cache",
                    "input-rmsnorm-qkv-rope-cache-scores-softmax",
                }
                and name in full_expected_buffers
                and name != "x_norm"
            ):
                full_ref = full_expected_buffers[name]
                full_diff = (
                    output.to(torch.float32) - full_ref.to(torch.float32)
                ).abs()
                print(f"{name}_full_ref_max_abs: {float(full_diff.max()):.6f}")
                print(f"{name}_full_ref_mean_abs: {float(full_diff.mean()):.6f}")
            print(f"{name}_errors: {len(errors)}")
            failed = failed or bool(errors)

    if args.verify and failed:
        raise SystemExit(1)
    gc.collect()


if __name__ == "__main__":
    main()
