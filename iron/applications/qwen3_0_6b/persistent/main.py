#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import gc
import shutil
import sys
import time
from pathlib import Path

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
from iron.applications.qwen3_0_6b.persistent.diagnostics import (  # noqa: E402
    run_qkv_diagnostic_bundle,
)
from iron.applications.qwen3_0_6b.persistent.layout import (  # noqa: E402
    default_packed_weights_dir,
    validate_packed_weight_artifact,
    write_packed_weight_artifact,
)
from iron.applications.qwen3_0_6b.persistent.generate_runner import (  # noqa: E402
    make_single_layer_final_only_op_for_position,
    run_generate,
)
from iron.applications.qwen3_0_6b.persistent.ops_core import (  # noqa: E402
    Qwen3PersistentInputRMSNorm,
    Qwen3PersistentInputRMSNormQKV,
)
from iron.applications.qwen3_0_6b.persistent.ops_mlp import (  # noqa: E402
    Qwen3PersistentPostAttnMLPDownResidual,
    Qwen3PersistentPostAttnRMSNormFullMLP,
    Qwen3PersistentPostAttnRMSNormMLPGateUp,
)
from iron.applications.qwen3_0_6b.persistent.ops_nlayer import (  # noqa: E402
    Qwen3PersistentNLayerFinalOnly,
)
from iron.applications.qwen3_0_6b.persistent.stage_runner import (  # noqa: E402
    print_stage_proof,
    run_compiled_stage,
)
from iron.applications.qwen3_0_6b.persistent.stages import (  # noqa: E402
    GENERATE_STAGE,
    N_LAYER_FINAL_ONLY_STAGE,
    STAGE_CHOICES,
)
from iron.applications.qwen3_0_6b.qwen3_preflight import (  # noqa: E402
    run_persistent_artifact_preflight,
)
from iron.common.context import AIEContext  # noqa: E402


def assert_standard_runtime_available():
    import pyxrt  # noqa: F401


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
        "--prepare-weights",
        action="store_true",
        help=(
            "Prepack HF Qwen3 weights into a persistent bf16 artifact and exit. "
            "The default output is <model_dir>/qwen3_iron_packed."
        ),
    )
    parser.add_argument(
        "--packed-weights-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing weights.bf16.bin and manifest.json, or the "
            "output directory for --prepare-weights."
        ),
    )
    parser.add_argument(
        "--require-packed-weights",
        action="store_true",
        help="Require --fast-generate to load prepacked weights instead of runtime packing.",
    )
    parser.add_argument(
        "--stage",
        choices=STAGE_CHOICES,
        default="input-rmsnorm",
        help="Persistent bring-up stage to compile/run",
    )
    parser.add_argument("--compile-only", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--verify-generate",
        action="store_true",
        help="Compare generated tokens against the cached CPU decode reference",
    )
    parser.add_argument(
        "--fast-generate",
        action="store_true",
        help=(
            "Use the n-layer final-only generate path with cached packed "
            "weight/cache XRT buffers."
        ),
    )
    parser.add_argument("--verify-repeat", type=int, default=1)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2,
        help="Number of new tokens to produce for --stage generate",
    )
    parser.add_argument(
        "--layer-chunk-size",
        type=int,
        default=1,
        help="Number of full layers per persistent fast-generate invocation",
    )
    parser.add_argument(
        "--qkv-diagnostic-bundle",
        type=Path,
        default=None,
        help="Run an isolated QKV diagnostic bundle",
    )
    parser.add_argument("--dump-proof", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.clean_build:
        shutil.rmtree(args.build_dir, ignore_errors=True)

    model_dir = resolve_model_dir(args.model, args.revision)
    model = Qwen3ForCausalLM(model_dir)
    if model.config.hidden_size != 1024:
        raise ValueError(
            f"expected Qwen3-0.6B hidden_size=1024, got {model.config.hidden_size}"
        )
    if model.config.intermediate_size != 3072:
        raise ValueError(
            "expected Qwen3-0.6B intermediate_size=3072, "
            f"got {model.config.intermediate_size}"
        )
    if args.max_seq_len < 256:
        raise ValueError("max_seq_len must be at least 256")
    if args.packed_weights_dir is None:
        args.packed_weights_dir = default_packed_weights_dir(model_dir)

    if args.prepare_weights:
        op_for_layout = make_single_layer_final_only_op_for_position(
            args,
            model,
            AIEContext(build_dir=args.build_dir),
            position=0,
        )
        start = time.perf_counter()
        manifest = write_packed_weight_artifact(
            model,
            args.packed_weights_dir,
            expected_per_layer_numel=op_for_layout.packed_weights_size,
        )
        elapsed = time.perf_counter() - start
        print("stage: prepare-weights")
        print(f"packed_weights_dir: {args.packed_weights_dir}")
        print(f"packed_weights_file: {args.packed_weights_dir / 'weights.bf16.bin'}")
        print(f"packed_manifest_file: {args.packed_weights_dir / 'manifest.json'}")
        print(f"packed_layers: {manifest['num_layers']}")
        print(f"packed_per_layer_numel: {manifest['per_layer_numel']}")
        print(f"packed_total_bytes: {manifest['total_bytes']}")
        print(f"prepare_weights_s: {elapsed:.6f}")
        return

    if args.require_packed_weights:
        validate_packed_weight_artifact(
            model,
            args.packed_weights_dir,
            expected_per_layer_numel=make_single_layer_final_only_op_for_position(
                args,
                model,
                AIEContext(build_dir=args.build_dir),
                position=0,
            ).packed_weights_size,
        )
        if not args.fast_generate:
            raise ValueError("--require-packed-weights is only used by --fast-generate")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    input_ids = encode_prompt(
        tokenizer,
        args.prompt,
        raw_prompt=args.raw_prompt,
        enable_thinking=args.enable_thinking,
    )
    if not args.compile_only:
        assert_standard_runtime_available()

    context = AIEContext(build_dir=args.build_dir)
    if args.qkv_diagnostic_bundle is not None:
        failed = run_qkv_diagnostic_bundle(args.qkv_diagnostic_bundle, model, context)
        if failed:
            raise SystemExit(1)
        return
    if args.stage == GENERATE_STAGE:
        failed = run_generate(args, model, tokenizer, input_ids, context)
        if args.verify_generate and failed:
            raise SystemExit(1)
        gc.collect()
        return

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
    elif args.stage == "post-attn-rmsnorm-mlp-gate-up":
        op = Qwen3PersistentPostAttnRMSNormMLPGateUp(
            hidden_size=model.config.hidden_size,
            intermediate_size=model.config.intermediate_size,
            epsilon=model.config.rms_norm_eps,
            context=context,
        )
    elif args.stage == "post-attn-mlp-down-residual":
        op = Qwen3PersistentPostAttnMLPDownResidual(
            hidden_size=model.config.hidden_size,
            intermediate_size=model.config.intermediate_size,
            context=context,
        )
    elif args.stage == "post-attn-rmsnorm-full-mlp":
        op = Qwen3PersistentPostAttnRMSNormFullMLP(
            hidden_size=model.config.hidden_size,
            intermediate_size=model.config.intermediate_size,
            epsilon=model.config.rms_norm_eps,
            context=context,
        )
    elif args.stage == N_LAYER_FINAL_ONLY_STAGE:
        op = Qwen3PersistentNLayerFinalOnly(
            hidden_size=model.config.hidden_size,
            q_size=model.config.num_attention_heads * model.config.head_dim,
            kv_size=model.config.num_key_value_heads * model.config.head_dim,
            head_dim=model.config.head_dim,
            max_seq_len=args.max_seq_len,
            position=input_ids.shape[1],
            intermediate_size=model.config.intermediate_size,
            layer_iterations=args.layer_chunk_size,
            epsilon=model.config.rms_norm_eps,
            context=context,
        )
    else:
        raise ValueError(f"unsupported stage after attention cleanup: {args.stage}")

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
        f"compute_cores={preflight.compute_cores} "
        f"max_fifo_buffered_bytes={preflight.max_fifo_buffered_bytes} "
        f"max_dma_tasks_per_fifo={preflight.max_dma_tasks_per_fifo} "
        f"max_tile_inputs={preflight.max_compute_tile_inputs} "
        f"max_tile_outputs={preflight.max_compute_tile_outputs} "
        f"non_advancing_acquires={preflight.non_advancing_acquires}"
    )
    if args.dump_proof:
        print_stage_proof(args, op)
    if args.compile_only:
        return

    op_func = op.get_callable()
    failed = run_compiled_stage(args, model, input_ids, op, op_func)
    if args.verify and failed:
        raise SystemExit(1)
    gc.collect()


if __name__ == "__main__":
    main()
