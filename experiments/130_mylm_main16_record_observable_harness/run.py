#!/usr/bin/env python3
"""Observe whether the raw MyLM main16 core can emit one compact record."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
PREV_RUN = REPO_ROOT / "experiments/129_mylm_main16_standalone_raw_kernel/run.py"

LLVM_AIE_BIN = REPO_ROOT / ".venv/lib/python3.12/site-packages/llvm-aie/bin"
MLIR_AIE_BIN = REPO_ROOT / ".venv/lib/python3.12/site-packages/mlir_aie/bin"
AIECC = MLIR_AIE_BIN / "aiecc"
LLVM_SIZE = LLVM_AIE_BIN / "llvm-size"
LLVM_READELF = LLVM_AIE_BIN / "llvm-readelf"
XRT_BIN = Path("/var/opt/xilinx/xrt/bin")

DEFAULT_MYLM_RAW = REPO_ROOT / "experiments/129_mylm_main16_standalone_raw_kernel/mylm_c2r2_program.bin"
DEFAULT_STATIC_73C80 = REPO_ROOT / "experiments/129_mylm_main16_standalone_raw_kernel/mylm_static_73c80.bin"
DEFAULT_STATIC_73D00 = REPO_ROOT / "experiments/129_mylm_main16_standalone_raw_kernel/mylm_static_73d00.bin"

DISPATCHER_PRESERVE_P5_OFFSET = 0x36DE
DISPATCHER_PRESERVE_P5_ORIGINAL = bytes.fromhex("ba72600a028b8027d3f7")
DISPATCHER_PRESERVE_P5_PATCHED = bytes.fromhex("ba72600a028b80f72c00")

RAW_COPY = EXPERIMENT_DIR / "mylm_c2r2_program.bin"
STATIC_73C80_COPY = EXPERIMENT_DIR / "mylm_static_73c80.bin"
STATIC_73D00_COPY = EXPERIMENT_DIR / "mylm_static_73d00.bin"
ELF = EXPERIMENT_DIR / "mylm_c2r2_main16_record_exec.elf"
MLIR = EXPERIMENT_DIR / "design.mlir"
TXN = EXPERIMENT_DIR / "design.txn.mlir"
XCLBIN = EXPERIMENT_DIR / "design.xclbin"
INSTS = EXPERIMENT_DIR / "design.bin"
PRJ_DIR = EXPERIMENT_DIR / "prj"
MANIFEST = EXPERIMENT_DIR / "mylm_main16_record_observable_harness.json"
REPORT = EXPERIMENT_DIR / "mylm_main16_record_observable_harness.md"

RECORD_DWORDS_PER_RECORD = 17
CHUNKS_PER_RECORD = 16
ACTIVATION_DWORDS_PER_CHUNK = 128
WEIGHT_DWORDS_PER_CHUNK = 1280


def activation_dwords(records: int) -> int:
    return records * CHUNKS_PER_RECORD * ACTIVATION_DWORDS_PER_CHUNK


def weight_dwords(records: int) -> int:
    return records * CHUNKS_PER_RECORD * WEIGHT_DWORDS_PER_CHUNK


def output_dwords(records: int) -> int:
    return records * RECORD_DWORDS_PER_RECORD


def _load_prev_helpers() -> Any:
    spec = importlib.util.spec_from_file_location("mylm_exp129", PREV_RUN)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper module: {PREV_RUN}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


HELPERS = _load_prev_helpers()
Segment = HELPERS.Segment
wrap_segments_as_aie_exec = HELPERS.wrap_segments_as_aie_exec
run_cmd = HELPERS.run_cmd
xrt_validate = HELPERS.xrt_validate
_short_output = HELPERS._short_output


def as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode(errors="replace")
    return str(value)


def json_safe(value: object) -> object:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, bytes):
        return as_text(value)
    return value


def copy_required(src: Path, dst: Path) -> bytes:
    if not src.exists():
        raise FileNotFoundError(src)
    if src.resolve() != dst.resolve():
        shutil.copy2(src, dst)
    return dst.read_bytes()


def maybe_patch_raw(raw: bytes, preserve_dispatcher_p5: bool) -> tuple[bytes, list[dict[str, Any]]]:
    patches: list[dict[str, Any]] = []
    if preserve_dispatcher_p5:
        start = DISPATCHER_PRESERVE_P5_OFFSET
        end = start + len(DISPATCHER_PRESERVE_P5_ORIGINAL)
        found = raw[start:end]
        if found != DISPATCHER_PRESERVE_P5_ORIGINAL:
            raise ValueError(
                "dispatcher preserve-p5 patch preimage mismatch at "
                f"0x{start:x}: got {found.hex()}, expected {DISPATCHER_PRESERVE_P5_ORIGINAL.hex()}"
            )
        raw = raw[:start] + DISPATCHER_PRESERVE_P5_PATCHED + raw[end:]
        patches.append(
            {
                "name": "preserve_dispatcher_p5",
                "offset": hex(start),
                "original": DISPATCHER_PRESERVE_P5_ORIGINAL.hex(),
                "patched": DISPATCHER_PRESERVE_P5_PATCHED.hex(),
                "meaning": "replace lda p5,[sp,#-68] with nopa in the same bundle, preserving caller p5",
            }
        )
    return raw, patches


def write_elf(
    raw_path: Path,
    static_73c80: Path,
    static_73d00: Path,
    *,
    preserve_dispatcher_p5: bool,
) -> dict[str, Any]:
    raw = copy_required(raw_path, RAW_COPY)
    raw, patches = maybe_patch_raw(raw, preserve_dispatcher_p5)
    RAW_COPY.write_bytes(raw)
    data_73c80 = copy_required(static_73c80, STATIC_73C80_COPY)
    data_73d00 = copy_required(static_73d00, STATIC_73D00_COPY)
    segments = [
        Segment(".text", 0x0, raw, 0x6, 0x5),
        Segment(".mylm_static_73c80", 0x73C80, data_73c80, 0x3, 0x6),
        Segment(".mylm_static_73d00", 0x73D00, data_73d00, 0x3, 0x6),
    ]
    ELF.write_bytes(wrap_segments_as_aie_exec(segments, entry=0))
    return {
        "raw_bytes": len(raw),
        "static_segments": [
            {"name": seg.name, "addr": hex(seg.addr), "bytes": len(seg.data)}
            for seg in segments[1:]
        ],
        "raw_patches": patches,
    }


def npu_writebd(column: int, bd_id: int, buffer_length: int, buffer_offset: int = 0) -> str:
    return (
        f"      aiex.npu.writebd {{bd_id = {bd_id} : i32, "
        f"buffer_length = {buffer_length} : i32, buffer_offset = {buffer_offset} : i32, "
        f"burst_length = 64 : i32, column = {column} : i32, "
        "d0_size = 0 : i32, d0_stride = 0 : i32, "
        "d0_zero_after = 0 : i32, d0_zero_before = 0 : i32, "
        "d1_size = 0 : i32, d1_stride = 0 : i32, "
        "d1_zero_after = 0 : i32, d1_zero_before = 0 : i32, "
        "d2_size = 0 : i32, d2_stride = 0 : i32, "
        "d2_zero_after = 0 : i32, d2_zero_before = 0 : i32, "
        "enable_packet = 0 : i32, iteration_current = 0 : i32, "
        "iteration_size = 0 : i32, iteration_stride = 0 : i32, "
        "lock_acq_enable = 0 : i32, lock_acq_id = 0 : i32, lock_acq_val = 0 : i32, "
        "lock_rel_id = 0 : i32, lock_rel_val = 0 : i32, "
        "next_bd = 0 : i32, out_of_order_id = 0 : i32, "
        "packet_id = 0 : i32, packet_type = 0 : i32, row = 0 : i32, "
        "use_next_bd = 0 : i32, valid_bd = 1 : i32}"
    )


def shim_bd_address(column: int, bd_id: int) -> int:
    return column * 0x02000000 + 0x1D004 + bd_id * 0x20


def address_patch(column: int, bd_id: int, arg_idx: int) -> str:
    return (
        f"      aiex.npu.address_patch {{addr = {shim_bd_address(column, bd_id)} : ui32, "
        f"arg_idx = {arg_idx} : i32, arg_plus = 0 : i32}}"
    )


def push_queue(column: int, direction: str, channel: int, bd_id: int) -> str:
    return (
        f"      aiex.npu.push_queue({column}, 0, {direction} : {channel}) "
        f"{{bd_id = {bd_id} : i32, issue_token = true, repeat_count = 0 : i32}}"
    )


def npu_sync(column: int, channel: int, direction: int) -> str:
    return (
        f"      aiex.npu.sync {{channel = {channel} : i32, column = {column} : i32, "
        f"column_num = 1 : i32, direction = {direction} : i32, row = 0 : i32, row_num = 1 : i32}}"
    )


def write_mlir(records: int) -> None:
    out_dwords = output_dwords(records)
    act_dwords = activation_dwords(records)
    wt_dwords = weight_dwords(records)
    MLIR.write_text(
        "\n".join(
            [
                "module @mylm_main16_record_observable_harness {",
                "  aie.device(npu2) {",
                "    %shim_in = aie.tile(2, 0)",
                "    %shim_tap = aie.tile(3, 0)",
                "    %t22 = aie.tile(2, 2)",
                "",
                "    aie.flow(%shim_in, DMA : 0, %t22, DMA : 0)",
                "    aie.flow(%shim_in, DMA : 1, %t22, DMA : 1)",
                "    aie.flow(%t22, DMA : 1, %shim_tap, DMA : 1)",
                "",
                '    %activation_ping = aie.buffer(%t22) {address = 32768 : i32, sym_name = "activation_ping"} : memref<128xi32>',
                '    %activation_pong = aie.buffer(%t22) {address = 49152 : i32, sym_name = "activation_pong"} : memref<128xi32>',
                '    %weight_ping = aie.buffer(%t22) {address = 10240 : i32, sym_name = "weight_ping"} : memref<2560xbf16>',
                '    %weight_pong = aie.buffer(%t22) {address = 16384 : i32, sym_name = "weight_pong"} : memref<2560xbf16>',
                '    %record_ping = aie.buffer(%t22) {address = 15388 : i32, sym_name = "record_ping"} : memref<17xi32>',
                '    %record_pong = aie.buffer(%t22) {address = 21532 : i32, sym_name = "record_pong"} : memref<17xi32>',
                "",
                '    %activation_empty = aie.lock(%t22, 0) {init = 2 : i32, sym_name = "activation_empty"}',
                '    %activation_full = aie.lock(%t22, 1) {init = 0 : i32, sym_name = "activation_full"}',
                '    %weight_empty = aie.lock(%t22, 2) {init = 2 : i32, sym_name = "weight_empty"}',
                '    %weight_full = aie.lock(%t22, 3) {init = 0 : i32, sym_name = "weight_full"}',
                '    %record_empty = aie.lock(%t22, 4) {init = 2 : i32, sym_name = "record_empty"}',
                '    %record_full = aie.lock(%t22, 5) {init = 0 : i32, sym_name = "record_full"}',
                '    %start_lock = aie.lock(%t22, 6) {init = 0 : i32, sym_name = "start_lock"}',
                "",
                "    %c22 = aie.core(%t22) {",
                "      aie.end",
                f'    }} {{elf_file = "{ELF.name}"}}',
                "",
                "    %m22 = aie.mem(%t22) {",
                "      %activation_dma = aie.dma_start(S2MM, 0, ^activation_ping_in, ^weight_start)",
                "    ^activation_ping_in:",
                "      aie.use_lock(%activation_empty, AcquireGreaterEqual, 1)",
                "      aie.dma_bd(%activation_ping : memref<128xi32>, 0, 128) {bd_id = 0 : i32, next_bd_id = 1 : i32}",
                "      aie.use_lock(%activation_full, Release, 1)",
                "      aie.next_bd ^activation_pong_in",
                "    ^activation_pong_in:",
                "      aie.use_lock(%activation_empty, AcquireGreaterEqual, 1)",
                "      aie.dma_bd(%activation_pong : memref<128xi32>, 0, 128) {bd_id = 1 : i32, next_bd_id = 0 : i32}",
                "      aie.use_lock(%activation_full, Release, 1)",
                "      aie.next_bd ^activation_ping_in",
                "",
                "    ^weight_start:",
                "      %weight_dma = aie.dma_start(S2MM, 1, ^weight_ping_in, ^record_start)",
                "    ^weight_ping_in:",
                "      aie.use_lock(%weight_empty, AcquireGreaterEqual, 1)",
                "      aie.dma_bd(%weight_ping : memref<2560xbf16>, 0, 2560) {bd_id = 2 : i32, next_bd_id = 3 : i32}",
                "      aie.use_lock(%weight_full, Release, 1)",
                "      aie.next_bd ^weight_pong_in",
                "    ^weight_pong_in:",
                "      aie.use_lock(%weight_empty, AcquireGreaterEqual, 1)",
                "      aie.dma_bd(%weight_pong : memref<2560xbf16>, 0, 2560) {bd_id = 3 : i32, next_bd_id = 2 : i32}",
                "      aie.use_lock(%weight_full, Release, 1)",
                "      aie.next_bd ^weight_ping_in",
                "",
                "    ^record_start:",
                "      %record_dma = aie.dma_start(MM2S, 1, ^record_ping_out, ^end)",
                "    ^record_ping_out:",
                "      aie.use_lock(%record_full, AcquireGreaterEqual, 1)",
                "      aie.dma_bd(%record_ping : memref<17xi32>, 0, 17) {bd_id = 4 : i32, next_bd_id = 5 : i32}",
                "      aie.use_lock(%record_empty, Release, 1)",
                "      aie.next_bd ^record_pong_out",
                "    ^record_pong_out:",
                "      aie.use_lock(%record_full, AcquireGreaterEqual, 1)",
                "      aie.dma_bd(%record_pong : memref<17xi32>, 0, 17) {bd_id = 5 : i32, next_bd_id = 4 : i32}",
                "      aie.use_lock(%record_empty, Release, 1)",
                "      aie.next_bd ^record_ping_out",
                "    ^end:",
                "      aie.end",
                "    }",
                "",
                f"    aie.runtime_sequence(%out: memref<{out_dwords}xi32>, %weights: memref<{wt_dwords}xi32>, %activation: memref<{act_dwords}xi32>) {{",
                npu_writebd(3, 2, out_dwords),
                address_patch(3, 2, 0),
                push_queue(3, "S2MM", 1, 2),
                npu_writebd(2, 12, act_dwords),
                address_patch(2, 12, 2),
                push_queue(2, "MM2S", 0, 12),
                npu_writebd(2, 14, wt_dwords),
                address_patch(2, 14, 1),
                push_queue(2, "MM2S", 1, 14),
                "      aiex.set_lock(%start_lock, 1)",
                npu_sync(3, 1, 0),
                "    }",
                "  }",
                "}",
                "",
            ]
        )
    )


def package_with_aiecc() -> None:
    if PRJ_DIR.exists():
        shutil.rmtree(PRJ_DIR)
    PRJ_DIR.mkdir()
    shutil.copy2(ELF, PRJ_DIR / ELF.name)
    env = os.environ.copy()
    env["PATH"] = f"{XRT_BIN}:{env.get('PATH', '')}"
    run_cmd(
        (
            str(AIECC),
            "-j1",
            "--no-compile",
            "--no-compile-host",
            "--no-xchesscc",
            "--no-xbridge",
            "--alloc-scheme=basic-sequential",
            "--peano",
            str(REPO_ROOT / ".venv/lib/python3.12/site-packages/llvm-aie"),
            "--aie-generate-xclbin",
            f"--xclbin-name={XCLBIN}",
            "--xclbin-kernel-name=MLIR_AIE",
            "--aie-generate-txn",
            f"--txn-name={TXN}",
            "--aie-generate-npu-insts",
            f"--npu-insts-name={INSTS}",
            f"--tmpdir={PRJ_DIR}",
            str(MLIR),
        ),
        env=env,
    )


def section_sizes(elf: Path) -> dict[str, int]:
    output = run_cmd((str(LLVM_SIZE), "-A", str(elf))).stdout
    sizes: dict[str, int] = {}
    for line in output.splitlines():
        fields = line.split()
        if len(fields) >= 2 and fields[0].startswith("."):
            sizes[fields[0]] = int(fields[1], 0)
    return sizes


def elf_header(elf: Path) -> dict[str, Any]:
    output = run_cmd((str(LLVM_READELF), "-h", "-l", "-S", str(elf))).stdout
    elf_type = re.search(r"Type:\s+(\S+)", output)
    load_segments = len(re.findall(r"^\s+LOAD\s", output, re.MULTILINE))
    return {
        "type": elf_type.group(1) if elf_type else None,
        "load_segments": load_segments,
    }


def txn_payload_sizes(txn: Path) -> list[int]:
    text = txn.read_text() if txn.exists() else ""
    return [
        int(words) * 4
        for words in re.findall(
            r"memref\.global\s+\"private\"\s+constant\s+@config_blockwrite_data_\d+\s*:\s*memref<(\d+)xi32>",
            text,
        )
    ]


def txn_blockwrite_addresses(txn: Path) -> list[int]:
    text = txn.read_text() if txn.exists() else ""
    return [int(address) for address in re.findall(r"aiex\.npu\.blockwrite\(%\d+\) \{address = (\d+) : ui32\}", text)]


def try_run_observable(timeout_s: int, records: int, activation_pong_flag: int) -> dict[str, Any]:
    env = os.environ.copy()
    env["PATH"] = f"{XRT_BIN}:{env.get('PATH', '')}"
    out_dwords = output_dwords(records)
    act_dwords = activation_dwords(records)
    wt_dwords = weight_dwords(records)
    code = f"""
import os, time, traceback
os.environ['PATH'] = {env["PATH"]!r}
import numpy as np
import torch
import aie.utils as aie_utils
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
from aie.utils.npukernel import NPUKernel

try:
    aie_utils.DefaultNPURuntime.cleanup()
except Exception:
    pass

try:
    kernel = NPUKernel(xclbin_path={str(XCLBIN)!r}, kernel_name='MLIR_AIE', insts_path={str(INSTS)!r})
    print('load_begin')
    handle = aie_utils.DefaultNPURuntime.load(kernel)
    print('load_ok')
    out_buf = XRTTensor(({out_dwords},), dtype=np.int32)
    activation = np.zeros(({act_dwords},), dtype=np.int32)
    if activation.size > 128:
        activation[128] = {activation_pong_flag}
    weights = np.zeros(({wt_dwords},), dtype=np.int32)
    activation_buf = XRTTensor.from_torch(torch.from_numpy(activation).to(torch.int32))
    weights_buf = XRTTensor.from_torch(torch.from_numpy(weights).to(torch.int32))
    start = time.time()
    result = aie_utils.DefaultNPURuntime.run(handle, [out_buf, weights_buf, activation_buf])
    elapsed = time.time() - start
    got = out_buf.to_torch().numpy().astype(np.int32)
    print('run_ok')
    print('elapsed=' + str(elapsed))
    print('result_type=' + type(result).__name__)
    print('npu_time=' + str(getattr(result, 'npu_time', None)))
    print('record=' + ','.join(hex(int(x) & 0xffffffff) for x in got.tolist()))
except Exception:
    print('exception_begin')
    traceback.print_exc()
    raise
finally:
    try:
        aie_utils.DefaultNPURuntime.cleanup()
    except Exception:
        traceback.print_exc()
"""
    started = time.time()
    try:
        result = run_cmd(
            (str(REPO_ROOT / ".venv/bin/python"), "-c", code),
            env=env,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timeout_waiting_for_record",
            "timeout_s": timeout_s,
            "elapsed_s": time.time() - started,
            "stdout": as_text(exc.stdout),
            "stderr": as_text(exc.stderr),
        }
    output = result.stdout + result.stderr
    record_match = re.search(r"^record=([^\n]+)", result.stdout, re.MULTILINE)
    record_words = record_match.group(1).split(",") if record_match else []
    if result.returncode == 0 and len(record_words) == out_dwords:
        status = "record_observed"
    elif "DRM_IOCTL_AMDXDNA_GET_INFO IOCTL failed" in output:
        status = "xrt_get_info_failed"
    elif "load_ok" in result.stdout:
        status = "loaded_but_no_record"
    else:
        status = "load_failed"
    return {
        "status": status,
        "returncode": result.returncode,
        "elapsed_s": time.time() - started,
        "record_words": record_words,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    if args.records <= 0:
        raise ValueError("--records must be positive")
    source_info = write_elf(
        args.mylm_raw,
        args.static_73c80,
        args.static_73d00,
        preserve_dispatcher_p5=args.preserve_dispatcher_p5,
    )
    write_mlir(args.records)
    package_with_aiecc()
    runtime = try_run_observable(args.runtime_timeout, args.records, args.activation_pong_flag)
    validate = xrt_validate(args.xrt_timeout)
    record_words = runtime.get("record_words") or []
    record_headers = record_words[0::RECORD_DWORDS_PER_RECORD]
    manifest: dict[str, Any] = {
        "status": runtime["status"],
        "source": {
            "mylm_raw": str(args.mylm_raw),
            "static_73c80": str(args.static_73c80),
            "static_73d00": str(args.static_73d00),
            **source_info,
        },
        "harness": {
            "tile": "c2r2",
            "records_to_observe": args.records,
            "activation_chunks": args.records * CHUNKS_PER_RECORD,
            "weight_chunks": args.records * CHUNKS_PER_RECORD,
            "activation_dwords": activation_dwords(args.records),
            "weight_dwords": weight_dwords(args.records),
            "record_dwords": output_dwords(args.records),
            "preserve_dispatcher_p5": args.preserve_dispatcher_p5,
            "activation_pong_flag": args.activation_pong_flag,
            "route": "shim2 MM2S0/1 -> c2r2 DMA0/1; c2r2 MM2S1 -> shim3 S2MM1",
        },
        "artifacts": {
            "elf": str(ELF.relative_to(REPO_ROOT)),
            "mlir": str(MLIR.relative_to(REPO_ROOT)),
            "txn": str(TXN.relative_to(REPO_ROOT)),
            "xclbin": str(XCLBIN.relative_to(REPO_ROOT)),
            "insts": str(INSTS.relative_to(REPO_ROOT)),
        },
        "elf": {
            "header": elf_header(ELF),
            "sections": section_sizes(ELF),
        },
        "transaction": {
            "payload_bytes": txn_payload_sizes(TXN),
            "blockwrite_addresses": [hex(address) for address in txn_blockwrite_addresses(TXN)],
        },
        "runtime_probe": runtime,
        "record_headers": record_headers,
        "unique_record_headers": sorted(set(record_headers)),
        "xrt_validate": validate,
    }
    return manifest


def render_report(manifest: dict[str, Any]) -> str:
    runtime = manifest["runtime_probe"]
    validate = manifest["xrt_validate"]
    headers = manifest.get("record_headers", [])
    header_preview = headers[:8]
    header_tail = headers[-4:] if len(headers) > 8 else []
    lines = [
        "# MyLM Main16 Record-Observable Harness",
        "",
        f"- Status: `{manifest['status']}`",
        f"- Route: `{manifest['harness']['route']}`",
        f"- Records requested: `{manifest['harness']['records_to_observe']}`",
        f"- Fed chunks: activation `{manifest['harness']['activation_chunks']}`, weight `{manifest['harness']['weight_chunks']}`",
        f"- Preserve dispatcher p5 patch: `{manifest['harness']['preserve_dispatcher_p5']}`",
        f"- Activation pong flag word: `{manifest['harness']['activation_pong_flag']}`",
        f"- ELF load segments: `{manifest['elf']['header']['load_segments']}`",
        f"- Transaction blockwrite addresses: `{manifest['transaction']['blockwrite_addresses']}`",
        f"- Runtime probe: `{runtime['status']}`",
        f"- Unique record headers: `{manifest.get('unique_record_headers', [])}`",
        f"- `xrt-smi validate`: `{validate['status']}`",
        "",
        "## Runtime Output",
        "",
        "```text",
        _short_output((runtime.get("stdout") or "") + (runtime.get("stderr") or ""), max_chars=1400),
        "```",
        "",
        "## Interpretation",
        "",
    ]
    if runtime["status"] == "record_observed":
        lines.extend(
            [
                "The raw MyLM c2r2 program loaded as a whole-core ELF, consumed the",
                "requested activation/weight stream windows through MLIR-AIE DMA/locks, and",
                "released `record_full` far enough for the shim S2MM `npu.sync` to complete.",
                "",
                f"Observed header preview: `{header_preview}`",
            ]
        )
        if header_tail:
            lines.append(f"Observed header tail: `{header_tail}`")
        if manifest["harness"]["preserve_dispatcher_p5"] and manifest["harness"]["activation_pong_flag"] == 1:
            lines.append(
                "This run forces the dispatcher-visible control pointer to remain on the caller-provided "
                "activation-pong buffer and sets that flag word to 1, which selects the normal path."
            )
    elif runtime["status"] == "timeout_waiting_for_record":
        lines.append(
            "The package loaded, but the host timed out waiting for the 17-dword record. "
            "That means the remaining mismatch is in route/locks/control or the raw core "
            "did not reach its first record emit under this standalone harness."
        )
    else:
        lines.append("The observable harness did not reach the record wait point cleanly; inspect stdout/stderr above.")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- MLIR: `{manifest['artifacts']['mlir']}`",
            f"- ELF: `{manifest['artifacts']['elf']}`",
            f"- XCLBIN: `{manifest['artifacts']['xclbin']}`",
            f"- Insts: `{manifest['artifacts']['insts']}`",
            f"- TXN: `{manifest['artifacts']['txn']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mylm-raw", type=Path, default=DEFAULT_MYLM_RAW)
    parser.add_argument("--static-73c80", type=Path, default=DEFAULT_STATIC_73C80)
    parser.add_argument("--static-73d00", type=Path, default=DEFAULT_STATIC_73D00)
    parser.add_argument("--records", type=int, default=1, help="17-dword records to wait for")
    parser.add_argument("--preserve-dispatcher-p5", action="store_true")
    parser.add_argument("--activation-pong-flag", type=int, default=0)
    parser.add_argument("--runtime-timeout", type=int, default=20)
    parser.add_argument("--xrt-timeout", type=int, default=20)
    return parser


def main() -> int:
    args = _build_arg_parser().parse_args()
    try:
        manifest = build_manifest(args)
    except Exception:
        failure = {
            "status": "experiment_failed",
            "traceback": traceback.format_exc(),
        }
        MANIFEST.write_text(json.dumps(json_safe(failure), indent=2, sort_keys=True) + "\n")
        REPORT.write_text(
            "# MyLM Main16 Record-Observable Harness\n\nExperiment failed.\n\n```text\n"
            + failure["traceback"]
            + "```\n"
        )
        print(f"wrote {REPORT}")
        print(f"wrote {MANIFEST}")
        raise
    MANIFEST.write_text(json.dumps(json_safe(manifest), indent=2, sort_keys=True) + "\n")
    REPORT.write_text(render_report(manifest))
    print(f"wrote {REPORT}")
    print(f"wrote {MANIFEST}")
    print(f"status: {manifest['status']}")
    return 0 if manifest["status"] in {"record_observed", "timeout_waiting_for_record"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
