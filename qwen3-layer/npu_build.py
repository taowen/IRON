"""Shared build and runtime helpers for qwen3-layer NPU cases."""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from pathlib import Path
from typing import Protocol

os.environ["PATH"] = "/var/opt/xilinx/xrt/bin:" + os.environ.get("PATH", "")

import aie.utils as aie_utils
from aie.utils.config import peano_install_dir, root_path
from aie.utils.npukernel import NPUKernel

EXPERIMENT_DIR = Path(__file__).parent
ROLE_KERNEL_SOURCES = {
    "edge_attention.o": "edge_attention.cc",
    "full_vector_station.o": "full_vector_station.cc",
    "main_projection_q4nx_fast.o": "main_projection_q4nx_fast.cc",
    "postprocess_qkv.o": "postprocess_qkv.cc",
    "swiglu.o": "swiglu.cc",
}
ROLE_KERNEL_HEADERS = ("qwen3_constants.h", "record_format.h")
LINK_WITH_RE = re.compile(r'link_with = "[^"]*/([^/"]+\.o)"')


class BuildHasher(Protocol):
    def update(self, data: bytes, /) -> None:
        ...


def run_command(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def _compile_aie_object(source_name: str, object_name: str) -> None:
    peano_dir = Path(peano_install_dir())
    mlir_aie_dir = Path(root_path())
    clang = peano_dir / "bin" / "clang++"
    include_path = mlir_aie_dir / "include"
    runtime_lib_include = mlir_aie_dir / "aie_runtime_lib" / "AIE2P"
    src = EXPERIMENT_DIR / source_name
    obj = EXPERIMENT_DIR / object_name
    cmd = [
        str(clang),
        "-O2",
        "-std=c++20",
        "--target=aie2p-none-unknown-elf",
        "-Wno-parentheses",
        "-Wno-attributes",
        "-Wno-macro-redefined",
        "-Wno-empty-body",
        "-Wno-missing-template-arg-list-after-template-kw",
        f"-I{include_path}",
        f"-I{runtime_lib_include}",
        "-c",
        str(src),
        "-o",
        str(obj),
    ]
    print(f"  Compiling {source_name}...")
    run_command(cmd)


def _linked_role_objects(mlir_text: str) -> tuple[str, ...]:
    seen: set[str] = set()
    objects: list[str] = []
    for object_name in LINK_WITH_RE.findall(mlir_text):
        if object_name not in ROLE_KERNEL_SOURCES:
            raise ValueError(f"unknown AIE role object in link_with: {object_name}")
        if object_name not in seen:
            seen.add(object_name)
            objects.append(object_name)
    return tuple(objects)


def _compile_linked_role_objects(object_names: tuple[str, ...]) -> None:
    for object_name in object_names:
        _compile_aie_object(ROLE_KERNEL_SOURCES[object_name], object_name)


def _update_build_key_file(hasher: BuildHasher, path: Path) -> None:
    hasher.update(path.name.encode())
    hasher.update(b"\0")
    hasher.update(path.read_bytes())
    hasher.update(b"\0")


def _build_key(mlir_text: str, object_names: tuple[str, ...], command: list[str]) -> str:
    hasher = hashlib.sha256()
    hasher.update(b"qwen3-layer-npu-build-v1\0")
    hasher.update("\n".join(command).encode())
    hasher.update(b"\0")
    hasher.update(mlir_text.encode())
    hasher.update(b"\0")
    for header_name in ROLE_KERNEL_HEADERS:
        _update_build_key_file(hasher, EXPERIMENT_DIR / header_name)
    for object_name in object_names:
        _update_build_key_file(hasher, EXPERIMENT_DIR / ROLE_KERNEL_SOURCES[object_name])
    return hasher.hexdigest()


def compile_mlir(mlir_path: Path, xclbin_path: Path, insts_path: Path) -> None:
    mlir_text = mlir_path.read_text()
    object_names = _linked_role_objects(mlir_text)
    mlir_aie_dir = Path(root_path())
    peano_dir = Path(peano_install_dir())
    aiecc = mlir_aie_dir / "bin" / "aiecc"
    cmd = [
        str(aiecc),
        "-v",
        "-j1",
        "--no-compile-host",
        "--no-xchesscc",
        "--no-xbridge",
        "--alloc-scheme=basic-sequential",
        "--peano",
        str(peano_dir),
        "--aie-generate-xclbin",
        f"--xclbin-name={xclbin_path}",
        "--xclbin-kernel-name=MLIR_AIE",
        "--aie-generate-npu-insts",
        f"--npu-insts-name={insts_path}",
        str(mlir_path),
    ]
    key_path = xclbin_path.with_suffix(".buildkey")
    build_key = _build_key(mlir_text, object_names, cmd)
    if xclbin_path.exists() and insts_path.exists() and key_path.exists() and key_path.read_text() == build_key:
        print(f"  Reusing cached NPU build: {xclbin_path}")
        return
    _compile_linked_role_objects(object_names)
    print("  Compiling MLIR...")
    run_command(cmd)
    key_path.write_text(build_key)


def load_kernel(xclbin_path: Path, insts_path: Path):
    kernel = NPUKernel(
        xclbin_path=str(xclbin_path),
        kernel_name="MLIR_AIE",
        insts_path=str(insts_path),
    )
    return aie_utils.DefaultNPURuntime.load(kernel)


def device() -> object:
    return aie_utils.DefaultNPURuntime.device()


def run(handle: object, buffers: list[object]) -> object:
    return aie_utils.DefaultNPURuntime.run(handle, buffers)


def cleanup() -> None:
    aie_utils.DefaultNPURuntime.cleanup()
