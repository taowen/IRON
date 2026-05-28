"""Shared build and runtime helpers for recipe NPU cases."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

os.environ["PATH"] = "/var/opt/xilinx/xrt/bin:" + os.environ.get("PATH", "")

import aie.utils as aie_utils
from aie.utils.config import peano_install_dir, root_path
from aie.utils.npukernel import NPUKernel

EXPERIMENT_DIR = Path(__file__).parent


def run_command(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def compile_aie_object(source_name: str, object_name: str) -> None:
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


def compile_bridge_kernel() -> None:
    compile_aie_object("dataflow_kernels.cc", "dataflow_kernels.o")


def compile_mlir(mlir_path: Path, xclbin_path: Path, insts_path: Path) -> None:
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
        "--peano",
        str(peano_dir),
        "--aie-generate-xclbin",
        f"--xclbin-name={xclbin_path}",
        "--xclbin-kernel-name=MLIR_AIE",
        "--aie-generate-npu-insts",
        f"--npu-insts-name={insts_path}",
        str(mlir_path),
    ]
    print("  Compiling MLIR...")
    run_command(cmd)


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
