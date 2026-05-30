#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MYLM_ROOT="${MYLM_ROOT:-/var/home/taowen/projects/MyLM}"
XRT_ROOT="${XRT_ROOT:-/var/opt/xilinx/xrt}"
CXX="${CXX:-g++}"
OUT="$ROOT/build/mylm_forward_probe"

mkdir -p "$ROOT/build"

"$CXX" -std=c++17 -O0 \
  -I "$MYLM_ROOT/tools/re/stubs" \
  -I "$MYLM_ROOT/src/include" \
  -I "$XRT_ROOT/include" \
  "$ROOT/tools/mylm_forward_probe.cpp" \
  -o "$OUT" \
  -L "$MYLM_ROOT/src/lib" \
  -Wl,-rpath,"$MYLM_ROOT/src/lib" \
  -Wl,-rpath,"$XRT_ROOT/lib64" \
  -L "$XRT_ROOT/lib64" \
  -lqwen3_npu -lq4_npu_eXpress -lmha -lgemm -ldequant -llm_head \
  -lxrt_coreutil -lxrt_core

echo "$OUT"
