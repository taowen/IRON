# 006 Q4NX Alias Lifetime Graph

This experiment turns the MyLM main16 Q4NX hot loop into an alias-aware
def/use table.

The hardware register names are already understood from AM027:

- `wlN` and `whN` are the 256-bit halves of `xN`;
- `bmllN/bmlhN/bmhlN/bmhhN` are 512-bit accumulator views;
- `cmlN/cmhN` are 1024-bit accumulator aliases;
- `dmN` is the 2048-bit accumulator alias.

The missing part is not the register vocabulary. The missing part is the MyLM
schedule: which alias is defined, consumed, and kept live across the eight
activation groups in `0x260..0x1850`.

This script derives:

- one row per disassembled op fragment;
- direct defs/uses;
- alias write/read families;
- semantic class hints such as activation load, Q4 unpack, UPS, dequant adjust,
  accumulator-to-bf16 conversion, and MAC;
- group-boundary live-through families.

