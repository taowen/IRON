# 030 Q4NX MIR Byte-Exact No-Op Numeric Gate

This experiment connects the byte-exact pre-bundled MIR object from experiment
029 to the direct-QKV NPU harness.

It patches the generated `.text` over MyLM's `0x260..0x1850` hot loop, then
checks that the full patched raw program is byte-identical to the original raw
program before building the xclbin.

This is intentionally a no-op replacement. Passing it proves the MIR object
extraction, raw patching, xclbin packaging, and NPU runtime path preserve MyLM
numerics before any bundle-level mutation is attempted.
