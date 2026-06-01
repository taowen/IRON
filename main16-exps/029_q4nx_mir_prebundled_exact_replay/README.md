# 029 Q4NX MIR Prebundled Exact Replay

This experiment is a MIR usage gate for the MyLM main16 Q4NX hot loop.

It proves the correct MIR boundary for hand-scheduled AIE2P code:

- use pre-bundled MIR `BUNDLE { ... }` when the schedule is already known;
- compile it with `--start-after=postmisched --skip-machine-alignment`;
- preserve MyLM address gaps as explicit `BUNDLE { NOP }` padding;
- compare generated `.text` bytes against the original raw hot-loop bytes.

Passing this gate means MIR can encode the MyLM hot-loop body exactly. It does
not mean a rewritten kernel is correct; that still needs the NPU numeric gate.
