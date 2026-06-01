# Q4NX MIR Hotloop Patch Numeric Gate

This experiment patches the generated MIR hot-loop body from experiment 026 into
the MyLM raw main16 program.

It keeps the MyLM `0x01f0` prologue, Q/K/V phase body, group-sum producer,
record emitter, DMA, and lock behavior. Only the `0x260..0x1850` hot-loop body
is replaced. This is the first numeric gate for the direct-MIR codegen route.
