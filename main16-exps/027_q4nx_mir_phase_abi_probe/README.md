# Q4NX MIR Phase ABI Probe

This experiment identifies the missing ABI layer between the full hot-loop MIR
object from experiment 026 and a runnable direct-QKV numeric harness.

It checks the MyLM Q/K/V phase body and the `0x01f0` microkernel prologue so the
next runnable kernel does not blindly jump into `0x260..0x1850` without the
required pointer/control/register state.
