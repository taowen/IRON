# Q4NX Dynamic Readback Probe

This probe isolates the first failure seen by experiment 016: the generated
dynamic arithmetic body emitted a valid record but all payload words were zero.

The program consumes the same first activation/weight chunk as the Q4NX tiny
gates, then writes raw scalar loads from candidate local addresses into the
compact record payload. It does not try to compute Q4NX values. The only goal is
to prove which address form can see DMA-written fields from source assembly.
