  .section .text,"ax",@progbits
  .globl __start
  .type __start,@function
  .p2align 4
__start:
  mova r0, #7
  mova r1, #35
.Lspin:
  add r0, r0, #1
  j #.Lspin
  nop
  nop
  nop
  nop
  nop
  .size __start, .-__start
