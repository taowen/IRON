MEMORY {
  program (RX) : ORIGIN = 0, LENGTH = 0x20000
  data (!RX) : ORIGIN = 0x7C200, LENGTH = 0x3E00
}
ENTRY(q4nx_main16_whole_program_entry)
SECTIONS {
  . = 0x0;
  .text : {
    *(.text.q4nx_main16_whole_program_entry)
    *(.text*)
  } > program
  .data : { *(.data*) *(.rodata*) } > data
  .bss : { *(.bss*) } > data
}
