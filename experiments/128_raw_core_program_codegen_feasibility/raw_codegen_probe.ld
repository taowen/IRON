MEMORY { program (RX) : ORIGIN = 0, LENGTH = 0x20000 }
ENTRY(__start)
SECTIONS { . = 0x0; .text : { *(.text*) } > program }
