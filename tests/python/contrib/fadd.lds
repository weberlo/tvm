OUTPUT_ARCH( "riscv" )
SECTIONS
{
  . = 0x10010000;
  .text : { *(.text) }
  . = 0x1001c350;
  .data : { *(.data) }
  . = 0x100286a0;
  .bss : { *(.bss) }
  .sbss : { *(.sbss) }
}
