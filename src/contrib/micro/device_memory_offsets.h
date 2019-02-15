#define PAGE_SIZE     4096

#define SECTION_TEXT  0
#define SECTION_DATA  50000
#define SECTION_BSS   100000
#define SECTION_ARGS  150000
#define SECTION_HEAP  200000

// TODO: Do we need to do this or is this already incorporated by the
// base_addr?
// 0x10010000 in decimal
//#define SECTION_TEXT  268500992
//#define SECTION_DATA  268550992
//#define SECTION_BSS   268600992
//#define SECTION_ARGS  268650992
//#define SECTION_HEAP  268700992

#define MEMORY_SIZE   500000
