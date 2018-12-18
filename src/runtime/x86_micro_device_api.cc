/*!
 *  Copyright (c) 2018 by Contributors
 * \file x86_micro_device_api.cc
 * \brief x86-emulated micro device API
 */
#include "x86_micro_device_api.h"
#include <sys/mman.h>
#include <dmlc/memory_io.h>
#include <mutex>
#include <errno.h>

namespace tvm {
namespace runtime {
  x86MicroDeviceAPI::x86MicroDeviceAPI(size_t num_bytes) 
    : size(num_bytes) {
    size = num_bytes;
    size_in_pages = (num_bytes + PAGE_SIZE - 1) / PAGE_SIZE;
    printf("allocated size %d num pages %d\n", size, size_in_pages);
    int prot = PROT_READ | PROT_WRITE | PROT_EXEC;
    int flags = MAP_ANONYMOUS | MAP_PRIVATE;
    base_addr = (uint8_t*) mmap(NULL, size_in_pages * PAGE_SIZE, prot, flags, -1, 0);
    printf("errno %d\n", errno);
    printf("base addr original %p\n", base_addr);
    //int buf_size = 10 * PAGE_SIZE;
    stream = new dmlc::MemoryStringStream(&args_buf);
    printf("initialized x86microdeviceapi %p\n", this);
  }

  x86MicroDeviceAPI::~x86MicroDeviceAPI() {
    Shutdown();
  }

  void x86MicroDeviceAPI::WriteToMemory(TVMContext ctx,
                     void* offset,
                     uint8_t* buf,
                     size_t num_bytes) {
    printf("writing x86 to memory\n");
    fflush(stdout);
    uint8_t* real_addr = GetRealAddr(offset);
    printf("real addr %p\n", real_addr);
    fflush(stdout);
    printf("num_bytes %d\n", num_bytes);
    fflush(stdout);
    int i = 0;
    uint8_t x;
    for (i = 0; i < num_bytes; i++)
      x = buf[i];
    printf("forloop 1 done\n");
    fflush(stdout);
    for (i = 0; i < num_bytes; i++)
      real_addr[i];
    printf("forloop 2 done\n");
    fflush(stdout);
    std::memcpy(real_addr, buf, num_bytes);
    printf("memcpy done\n");
    fflush(stdout);
  }

  void x86MicroDeviceAPI::ReadFromMemory(TVMContext ctx,
                      void* offset,
                      uint8_t* buf,
                      size_t num_bytes) {
    uint8_t* real_addr = GetRealAddr(offset);
    std::memcpy(buf, real_addr, num_bytes);
  }

  void x86MicroDeviceAPI::ChangeMemoryProtection(TVMContext ctx,
                              void* offset,
                              int prot,
                              size_t num_bytes) {
    // needs to be page aligned
    // we assume all memory is executable for now, so this isn't called
    uint8_t* real_addr = GetRealAddr(offset);
    mprotect(real_addr, num_bytes, prot);
  }

  /* Description of the args section
   * 
   * it will have void* args_location
   */

  void x86MicroDeviceAPI::Execute(TVMContext ctx, TVMArgs args, TVMRetValue *rv, void* offset) {
    // TODO: need to call init stub with this addr after copying args?
    // how will init stub know which function to call?
    // TODO: use MemoryIO or Stream from dmlc_core to write args in binary
    // TODO: should args section choice be at the micro level?
    // CopyArgs(args, rv);
    printf("running execute in x86MicroDeviceAPI\n");
    fflush(stdout);
    printf("sizeof args %d\n", (size_t) sizeof(args));
    // TODO: copy individual args -- how to deep copy?
    const TVMValue* values = args.values;
    const int* type_codes = args.type_codes;
    int num_args = args.num_args;
    //stream->Write((void*) &args, (size_t) sizeof(args)); // find correct/exact size
    void* args_section = (void *)(30 * PAGE_SIZE);
    printf("Execute: writing to memory x86MicroDeviceAPI\n");
    fflush(stdout);
    printf("size of args_buf %d\n", sizeof(args_buf));
    fflush(stdout);
    WriteToMemory(ctx, args_section, (uint8_t*) &args_buf[0], (size_t) sizeof(args_buf));
    //ReadFromMemory(ctx, args_section, read_buffer, num_args * (sizeof(TVMValue) + sizeof(int)) );
    printf("Execute: wrote args to memory\n");
    fflush(stdout);
    uint8_t* real_addr = GetRealAddr(offset);
    // This should be the function signature if it's to know where things are
    printf("Execute: calling function\n");
    fflush(stdout);
    void (*func)(const void*, const void*, int) = (void (*)(const void*, const void*, int)) real_addr;
    func(args.values, args.type_codes, args.num_args);
    printf("Execute: called function\n");
    fflush(stdout);
  }

  void x86MicroDeviceAPI::Reset(TVMContext ctx) {
  }

} // namespace runtime
} // namespace tvm
