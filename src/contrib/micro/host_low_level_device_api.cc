/*!
 *  Copyright (c) 2018 by Contributors
 * \file host_low_level_device_api.cc
 * \brief emulated micro device implementation on host machine
 */
#include <sys/mman.h>
#include <iostream>
#include <mutex>
#include <errno.h>
#include <tvm/runtime/c_runtime_api.h>
#include "host_low_level_device_api.h"
#include "allocator_stream.h"
#include "device_memory_offsets.h"

namespace tvm {
namespace runtime {
  HostLowLevelDeviceAPI::HostLowLevelDeviceAPI(size_t num_bytes)
    : size(num_bytes) {
    size = num_bytes;
    size_t size_in_pages = (num_bytes + PAGE_SIZE - 1) / PAGE_SIZE;
    int mmap_prot = PROT_READ | PROT_WRITE | PROT_EXEC;
    int mmap_flags = MAP_ANONYMOUS | MAP_PRIVATE;
    base_addr = (uint8_t*) mmap(NULL, size_in_pages * PAGE_SIZE, mmap_prot, mmap_flags, -1, 0);
    stream = new AllocatorStream(&args_buf);
  }

  HostLowLevelDeviceAPI::~HostLowLevelDeviceAPI() {
    Shutdown();
  }

  void HostLowLevelDeviceAPI::Write(TVMContext ctx,
                     void* offset,
                     uint8_t* buf,
                     size_t num_bytes) {
    uint8_t* real_addr = GetRealAddr(offset);
    std::memcpy(real_addr, buf, num_bytes);
  }

  void HostLowLevelDeviceAPI::Read(TVMContext ctx,
                      void* offset,
                      uint8_t* buf,
                      size_t num_bytes) {
    uint8_t* real_addr = GetRealAddr(offset);
    std::memcpy(buf, real_addr, num_bytes);
  }

  void HostLowLevelDeviceAPI::Execute(TVMContext ctx, TVMArgs args, TVMRetValue *rv, void* offset) {
    // TODO: should args section choice be at the micro level? no
    void* args_section = (void *) SECTION_ARGS;
    WriteTVMArgsToStream(args, stream, base_addr, SECTION_ARGS);
    Write(ctx, args_section, (uint8_t*) args_buf.c_str(), (size_t) stream->GetBufferSize());
    uint8_t* real_addr = GetRealAddr(offset);
    void (*func)(void) = (void (*)(void)) real_addr;
    func();
  }

  // host does not need a reset
  void HostLowLevelDeviceAPI::Reset(TVMContext ctx) {
  }

} // namespace runtime
} // namespace tvm
