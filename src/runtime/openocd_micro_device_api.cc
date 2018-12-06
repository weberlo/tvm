/*!
 *  Copyright (c) 2018 by Contributors
 * \file x86_micro_device_api.cc
 * \brief x86-emulated micro device API
 */
#include <tvm/runtime/micro_device_api.h>
#include <sys/mman.h>

namespace tvm {
namespace runtime {
class OpenOCDMicroDeviceAPI final : public MicroDeviceAPI {
  public:
    OpenOCDMicroDeviceAPI(size_t num_bytes) {
      size = num_bytes;
      size_in_pages = (num_bytes + PAGE_SIZE - 1) / PAGE_SIZE;
      int prot = PROT_READ | PROT_WRITE | PROT_EXEC;
      int flags = MAP_ANONYMOUS;
      base_addr = (uint8_t*) mmap(NULL, size, prot, flags, -1, 0);
      // TODO: init socket connection to openocd server, and use that memory
    }

    void WriteToMemory(TVMContext ctx,
                       void* offset,
                       uint8_t* buf,
                       size_t num_bytes) final {
      uint8_t* real_addr = GetRealAddr(offset);
      std::memcpy(real_addr, buf, num_bytes);
    }

    void ReadFromMemory(TVMContext ctx,
                        void* offset,
                        uint8_t* buf,
                        size_t num_bytes) final {
      uint8_t* real_addr = GetRealAddr(offset);
      std::memcpy(buf, real_addr, num_bytes);
    }

    void ChangeMemoryProtection(TVMContext ctx,
                                void* offset,
                                int prot,
                                size_t num_bytes) final {
      // no-op because openocd doesn't do this
    }

    void Execute(TVMContext ctx, void* offset) final {
      // TODO: need to maybe call init stub that calls the correct func
      // args need to be in binary/readable format in specific memory location
      Reset();
      uint8_t* real_addr = GetRealAddr(offset);
      void (*func)(void) = (void (*)(void)) real_addr;
      func();
    }


    void Reset(TVMContext ctx) final {
      // TODO: this seems required in openocd, send reset command to server
    }

  private:
    size_t size;
    size_t size_in_pages;
    // TODO: will the riscv device have a contiguous addr range?
    uint8_t* base_addr;

    void* GetOffset(uint8_t* real_addr) {
      return (void *) (real_addr - base_addr);
    }

    uint8_t* GetRealAddr(void*  offset) {
      return base_addr + reinterpret_cast<std::uintptr_t>(offset);
    }
};

} // namespace runtime
} // namespace tvm
