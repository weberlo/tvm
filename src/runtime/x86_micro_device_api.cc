/*!
 *  Copyright (c) 2018 by Contributors
 * \file x86_micro_device_api.cc
 * \brief x86-emulated micro device API
 */
#include <tvm/runtime/micro_device_api.h>
#include <sys/mman.h>
#include <mutex>

namespace tvm {
namespace runtime {
class x86MicroDeviceAPI final : public MicroDeviceAPI {
  public:
    x86MicroDeviceAPI(size_t num_bytes) 
      : size(num_bytes) {
      size = num_bytes;
      size_in_pages = (num_bytes + PAGE_SIZE - 1) / PAGE_SIZE;
      int prot = PROT_READ | PROT_WRITE | PROT_EXEC;
      int flags = MAP_ANONYMOUS;
      base_addr = (uint8_t*) mmap(NULL, size, prot, flags, -1, 0);
    }

    ~x86MicroDeviceAPI() {
      Shutdown();
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
      // needs to be page aligned
      // we assume all memory is executable for now, so this isn't called
      uint8_t* real_addr = GetRealAddr(offset);
      mprotect(real_addr, num_bytes, prot);
    }

    void Execute(TVMContext ctx, void* offset) final {
      // TODO: need to call init stub with this addr after copying args?
      // args need to be in binary mode, readable for function
      uint8_t* real_addr = GetRealAddr(offset);
      void (*func)(void) = (void (*)(void)) real_addr;
      func();
    }

    void Reset(TVMContext ctx) final {
    }

    static std::shared_ptr<MicroDeviceAPI> Create(size_t num_bytes);

    static std::shared_ptr<MicroDeviceAPI> Get(int table_index);

  private:
    size_t size;
    size_t size_in_pages;
    uint8_t* base_addr;

    void* GetOffset(uint8_t* real_addr) {
      return (void*) (real_addr - base_addr);
    }

    uint8_t* GetRealAddr(void* offset) {
      return base_addr + reinterpret_cast<std::uintptr_t>(offset);
    }

    void Shutdown() {
      munmap(base_addr, size);
    }
};

std::shared_ptr<MicroDeviceAPI> x86MicroDeviceAPI::Create(size_t num_bytes) {
  std::shared_ptr<x86MicroDeviceAPI> micro_dev = 
    std::make_shared<x86MicroDeviceAPI>(num_bytes);
  micro_dev->table_index_ = MicroDevTable::Global()->Insert(micro_dev);
  return micro_dev;
}

std::shared_ptr<MicroDeviceAPI> x86MicroDeviceAPI::Get(int table_index) {
  return MicroDevTable::Global()->Get(table_index);
}

} // namespace runtime
} // namespace tvm
