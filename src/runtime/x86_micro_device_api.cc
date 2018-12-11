/*!
 *  Copyright (c) 2018 by Contributors
 * \file x86_micro_device_api.cc
 * \brief x86-emulated micro device API
 */
#include <tvm/runtime/micro_device_api.h>
#include <sys/mman.h>
#include <dmlc/memory_io.h>
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
      //int buf_size = 10 * PAGE_SIZE;
      stream = new dmlc::MemoryStringStream(&args_buf);
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

    /* Description of the args section
     * 
     * it will have void* args_location
     */

    void Execute(TVMContext ctx, TVMArgs args, TVMRetValue *rv, void* offset) final {
      // TODO: need to call init stub with this addr after copying args?
      // how will init stub know which function to call?
      // TODO: use MemoryIO or Stream from dmlc_core to write args in binary
      // TODO: should args section choice be at the micro level?
      // CopyArgs(args, rv);
      stream->Write((void*) &args, (size_t) 10 * PAGE_SIZE); // find correct/exact size
      void* args_section = (void *)(30 * PAGE_SIZE);
      WriteToMemory(ctx, args_section, (uint8_t*) &args_buf[0], (size_t) 10 * PAGE_SIZE);
      uint8_t* real_addr = GetRealAddr(offset);
      // This should be the function signature if it's to know where things are
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
    uint8_t* args_section;
    dmlc::MemoryStringStream* stream;
    std::string args_buf;

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
