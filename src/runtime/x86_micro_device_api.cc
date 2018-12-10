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
    int table_index_{0};

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

struct MicroDevTable {
 public:
  static constexpr int kMaxMicroDevion = 1;
  // Get global singleton
  static MicroDevTable* Global() {
    static MicroDevTable inst;
    return &inst;
  }
  // Get session from table
  // TODO: does this have to be a shared_ptr? reference should work I think
  std::shared_ptr<x86MicroDeviceAPI> Get(int index) {
    CHECK(index >= 0 && index < kMaxMicroDevion);
    return tbl_[index].lock();
  }
  // Insert session into table.
  int Insert(std::shared_ptr<x86MicroDeviceAPI> ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (int i = 0; i < kMaxMicroDevion; ++i) {
      if (tbl_[i].lock() == nullptr) {
        tbl_[i] = ptr; return i;
      }
    }
    LOG(FATAL) << "maximum number of micro session reached";
    return 0;
  }

 private:
  // The mutex
  std::mutex mutex_;
  // Use weak_ptr intentionally
  // If the RPCDevion get released, the pointer session will be released
  std::array<std::weak_ptr<x86MicroDeviceAPI>, kMaxMicroDevion> tbl_;
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
