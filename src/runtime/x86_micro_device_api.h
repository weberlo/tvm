/*!
 *  Copyright (c) 2018 by Contributors
 * \file x86_micro_device_api.h
 * \brief  kernels
 */
#ifndef TVM_RUNTIME_X86_MICRO_DEVICE_API_H_
#define TVM_RUNTIME_X86_MICRO_DEVICE_API_H_

#include <tvm/runtime/micro_device_api.h>
#include <tvm/runtime/module.h>
#include <memory>
#include <vector>
#include <string>
#include <sys/mman.h>
#include "allocator_stream.h"
#include <mutex>

namespace tvm {
namespace runtime {

struct x86MicroDevTable;

class x86MicroDeviceAPI final : public MicroDeviceAPI {
  public:
    x86MicroDeviceAPI(size_t num_bytes); 
    
    ~x86MicroDeviceAPI();

    void WriteToMemory(TVMContext ctx,
                       void* offset,
                       uint8_t* buf,
                       size_t num_bytes) final; 

    void ReadFromMemory(TVMContext ctx,
                        void* offset,
                        uint8_t* buf,
                        size_t num_bytes) final; 

    void ChangeMemoryProtection(TVMContext ctx,
                                void* offset,
                                int prot,
                                size_t num_bytes) final;

    void Execute(TVMContext ctx, TVMArgs args, TVMRetValue *rv, void* offset) final;

    void Reset(TVMContext ctx) final;

    static std::shared_ptr<x86MicroDeviceAPI> Create(size_t num_bytes);
    static std::shared_ptr<x86MicroDeviceAPI> Get(int table_index);

    int x = 992;
    uint8_t* base_addr;
  private:
    size_t size;
    size_t size_in_pages;
    uint8_t* args_section;
    AllocatorStream* stream;
    std::string args_buf;
    int table_index_{0};

    inline void* GetOffset(uint8_t* real_addr) {
      return (void*) (real_addr - base_addr);
    }

    inline uint8_t* GetRealAddr(void* offset) {
      printf("base addr %p\n", base_addr);;
      printf("offset ptr %p\n", offset);
      printf("offset int %d\n", reinterpret_cast<std::uintptr_t>(offset));;
      //printf("offset int %" PRIxPTR "\n", reinterpret_cast<std::uintptr_t>(offset));;
      return base_addr + reinterpret_cast<std::uintptr_t>(offset);
    }

    inline void Shutdown() {
      munmap(base_addr, size);
    }
};

struct x86MicroDevTable {
 public:
  static constexpr int kMaxMicroDevice = 1;
  // Get global singleton
  static x86MicroDevTable* Global() {
    static x86MicroDevTable inst;
    return &inst;
  }
  // Get session from table
  std::shared_ptr<x86MicroDeviceAPI> Get(int index) {
    CHECK(index >= 0 && index < kMaxMicroDevice);
    return tbl_[index].lock();
  }
  // Insert session into table.
  int Insert(std::shared_ptr<x86MicroDeviceAPI> ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (int i = 0; i < kMaxMicroDevice; ++i) {
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
  // If the RPCDevice get released, the pointer session will be released
  std::array<std::weak_ptr<x86MicroDeviceAPI>, kMaxMicroDevice> tbl_;
};

  inline std::shared_ptr<x86MicroDeviceAPI> x86MicroDeviceAPI::Create(size_t num_bytes) {
    std::shared_ptr<x86MicroDeviceAPI> micro_dev = 
      std::make_shared<x86MicroDeviceAPI>(num_bytes);
    micro_dev->table_index_ = x86MicroDevTable::Global()->Insert(micro_dev);
    return micro_dev;
  }

  inline std::shared_ptr<x86MicroDeviceAPI> x86MicroDeviceAPI::Get(int table_index) {
    return x86MicroDevTable::Global()->Get(table_index);
  }

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_X86_MICRO_DEVICE_API_H_
