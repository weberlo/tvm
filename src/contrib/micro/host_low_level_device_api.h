/*!
 *  Copyright (c) 2018 by Contributors
 * \file host_low_level_device_api.h
 * \brief low-level host device
 */
#ifndef TVM_RUNTIME_HOST_LOW_LEVEL_DEVICE_API_H_
#define TVM_RUNTIME_HOST_LOW_LEVEL_DEVICE_API_H_

#include <tvm/runtime/low_level_device_api.h>
#include <tvm/runtime/module.h>
#include <memory>
#include <vector>
#include <string>
#include <sys/mman.h>
#include "allocator_stream.h"
#include <mutex>

namespace tvm {
namespace runtime {

struct HostLowLevelDevTable;

class HostLowLevelDeviceAPI final : public LowLevelDeviceAPI {
  public:
    HostLowLevelDeviceAPI(size_t num_bytes); 
    
    ~HostLowLevelDeviceAPI();

    void Write(TVMContext ctx,
               void* offset,
               uint8_t* buf,
               size_t num_bytes) final; 

    void Read(TVMContext ctx,
              void* offset,
              uint8_t* buf,
              size_t num_bytes) final; 

    void Execute(TVMContext ctx, TVMArgs args, TVMRetValue *rv, void* offset) final;

    void Reset(TVMContext ctx) final;

    static std::shared_ptr<HostLowLevelDeviceAPI> Create(size_t num_bytes);
    static std::shared_ptr<HostLowLevelDeviceAPI> Get(int table_index);

    uint8_t* base_addr;

  private:
    size_t size;
    AllocatorStream* stream;
    std::string args_buf;
    int table_index_{0};

    inline void* GetOffset(uint8_t* real_addr) {
      return (void*) (real_addr - base_addr);
    }

    inline uint8_t* GetRealAddr(void* offset) {
      return base_addr + reinterpret_cast<std::uintptr_t>(offset);
    }

    inline void Shutdown() {
      munmap(base_addr, size);
    }
};

struct HostLowLevelDevTable {
 public:
  static constexpr int kMaxMicroDev = 1;
  // Get global singleton
  static HostLowLevelDevTable* Global() {
    static HostLowLevelDevTable inst;
    return &inst;
  }
  // Get session from table
  std::shared_ptr<HostLowLevelDeviceAPI> Get(int index) {
    CHECK(index >= 0 && index < kMaxMicroDev);
    return tbl_[index].lock();
  }
  // Insert session into table.
  int Insert(std::shared_ptr<HostLowLevelDeviceAPI> ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (int i = 0; i < kMaxMicroDev; ++i) {
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
  std::array<std::weak_ptr<HostLowLevelDeviceAPI>, kMaxMicroDev> tbl_;
};

  inline std::shared_ptr<HostLowLevelDeviceAPI> HostLowLevelDeviceAPI::Create(size_t num_bytes) {
    std::shared_ptr<HostLowLevelDeviceAPI> micro_dev = 
      std::make_shared<HostLowLevelDeviceAPI>(num_bytes);
    micro_dev->table_index_ = HostLowLevelDevTable::Global()->Insert(micro_dev);
    return micro_dev;
  }

  inline std::shared_ptr<HostLowLevelDeviceAPI> HostLowLevelDeviceAPI::Get(int table_index) {
    return HostLowLevelDevTable::Global()->Get(table_index);
  }

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_HOST_LOW_LEVEL_DEVICE_API_H_
