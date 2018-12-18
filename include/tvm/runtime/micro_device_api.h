/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/runtime/micro_device_api.h
 * \brief Abstract micro device memory management API
 */
#ifndef TVM_RUNTIME_MICRO_DEVICE_API_H_
#define TVM_RUNTIME_MICRO_DEVICE_API_H_

#include <cstring>
#include <mutex>
#include "packed_func.h"
#include "c_runtime_api.h"
#define PAGE_SIZE 4096

namespace tvm {
namespace runtime {

class MicroDeviceAPI {
  public:
  /*! \brief virtual destructor */
  virtual ~MicroDeviceAPI() {}

  // ignore TVMContext for now, it's not useful
  virtual void WriteToMemory(TVMContext ctx, void* offset, uint8_t* buf, size_t num_bytes) = 0;

  virtual void ReadFromMemory(TVMContext ctx, void* offset, uint8_t* buf, size_t num_bytes) = 0;

  virtual void ChangeMemoryProtection(TVMContext ctx, void* offset, int prot, size_t num_bytes) = 0;

  virtual void Execute(TVMContext ctx, TVMArgs args, TVMRetValue *rv, void* offset) = 0;

  virtual void Reset(TVMContext ctx) = 0;

  static std::shared_ptr<MicroDeviceAPI> Get(int table_index) {
    printf("static get called\n");
  }

  static std::shared_ptr<MicroDeviceAPI> Create(size_t num_bytes) {
    printf("static create called\n");
  }

  protected:
  // TODO: figure out endianness and mem alignment
  int endianness;
  size_t size;
};

struct MicroDevTable {
 public:
  static constexpr int kMaxMicroDevice = 1;
  // Get global singleton
  static MicroDevTable* Global() {
    static MicroDevTable inst;
    return &inst;
  }
  // Get session from table
  std::shared_ptr<MicroDeviceAPI> Get(int index) {
    CHECK(index >= 0 && index < kMaxMicroDevice);
    return tbl_[index].lock();
  }
  // Insert session into table.
  int Insert(std::shared_ptr<MicroDeviceAPI> ptr) {
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
  std::array<std::weak_ptr<MicroDeviceAPI>, kMaxMicroDevice> tbl_;
};


} // namespace runtime
} // namespace tvm
#endif // TVM_RUNTIME_MICRO_DEVICE_API_H_
