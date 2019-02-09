/*!
 *  Copyright (c) 2018 by Contributors
 * \file low_level_device_api.h
 * \brief Abstract low-level device management API
 */
#ifndef TVM_RUNTIME_LOW_LEVEL_DEVICE_API_H_
#define TVM_RUNTIME_LOW_LEVEL_DEVICE_API_H_

#include <cstring>
#include <mutex>
#include "packed_func.h"
#include "c_runtime_api.h"

namespace tvm {
namespace runtime {

class LowLevelDeviceAPI {
  public:
  /*! \brief virtual destructor */
  virtual ~LowLevelDeviceAPI() {}

  virtual void Write(TVMContext ctx, void* offset, uint8_t* buf, size_t num_bytes) = 0;

  virtual void Read(TVMContext ctx, void* offset, uint8_t* buf, size_t num_bytes) = 0;

  virtual void Execute(TVMContext ctx, TVMArgs args, TVMRetValue *rv, void* offset) = 0;

  virtual void Reset(TVMContext ctx) = 0;

  static std::shared_ptr<LowLevelDeviceAPI> Get(int table_index) {
    std::shared_ptr<LowLevelDeviceAPI> ret;
    return ret;
  }

  static std::shared_ptr<LowLevelDeviceAPI> Create(size_t num_bytes) {
    std::shared_ptr<LowLevelDeviceAPI> ret;
    return ret;
  }

  protected:
  // TODO: figure out endianness and memory alignment
  int endianness;
  size_t size;
};

struct LowLevelDeviceTable {
 public:
  static constexpr int kMaxLowLevelDevice = 1;
  // Get global singleton
  static LowLevelDeviceTable* Global() {
    static LowLevelDeviceTable inst;
    return &inst;
  }
  // Get session from table
  std::shared_ptr<LowLevelDeviceAPI> Get(int index) {
    CHECK(index >= 0 && index < kMaxLowLevelDevice);
    return tbl_[index].lock();
  }
  // Insert session into table.
  int Insert(std::shared_ptr<LowLevelDeviceAPI> ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (int i = 0; i < kMaxLowLevelDevice; ++i) {
      if (tbl_[i].lock() == nullptr) {
        tbl_[i] = ptr; return i;
      }
    }
    LOG(FATAL) << "maximum number of low-level device sessions reached";
    return 0;
  }

 private:
  // The mutex
  std::mutex mutex_;
  // If the device gets released, the pointer session will be released
  std::array<std::weak_ptr<LowLevelDeviceAPI>, kMaxLowLevelDevice> tbl_;
};


} // namespace runtime
} // namespace tvm
#endif // TVM_RUNTIME_LOW_LEVEL_DEVICE_API_H_
