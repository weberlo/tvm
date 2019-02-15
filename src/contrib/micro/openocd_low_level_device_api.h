/*!
 *  Copyright (c) 2018 by Contributors
 * \file OpenOCD_micro_device_api.h
 * \brief  kernels
 */
#ifndef TVM_RUNTIME_OPENOCD_LOW_LEVEL_DEVICE_API_H_
#define TVM_RUNTIME_OPENOCD_LOW_LEVEL_DEVICE_API_H_

#include <memory>
#include <vector>
#include <string>
#include <sys/mman.h>
#include <mutex>
#include <tvm/runtime/low_level_device_api.h>
#include <tvm/runtime/module.h>
#include "allocator_stream.h"
#include "tcl_socket.h"

namespace tvm {
namespace runtime {

class OpenOCDLowLevelDeviceAPI final : public LowLevelDeviceAPI {
  public:
    OpenOCDLowLevelDeviceAPI(size_t num_bytes);

    ~OpenOCDLowLevelDeviceAPI();

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

    static std::shared_ptr<OpenOCDLowLevelDeviceAPI> Create(size_t num_bytes);
    static std::shared_ptr<OpenOCDLowLevelDeviceAPI> Get(int table_index);

    // TODO: Make this private.
    // Returns the reply.
    std::string SendCommand(std::string cmd);

    uint8_t* base_addr;

  private:
    size_t size;
    AllocatorStream* stream;
    std::string args_buf;
    int table_index_{0};

    tvm::common::TclSocket socket;

    // TODO: Just make this a `char`.
    static const constexpr char *kCommandTerminateToken = "\x1a";
    static const constexpr size_t kSendBufSize = 4096;
    static const constexpr size_t kReplyBufSize = 4096;
    static const constexpr ssize_t kWordLen = 8;

    inline void* GetOffset(uint8_t* real_addr) {
      return (void*) (real_addr - base_addr);
    }

    inline uint8_t* GetRealAddr(void* offset) {
      return base_addr + reinterpret_cast<std::uintptr_t>(offset);
    }

    // Returns the reply.
    //std::string SendCommand(std::string cmd);
};

struct OpenOCDLowLevelDevTable {
 public:
  static constexpr int kMaxLowLevelDevice = 1;
  // Get global singleton
  static OpenOCDLowLevelDevTable* Global() {
    static OpenOCDLowLevelDevTable inst;
    return &inst;
  }
  // Get session from table
  std::shared_ptr<OpenOCDLowLevelDeviceAPI> Get(int index) {
    CHECK(index >= 0 && index < kMaxLowLevelDevice);
    return tbl_[index].lock();
  }
  // Insert session into table.
  int Insert(std::shared_ptr<OpenOCDLowLevelDeviceAPI> ptr) {
    std::lock_guard<std::mutex> lock(mutex_);
    for (int i = 0; i < kMaxLowLevelDevice; ++i) {
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
  std::array<std::weak_ptr<OpenOCDLowLevelDeviceAPI>, kMaxLowLevelDevice> tbl_;
};

inline std::shared_ptr<OpenOCDLowLevelDeviceAPI> OpenOCDLowLevelDeviceAPI::Create(size_t num_bytes) {
  std::shared_ptr<OpenOCDLowLevelDeviceAPI> micro_dev =
    std::make_shared<OpenOCDLowLevelDeviceAPI>(num_bytes);
  micro_dev->table_index_ = OpenOCDLowLevelDevTable::Global()->Insert(micro_dev);
  return micro_dev;
}

inline std::shared_ptr<OpenOCDLowLevelDeviceAPI> OpenOCDLowLevelDeviceAPI::Get(int table_index) {
  return OpenOCDLowLevelDevTable::Global()->Get(table_index);
}

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_OPENOCD_LOW_LEVEL_DEVICE_API_H_
