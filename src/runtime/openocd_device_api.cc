/*!
 *  Copyright (c) 2018 by Contributors
 * \file openocd_device_api.cc
 * \brief openocd device API
 */
#include <dmlc/logging.h>
#include <dmlc/thread_local.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/micro_device_api.h>
#include <cstdlib>
#include <cstring>
#include "workspace_pool.h"

namespace tvm {
namespace runtime {

class OpenOCDDeviceAPI final : public DeviceAPI {
  public:
  void SetDevice(TVMContext ctx) final {}
  void GetAttr(TVMContext ctx, DeviceAttrKind kind, TVMRetValue* rv) final {
    if (kind == kExist) {
      *rv = 1;
    }
  }

  void* AllocDataSpace(TVMContext ctx,
                       size_t nbytes,
                       size_t alignment,
                       TVMType type_hint) final {
    // emulates silly heap section by incrementing last_alloc_ pointer
    // TODO: how to get microdevice object from OpenOCDModule for better allocation?
    // is alignment/type_hint necessary? how? with respect to what?
    printf("called allocdataspace\n");
    CHECK (last_alloc_ + nbytes <= (uint8_t *)(50 * PAGE_SIZE))
      << "out of allocation space\n";
    void* ptr = last_alloc_;
    last_alloc_ = last_alloc_ + nbytes;
    return ptr;
  }

  void FreeDataSpace(TVMContext ctx, void* ptr) final {
    // TODO: make a better allocator to deal with frees, for now it should be ok
    printf("called freedataspace\n");
  }

  void CopyDataFromTo(const void* from,
                      size_t from_offset,
                      void* to,
                      size_t to_offset,
                      size_t size,
                      TVMContext ctx_from,
                      TVMContext ctx_to,
                      TVMType type_hint,
                      TVMStreamHandle stream) final {
    // TODO: how exactly does this work? cpu_device_api just does memcpy, but what f it had to copy to the riscv board? are contexts single device only?
    printf("called copydatafromto\n");
  }

  void StreamSync(TVMContext ctx, TVMStreamHandle stream) final {
    printf("called streamsync\n");
  }

  void* AllocWorkspace(TVMContext ctx, size_t size, TVMType type_hint) final;
  void FreeWorkspace(TVMContext ctx, void* data) final;

  static const std::shared_ptr<OpenOCDDeviceAPI>& Global() {
    static std::shared_ptr<OpenOCDDeviceAPI> inst =
        std::make_shared<OpenOCDDeviceAPI>();
    return inst;
  }

  private:
  // TODO: make dynamic heap sizes
  uint8_t* last_alloc_ = (uint8_t *) 40 * PAGE_SIZE;
};

struct OpenOCDWorkspacePool : public WorkspacePool {
  OpenOCDWorkspacePool() :
    // TODO: kDLOpenOCD or kDLExtDev? was originally kDLCPU
    WorkspacePool(kDLExtDev, OpenOCDDeviceAPI::Global()) {}
};

void* OpenOCDDeviceAPI::AllocWorkspace(TVMContext ctx,
                                   size_t size,
                                   TVMType type_hint) {
  return dmlc::ThreadLocalStore<OpenOCDWorkspacePool>::Get()
      ->AllocWorkspace(ctx, size);
}

void OpenOCDDeviceAPI::FreeWorkspace(TVMContext ctx, void* data) {
  dmlc::ThreadLocalStore<OpenOCDWorkspacePool>::Get()->FreeWorkspace(ctx, data);
}

TVM_REGISTER_GLOBAL("device_api.openocd")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    DeviceAPI* ptr = OpenOCDDeviceAPI::Global().get();
    *rv = static_cast<void*>(ptr);
  });
} // namespace runtime
} // namespace tvm
