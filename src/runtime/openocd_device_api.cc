/*!
 *  Copyright (c) 2018 by Contributors
 * \file openocd_device_api.cc
 * \brief openocd device API
 */
#include <dmlc/logging.h>
#include <dmlc/thread_local.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/device_api.h>
#include <cstdlib>
#include <cstring>
#include "workspace_pool.h"

namespace tvm {
namespace runtime {
class OpenOCDDeviceAPI final : public DeviceAPI {
  // TODO: where should the binary/.so loading be done?
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
    void* ptr;
    printf("called allocdataspace\n");
    return ptr;
  }

  void FreeDataSpace(TVMContext ctx, void* ptr) final {
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
};

// TODO: OpenOCDWorkspacePool?
struct OpenOCDWorkspacePool : public WorkspacePool {
  OpenOCDWorkspacePool() :
    // TODO: kDLOpenOCD?
    // TODO: kDLExtDev? was kDLCPU
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
