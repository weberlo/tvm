/*!
 *  Copyright (c) 2018 by Contributors
 * \file micro_device_api.cc
 * \brief MicroDeviceAPI
 */
#include <dmlc/logging.h>
#include <dmlc/thread_local.h>
#include <tvm/runtime/registry.h>
#include <tvm/runtime/device_api.h>
#include <tvm/runtime/low_level_device_api.h>
#include <cstdlib>
#include <cstring>
#include "host_low_level_device_api.h"
#include "device_memory_offsets.h"
#include "../../runtime/workspace_pool.h"

namespace tvm {
namespace runtime {

class MicroDeviceAPI final : public DeviceAPI {
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
    // TODO: is alignment/type_hint necessary? how? with respect to what?
    std::shared_ptr<LowLevelDeviceAPI> md_ = GetMicroDev(ctx);
    CHECK (last_alloc_ + nbytes <= (uint8_t *) MEMORY_SIZE)
      << "out of allocation space\n";
    void* ptr = last_alloc_;
    last_alloc_ = last_alloc_ + nbytes;
    return ptr;
  }

  void FreeDataSpace(TVMContext ctx, void* ptr) final {
    // TODO: implement better allocator with frees
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
    uint8_t buffer[size];
    if (ctx_from.device_type == kDLMicroDev && ctx_to.device_type == kDLMicroDev) {
      std::shared_ptr<LowLevelDeviceAPI> from_md = GetMicroDev(ctx_from);
      std::shared_ptr<LowLevelDeviceAPI> to_md = GetMicroDev(ctx_to);
      from_md->Read(ctx_from, (char*)(from) + from_offset, buffer, size);
      to_md->Write(ctx_to, (char*)(to) + to_offset, buffer, size);
    } else if (ctx_from.device_type == kDLMicroDev && ctx_to.device_type == kDLCPU) {
      std::shared_ptr<LowLevelDeviceAPI> from_md = GetMicroDev(ctx_from);
      from_md->Read(ctx_from, (char*)(from) + from_offset, 
                     buffer, size);
      memcpy(static_cast<char*>(to) + to_offset, buffer, size);

    } else if (ctx_from.device_type  == kDLCPU && ctx_to.device_type == kDLMicroDev) {
      std::shared_ptr<LowLevelDeviceAPI> to_md = GetMicroDev(ctx_to);
      to_md->Write(ctx_to, (char*)(to) + to_offset, 
                    (uint8_t*)(from) + from_offset, size);
    } else {
      LOG(FATAL) << "expect copy from/to Micro or between Micro\n";
    }
  }

  void StreamSync(TVMContext ctx, TVMStreamHandle stream) final {
  }

  void* AllocWorkspace(TVMContext ctx, size_t size, TVMType type_hint) final;
  void FreeWorkspace(TVMContext ctx, void* data) final;

  static const std::shared_ptr<MicroDeviceAPI>& Global() {
    static std::shared_ptr<MicroDeviceAPI> inst =
        std::make_shared<MicroDeviceAPI>();
    return inst;
  }

  private:
  uint8_t* last_alloc_ = (uint8_t *) SECTION_HEAP;

  std::shared_ptr<HostLowLevelDeviceAPI> GetMicroDev(TVMContext ctx) {
    int dev_type = ctx.device_type;
    CHECK_EQ(dev_type, kDLMicroDev);
    int tbl_index = 0;
    // TODO: How to ensure the right type?
    return HostLowLevelDeviceAPI::Get(tbl_index);
  }
};

struct MicroWorkspacePool : public WorkspacePool {
  MicroWorkspacePool() :
    WorkspacePool(kDLMicroDev, MicroDeviceAPI::Global()) {}
};

void* MicroDeviceAPI::AllocWorkspace(TVMContext ctx,
                                   size_t size,
                                   TVMType type_hint) {
  return dmlc::ThreadLocalStore<MicroWorkspacePool>::Get()
      ->AllocWorkspace(ctx, size);
}

void MicroDeviceAPI::FreeWorkspace(TVMContext ctx, void* data) {
  dmlc::ThreadLocalStore<MicroWorkspacePool>::Get()->FreeWorkspace(ctx, data);
}

TVM_REGISTER_GLOBAL("device_api.micro_dev")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    DeviceAPI* ptr = MicroDeviceAPI::Global().get();
    *rv = static_cast<void*>(ptr);
  });
} // namespace runtime
} // namespace tvm
