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

// TODO: implement something like this? (from VTA)
struct DataBuffer {
  /*! \return Virtual address of the data. */
  void* virt_addr() const {
    return data_;
  }
  /*! \return Physical address of the data. */
  uint32_t phy_addr() const {
    return phy_addr_;
  }
  /*!
   * \brief Invalidate the cache of given location in data buffer.
   * \param offset The offset to the data.
   * \param size The size of the data.
   */
  void InvalidateCache(size_t offset, size_t size) {
    if (!kBufferCoherent) {
      VTAInvalidateCache(phy_addr_ + offset, size);
    }
  }
  /*!
   * \brief Invalidate the cache of certain location in data buffer.
   * \param offset The offset to the data.
   * \param size The size of the data.
   */
  void FlushCache(size_t offset, size_t size) {
    if (!kBufferCoherent) {
      VTAFlushCache(phy_addr_ + offset, size);
    }
  }
  /*!
   * \brief Allocate a buffer of a given size.
   * \param size The size of the buffer.
   */
  static DataBuffer* Alloc(size_t size) {
    void* data = VTAMemAlloc(size, 1);
    CHECK(data != nullptr);
    DataBuffer* buffer = new DataBuffer();
    buffer->data_ = data;
    buffer->phy_addr_ = VTAMemGetPhyAddr(data);
    return buffer;
  }
  /*!
   * \brief Free the data buffer.
   * \param buffer The buffer to be freed.
   */
  static void Free(DataBuffer* buffer) {
    VTAMemFree(buffer->data_);
    delete buffer;
  }
  /*!
   * \brief Create data buffer header from buffer ptr.
   * \param buffer The buffer pointer.
   * \return The corresponding data buffer header.
   */
  static DataBuffer* FromHandle(const void* buffer) {
    return const_cast<DataBuffer*>(
        reinterpret_cast<const DataBuffer*>(buffer));
  }

 private:
  /*! \brief The internal data. */
  void* data_;
  /*! \brief The physical address of the buffer, excluding header. */
  uint32_t phy_addr_;
}

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
    // TODO: check VTA, maybe make it similar?
    md_->WriteToMemory();
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

  private:
};

// TODO: OpenOCDWorkspacePool?
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
