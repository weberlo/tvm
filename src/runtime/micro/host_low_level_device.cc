/*!
 *  Copyright (c) 2019 by Contributors
 * \file host_low_level_device.cc
 * \brief emulated low-level micro device implementation on host machine
 */

#include <sys/mman.h>
#include <cstring>
#include "low_level_device.h"
#include "micro_common.h"

namespace tvm {
namespace runtime {
/*!
 * \brief emulated low-level device on host machine
 */
class HostLowLevelDevice final : public LowLevelDevice {
 public:
  /*!
   * \brief constructor to initialize on-host memory region to act as device
   * \param num_bytes size of the emulated on-device memory region
   */
  HostLowLevelDevice(size_t num_bytes)
    : size_(num_bytes) {
    size_t size_in_pages = (num_bytes + kPageSize - 1) / kPageSize;
    int mmap_prot = PROT_READ | PROT_WRITE | PROT_EXEC;
    int mmap_flags = MAP_ANONYMOUS | MAP_PRIVATE;
    base_addr_ = mmap(nullptr, size_in_pages * kPageSize,
                      mmap_prot, mmap_flags, -1, 0);
  }

  /*!
   * \brief destructor to deallocate on-host device region
   */
  ~HostLowLevelDevice() {
    munmap(base_addr_, size_);
  }

  void Write(void* offset,
             void* buf,
             size_t num_bytes) final {
    void* addr = GetAddr(offset, base_addr_);
    std::memcpy(addr, buf, num_bytes);
  }

  void Read(void* offset,
            void* buf,
            size_t num_bytes) final {
    void* addr = GetAddr(offset, base_addr_);
    std::memcpy(buf, addr, num_bytes);
  }

  void Execute(void* func_addr, void* breakpoint) final {
    void (*func)(void) = (void (*)(void)) func_addr;
    func();
  }

  const void* base_addr() const final {
    return base_addr_;
  }

 private:
  /*! \brief base address of the micro device memory region */
  void* base_addr_;
  /*! \brief size of memory region */
  size_t size_;
};

const std::shared_ptr<LowLevelDevice> HostLowLevelDeviceCreate(size_t num_bytes) {
  std::shared_ptr<LowLevelDevice> lld = 
      std::make_shared<HostLowLevelDevice>(num_bytes);
  return lld;
}
} // namespace runtime
} // namespace tvm
