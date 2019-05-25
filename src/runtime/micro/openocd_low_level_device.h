/*!
 *  Copyright (c) 2019 by Contributors
 * \file openocd_low_level_device.h
 * \brief openocd low-level device to interface with micro devices over JTAG
 */
#ifndef TVM_RUNTIME_MICRO_OPENOCD_LOW_LEVEL_DEVICE_API_H_
#define TVM_RUNTIME_MICRO_OPENOCD_LOW_LEVEL_DEVICE_API_H_

// TODO(weberlo): do we need this all?
// Socket stuff
#include <fcntl.h>
#include <netdb.h>
#include <errno.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/ioctl.h>

#include "low_level_device.h"
#include "tcl_socket.h"

namespace tvm {
namespace runtime {

// TODO(weberlo): Add implementation for this device.
// TODO(weberlo): Docs everywhere

/*!
 * \brief openocd low-level device for uTVM micro devices connected over JTAG
 */
class OpenOCDLowLevelDevice final : public LowLevelDevice {
 public:
  /*!
   * \brief constructor to initialize connection to openocd device
   * \param port port of the OpenOCD server to connect to
   */
  explicit OpenOCDLowLevelDevice(int port);

  /*!
   * \brief destructor to close openocd device connection
   */
  virtual ~OpenOCDLowLevelDevice();

  void Write(DevBaseOffset offset, void* buf, size_t num_bytes) final;

  void Read(DevBaseOffset offset, void* buf, size_t num_bytes) final;

  void Execute(DevBaseOffset func_offset, DevBaseOffset breakpoint) final;

  void SetBreakpoint(DevBaseOffset breakpoint) {
    breakpoint_ = base_addr_ + breakpoint;
    std::cout << "setting breakpoint to 0x" << std::hex << breakpoint_.value() << std::endl;
  }

  void SetStackTop(DevBaseOffset stack_top) {
    stack_top_ = base_addr_ + stack_top;
    std::cout << "setting stack top to 0x" << std::hex << stack_top_.value() << std::endl;
  }

  DevBaseAddr base_addr() const final {
    return base_addr_;
  }

  DevAddr breakpoint() const {
    CHECK(breakpoint_ != nullptr) << "breakpoint was never initialized";
    return breakpoint_;
  }

  DevAddr stack_top() const {
    CHECK(stack_top_ != nullptr) << "stack top was never initialized";
    return stack_top_;
  }

  const char* device_type() const final {
    return "openocd";
  }

 private:
  /*! \brief base address of the micro device memory region */
  DevBaseAddr base_addr_;
  /*! \brief TODO */
  DevAddr breakpoint_;
  /*! \brief TODO */
  DevAddr stack_top_;
  /*! \brief TODO */
  TclSocket socket_;

  static const constexpr ssize_t kWordLen = 8;
};

/*!
 * \brief connect to OpenOCD and create an OpenOCD low-level device
 * \param port port of the OpenOCD server to connect to
 */
const std::shared_ptr<LowLevelDevice> OpenOCDLowLevelDeviceCreate(int port);

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_MICRO_OPENOCD_LOW_LEVEL_DEVICE_API_H_
