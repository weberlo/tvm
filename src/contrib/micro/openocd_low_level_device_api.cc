/*!
 *  Copyright (c) 2018 by Contributors
 * \file openocd_micro_device_api.cc
 * \brief OpenOCD micro device API
 */
#include <sys/mman.h>
#include <sstream>
#include "tcl_socket.h"
#include "openocd_low_level_device_api.h"

namespace tvm {
namespace runtime {

OpenOCDLowLevelDeviceAPI::OpenOCDLowLevelDeviceAPI(size_t num_bytes)
  : size(num_bytes) {
    socket.Bind(tvm::common::SockAddr("127.0.0.1", 6666));
    socket.Create();
    base_addr = 0x0;
}

void OpenOCDLowLevelDeviceAPI::Write(TVMContext ctx,
                                          void* offset,
                                          uint8_t* buf,
                                          size_t num_bytes) {
  std::stringstream cmds;
  cmds << "array unset input";
  cmds << OpenOCDLowLevelDeviceAPI::kCommandToken;
  cmds << "array set input { 0 0x1 1 0x2 2 0x3 3 0x4 4 0x5 }";
  cmds << kCommandToken;
  cmds << "array2mem input 0x20 268501088 5";
  cmds << kCommandToken;
}

void OpenOCDLowLevelDeviceAPI::Read(TVMContext ctx,
                                           void* offset,
                                           uint8_t* buf,
                                           size_t num_bytes) {
  std::stringstream cmds;
  cmds << "mem2array output 32 0x10010060 32";
  cmds << kCommandToken;
  cmds << "ocd_echo $output";
  cmds << kCommandToken;
}

void OpenOCDLowLevelDeviceAPI::Execute(TVMContext ctx,
                                    TVMArgs args,
                                    TVMRetValue *rv,
                                    void* offset) {
  std::stringstream cmds;
  cmds << "reset run";
  cmds << kCommandToken;
}

void OpenOCDLowLevelDeviceAPI::Reset(TVMContext ctx) {
  std::stringstream cmds;
  cmds << "reset halt";
  cmds << kCommandToken;
}

inline std::shared_ptr<OpenOCDLowLevelDeviceAPI> OpenOCDLowLevelDeviceAPI::Create(size_t num_bytes) {
  std::shared_ptr<OpenOCDLowLevelDeviceAPI> micro_dev =
    std::make_shared<OpenOCDLowLevelDeviceAPI>(num_bytes);
  micro_dev->table_index_ = OpenOCDLowLevelDevTable::Global()->Insert(micro_dev);
  return micro_dev;
}

inline std::shared_ptr<OpenOCDLowLevelDeviceAPI> OpenOCDLowLevelDeviceAPI::Get(int table_index) {
  return OpenOCDLowLevelDevTable::Global()->Get(table_index);
}

} // namespace runtime
} // namespace tvm

