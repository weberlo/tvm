/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2019 by Contributors
 * \file openocd_low_level_device.cc
 * \brief openocd low-level device to interface with micro devices over JTAG
 */

#include "openocd_low_level_device.h"

// IO
#include<iostream>
#include<iomanip>
// Thread Sleeping
#include <unistd.h>

#include "micro_common.h"

namespace tvm {
namespace runtime {

// TODO(weberlo): Can we query the device for how much memory it has?

OpenOCDLowLevelDevice::OpenOCDLowLevelDevice(int port)
    : socket_() {
  socket_.Connect(tvm::common::SockAddr("127.0.0.1", port));
  socket_.SendCommand("reset halt");
  // TODO: NO HARDCODE.
  base_addr_ = DevBaseAddr(0x10010000);
}

OpenOCDLowLevelDevice::~OpenOCDLowLevelDevice() {
  // socket_.Close();
}

void OpenOCDLowLevelDevice::Read(DevBaseOffset offset, void* buf, size_t num_bytes) {
  if (num_bytes == 0) {
    return;
  }
  {
    DevAddr addr = base_addr_ + offset;
    std::stringstream read_cmd;
    read_cmd << "mem2array output";
    read_cmd << " " << std::dec << kWordLen;
    read_cmd << " 0x" << std::hex << addr.cast_to<uintptr_t>();
    read_cmd << " " << std::dec << num_bytes;
    socket_.SendCommand(read_cmd.str());
  }

  {
    std::string reply = socket_.SendCommand("ocd_echo $output");
    std::stringstream values(reply);
    char* char_buf = reinterpret_cast<char*>(buf);
    ssize_t req_bytes_remaining = num_bytes;
    // TODO: Don't assume the stream has enough values.
    //bool stream_has_values = true;
    while (req_bytes_remaining > 0) {
      // The response from this command pairs indices with the contents of the
      // memory at that index.
      uint32_t index;
      uint32_t val;
      values >> index;
      // Read the value into `curr_val`, instead of reading directly into
      // `buf_iter`, because otherwise it's interpreted as the ASCII value and
      // not the integral value.
      values >> val;
      char_buf[index] = static_cast<uint8_t>(val);
      req_bytes_remaining--;
    }
  }
}

void OpenOCDLowLevelDevice::Write(DevBaseOffset offset, void* buf, size_t num_bytes) {
  if (num_bytes == 0) {
    return;
  }
  // Clear `input` array.
  socket_.SendCommand("array unset input");
  // Build a command to set the value of `input`.
  {
    std::stringstream input_set_cmd;
    input_set_cmd << "array set input { ";
    char* char_buf = reinterpret_cast<char*>(buf);
    for (size_t i = 0; i < num_bytes; i++) {
      // In a Tcl `array set` commmand, we need to pair the array indices with
      // their values.
      input_set_cmd << i << " ";
      // Need to cast to uint, so the number representation of `buf[i]` is
      // printed, and not the ASCII representation.
      input_set_cmd << static_cast<uint32_t>(char_buf[i]) << " ";
    }
    input_set_cmd << "}";
    socket_.SendCommand(input_set_cmd.str());
  }
  {
    DevAddr addr = base_addr_ + offset;
    std::stringstream write_cmd;
    write_cmd << "array2mem input";
    write_cmd << " " << std::dec << kWordLen;
    write_cmd << " 0x" << std::hex << addr.cast_to<uintptr_t>();
    write_cmd << " " << std::dec << num_bytes;
    socket_.SendCommand(write_cmd.str());
  }
}

void OpenOCDLowLevelDevice::Execute(DevBaseOffset func_offset, DevBaseOffset useless) {
  socket_.SendCommand("halt 0", true);

  // Set up the stack pointer.  We need to do this every time, because `reset
  // halt` wipes it out.
  DevAddr stack_end = stack_top() - 8;
  std::stringstream sp_set_cmd;
  sp_set_cmd << "reg sp 0x" << std::hex << stack_end.cast_to<uintptr_t>();
  socket_.SendCommand(sp_set_cmd.str(), true);

  // Set a breakpoint at the beginning of `UTVMDone`.
  std::stringstream bp_set_cmd;
  bp_set_cmd << "bp 0x" << std::hex << breakpoint().cast_to<uintptr_t>() << " 2";
  socket_.SendCommand(bp_set_cmd.str(), true);

  char tmp;
  std::cout << "[PRESS ENTER TO CONTINUE]";
  std::cin >> tmp;

  // DevAddr func_addr = base_addr_ + func_offset;
  // std::stringstream resume_cmd;
  // resume_cmd << "resume 0x" << std::hex << func_addr.cast_to<uintptr_t>();
  // socket_.SendCommand(resume_cmd.str(), true);

  // // size_t millis_to_wait = 10 * 1000;
  // // std::stringstream wait_halt_cmd;
  // // resume_cmd << "wait_halt " << std::dec << millis_to_wait;
  // // socket_.SendCommand(wait_halt_cmd.str(), true);
  // socket_.SendCommand("wait_halt 10000", true);

  // socket_.SendCommand("halt 0", true);

  // Remove the breakpoint.
  std::stringstream bp_rm_cmd;
  bp_rm_cmd << "rbp 0x" << std::hex << breakpoint().cast_to<uintptr_t>();
  socket_.SendCommand(bp_rm_cmd.str(), true);
}

const std::shared_ptr<LowLevelDevice> OpenOCDLowLevelDeviceCreate(int port) {
  std::shared_ptr<LowLevelDevice> lld =
      std::make_shared<OpenOCDLowLevelDevice>(port);
  return lld;
}

}  // namespace runtime
}  // namespace tvm
