/*!
 *  Copyright (c) 2018 by Contributors
 * \file openocd_micro_device_api.cc
 * \brief OpenOCD micro device API
 */
#include <sys/mman.h>
#include <sstream>

// Socket stuff
#include <fcntl.h>
#include <netdb.h>
#include <errno.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
// IO
#include<iostream>
#include<iomanip>
// Thread Sleeping
#include <unistd.h>

#include "openocd_low_level_device_api.h"
#include "device_memory_offsets.h"

namespace tvm {
namespace runtime {

// TODO: DON'T DUPLICATE THESE METHODS FROM `micro_module.cc`.

FILE* ExecuteCommandWithOutput(std::string cmd) {
  FILE* f;
  f = popen(cmd.c_str(), "r");
  if (f == NULL) {
    CHECK(false)
      << "error in popen " + cmd + "\n";
  }
  return f;
}

void* OpenOCDLowLevelDeviceAPI::GetSymbol(const char* name) {
  uint8_t* addr;
  // Use `nm` with the `-C` option to demangle symbols before grepping.
  // TODO: NO HARDCODE
  std::string cmd = std::string("riscv64-unknown-elf-nm -C fadd.obj.bin | grep -w ") + std::string(name);
  FILE* f = ExecuteCommandWithOutput(cmd);
  if (!fscanf(f, "%p", &addr)) {
    addr = nullptr;
    std::cerr << "Could not find address for symbol \"" << name << "\"" << std::endl;
  }
  return (void*)(addr - base_addr);
}

OpenOCDLowLevelDeviceAPI::OpenOCDLowLevelDeviceAPI(size_t num_bytes)
  : size(num_bytes) {
    socket.Create();
    socket.SetKeepAlive(true);
    socket.Connect(tvm::common::SockAddr("127.0.0.1", 6666));
    // TODO: NO HARDCODE.
    base_addr = (uint8_t*) 0x10010000;
    args_offs = reinterpret_cast<uintptr_t>(GetSymbol("args_section"));
    stream = new AllocatorStream(&args_buf);
}

OpenOCDLowLevelDeviceAPI::~OpenOCDLowLevelDeviceAPI() {
  // socket.Close();
}

void OpenOCDLowLevelDeviceAPI::Write(TVMContext ctx,
                                     void* offset,
                                     uint8_t* buf,
                                     size_t num_bytes) {
  uint8_t* real_addr = GetRealAddr(offset);
  // Clear `input` array.
  SendCommand("array unset input");
  // Build a command to set the value of `input`.
  {
    std::stringstream input_set_cmd;
    input_set_cmd << "array set input { ";
    for (size_t i = 0; i < num_bytes; i++) {
      // In a Tcl `array set` commmand, we need to pair the array indices with
      // their values.
      input_set_cmd << i << " ";
      // Need to cast to uint, so the number representation of `buf[i]` is
      // printed, and not the ASCII representation.
      input_set_cmd << static_cast<unsigned int>(buf[i]) << " ";
    }
    input_set_cmd << "}";
    SendCommand(input_set_cmd.str());
  }
  {
    std::stringstream write_cmd;
    write_cmd << "array2mem input";
    write_cmd << " " << kWordLen;
    write_cmd << " " << reinterpret_cast<uint64_t>(real_addr);
    write_cmd << " " << num_bytes;
    SendCommand(write_cmd.str());
  }
}

void OpenOCDLowLevelDeviceAPI::Read(TVMContext ctx,
                                    void* offset,
                                    uint8_t* buf,
                                    size_t num_bytes) {
  uint8_t* real_addr = GetRealAddr(offset);
  {
    std::stringstream read_cmd;
    read_cmd << "mem2array output";
    read_cmd << " " << kWordLen;
    read_cmd << " " << reinterpret_cast<uint64_t>(real_addr);
    read_cmd << " " << num_bytes;
    SendCommand(read_cmd.str());
  }

  {
    std::string reply = SendCommand("ocd_echo $output");
    std::stringstream values(reply);
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
      buf[index] = static_cast<uint8_t>(val);
      req_bytes_remaining--;
    }
    /*
    ssize_t extra_recv_bytes = 0;
    while (stream_has_values) {
      int garbage;
      values >> garbage;
      stream_has_values = (values >> garbage);
      extra_recv_bytes++;
    }
    */
    /*
    if (extra_recv_bytes > 0) {
      std::cerr << std::dec << extra_recv_bytes << " more bytes received in "
        "response than were requested" << std::endl;
      //std::cerr << "`buf`: ";
      //for (uint8_t *buf_iter = buf; buf_iter < buf + num_bytes; buf_iter++) {
      //  std::cerr << " " << std::hex << static_cast<unsigned int>(*buf_iter);
      //}
      //std::cerr << std::endl;
      //std::cerr << "`reply`: " << reply << std::endl;
    } else if (req_bytes_remaining > 0) {
      std::cerr << std::dec << req_bytes_remaining << " fewer bytes received in "
        "response than were requested" << std::endl;
    }
    */
  }
}

void OpenOCDLowLevelDeviceAPI::Execute(TVMContext ctx,
                                       TVMArgs args,
                                       TVMRetValue *rv,
                                       void* offset) {
  void* args_section = (void *) args_offs;
  WriteTVMArgsToStream(args, stream, base_addr, args_offs);
  Write(ctx, args_section, (uint8_t*) args_buf.c_str(), (size_t) stream->GetBufferSize());

  uint8_t* real_addr = GetRealAddr(offset);

  SendCommand("reset halt", true);

  // Set up the stack pointer.  We need to do this every time, because `reset
  // halt` wipes it out.
  uintptr_t stack_addr = reinterpret_cast<uintptr_t>(base_addr) +
    reinterpret_cast<uintptr_t>(GetSymbol("stack_section")) + 64000 - 8;
  std::stringstream sp_set_cmd;
  sp_set_cmd << "reg sp 0x" << std::hex << stack_addr;
  SendCommand(sp_set_cmd.str(), true);

  // Set a breakpoint at the end of `main`.
  uintptr_t done_bp_addr = reinterpret_cast<uintptr_t>(base_addr) +
    reinterpret_cast<uintptr_t>(GetSymbol("main")) + 0xe;
  std::stringstream bp_cmd;
  bp_cmd << "bp 0x" << std::hex << done_bp_addr << " 2";
  SendCommand(bp_cmd.str(), true);

  std::stringstream resume_cmd;
  resume_cmd << "resume 0x" << std::hex << reinterpret_cast<uint64_t>(real_addr);
  SendCommand(resume_cmd.str(), true);
  // SendCommand("resume", true);

  // std::string halted = "halted";
  // std::string reply;
  // while (reply.compare(0, halted.length(), halted) != 0) {
  //   // for (int32_t i = 0; i < 10; i++) {
  //   //   usleep(30000);
  //   // }
  //   usleep(1000000);
  //   reply = SendCommand("riscv.cpu curstate", true);
  // }

  // usleep(1000000);
  // SendCommand("riscv.cpu curstate", true);
  // usleep(1000000);
  SendCommand("halt 0", true);

  // SendCommand("ocd_reg sp", true);
  // SendCommand("ocd_reg pc", true);
}

void OpenOCDLowLevelDeviceAPI::Reset(TVMContext ctx) {
  SendCommand("reset halt");
}

std::string OpenOCDLowLevelDeviceAPI::SendCommand(std::string cmd, bool verbose) {
  std::stringstream cmd_builder;
  cmd_builder << cmd;
  cmd_builder << kCommandTerminateToken;
  std::string full_cmd = cmd_builder.str();
  socket.Send(full_cmd.data(), full_cmd.length());
  if (verbose)
    std::cout << "SEND: " << full_cmd << std::endl;

  std::stringstream reply_builder;
  static char reply_buf[kReplyBufSize];
  ssize_t bytes_read;
  do {
    // Leave room at the end of `reply_buf` to tack on a null terminator.
    bytes_read = socket.Recv(reply_buf, kReplyBufSize - 1);
    reply_buf[bytes_read] = '\0';
    reply_builder << reply_buf;
  } while (bytes_read == kReplyBufSize - 1);
  // -1 signals an error from `socket`.
  if (bytes_read == -1) {
    std::cerr << "error reading OpenOCD response" << std::endl;
  }
  std::string reply = reply_builder.str();
  // Make sure the reply ends in a command terminator.
  if (reply[reply.length()-1] != kCommandTerminateToken[0]) {
    std::cerr << "missing command terminator in OpenOCD response \"" << reply << "\" to command \"" << cmd << "\"" << std::endl;
  }

  if (verbose)
    std::cout << "RECV: " << reply << std::endl;

  return reply;
}

} // namespace runtime
} // namespace tvm
