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
// Hex Conversion
#include <iomanip>

#include "openocd_low_level_device_api.h"

namespace tvm {
namespace runtime {

OpenOCDLowLevelDeviceAPI::OpenOCDLowLevelDeviceAPI(size_t num_bytes)
  : size(num_bytes) {
    socket.Create();
    socket.SetKeepAlive(true);
    socket.Connect(tvm::common::SockAddr("127.0.0.1", 6666));
    base_addr = 0x0;
}

OpenOCDLowLevelDeviceAPI::~OpenOCDLowLevelDeviceAPI() { }

void OpenOCDLowLevelDeviceAPI::Write(TVMContext ctx,
                                     void* offset,
                                     uint8_t* buf,
                                     size_t num_bytes) {
  // Clear `input` array.
  SendCommand("array unset input");
  // Build a command to set the value of `input`.
  {
    std::stringstream input_set_cmd;
    input_set_cmd << "array set input { ";
    for (size_t i = 0; i < num_bytes; i++) {
      // TODO: Do the values need to be hex-encoded?
      // In a Tcl `array set` commmand, we need to pair the array indices with
      // their values.
      input_set_cmd << std::dec << i << " ";
      // Need to cast to uint, so the number representation of `buf[i]` is
      // printed, and not the ASCII representation.
      input_set_cmd << "0x" << std::hex << static_cast<unsigned int>(buf[i]) << " ";
    }
    input_set_cmd << "}";
    SendCommand(input_set_cmd.str());
  }
  {
    std::stringstream write_cmd;
    // Set the word length to be 8, so we can send a byte at a time.
    ssize_t word_len = 8;
    write_cmd << "array2mem input";
    write_cmd << " 0x" << std::hex << word_len;
    write_cmd << " " << std::dec << offset;
    write_cmd << " 0x" << std::hex << num_bytes;
    SendCommand(write_cmd.str());
  }
}

void OpenOCDLowLevelDeviceAPI::Read(TVMContext ctx,
                                    void* offset,
                                    uint8_t* buf,
                                    size_t num_bytes) {
  {
    std::stringstream read_cmd;
    // Set the word length to be 8, so we can send a byte at a time.
    ssize_t word_len = 8;
    read_cmd << "mem2array output";
    read_cmd << " " << std::dec << word_len;
    read_cmd << " 0x" << std::hex << offset;
    read_cmd << " " << std::dec << num_bytes;
    SendCommand(read_cmd.str());
  }

  {
    std::string reply = SendCommand("ocd_echo $output");
    // TODO: Figure out why the below is true.
    // The response from this command intersperses unrelated values with the
    // contents of the memory that was read, so we grab every other value.
    std::stringstream values(reply);
    uint8_t *buf_iter = buf;
    ssize_t req_bytes_remaining = num_bytes;
    // TODO: Don't assume the stream has enough values.
    //bool stream_has_values = true;
    while (req_bytes_remaining > 0) {
      int garbage;
      values >> garbage;
      values >> *buf_iter;
      buf_iter++;
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
  SendCommand("reset run");
}

void OpenOCDLowLevelDeviceAPI::Reset(TVMContext ctx) {
  SendCommand("reset halt");
}

std::string OpenOCDLowLevelDeviceAPI::SendCommand(std::string cmd) {
  std::stringstream cmd_builder;
  cmd_builder << cmd;
  cmd_builder << kCommandTerminateToken;
  std::string full_cmd = cmd_builder.str();
  //std::cout << "SEND: " << full_cmd << std::endl;
  socket.Send(full_cmd.data(), full_cmd.length());

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
    std::cerr << "missing command terminator in OpenOCD response" << std::endl;
  }
  //std::cout << "RECV: " << reply << std::endl;

  //std::cout << "WAITING..." << std::endl;
  //std::string temp;
  //std::getline(std::cin, temp);
  return reply;
}

} // namespace runtime
} // namespace tvm
