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
// Thread Sleeping
#include <unistd.h>

#include "openocd_low_level_device_api.h"
#include "device_memory_offsets.h"

namespace tvm {
namespace runtime {

static void WriteTVMArgsToStream(TVMArgs args, AllocatorStream* stream, void* base_addr) {
  const TVMValue* values = args.values;
  const int* type_codes = args.type_codes;
  int num_args = args.num_args;
  size_t args_offset = stream->Allocate(sizeof(TVMValue*) * num_args
      + sizeof(const int*) * num_args + sizeof(int));
  stream->Seek(args_offset + sizeof(TVMValue*) * num_args);
  stream->Write(type_codes, sizeof(const int*) * num_args);
  stream->Write(&num_args, sizeof(int));
  for (int i = 0; i < num_args; i++) {
    switch(type_codes[i]) {
      case kDLInt:
        printf("was int\n");
        break;
      case kDLUInt:
        printf("was uint\n");
        break;
      case kDLFloat:
        printf("was float\n");
        break;
      case kStr:
        printf("was str\n");
        break;
      case kBytes:
        printf("was bytes\n");
        break;
      case kHandle:
        printf("was handle\n");
        break;
      case kNull:
        printf("was null\n");
        break;
      case kNodeHandle:
        printf("was nodehandle\n");
        break;
      case kArrayHandle:
        printf("was arrayhandle\n");
        break;
      case kTVMType:
        printf("was tvmtype\n");
        break;
      case kTVMContext:
        printf("was tvmctx\n");
        break;
      case kFuncHandle:
        printf("was funchandle\n");
        break;
      case kModuleHandle:
        printf("was modulehandle\n");
        break;
      case kNDArrayContainer:
        {
          TVMArray* tarr = (TVMArray*)(values[i].v_handle);
          size_t tarr_offset = stream->Allocate(sizeof(TVMArray));
          size_t shape_size = 1;
          for (int dim = 0; dim < tarr->ndim; dim++)
            shape_size *= tarr->shape[dim];
          size_t shape_offset = stream->Allocate(sizeof(int64_t) * tarr->ndim);
          stream->Seek(shape_offset);
          stream->Write(tarr->shape, sizeof(int64_t) * tarr->ndim);
          size_t strides_offset = 0;
          if (tarr->strides != NULL) {
            strides_offset = stream->Allocate(sizeof(int64_t) * tarr->ndim);
            stream->Seek(strides_offset);
            stream->Write(tarr->strides, sizeof(int64_t) * tarr->ndim);
          }
          stream->Seek(tarr_offset);
          stream->Write(tarr, sizeof(TVMArray));
          void* data_addr = (uint8_t*) base_addr + reinterpret_cast<std::uintptr_t>(tarr->data) - SECTION_ARGS;
          void* shape_addr = (uint8_t*) base_addr + shape_offset;
          void* strides_addr = NULL;
          if (tarr->strides != NULL)
            strides_addr = (uint8_t*) base_addr + strides_offset;
          stream->Seek(tarr_offset);
          stream->Write(&data_addr, sizeof(void*));
          stream->Seek(tarr_offset + sizeof(void*) + sizeof(DLContext)
              + sizeof(int) + sizeof(DLDataType));
          stream->Write(&shape_addr, sizeof(void*));
          stream->Write(&strides_addr, sizeof(void*));
          void* tarr_addr = (uint8_t*) base_addr + tarr_offset;
          stream->Seek(args_offset + sizeof(TVMValue*) * i);
          stream->Write(&tarr_addr, sizeof(void*));
          break;
        }
      default:
        printf("couldn't process type code %d\n", type_codes[i]);
        break;
    }
  }
}

OpenOCDLowLevelDeviceAPI::OpenOCDLowLevelDeviceAPI(size_t num_bytes)
  : size(num_bytes) {
    socket.Create();
    socket.SetKeepAlive(true);
    socket.Connect(tvm::common::SockAddr("127.0.0.1", 6666));
    // TODO: NO HARDCODE.
    base_addr = (uint8_t*) 0x10010000;
    stream = new AllocatorStream(&args_buf);
}

OpenOCDLowLevelDeviceAPI::~OpenOCDLowLevelDeviceAPI() { }

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
      // TODO: Do the values need to be hex-encoded?
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
      // TODO: Figure out why the below is true.
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
  // TODO: should args section choice be at the micro level? no
  void* args_section = (void *) SECTION_ARGS;
  WriteTVMArgsToStream(args, stream, base_addr + SECTION_ARGS);
  Write(ctx, args_section, (uint8_t*) args_buf.c_str(), (size_t) stream->GetBufferSize());

  uint8_t* real_addr = GetRealAddr(offset);
  std::cout << "ATTEMPTING TO EXECUTE AT " << reinterpret_cast<uint64_t>(real_addr) << std::endl;
  //void (*func)(void) = (void (*)(void)) real_addr;
  //func();
  // TODO: Figure out how to begin execution at an arbitrary address.
  SendCommand("reset halt");
  std::stringstream resume_cmd;
  resume_cmd << "resume " << reinterpret_cast<uint64_t>(real_addr);
  SendCommand(resume_cmd.str());
  for (int i = 0; i < 15; i++) {
    usleep(5000000 / 15);
  }
  SendCommand("reset halt");
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
