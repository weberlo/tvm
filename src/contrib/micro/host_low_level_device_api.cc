/*!
 *  Copyright (c) 2018 by Contributors
 * \file host_low_level_device_api.cc
 * \brief emulated micro device implementation on host machine
 */
#include <sys/mman.h>
#include <iostream>
#include <mutex>
#include <errno.h>
#include <tvm/runtime/c_runtime_api.h>
#include "host_low_level_device_api.h"
#include "allocator_stream.h"
#include "device_memory_offsets.h"

namespace tvm {
namespace runtime {
  HostLowLevelDeviceAPI::HostLowLevelDeviceAPI(size_t num_bytes) 
    : size(num_bytes) {
    size = num_bytes;
    size_in_pages = (num_bytes + PAGE_SIZE - 1) / PAGE_SIZE;
    int mmap_prot = PROT_READ | PROT_WRITE | PROT_EXEC;
    int mmap_flags = MAP_ANONYMOUS | MAP_PRIVATE;
    base_addr = (uint8_t*) mmap(NULL, size_in_pages * PAGE_SIZE, mmap_prot, mmap_flags, -1, 0);
    stream = new AllocatorStream(&args_buf);
  }

  HostLowLevelDeviceAPI::~HostLowLevelDeviceAPI() {
    Shutdown();
  }

  void HostLowLevelDeviceAPI::Write(TVMContext ctx,
                     void* offset,
                     uint8_t* buf,
                     size_t num_bytes) {
    uint8_t* real_addr = GetRealAddr(offset);
    std::memcpy(real_addr, buf, num_bytes);
  }

  void HostLowLevelDeviceAPI::Read(TVMContext ctx,
                      void* offset,
                      uint8_t* buf,
                      size_t num_bytes) {
    uint8_t* real_addr = GetRealAddr(offset);
    std::memcpy(buf, real_addr, num_bytes);
  }

  void WriteTVMArgsToStream(TVMArgs args, AllocatorStream* stream, void* base_addr) {
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

  void HostLowLevelDeviceAPI::Execute(TVMContext ctx, TVMArgs args, TVMRetValue *rv, void* offset) {
    // TODO: should args section choice be at the micro level? no
    void* args_section = (void *) SECTION_ARGS;
    WriteTVMArgsToStream(args, stream, base_addr + SECTION_ARGS);
    Write(ctx, args_section, (uint8_t*) args_buf.c_str(), (size_t) stream->GetBufferSize());
    uint8_t* real_addr = GetRealAddr(offset);
    void (*func)(void) = (void (*)(void)) real_addr;
    func();
  }

  // host does not need a reset 
  void HostLowLevelDeviceAPI::Reset(TVMContext ctx) {
  }

} // namespace runtime
} // namespace tvm
