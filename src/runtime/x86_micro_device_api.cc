/*!
 *  Copyright (c) 2018 by Contributors
 * \file x86_micro_device_api.cc
 * \brief x86-emulated micro device API
 */
#include <sys/mman.h>
#include <iostream>
#include <mutex>
#include <errno.h>
#include <tvm/runtime/c_runtime_api.h>
#include "x86_micro_device_api.h"
#include "allocator_stream.h"

namespace tvm {
namespace runtime {
  x86MicroDeviceAPI::x86MicroDeviceAPI(size_t num_bytes) 
    : size(num_bytes) {
    size = num_bytes;
    size_in_pages = (num_bytes + PAGE_SIZE - 1) / PAGE_SIZE;
    printf("allocated size %d num pages %d\n", size, size_in_pages);
    int prot = PROT_READ | PROT_WRITE | PROT_EXEC;
    int flags = MAP_ANONYMOUS | MAP_PRIVATE;
    base_addr = (uint8_t*) mmap(NULL, size_in_pages * PAGE_SIZE, prot, flags, -1, 0);
    printf("errno %d\n", errno);
    printf("base addr original %p\n", base_addr);
    //int buf_size = 10 * PAGE_SIZE;
    stream = new AllocatorStream(&args_buf);
    printf("initialized x86microdeviceapi %p\n", this);
  }

  x86MicroDeviceAPI::~x86MicroDeviceAPI() {
    Shutdown();
  }

  void x86MicroDeviceAPI::WriteToMemory(TVMContext ctx,
                     void* offset,
                     uint8_t* buf,
                     size_t num_bytes) {
    //printf("writing x86 to memory\n");
    fflush(stdout);
    uint8_t* real_addr = GetRealAddr(offset);
    //printf("real addr %p\n", real_addr);
    fflush(stdout);
    //printf("num_bytes %d\n", num_bytes);
    fflush(stdout);
    int i = 0;
    uint8_t x;
    for (i = 0; i < num_bytes; i++)
      x = buf[i];
    //printf("forloop 1 done\n");
    fflush(stdout);
    for (i = 0; i < num_bytes; i++)
      real_addr[i];
    //printf("forloop 2 done\n");
    fflush(stdout);
    std::memcpy(real_addr, buf, num_bytes);
    //printf("memcpy done\n");
    fflush(stdout);
  }

  void x86MicroDeviceAPI::ReadFromMemory(TVMContext ctx,
                      void* offset,
                      uint8_t* buf,
                      size_t num_bytes) {
    uint8_t* real_addr = GetRealAddr(offset);
    std::memcpy(buf, real_addr, num_bytes);
  }

  void x86MicroDeviceAPI::ChangeMemoryProtection(TVMContext ctx,
                              void* offset,
                              int prot,
                              size_t num_bytes) {
    // needs to be page aligned
    // we assume all memory is executable for now, so this isn't called
    uint8_t* real_addr = GetRealAddr(offset);
    mprotect(real_addr, num_bytes, prot);
  }

  void WriteTVMArgsToStream(TVMArgs args, AllocatorStream* stream, void* base_addr) {
    const TVMValue* values = args.values;
    const int* type_codes = args.type_codes;
    int num_args = args.num_args;
		size_t args_offset = stream->Allocate(sizeof(TVMValue*) * num_args
										 			+ sizeof(const int*) * num_args + sizeof(int));
    printf("args offset %d\n", args_offset);
    stream->Seek(args_offset + sizeof(TVMValue*) * num_args);
    stream->Write(type_codes, sizeof(const int*) * num_args);
    stream->Write(&num_args, sizeof(int));
    printf("type_codes and num_args written\n");
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
            printf("was arraycontainer\n");
            TVMArray* tarr = (TVMArray*)(values[i].v_handle);
            size_t tarr_offset = stream->Allocate(sizeof(TVMArray));
            size_t data_offset;
            size_t shape_size = 1;
            for (int dim = 0; dim < tarr->ndim; dim++)
              shape_size *= tarr->shape[dim];
            printf("shape size %d\n", shape_size);
            if (tarr->dtype.code == kDLInt) {
              printf("values dtype 0\n");
              data_offset = stream->Allocate(sizeof(int) * shape_size);
              stream->Seek(data_offset);
              stream->Write(tarr->data, sizeof(int) * shape_size);
              printf("wrote data arr\n");
            }
            if (tarr->dtype.code == kDLUInt) {
              printf("values dtype 1\n");
              data_offset = stream->Allocate(sizeof(unsigned int) * shape_size);
              stream->Seek(data_offset);
              stream->Write(tarr->data, sizeof(unsigned int) * shape_size);
              printf("wrote data arr\n");
            }
            if (tarr->dtype.code == kDLFloat) {
              printf("values dtype 2\n");
              data_offset = stream->Allocate(sizeof(float) * shape_size);
              printf("size %d\n", stream->GetBufferSize());
              stream->Seek(data_offset);
              stream->Write(tarr->data, sizeof(float) * shape_size);
              printf("wrote data arr\n");
            }
            size_t shape_offset = stream->Allocate(sizeof(int64_t) * tarr->ndim);
            stream->Seek(shape_offset);
            stream->Write(tarr->shape, sizeof(int64_t) * tarr->ndim);
            printf("shape allocated\n");
            fflush(stdout);
            size_t strides_offset = 0;
            if (tarr->strides != NULL) {
              strides_offset = stream->Allocate(sizeof(int64_t) * tarr->ndim);
              stream->Seek(strides_offset);
              stream->Write(tarr->strides, sizeof(int64_t) * tarr->ndim);
            }
            printf("everything allocated\n");
            fflush(stdout);
            stream->Seek(tarr_offset);
            stream->Write(tarr, sizeof(TVMArray)); 
            void* data_addr = base_addr + data_offset;
            void* shape_addr = base_addr + shape_offset;
            void* strides_addr = NULL;
            if (tarr->strides != NULL)
              strides_addr = base_addr + strides_offset;
            stream->Seek(tarr_offset);
            stream->Write(&data_addr, sizeof(void*));
            stream->Seek(tarr_offset + sizeof(void*) + sizeof(DLContext) 
                        + sizeof(int) + sizeof(DLDataType));
            stream->Write(&shape_addr, sizeof(void*));
            stream->Write(&strides_addr, sizeof(void*));
            printf("tvmarray written\n");
            void* tarr_addr = base_addr + tarr_offset;
            printf("tarr %p\n", tarr_addr);
            stream->Seek(args_offset + sizeof(TVMValue*) * i);
            stream->Write(&tarr_addr, sizeof(void*));
            printf("values %d written\n", i);
            fflush(stdout);
            break;
          }
        default:
          {
            printf("couldn't process type code %d\n", type_codes[i]);
            break;
          }
      }
    }
  }

  void dum(void* args)
  {
    printf("dum %p %d\n", (int*)args, ((int*)args)[0]);
  }

  void x86MicroDeviceAPI::Execute(TVMContext ctx, TVMArgs args, TVMRetValue *rv, void* offset) {
    // TODO: should args section choice be at the micro level? likely no
    printf("running execute in x86MicroDeviceAPI\n");
    fflush(stdout);
    void* args_section = (void *)(30 * PAGE_SIZE);
    printf("Execute: writing to memory x86MicroDeviceAPI\n");
    fflush(stdout);
    printf("size of args_buf %d\n", sizeof(args_buf));
    fflush(stdout);
    WriteTVMArgsToStream(args, stream, base_addr + (30 * PAGE_SIZE));
    printf("size of stream %d\n", stream->GetBufferSize());
    int buf_size = stream->GetBufferSize();
    WriteToMemory(ctx, args_section, (uint8_t*) args_buf.c_str(), (size_t) stream->GetBufferSize());
    TVMValue* val0 = reinterpret_cast<TVMValue*>(base_addr + (30 * PAGE_SIZE));
    //TVMValue val01 = reinterpret_cast<TVMValue>(base_addr + (30 * PAGE_SIZE));
    int32_t* ids = reinterpret_cast<int32_t*>(base_addr + (30 * PAGE_SIZE) + sizeof(TVMValue*)*3);
    void** ags = (void**)(base_addr + (20*PAGE_SIZE) + sizeof(void**));
    dum(*ags);
    printf("ids is %p %d %d %d\n", ids, ids[0], ids[1], ids[2]);
    printf("ptr is %d\n", ((TVMArray*)(val0[1].v_handle))->shape[0]);
    printf("ptr is %d\n", ((TVMArray*)(val0[1].v_handle))->strides);
    printf("args ptr is %p\n", *(TVMValue*)(base_addr + (20 * PAGE_SIZE)));
    printf("args ptr is %p\n", (base_addr + (30 * PAGE_SIZE)));
    printf("func ptr is %p\n", (base_addr + (20 * PAGE_SIZE)+ 24) );
    //for(int x = 0; x < 1024; x++)
    //  printf("%lf ", ((float*)((TVMArray*)(val0[1].v_handle))->data)[x]);
    //printf("ptr is %p\n", *(base_addr + (30 * PAGE_SIZE)));
    //printf("ptr is %p\n", *(base_addr + (30 * PAGE_SIZE) + sizeof(TVMValue*)));
    //printf("args val 0 ndim %d\n", ((TVMArray*)val0)->ndim);
    //printf("args val 0 ndim %d\n", ((TVMArray*)&val)->ndim);
    //printf("args val 0 ndim %d\n", ids[0]);
    printf("Execute: wrote args to memory\n");
    fflush(stdout);
    uint8_t* real_addr = GetRealAddr(offset);
    // This should be the function signature if it's to know where things are
    printf("Execute: calling function\n");
    fflush(stdout);
    //void (*func)(const void*, const void*, int) = (void (*)(const void*, const void*, int)) real_addr;
    void (*func)(void) = (void (*)(void)) real_addr;
    func();
    // TODO: copy-back phase
    printf("Execute: called function\n");
    fflush(stdout);
    const TVMValue* values = args.values;
    for(int x = 0; x < 1024; x++)
      printf("%lf ", ((float*)((TVMArray*)(val0[2].v_handle))->data)[x]);
      //printf("%lf ", ((float*)((TVMArray*)(values[2].v_handle))->data)[x]);
    //ReadFromMemory(ctx, ((TVMArray*)(args_section + sizeof(TVMValue*) * 2))->data, 
    //    (uint8_t*) ((TVMArray*)(values[2].v_handle))->data, 1024 * 4);
    fflush(stdout);
  }

  void x86MicroDeviceAPI::Reset(TVMContext ctx) {
  }

} // namespace runtime
} // namespace tvm
