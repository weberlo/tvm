/*!
 *  Copyright (c) 2018 by Contributors
 * \file allocator_stream.h
 * \brief allocator stream utility.
 */
#ifndef TVM_RUNTIME_ALLOCATOR_STREAM_H_
#define TVM_RUNTIME_ALLOCATOR_STREAM_H_

#include <stdio.h>
#include <cstring>
#include <string>
#include <algorithm>
#include <vector>
#include <dmlc/memory_io.h>

namespace tvm {
namespace runtime {
struct AllocatorStream : public dmlc::SeekStream {
 public:
  /*!
   * \brief constructor
   * \param p_buffer the pointer to the string.
   */
  explicit AllocatorStream(std::string *p_buffer)
      : p_buffer_(p_buffer) {
    curr_ptr_ = 0;
		max_ptr_ = 0;
  }
  virtual size_t Read(void *ptr, size_t size) {
    CHECK(curr_ptr_ <= p_buffer_->length());
    CHECK(curr_ptr_ + size <= max_ptr_);
    size_t nread = std::min(p_buffer_->length() - curr_ptr_, size);
    if (nread != 0) std::memcpy(ptr, &(*p_buffer_)[0] + curr_ptr_, nread);
    curr_ptr_ += nread;
    return nread;
  }
  virtual void Write(const void *ptr, size_t size) {
    if (size == 0) return;
    CHECK(curr_ptr_ + size <= max_ptr_);
    if (curr_ptr_ + size > p_buffer_->length()) {
      p_buffer_->resize(curr_ptr_+size);
    }
    std::memcpy(&(*p_buffer_)[0] + curr_ptr_, ptr, size);
    curr_ptr_ += size;
  }
  virtual void Seek(size_t pos) {
    curr_ptr_ = static_cast<size_t>(pos);
  }
  virtual size_t Tell(void) {
    return curr_ptr_;
  }
	size_t Allocate(size_t size) {
		size_t ret = max_ptr_;
		max_ptr_ += size;
		return ret;
	}
  size_t GetBufferSize() {
    return max_ptr_;
  }

 private:
  /*! \brief in memory buffer */
  std::string *p_buffer_;
  /*! \brief current pointer */
  size_t curr_ptr_;
  /*! \brief maximum pointer */
  size_t max_ptr_;
};

// TODO: Should this be here?
/*
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
*/

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_ALLOCATOR_STREAM_H_
