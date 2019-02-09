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
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_ALLOCATOR_STREAM_H_
