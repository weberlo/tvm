/*!
 *  Copyright (c) 2019 by Contributors
 * \file target_data_layout_encoder.h
 * \brief uTVM data layout encoder
 */
#ifndef TVM_RUNTIME_MICRO_TARGET_DATA_LAYOUT_ENCODER_H_
#define TVM_RUNTIME_MICRO_TARGET_DATA_LAYOUT_ENCODER_H_

#include <dmlc/memory_io.h>

#include <algorithm>
#include <cstring>
#include <iomanip>
#include <memory>
#include <string>
#include <vector>

#include "device/utvm_runtime.h"

namespace tvm {
namespace runtime {

/*!
 * \brief data encoder for uTVM that builds a host-side buffer
 */
class TargetDataLayoutEncoder {
 public:
  /*!
   * \brief helper class for writing into `TargetDataLayoutEncoder`
   */
  template <typename T>
  class Slot {
   public:
    /*!
     * \brief constructor
     * \param parent pointer to parent encoder
     * \param start_offset start byte offset of the slot in the backing buffer
     * \param size size (in bytes) of the memory region allocated for this slot
     * \param dev_start_addr start address of the slot in the device's memory
     */
    Slot(TargetDataLayoutEncoder* parent, size_t start_offset, size_t size, void* dev_start_addr);

    ~Slot();

    /*!
     * \brief writes `sizeof(T) * num_elems` bytes of data from `src_ptr`
     * \param src_ptr address of the buffer to be read from
     * \param num_elems number of elements in array (defaults to 1)
     */
    void Write(const T* src_ptr, size_t num_elems = 1);

    /*!
     * \brief returns start address of the slot in device memory
     * \return device start address
     */
    void* dev_start_addr();

    /*!
     * \brief returns number of bytes allocated for this slot
     * \return size of this slot
     */
    size_t size();

   private:
    /*! \brief pointer to parent encoder */
    TargetDataLayoutEncoder* parent_;
    /*! \brief start offset of the slot in the parent's backing parent_buffer */
    size_t start_offset_;
    /*! \brief current offset relative to the start offset of this slot */
    size_t curr_offset_;
    /*! \brief size (in bytes) of the memory region allocated for this slot */
    size_t size_;
    /*! \brief start address of the slot in the device's memory */
    void* dev_start_addr_;
  };

  /*!
   * \brief constructor
   * \param dev_start_addr start address of the encoder in device memory
   * \param dev_base_addr base address of the device
   */
  explicit TargetDataLayoutEncoder(void* dev_start_addr, const void* dev_base_addr)
      : buf_(std::vector<uint8_t>()),
        curr_offset_(0),
        dev_start_addr_(dev_start_addr),
        dev_base_addr_(dev_base_addr) {}

  /*!
   * \brief allocates a slot for `sizeof(T) * num_elems` bytes of data
   * \param num_elems number of elements of type `T` being allocated (defaults to 1)
   * \return slot of size `sizeof(T) * num_elems` bytes
   */
  template <typename T>
  Slot<T> Alloc(size_t num_elems = 1) {
    size_t size = sizeof(T) * num_elems;
    if (curr_offset_ + size > buf_.size()) {
      buf_.resize(curr_offset_ + size);
    }
    size_t slot_start_offset = curr_offset_;
    curr_offset_ += size;
    return Slot<T>(this, slot_start_offset, size, GetDevAddr(slot_start_offset));
  }

  /*!
   * \brief returns the corresponding device address for the offset `offset`
   * \param offset byte offset from the beginning of the backing buffer
   * \return device address
   */
  void* GetDevAddr(size_t offset) {
    return reinterpret_cast<uint8_t*>(dev_start_addr_) + offset;
  }

  /*!
   * \brief returns the array backing the encoder's buffer
   * \return array backing the encoder's buffer
   */
  uint8_t* data() {
    return buf_.data();
  }

  /*!
   * \brief returns current size of the encoder's buffer
   * \return buffer size
   */
  size_t buf_size() {
    return buf_.size();
  }

 private:
  /*! \brief in-memory backing buffer */
  std::vector<uint8_t> buf_;
  /*! \brief current offset */
  size_t curr_offset_;
  /*! \brief start address of the encoder in device memory */
  void* dev_start_addr_;
  /*! \brief base address of the device */
  const void* dev_base_addr_;
};

template <typename T>
TargetDataLayoutEncoder::Slot<T>::Slot(TargetDataLayoutEncoder* parent, size_t start_offset,
                                       size_t size, void* dev_start_addr)
    : parent_(parent),
      start_offset_(start_offset),
      curr_offset_(0),
      size_(size),
      dev_start_addr_(dev_start_addr) {}

template <typename T>
TargetDataLayoutEncoder::Slot<T>::~Slot() {
  CHECK(curr_offset_ == size_) << "unwritten space in slot";
}

template <typename T>
void TargetDataLayoutEncoder::Slot<T>::Write(const T* src_ptr, size_t num_elems) {
  if (num_elems == 0) return;
  size_t size = sizeof(T) * num_elems;
  CHECK(curr_offset_ + size <= size_) << "not enough space in slot";
  uint8_t* curr_ptr = &(parent_->data())[start_offset_ + curr_offset_];
  std::memcpy(curr_ptr, src_ptr, size);
  curr_offset_ += size;
}

template <typename T>
void* TargetDataLayoutEncoder::Slot<T>::dev_start_addr() {
  return dev_start_addr_;
}

template <typename T>
size_t TargetDataLayoutEncoder::Slot<T>::size() {
  return size_;
}

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_MICRO_TARGET_DATA_LAYOUT_ENCODER_H_
