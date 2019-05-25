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
 * \file micro_section.h
 */
#ifndef TVM_RUNTIME_MICRO_MICRO_SECTION_H_
#define TVM_RUNTIME_MICRO_MICRO_SECTION_H_

namespace tvm {
namespace runtime {

// TODO(weberlo): Move everything into `tvm::runtime::micro` instead of
// `tvm::runtime`?  If so, rename this to `SectionAllocator`.

/*!
 * \brief TODO (something something on-device memory section)
 */
class MicroSection {
 public:
  /*!
   * \brief constructor that specifies section boundaries
   * \param location location of the section
   */
  MicroSection(SectionLocation location)
    : start_offset_(location.start),
      size_(0),
      capacity_(location.size) { }

  /*!
   * \brief destructor
   */
  ~MicroSection() {}

  /*!
   * \brief memory allocator
   * \param size size of allocated memory in bytes
   * \return pointer to allocated memory region in section, nullptr if out of space
   */
  DevBaseOffset Allocate(size_t size) {
    CHECK(size_ + size < capacity_)
        << "cannot alloc " << size << " bytes in section with start_addr " <<
        start_offset_.value();
    DevBaseOffset alloc_ptr = start_offset_ + size_;
    size_ += size;
    alloc_map_[alloc_ptr.value()] = size;
    return alloc_ptr;
  }

  /*!
   * \brief free prior allocation from section
   * \param offs offset to allocated memory
   * \note simple allocator scheme, more complex versions will be implemented later
   */
  void Free(DevBaseOffset offs) {
    std::uintptr_t ptr = offs.value();
    CHECK(alloc_map_.find(ptr) != alloc_map_.end()) << "freed pointer was never allocated";
    alloc_map_.erase(ptr);
    if (alloc_map_.empty()) {
      size_ = 0;
    }
  }

  /*!
   * \brief TODO
   */
  DevBaseOffset start_offset() const { return start_offset_; }

  // TODO(weberlo): clean up the naming between the next two methods.

  /*!
   * \brief TODO
   */
  DevBaseOffset curr_end_offset() const { return start_offset_ + size_; }

  /*!
   * \brief TODO
   */
  DevBaseOffset max_end_offset() const { return start_offset_ + capacity_; }

  /*!
   * \brief TODO
   */
  size_t size() const { return size_; }

  /*!
   * \brief TODO
   */
  size_t capacity() const { return capacity_; }

 private:
  /*! \brief start address of the section */
  DevBaseOffset start_offset_;
  /*! \brief current size of the section */
  size_t size_;
  /*! \brief total storage capacity of the section */
  size_t capacity_;
  /*! \brief allocation map for allocation sizes */
  std::unordered_map<std::uintptr_t, size_t> alloc_map_;
};

}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_MICRO_MICRO_SECTION_H_
