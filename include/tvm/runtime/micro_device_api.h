/*!
 *  Copyright (c) 2018 by Contributors
 * \file tvm/runtime/micro_device_api.h
 * \brief Abstract micro device memory management API
 */
#ifndef TVM_RUNTIME_MICRO_DEVICE_API_H_
#define TVM_RUNTIME_MICRO_DEVICE_API_H_

#include <cstring>
#include "packed_func.h"
#include "c_runtime_api.h"
#define PAGE_SIZE 4096

namespace tvm {
namespace runtime {

class MicroDeviceAPI {
  public:
  /*! \brief virtual destructor */
  virtual ~MicroDeviceAPI() {}

  // TODO: Do I need the TVMContext? What to do with it?
  virtual void WriteToMemory(TVMContext ctx, void* offset, uint8_t* buf, size_t num_bytes);

  virtual void ReadFromMemory(TVMContext ctx, void* offset, uint8_t* buf, size_t num_bytes);

  virtual void ChangeMemoryProtection(TVMContext ctx, void* offset, int prot, size_t num_bytes);

  virtual void Execute(TVMContext ctx, void* offset);

  virtual void Reset(TVMContext ctx);

  static std::shared_ptr<MicroDeviceAPI> Get(int table_index);

  static std::shared_ptr<MicroDeviceAPI> Create(size_t num_bytes);

  protected:
  // TODO: figure out endianness and mem alignment
  int endianness;
  size_t size;
};
  
} // namespace runtime
} // namespace tvm
#endif // TVM_RUNTIME_MICRO_DEVICE_API_H_
