/*!
 *  Copyright (c) 2018 by Contributors
 * \file micro_module.h
 * \brief MicroModule 
 */
#ifndef TVM_RUNTIME_MICRO_MODULE_H_
#define TVM_RUNTIME_MICRO_MODULE_H_

#include <tvm/runtime/packed_func.h>
#include <memory>
#include <vector>
#include <string>
#include "../meta_data.h"

namespace tvm {
namespace runtime {
/*!
 * \brief create a MicroModule for micro devices 
 */
Module MicroModuleCreate(
    std::string data,
    std::string fmt,
    std::unordered_map<std::string, FunctionInfo> fmap,
    std::string source);
}  // namespace runtime
}  // namespace tvm
#endif  // TVM_RUNTIME_MICRO_MODULE_H_
