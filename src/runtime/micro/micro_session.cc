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
 * \file micro_session.cc
 * \brief session to manage multiple micro modules
 */

#include <tvm/runtime/registry.h>
#include <memory>
#include <string>
#include "micro_session.h"
#include "low_level_device.h"
#include "host_low_level_device.h"
#include "openocd_low_level_device.h"
#include "target_data_layout_encoder.h"

namespace tvm {
namespace runtime {

MicroSession::MicroSession() { }

MicroSession::~MicroSession() { }

void MicroSession::InitSession(const TVMArgs& args) {
  DevBaseOffset curr_start_offset = kDeviceStart;
  for (size_t i = 0; i < static_cast<size_t>(SectionKind::kNumKinds); i++) {
    size_t section_size = GetDefaultSectionSize(static_cast<SectionKind>(i));
    sections_[i] = std::make_shared<MicroSection>(SectionLocation {
      .start = curr_start_offset,
      .size = section_size,
    });
    curr_start_offset += section_size;
  }
  memory_size_ = curr_start_offset.cast_to<size_t>();

  std::string device_type = args[0];
  std::string binary_path = args[1];
  // TODO(weberlo): make device type enum
  if (device_type == "host") {
    low_level_device_ = HostLowLevelDeviceCreate(memory_size_);
    binutil_prefix_ = "";
  } else if (device_type == "openocd") {
    int port = args[2];
    low_level_device_ = OpenOCDLowLevelDeviceCreate(port);
    binutil_prefix_ = "riscv64-unknown-elf-";
  } else {
    LOG(FATAL) << "Unsupported micro low-level device";
  }
  SetInitBinaryPath(args[1]);
  CHECK(!init_binary_path_.empty()) << "init library not initialized";
  init_stub_info_ = LoadBinary(init_binary_path_);
  utvm_main_symbol_ = init_symbol_map()["UTVMMain"];
  utvm_done_symbol_ = init_symbol_map()["UTVMDone"];

  std::cout << "[InitSession]" << std::endl;
  PrintSymbol<void*>(init_symbol_map(), "task");
  PrintSymbol<void*>(init_symbol_map(), "utvm_workspace_begin");
  PrintSymbol<void*>(init_symbol_map(), "utvm_workspace_curr");
  // PrintSymbol<size_t>(init_symbol_map(), "num_active_allocs");
  PrintSymbol<void*>(init_symbol_map(), "last_error");
  PrintSymbol<void*>(init_symbol_map(), "return_code");
  PrintSymbol<void*>(init_symbol_map(), "UTVMDone");
  PrintSymbol<void*>(init_symbol_map(), "UTVMMain");
  PrintSymbol<void*>(init_symbol_map(), "TVMBackendAllocWorkspace");
  PrintSymbol<void*>(init_symbol_map(), "TVMBackendFreeWorkspace");
  PrintSymbol<void*>(init_symbol_map(), "TVMAPISetLastError");

  if (device_type == "openocd") {
    // Set OpenOCD device's breakpoint and stack pointer.
    auto ocd_device = std::dynamic_pointer_cast<OpenOCDLowLevelDevice>(low_level_device_);
    ocd_device->SetBreakpoint(utvm_done_symbol_);
    auto stack_section = GetSection(SectionKind::kStack);
    ocd_device->SetStackTop(stack_section->max_end_offset());
  }

  // TODO(weberlo): Make `DevSymbolWrite` func?
  // Patch workspace pointers to the start of the workspace section.
  DevBaseOffset workspace_start_hole_offset = init_symbol_map()["utvm_workspace_begin"];
  DevBaseOffset workspace_end_hole_offset = init_symbol_map()["utvm_workspace_end"];
  DevBaseOffset workspace_start_offset = GetSection(SectionKind::kWorkspace)->start_offset();
  DevBaseOffset workspace_end_offset = GetSection(SectionKind::kWorkspace)->max_end_offset();
  void* workspace_start_addr =
      (workspace_start_offset + low_level_device_->base_addr()).cast_to<void*>();
  void* workspace_end_addr =
      (workspace_end_offset + low_level_device_->base_addr()).cast_to<void*>();
  low_level_device()->Write(workspace_start_hole_offset, &workspace_start_addr, sizeof(void*));
  low_level_device()->Write(workspace_end_hole_offset, &workspace_end_addr, sizeof(void*));
}

void MicroSession::EndSession() {
  // text_allocator_ = nullptr;
  // rodata_allocator_ = nullptr;
  // data_allocator_ = nullptr;
  // bss_allocator_ = nullptr;
  // args_allocator_ = nullptr;
  // stack_allocator_ = nullptr;
  // heap_allocator_ = nullptr;

  low_level_device_ = nullptr;
}

DevBaseOffset MicroSession::AllocateInSection(SectionKind type, size_t size) {
    return GetSection(type)->Allocate(size);
}

void MicroSession::FreeInSection(SectionKind type, DevBaseOffset ptr) {
    return GetSection(type)->Free(ptr);
}

std::string MicroSession::ReadString(DevBaseOffset str_offset) {
  std::stringstream result;
  // TODO(weberlo): remove the use of static and use strings
  static char buf[256];
  size_t i = 256;
  while (i == 256) {
    low_level_device()->Read(str_offset, reinterpret_cast<void*>(buf), 256);
    i = 0;
    while (i < 256) {
      if (buf[i] == 0) break;
      result << buf[i];
      i++;
    }
    str_offset = str_offset + i;
  }
  return result.str();
}

void MicroSession::PushToExecQueue(DevBaseOffset func, const TVMArgs& args) {
  int32_t (*func_dev_addr)(void*, void*, int32_t) =
      reinterpret_cast<int32_t (*)(void*, void*, int32_t)>(
      (func + low_level_device()->base_addr()).value());

  // Create an allocator stream for the memory region after the most recent
  // allocation in the args section.
  DevAddr args_addr =
      low_level_device()->base_addr() + GetSection(SectionKind::kArgs)->curr_end_offset();
  TargetDataLayoutEncoder encoder(args_addr);

  EncoderAppend(&encoder, args);
  // Flush `stream` to device memory.
  DevBaseOffset stream_dev_offset =
      GetSection(SectionKind::kArgs)->Allocate(encoder.buf_size());
  low_level_device()->Write(stream_dev_offset,
                            reinterpret_cast<void*>(encoder.data()),
                            encoder.buf_size());

  UTVMTask task = {
      .func = func_dev_addr,
      .args = args_addr.cast_to<UTVMArgs*>(),
  };
  // TODO(mutinifni): handle bits / endianness
  // Write the task.
  low_level_device()->Write(init_symbol_map()["task"], &task, sizeof(UTVMTask));

  // // Zero out the last error.
  // int32_t last_error = 0;
  // low_level_device()->Write(init_symbol_map()["last_error"], &last_error, sizeof(int32_t));

  low_level_device()->Execute(utvm_main_symbol_, utvm_done_symbol_);

  // // Check if there was an error during execution.  If so, log it.
  // CheckDeviceError();
}

BinaryInfo MicroSession::LoadBinary(std::string binary_path) {
  SectionLocation text;
  SectionLocation rodata;
  SectionLocation data;
  SectionLocation bss;

  text.size = GetSectionSize(binary_path, SectionKind::kText, binutil_prefix_);
  rodata.size = GetSectionSize(binary_path, SectionKind::kRodata, binutil_prefix_);
  data.size = GetSectionSize(binary_path, SectionKind::kData, binutil_prefix_);
  bss.size = GetSectionSize(binary_path, SectionKind::kBss, binutil_prefix_);

  text.start = AllocateInSection(SectionKind::kText, text.size);
  rodata.start = AllocateInSection(SectionKind::kRodata, rodata.size);
  data.start = AllocateInSection(SectionKind::kData, data.size);
  bss.start = AllocateInSection(SectionKind::kBss, bss.size);
  CHECK(text.start != nullptr && rodata.start != nullptr && data.start != nullptr &&
        bss.start != nullptr) << "not enough space to load module on device";
  const DevBaseAddr base_addr = low_level_device_->base_addr();
  std::cout << "[LoadBinary]" << std::endl;
  std::cout << "  text: start=" << text.start.cast_to<void*>() << ", size=" << std::dec << text.size << std::endl;
  std::cout << "  rodata: start=" << rodata.start.cast_to<void*>() << ", size=" << std::dec << rodata.size << std::endl;
  std::cout << "  data: start=" << data.start.cast_to<void*>() << ", size=" << std::dec << data.size << std::endl;
  std::cout << "  bss: start=" << bss.start.cast_to<void*>() << ", size=" << std::dec << bss.size << std::endl;

  std::string relocated_bin = RelocateBinarySections(
      binary_path,
      text.start + base_addr,
      rodata.start + base_addr,
      data.start + base_addr,
      bss.start + base_addr,
      binutil_prefix_);
  std::string text_contents = ReadSection(relocated_bin, SectionKind::kText, binutil_prefix_);
  std::string rodata_contents = ReadSection(relocated_bin, SectionKind::kRodata, binutil_prefix_);
  std::string data_contents = ReadSection(relocated_bin, SectionKind::kData, binutil_prefix_);
  std::string bss_contents = ReadSection(relocated_bin, SectionKind::kBss, binutil_prefix_);
  low_level_device_->Write(text.start, &text_contents[0], text.size);
  low_level_device_->Write(rodata.start, &rodata_contents[0], rodata.size);
  low_level_device_->Write(data.start, &data_contents[0], data.size);
  low_level_device_->Write(bss.start, &bss_contents[0], bss.size);
  SymbolMap symbol_map {relocated_bin, base_addr, binutil_prefix_};
  return BinaryInfo {
      .text = text,
      .rodata = rodata,
      .data = data,
      .bss = bss,
      .symbol_map = symbol_map,
  };
}

void MicroSession::SetInitBinaryPath(std::string path) {
  init_binary_path_ = path;
}

DevAddr MicroSession::EncoderAppend(TargetDataLayoutEncoder* encoder, const TVMArgs& args) {
  auto utvm_args_slot = encoder->Alloc<UTVMArgs>();

  const int* type_codes = args.type_codes;
  int num_args = args.num_args;

  auto tvm_vals_slot = encoder->Alloc<TVMValue>(num_args);
  auto type_codes_slot = encoder->Alloc<const int>(num_args);

  for (int i = 0; i < num_args; i++) {
    switch (type_codes[i]) {
      case kNDArrayContainer:
      case kArrayHandle: {
        TVMArray* arr_handle = args[i];
        void* arr_ptr = EncoderAppend(encoder, *arr_handle).cast_to<void*>();
        TVMValue val;
        val.v_handle = arr_ptr;
        tvm_vals_slot.WriteValue(val);
        break;
      }
      // TODO(weberlo): Implement `double` and `int64` case.
      case kDLFloat:
      case kDLInt:
      case kDLUInt:
      default:
        LOG(FATAL) << "Unsupported type code for writing args: " << type_codes[i];
        break;
    }
  }
  type_codes_slot.WriteArray(type_codes, num_args);

  UTVMArgs dev_args = {
    .values = tvm_vals_slot.start_addr().cast_to<TVMValue*>(),
    .type_codes = type_codes_slot.start_addr().cast_to<int*>(),
    .num_args = num_args,
  };
  utvm_args_slot.WriteValue(dev_args);
  return utvm_args_slot.start_addr();
}

DevAddr MicroSession::EncoderAppend(TargetDataLayoutEncoder* encoder, const TVMArray& arr) {
  auto tvm_arr_slot = encoder->Alloc<TVMArray>();
  auto shape_slot = encoder->Alloc<int64_t>(arr.ndim);

  // `shape` and `strides` are stored on the host, so we need to write them to
  // the device first. The `data` field is already allocated on the device and
  // is a device pointer, so we don't need to write it.
  shape_slot.WriteArray(arr.shape, arr.ndim);
  DevAddr shape_addr = shape_slot.start_addr();
  DevAddr strides_addr = DevAddr(nullptr);
  if (arr.strides != nullptr) {
    auto stride_slot = encoder->Alloc<int64_t>(arr.ndim);
    stride_slot.WriteArray(arr.strides, arr.ndim);
    strides_addr = stride_slot.start_addr();
  }

  // Copy `arr`, update the copy's pointers to be device pointers, then
  // write the copy to `tvm_arr_slot`.
  TVMArray dev_arr = arr;
  // Update the device type to look like a host, because codegen generates
  // checks that it is a host array.
  CHECK(dev_arr.ctx.device_type == static_cast<DLDeviceType>(kDLMicroDev))
    << "attempt to write TVMArray with non-micro device type";
  dev_arr.ctx.device_type = DLDeviceType::kDLCPU;
  // Add the base address of the device to the array's data's device offset to
  // get a device address.
  DevBaseOffset arr_offset(reinterpret_cast<std::uintptr_t>(arr.data));
  dev_arr.data = (low_level_device()->base_addr() + arr_offset).cast_to<void*>();
  dev_arr.shape = shape_addr.cast_to<int64_t*>();
  dev_arr.strides = strides_addr.cast_to<int64_t*>();
  tvm_arr_slot.WriteValue(dev_arr);
  return tvm_arr_slot.start_addr();
}

void MicroSession::CheckDeviceError() {
  int32_t return_code = DevSymbolRead<int32_t>(init_symbol_map(), "return_code");
  if (return_code) {
    std::uintptr_t last_error = DevSymbolRead<std::uintptr_t>(init_symbol_map(), "last_error");
    DevBaseOffset last_err_offset =
        DevAddr(last_error) - low_level_device()->base_addr();
    // Then read the string from device to host and log it.
    std::string last_error_str = ReadString(last_err_offset);
    // LOG(FATAL) << "error during micro function execution:\n"
    std::cerr << "error during micro function execution:\n"
               << "  return code: " << std::dec << return_code << "\n"
               << "  dev str addr: 0x" << std::hex << last_error << "\n"
               << "  dev str data: " << last_error_str;
  }
}

/*!
  * \brief TODO
  */
template <typename T>
T MicroSession::DevSymbolRead(SymbolMap& symbol_map, const std::string& symbol) {
  DevBaseOffset sym_offset = symbol_map[symbol];
  T result;
  low_level_device()->Read(sym_offset, &result, sizeof(T));
  return result;
}

/*!
  * \brief TODO
  */
template <typename T>
void MicroSession::PrintSymbol(SymbolMap& symbol_map, const std::string& symbol) {
  T val = DevSymbolRead<T>(symbol_map, symbol);
  std::cout << "  " << symbol << " (0x" << std::hex
            << (low_level_device()->base_addr() + symbol_map[symbol]).value()
            << "): " << val
            << std::endl;
}

// initializes micro session and low-level device from Python frontend
TVM_REGISTER_GLOBAL("micro._InitSession")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    std::shared_ptr<MicroSession> session = MicroSession::Global();
    session->InitSession(args);
    });

// ends micro session and destructs low-level device from Python frontend
TVM_REGISTER_GLOBAL("micro._EndSession")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    std::shared_ptr<MicroSession> session = MicroSession::Global();
    session->EndSession();
    });
}  // namespace runtime
}  // namespace tvm
