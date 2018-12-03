/*!
*  Copyright (c) 2018 by Contributors
* \file openocd_module.cc
*/
#include <tvm/runtime/registry.h>
#include <vector>
#include <array>
#include <string>
#include <mutex>
#include "pack_args.h"
#include "thread_storage_scope.h"
#include "meta_data.h"
#include "file_util.h"

namespace tvm {
namespace runtime {

// Module to support thread-safe multi-GPU execution.
// cuModule is a per-GPU module
// The runtime will contain a per-device module table
// The modules will be lazily loaded
class OpenOCDModuleNode : public runtime::ModuleNode {
public:
 explicit OpenOCDModuleNode(std::string data,
                         std::string fmt,
                         std::unordered_map<std::string, FunctionInfo> fmap,
                         std::string source)
     : data_(data), fmt_(fmt), fmap_(fmap), source_(source) {
  printf("Called constructor of OpenOCDModuleNode\n");
 }
 // destructor
 ~OpenOCDModuleNode() {
   printf("Called destructor of OpenOCDModuleNode\n");
 }

 const char* type_key() const final {
   return "openocd";
 }

 PackedFunc GetFunction(
     const std::string& name,
     const std::shared_ptr<ModuleNode>& sptr_to_self) final;

 void SaveToFile(const std::string& file_name,
                 const std::string& format) final {
   printf("Called SaveToFile of OpenOCDModuleNode\n");
   std::string fmt = GetFileFormat(file_name, format);
   std::string meta_file = GetMetaFilePath(file_name);
   if (fmt == "cc") {
     CHECK_NE(source_.length(), 0);
     SaveMetaDataToFile(meta_file, fmap_);
     SaveBinaryToFile(file_name, source_);
   } else {
     CHECK_EQ(fmt, fmt_)
         << "Can only save to format=" << fmt_;
     SaveMetaDataToFile(meta_file, fmap_);
     SaveBinaryToFile(file_name, data_);
   }
 }

 void SaveToBinary(dmlc::Stream* stream) final {
   printf("Called SaveToBinary of OpenOCDModuleNode\n");
   stream->Write(fmt_);
   stream->Write(fmap_);
   stream->Write(data_);
 }

 std::string GetSource(const std::string& format) final {
   printf("Called GetSource of OpenOCDModuleNode\n");
   if (format == fmt_) return data_;
   if (source_.length() != 0) {
     return source_;
   } else {
     if (fmt_ == "ptx") return data_;
     return "";
   }
 }

 void* GetFunc(int device_id, const std::string& func_name) {
   printf("Called GetFunc of OpenOCDModuleNode\n");
   void* func;
   return func;
 }
 // get a global var from primary context in device_id
 void GetGlobal(int device_id,
                       const std::string& global_name,
                       size_t expect_nbytes) {
   printf("Called GetGlobal of OpenOCDModuleNode\n");
 }

private:
 // the binary data
 std::string data_;
 // The format
 std::string fmt_;
 // function information table.
 std::unordered_map<std::string, FunctionInfo> fmap_;
 // The cuda source.
 std::string source_;
 // internal mutex when updating the module
 std::mutex mutex_;
};

// a wrapped function class to get packed fucn.
class OpenOCDWrappedFunc {
public:
 // initialize the OpenOCD function.
 void Init(OpenOCDModuleNode* m,
           std::shared_ptr<ModuleNode> sptr,
           const std::string& func_name,
           size_t num_void_args,
           const std::vector<std::string>& thread_axis_tags) {
   printf("Called Init of OpenOCDModuleNode\n");
 }

 // invoke the function with void arguments
 void operator()(TVMArgs args,
                 TVMRetValue* rv,
                 void** void_args) const {
   printf("Called Operator() of OpenOCDModuleNode\n");
 }

private:
 // internal module
 OpenOCDModuleNode* m_;
 // the resource holder
 std::shared_ptr<ModuleNode> sptr_;
 // The name of the function.
 std::string func_name_;
 // Device function cache per device.
 // thread axis configuration
 ThreadAxisConfig thread_axis_cfg_;
};

class OpenOCDPrepGlobalBarrier {
public:
 OpenOCDPrepGlobalBarrier(OpenOCDModuleNode* m,
                       std::shared_ptr<ModuleNode> sptr)
     : m_(m), sptr_(sptr) {
   printf("Called of OpenOCDPrepGlobalBarrier\n");
 }

 void operator()(const TVMArgs& args, TVMRetValue* rv) const {
   printf("Called Operator() OpenOCDPrepGlobalBarrier\n");
 }

private:
 // internal module
 OpenOCDModuleNode* m_;
 // the resource holder
 std::shared_ptr<ModuleNode> sptr_;
 // mark as mutable, to enable lazy initialization
};

PackedFunc OpenOCDModuleNode::GetFunction(
     const std::string& name,
     const std::shared_ptr<ModuleNode>& sptr_to_self) {
  printf("Called GetFunction of OpenOCDModuleNode\n");
  PackedFunc f;
  return f;
}

Module OpenOCDModuleCreate(
   std::string data,
   std::string fmt,
   std::unordered_map<std::string, FunctionInfo> fmap,
   std::string source) {
  printf("Called OpenOCDModuleCreate of OpenOCDModuleNode\n");
 std::shared_ptr<OpenOCDModuleNode> n =
     std::make_shared<OpenOCDModuleNode>(data, fmt, fmap, source);
 return Module(n);
}

// Load module from module.
Module OpenOCDModuleLoadFile(const std::string& file_name,
                         const std::string& format) {
 printf("Called OpenOCDModuleLoadFile of OpenOCDModuleNode\n");
 std::string data;
 std::unordered_map<std::string, FunctionInfo> fmap;
 std::string fmt = GetFileFormat(file_name, format);
 std::string meta_file = GetMetaFilePath(file_name);
 std::cout << file_name << "  " << meta_file << std::endl;
 LoadBinaryFromFile(file_name, &data);
 return OpenOCDModuleCreate(data, fmt, fmap, std::string());
}

Module OpenOCDModuleLoadBinary(void* strm) {
  printf("Called OpenOCDModuleLoadBinary of OpenOCDModuleNode\n");
 dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
 std::string data;
 std::unordered_map<std::string, FunctionInfo> fmap;
 std::string fmt;
 stream->Read(&fmt);
 stream->Read(&fmap);
 stream->Read(&data);
 return OpenOCDModuleCreate(data, fmt, fmap, std::string());
}

TVM_REGISTER_GLOBAL("dump_text_c")
.set_body([](TVMArgs args, TVMRetValue* rv) {
    PackedFunc f = args[0];
    f();
 });

TVM_REGISTER_GLOBAL("module.loadfile_openocd")
.set_body([](TVMArgs args, TVMRetValue* rv) {
   *rv = OpenOCDModuleLoadFile(args[0], args[1]);
 });

// lel?
TVM_REGISTER_GLOBAL("module.loadfile_ptx")
.set_body([](TVMArgs args, TVMRetValue* rv) {
   *rv = OpenOCDModuleLoadFile(args[0], args[1]);
 });

// lel??
TVM_REGISTER_GLOBAL("module.loadbinary_openocd")
.set_body([](TVMArgs args, TVMRetValue* rv) {
   *rv = OpenOCDModuleLoadBinary(args[0]);
 });
}  // namespace runtime
}  // namespace tvm
