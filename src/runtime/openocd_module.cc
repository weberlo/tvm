/*!
*  Copyright (c) 2018 by Contributors
* \file openocd_module.cc
*/
#include <tvm/runtime/registry.h>
#include <tvm/runtime/micro_device_api.h>
#include <vector>
#include <array>
#include <string>
#include <mutex>
#include <iostream>
#include <fstream>
#include "pack_args.h"
#include "thread_storage_scope.h"
#include "meta_data.h"
#include "file_util.h"

#define PAGE_SIZE 4096

namespace tvm {
namespace runtime {
class OpenOCDModuleNode : public runtime::ModuleNode {
public:
 explicit OpenOCDModuleNode(std::string data,
                         std::string fmt,
                         std::unordered_map<std::string, FunctionInfo> fmap,
                         std::string source)
     : data_(data), fmt_(fmt), fmap_(fmap), source_(source) {
  printf("Called constructor of OpenOCDModuleNode\n");
  // TODO: nothing here, right?
 }

 // destructor
 ~OpenOCDModuleNode() {
    printf("Called destructor of OpenOCDModuleNode\n");
    Unload();
 }

 const char* type_key() const final {
   return "openocd";
 }

 PackedFunc GetFunction(
     const std::string& name,
     const std::shared_ptr<ModuleNode>& sptr_to_self) final;

 void SaveToFile(const std::string& file_name,
                 const std::string& format) final {
   // TODO: Do we need this? I think no.
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
   // TODO: Do we need this? I think no.
   printf("Called SaveToBinary of OpenOCDModuleNode\n");
   stream->Write(fmt_);
   stream->Write(fmap_);
   stream->Write(data_);
 }

 std::string GetSource(const std::string& format) final {
   // TODO: Do we need this? I think no.
   printf("Called GetSource of OpenOCDModuleNode\n");
   if (format == fmt_) return data_;
   if (source_.length() != 0) {
     return source_;
   } else {
     return "";
   }
 }

private:
  // the binary data
  std::string data_;
  // The format
  std::string fmt_;
  // function information table.
  std::unordered_map<std::string, FunctionInfo> fmap_;
  // The module source
  std::string source_;
  // internal mutex when updating the module
  std::mutex mutex_;
  // MicroDeviceAPI handle
  MicroDeviceAPI md_;

  void ExecuteCommand(std::string cmd, char* args[]) {
    // TODO: host OS specific code?
    int pid = fork();
    if (pid) {
      wait(0);
    } else {
      // Assumes cmd is in $PATH
      int ret = execvp(cmd.c_str(), args);
      CHECK(false)
        << "error in execvp " + cmd.c_str() + " " + args + "\n";
    }
  }

  FILE* ExecuteCommandWithOutput(std::string cmd) {
    FILE* f;
    f = popen(cmd.c_str(), "r");
    if (f == NULL) {
      CHECK(false)
        << "error in popen " + cmd.c_str() + "\n";
    }
    return f;
  }

  void DumpSection(std::string binary, std::string section) {
    std::string cmd = "objcopy";
    char *args[] = {cmd.c_str(), 
                    "--dump-section", 
                    "." + section "=" + section + ".bin", 
                    binary, 
                    NULL};
    ExecuteCommand(cmd, args);
  }

  void LoadSection(std::string section, void* addr) {
    std::ifstream f(section+".bin", std::ios::in | std::ios::binary | std::ios::ate);
    if (!f) {
      CHECK(false)
        << "error occurred in opening " + section + ".bin\n";
      return;
    }
    size_t size = f.tellg();
    char* buf[size];
    f.seekg(0, std::ios::beg);
    f.read(buf, size);
    f.close();
    // TODO: what is the ctx here?
    md_->WriteToMemory(ctx, addr, (int8_t *) buf, size);
  }

  void FindSectionSize(std::string binary, std::string section) {
    // TODO: add size finding function, currently using static sizes
  }

  void CustomLink(std::string object,
                  string:: binary,
                  void* text, 
                  void* data, 
                  void* bss) {
    std::string cmd = "ld";
    std::string text_addr;
    std::string data_addr;
    std::string bss_addr;
    sprintf(text_addr, "%p", text);
    sprintf(data_addr, "%p", text);
    sprintf(bss_addr, "%p", text);
    char *args[] = {cmd.c_str(), 
                    object.c_str(),
                    "-Ttext", text_addr,
                    "-Tdata", data_addr,
                    "-Tbss", bss_addr,
                    "-o", binary.c_str(), 
                    NULL};
    ExecuteCommand(cmd, args);
  }

  // load the library
  void Load(const std::string& name) {
    // TODO: static microdevice size of 30 pages, change to dynamic eventually
    md_ = MicroDeviceAPI(name.c_str(), 30 * PAGE_SIZE);
    // 10 pages each of text, data and bss
    CustomLink(object, binary, (void*)0, (void*)(10 * PAGE_SIZE), (void*)(20 * PAGE_SIZE));
    DumpSection(binary, "text");
    DumpSection(binary, "data");
    DumpSection(binary, "bss");
    LoadSection("text", (void *) 0);
    LoadSection("data", (void *)(20 * PAGE_SIZE));
    LoadSection("bss", (void *)(20 * PAGE_SIZE));
  }

  void* GetSymbol(const char* name) {
    void* addr;
    std::string cmd = "nm -C " + binary + " | grep -w " + name;
    FILE* f = ExecuteCommandWithOutput(cmd);
    fscanf(f, "%p", &addr);
    return addr;
  }

  void Unload() {
    md_->Reset();
  }
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
   m_ = m;
   sptr_ = sptr;
   entry_ = entry;
   func_name_ = func_name;
   arg_size_ = arg_size;
   thread_axis_cfg_.Init(arg_size.size(), thread_axis_tags);
 }

 // invoke the function with void arguments
 void operator()(TVMArgs args,
                 TVMRetValue* rv,
                 void** void_args) const {
   printf("Called Operator() of OpenOCDModuleNode\n");
   // TODO: Copy args to on-device section
   // TODO: Call OpenOCDMicroDeviceAPI->Run()
   m->Run();
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

// TODO: complete this by wrapping around MicroDeviceAPIs function
PackedFunc OpenOCDModuleNode::GetFunction(
     const std::string& name,
     const std::shared_ptr<ModuleNode>& sptr_to_self) {
  printf("Called GetFunction of OpenOCDModuleNode\n");
  BackendPackedCFunc faddr;
  if (name == runtime::symbol::tvm_module_main) {
    const char* entry_name = reinterpret_cast<const char*>(
        GetSymbol(runtime::symbol::tvm_module_main));
    CHECK(entry_name!= nullptr)
        << "Symbol " << runtime::symbol::tvm_module_main << " is not presented";
    faddr = reinterpret_cast<BackendPackedCFunc>(GetSymbol(entry_name));
  } else {
    faddr = reinterpret_cast<BackendPackedCFunc>(GetSymbol(name.c_str()));
  }
  if (faddr == nullptr) return PackedFunc();
  OpenOCDWrappedFunc f;
  f.Init(args);
  return PackedFunc();
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
   // TODO: Do we need this? I think no.
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
   // TODO: Do we need this? I think no.
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

TVM_REGISTER_GLOBAL("module.loadfile_openocd")
.set_body([](TVMArgs args, TVMRetValue* rv) {
   *rv = OpenOCDModuleLoadFile(args[0], args[1]);
 });

TVM_REGISTER_GLOBAL("module.loadbinary_openocd")
.set_body([](TVMArgs args, TVMRetValue* rv) {
   *rv = OpenOCDModuleLoadBinary(args[0]);
 });
}  // namespace runtime
}  // namespace tvm
