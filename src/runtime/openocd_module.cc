/*!
*  Copyright (c) 2018 by Contributors
* \file openocd_module.cc
*/
#include <tvm/runtime/registry.h>
#include <tvm/runtime/micro_device_api.h>
#include <dmlc/memory_io.h>
#include <vector>
#include <array>
#include <string>
#include <mutex>
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <stdio.h>
#include "pack_args.h"
#include "meta_data.h"
#include "file_util.h"
#include "module_util.h"

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
     return "";
   }
 }

 void Run(TVMContext ctx, TVMArgs args, TVMRetValue *rv, void* addr) {
   md_->Execute(ctx, args, rv, addr);
 }

 void Init(const std::string& name) {
   Load(name);
   if (auto *ctx_addr =
       reinterpret_cast<void**>(GetSymbol(runtime::symbol::tvm_module_ctx))) {
     *ctx_addr = this;
   }
   InitContextFunctions([this](const char* fname) {
       return GetSymbol(fname);
     });
   // Load the imported modules
   const char* dev_mblob =
       reinterpret_cast<const char*>(
           GetSymbol(runtime::symbol::tvm_dev_mblob));
   if (dev_mblob != nullptr) {
     ImportModuleBlob(dev_mblob, &imports_);
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
  // The module binary 
  std::string binary_;
  // internal mutex when updating the module
  std::mutex mutex_;
  // some context variable - unneeded for now
  TVMContext ctx_;
  // MicroDeviceAPI handle
  std::shared_ptr<MicroDeviceAPI> md_;

  // TODO: API okay? what is distinguishing factor btwn x86 / OpenOCD here?
  std::shared_ptr<MicroDeviceAPI> MicroDeviceConnect(size_t num_bytes) {
    return MicroDeviceAPI::Create(num_bytes);
  }

  void ExecuteCommand(std::string cmd, char* args[]) {
    // TODO: this is linux specific code, make windows compatible eventually
    int pid = fork();
    if (pid) {
      wait(0);
    } else {
      // Assumes cmd is in $PATH
      int ret = execvp(cmd.c_str(), args);
      CHECK(ret == 0)
        << "error in execvp " + cmd + "\n";
    }
  }

  FILE* ExecuteCommandWithOutput(std::string cmd) {
    FILE* f;
    f = popen(cmd.c_str(), "r");
    if (f == NULL) {
      CHECK(false)
        << "error in popen " + cmd + "\n";
    }
    return f;
  }

  void DumpSection(std::string binary, std::string section) {
    std::string cmd = "objcopy";
    char* args[] = {(char *) cmd.c_str(), 
                    "--dump-section", 
                    (char *)("." + section + "=" + section + ".bin").c_str(), 
                    (char *) binary.c_str(), 
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
    char buf[size];
    f.seekg(0, std::ios::beg);
    f.read(buf, size);
    f.close();
    md_->WriteToMemory(ctx_, addr, (uint8_t *) buf, size);
  }

  void FindSectionSize(std::string binary, std::string section) {
    // TODO: add size finding function, currently using static sizes
  }

  void CustomLink(std::string object,
                  std::string binary,
                  void* text, 
                  void* data, 
                  void* bss) {
    std::string cmd = "ld";
    char text_addr[16];
    char data_addr[16];
    char bss_addr[16];
    sprintf(text_addr, "%p", text);
    sprintf(data_addr, "%p", data);
    sprintf(bss_addr, "%p", bss);
    char* args[] = {(char *) cmd.c_str(), 
                    (char *) object.c_str(),
                    "-Ttext", text_addr,
                    "-Tdata", data_addr,
                    "-Tbss", bss_addr,
                    "-o", (char *) binary.c_str(), 
                    NULL};
    ExecuteCommand(cmd, args);
  }

  // load the library
  void Load(const std::string& name) {
    // TODO: static microdevice size of 50 pages, change to dynamic eventually
    // TODO: function call arguments in callargs section of 10 pages
    // what to do with the name? that should be the object right
    size_t total_memory = 50 * PAGE_SIZE;
    md_ = MicroDeviceConnect(total_memory);
    // maybe global ptr like with DeviceAPI?
    // TODO: where to get these binary and object file/names from?
    std::string binary = "";
    binary_ = binary;
    std::string object = "";
    // 10 pages each of text, data and bss
    CustomLink(object, binary, (void*)0, (void*)(10 * PAGE_SIZE), (void*)(20 * PAGE_SIZE));
    DumpSection(binary, "text");
    DumpSection(binary, "data");
    DumpSection(binary, "bss");
    LoadSection("text", (void *) 0);
    LoadSection("data", (void *)(10 * PAGE_SIZE));
    LoadSection("bss", (void *)(20 * PAGE_SIZE));
  }

  void* GetSymbol(const char* name) {
    void* addr;
    std::string cmd = "nm -C " + binary_ + " | grep -w " + name;
    FILE* f = ExecuteCommandWithOutput(cmd);
    if (!fscanf(f, "%p", &addr)) {
      addr = nullptr;
    }
    return addr;
  }

  void Unload() {
    md_->Reset(ctx_);
  }
};

// a wrapped function class to get packed fucn.
class OpenOCDWrappedFunc {
public:
 // initialize the OpenOCD function.
 void Init(OpenOCDModuleNode* m,
           std::shared_ptr<ModuleNode> sptr,
           const std::string& func_name,
           size_t num_void_args) {
   printf("Called Init of OpenOCDModuleNode\n");
   m_ = m;
   sptr_ = sptr;
   func_name_ = func_name;
   // TODO: initialize offset addr of function
 }

 // invoke the function with void arguments
 void operator()(TVMArgs args,
                 TVMRetValue* rv,
                 void** void_args) const {
   printf("Called Operator() of OpenOCDModuleNode\n");
   m_->Run(ctx_, args, rv, addr);
 }

private:
 // internal module
 OpenOCDModuleNode* m_;
 // the resource holder
 std::shared_ptr<ModuleNode> sptr_;
 // The name of the function.
 std::string func_name_;
 // address of the function to be called
 void* addr; 
 // dummy context variable, unneeded for OpenOCD
 TVMContext ctx_; 
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
  // TODO: wrap f
  // f.Init(args);
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

// re-examine
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
