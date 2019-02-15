/*!
*  Copyright (c) 2018 by Contributors
* \file micro_module.cc
*/
#include <tvm/runtime/registry.h>
#include <tvm/runtime/module.h>
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
#include "../../runtime/meta_data.h"
#include "../../runtime/pack_args.h"
#include "../../runtime/file_util.h"
#include "../../runtime/module_util.h"
#include "device_memory_offsets.h"
//#include "host_low_level_device_api.h"
#include "openocd_low_level_device_api.h"
// Thread Sleeping
#include <unistd.h>

namespace tvm {
namespace runtime {
class MicroModuleNode final : public ModuleNode {
public:
  /*
 explicit MicroModuleNode(std::string data,
                         std::string fmt,
                         std::unordered_map<std::string, FunctionInfo> fmap,
                         std::string source)
     : data_(data), fmt_(fmt), fmap_(fmap), source_(source) {
  printf("Called constructor of MicroModuleNode\n");
 }*/

 // destructor
 ~MicroModuleNode() {
    Unload();
 }

 const char* type_key() const final {
   return "micro";
 }

 PackedFunc GetFunction(
     const std::string& name,
     const std::shared_ptr<ModuleNode>& sptr_to_self) final;

 void Run(TVMContext ctx, TVMArgs args, TVMRetValue *rv, void* addr) {
	 int num_args = args.num_args;
	 void* base_addr = md_->base_addr + SECTION_ARGS;
   void* values_addr = base_addr;
   void* type_codes_addr = (char*) values_addr + sizeof(TVMValue*) * num_args;
   void* num_args_addr = (char*) type_codes_addr + sizeof(const int*) * num_args;
	 void* fadd_addr = GetSymbol("fadd");
   void* func_addr = (char*) md_->base_addr + reinterpret_cast<std::uintptr_t>(fadd_addr);
   md_->Write(ctx, GetSymbol("args"), (uint8_t*) &values_addr, (size_t) sizeof(void**));
   md_->Write(ctx, GetSymbol("arg_type_ids"), (uint8_t*)  &type_codes_addr, (size_t) sizeof(void**));
   md_->Write(ctx, GetSymbol("num_args"), (uint8_t*)  &num_args_addr, (size_t) sizeof(int32_t*));
   md_->Write(ctx, GetSymbol("func"), (uint8_t*)  &func_addr, (size_t) sizeof(void*));
   md_->Execute(ctx, args, rv, addr);
 }

 void Init(const std::string& name) {
   Load(name);
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
  // OpenOCDLowLevelDeviceAPI handle
  std::shared_ptr<OpenOCDLowLevelDeviceAPI> md_;

  // TODO: API? what is distinguishing factor btwn OpenOCD / OpenOCD here?
  std::shared_ptr<OpenOCDLowLevelDeviceAPI> OpenOCDLowLevelDeviceConnect(size_t num_bytes) {
    std::shared_ptr<OpenOCDLowLevelDeviceAPI> ret = OpenOCDLowLevelDeviceAPI::Create(num_bytes);
    return ret;
  }

  void ExecuteCommand(std::string cmd, const char* const args[]) {
    // TODO: this is linux specific code, make windows compatible eventually
    int pid = fork();
    if (pid) {
      wait(0);
    } else {
      int ret = execvp(cmd.c_str(), const_cast<char* const*>(args));
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
    const char* const args[] = {(char *) cmd.c_str(),
                          "--dump-section",
                          (char *)("." + section + "=" + section + ".bin").c_str(),
                          (char *) binary.c_str(),
                          nullptr};
    ExecuteCommand(cmd, args);
  }

  void LoadSection(std::string section, void* addr) {
    std::ifstream f(section+".bin", std::ios::in | std::ios::binary | std::ios::ate);
    if (!f) {
      printf("Error loading section %s\n", section.c_str());
      return;
    }
    size_t size = f.tellg();
    char buf[size];
    f.seekg(0, std::ios::beg);
    f.read(buf, size);
    f.close();
    md_->Write(ctx_, addr, (uint8_t *) buf, size);
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
    char text_addr[20];
    char data_addr[20];
    char bss_addr[20];
    sprintf(text_addr, "%p", text);
    sprintf(data_addr, "%p", data);
    sprintf(bss_addr, "%p", bss);
    printf("%p %p %p\n", text, data, bss);
    if(text == NULL)
      sprintf(text_addr, "0x0");
    if(data == NULL)
      sprintf(data_addr, "0x0");
    if(bss == NULL)
      sprintf(bss_addr, "0x0");

    const char* const args[] = {(char *) cmd.c_str(),
                    (char *) object.c_str(),
                    "-Ttext", text_addr,
                    "-Tdata", data_addr,
                    "-Tbss", bss_addr,
                    "-o", (char *) binary.c_str(),
                    nullptr};
    ExecuteCommand(cmd, args);
  }

  void PrintArray(uint8_t *arr, size_t len) {
    std::cout << "[";
    if (len > 0) {
      std::cout << static_cast<uint32_t>(arr[0]);
    }
    for (size_t i = 1; i < len; i++) {
      std::cout << ", " << static_cast<uint32_t>(arr[i]);
    }
    std::cout << "]" << std::endl;
  }

  void Load(const std::string& name) {
    size_t total_memory = MEMORY_SIZE;
    md_ = OpenOCDLowLevelDeviceConnect(total_memory);

    /*
    std::string binary = name + ".bin";
    binary_ = binary;
    CustomLink(name, binary,
        md_->base_addr + SECTION_TEXT,
        md_->base_addr + SECTION_DATA,
        md_->base_addr + SECTION_BSS);
    DumpSection(binary, "text");
    DumpSection(binary, "data");
    DumpSection(binary, "bss");
    LoadSection("text", NULL);
    LoadSection("data", (void *) SECTION_DATA);
    LoadSection("bss", (void *) SECTION_BSS);
    */

    /*
    TVMContext ctx;
    void *addr = (void*) 0x0;
    size_t arr_len = 5;
    uint8_t write[] = "Farts";
    //uint8_t write[] = {2, 4, 6, 8, 10};
    uint8_t read[] = {0, 0, 0, 0, 0};

    std::cout << "Write: ";
    PrintArray(write, arr_len);
    std::cout << "Read: ";
    PrintArray(read, arr_len);

    md_->Write(ctx, addr, write, arr_len);
    md_->Read(ctx, addr, read, arr_len);

    std::cout << "Write: ";
    PrintArray(write, arr_len);
    std::cout << "Read: ";
    PrintArray(read, arr_len);
    */
    md_->SendCommand("reset run");
    for (int i = 0; i < 15; i++) {
      usleep(5000000 / 15);
    }
    md_->SendCommand("reset halt");
  }

  void* GetSymbol(const char* name) {
    uint8_t* addr;
    std::string cmd = "nm -C " + binary_ + " | grep -w " + name;
    FILE* f = ExecuteCommandWithOutput(cmd);
    if (!fscanf(f, "%p", &addr)) {
      addr = nullptr;
    }
    return (void*)(addr - md_->base_addr);
  }

  void Unload() {
    md_->Reset(ctx_);
  }
};

// a wrapped function class to get packed func.
class MicroWrappedFunc {
public:
 // initialize the Micro function.
 void Init(MicroModuleNode* m,
           std::shared_ptr<ModuleNode> sptr,
           const std::string& func_name,
           void* func_addr) {
   m_ = m;
   sptr_ = sptr;
   func_name_ = func_name;
   func_addr_ = func_addr;
 }

 // invoke the function with void arguments
 void operator()(TVMArgs args,
                 TVMRetValue* rv,
                 void** void_args) const {
   m_->Run(ctx_, args, rv, func_addr_);
 }

private:
 // internal module
 MicroModuleNode* m_;
 // the resource holder
 std::shared_ptr<ModuleNode> sptr_;
 // The name of the function.
 std::string func_name_;
 // address of the function to be called
 void* func_addr_;
 // dummy context variable, unneeded for Micro
 TVMContext ctx_;
};

PackedFunc MicroModuleNode::GetFunction(
     const std::string& name,
     const std::shared_ptr<ModuleNode>& sptr_to_self) {
  BackendPackedCFunc faddr;
	std::string name2 = "main";
  if (name2 == runtime::symbol::tvm_module_main) {
    const char* entry_name = reinterpret_cast<const char*>(
        GetSymbol(runtime::symbol::tvm_module_main));
    CHECK(entry_name!= nullptr)
        << "Symbol " << runtime::symbol::tvm_module_main << " is not presented";
    faddr = reinterpret_cast<BackendPackedCFunc>(GetSymbol(entry_name));
  } else {
    faddr = reinterpret_cast<BackendPackedCFunc>(GetSymbol(name2.c_str()));
  }
  MicroWrappedFunc f;
  f.Init(this, sptr_to_self, name, (void*) faddr);
  return PackFuncVoidAddr(f, std::vector<TVMType>());
}


Module MicroModuleCreate() {
  std::shared_ptr<MicroModuleNode> n =
     std::make_shared<MicroModuleNode>();
  return Module(n);
}

// Load module from module.
Module MicroModuleLoadFile(const std::string& file_name,
                           const std::string& format) {
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt = GetFileFormat(file_name, format);
  std::string meta_file = GetMetaFilePath(file_name);
  std::cout << file_name << "  " << meta_file << std::endl;
  LoadBinaryFromFile(file_name, &data);
//  return MicroModuleCreate(data, fmt, fmap, std::string());
  return Module();
}

Module MicroModuleLoadBinary(void* strm) {
  dmlc::Stream* stream = static_cast<dmlc::Stream*>(strm);
  std::string data;
  std::unordered_map<std::string, FunctionInfo> fmap;
  std::string fmt;
  stream->Read(&fmt);
  stream->Read(&fmap);
  stream->Read(&data);
//  return MicroModuleCreate(data, fmt, fmap, std::string());
  return Module();
}

TVM_REGISTER_GLOBAL("module.loadfile_micro_dev")
.set_body([](TVMArgs args, TVMRetValue* rv) {
  std::shared_ptr<MicroModuleNode> n = std::make_shared<MicroModuleNode>();
  n->Init(args[0]);
  *rv = runtime::Module(n);
//  *rv = MicroModuleLoadFile(args[0], args[1]);
//  TODO: this doesn't seem to do anything, so rethink this
 });

TVM_REGISTER_GLOBAL("module.loadbinary_micro_dev")
.set_body([](TVMArgs args, TVMRetValue* rv) {
   *rv = MicroModuleLoadBinary(args[0]);
 });
}  // namespace runtime
}  // namespace tvm

