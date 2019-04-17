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

#include <iomanip>

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
   uintptr_t args_section_addr = reinterpret_cast<uintptr_t>(md_->base_addr) +
     md_->args_offs;

   uintptr_t args_addr = args_section_addr;
   md_->Write(ctx,
              GetSymbol("args"),
              (uint8_t*) &args_addr,
              (size_t) sizeof(uint8_t*));

   uintptr_t arg_type_ids_addr = args_addr + sizeof(TVMValue*) * num_args;
   md_->Write(ctx,
              GetSymbol("arg_type_ids"),
              (uint8_t*) &arg_type_ids_addr,
              (size_t) sizeof(uint8_t*));

   uintptr_t num_args_addr = arg_type_ids_addr + sizeof(const int*) * num_args;
   md_->Write(ctx,
              GetSymbol("num_args"),
              (uint8_t*) &num_args_addr,
              (size_t) sizeof(uint8_t*));

   uintptr_t fadd_addr = reinterpret_cast<uintptr_t>(md_->base_addr) +
     reinterpret_cast<uintptr_t>(GetSymbol("fadd"));
   md_->Write(ctx,
              GetSymbol("func"),
              (uint8_t*) &fadd_addr,
              (size_t) sizeof(void*));

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
    // std::cout << "Executing command: \"";
    // for (char** args_iter = const_cast<char**>(args); *args_iter; args_iter++) {
    //   std::cout << *args_iter << " ";
    // }
    // std::cout << std::endl;

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
    //std::string cmd = "objcopy";
    std::string cmd = "riscv64-unknown-elf-objcopy";
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
      // printf("Error loading section %s\n", section.c_str());
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
    {
      std::stringstream ld_script;
      ld_script << "OUTPUT_ARCH( \"riscv\" )" << std::endl;
      ld_script << "SECTIONS" << std::endl;
      ld_script << "{" << std::endl;
      ld_script << "  . = " << std::hex << text << ";" << std::endl;
      ld_script << "  .text : { *(.text) }"<< std::endl;
      ld_script << "  . = " << std::hex << data << ";" << std::endl;
      ld_script << "  .data : { *(.data) }"<< std::endl;
      ld_script << "  . = " << std::hex << bss << ";" << std::endl;
      ld_script << "  .bss : { *(.bss) }"<< std::endl;
      ld_script << "  .sbss : { *(.sbss) }"<< std::endl;
      ld_script << "}" << std::endl;
      std::ofstream ld_script_file("fadd.lds");
      ld_script_file << ld_script.str();
      ld_script_file.close();
    }

    //std::string cmd = "ld";
    //std::string cmd = "riscv64-unknown-elf-ld";
    // char text_addr[20];
    // char data_addr[20];
    // char bss_addr[20];
    // sprintf(text_addr, "%p", text);
    // sprintf(data_addr, "%p", data);
    // sprintf(bss_addr, "%p", bss);
    // if(text == NULL)
    //   sprintf(text_addr, "0x0");
    // if(data == NULL)
    //   sprintf(data_addr, "0x0");
    // if(bss == NULL)
    //   sprintf(bss_addr, "0x0");

    // const char* const args[] = {(char *) cmd.c_str(),
    //                             (char *) object.c_str(),
    //                             "-Ttext", text_addr,
    //                             "-Tdata", data_addr,
    //                             "-Tbss", bss_addr,
    //                             "-o", (char *) binary.c_str(),
    //                             nullptr};

    std::string cmd = "riscv64-unknown-elf-g++";

    // riscv64-unknown-elf-g++ -c -g -Og -o "${OBJ_NAME}" "${SRC_NAME}" -I "$HOME/tvm-riscv/include" -I "$HOME/tvm-riscv/3rdparty/dlpack/include" 
    // riscv64-unknown-elf-g++ -g -Og -T "$HOME/utvm-conf/spike.lds" -nostartfiles -o "${BIN_NAME}" "${OBJ_NAME}"

    const char* const args[] = {(char *) cmd.c_str(),
                                "-g",
                                "-Og",
                                "-T", "fadd.lds",
                                "-nostartfiles",
                                "-o", (char *) binary.c_str(),
                                (char *) object.c_str(),
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

    std::string binary = name + ".bin";
    binary_ = binary;
    CustomLink(name, binary,
        md_->base_addr + SECTION_TEXT,
        md_->base_addr + SECTION_DATA,
        md_->base_addr + SECTION_BSS);
    DumpSection(binary, "text");
    DumpSection(binary, "data");
    DumpSection(binary, "bss");
    LoadSection("text", (void *) SECTION_TEXT);
    LoadSection("data", (void *) SECTION_DATA);
    LoadSection("bss", (void *) SECTION_BSS);
  }

  /*
   * Returns the offset from the base address to the symbol with name `name`.
   */
  void* GetSymbol(const char* name) {
    uint8_t* addr;
    // Use `nm` with the `-C` option to demangle symbols before grepping.
    std::string cmd = "riscv64-unknown-elf-nm -C " + binary_ + " | grep -w " + name;
    FILE* f = ExecuteCommandWithOutput(cmd);
    if (!fscanf(f, "%p", &addr)) {
      addr = nullptr;
      std::cerr << "Could not find address for symbol \"" << name << "\"" << std::endl;
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

  /*
  BackendPackedCFunc faddr;
  if (name2 == runtime::symbol::tvm_module_main) {
    const char* entry_name = reinterpret_cast<const char*>(
        GetSymbol(runtime::symbol::tvm_module_main));
    CHECK(entry_name != nullptr)
        << "Symbol " << runtime::symbol::tvm_module_main << " is not presented";
    faddr = reinterpret_cast<BackendPackedCFunc>(GetSymbol(entry_name));
  } else {
  }
  */

  // We don't use `name`, because we always want to call `main`.  The actual
  // function routing is done at execution time, by patching a function pointer
  // that `main` calls.
  void* faddr = GetSymbol("main");
  MicroWrappedFunc f;
  f.Init(this, sptr_to_self, name, faddr);
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
    assert(false);
   *rv = MicroModuleLoadBinary(args[0]);
 });
}  // namespace runtime
}  // namespace tvm

