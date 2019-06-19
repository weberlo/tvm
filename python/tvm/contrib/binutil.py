# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Utilities for binary file manipulation"""
import os
import subprocess
from . import util
from .._ffi.base import py_str
from ..api import register_func

@register_func("tvm_callback_get_section_size")
def tvm_callback_get_section_size(binary_path, section_name, binutil_prefix):
    """Finds size of the section in the binary.
    Assumes `size` shell command exists (typically works only on Linux machines)

    Parameters
    ----------
    binary_path : str
        path of the binary file

    section_name : str
        name of section

    binutil_prefix : str
        TODO

    Return
    ------
    size : integer
        size of the section in bytes
    """
    if not os.path.isfile(binary_path):
        raise RuntimeError("no such file \"{}\"".format(binary_path))
    # TODO(weberlo): refactor all of the subprocess shit into a helper.
    # We use the "-A" flag here to get the ".rodata" section's size, which is
    # not included by default.
    size_proc = subprocess.Popen(["{}size".format(binutil_prefix), "-A", binary_path], stdout=subprocess.PIPE)
    (size_output, _) = size_proc.communicate()
    size_output = size_output.decode("utf-8")
    if size_proc.returncode != 0:
        msg = "error in finding section size:\n"
        msg += py_str(out)
        raise RuntimeError(msg)

    # nm_proc = subprocess.Popen(["{}nm".format(binutil_prefix), "-S", binary_path], stdout=subprocess.PIPE)
    # (nm_output, _) = nm_proc.communicate()
    # nm_output = nm_output.decode("utf-8")
    # if nm_proc.returncode != 0:
    #     msg = "error in finding section nm:\n"
    #     msg += py_str(out)
    #     raise RuntimeError(msg)
    # print(nm_output)

    # TODO(weberlo): Refactor this method and `*relocate_binary` so they are
    # both aware of [".bss", ".sbss", ".sdata"] being relocated to ".bss".
    SECTION_MAPPING = {
        ".text": [".text"],
        ".rodata": [".rodata"],
        ".data": [".data"],
        ".bss": [".bss", ".sbss", ".sdata"],
    }
    sections_to_sum = SECTION_MAPPING["." + section_name]
    section_size = 0
    # Skip the first two header lines in the `size` output.
    for line in size_output.split("\n")[2:]:
        tokens = list(filter(lambda s: len(s) != 0, line.split(" ")))
        if len(tokens) != 3:
            continue
        entry_name = tokens[0]
        entry_size = int(tokens[1])
        # if entry_name.startswith("." + section_name):
        #     # The `.rodata` section should be the only section for which we
        #     # need to collect the size from *multiple* entries in the command
        #     # output.
        #     if section_size != 0 and not entry_name.startswith(".rodata"):
        #         raise RuntimeError(
        #             "multiple entries in `size` output for section {}".format(section_name))
        #     section_size += entry_size
        if entry_name in sections_to_sum:
            section_size += entry_size
    # TODO(weberlo): remove any constant multiplier and calculate it correctly.
    # return section_size + 16
    return section_size + 32


AYY = False
TEMPDIR_REFS = []

@register_func("tvm_callback_relocate_binary")
def tvm_callback_relocate_binary(binary_path, text_addr, rodata_addr, data_addr, bss_addr, binutil_prefix):
    """Relocates sections in the binary to new addresses

    Parameters
    ----------
    binary_path : str
        path of the binary file

    text_addr : str
        text section absolute address

    rodata_addr : str
        rodata section absolute address

    data_addr : str
        data section absolute address

    bss_addr : str
        bss section absolute address

    binutil_prefix : str
        TODO

    Return
    ------
    rel_bin : bytearray
        the relocated binary
    """
    tmp_dir = util.tempdir()
    rel_obj_path = tmp_dir.relpath("relocated.o")
    ld_script_contents = ""
    # TODO(weberlo): this is a fukn hack
    if binutil_prefix == "riscv64-unknown-elf-":
        print("relocating RISC-V binary...")
        ld_script_contents += "OUTPUT_ARCH( \"riscv\" )\n\n"
        pass
    # TODO(weberlo): Should ".sdata" and ".sbss" be linked into the ".bss"
    # section?
    # TODO(weberlo): Generate the script in a more procedural manner.
    ld_script_contents += """
SECTIONS
{
  . = %s;
  . = ALIGN(8);
  .text :
  {
    *(.text)
    . = ALIGN(8);
    *(.text*)
  }
  . = %s;
  . = ALIGN(8);
  .rodata :
  {
    *(.rodata)
    . = ALIGN(8);
    *(.rodata*)
  }
  . = %s;
  . = ALIGN(8);
  .data :
  {
    *(.data)
    . = ALIGN(8);
    *(.data*)
  }
  . = %s;
  . = ALIGN(8);
  .bss :
  {
    *(.bss)
    . = ALIGN(8);
    *(.bss*)
    . = ALIGN(8);
    *(.sbss)
    . = ALIGN(8);
    *(.sdata)
  }
}
    """ % (text_addr, rodata_addr, data_addr, bss_addr)
    rel_ld_script_path = tmp_dir.relpath("relocated.lds")
    with open(rel_ld_script_path, "w") as f:
        f.write(ld_script_contents)
    ld_proc = subprocess.Popen(["{}ld".format(binutil_prefix), binary_path,
                                "-T", rel_ld_script_path,
                                "-o", rel_obj_path],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    (out, _) = ld_proc.communicate()
    if ld_proc.returncode != 0:
        msg = "linking error using ld:\n"
        msg += py_str(out)
        raise RuntimeError(msg)
    print("reloc'ed obj path is \"{}\"".format(rel_obj_path))

    global AYY
    if not AYY:
        with open("/home/weberlo/utvm-conf/device-gdb/.gdbinit", "r") as f:
            gdbinit_contents = f.read().split("\n")
        for i, line in enumerate(gdbinit_contents):
            if line.startswith("file"):
                gdbinit_contents[i] = "file {}".format(rel_obj_path)
                break
        with open("/home/weberlo/utvm-conf/device-gdb/.gdbinit", "w") as f:
            f.write("\n".join(gdbinit_contents))
        AYY = True
    else:
        with open("/home/weberlo/utvm-conf/device-gdb/.gdbinit", "r") as f:
            gdbinit_contents = f.read().split("\n")
        replaced_any = False
        line_to_add = "add-symbol-file {}".format(rel_obj_path)
        for i, line in enumerate(gdbinit_contents):
            if line.startswith("add-symbol-file"):
                gdbinit_contents[i] = line_to_add
                replaced_any = True
        if not replaced_any:
            gdbinit_contents.append(line_to_add)
        with open("/home/weberlo/utvm-conf/device-gdb/.gdbinit", "w") as f:
            f.write("\n".join(gdbinit_contents))
    global TEMPDIR_REFS
    TEMPDIR_REFS.append(tmp_dir)

    with open(rel_obj_path, "rb") as f:
        rel_bin = bytearray(f.read())
    return rel_bin


@register_func("tvm_callback_read_binary_section")
def tvm_callback_read_binary_section(binary, section, binutil_prefix):
    """Returns the contents of the specified section in the binary byte array

    Parameters
    ----------
    binary : bytearray
        contents of the binary

    section : str
        type of section

    binutil_prefix : str
        TODO

    Return
    ------
    section_bin : bytearray
        contents of the read section
    """
    tmp_dir = util.tempdir()
    tmp_bin = tmp_dir.relpath("temp.bin")
    tmp_section = tmp_dir.relpath("tmp_section.bin")
    with open(tmp_bin, "wb") as out_file:
        out_file.write(bytes(binary))
    objcopy_proc = subprocess.Popen(["{}objcopy".format(binutil_prefix), "--dump-section",
                                     ".{}={}".format(section, tmp_section),
                                     tmp_bin],
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT)
    (out, _) = objcopy_proc.communicate()
    if objcopy_proc.returncode != 0:
        msg = "error in using objcopy:\n"
        msg += py_str(out)
        raise RuntimeError(msg)
    if os.path.isfile(tmp_section):
        # Get section content if it exists.
        with open(tmp_section, "rb") as f:
            section_bin = bytearray(f.read())
    else:
        # Return empty bytearray if the section does not exist.
        section_bin = bytearray("", "utf-8")
    return section_bin


@register_func("tvm_callback_get_symbol_map")
def tvm_callback_get_symbol_map(binary, binutil_prefix):
    """Obtains a map of symbols to addresses in the passed binary

    Parameters
    ----------
    binary : bytearray
        contents of the binary

    binutil_prefix : str
        TODO

    Return
    ------
    map_str : str
        map of defined symbols to addresses, encoded as a series of
        alternating newline-separated keys and values
    """
    tmp_dir = util.tempdir()
    tmp_obj = tmp_dir.relpath("tmp_obj.bin")
    with open(tmp_obj, "wb") as out_file:
        out_file.write(bytes(binary))
    nm_proc = subprocess.Popen(["{}nm".format(binutil_prefix), "-C", "--defined-only", tmp_obj],
                               stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT)
    (out, _) = nm_proc.communicate()
    if nm_proc.returncode != 0:
        msg = "error in using nm:\n"
        msg += py_str(out)
        raise RuntimeError(msg)
    out = out.decode("utf8").splitlines()
    map_str = ""
    for line in out:
        line = line.split()
        map_str += line[2] + "\n"
        map_str += line[0] + "\n"
    return map_str
