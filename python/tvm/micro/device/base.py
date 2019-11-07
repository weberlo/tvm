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
"""Base definitions for MicroTVM config"""

import glob
import os
import sys
from enum import Enum
from pathlib import Path

from tvm.contrib import util as _util
from tvm.contrib.binutil import run_cmd
from tvm._ffi.libinfo import find_include_path
from tvm.micro import LibType, get_micro_host_driven_dir, get_micro_device_dir

BINUTIL_REGISTRY = {}

def register_binutil(binutil):
    name = binutil.device_id()
    if name in BINUTIL_REGISTRY:
        raise RuntimeError(f'"{name}" already exists in the binutil registry')
    BINUTIL_REGISTRY[name] = binutil


def get_binutil(device_id):
    """Get matching MicroBinutil subclass from `device_id`

    Parameters
    ----------
    device_id : str
        unique identifier for the target device

    Return
    ------
    binutil : MicroBinutil
        MicroBinutil subclass
    """
    if device_id not in BINUTIL_REGISTRY:
        raise RuntimeError(f'"{device_id}" does not exist in the binutil registry')
    binutil = BINUTIL_REGISTRY[device_id]
    return binutil()


class MicroBinutil:
    """Base class for GCC-specific library compilation for MicroTVM

    Parameters
    ----------
    toolchain_prefix : str
        toolchain prefix to be used. For example, a prefix of
        "riscv64-unknown-elf-" means "riscv64-unknown-elf-gcc" is used as
        the compiler and "riscv64-unknown-elf-ld" is used as the linker,
        etc.
    """
    def __init__(self, toolchain_prefix):
        self._toolchain_prefix = toolchain_prefix

    def create_lib(self, obj_path, src_path, lib_type, options=None):
        """Compiles code into a binary for the target micro device.

        Parameters
        ----------
        obj_path : Optional[str]
            path to generated object file (defaults to same directory as `src_path`)

        src_path : str
            path to source file

        lib_type : micro.LibType
            whether to compile a MicroTVM runtime or operator library

        options : List[str]
            additional options to pass to GCC
        """
        print('[MicroBinutil.create_lib]')
        print('  EXTENDED OPTIONS')
        print(f'    {obj_path}')
        print(f'    {src_path}')
        print(f'    {lib_type}')
        print(f'    {options}')
        base_compile_cmd = [
                f'{self.toolchain_prefix()}gcc',
                '-std=c11',
                '-Wall',
                '-Wextra',
                '--pedantic',
                '-c',
                '-O0',
                '-g',
                '-nostartfiles',
                '-nodefaultlibs',
                '-nostdlib',
                '-fdata-sections',
                '-ffunction-sections',
                ]
        if options is not None:
            base_compile_cmd += options

        src_paths = []
        include_paths = find_include_path() + [get_micro_host_driven_dir()]
        ld_script_path = None
        tmp_dir = _util.tempdir()
        if lib_type == LibType.RUNTIME:
            dev_dir = self._get_device_source_dir()

            print(dev_dir)
            dev_src_paths = glob.glob(f'{dev_dir}/*.[csS]')
            print(dev_src_paths)
            # there needs to at least be a utvm_timer.c file
            assert dev_src_paths
            assert 'utvm_timer.c' in map(os.path.basename, dev_src_paths)

            src_paths += dev_src_paths
        elif lib_type == LibType.OPERATOR:
            # create a temporary copy of the source, so we can inject the dev lib
            # header without modifying the original.
            temp_src_path = tmp_dir.relpath('temp.c')
            with open(src_path, 'r') as f:
                src_lines = f.read().splitlines()
            src_lines.insert(0, '#include "utvm_device_dylib_redirect.c"')
            with open(temp_src_path, 'w') as f:
                f.write('\n'.join(src_lines))
            src_path = temp_src_path

            base_compile_cmd += ['-c']
        else:
            raise RuntimeError('unknown lib type')

        src_paths += [src_path]

        print(f'include paths: {include_paths}')
        for path in include_paths:
            base_compile_cmd += ['-I', path]

        prereq_obj_paths = []
        for src_path in src_paths:
            curr_obj_path = Path(src_path).with_suffix('.o').name
            assert curr_obj_path not in prereq_obj_paths
            prereq_obj_paths.append(curr_obj_path)
            curr_compile_cmd = base_compile_cmd + [src_path, '-o', curr_obj_path]
            run_cmd(curr_compile_cmd)

        ld_cmd = [f'{self.toolchain_prefix()}ld', '-relocatable']
        ld_cmd += prereq_obj_paths
        ld_cmd += ['-o', obj_path]
        run_cmd(ld_cmd)

    def _get_device_source_dir(self):
        """Grabs the source directory for device-specific uTVM files"""
        dev_subdir = '/'.join(self.__class__.device_id().split('.'))
        return get_micro_device_dir() + '/' + dev_subdir

    def device_id(self):
        raise RuntimeError('no device ID for abstract MicroBinutil')

    def toolchain_prefix(self):
        return self._toolchain_prefix
