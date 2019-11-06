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

"""Base definitions for micro."""

from __future__ import absolute_import

import logging
import os
import sys
from enum import Enum
from pathlib import Path

import tvm
from tvm.contrib import util as _util
from tvm.contrib import cc as _cc
from .._ffi.function import _init_api
from .._ffi.libinfo import find_include_path

class LibType(Enum):
    RUNTIME = 0
    OPERATOR = 1


class Session:
    """MicroTVM Device Session

    Parameters
    ----------
    device_type : str
        type of low-level device

    toolchain_prefix : str
        toolchain prefix to be used. For example, a prefix of
        "riscv64-unknown-elf-" means "riscv64-unknown-elf-gcc" is used as
        the compiler and "riscv64-unknown-elf-ld" is used as the linker,
        etc.

    Example
    --------
    .. code-block:: python

      c_mod = ...  # some module generated with "c" as the target
      dev_config = micro.device.stm32f746xx.default_config('127.0.0.1', 6666)
      with tvm.micro.Session(dev_config):
          c_mod.export_library(
              lib_obj_path,
              fcompile=tvm.micro.cross_compiler(dev_config['binutil'],
              tvm.micro.LibType.OPERATOR))
          micro_mod = tvm.module.load(lib_obj_path, 'micro_dev')
    """

    def __init__(self, config):
        self._check_system()
        self._check_config(config)

        self.binutil = tvm.micro.device.get_binutil(config['binutil'])
        self.mem_layout = config['mem_layout']
        self.word_size = config['word_size']
        self.thumb_mode = config['thumb_mode']
        self.comms_method = config['comms_method']

        # First, find and compile runtime library.
        runtime_src_path = os.path.join(_get_micro_host_driven_dir(), 'utvm_runtime.c')
        tmp_dir = _util.tempdir()
        runtime_obj_path = tmp_dir.relpath('utvm_runtime.obj')
        self.binutil.create_lib(runtime_obj_path, runtime_src_path, LibType.RUNTIME)
        #input(f'check {runtime_obj_path}: ')

        comms_method = config['comms_method']
        if comms_method == 'openocd':
            server_addr = config['server_addr']
            server_port = config['server_port']
        elif comms_method == 'host':
            server_addr = ''
            server_port = 0
        else:
            raise RuntimeError(f'unknown communication method: f{self.comms_method}')

        self.module = _CreateSession(
            comms_method,
            runtime_obj_path,
            self.binutil.toolchain_prefix(),
            self.mem_layout['text'].get('start', 0),
            self.mem_layout['text']['size'],
            self.mem_layout['rodata'].get('start', 0),
            self.mem_layout['rodata']['size'],
            self.mem_layout['data'].get('start', 0),
            self.mem_layout['data']['size'],
            self.mem_layout['bss'].get('start', 0),
            self.mem_layout['bss']['size'],
            self.mem_layout['args'].get('start', 0),
            self.mem_layout['args']['size'],
            self.mem_layout['heap'].get('start', 0),
            self.mem_layout['heap']['size'],
            self.mem_layout['workspace'].get('start', 0),
            self.mem_layout['workspace']['size'],
            self.mem_layout['stack'].get('start', 0),
            self.mem_layout['stack']['size'],
            self.word_size,
            self.thumb_mode,
            server_addr,
            server_port)
        self._enter = self.module['enter']
        self._exit = self.module['exit']

    def create_micro_mod(self, c_mod):
        """Produces a micro module from a given module.

        Parameters
        ----------
        c_mod : tvm.module.Module
            module with "c" as its target backend

        Return
        ------
        micro_mod : tvm.module.Module
            micro module for the target device
        """
        print('[create_micro_mod]')
        temp_dir = _util.tempdir()
        lib_obj_path = temp_dir.relpath('dev_lib.obj')
        c_mod.export_library(
                lib_obj_path,
                fcompile=cross_compiler(self.binutil, LibType.OPERATOR))
        micro_mod = tvm.module.load(lib_obj_path)
        return micro_mod

    def _check_system(self):
        """Check if the user's system is supported by MicroTVM.

        Raises error if not supported.
        """
        if not sys.platform.startswith('linux'):
            raise RuntimeError('MicroTVM is currently only supported on Linux')
        # TODO(weberlo): Add 32-bit support.
        # It's primarily the compilation pipeline that isn't compatible.
        if sys.maxsize <= 2**32:
            raise RuntimeError('MicroTVM is currently only supported on 64-bit platforms')

    def _check_config(self, config):
        """Check if the given configuration is valid."""
        #if device_type == "host":
        #    pass
        #elif device_type == "openocd":
        #    assert "base_addr" in args
        #    assert "server_addr" in args
        #    assert "port" in args

    def __enter__(self):
        self._enter()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self._exit()


def cross_compiler(dev_binutil, lib_type):
    """Creates a cross compile function that wraps `create_micro_lib`.

    For use in `tvm.module.Module.export_library`.

    Parameters
    ----------
    lib_type: DFSDF

    Return
    ------
    func : Callable[[str, str, Optional[str]], None]
        cross compile function taking a destination path for the object file
        and a path for the input source file.

    Example
    --------
    .. code-block:: python

      c_mod = ...  # some module generated with "c" as the target
      fcompile = tvm.micro.cross_compiler(lib_type=LibType.OPERATOR)
      c_mod.export_library('dev_lib.obj', fcompile=fcompile)
    """
    if isinstance(dev_binutil, str):
        dev_binutil = tvm.micro.device.get_binutil(dev_binutil)

    def compile_func(obj_path, src_path, **kwargs):
        if isinstance(obj_path, list):
            obj_path = obj_path[0]
        if isinstance(src_path, list):
            src_path = src_path[0]
        dev_binutil.create_lib(obj_path, src_path, lib_type, kwargs.get('options', None))
    return _cc.cross_compiler(compile_func, output_format='obj')


def _get_micro_host_driven_dir():
    """Get directory path for uTVM host-driven runtime source files.

    Return
    ------
    micro_device_dir : str
        directory path
    """
    micro_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
    micro_host_driven_dir = os.path.join(micro_dir, '..', '..', '..',
                                         'src', 'runtime', 'micro', 'host_driven')
    return micro_host_driven_dir


def _get_micro_device_dir():
    """Get directory path for TODO

    Return
    ------
    micro_device_dir : str
        directory path
    """
    micro_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
    micro_device_dir = os.path.join(micro_dir, '..', '..', '..',
                                    'src', 'runtime', 'micro', 'device')
    return micro_device_dir


_init_api('tvm.micro', 'tvm.micro.base')
