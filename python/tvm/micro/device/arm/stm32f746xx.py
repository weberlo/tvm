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
"""Compilation and config definitions for Arm STM32F746XX devices"""
from .. import create_micro_lib_base, register_device, gen_mem_layout, MemConstraint

DEVICE_ID = 'arm.stm32f746xx'
TOOLCHAIN_PREFIX = 'arm-none-eabi-'
WORD_SIZE = 4
#
# [Device Memory Layout]
#   RAM   (rwx) : START = 0x20000000, LENGTH = 320K
#   Flash (rx)  : START = 0x8000000,  LENGTH = 1024K
#
BASE_ADDR = 0x20000000
AVAILABLE_MEM = 320000
DEFAULT_SECTION_CONSTRAINTS = {
    'text': (18000, MemConstraint.ABSOLUTE_BYTES),
    'rodata': (100, MemConstraint.ABSOLUTE_BYTES),
    'data': (100, MemConstraint.ABSOLUTE_BYTES),
    'bss': (600, MemConstraint.ABSOLUTE_BYTES),
    'args': (4096, MemConstraint.ABSOLUTE_BYTES),
    'heap': (100.0, MemConstraint.WEIGHT),
    'workspace': (64000, MemConstraint.ABSOLUTE_BYTES),
    'stack': (32, MemConstraint.ABSOLUTE_BYTES),
}

def create_micro_lib(obj_path, src_path, lib_type, options=None, lib_src_paths=None):
    """Wrapper over `create_micro_lib_base` to add device-specific options

    Parameters
    ----------
    obj_path : str
        path to generated object file

    src_path : str
        path to source file

    lib_type : micro.LibType
        whether to compile a MicroTVM runtime or operator library

    options : Optional[List[str]]
        additional options to pass to GCC

    lib_src_paths : Optional[List[str]]
        TODO
    """
    if options is None:
        options = []
    options += [
        '-march=armv7e-m',
        '-mcpu=cortex-m7',
        '-mlittle-endian',
        '-mfloat-abi=hard',
        # TODO try this one?
        #'-mfpu=fpv5-d16',
        '-mfpu=fpv5-sp-d16',
        '-mthumb',
        '-ffast-math',
        '-gdwarf-5',
        '-DARM_MATH_CM7',
        '-D__FPU_PRESENT=1U',
        '-DARM_MATH_DSP',
        ]
    create_micro_lib_base(
        obj_path, src_path, TOOLCHAIN_PREFIX, DEVICE_ID, lib_type, options=options, lib_src_paths=lib_src_paths)


def generate_config(server_addr, server_port, section_constraints=None):
    """Generates a configuration for Arm STM32F746XX devices

    Parameters
    ----------
    server_addr : str
        address of OpenOCD server to connect to

    server_port : int
        port of OpenOCD server to connect to

    TODO correct type annotation?
    section_constraints: Optional[Dict[str, Tuple[Number, MemConstraint]]]
        TODO

    Return
    ------
    config : Dict[str, Any]
        MicroTVM config dict for this device
    """
    #'mem_layout': gen_mem_layout(OrderedDict([
    #    ('text', (10000, MemConstraint.ABSOLUTE_BYTES)),
    #    ('rodata', (100, MemConstraint.ABSOLUTE_BYTES)),
    #    ('data', (100, MemConstraint.ABSOLUTE_BYTES)),
    #    ('bss', (600, MemConstraint.ABSOLUTE_BYTES)),
    #    ('args', (4096, MemConstraint.ABSOLUTE_BYTES)),
    #    ('heap', (50.0, MemConstraint.WEIGHT)),
    #    ('workspace', (2048, MemConstraint.ABSOLUTE_BYTES)),
    #    ('stack', (32, MemConstraint.ABSOLUTE_BYTES)),
    #])),
    if section_constraints is None:
        section_constraints = DEFAULT_SECTION_CONSTRAINTS
    return {
        'device_id': DEVICE_ID,
        'toolchain_prefix': TOOLCHAIN_PREFIX,
        'mem_layout': gen_mem_layout(BASE_ADDR, AVAILABLE_MEM, WORD_SIZE, section_constraints),
        'word_size': WORD_SIZE,
        'thumb_mode': True,
        'use_device_timer': False,
        'comms_method': 'openocd',
        'server_addr': server_addr,
        'server_port': server_port,
    }


register_device(DEVICE_ID, {
    'create_micro_lib': create_micro_lib,
    'generate_config': generate_config,
})
