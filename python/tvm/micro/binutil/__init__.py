import os
import sys
from enum import Enum
from pathlib import Path

from tvm.contrib import util as _util
from tvm.contrib.binutil import run_cmd
from tvm._ffi.libinfo import find_include_path
from tvm.micro import LibType

#from . import host
#from . import riscv_spike
#from . import stm32f746xx

#class Device(Enum):
#    HOST = 0
#    RISCV_SPIKE = 1
#    STM32F746XX = 2


class MicroBinutil:
    def __init__(self, toolchain_prefix):
        print(f'creating microbinutil with {toolchain_prefix}')
        self._toolchain_prefix = toolchain_prefix

    def create_lib(self, obj_path, src_path, lib_type, options=None):
        """Compiles code into a binary for the target micro device.

        Parameters
        ----------
        obj_path : Optional[str]
            path to generated object file (defaults to same directory as `src_path`)

        src_path : str
            path to source file

        toolchain_prefix : str
            toolchain prefix to be used

        include_dev_lib_header : bool
            whether to include the device library header containing definitions of
            library functions.
        """
        print('OPTIONS')
        print(f'{obj_path}')
        print(f'{src_path}')
        print(f'{lib_type}')
        print(f'{options}')
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

        #if toolchain_prefix == 'arm-none-eabi-':
        #    device_id = 'stm32f746'
        #    base_compile_cmd += [
        #        '-mcpu=cortex-m7',
        #        '-mlittle-endian',
        #        '-mfloat-abi=hard',
        #        '-mfpu=fpv5-sp-d16',
        #        '-mthumb',
        #        '-gdwarf-5'
        #        ]
        #elif toolchain_prefix == '':
        #    device_id = 'host'
        #    if sys.maxsize > 2**32 and sys.platform.startswith('linux'):
        #        base_compile_cmd += ['-mcmodel=large']
        #else:
        #    assert False

        src_paths = []
        include_paths = find_include_path() + [_get_micro_host_driven_dir()]
        ld_script_path = None
        tmp_dir = _util.tempdir()
        if lib_type == LibType.RUNTIME:
            import glob
            dev_dir = _get_micro_device_dir() + '/' + self.device_id()

            print(dev_dir)
            dev_src_paths = glob.glob(f'{dev_dir}/*.[csS]')
            print(dev_src_paths)
            # there needs to at least be a utvm_timer.c file
            assert dev_src_paths

            src_paths += dev_src_paths
            # TODO: configure this
            #include_paths += [dev_dir]
            CMSIS_PATH = '/home/pratyush/Code/nucleo-interaction-from-scratch/stm32-cube'
            include_paths += [f'{CMSIS_PATH}/Drivers/CMSIS/Include']
            include_paths += [f'{CMSIS_PATH}/Drivers/CMSIS/Device/ST/STM32F7xx/Include']
            include_paths += [f'{CMSIS_PATH}/Drivers/STM32F7xx_HAL_Driver/Inc']
            include_paths += [f'{CMSIS_PATH}/Drivers/BSP/STM32F7xx_Nucleo_144']
            include_paths += [f'{CMSIS_PATH}/Drivers/BSP/STM32746G-Discovery']
        elif lib_type == LibType.OPERATOR:
            # Create a temporary copy of the source, so we can inject the dev lib
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
            curr_obj_path = self._get_unique_obj_name(src_path, prereq_obj_paths, tmp_dir)
            prereq_obj_paths.append(curr_obj_path)
            curr_compile_cmd = base_compile_cmd + [src_path, '-o', curr_obj_path]
            run_cmd(curr_compile_cmd)

        ld_cmd = [f'{self.toolchain_prefix()}ld', '-relocatable']
        ld_cmd += prereq_obj_paths
        ld_cmd += ['-o', obj_path]
        run_cmd(ld_cmd)
        print(f'compiled obj {obj_path}')

    def _get_unique_obj_name(self, src_path, obj_paths, tmp_dir):
        res = tmp_dir.relpath(Path(src_path).with_suffix('.o').name)
        i = 2
        # if the name collides, try increasing numeric suffixes until the name doesn't collide
        while res in obj_paths:
            res = tmp_dir.relpath(Path(os.path.basename(src_path).split('.')[0] + str(i)).with_suffix('.o').name)
            i += 1
        return res

    def device_id(self):
        raise RuntimeError('no device ID for abstract MicroBinutil')

    def toolchain_prefix(self):
        return self._toolchain_prefix


class HostBinutil(MicroBinutil):
    def __init__(self):
        super(HostBinutil, self).__init__('')

    def create_lib(self, obj_path, src_path, lib_type, options=None):
        if options is None:
            options = []
        if sys.maxsize > 2**32 and sys.platform.startswith('linux'):
            options += ['-mcmodel=large']
        super(HostBinutil, self).create_lib(obj_path, src_path, lib_type, options=options)

    def device_id(self):
        return 'host'


class ArmBinutil(MicroBinutil):
    def __init__(self):
        super(ArmBinutil, self).__init__('arm-none-eabi-')

    def create_lib(self, obj_path, src_path, lib_type, options=None):
        if options is None:
            options = []
        options += [
            '-mcpu=cortex-m7',
            '-mlittle-endian',
            '-mfloat-abi=hard',
            '-mfpu=fpv5-sp-d16',
            '-mthumb',
            '-gdwarf-5',
            '-DSTM32F746xx'
            ]
        super(ArmBinutil, self).create_lib(obj_path, src_path, lib_type, options=options)

    def device_id(self):
        return 'stm32f746xx'


#def get_device_binutil(device):
#    # TODO make all binutils singletons if they remain stateless
#    if device_id == Device.HOST:
#        return HostBinutil()
#    elif device_id == Device.RISCV_SPIKE:
#        return SpikeBinutil()
#    elif device_id == Device.STM32F746XX:
#        return ArmBinutil()
#    else:
#        raise RuntimeError(f'invalid device: {device}')


def _get_micro_host_driven_dir():
    """Get directory path for uTVM host-driven runtime source files.

    Return
    ------
    micro_device_dir : str
        directory path
    """
    micro_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
    micro_host_driven_dir = os.path.join(micro_dir, "..", "..", "..", "..",
                                         "src", "runtime", "micro", "host_driven")
    return micro_host_driven_dir


def _get_micro_device_dir():
    """Get directory path for TODO

    Return
    ------
    micro_device_dir : str
        directory path
    """
    micro_dir = os.path.dirname(os.path.realpath(os.path.expanduser(__file__)))
    micro_device_dir = os.path.join(micro_dir, "..", "..", "..", "..",
                                    "src", "runtime", "micro", "device")
    return micro_device_dir


