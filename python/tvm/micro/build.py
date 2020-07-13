import copy
import glob
import logging
import os
import re
from tvm.contrib import util


_LOG = logging.getLogger(__name__)


class Workspace:

  def __init__(self, root=None, debug=False):
    if debug or root is not None:
      with util.TempDirectory.set_keep_for_debug():
        self.tempdir = util.tempdir(custom_path=root)
        _LOG.info('Created debug mode workspace at: %s', self.tempdir.temp_dir)
    else:
      self.tempdir = util.tempdir()

  def relpath(self, path):
    return self.tempdir.relpath(path)

  def listdir(self):
    return self.tempdir.listdir()

  @property
  def path(self):
    return self.tempdir.temp_dir


FUNC_RE = re.compile('^TVM_DLL int32_t ([^(]+)\(')


def _generate_mod_wrapper(src_path):
  funcs = []
  with open(src_path) as src_f:
    for line in src_f:
      m = FUNC_RE.match(line)
      if m:
        funcs.append(m.group(1))

  encoded_funcs = f'\\{len(funcs):03o}' + '\\0'.join(funcs)
  lines = [
      '#include <tvm/runtime/c_runtime_api.h>',
      '#include <tvm/runtime/crt/module.h>',
      '#include <stdio.h>',
      '',
      'static TVMBackendPackedCFunc funcs[] = {',
  ]
  for f in funcs:
    lines.append(f'    &{f},')
  lines += [
      '};',
      'static const TVMFuncRegistry system_lib_registry = {',
      f'       "{encoded_funcs}\\0",',
      '        funcs,',
      '};',
      'static const TVMModule system_lib = {',
      '    &system_lib_registry,',
      '};',
      '',
      'const TVMModule* TVMSystemLibEntryPoint(void) {',
      '    fprintf(stderr, "create system lib!! %p\\n", system_lib.registry->funcs[0]);',
      '    return &system_lib;',
      '}',
      '',   # blank line to end the file
  ]
  with open(src_path, 'a') as wrapper_f:
    wrapper_f.write('\n'.join(lines))


CRT_RUNTIME_LIB_NAMES = ['common', 'rpc_server']


TVM_ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))


CRT_ROOT_DIR = os.path.join(TVM_ROOT_DIR, 'src', 'runtime', 'crt')


RUNTIME_LIB_SRC_DIRS = (
  [os.path.join(TVM_ROOT_DIR,
                '3rdparty/mbed-os/targets/TARGET_NORDIC/TARGET_NRF5x/TARGET_SDK_11/libraries/crc16')] +
  [os.path.join(CRT_ROOT_DIR, n) for n in CRT_RUNTIME_LIB_NAMES])


RUNTIME_SRC_REGEX = re.compile('^.*\.cc?$', re.IGNORECASE)


_CRT_DEFAULT_OPTIONS = {
  'ccflags': ['-std=c++11'],
  'include_dirs': [f'{TVM_ROOT_DIR}/include',
                   f'{TVM_ROOT_DIR}/3rdparty/dlpack/include',
                   f'{TVM_ROOT_DIR}/3rdparty/mbed-os/targets/TARGET_NORDIC/TARGET_NRF5x/TARGET_SDK_11/libraries/crc16/',
                   f'{TVM_ROOT_DIR}/3rdparty/dmlc-core/include'],
}


def DefaultOptions():
  return copy.deepcopy(_CRT_DEFAULT_OPTIONS)


def build_static_runtime(workspace, compiler, module, lib_opts=None, bin_opts=None):
  """Build the on-device runtime, statically linking the given modules.

  Parameters
  ----------
  compiler : tvm.micro.Compiler
      Compiler instance used to build the runtime.

  module : IRModule
      Module to statically link.

  lib_opts : dict
      Extra kwargs passed to Library(),

  bin_opts : dict
      Extra kwargs passed to Binary(),

  Returns
  -------
  MicroBinary :
      The compiled runtime.
  """
  lib_opts = _CRT_DEFAULT_OPTIONS if lib_opts is None else lib_opts
  bin_opts = _CRT_DEFAULT_OPTIONS if bin_opts is None else bin_opts

  crt_path = os.path.realpath(
    os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src', 'runtime', 'crt'))

  mod_build_dir = workspace.relpath(os.path.join('build', 'module'))
  os.makedirs(mod_build_dir)
  mod_src_dir = workspace.relpath(os.path.join('src', 'module'))
  os.makedirs(mod_src_dir)
  mod_src_path = os.path.join(mod_src_dir, 'module.c')
  module.save(mod_src_path, 'cc')

#  mod_wrapper_path = os.path.join(mod_src_dir, 'module-wrapper.c')
  _generate_mod_wrapper(mod_src_path)

  libs = []
  libs.append(compiler.Library(mod_build_dir, [mod_src_path], lib_opts))

  for lib_src_dir in RUNTIME_LIB_SRC_DIRS:
    lib_name = os.path.basename(lib_src_dir)
    lib_build_dir = workspace.relpath(f'build/{lib_name}')
    os.makedirs(lib_build_dir)

    lib_srcs = []
    for p in os.listdir(lib_src_dir):
      if RUNTIME_SRC_REGEX.match(p):
        lib_srcs.append(os.path.join(lib_src_dir, p))

    libs.append(compiler.Library(lib_build_dir, lib_srcs, lib_opts))

  runtime_build_dir = workspace.relpath(f'build/runtime')
  os.makedirs(runtime_build_dir)
  return compiler.Binary(runtime_build_dir, libs, bin_opts)
