import contextlib
import copy
import glob
import os

import numpy

import tvm
import tvm.micro

from tvm.contrib import device_util
from tvm.contrib import util

DEBUG = False

def _setup(test_func):
  # target = tvm.target.create('c -mcpu=x86-64')
  target = tvm.target.create('c -mcpu=native')

  A = tvm.te.placeholder((2,), dtype='int8')
  B = tvm.te.placeholder((1,), dtype='int8')
  C = tvm.te.compute(A.shape, lambda i: A[i] + B[0], name='C')

  s = tvm.te.create_schedule(C.op)

  with tvm.transform.PassContext(opt_level=3, config={'tir.disable_vectorize': True}):
    mod = tvm.build(s, [A, B, C], target, target_host=target, name='add')

  workspace = tvm.micro.Workspace(debug=True)

  compiler = tvm.micro.DefaultCompiler(target=target)
  opts = tvm.micro.DefaultOptions()
  opts['include_dirs'].append(os.path.join(tvm.micro.TVM_ROOT_DIR, 'src', 'runtime', 'crt', 'host'))
  # lib_opts = copy.deepcopy(bin_opts)
  # lib_opts['profile']['common'].append('-Werror')
  # lib_opts['cflags'] = ['-Wno-error=incompatible-pointer-types']

  micro_binary = tvm.micro.build_static_runtime(workspace, compiler, mod, opts, opts)
  device_transport = device_util.DeviceTransportLauncher({
    'num_instances': 1,
    'use_tracker': False,
  })

  # with contextlib.ExitStack() as exit_stack:
  flasher_kw = {
    'debug': DEBUG,
  }

  flasher = compiler.Flasher(**flasher_kw)
  with tvm.micro.Session(binary=micro_binary, flasher=flasher) as sess:
    test_func(sess)


def test_compile_runtime(sess):
  """Test compiling the on-device runtime."""
  A_data = tvm.nd.array(numpy.array([2, 3], dtype='int8'), ctx=sess.context)
  B_data = tvm.nd.array(numpy.array([4], dtype='int8'), ctx=sess.context)
  C_data = tvm.nd.array(numpy.array([0, 0], dtype='int8'), ctx=sess.context)

  system_lib = sess._rpc.system_lib()
  print('got system lib', system_lib)
  system_lib.get_function('add')(A_data, B_data, C_data)
  system_lib.get_function('add')
  print('got data!', C_data.asnumpy())
  assert (C_data.asnumpy() == numpy.array([6, 7])).all()


def test_dev_timer(sess):
  system_lib = sess._rpc.system_lib()
  print('got system lib', system_lib)
  start_timer = system_lib.get_function('TVMPlatformTimerStart')
  stop_timer = system_lib.get_function('TVMPlatformTimerStop')
  start_timer()
  exec_time = stop_timer()
  print(f'exec time took {exec_time} seconds')


def test_time_evaluator(sess):
  A_data = tvm.nd.array(numpy.array([2, 3], dtype='int8'), ctx=sess.context)
  B_data = tvm.nd.array(numpy.array([4], dtype='int8'), ctx=sess.context)
  C_data = tvm.nd.array(numpy.array([0, 0], dtype='int8'), ctx=sess.context)

  system_lib = sess._rpc.system_lib()
  print('got system lib', system_lib)
  timer_func = system_lib.time_evaluator('add', sess.context, number=10000, repeat=5)
  print('got timer_func', timer_func)
  time_res = timer_func(A_data, B_data, C_data)
  print('time result: ', time_res)
  print('time mean: ', time_res.mean)
  print(C_data.asnumpy())


if __name__ == '__main__':
  # _setup(test_compile_runtime)
  # _setup(test_dev_timer)
  _setup(test_time_evaluator)
