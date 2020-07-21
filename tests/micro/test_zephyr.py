import contextlib
import copy
import glob
import os
import shutil

import numpy as np

import tvm
import tvm.micro

from tvm.micro.contrib import zephyr
from tvm.contrib import device_util
from tvm.contrib import util
import topi
from topi import get_const_tuple
from topi.testing import conv2d_nchw_python

BUILD = True
DEBUG = False

# TARGET = tvm.target.target.micro('stm32f746xx')
TARGET = tvm.target.target.micro('nrf5340')

def _make_sess_from_op(sched, arg_bufs, op_name):
  with tvm.transform.PassContext(opt_level=3, config={'tir.disable_vectorize': True}):
    # TODO(weberlo) very fundamental question, but why do we need to pass in
    # the arg_bufs, when it should be derivable from the schedule?
    mod = tvm.build(sched, arg_bufs, TARGET, target_host=TARGET, name=op_name)

  workspace = tvm.micro.Workspace(debug=True)

#  rpc_session = tvm.rpc.connect('127.0.0.1', 9090)
  project_dir = os.path.expanduser('~/ws/utvm-zephyr-runtime')
  compiler = zephyr.ZephyrCompiler(
    project_dir=project_dir,
#    board='nucleo_f746zg',
    board='nrf5340pdk_nrf5340_cpuapp',
    zephyr_toolchain_variant='gnuarmemb',
    env_vars={'GNUARMEMB_TOOLCHAIN_PATH': '~/ws/gcc-arm-none-eabi-9-2020-q2-update'},
  )

  root_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..'))
  bin_opts = tvm.micro.DefaultOptions()
  bin_opts.setdefault('profile', {})['common'] = ['-Wno-unused-variable']
#  bin_opts.setdefault('ccflags', []).append('-std=gnu++14')
  bin_opts.setdefault('ldflags', []).append('-std=gnu++14')
  bin_opts.setdefault('include_dirs', []).append(f'{project_dir}/crt')
  bin_opts.setdefault('include_dirs', []).append(os.path.realpath(os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'runtime', 'crt', 'include')))

  lib_opts = copy.deepcopy(bin_opts)
  lib_opts['profile']['common'].append('-Werror')
  lib_opts['cflags'] = ['-Wno-error=incompatible-pointer-types']

  if BUILD:
    micro_binary = tvm.micro.build_static_runtime(workspace, compiler, mod, lib_opts, bin_opts)
  lib_opts['cflags'].pop()

#   device_transport = device_util.DeviceTransportLauncher({
#     'num_instances': 1,
#     'use_tracker': False,
#   })

  # with contextlib.ExitStack() as exit_stack:
  flasher_kw = {
      # 'nrfjprog_snr': '960104913',
      # 'nrfjprog_snr': '960148252',
      # 'nrfjprog_snr': '960165313',
#      'debug': DEBUG,
#      'debug_remote_hostport': '{}:{:d}'.format(*device_transport.openocd_gdb_host_port_tuple(0)),
  }

  debug_rpc_session = tvm.rpc.connect('127.0.0.1', 9090)

#     if DEBUG:
#       flasher_kw['debug_wrapping_context_manager'] = device_transport.launch(None)
  # flasher = compiler.Flasher(**flasher_kw) #, openocd_serial='066DFF343039564157214347')

  flasher = compiler.Flasher(
    **flasher_kw,
    openocd_serial='066DFF343039564157214347',
    debug_rpc_session=debug_rpc_session)
  if BUILD:
    # sess = exit_stack.enter_context(tvm.micro.Session(binary=micro_binary, flasher=flasher))
    return tvm.micro.Session(binary=micro_binary, flasher=flasher)
  else:
    # sess = exit_stack.enter_context(tvm.micro.Session(transport_context_manager=flasher.Transport(
    #   micro_binary=tvm.micro.MicroBinary('/private/var/folders/9y/3j808g591ln3kys4qpyl3qmc0000gn/T/tvm-debug-mode-tempdirs/2020-06-30T09-34-07___s8a9ul7f/00000/build/runtime', 'zephyr/zephyr.bin', debug_files=['zephyr/zephyr.elf'], labelled_files={'device_tree': ['zephyr/zephyr.dts'], 'cmake_cache': ['CMakeCache.txt']}))))
    return tvm.micro.Session(transport_context_manager=flasher.Transport(
      micro_binary=tvm.micro.MicroBinary('/private/var/folders/9y/3j808g591ln3kys4qpyl3qmc0000gn/T/tvm-debug-mode-tempdirs/2020-06-30T09-34-07___s8a9ul7f/00000/build/runtime', 'zephyr/zephyr.bin', debug_files=['zephyr/zephyr.elf'], labelled_files={'device_tree': ['zephyr/zephyr.dts'], 'cmake_cache': ['CMakeCache.txt']})))


def test_compile_runtime():
  """Test compiling the on-device runtime."""
  A = tvm.te.placeholder((2,), dtype='int8')
  B = tvm.te.placeholder((1,), dtype='int8')
  C = tvm.te.compute(A.shape, lambda i: A[i] + B[0], name='C')
  sched = tvm.te.create_schedule(C.op)

  sess = _make_sess_from_op(sched, [A, B, C], 'add')
  with sess:
    A_data = tvm.nd.array(np.array([2, 3], dtype='int8'), ctx=sess.context)
    assert (A_data.asnumpy() == np.array([2, 3])).all()
    B_data = tvm.nd.array(np.array([4], dtype='int8'), ctx=sess.context)
    assert (B_data.asnumpy() == np.array([4])).all()
    C_data = tvm.nd.array(np.array([0, 0], dtype='int8'), ctx=sess.context)
    assert (C_data.asnumpy() == np.array([0, 0])).all()

    print('get system lib')
    system_lib = sess.get_system_lib()
    print('got system lib', system_lib)
    system_lib.get_function('add')(A_data, B_data, C_data)
    print('got data!', C_data.asnumpy())
    assert (C_data.asnumpy() == np.array([6, 7])).all()


def test_time_eval_int_add():
  number = 10
  repeat = 5
  min_repeat_ms = 10

  A = tvm.te.placeholder((2,), dtype='int8')
  B = tvm.te.placeholder((1,), dtype='int8')
  C = tvm.te.compute(A.shape, lambda i: A[i] + B[0], name='C')
  sched = tvm.te.create_schedule(C.op)

  sess = _make_sess_from_op(sched, [A, B, C], 'add')
  with sess:
    A_data = tvm.nd.array(np.array([2, 3], dtype='int8'), ctx=sess.context)
    B_data = tvm.nd.array(np.array([4], dtype='int8'), ctx=sess.context)
    C_data = tvm.nd.array(np.zeros([0, 0], dtype='int8'), ctx=sess.context)

    print('[Getting System Lib]')
    system_lib = sess.get_system_lib()
    print('[Getting Time Evaluator]')
    timer_func = system_lib.time_evaluator(
      'add', sess.context,
      number=number, repeat=repeat, min_repeat_ms=min_repeat_ms)
    print('[Running Time Evaluator]')
    time_res = timer_func(A_data, B_data, C_data)
    print('[Finished Time Evaluator]')
    import pdb; pdb.set_trace()
    assert len(time_res.results) == repeat
    assert time_res.mean > 0.0
    # make sure the function actually ran
    assert (C_data.asnumpy() == np.array([6, 7])).all()


def test_time_eval_fp32_conv2d():
  number = 10000
  repeat = 3
  min_repeat_ms = 10

  strides = 1
  padding = 1

  I = tvm.te.placeholder((1, 3, 5, 5), dtype='float32')
  F = tvm.te.placeholder((4, 3, 3, 3), dtype='float32')
  C = topi.nn.conv2d(I, F, strides=strides, padding=padding, dilation=1, layout='NCHW', out_dtype='float32')
  sched = tvm.te.create_schedule(C.op)

  sess = _make_sess_from_op(sched, [I, F, C], 'conv2d')
  with sess:
    I_np = np.random.rand(*(get_const_tuple(I.shape))).astype('float32')
    F_np = np.random.rand(*(get_const_tuple(F.shape))).astype('float32')
    I_data = tvm.nd.array(I_np, ctx=sess.context)
    F_data = tvm.nd.array(F_np, ctx=sess.context)
    C_data = tvm.nd.array(np.zeros(get_const_tuple(C.shape)).astype('float32'), ctx=sess.context)

    print('[Getting System Lib]')
    system_lib = sess.get_system_lib()
    print('[Getting Time Evaluator]')
    timer_func = system_lib.time_evaluator(
      'conv2d', sess.context,
      number=number, repeat=repeat, min_repeat_ms=min_repeat_ms)
    print('[Running Time Evaluator]')
    time_res = timer_func(I_data, F_data, C_data)
    print('[Finished Time Evaluator]')
    import pdb; pdb.set_trace()
    assert len(time_res.results) == repeat
    assert time_res.mean > 0.0
    host_res = conv2d_nchw_python(I_np, F_np, strides, padding)
    assert np.testing.assert_allclose(C_data.asnumpy(), host_res, rtol=1e-3, atol=1e-5)


if __name__ == '__main__':
  # test_compile_runtime()
  test_time_eval_int_add()
  # test_time_eval_fp32_conv2d()
