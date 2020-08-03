import contextlib
import copy
import glob
import os
import sys
import subprocess
import textwrap

import numpy as np

import tvm
import tvm.relay
import tvm.micro

from tvm.micro.contrib import zephyr
from tvm.contrib import device_util
from tvm.contrib import util
import topi
from topi import get_const_tuple
from topi.testing import conv2d_nchw_python

BUILD = True
DEBUG = False

# TODO(weberlo) fix bug with sessions not being reusable
# ADD_SESS = None
# IDENT_SESS = None

def _make_sess_from_op(op_name, sched, arg_bufs):
  with tvm.transform.PassContext(opt_level=3, config={'tir.disable_vectorize': True}):
    mod = tvm.build(sched, arg_bufs, TARGET, target_host=TARGET, name=op_name)

  if TARGET.mcpu == 'x86-64':
    return _make_x86_64_sess(mod)
  elif TARGET.mcpu == 'cortex-m33':
    return _make_cortex_m33_sess(mod)
  else:
    assert False, 'device not supported by this test'


def _make_x86_64_sess(mod):
  workspace = tvm.micro.Workspace(debug=True)

  compiler = tvm.micro.DefaultCompiler(target=TARGET)
  opts = tvm.micro.DefaultOptions(os.path.join(tvm.micro.CRT_ROOT_DIR, 'host'))

  micro_binary = tvm.micro.build_static_runtime(
    # the x86 compiler *expects* you to give the exact same dictionary for both
    # lib_opts and bin_opts. so the library compiler is mutating lib_opts and
    # the binary compiler is expecting those mutations to be in bin_opts.
    # TODO(weberlo) fix this very bizarre behavior
    workspace, compiler, mod, lib_opts=opts['bin_opts'], bin_opts=opts['bin_opts'])

  flasher_kw = {
    'debug': DEBUG,
  }
  flasher = compiler.Flasher(**flasher_kw)
  return tvm.micro.Session(binary=micro_binary, flasher=flasher)


def _make_cortex_m33_sess(mod):
  workspace = tvm.micro.Workspace(debug=True)

  project_dir = os.path.expanduser('~/ws/utvm-zephyr-runtime')
  compiler = zephyr.ZephyrCompiler(
    project_dir=project_dir,
    board='nrf5340pdk_nrf5340_cpuapp',
    zephyr_toolchain_variant='gnuarmemb',
    env_vars={'GNUARMEMB_TOOLCHAIN_PATH': '~/ws/gcc-arm-none-eabi-9-2020-q2-update'},
  )

  opts = tvm.micro.DefaultOptions(f'{project_dir}/crt')

  if BUILD:
    micro_binary = tvm.micro.build_static_runtime(
      workspace, compiler, mod, **opts)

  # debug_rpc_session = tvm.rpc.connect('127.0.0.1', 9090)
  debug_rpc_session = None

  flasher = compiler.Flasher(
    openocd_serial='066DFF343039564157214347',
    debug_rpc_session=debug_rpc_session)
  if BUILD:
    return tvm.micro.Session(binary=micro_binary, flasher=flasher)
  else:
    return tvm.micro.Session(transport_context_manager=flasher.Transport(
      micro_binary=tvm.micro.MicroBinary(
        '/private/var/folders/9y/3j808g591ln3kys4qpyl3qmc0000gn/T/tvm-debug-mode-tempdirs/2020-06-30T09-34-07___s8a9ul7f/00000/build/runtime',
        'zephyr/zephyr.bin',
        debug_files=['zephyr/zephyr.elf'],
        labelled_files={'device_tree': ['zephyr/zephyr.dts'], 'cmake_cache': ['CMakeCache.txt']})))


def _make_add_sess():
  A = tvm.te.placeholder((2,), dtype='int8')
  B = tvm.te.placeholder((1,), dtype='int8')
  C = tvm.te.compute(A.shape, lambda i: A[i] + B[0], name='C')
  sched = tvm.te.create_schedule(C.op)
  return _make_sess_from_op('add', sched, [A, B, C])


def _make_ident_sess():
  A = tvm.te.placeholder((2,), dtype='int8')
  B = tvm.te.compute(A.shape, lambda i: A[i], name='B')
  sched = tvm.te.create_schedule(B.op)
  return _make_sess_from_op('ident', sched, [A, B])


def test_compile_runtime():
  """Test compiling the on-device runtime."""
  with _make_add_sess() as sess:
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
  number = 3
  repeat = 5
  min_repeat_ms = 0

  with _make_add_sess() as sess:
    A_data = tvm.nd.array(np.array([2, 3], dtype='int8'), ctx=sess.context)
    B_data = tvm.nd.array(np.array([4], dtype='int8'), ctx=sess.context)
    C_data = tvm.nd.array(np.array([0, 0], dtype='int8'), ctx=sess.context)

    system_lib = sess.get_system_lib()
    timer_func = system_lib.time_evaluator(
      'add', sess.context,
      number=number, repeat=repeat, min_repeat_ms=min_repeat_ms)
    time_res = timer_func(A_data, B_data, C_data)
    print(f'time_res: {time_res}')
    assert len(time_res.results) == repeat
    assert time_res.mean > 0.0
    # make sure the function actually ran
    assert (C_data.asnumpy() == np.array([6, 7])).all()


def test_time_eval_fp32_conv2d():
  number = 100
  repeat = 3
  min_repeat_ms = 10

  strides = 1
  padding = 1

  I = tvm.te.placeholder((1, 3, 5, 5), dtype='float32')
  F = tvm.te.placeholder((4, 3, 3, 3), dtype='float32')
  C = topi.nn.conv2d(I, F, strides=strides, padding=padding, dilation=1, layout='NCHW', out_dtype='float32')
  sched = tvm.te.create_schedule(C.op)
  op_name = 'conv2d'

  sess = _make_sess_from_op(op_name, sched, [I, F, C])
  with sess:
    I_np = np.random.rand(*(get_const_tuple(I.shape))).astype('float32')
    F_np = np.random.rand(*(get_const_tuple(F.shape))).astype('float32')
    I_data = tvm.nd.array(I_np, ctx=sess.context)
    F_data = tvm.nd.array(F_np, ctx=sess.context)
    C_data = tvm.nd.array(np.zeros(get_const_tuple(C.shape)).astype('float32'), ctx=sess.context)

    system_lib = sess.get_system_lib()
    timer_func = system_lib.time_evaluator(
      op_name, sess.context,
      number=number, repeat=repeat, min_repeat_ms=min_repeat_ms)
    time_res = timer_func(I_data, F_data, C_data)
    print(f'time_res: {time_res}')
    assert len(time_res.results) == repeat
    assert time_res.mean > 0.0
    # compare against python reference impl
    host_res = conv2d_nchw_python(I_np, F_np, strides, padding)
    C_np = C_data.asnumpy()
    print('utvm res!', C_np)
    print('host res!', host_res)
    np.testing.assert_allclose(C_np, host_res, rtol=1e-3, atol=1e-5)


def test_time_eval_many_runs():
  number = 1
  repeat = 5
  min_repeat_ms = 0

  with _make_add_sess() as sess:
    A_data = tvm.nd.array(np.array([2, 3], dtype='int8'), ctx=sess.context)
    B_data = tvm.nd.array(np.array([4], dtype='int8'), ctx=sess.context)
    C_data = tvm.nd.array(np.array([0, 0], dtype='int8'), ctx=sess.context)

    system_lib = sess.get_system_lib()
    # run the time evaluator many times to ensure we don't ruin device state
    # after each execution
    timer_func = system_lib.time_evaluator(
      'add', sess.context,
      number=number, repeat=repeat)
    for i in range(50):
      print(timer_func(A_data, B_data, C_data))


def test_type_check():
  """Test runtime type checking."""
  with _make_ident_sess() as sess:
    A_data = tvm.nd.array(np.array([2, 3], dtype='int8'), ctx=sess.context)
    # NOTE we have made an incorrect call to `np.ones`. we should have given
    # the shape `(2,)`. we would like to pick up this error on the device's
    # runtime.
    B_data = tvm.nd.array(np.ones([0, 0], dtype='int8'), ctx=sess.context)

    print('get system lib')
    system_lib = sess.get_system_lib()
    print('got system lib', system_lib)
    try:
      system_lib.get_function('ident')(A_data, B_data)
      assert False, 'no runtime type error was raised'
    except Exception as e:
      assert 'arg1.ndim is expected to equal 1' in str(e)


def test_many_tensor_alloc_deallocs():
  with _make_ident_sess() as sess:
    def make_and_del_tensor():
      # alloc
      A = tvm.nd.array(np.ones((8,), dtype='int8'), ctx=sess.context)
      # read
      A.asnumpy()
      # dealloc

    for i in range(20):
      make_and_del_tensor()


def test_autotvm():
  target = tvm.target.create('c -mcpu=x86-64')
  model = tvm.relay.fromtext("""\
      v0.0.4
      def @main(%data: Tensor[(1, 3, 32, 32), int8], %weight: Tensor[(3, 3, 5, 5), int8]) {
          nn.conv2d(
               %data,
               %weight,
               padding=[2, 2],
               channels=3,
               kernel_size=[5, 5],
               data_layout="NCHW",
               kernel_layout="OIHW",
               out_dtype="int32")

      }
      """)
  with tvm.transform.PassContext(opt_level=3, config={'tir.disable_vectorize': True}):
    tasks = tvm.autotvm.task.extract_from_program(model, {}, target)

  assert len(tasks) == 1
  tuner = tvm.autotvm.tuner.GATuner(tasks[0])

  workspace = tvm.micro.Workspace(debug=True)
  compiler = tvm.micro.DefaultCompiler(target=target)
  lib_opts = tvm.micro.DefaultOptions()
  lib_opts['include_dirs'].append(os.path.join(tvm.micro.TVM_ROOT_DIR, 'src', 'runtime', 'crt', 'host'))
  lib_opts['include_dirs'].append(os.path.join(tvm.micro.TVM_ROOT_DIR, 'src', 'runtime', 'crt', 'include'))

  adapter = tvm.micro.AutoTvmAdapter(workspace, compiler, compiler.flasher_factory,
                                     lib_opts=lib_opts)

  builder = tvm.autotvm.LocalBuilder(
      build_func=adapter.StaticRuntime,
      n_parallel=1,
      build_kwargs={'build_option': {'tir.disable_vectorize': True}},
      do_fork=False)
  runner = tvm.autotvm.LocalRunner(
      number=1, repeat=1, timeout=0, code_loader=adapter.CodeLoader)

  measure_option = tvm.autotvm.measure_option(builder=builder, runner=runner)
  tuner.tune(n_trial=5,
             measure_option=measure_option,
             callbacks=[],
             si_prefix='k')




if __name__ == '__main__':
  # TODO(weberlo) remove before mainlining
  if len(sys.argv) == 2:
    TARGET = tvm.target.target.micro(sys.argv[1])
  else:
    print('missing micro target specifier. assuming x86-64')
    TARGET = tvm.target.target.micro('x86-64')
  assert 'micro-runtime' in TARGET.keys
  print(f'using target: {TARGET}')

  test_compile_runtime()
  test_time_eval_int_add()
  test_time_eval_fp32_conv2d()
  test_time_eval_many_runs()
  test_type_check()
  test_many_tensor_alloc_deallocs()
  import logging
  logging.basicConfig(level=logging.DEBUG)
  test_autotvm()
