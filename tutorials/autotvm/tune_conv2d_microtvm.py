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
"""
Auto-tuning matrix multiplication on an ARM Cortex-M7 STM32F746 Board
=====================================================================
**Author**: `Logan Weber <https://github.com/weberlo>`_

Auto-tuning for a specific accelerator design is critical for getting the best
performance for any given operator. This is a tutorial showcases how to tune a
whole convolutional network on VTA.

The operator implementation for VTA in TVM is written in template form.
The template has many tunable knobs (tile factor, virtual threads, etc).
We will tune all convolution operators in the neural network. After tuning,
we produce a log file which stores the best schedule parameters for all tuned
operators. When the TVM compiler compiles these operators, it will query this
log file to get the best knob parameters.

"""
import logging
import os
import sys

from mxnet.gluon.model_zoo import vision
import numpy as np
from PIL import Image

import topi
import tvm
from tvm import rpc, autotvm, relay
from tvm.contrib import graph_runtime, util, download
#from tvm.autotvm.measure.measure_methods import request_remote
from tvm.autotvm.tuner import XGBTuner, GATuner, RandomTuner, GridSearchTuner
import tvm.micro as micro

# First, locate your OpenOCD script directory (e.g.,
# OPENOCD_SCRIPT_DIR=/usr/share/openocd/scripts) and run `openocd -f
# $(OPENOCD_SCRIPT_DIR)/interface/stlink-v2-1.cfg -f $(OPENOCD_SCRIPT_DIR)/target/stm32f7x.cfg` in one terminal.
#
# Then, run `python -m tvm.exec.rpc_tracker --host 0.0.0.0 --port=9190` in another terminal.
#
# Then, run `python -m tvm.micro.rpc_server --tracker=0.0.0.0:9190 --key=micro
# --dev-config=<config_file>` in another terminal.

logging.getLogger('autotvm').setLevel(logging.DEBUG)
logging.getLogger('autotvm').addHandler(logging.StreamHandler(sys.stdout))

DEV_CONFIG = micro.device.arm.stm32f746xx.default_config('127.0.0.1', 6666)
DEV_CREATE_MICRO_LIB = micro.device.get_device_funcs(DEV_CONFIG['device_id'])['create_micro_lib']
#DEV_CONFIG = micro.device.host.default_config()

DEVICE = 'arm-cortex-m'
TARGET = tvm.target.create('c -device=micro_dev')

N_TRIAL = 1500
EARLY_STOPPING = 800
N_PER_TRIAL = 1
N_PARALLEL = 8

SERVER_ADDR = '0.0.0.0'
SERVER_PORT = 9190

LOG_FILE_NAME = f'{DEVICE}.log'
if os.path.exists(LOG_FILE_NAME):
    os.remove(LOG_FILE_NAME)

#N, L, M = 32, 32, 32
#DTYPE = 'float32'
#
#@autotvm.template
#def matmul(N, L, M, dtype):
#    A = tvm.placeholder((N, L), name='A', dtype=dtype)
#    B = tvm.placeholder((L, M), name='B', dtype=dtype)
#
#    k = tvm.reduce_axis((0, L), name='k')
#    C = tvm.compute((N, M), lambda i, j: tvm.sum(A[i, k] * B[k, j], axis=k), name='C')
#    s = tvm.create_schedule(C.op)
#
#    # schedule
#    y, x = s[C].op.axis
#    k = s[C].op.reduce_axis[0]
#
#    # define schedule space
#    cfg = autotvm.get_config()
#    cfg.define_split('tile_y', y, num_outputs=2)
#    cfg.define_split('tile_x', x, num_outputs=2)
#
#    # schedule according to config
#    yo, yi = cfg['tile_y'].apply(s, C, y)
#    xo, xi = cfg['tile_x'].apply(s, C, x)
#
#    s[C].reorder(yo, xo, k, yi, xi)
#
#    return s, [A, B, C]

N, H, W, CO, CI, KH, KW = 1, 32, 32, 4, 4, 3, 3
STRIDES, PADDING, DILATION = (1, 1), (1, 1), 1
LAYOUT = 'NCHW'
OUT_DTYPE = 'float32'
TIMEOUT = 0
assert N == 1, "Only consider batch_size = 1 in this template"

@autotvm.template
def conv2d(N, H, W, CO, CI, KH, KW, strides, padding, dilation, layout, out_dtype):
    data = tvm.placeholder((N, CI, H, W), name='data')
    kernel = tvm.placeholder((CO, CI, KH, KW), name='kernel')

    cfg = autotvm.get_config()
    conv = topi.arm_cpu.conv2d.conv2d_arm_cpu(
            cfg, data, kernel, strides, padding, dilation, layout, out_dtype)
    sched = topi.arm_cpu.conv2d.schedule_conv2d_nchw_arm_cpu(cfg, [conv])
    return sched, [data, kernel, conv]


def tune():
    print('[TUNE]')
    task = autotvm.task.create(conv2d,
            args=(N, H, W, CO, CI, KH, KW, STRIDES, PADDING, DILATION, LAYOUT, OUT_DTYPE),
            target=TARGET)

    measure_option = autotvm.measure_option(
            builder=autotvm.LocalBuilder(
                build_func=tvm.micro.cross_compiler(DEV_CREATE_MICRO_LIB, micro.LibType.OPERATOR)),
            runner=autotvm.RPCRunner('micro', SERVER_ADDR, SERVER_PORT, n_parallel=N_PARALLEL, number=N_PER_TRIAL, timeout=TIMEOUT)
            )

    # create tmp log file
    tmp_log_file = LOG_FILE_NAME + '.tmp'
    if os.path.exists(tmp_log_file):
        os.remove(tmp_log_file)

    #tuner = RandomTuner(task)
    tuner = XGBTuner(task, loss_type='rank')

    # do tuning
    tuner.tune(n_trial=min(N_TRIAL, len(task.config_space)),
            early_stopping=EARLY_STOPPING,
            measure_option=measure_option,
            callbacks=[
                autotvm.callback.progress_bar(N_TRIAL, prefix='[Conv2D Task]'),
                autotvm.callback.log_to_file(tmp_log_file)])

    # pick best records to a cache file
    autotvm.record.pick_best(tmp_log_file, LOG_FILE_NAME)
    os.remove(tmp_log_file)


def evaluate():
    print('[EVALUATE]')

    def eval_mod(c_mod):
        with micro.Session(DEV_CONFIG) as sess:
            micro_mod = sess.create_micro_mod(c_mod)
            micro_func = micro_mod['conv2d']
            ctx = tvm.micro_dev(0)

            # check correctness
            data_np = np.random.uniform(size=(N, CI, H, W)).astype(np.float32)
            kernel_np = np.random.uniform(size=(CO, CI, KH, KW)).astype(np.float32)

            c_tvm = tvm.nd.empty([N, CO, H, W], ctx=ctx)
            res = micro_func(tvm.nd.array(data_np, ctx), tvm.nd.array(kernel_np, ctx), c_tvm)
            print(f'cycle count: {res}')
        return res
            #tvm.testing.assert_allclose(c_np, c_tvm.asnumpy(), rtol=1e-2)

    # compile with default schedule
    with TARGET:
        data = tvm.placeholder((N, CI, H, W), name='data')
        kernel = tvm.placeholder((CO, CI, KH, KW), name='kernel')
        conv = topi.nn.conv2d_nchw(data, kernel, STRIDES, PADDING, DILATION, OUT_DTYPE)
        #sched, arg_bufs = conv2d(N, H, W, CO, CI, KH, KW, STRIDES, PADDING, DILATION, LAYOUT, OUT_DTYPE)
        sched = tvm.create_schedule([conv.op])
        c_mod = tvm.build(sched, [data, kernel, conv], name='conv2d')
    default_cycle_count = eval_mod(c_mod)

    # compile kernels with history best records
    with autotvm.apply_history_best(LOG_FILE_NAME):
        with TARGET:
            sched, arg_bufs = conv2d(N, H, W, CO, CI, KH, KW, STRIDES, PADDING, DILATION, LAYOUT, OUT_DTYPE)
            c_mod = tvm.build(sched, arg_bufs, name='conv2d')
    autotune_cycle_count = eval_mod(c_mod)

    print(f'SPEEDUP: {default_cycle_count / autotune_cycle_count}')


if __name__ == '__main__':
    tune()
    input('finished tuning...')
    evaluate()
