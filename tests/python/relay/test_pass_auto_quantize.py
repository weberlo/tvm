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
import numpy as np
import pytest

import tvm
from tvm import te
from tvm import relay
from tvm.relay import testing
from tvm.relay.expr import Call
from topi import get_const_tuple


def quantize_and_build(out):
    f = relay.Function(relay.analysis.free_vars(out), out)
    mod, params = testing.create_workload(f)

    with relay.quantize.qconfig(skip_conv_layers=[]):
        qmod = relay.quantize.quantize(mod, params)

    relay.build(qmod, "llvm", params=params)

    return qmod

def test_mul_rewrite():
    """a test case where rhs of mul is not constant"""
    data = relay.var("data", shape=(1, 16, 64, 64))
    multiplier = relay.sigmoid(relay.var("data", shape=(1, 16, 1, 1)))
    conv = relay.nn.conv2d(data, relay.var("weight"),
                           kernel_size=(3, 3),
                           padding=(1, 1),
                           channels=16)
    act = relay.nn.relu(data=conv)

    quantize_and_build(act * multiplier)

    pool = relay.nn.global_avg_pool2d(data=act)

    quantize_and_build(act * pool)

def test_batch_flatten_rewrite():

    data = relay.var("data", shape=(1, 16, 64, 64), dtype="float32")

    out = relay.nn.conv2d(data, relay.var("weight"),
                          kernel_size=(3, 3),
                          padding=(1, 1),
                          channels=16)

    out = relay.nn.batch_flatten(out)

    qmod = quantize_and_build(out)

    def _check_batch_flatten(node):
        if isinstance(node, Call):
            if(node.op.name == "nn.batch_flatten"):
               assert node.checked_type.dtype == "int8"

    # check if batch_flatten is quantized
    relay.analysis.post_order_visit(qmod["main"], _check_batch_flatten)

def get_calibration_dataset(input_name):
    dataset = []
    for i in range(5):
        data = np.random.uniform(size=(1, 3, 224, 224))
        dataset.append({input_name: data})
    return dataset


@pytest.mark.parametrize("create_target", [True, False])
def test_calibrate_target(create_target):
    mod, params = testing.resnet.get_workload(num_layers=18)
    dataset = get_calibration_dataset("data")
    with relay.quantize.qconfig(calibrate_mode="kl_divergence"):
        if create_target:
            with tvm.target.create("llvm"):
                relay.quantize.quantize(mod, params, dataset)
        else:
            # current_target = None
            relay.quantize.quantize(mod, params, dataset)


def test_calibrate_memory_bound():
    mod, params = testing.resnet.get_workload(num_layers=18)
    dataset = get_calibration_dataset("data")
    import multiprocessing
    num_cpu = multiprocessing.cpu_count()
    with relay.quantize.qconfig(calibrate_mode="kl_divergence",
                                calibrate_chunk_by=num_cpu):
        relay.quantize.quantize(mod, params, dataset)



#############################################
# ALLOWED_DTYPES AND PARTITION_RESULT TESTS #
#############################################

def gen_rand_np(tt, low, high):
    if 'int' in tt.dtype:
        return np.random.randint(low, high, size=get_const_tuple(tt.shape), dtype=tt.dtype)
    elif 'float' in tt.dtype:
        return np.random.uniform(low, high, size=get_const_tuple(tt.shape)).astype(tt.dtype)
    else:
        assert False, 'unknown dtype'


def gen_rand_tvm(tt, low, high):
    return tvm.nd.array(gen_rand_np(tt, low, high), ctx=tvm.cpu(0))


def verify(mod, params):
    # TODO is the `with qconfig` shit even a good idea? why can't it just be a
    # config dict passed to `quantize`?
    base_cfg = {
      'calibrate_mode': 'global_scale',
      'global_scale': 8.0,
      'nbit_activation': 8,
      'dtype_activation': "int8",
      'skip_conv_layers': [],
      'skip_dense_layers': False,
      'dtype_input': "int8",
      'dtype_weight': "int8",
    }
    with relay.quantize.qconfig(**base_cfg):
        full_mod = relay.quantize.quantize(mod, params)
    with relay.quantize.qconfig(
            **base_cfg,
            allowed_dtypes=['int8'],
            partition_result=True):
        pre_mod, mid_mod, post_mod = relay.quantize.quantize(mod, params)

    pre_func = pre_mod['main']
    mid_func = mid_mod['main']
    post_func = post_mod['main']
    fused_mod = tvm.IRModule(functions={
        relay.GlobalVar('quantize_inputs'): pre_func,
        relay.GlobalVar('quantized_main'): mid_func,
        relay.GlobalVar('dequantize_outputs'): post_func,
    })

    sb = relay.ScopeBuilder()
    fused_mod_main_params = [relay.Var(param.name_hint) for param in pre_func.params]
    quantized_inputs = sb.let('quantized_inputs', relay.Call(
        fused_mod.get_global_var('quantize_inputs'),
        fused_mod_main_params
    ))
    quantized_outputs = sb.let('quantized_outputs', relay.Call(
        fused_mod.get_global_var('quantized_main'),
        [relay.TupleGetItem(quantized_inputs, i) for i in range(len(pre_func.ret_type.fields))]
    ))
    dequantized_outputs = sb.let('dequantized_outputs', relay.Call(
        fused_mod.get_global_var('dequantize_outputs'),
        # TODO quit assuming we'll only have a single output?
        [quantized_outputs]
    ))
    sb.ret(dequantized_outputs)
    fused_mod['main'] = relay.Function(fused_mod_main_params, sb.get())

    params = [
        gen_rand_tvm(param.type_annotation, 0, 1)
        for param in fused_mod['main'].params
    ]

    vm = relay.create_executor('vm', ctx=tvm.cpu(0), target='llvm', mod=fused_mod)
    fused_mod_result = vm.evaluate()(*params)

    vm = relay.create_executor('vm', ctx=tvm.cpu(0), target='llvm', mod=full_mod)
    full_mod_result = vm.evaluate()(*params)

    tvm.testing.assert_allclose(full_mod_result.asnumpy(), fused_mod_result.asnumpy())


# TODO remove this test before upstreaming
@pytest.mark.xfail(raises=RuntimeError)
def test_cifar10():
    import onnx
    import os
    onnx_model = onnx.load(os.path.dirname(os.path.abspath(__file__)) + '/cifar10.onnx')
    mod, params = relay.frontend.from_onnx(onnx_model, {"data": (1, 3, 32, 32)})
    verify(mod, params)


def get_param_type(func, name):
    for param in func.params:
        if param.name_hint == name:
            return param.checked_type


def test_conv_quant():
    func = relay.fromtext("""
    v0.0.4
    fn (%x: Tensor[(1, 4, 16, 16), float32],
        %w: Tensor[(4, 4, 3, 3), float32]) -> Tensor[(1, 4, 16, 16), float32] {
      nn.conv2d(%x, %w,
        padding=[1, 1, 1, 1],
        channels=4,
        kernel_size=[3, 3])
    }
    """)
    mod = tvm.IRModule.from_expr(func)
    weight_ty = mod['main'].params[1].checked_type
    params = {
        'w': gen_rand_tvm(weight_ty, 0, 1)
    }
    verify(mod, params)


@pytest.mark.xfail(raises=RuntimeError)
def test_start_with_unquantizable():
    # start with avgpool, since it isn't currently supported
    func = relay.fromtext("""
    v0.0.4
    fn (%x: Tensor[(1, 4, 16, 16), float32],
        %w: Tensor[(4, 4, 3, 3), float32]) -> Tensor[(1, 4, 8, 8), float32] {
      %0 = nn.avg_pool2d(%x, pool_size=[3, 3], strides=[2, 2], padding=[0, 0, 0, 0], ceil_mode=True);
      nn.conv2d(%0, %w,
        padding=[1, 1, 1, 1],
        channels=4,
        kernel_size=[3, 3])
    }
    """)
    mod = tvm.IRModule.from_expr(func)
    weight_ty = mod['main'].params[1].checked_type
    params = {
        'w': gen_rand_tvm(weight_ty, 0, 1)
    }
    verify(mod, params)


# TODO with respect to partitioning, what should we do in cases where the
# function is so small the quantizer doesn't quantize anything?
# TODO it *should* be able to quantize a single `add`. might need to patch
# it so it can
def test_add_quant():
    func = relay.fromtext("""
    v0.0.4
    fn (%x: Tensor[(10, 10), float32],
        %y: Tensor[(10, 10), float32]) {
      add(%x, %y)
    }
    """)
    mod = tvm.IRModule.from_expr(func)
    params = {}
    verify(mod, params)


def test_multiple_arg_conversions():
    mod = tvm.IRModule.from_expr(relay.fromtext("""
    v0.0.4
    fn (%x1: Tensor[(1, 4, 16, 16), float32],
        %w1: Tensor[(4, 4, 3, 3), float32],
        %x2: Tensor[(1, 4, 16, 16), float32],
        %w2: Tensor[(4, 4, 3, 3), float32]
        ) -> Tensor[(1, 4, 16, 16), float32] {
      %0 = nn.conv2d(%x1, %w1,
        padding=[1, 1, 1, 1],
        channels=4,
        kernel_size=[3, 3]);
      %1 = nn.conv2d(%x2, %w2,
        padding=[1, 1, 1, 1],
        channels=4,
        kernel_size=[3, 3]);
      add(%0, %1)
    }
    """))

    w1_ty = get_param_type(mod['main'], 'w1')
    w2_ty = get_param_type(mod['main'], 'w2')
    params = {
        'w1': gen_rand_tvm(w1_ty, 0, 1),
        'w2': gen_rand_tvm(w2_ty, 0, 1),
    }
    verify(mod, params)


if __name__ == "__main__":
    test_mul_rewrite()
    test_batch_flatten_rewrite()
    test_calibrate_target(False)
    test_calibrate_target(True)
    test_calibrate_memory_bound()

    test_cifar10()
    test_conv_quant()
    test_start_with_unquantizable()
    test_add_quant()
    test_multiple_arg_conversions()

    # TODO add test that includes float32 in allowed_dtypes, and ensure every
    # test doesn't throw an exception
    assert False, 'TODO add tests that play with config knobs more'
