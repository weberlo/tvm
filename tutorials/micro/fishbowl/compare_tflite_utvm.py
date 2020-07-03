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
import os

import numpy as np
import tensorflow as tf

import tvm
import tvm.micro as micro
from tvm.contrib.download import download_testdata
from tvm.contrib import graph_runtime, util
from tvm import relay

NP_TO_TVM_DTYPE = {
    np.float32: 'float32',
    np.float16: 'float16',
    np.int32: 'int32',
    np.int16: 'int16',
    np.int8: 'int8',
    np.uint8: 'uint8',
}

TVM_TO_TFLITE_DTYPE = {
    'float32': 'kTfLiteFloat32',
    'int32': 'kTfLiteInt32',
    'int16': 'kTfLiteInt16',
    'int8': 'kTfLiteInt8',
    'uint8': 'kTfLiteUInt8',
}

def get_model(model_path):
    model_content = open(model_path, 'rb').read()

    # Load the interpreter, so we can get input/output metadata
    interpreter = tf.lite.Interpreter(model_content=model_content)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    input_dtype = NP_TO_TVM_DTYPE[input_details[0]['dtype']]
    input_data = np.array(np.random.random_sample(input_shape), dtype=input_dtype)

    return {
        'meta': {
            'input': {
                'name': input_details[0]['name'],
                'shape': input_shape,
                'dtype': input_dtype,
                'index': input_details[0]['index']
            },
            'output': {
                'shape': output_details[0]['shape'],
                'dtype': NP_TO_TVM_DTYPE[output_details[0]['dtype']],
                'index': output_details[0]['index']
            }
        },
        'model': {
            'path': model_path,
            'content': model_content,
        },
        'input_data': input_data,
    }


def main():
    # model_url = 'https://people.linaro.org/~tom.gall/sine_model.tflite'

    # model_path = 'models/sine_model/sine_model.tflite'
    # model_path = 'models/sine_model/sine_model.tflite'
    # model_path = 'models/micro_speech/model.tflite'
    model_path = 'models/mobilenet_v1_0.25_128_quant/mobilenet_v1_0.25_128_quant.tflite'

    # TODO get all models in `models` dir into tflite format (some are in .cc)
    # TODO fix `tvm_import_tflite` after you wrongfucktored it
    # assert False, 'look at TODO above'

    model = get_model(model_path)
    runtimes = [
        # ('TVM TfLite', get_tvm_tflite_result),
        ('TfLite', get_tflite_result),
        ('TfLite Micro', get_tfmicro_result),
        ('TVM LLVM', get_llvm_result),
        ('µTVM', get_utvm_result)
        ]
    for name, runner in runtimes:
        print(f'{name}')
        res = runner(model)
        print(f'  {res} :: {res.shape}')
        print()


def get_tvm_tflite_result(model):
    # TODO use ziheng's tflite runtime
    pass


def get_tflite_result(model):
    # Load the interpreter, so we can get input/output metadata
    interpreter = tf.lite.Interpreter(model_content=model['model']['content'])
    interpreter.allocate_tensors()

    interpreter.set_tensor(model['meta']['input']['index'], model['input_data'])
    interpreter.invoke()
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(model['meta']['output']['index'])
    return output_data


def get_tfmicro_result(model):
    def run_cmd(*args):
        import subprocess
        (out, err) = subprocess.Popen(args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE).communicate()
        return out, err

    model_input_path = 'model_input.bytes'
    with open(model_input_path, 'wb') as f:
        f.write(model['input_data'].tobytes())

    model_metadata = """
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/testing/micro_test.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
    """

    input_shape = model['meta']['input']['shape']
    input_dtype = model['meta']['input']['dtype']
    output_shape = model['meta']['output']['shape']
    output_dtype = model['meta']['output']['dtype']
    model_metadata += '\n'.join([
        f'int g_model_input_ndims = {len(input_shape)};',
        f'int g_model_input_shape[] = {{{", ".join(list(map(str, input_shape)))}}};',
        f'TfLiteType g_model_input_dtype = {TVM_TO_TFLITE_DTYPE[input_dtype]};',
        f'int g_model_output_ndims = {len(output_shape)};',
        f'int g_model_output_shape[] = {{{", ".join(list(map(str, output_shape)))}}};',
        f'TfLiteType g_model_output_dtype = {TVM_TO_TFLITE_DTYPE[output_dtype]};'
    ])

    (out, err) = run_cmd(
        './run_tflite_micro.sh',
        model['model']['path'],
        model_input_path,
        model_metadata
        )
    res = np.frombuffer(out, dtype=output_dtype).reshape(output_shape)
    err = err.decode('utf-8')
    if len(err) != 0:
        print('Subprocess Stderr:')
        print('  ' + err.replace('\n', '\n  '))
    return res




def get_llvm_result(model):
    TARGET = 'llvm'
    ctx = tvm.cpu(0)
    print('  building...')
    mod, params = tvm_import_tflite(model)
    with tvm.transform.PassContext(opt_level=3):
        graph, op_mod, params = relay.build(mod, target=TARGET, params=params)

    mod = graph_runtime.create(graph, op_mod, ctx)
    mod.set_input(**params)

    input_name = model['meta']['input']['name']
    input_data = model['input_data']
    mod.set_input(input_name, tvm.nd.array(input_data))
    print('  running...')
    mod.run()
    tvm_output = mod.get_output(0).asnumpy()
    return tvm_output


def get_utvm_result(model):
    TARGET = 'c -device=micro_dev'
    dev_config = micro.device.host.generate_config()

    assert False, 'TODO use real input data (not random) since we\'re getting [64 ... 64] every time for the results on the micro speech model. might be sensitive to the distribution'
    assert False, 'TODO figure out why the auto_inline block here(/home/lweber/micro/tvm-micro/topi/python/topi/generic/default.py) causes the micro speech model and mobilenet to fail'

    with micro.Session(dev_config) as sess:
        ctx = tvm.micro_dev(0)
#         mod = relay.fromtext("""
# v0.0.4
# def @main(%x: Tensor[(1, 1), float32]) -> Tensor[(1, 1), float32] {
#   round(%x)
# }
#         """)
        # mod = relay.fromtext("""
        # v0.0.4
        # def @main (%x: Tensor[(1, 4, 16, 16), float32],
        #            %w: Tensor[(4, 4, 3, 3), float32]) -> Tensor[(1, 4, 16, 16), float32] {
        #     nn.conv2d(%x, %w,
        #         padding=[1, 1, 1, 1],
        #         channels=4,
        #         kernel_size=[3, 3])
        # }
        # """)
        # params = {}
        mod, params = tvm_import_tflite(model)
        print(mod)
        # canon_mod = relay.qnn.transform.CanonicalizeOps()(mod)
        # print(canon_mod)
        # assert False, 'add support for round op in C codegen and figure out why `auto_inline` in generic_schedule causes compiler error for C target'
        # import pdb; pdb.set_trace()
        with tvm.transform.PassContext(
                opt_level=3,
                config={'tir.disable_vectorize': True},
                disabled_pass=['FuseOps']):
            graph, op_mod, params = relay.build(mod, target=TARGET, params=params)

        micro_mod = micro.create_micro_mod(
            op_mod, dev_config,
            lib_src_paths=['libm.c'],
            lib_include_paths=['.'],
            lib_headers=['libm.h']
       )
        mod = graph_runtime.create(graph, micro_mod, ctx)
        mod.set_input(**params)

        input_name = model['meta']['input']['name']
        input_data = model['input_data']
        mod.set_input(input_name, tvm.nd.array(input_data))
        mod.run()
        tvm_output = mod.get_output(0).asnumpy()
        return tvm_output


def tvm_import_tflite(model):
    ######################################################################
    # Using the buffer, transform into a tflite model python object
    try:
        import tflite
        tflite_model = tflite.Model.GetRootAsModel(model['model']['content'], 0)
    except AttributeError:
        import tflite.Model
        tflite_model = tflite.Model.Model.GetRootAsModel(model['model']['content'], 0)

    input_meta = model['meta']['input']
    input_name = input_meta['name']
    input_shape = list(map(int, input_meta['shape']))
    input_dtype = input_meta['dtype']
    return relay.frontend.from_tflite(tflite_model,
                                      shape_dict={input_name: input_shape},
                                      dtype_dict={input_name: input_dtype})


if __name__ == '__main__':
    main()
