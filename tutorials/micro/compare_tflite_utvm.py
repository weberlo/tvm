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

model_url = 'https://people.linaro.org/~tom.gall/sine_model.tflite'
model_file = 'sine_model.tflite'
model_path = download_testdata(model_url, model_file, module='data')

# Load the TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

NP_TO_TVM_DTYPE = {
    np.float32: 'float32',
    np.float16: 'float16',
    np.int32: 'int32',
    np.int16: 'int16',
    np.int8: 'int8',
}

TVM_TO_TFLITE_DTYPE = {
    'float32': 'kTfLiteFloat32',
    'int32': 'kTfLiteInt32',
    'int16': 'kTfLiteInt16',
    'int8': 'kTfLiteInt8',
}

input_shape = input_details[0]['shape']
input_dtype = NP_TO_TVM_DTYPE[input_details[0]['dtype']]
output_shape = output_details[0]['shape']
output_dtype = NP_TO_TVM_DTYPE[output_details[0]['dtype']]
# Test the model on random input data.
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

def main():
    print(get_tflite_result())
    print(get_tfmicro_result())
    print(get_utvm_result())


def get_tflite_result():
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0][0]


def get_tfmicro_result():
    def run_cmd(*args):
        import subprocess
        (out, err) = subprocess.Popen(args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE).communicate()
        return out, err

    lite_model_path = model_path
    model_input_path = 'model_input.bytes'
    with open(model_input_path, 'wb') as f:
        f.write(input_data.tobytes())

    model_metadata = """
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/testing/micro_test.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>
    """

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
        lite_model_path,
        model_input_path,
        model_metadata
        )
    # out = out.decode('utf-8')
    res = np.frombuffer(out, dtype=output_dtype)
    # err = err.decode('utf-8')
    # print('Subprocess Stdout:')
    # print('  ' + out.replace('\n', '\n  '))
    # print('Subprocess Stderr:')
    # print('  ' + err.replace('\n', '\n  '))
    # return float(out)
    return res[0]


def get_utvm_result():
    tflite_model_buf = open(model_path, "rb").read()

    ######################################################################
    # Using the buffer, transform into a tflite model python object
    try:
        import tflite
        tflite_model = tflite.Model.GetRootAsModel(tflite_model_buf, 0)
    except AttributeError:
        import tflite.Model
        tflite_model = tflite.Model.Model.GetRootAsModel(tflite_model_buf, 0)

    ######################################################################
    # Parse the python model object to convert it into a relay module
    # and weights.
    # It is important to note that the input tensor name must match what
    # is contained in the model.
    #
    # If you are unsure what that might be, this can be discovered by using
    # the visualize.py script within the Tensorflow project.
    # See : How do I inspect a .tflite file? `<https://www.tensorflow.org/lite/guide/faq>`_

    input_tensor = input_details[0]['name']
    input_shape = tuple(input_details[0]['shape'])
    input_dtype = NP_TO_TVM_DTYPE[input_details[0]['dtype']]

    mod, params = relay.frontend.from_tflite(tflite_model,
                                            shape_dict={input_tensor: input_shape},
                                            dtype_dict={input_tensor: input_dtype})

    TARGET = 'c -device=micro_dev'
    dev_config = micro.device.host.generate_config()

    with micro.Session(dev_config) as sess:
        ctx = tvm.micro_dev(0)
        with tvm.transform.PassContext(
                opt_level=3,
                config={'tir.disable_vectorize': True},
                disabled_pass=['FuseOps']):
            graph, c_mod, params = relay.build(mod, target=TARGET, params=params)

        micro_mod = micro.create_micro_mod(c_mod, dev_config)
        mod = graph_runtime.create(graph, micro_mod, ctx)
        mod.set_input(**params)

        # The model consumes a single float32 value and returns a predicted
        # sine value.
        # To pass the input value we construct a tvm.nd.array object
        # with a single contrived number as input. For this model values of
        # 0 to 2Pi are acceptable.
        mod.set_input(input_tensor, tvm.nd.array(input_data))
        mod.run()
        tvm_output = mod.get_output(0).asnumpy()
        return tvm_output[0][0]


if __name__ == '__main__':
    main()
