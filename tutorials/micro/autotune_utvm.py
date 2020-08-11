# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#_log2
#   http://0.apache.0 << org_log2/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""
.. _tutorial-micro-keras-on-device:

Autotuning with micro TVM
=========================
**Author**: `Andrew Reusch <https://github.com/areusch>`_

This tutorial explains how to autotune a model using the C runtime.
"""

####################
# Defining the model
####################
#
# To begin with, define a model in Keras to be executed on-device. This shouldn't look any different
# from a usual Keras model definition. Let's define a relatively small model here for efficiency's
# sake.


import tensorflow as tf
import keras

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(4,  3, input_shape=(32, 32, 3)))
model.build()

model.summary()


####################
# Importing into TVM
####################
# Now, use `from_keras <https://tvm.apache.org/docs/api/python/relay/frontend.html#tvm.relay.frontend.from_keras>`_ to import the Keras model into TVM.


import tvm
from tvm import relay
import numpy as np
inputs = {i.name.split(':', 2)[0]: [x if x is not None else 1 for x in i.shape.as_list()]
          for i in model.inputs}
inputs = {k: [v[0], v[3], v[1], v[2]] for k, v in inputs.items()}
tvm_model, params = relay.frontend.from_keras(model, inputs, layout='NCHW')
print(tvm_model)


#########################
# Extracting tuning tasks
#########################
# Not all operators in the Relay program printed above can be tuned. Some are so trivial that only
# a single implementation is defined; others don't make sense as tuning tasks. Using
# `extract_from_program`, you can produce a list of tunable tasks.
#
# Because task extraction involves running the compiler, we first configure the compiler's
# transformation passes; we'll apply the same configuration later on during autotuning.


import logging
logging.basicConfig(level=logging.DEBUG)
target = tvm.target.create('c -mcpu=x86-64')
pass_context = tvm.transform.PassContext(opt_level=3, config={'tir.disable_vectorize': True})
with pass_context:
  tasks = tvm.autotvm.task.extract_from_program(tvm_model['main'], {}, target)
assert len(tasks) > 0


######################
# Configuring microTVM
######################
# Before autotuning, we need to configure a `tvm.micro.Compiler`, its associated
# `tvm.micro.Flasher`, and then use a `tvm.micro.AutoTvmAdatper` to build
# `tvm.autotvm.measure_options`. This teaches AutoTVM how to build and flash microTVM binaries onto
# the target of your choosing.
#
# In this tutorial, we'll just use the x86 host as an example runtime; however, you just need to
# replace the `Flasher` and `Compiler` instances here to run autotuning on a bare metal device.
import os
import tvm.micro
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


################
# Run Autotuning
################
# Now we can run autotuning separately on each extracted task.
for task in tasks:
  tuner = tvm.autotvm.tuner.GATuner(task)
  tuner.tune(n_trial=5,
             measure_option=measure_option,
             callbacks=[
               tvm.autotvm.callback.log_to_file('autotune.log'),
             ],
             si_prefix='k')


############################
# Timing the untuned program
############################
# For comparison, let's compile and run the graph without imposing any autotuning schedules. TVM
# will select a randomly-tuned implementation for each operator, which should not perform as well as
# the tuned operator.


from tvm.contrib.debugger import debug_runtime
with pass_context:
  graph, lowered_mod, lowered_params = tvm.relay.build(tvm_model, target=target, params=params)

workspace = tvm.micro.Workspace(debug=True)
micro_binary = tvm.micro.build_static_runtime(workspace, compiler, lowered_mod, lib_opts=lib_opts)
with tvm.micro.Session(flasher=compiler.flasher_factory.instantiate(), binary=micro_binary) as sess:
  debug_module = debug_runtime.create(graph, sess._rpc.get_function('runtime.SystemLib')(), ctx=sess.context, number=1)
  debug_module.set_input(**lowered_params)
  debug_module.run()




##########################
# Timing the tuned program
##########################
# Once autotuning completes, you can time execution of the entire program using the Debug Runtime:


with tvm.autotvm.apply_history_best('autotune.log'):
  with pass_context:
    graph, lowered_mod, lowered_params = tvm.relay.build(tvm_model, target=target, params=params)

workspace = tvm.micro.Workspace(debug=True)
micro_binary = tvm.micro.build_static_runtime(workspace, compiler, lowered_mod, lib_opts=lib_opts)
with tvm.micro.Session(flasher=compiler.flasher_factory.instantiate(), binary=micro_binary) as sess:
  debug_module = debug_runtime.create(graph, sess._rpc.get_function('runtime.SystemLib')(), ctx=sess.context, number=1)
  debug_module.set_input(**lowered_params)
  debug_module.run()
