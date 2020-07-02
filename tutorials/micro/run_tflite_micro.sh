#!/usr/bin/env bash

set -e

tflite_model_path="$1"
model_input_path="$2"
model_metadata="$3"

function as_c {
  in_path="$1"
  arr_name="$2"

  replace_text=$(echo "${in_path}" | sed "s/\//_/g" | sed "s/\./_/g")
  # Convert model to a C source file and set array var's name
  xxd -i "${in_path}" | sed "s/${replace_text}/${arr_name}/g"
}

as_c "${tflite_model_path}" g_model > model.cc
as_c "${model_input_path}" g_model_input >> model.cc
echo "${model_metadata}" >> model.cc

TF_PATH="$HOME/micro/tensorflow"

g++ -std=c++11 -DTF_LITE_STATIC_MEMORY -DNDEBUG -O3 \
  -DTF_LITE_DISABLE_X86_NEON \
  "-I${TF_PATH}" \
  "-I${TF_PATH}/tensorflow/lite/micro/tools/make/downloads/" \
  "-I${TF_PATH}/tensorflow/lite/micro/tools/make/downloads/gemmlowp" \
  "-I${TF_PATH}/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include" \
  "-I${TF_PATH}/tensorflow/lite/micro/tools/make/downloads/ruy" \
  "-I${TF_PATH}/tensorflow/lite/micro/tools/make/downloads/kissfft" \
  model.cc \
  driver.cc \
  "${TF_PATH}/tensorflow/lite/micro/tools/make/gen/linux_x86_64/lib/libtensorflow-microlite.a" \
  -lm \
  -o run_model

./run_model

rm run_model
