#!/usr/bin/env bash

set -e

tflite_model_path="$1"
model_replace_text="$2"
model_input_replace_text="$3"

# Convert model to a C source file
xxd -i "$tflite_model_path" > model.cc
# Update variable names
sed -i "s/${model_replace_text}/g_model/g" model.cc

# Convert model input to a C source file
xxd -i model_input.bytes > model_input.cc
# Update variable names
sed -i "s/${model_input_replace_text}/g_model_input/g" model_input.cc

TF_PATH="$HOME/micro/tensorflow"

g++ -std=c++11 -DTF_LITE_STATIC_MEMORY -DNDEBUG -O3 \
  -DTF_LITE_DISABLE_X86_NEON \
  "-I${TF_PATH}" \
  "-I${TF_PATH}/tensorflow/lite/micro/tools/make/downloads/" \
  "-I${TF_PATH}/tensorflow/lite/micro/tools/make/downloads/gemmlowp" \
  "-I${TF_PATH}/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include" \
  "-I${TF_PATH}/tensorflow/lite/micro/tools/make/downloads/ruy" \
  "-I${TF_PATH}/tensorflow/lite/micro/tools/make/downloads/kissfft" \
  -c \
  model.cc \
  -o model.o

g++ -std=c++11 -DTF_LITE_STATIC_MEMORY -DNDEBUG -O3 \
  -DTF_LITE_DISABLE_X86_NEON \
  "-DINPUT_DATA=${input_val}" \
  "-I${TF_PATH}" \
  "-I${TF_PATH}/tensorflow/lite/micro/tools/make/downloads/" \
  "-I${TF_PATH}/tensorflow/lite/micro/tools/make/downloads/gemmlowp" \
  "-I${TF_PATH}/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include" \
  "-I${TF_PATH}/tensorflow/lite/micro/tools/make/downloads/ruy" \
  "-I${TF_PATH}/tensorflow/lite/micro/tools/make/downloads/kissfft" \
  -c \
  driver.cc \
  -o driver.o

g++ -std=c++11 -DTF_LITE_STATIC_MEMORY -DNDEBUG -O3 \
  -DTF_LITE_DISABLE_X86_NEON \
  "-I${TF_PATH}" \
  "-I${TF_PATH}/tensorflow/lite/micro/tools/make/downloads/" \
  "-I${TF_PATH}/tensorflow/lite/micro/tools/make/downloads/gemmlowp" \
  "-I${TF_PATH}/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include" \
  "-I${TF_PATH}/tensorflow/lite/micro/tools/make/downloads/ruy" \
  "-I${TF_PATH}/tensorflow/lite/micro/tools/make/downloads/kissfft" \
  driver.o \
  model.o \
  "${TF_PATH}/tensorflow/lite/micro/tools/make/gen/linux_x86_64/lib/libtensorflow-microlite.a" \
  -lm \
  -o run_model

./run_model

rm run_model
