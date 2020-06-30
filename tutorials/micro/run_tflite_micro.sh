#!/usr/bin/env bash

set -e

tflite_model="$1"
tflite_micro_model="$2"
replace_text="$3"
input_val="$4"

# Convert to a C source file
xxd -i "$tflite_model" > "$tflite_micro_model"
# Update variable names
sed -i "s/${replace_text}/g_model/g" "$tflite_micro_model"

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
  run_model.cc \
  -o run_model.o

g++ -std=c++11 -DTF_LITE_STATIC_MEMORY -DNDEBUG -O3 \
  -DTF_LITE_DISABLE_X86_NEON \
  "-I${TF_PATH}" \
  "-I${TF_PATH}/tensorflow/lite/micro/tools/make/downloads/" \
  "-I${TF_PATH}/tensorflow/lite/micro/tools/make/downloads/gemmlowp" \
  "-I${TF_PATH}/tensorflow/lite/micro/tools/make/downloads/flatbuffers/include" \
  "-I${TF_PATH}/tensorflow/lite/micro/tools/make/downloads/ruy" \
  "-I${TF_PATH}/tensorflow/lite/micro/tools/make/downloads/kissfft" \
  run_model.o \
  model.o \
  "${TF_PATH}/tensorflow/lite/micro/tools/make/gen/linux_x86_64/lib/libtensorflow-microlite.a" \
  -lm \
  -o run_model

./run_model

rm run_model
