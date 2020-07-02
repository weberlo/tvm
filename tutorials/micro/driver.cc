#include <iostream>

#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/micro/testing/micro_test.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

#include "model.h"

TF_LITE_MICRO_TESTS_BEGIN // {
{
  // Set up logging
  tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model = ::tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                        "Model provided is schema version %d not equal "
                        "to supported version %d.\n",
                        model->version(), TFLITE_SCHEMA_VERSION);
  }

  // This pulls in all the operation implementations we need
  tflite::AllOpsResolver resolver;

  // Create an area of memory to use for input, output, and intermediate arrays.

  // // Minimum arena size, at the time of writing. After allocating tensors
  // // you can retrieve this value by invoking interpreter.arena_used_bytes().
  // const int model_arena_size = 2468;
  // /* Extra headroom for model + alignment + future interpreter changes */
  // const int extra_arena_size = 570 + 16 + 100;
  // const int tensor_arena_size = model_arena_size + extra_arena_size;

  // TODO(weberlo) actually compute required space
  const int tensor_arena_size = 1000*1024;
  uint8_t tensor_arena[tensor_arena_size];

  // Build an interpreter to run the model with
  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena,
                                       tensor_arena_size, error_reporter);
  // Allocate memory from the tensor_arena for the model's tensors
  TF_LITE_MICRO_EXPECT_EQ(interpreter.AllocateTensors(), kTfLiteOk);

  // // Alert for substantial increase in arena size usage.
  // TF_LITE_MICRO_EXPECT_LE(interpreter.arena_used_bytes(),
  //                       model_arena_size + 100);
  // Obtain a pointer to the model's input tensor
  TfLiteTensor* input = interpreter.input(0);

  // Make sure the input has the properties we expect
  TF_LITE_MICRO_EXPECT_NE(nullptr, input);
  // The property "dims" tells us the tensor's shape. It has one element for
  // each dimension. Our input is a 2D tensor containing 1 element, so "dims"
  // should have size 2.
  TF_LITE_MICRO_EXPECT_EQ(g_model_input_ndims, input->dims->size);
  // The value of each element gives the length of the corresponding tensor.
  // We should expect two single element tensors (one is contained within the
  // other).

  TF_LITE_MICRO_EXPECT_EQ(g_model_input_dtype, input->type);
  size_t input_nbytes = 4;
  for (int i = 0; i < g_model_input_ndims; i++) {
    TF_LITE_MICRO_EXPECT_EQ(g_model_input_shape[i], input->dims->data[i]);
    input_nbytes *= g_model_input_shape[i];
  }
  TF_LITE_MICRO_EXPECT_EQ(input_nbytes, g_model_input_len);

  // Provide an input value
  if (input->type == kTfLiteFloat32) {
    memset(input->data.f, 0, input->bytes);
    memcpy(input->data.f,
           &g_model_input,
           g_model_input_len);
  } else {
    // unhandled dtype
    TF_LITE_MICRO_EXPECT_EQ(true, false);
  }

  // Run the model on this input and check that it succeeds
  TfLiteStatus invoke_status = interpreter.Invoke();
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, invoke_status);

  // Obtain a pointer to the output tensor and make sure it has the
  // properties we expect. It should be the same as the input tensor.
  TfLiteTensor* output = interpreter.output(0);
  TF_LITE_MICRO_EXPECT_EQ(g_model_output_dtype, output->type);
  size_t output_nbytes = 4;
  for (int i = 0; i < g_model_output_ndims; i++) {
    TF_LITE_MICRO_EXPECT_EQ(g_model_output_shape[i], output->dims->data[i]);
    output_nbytes *= g_model_output_shape[i];
  }

  fwrite(output->data.raw, 1, output_nbytes, stdout);
}

}
