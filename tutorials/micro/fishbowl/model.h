#ifndef TENSORFLOW_LITE_MICRO_MODEL_H_
#define TENSORFLOW_LITE_MICRO_MODEL_H_

extern const unsigned char g_model[];
extern const int g_model_len;

extern const unsigned char g_model_input[];
extern const unsigned int g_model_input_len;

extern const int g_model_input_ndims;
extern const int g_model_input_shape[];
extern const TfLiteType g_model_input_dtype;

extern const int g_model_output_ndims;
extern const int g_model_output_shape[];
extern const TfLiteType g_model_output_dtype;

#endif  // TENSORFLOW_LITE_MICRO_MODEL_H_
