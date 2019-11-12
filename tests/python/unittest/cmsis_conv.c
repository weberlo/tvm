#include <stdint.h>
#include <utvm_runtime.h>
#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/c_backend_api.h>

#include <arm_math.h>
#include <arm_nnfunctions.h>

#define CONV2_IN_DIM 16
#define CONV2_IN_CH 32
#define CONV2_KER_DIM 5
#define CONV2_PAD 2
#define CONV2_STRIDE 1
#define CONV2_OUT_CH 32
#define CONV2_OUT_DIM 16

#define CONV2_BIAS_LSHIFT 0
#define CONV2_OUT_RSHIFT 9

#define CONV2_BIAS {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0}
static q7_t conv2_bias[CONV2_OUT_CH] = CONV2_BIAS;

int32_t arm_conv_wrapper(TVMValue* arg_values, int* arg_type_codes, int32_t num_args) {
  void* data_handle = (((TVMValue*)arg_values)[0].v_handle);
  void* kernel_handle = (((TVMValue*)arg_values)[1].v_handle);
  void* output_handle = (((TVMValue*)arg_values)[2].v_handle);

  int32_t dev_type = (((TVMArray*)data_handle)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)data_handle)[0].ctx.device_id);

  int8_t* data = (int8_t*)(((TVMArray*)data_handle)[0].data);
  int64_t* data_shape = (int64_t*)(((TVMArray*)data_handle)[0].shape);
  int64_t* data_strides = (int64_t*)(((TVMArray*)data_handle)[0].strides);
  int8_t* kernel = (int8_t*)(((TVMArray*)kernel_handle)[0].data);
  int64_t* kernel_shape = (int64_t*)(((TVMArray*)kernel_handle)[0].shape);
  int64_t* kernel_strides = (int64_t*)(((TVMArray*)kernel_handle)[0].strides);
  int8_t* output = (int8_t*)(((TVMArray*)output_handle)[0].data);
  int64_t* output_shape = (int64_t*)(((TVMArray*)output_handle)[0].shape);
  int64_t* output_strides = (int64_t*)(((TVMArray*)output_handle)[0].strides);

  //void* buffer1 = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)(32768 * sizeof(int8_t)), 2, 8);
  //void* buffer2 = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)(8192 * sizeof(int8_t)), 2, 8);
  void* col_buffer = TVMBackendAllocWorkspace(1, dev_id, (uint64_t)(6400 * sizeof(int8_t)), 2, 8);
  if (col_buffer == NULL) {
    return UTVM_ERR_ALLOC_TOO_LARGE;
  }

  arm_convolve_HWC_q7_fast(
    /* Im_in      */  data,
    /* dim_im_in  */  CONV2_IN_DIM,
    /* ch_im_in   */  CONV2_IN_CH,
    /* wt         */  kernel,
    /* ch_im_out  */  CONV2_OUT_CH,
    /* dim_kernel */  CONV2_KER_DIM,
    /* padding    */  CONV2_PAD,
    /* stride     */  CONV2_STRIDE,
    /* bias       */  conv2_bias,
    /* bias_shift */  CONV2_BIAS_LSHIFT,
    /* out_shift  */  CONV2_OUT_RSHIFT,
    /* Im_out     */  output,
    /* dim_im_out */  CONV2_OUT_DIM,
    /* bufferA    */  (q15_t*)col_buffer,
    /* bufferB    */  NULL);

  int32_t res = TVMBackendFreeWorkspace(1, dev_id, col_buffer);
  if (res != 0) {
    return res;
  }
  return 0;
}
