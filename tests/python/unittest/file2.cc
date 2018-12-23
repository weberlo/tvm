#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/c_backend_api.h"
//extern void* __tvm_module_ctx = NULL;
static void* __tvm_set_device_packed = NULL;
#ifdef __cplusplus
extern "C"
#endif
TVM_DLL int32_t fadd( void* args,  void* arg_type_ids, int32_t num_args) {
  TVMValue stack[2];
  void* stack_tcode = stack;
  TVMValue stack1[3];
  void* stack_value = stack1;
  if (!((num_args == 3))) {
    return -1;
  }
  void* arg0 = (((TVMValue*)args)[0].v_handle);
  int32_t arg0_code = (( int32_t*)arg_type_ids)[0];
  void* arg1 = (((TVMValue*)args)[1].v_handle);
  int32_t arg1_code = (( int32_t*)arg_type_ids)[1];
  void* arg2 = (((TVMValue*)args)[2].v_handle);
  int32_t arg2_code = (( int32_t*)arg_type_ids)[2];
  float* A = (float*)(((TVMArray*)arg0)[0].data);
  int64_t* arg0_shape = (int64_t*)(((TVMArray*)arg0)[0].shape);
  int64_t* arg0_strides = (int64_t*)(((TVMArray*)arg0)[0].strides);
  if (!(arg0_strides == NULL)) {
    if (!((1 == ((int32_t)arg0_strides[0])))) {
      return -1;
    }
  }
  int32_t dev_type = (((TVMArray*)arg0)[0].ctx.device_type);
  int32_t dev_id = (((TVMArray*)arg0)[0].ctx.device_id);
  float* B = (float*)(((TVMArray*)arg1)[0].data);
  int64_t* arg1_shape = (int64_t*)(((TVMArray*)arg1)[0].shape);
  int64_t* arg1_strides = (int64_t*)(((TVMArray*)arg1)[0].strides);
  if (!(arg1_strides == NULL)) {
    if (!((1 == ((int32_t)arg1_strides[0])))) {
      return -1;
    }
  }
  float* C = (float*)(((TVMArray*)arg2)[0].data);
  int64_t* arg2_shape = (int64_t*)(((TVMArray*)arg2)[0].shape);
  int64_t* arg2_strides = (int64_t*)(((TVMArray*)arg2)[0].strides);
  if (!(arg2_strides == NULL)) {
    if (!((1 == ((int32_t)arg2_strides[0])))) {
      return -1;
    }
  }
  if (!(((((arg0_code == 3) || (arg0_code == 13)) || (arg0_code == 7)) || (arg0_code == 4)))) {
    return -1;
  }
  if (!(((((arg1_code == 3) || (arg1_code == 13)) || (arg1_code == 7)) || (arg1_code == 4)))) {
    return -1;
  }
  if (!(((((arg2_code == 3) || (arg2_code == 13)) || (arg2_code == 7)) || (arg2_code == 4)))) {
    return -1;
  }
  if (!((1 == (((TVMArray*)arg0)[0].ndim)))) {
    return -1;
  }
  if (!(((((((TVMArray*)arg0)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg0)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg0)[0].dtype.lanes) == (uint16_t)1)))) {
    return -1;
  }
  if (!((((int32_t)arg0_shape[0]) == 1024))) {
    return -1;
  }
  if (!(((((TVMArray*)arg0)[0].byte_offset) == (uint64_t)0))) {
    return -1;
  }
  if (!((1 == (((TVMArray*)arg1)[0].ndim)))) {
    return -1;
  }
  if (!(((((((TVMArray*)arg1)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg1)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg1)[0].dtype.lanes) == (uint16_t)1)))) {
    return -1;
  }
  if (!((((int32_t)arg1_shape[0]) == 1024))) {
    return -1;
  }
  if (!(((((TVMArray*)arg1)[0].byte_offset) == (uint64_t)0))) {
    return -1;
  }
  if (!((dev_type == (((TVMArray*)arg1)[0].ctx.device_type)))) {
    return -1;
  }
  if (!((dev_id == (((TVMArray*)arg1)[0].ctx.device_id)))) {
    return -1;
  }
  if (!((1 == (((TVMArray*)arg2)[0].ndim)))) {
    return -1;
  }
  if (!(((((((TVMArray*)arg2)[0].dtype.code) == (uint8_t)2) && ((((TVMArray*)arg2)[0].dtype.bits) == (uint8_t)32)) && ((((TVMArray*)arg2)[0].dtype.lanes) == (uint16_t)1)))) {
    return -1;
  }
  if (!((((int32_t)arg2_shape[0]) == 1024))) {
    return -1;
  }
  if (!(((((TVMArray*)arg2)[0].byte_offset) == (uint64_t)0))) {
    return -1;
  }
  if (!((dev_type == (((TVMArray*)arg2)[0].ctx.device_type)))) {
    return -1;
  }
  if (!((dev_id == (((TVMArray*)arg2)[0].ctx.device_id)))) {
    return -1;
  }
  for (int32_t i0 = 0; i0 < 1024; ++i0) {
    C[i0] = (A[i0] + B[i0]);
  }
  return 0;
}

void* args;
void* arg_type_ids;
int32_t* num_args;
int (*func)(void*, void*, int32_t);

// init stub
int main()
{
  func(args, arg_type_ids, *num_args);
  return 0;
}
