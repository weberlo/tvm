/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 *  Copyright (c) 2019 by Contributors
 * \file utvm_runtime.cc
 * \brief uTVM runtime
 *
 * All function calls go through the externally defined `UTVMInit`, which
 * performs device-specific setup, then calls `UTVMMain`.  `UTVMMain` then
 * calls the function in `utvm_task` with the arguments from the task.
 *
 * Additionally included in this file are definitions for some of the most
 * common functions used in the C runtime API.
 */
#ifdef __cplusplus
extern "C" {
#endif

#include "utvm_runtime.h"

// Task pointers must be patched before calling a function.
  /*
UTVMTask utvm_task = {
    .func = NULL,
    .arg_values = NULL,
    .arg_type_codes = NULL,
    .num_args = 0,
};
*/

// TODO(weberlo): make all of these volatile

volatile UTVMTask utvm_tasks[20] = { };
volatile uint32_t utvm_num_tasks = 0;

volatile uint32_t utvm_word_size = 0;

// These pointers are patched at load time to point to the workspace section.
volatile char* utvm_workspace_start = NULL;  // NOLINT(*)
volatile char* utvm_workspace_end = NULL;    // NOLINT(*)
volatile char* utvm_workspace_curr = NULL;   // NOLINT(*)
// Keep track of how many active allocations there are on the workspace.
volatile uint32_t utvm_num_active_allocs = 0;

volatile int32_t utvm_return_code = 0;        // NOLINT(*)

volatile uint32_t utvm_task_time = 0;

volatile uint32_t utvm_done = 0;

// Gets called by UTVMInit, after device-specific initialization is finished.
void UTVMMain() {
  utvm_done = 0;
  utvm_workspace_curr = utvm_workspace_start;
  utvm_num_active_allocs = 0;
  utvm_return_code = UTVM_ERR_NOT_FINISHED;
  utvm_task_time = 0;
  UTVMTimerReset();
  int32_t err = UTVMTimerStart();
  if (err < 0) {
    utvm_return_code = err;
    UTVMDone();
  }
  for (int i = 0; i < utvm_num_tasks; i++) {
    utvm_return_code = utvm_tasks[i].func(
            (void*) utvm_tasks[i].arg_values,      // NOLINT(*)
            (void*) utvm_tasks[i].arg_type_codes,  // NOLINT(*)
            utvm_tasks[i].num_args);
    if (utvm_return_code < 0) {
      break;
    }
  }
  UTVMTimerStop();
  utvm_task_time = UTVMTimerRead();
  if (utvm_task_time < 0) {
    utvm_return_code = utvm_task_time;
  }
  if (utvm_return_code == UTVM_ERR_NOT_FINISHED) {
    utvm_return_code = UTVM_ERR_OK;
  }
  UTVMDone();
}

// We use a dummy function to signal execution is finished for device
// backends which require breakpoints.
void __attribute__ ((noinline)) UTVMDone() {
  utvm_done = 1;
}

// TODO(weberlo): modify C codegen to generate UTVM error codes for the failure
// modes in workspace alloc/free.

void* TVMBackendAllocWorkspace(int device_type, int device_id, uint64_t size,
                               int dtype_code_hint, int dtype_bits_hint) {
  // Align up to 8 bytes.
  utvm_workspace_curr +=
    (utvm_word_size - ((uintptr_t) utvm_workspace_curr % utvm_word_size)) % utvm_word_size;  // NOLINT(*)
  if (utvm_workspace_curr + size > utvm_workspace_end) {
    // Out of space in workspace.
    return NULL;
  }
  void* ret_ptr = (void*) utvm_workspace_curr;  // NOLINT(*)
  utvm_workspace_curr += size;
  utvm_num_active_allocs++;
  return ret_ptr;
}

int TVMBackendFreeWorkspace(int device_type, int device_id, void* ptr) {
  utvm_num_active_allocs--;
  if (utvm_num_active_allocs < 0) {
    TVMAPISetLastError("free called with no active workspace allocations");
    // Reset allocations and workspace (for future task executions).
    utvm_num_active_allocs = 0;
    utvm_workspace_curr = utvm_workspace_start;
    return UTVM_ERR_NO_ACTIVE_ALLOCS;
  } else if (utvm_num_active_allocs == 0) {
    // No more allocations.  Reset workspace.
    utvm_workspace_curr = utvm_workspace_start;
    return UTVM_ERR_OK;
  } else {
    return UTVM_ERR_OK;
  }
}

void TVMAPISetLastError(const char* msg) { }

void *memset(void *s, int c, size_t n) {
  char *p = s;
  while (n > 0) {
    *p = (char) c;
    p++;
    n--;
  }
}

#ifdef __cplusplus
}  // TVM_EXTERN_C
#endif
