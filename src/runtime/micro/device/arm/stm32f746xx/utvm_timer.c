#ifdef __cplusplus
extern "C" {
#endif

#include "utvm_runtime.h"

// There are two implementations of cycle counters on the STM32F7X: SysTick and
// CYCCNT.  SysTick is preferred, as it gives better error handling, but the
// counter is only 24 bits wide.  If a larger timer is needed, use the CYCCNT
// implementation, which has a 32-bit counter.
#define USE_SYSTICK

#ifdef USE_SYSTICK

#define SYST_CSR    (*((volatile unsigned long *) 0xE000E010))
#define SYST_RVR    (*((volatile unsigned long *) 0xE000E014))
#define SYST_CVR    (*((volatile unsigned long *) 0xE000E018))
#define SYST_CALIB  (*((volatile unsigned long *) 0xE000E01C))

#define SYST_CSR_ENABLE     0
#define SYST_CSR_TICKINT    1
#define SYST_CSR_CLKSOURCE  2
#define SYST_COUNTFLAG      16

#define SYST_CALIB_NOREF  31
#define SYST_CALIB_SKEW   30

unsigned long start_time = 0;
unsigned long stop_time = 0;

int32_t UTVMTimerStart() {
  SYST_CSR = (1 << SYST_CSR_ENABLE) | (1 << SYST_CSR_CLKSOURCE);
  // wait until timer starts
  while (SYST_CVR == 0);
  start_time = SYST_CVR;
  return UTVM_ERR_OK;
}

void UTVMTimerStop() {
  SYST_CSR = 0;
  stop_time = SYST_CVR;
}

void UTVMTimerReset() {
  SYST_CSR = 0;
  // maximum reload value (24-bit)
  SYST_RVR = (~((unsigned long) 0)) >> 8;
  SYST_CVR = 0;
}

uint32_t UTVMTimerRead() {
  if (SYST_CSR & SYST_COUNTFLAG) {
    TVMAPISetLastError("timer overflowed");
    return UTVM_ERR_TIMER_OVERFLOW;
  } else {
    return start_time - stop_time;
  }
}

#else  // !USE_SYSTICK

#define DWT_CTRL    (*((volatile unsigned long *) 0xE0001000))
#define DWT_CYCCNT  (*((volatile unsigned long *) 0xE0001004))

#define DWT_CTRL_NOCYCCNT   25
#define DWT_CTRL_CYCCNTENA  0

unsigned long start_time = 0;
unsigned long stop_time = 0;

void UTVMTimerReset() {
  DWT_CYCCNT = 0;
}

int32_t UTVMTimerStart() {
  if (DWT_CTRL & DWT_CTRL_NOCYCCNT) {
    TVMAPISetLastError("cycle counter not implemented on device");
    return UTVM_ERR_TIMER_NOT_IMPLEMENTED;
  }
  start_time = DWT_CYCCNT;
  DWT_CTRL |= (1 << DWT_CTRL_CYCCNTENA);
}

void UTVMTimerStop() {
  stop_time = DWT_CYCCNT;
  DWT_CTRL &= ~(1 << DWT_CTRL_CYCCNTENA);
}

int32_t UTVMTimerRead() {
  // even with this check, we can't know for sure if the timer has overflowed
  // (it may have overflowed and gone past `start_time`).
  if (stop_time > start_time) {
    return stop_time - start_time;
  } else {
    return UTVM_ERR_TIMER_OVERFLOW;
  }
}

#endif  // USE_SYSTICK

#ifdef __cplusplus
}  // TVM_EXTERN_C
#endif
