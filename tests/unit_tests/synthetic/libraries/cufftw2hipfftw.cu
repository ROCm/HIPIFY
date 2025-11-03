// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --amap --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args -D__CUDA_API_VERSION_INTERNAL -ferror-limit=500

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <stdio.h>
// CHECK: #include "hipfft/hipfftw.h"
#include "cufftw.h"
// CHECK-NOT: #include "hipfftw.h"

int main() {
  printf("26. cufftw API to hipfftw API synthetic test\n");

  // CHECK: int W_FORWARD = FFTW_FORWARD;
  // CHECK-NEXT: int W_BACKWARD = FFTW_BACKWARD;
  // CHECK-NEXT: int W_ESTIMATE = FFTW_ESTIMATE;
  // CHECK-NEXT: int W_MEASURE = FFTW_MEASURE;
  // CHECK-NEXT: int W_PATIENT = FFTW_PATIENT;
  // CHECK-NEXT: int W_EXHAUSTIVE = FFTW_EXHAUSTIVE;
  // CHECK-NEXT: int W_WISDOM_ONLY = FFTW_WISDOM_ONLY;
  // CHECK-NEXT: int W_DESTROY_INPUT = FFTW_DESTROY_INPUT;
  // CHECK-NEXT: int W_PRESERVE_INPUT = FFTW_PRESERVE_INPUT;
  // CHECK-NEXT: int W_UNALIGNED = FFTW_UNALIGNED;
  int W_FORWARD = FFTW_FORWARD;
  int W_BACKWARD = FFTW_BACKWARD;
  int W_ESTIMATE = FFTW_ESTIMATE;
  int W_MEASURE = FFTW_MEASURE;
  int W_PATIENT = FFTW_PATIENT;
  int W_EXHAUSTIVE = FFTW_EXHAUSTIVE;
  int W_WISDOM_ONLY = FFTW_WISDOM_ONLY;
  int W_DESTROY_INPUT = FFTW_DESTROY_INPUT;
  int W_PRESERVE_INPUT = FFTW_PRESERVE_INPUT;
  int W_UNALIGNED = FFTW_UNALIGNED;

  // CHECK: fftw_complex w_complex;
  // CHECK-NEXT: fftwf_complex wf_complex;
  fftw_complex w_complex;
  fftwf_complex wf_complex;

  // CHECK: fftw_plan w_plan = nullptr;
  // CHECK-NEXT: fftwf_plan twf_plan = nullptr;
  fftw_plan w_plan = nullptr;
  fftwf_plan twf_plan = nullptr;

  return 0;
} 
