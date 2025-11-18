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

  // CHECK: fftw_complex w_complex, w_complex_in, w_complex_out;
  // CHECK-NEXT: fftwf_complex wf_complex, wf_complex_in, wf_complex_out;
  fftw_complex w_complex, w_complex_in, w_complex_out;
  fftwf_complex wf_complex, wf_complex_in, wf_complex_out;

  // CHECK: fftw_plan w_plan = nullptr;
  // CHECK-NEXT: fftwf_plan twf_plan = nullptr;
  fftw_plan w_plan = nullptr;
  fftwf_plan twf_plan = nullptr;

  int n = 0;
  int n0 = 0;
  int n1 = 0;
  int n2 = 0;
  int sign = 0;
  int rank = 0;
  unsigned int flags = 0;
  double d_in = 0.0f;
  double d_out = 0.0f;
  float f_in = 0.0f;
  float f_out = 0.0f;

  // CUDA: fftw_plan CUFFTAPI fftw_plan_dft_1d(int n, fftw_complex* in, fftw_complex* out, int sign, unsigned flags);
  // HIP: HIPFFT_EXPORT fftw_plan fftw_plan_dft_1d(int n, fftw_complex * in, fftw_complex * out, int sign, unsigned flags);
  // CHECK: w_plan = fftw_plan_dft_1d(n, &w_complex_in, &w_complex_out, sign, flags);
  w_plan = fftw_plan_dft_1d(n, &w_complex_in, &w_complex_out, sign, flags);

  // CUDA: fftw_plan CUFFTAPI fftw_plan_dft_2d(int n0, int n1, fftw_complex* in, fftw_complex* out, int sign, unsigned flags);
  // HIP: HIPFFT_EXPORT fftw_plan fftw_plan_dft_2d(int n0, int n1, fftw_complex * in, fftw_complex * out, int sign, unsigned flags);
  // CHECK: w_plan = fftw_plan_dft_2d(n0, n1, &w_complex_in, &w_complex_out, sign, flags);
  w_plan = fftw_plan_dft_2d(n0, n1, &w_complex_in, &w_complex_out, sign, flags);

  // CUDA: fftw_plan CUFFTAPI fftw_plan_dft_3d(int n0, int n1, int n2, fftw_complex* in, fftw_complex* out, int sign, unsigned flags);
  // HIP: HIPFFT_EXPORT fftw_plan fftw_plan_dft_3d(int n0, int n1, int n2, fftw_complex* in, fftw_complex* out, int sign, unsigned flags);
  // CHECK: w_plan = fftw_plan_dft_3d(n0, n1, n2, &w_complex_in, &w_complex_out, sign, flags);
  w_plan = fftw_plan_dft_3d(n0, n1, n2, &w_complex_in, &w_complex_out, sign, flags);

  // CUDA: fftw_plan CUFFTAPI fftw_plan_dft(int rank, const int* n, fftw_complex* in, fftw_complex* out, int sign, unsigned flags);
  // HIP: HIPFFT_EXPORT fftw_plan fftw_plan_dft(int rank, const int* n, fftw_cBomplex* in, fftw_complex* out, int sign, unsigned flags);
  // CHECK: w_plan = fftw_plan_dft(rank, &n, &w_complex_in, &w_complex_out, sign, flags);
  w_plan = fftw_plan_dft(rank, &n, &w_complex_in, &w_complex_out, sign, flags);

  // CUDA: fftw_plan CUFFTAPI fftw_plan_dft_r2c_1d(int n, double* in, fftw_complex* out, unsigned flags);
  // HIP: HIPFFT_EXPORT fftw_plan fftw_plan_dft_r2c_1d(int n, double* in, fftw_complex* out, unsigned flags);
  // CHECK: w_plan = fftw_plan_dft_r2c_1d(n, &d_in, &w_complex_out, flags);
  w_plan = fftw_plan_dft_r2c_1d(n, &d_in, &w_complex_out, flags);

  // CUDA: fftw_plan CUFFTAPI fftw_plan_dft_r2c_2d(int n0, int n1, double* in, fftw_complex* out, unsigned flags);
  // HIP: HIPFFT_EXPORT fftw_plan fftw_plan_dft_r2c_2d(int n0, int n1, double* in, fftw_complex * out, unsigned flags);
  // CHECK: w_plan = fftw_plan_dft_r2c_2d(n0, n1, &d_in, &w_complex_out, flags);
  w_plan = fftw_plan_dft_r2c_2d(n0, n1, &d_in, &w_complex_out, flags);

  // CUDA: fftw_plan CUFFTAPI fftw_plan_dft_r2c_3d(int n0, int n1, int n2, double* in, fftw_complex* out, unsigned flags);
  // HIP: HIPFFT_EXPORT fftw_plan fftw_plan_dft_r2c_3d(int n0, int n1, int n2, double* in, fftw_complex * out, unsigned flags);
  // CHECK: w_plan = fftw_plan_dft_r2c_3d(n0, n1, n2, &d_in, &w_complex_out, flags);
  w_plan = fftw_plan_dft_r2c_3d(n0, n1, n2, &d_in, &w_complex_out, flags);

  // CUDA: fftw_plan CUFFTAPI fftw_plan_dft_r2c(int rank, const int* n, double* in, fftw_complex* out, unsigned flags);
  // HIP: HIPFFT_EXPORT fftw_plan fftw_plan_dft_r2c(int rank, const int* n, double* in, fftw_complex * out, unsigned flags);
  // CHECK: w_plan = fftw_plan_dft_r2c(rank, &n, &d_in, &w_complex_out, flags);
  w_plan = fftw_plan_dft_r2c(rank, &n, &d_in, &w_complex_out, flags);

  // CUDA: fftw_plan CUFFTAPI fftw_plan_dft_c2r_1d(int n, fftw_complex* in, double* out, unsigned flags);
  // HIP: HIPFFT_EXPORT fftw_plan fftw_plan_dft_c2r_1d(int n, fftw_complex* in, double* out, unsigned flags);
  // CHECK: w_plan = fftw_plan_dft_c2r_1d(n, &w_complex_in, &d_out, flags);
  w_plan = fftw_plan_dft_c2r_1d(n, &w_complex_in, &d_out, flags);

  // CUDA: fftw_plan CUFFTAPI fftw_plan_dft_c2r_2d(int n0, int n1, fftw_complex* in, double* out, unsigned flags);
  // HIP: HIPFFT_EXPORT fftw_plan fftw_plan_dft_c2r_2d(int n0, int n1, fftw_complex * in, double* out, unsigned flags);
  // CHECK: w_plan = fftw_plan_dft_c2r_2d(n0, n1, &w_complex_in, &d_out, flags);
  w_plan = fftw_plan_dft_c2r_2d(n0, n1, &w_complex_in, &d_out, flags);

  // CUDA: fftw_plan CUFFTAPI fftw_plan_dft_c2r_3d(int n0, int n1, int n2, fftw_complex* in, double* out, unsigned flags);
  // HIP: HIPFFT_EXPORT fftw_plan fftw_plan_dft_c2r_3d(int n0, int n1, int n2, fftw_complex * in, double* out, unsigned flags);
  // CHECK: w_plan = fftw_plan_dft_c2r_3d(n0, n1, n2, &w_complex_in, &d_out, flags);
  w_plan = fftw_plan_dft_c2r_3d(n0, n1, n2, &w_complex_in, &d_out, flags);

  // CUDA: fftw_plan CUFFTAPI fftw_plan_dft_c2r(int rank, const int* n, fftw_complex* in, double* out, unsigned flags);
  // HIP: HIPFFT_EXPORT fftw_plan fftw_plan_dft_c2r(int rank, const int* n, fftw_complex* in, double* out, unsigned flags);
  // CHECK: w_plan = fftw_plan_dft_c2r(rank, &n, &w_complex_in, &d_out, flags);
  w_plan = fftw_plan_dft_c2r(rank, &n, &w_complex_in, &d_out, flags);

  // CUDA: void CUFFTAPI fftw_execute(const fftw_plan plan);
  // HIP: HIPFFT_EXPORT void fftw_execute(const fftw_plan plan);
  // CHECK: fftw_execute(w_plan);
  fftw_execute(w_plan);

  // CUDA: fftwf_plan CUFFTAPI fftwf_plan_dft_1d(int n, fftwf_complex* in, fftwf_complex* out, int sign, unsigned flags);
  // HIP: HIPFFT_EXPORT fftwf_plan fftwf_plan_dft_1d(int n, fftwf_complex* in, fftwf_complex* out, int sign, unsigned flags);
  // CHECK: twf_plan = fftwf_plan_dft_1d(n, &wf_complex_in, &wf_complex_out, sign, flags);
  twf_plan = fftwf_plan_dft_1d(n, &wf_complex_in, &wf_complex_out, sign, flags);

  // CUDA: fftwf_plan CUFFTAPI fftwf_plan_dft_2d(int n0, int n1, fftwf_complex* in, fftwf_complex* out, int sign, unsigned flags);
  // HIP: HIPFFT_EXPORT fftwf_plan fftwf_plan_dft_2d(int n0, int n1, fftwf_complex* in, fftwf_complex* out, int sign, unsigned flags);
  // CHECK: twf_plan = fftwf_plan_dft_2d(n0, n1, &wf_complex_in, &wf_complex_out, sign, flags);
  twf_plan = fftwf_plan_dft_2d(n0, n1, &wf_complex_in, &wf_complex_out, sign, flags);

  // CUDA: fftwf_plan CUFFTAPI fftwf_plan_dft_3d(int n0, int n1, int n2, fftwf_complex* in, fftwf_complex* out, int sign, unsigned flags);
  // HIP: HIPFFT_EXPORT fftwf_plan fftwf_plan_dft_3d(int n0, int n1, int n2, fftwf_complex* in, fftwf_complex* out, int sign, unsigned flags);
  // CHECK: twf_plan = fftwf_plan_dft_3d(n0, n1, n2, &wf_complex_in, &wf_complex_out, sign, flags);
  twf_plan = fftwf_plan_dft_3d(n0, n1, n2, &wf_complex_in, &wf_complex_out, sign, flags);

  // CUDA: fftwf_plan CUFFTAPI fftwf_plan_dft(int rank, const int* n, fftwf_complex* in, fftwf_complex* out, int sign, unsigned flags);
  // HIP: HIPFFT_EXPORT fftwf_plan fftwf_plan_dft(int rank, const int* n, fftwf_complex* in, fftwf_complex* out, int sign, unsigned flags);
  // CHECK: twf_plan = fftwf_plan_dft(rank, &n, &wf_complex_in, &wf_complex_out, sign, flags);
  twf_plan = fftwf_plan_dft(rank, &n, &wf_complex_in, &wf_complex_out, sign, flags);

  // CUDA: fftwf_plan CUFFTAPI fftwf_plan_dft_r2c_1d(int n, float* in, fftwf_complex* out, unsigned flags);
  // HIP: fftwf_plan CUFFTAPI fftwf_plan_dft_r2c_1d(int n, float* in, fftwf_complex* out, unsigned flags);
  // CHECK: twf_plan = fftwf_plan_dft_r2c_1d(n, &f_in, &wf_complex_out, flags);
  twf_plan = fftwf_plan_dft_r2c_1d(n, &f_in, &wf_complex_out, flags);

  // CUDA: fftwf_plan CUFFTAPI fftwf_plan_dft_r2c_2d(int n0, int n1, float* in, fftwf_complex* out, unsigned flags);
  // HIP: HIPFFT_EXPORT fftwf_plan fftwf_plan_dft_r2c_2d(int n0, int n1, float* in, fftwf_complex* out, unsigned flags);
  // CHECK: twf_plan = fftwf_plan_dft_r2c_2d(n0, n1, &f_in, &wf_complex_out, flags);
  twf_plan = fftwf_plan_dft_r2c_2d(n0, n1, &f_in, &wf_complex_out, flags);

  // CUDA: fftwf_plan CUFFTAPI fftwf_plan_dft_r2c_3d(int n0, int n1, int n2, float* in, fftwf_complex* out, unsigned flags);
  // HIP: HIPFFT_EXPORT fftwf_plan fftwf_plan_dft_r2c_3d(int n0, int n1, int n2, float* in, fftwf_complex* out, unsigned flags);
  // CHECK: twf_plan = fftwf_plan_dft_r2c_3d(n0, n1, n2, &f_in, &wf_complex_out, flags);
  twf_plan = fftwf_plan_dft_r2c_3d(n0, n1, n2, &f_in, &wf_complex_out, flags);

  // CUDA: fftwf_plan CUFFTAPI fftwf_plan_dft_r2c(int rank, const int* n, float* in, fftwf_complex* out, unsigned flags);
  // HIP: HIPFFT_EXPORT fftwf_plan fftwf_plan_dft_r2c(int rank, const int* n, float* in, fftwf_complex* out, unsigned flags);
  // CHECK: twf_plan = fftwf_plan_dft_r2c(rank, &n, &f_in, &wf_complex_out, flags);
  twf_plan = fftwf_plan_dft_r2c(rank, &n, &f_in, &wf_complex_out, flags);

  // CUDA: fftwf_plan CUFFTAPI fftwf_plan_dft_c2r_1d(int n, fftwf_complex* in, float* out, unsigned flags);
  // HIP: HIPFFT_EXPORT fftwf_plan fftwf_plan_dft_c2r_1d(int n, fftwf_complex* in, float* out, unsigned flags);
  // CHECK: twf_plan = fftwf_plan_dft_c2r_1d(n, &wf_complex_in, &f_out, flags);
  twf_plan = fftwf_plan_dft_c2r_1d(n, &wf_complex_in, &f_out, flags);

  // CUDA: fftwf_plan CUFFTAPI fftwf_plan_dft_c2r_2d(int n0, int n1, fftwf_complex* in, float* out, unsigned flags);
  // HIP: HIPFFT_EXPORT fftwf_plan fftwf_plan_dft_c2r_2d(int n0, int n1, fftwf_complex* in, float* out, unsigned flags);
  // CHECK: twf_plan = fftwf_plan_dft_c2r_2d(n0, n1, &wf_complex_in, &f_out, flags);
  twf_plan = fftwf_plan_dft_c2r_2d(n0, n1, &wf_complex_in, &f_out, flags);

  // CUDA: fftwf_plan CUFFTAPI fftwf_plan_dft_c2r_3d(int n0, int n1, int n2, fftwf_complex* in, float* out, unsigned flags);
  // HIP: HIPFFT_EXPORT fftwf_plan fftwf_plan_dft_c2r_3d(int n0, int n1, int n2, fftwf_complex* in, float* out, unsigned flags);
  // CHECK: twf_plan = fftwf_plan_dft_c2r_3d(n0, n1, n2, &wf_complex_in, &f_out, flags);
  twf_plan = fftwf_plan_dft_c2r_3d(n0, n1, n2, &wf_complex_in, &f_out, flags);

  // CUDA: fftwf_plan CUFFTAPI fftwf_plan_dft_c2r(int rank, const int* n, fftwf_complex* in, float* out, unsigned flags);
  // HIP: HIPFFT_EXPORT fftwf_plan fftwf_plan_dft_c2r(int rank, const int* n, fftwf_complex* in, float* out, unsigned flags);
  // CHECK: twf_plan = fftwf_plan_dft_c2r(rank, &n, &wf_complex_in, &f_out, flags);
  twf_plan = fftwf_plan_dft_c2r(rank, &n, &wf_complex_in, &f_out, flags);

  return 0;
}
