// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --amap --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args -ferror-limit=500

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <stdio.h>
// CHECK: #include "hip/hip_fp8.h"
#include "cuda_fp8.h"
// CHECK-NOT: #include "hip/hip_fp8.h"
// CHECK-NOT: #include "cuda_fp8.h"

#if CUDA_VERSION >= 12080
// CHECK: #include "hip/hip_fp4.h"
#include "cuda_fp4.h"
// CHECK-NOT: #include "hip/hip_fp4.h"
// CHECK-NOT: #include "cuda_fp4.h"
#endif

int main() {
  printf("24.12080, CUDA Device API to HIP Device API synthetic test\n");

  double dx = 0.0f;
  float fx = 0.0f;
  double2 d2 = { 0.0f, 0.0f };
  float2 f2 = { 0.0f, 0.0f };
  __half_raw hrx = { 0 };
  __half2_raw h2rx = { 0, 0 };

  // CHECK: hipRoundMode RoundMode;
  cudaRoundMode RoundMode;

#if CUDA_VERSION >= 11000
  // CHECK: __hip_bfloat16_raw bf16r = { 0 };
  __nv_bfloat16_raw bf16r = { 0 };

  // CHECK: __hip_bfloat162_raw bf162r = { 0, 0 };
  __nv_bfloat162_raw bf162r = { 0, 0 };
#endif

#if CUDA_VERSION >= 12080
  // CHECK: __hip_fp4_storage_t fp4_storage_t;
  // CHECK-NEXT: __hip_fp4x2_storage_t fp4x2_storage_t;
  // CHECK-NEXT: __hip_fp4x4_storage_t fp4x4_storage_t;
  // CHECK-NEXT: __hip_fp4_interpretation_t fp4_interpretation_t;
  // CHECK-NEXT: __hip_fp4_interpretation_t fp4_interpretation = __HIP_E2M1;
  // CHECK-NEXT: __hip_fp4_e2m1 fp4_e2m1;
  // CHECK-NEXT: __hip_fp4x2_e2m1 fp4x2_e2m1;
  // CHECK-NEXT: __hip_fp4x4_e2m1 fp4x4_e2m1;
  __nv_fp4_storage_t fp4_storage_t;
  __nv_fp4x2_storage_t fp4x2_storage_t;
  __nv_fp4x4_storage_t fp4x4_storage_t;
  __nv_fp4_interpretation_t fp4_interpretation_t;
  __nv_fp4_interpretation_t fp4_interpretation = __NV_E2M1;
  __nv_fp4_e2m1 fp4_e2m1;
  __nv_fp4x2_e2m1 fp4x2_e2m1;
  __nv_fp4x4_e2m1 fp4x4_e2m1;

  // CHECK: __hip_fp8_e8m0 fp8_e8m0;
  __nv_fp8_e8m0 fp8_e8m0;

  // CUDA: __CUDA_HOSTDEVICE_FP4_DECL__ __nv_fp4_storage_t __nv_cvt_double_to_fp4(const double x, const __nv_fp4_interpretation_t fp4_interpretation, const enum cudaRoundMode rounding);
  // HIP: __FP4_HOST_DEVICE_STATIC__ __hip_fp4_storage_t __hip_cvt_double_to_fp4(const double x, const __hip_fp4_interpretation_t, const enum hipRoundMode);
  // CHECK: fp4_storage_t = __hip_cvt_double_to_fp4(dx, fp4_interpretation_t, RoundMode);
  fp4_storage_t = __nv_cvt_double_to_fp4(dx, fp4_interpretation_t, RoundMode);

  // CUDA: __CUDA_HOSTDEVICE_FP4_DECL__ __nv_fp4x2_storage_t __nv_cvt_double2_to_fp4x2(const double2 x, const __nv_fp4_interpretation_t fp4_interpretation, const enum cudaRoundMode rounding);
  // HIP: __FP4_HOST_DEVICE_STATIC__ __hip_fp4x2_storage_t __hip_cvt_double2_to_fp4x2(const double2 x, const __hip_fp4_interpretation_t, const enum hipRoundMode);
  // CHECK: fp4x2_storage_t = __hip_cvt_double2_to_fp4x2(d2, fp4_interpretation_t, RoundMode);
  fp4x2_storage_t = __nv_cvt_double2_to_fp4x2(d2, fp4_interpretation_t, RoundMode);

  // CUDA: __CUDA_HOSTDEVICE_FP4_DECL__ __nv_fp4_storage_t __nv_cvt_float_to_fp4(const float x, const __nv_fp4_interpretation_t fp4_interpretation, const enum cudaRoundMode rounding);
  // HIP: __FP4_HOST_DEVICE_STATIC__ __hip_fp4_storage_t __hip_cvt_float_to_fp4(const float x, const __hip_fp4_interpretation_t, const enum hipRoundMode);
  // CHECK: fp4_storage_t = __hip_cvt_float_to_fp4(fx, fp4_interpretation_t, RoundMode);
  fp4_storage_t = __nv_cvt_float_to_fp4(fx, fp4_interpretation_t, RoundMode);

  // CUDA: __CUDA_HOSTDEVICE_FP4_DECL__ __nv_fp4x2_storage_t __nv_cvt_float2_to_fp4x2(const float2 x, const __nv_fp4_interpretation_t fp4_interpretation, const enum cudaRoundMode rounding);
  // HIP: __FP4_HOST_DEVICE_STATIC__ __hip_fp4x2_storage_t __hip_cvt_float2_to_fp4x2(const float2 x, const __hip_fp4_interpretation_t, const enum hipRoundMode);
  // CHECK: fp4x2_storage_t = __hip_cvt_float2_to_fp4x2(f2, fp4_interpretation_t, RoundMode);
  fp4x2_storage_t = __nv_cvt_float2_to_fp4x2(f2, fp4_interpretation_t, RoundMode);

  // CUDA: __CUDA_HOSTDEVICE_FP4_DECL__ __nv_fp4_storage_t __nv_cvt_bfloat16raw_to_fp4(const __nv_bfloat16_raw x, const __nv_fp4_interpretation_t fp4_interpretation, const enum cudaRoundMode rounding);
  // HIP: __FP4_HOST_DEVICE_STATIC__ __hip_fp4_storage_t __hip_cvt_bfloat16raw_to_fp4(const __hip_bfloat16_raw x, const __hip_fp4_interpretation_t, const enum hipRoundMode);
  // CHECK: fp4_storage_t = __hip_cvt_bfloat16raw_to_fp4(bf16r, fp4_interpretation_t, RoundMode);
  fp4_storage_t = __nv_cvt_bfloat16raw_to_fp4(bf16r, fp4_interpretation_t, RoundMode);

  // CUDA: __CUDA_HOSTDEVICE_FP4_DECL__ __nv_fp4x2_storage_t __nv_cvt_bfloat16raw2_to_fp4x2(const __nv_bfloat162_raw x, const __nv_fp4_interpretation_t fp4_interpretation, const enum cudaRoundMode rounding);
  // HIP: __FP4_HOST_DEVICE_STATIC__ __hip_fp4x2_storage_t __hip_cvt_bfloat16raw2_to_fp4x2(const __hip_bfloat162_raw x, const __hip_fp4_interpretation_t, const enum hipRoundMode);
  // CHECK: fp4x2_storage_t = __hip_cvt_bfloat16raw2_to_fp4x2(bf162r, fp4_interpretation_t, RoundMode);
  fp4x2_storage_t = __nv_cvt_bfloat16raw2_to_fp4x2(bf162r, fp4_interpretation_t, RoundMode);

  // CUDA: __CUDA_HOSTDEVICE_FP4_DECL__ __nv_fp4_storage_t __nv_cvt_halfraw_to_fp4(const __half_raw x, const __nv_fp4_interpretation_t fp4_interpretation, const enum cudaRoundMode rounding);
  // HIP: __FP4_HOST_DEVICE_STATIC__ __hip_fp4_storage_t __hip_cvt_halfraw_to_fp4(const __half_raw x, const __hip_fp4_interpretation_t, const enum hipRoundMode );
  // CHECK: fp4_storage_t = __hip_cvt_halfraw_to_fp4(hrx, fp4_interpretation_t, RoundMode);
  fp4_storage_t = __nv_cvt_halfraw_to_fp4(hrx, fp4_interpretation_t, RoundMode);

  // CUDA: __CUDA_HOSTDEVICE_FP4_DECL__ __nv_fp4x2_storage_t __nv_cvt_halfraw2_to_fp4x2(const __half2_raw x, const __nv_fp4_interpretation_t fp4_interpretation, const enum cudaRoundMode rounding);
  // HIP: __FP4_HOST_DEVICE_STATIC__ __hip_fp4x2_storage_t __hip_cvt_halfraw2_to_fp4x2(const __half2_raw x, const __hip_fp4_interpretation_t, const enum hipRoundMode);
  // CHECK: fp4x2_storage_t = __hip_cvt_halfraw2_to_fp4x2(h2rx, fp4_interpretation_t, RoundMode);
  fp4x2_storage_t = __nv_cvt_halfraw2_to_fp4x2(h2rx, fp4_interpretation_t, RoundMode);

  // CUDA: __CUDA_HOSTDEVICE_FP4_DECL__ __half_raw __nv_cvt_fp4_to_halfraw(const __nv_fp4_storage_t x, const __nv_fp4_interpretation_t fp4_interpretation);
  // HIP: __FP4_HOST_DEVICE_STATIC__ __half_raw __hip_cvt_fp4_to_halfraw(const __hip_fp4_storage_t x, const __hip_fp4_interpretation_t);
  // CHECK: hrx = __hip_cvt_fp4_to_halfraw(fp4_storage_t, fp4_interpretation_t);
  hrx = __nv_cvt_fp4_to_halfraw(fp4_storage_t, fp4_interpretation_t);

  // CUDA: __CUDA_HOSTDEVICE_FP4_DECL__ __half2_raw __nv_cvt_fp4x2_to_halfraw2(const __nv_fp4x2_storage_t x, const __nv_fp4_interpretation_t fp4_interpretation);
  // HIP: __FP4_HOST_DEVICE_STATIC__ __half2_raw __hip_cvt_fp4x2_to_halfraw2(const __hip_fp4x2_storage_t x, const __hip_fp4_interpretation_t);
  // CHECK: h2rx = __hip_cvt_fp4x2_to_halfraw2(fp4x2_storage_t, fp4_interpretation_t);
  h2rx = __nv_cvt_fp4x2_to_halfraw2(fp4x2_storage_t, fp4_interpretation_t);
#endif

  return 0;
}
