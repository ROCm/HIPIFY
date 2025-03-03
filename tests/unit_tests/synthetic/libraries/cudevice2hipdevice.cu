// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --amap --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args -ferror-limit=500

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <stdio.h>
// CHECK: #include "hip/hip_fp8.h"
#include "cuda_fp8.h"
// CHECK-NOT: #include "hip/hip_fp8.h"
// CHECK-NOT: #include "cuda_fp8.h"

int main() {
  printf("24. CUDA Device API to HIP Device API synthetic test\n");

  double da = 0.0f;
  double dx = 0.0f;
  float fa = 0.0f;
  float fx = 0.0f;
  short int shi = 0;
  unsigned short int ushi = 0;
  double2 d2 = { 0.0f, 0.0f };
  float2 f2 = { 0.0f, 0.0f };
  __half hx = { 0.0f };
  __half hy = { 0.0f };
  __half2 h2 = { 0.0f, 0.0f };
  __half_raw hrx = { 0 };
  __half2_raw h2rx = { 0, 0 };

#if CUDA_VERSION >= 11000
  // CHECK: __hip_bfloat16 _bf16 = { 0.0f };
  // CHECK-NEXT: __hip_bfloat16 bf16a = { 0.0f };
  // CHECK-NEXT: __hip_bfloat16 bf16b = { 0.0f };
  __nv_bfloat16 _bf16 = { 0.0f };
  __nv_bfloat16 bf16a = { 0.0f };
  __nv_bfloat16 bf16b = { 0.0f };

  // CHECK: hip_bfloat16 bf16 = { 0 };
  nv_bfloat16 bf16 = { 0 };

  // CHECK: __hip_bfloat16_raw bf16r = { 0 };
  __nv_bfloat16_raw bf16r = { 0 };

  // CHECK: __hip_bfloat162 bf162 = { 0, 0 };
  // CHECK-NEXT: __hip_bfloat162 bf162a = { 0, 0 };
  // CHECK-NEXT: __hip_bfloat162 bf162b = { 0, 0 };
  __nv_bfloat162 bf162 = { 0, 0 };
  __nv_bfloat162 bf162a = { 0, 0 };
  __nv_bfloat162 bf162b = { 0, 0 };

  // CHECK: __hip_bfloat162_raw bf162r = { 0, 0 };
  __nv_bfloat162_raw bf162r = { 0, 0 };

  // CUDA: __CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __double2bfloat16(const double a);
  // HIP: __BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __double2bfloat16(const double a)
  // CHECK: _bf16 = __double2bfloat16(da);
  _bf16 = __double2bfloat16(da);

  // CUDA: __CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat16 __float2bfloat16(const float a);
  // HIP: __BF16_HOST_DEVICE_STATIC__ __hip_bfloat16 __float2bfloat16(float f);
  // CHECK: _bf16 = __float2bfloat16(fa);
  _bf16 = __float2bfloat16(fa);

  // CUDA: __CUDA_HOSTDEVICE_BF16_DECL__ float __bfloat162float(const __nv_bfloat16 a);
  // HIP: __BF16_HOST_DEVICE_STATIC__ float __bfloat162float(__hip_bfloat16 a);
  // CHECK: _bf16 = __bfloat162float(fa);
  _bf16 = __bfloat162float(fa);

  // CUDA: __CUDA_HOSTDEVICE_BF16_DECL__ float2 __bfloat1622float2(const __nv_bfloat162 a);
  // HIP: __BF16_HOST_DEVICE_STATIC__ float2 __bfloat1622float2(const __hip_bfloat162 a);
  // CHECK: f2 = __bfloat1622float2(bf162);
  f2 = __bfloat1622float2(bf162);

  // CUDA: __CUDA_HOSTDEVICE_BF16_DECL__ __nv_bfloat162 __float22bfloat162_rn(const float2 a);
  // HIP: __BF16_HOST_DEVICE_STATIC__ __hip_bfloat162 __float22bfloat162_rn(const float2 a);
  // CHECK: bf162 = __float22bfloat162_rn(f2);
  bf162 = __float22bfloat162_rn(f2);
#endif

#if CUDA_VERSION >= 11080
  // CHECK: __hip_fp8_storage_t fp8_storage_t;
  __nv_fp8_storage_t fp8_storage_t;

  // CHECK: __hip_fp8x2_storage_t fp8x2_storage_t;
  __nv_fp8x2_storage_t fp8x2_storage_t;

  // CHECK: __hip_fp8x4_storage_t fp8x4_storage_t;
  __nv_fp8x4_storage_t fp8x4_storage_t;

  // CHECK: __hip_fp8_e5m2_fnuz fp8_e5m2;
  __nv_fp8_e5m2 fp8_e5m2;

  // CHECK: __hip_fp8x2_e5m2_fnuz fp8x2_e5m2;
  __nv_fp8x2_e5m2 fp8x2_e5m2;

  // CHECK: __hip_fp8_e4m3_fnuz fp8_e4m3;
  __nv_fp8_e4m3 fp8_e4m3;

  // CHECK: __hip_fp8x2_e4m3_fnuz fp8x2_e4m3;
  __nv_fp8x2_e4m3 fp8x2_e4m3;

  // CHECK: __hip_fp8x4_e4m3_fnuz fp8x4_e4m3;
  __nv_fp8x4_e4m3 fp8x4_e4m3;

  // CHECK: __hip_saturation_t saturation_t;
  // CHECK-NEXT: __hip_saturation_t NOSAT = __HIP_NOSAT;
  // CHECK-NEXT: __hip_saturation_t SATFINITE = __HIP_SATFINITE;
  __nv_saturation_t saturation_t;
  __nv_saturation_t NOSAT = __NV_NOSAT;
  __nv_saturation_t SATFINITE = __NV_SATFINITE;

  // CHECK: __hip_fp8_interpretation_t fp8_interpretation_t;
  // CHECK-NEXT: __hip_fp8_interpretation_t E4M3 = __HIP_E4M3_FNUZ;
  // CHECK-NEXT: __hip_fp8_interpretation_t E5M2 = __HIP_E5M2_FNUZ;
  __nv_fp8_interpretation_t fp8_interpretation_t;
  __nv_fp8_interpretation_t E4M3 = __NV_E4M3;
  __nv_fp8_interpretation_t E5M2 = __NV_E5M2;

  // CHECK: __hip_fp8x4_e5m2_fnuz fp8x4_e5m2;
  __nv_fp8x4_e5m2 fp8x4_e5m2;

  // CUDA: __CUDA_HOSTDEVICE_FP8_DECL__ __nv_fp8_storage_t __nv_cvt_double_to_fp8(const double x, const __nv_saturation_t saturate, const __nv_fp8_interpretation_t fp8_interpretation);
  // HIP: __FP8_HOST_DEVICE_STATIC__ __hip_fp8_storage_t __hip_cvt_double_to_fp8(const double d, const __hip_saturation_t sat, const __hip_fp8_interpretation_t type);
  // CHECK: fp8_storage_t = __hip_cvt_double_to_fp8(dx, saturation_t, fp8_interpretation_t);
  fp8_storage_t = __nv_cvt_double_to_fp8(dx, saturation_t, fp8_interpretation_t);

  // CUDA: __CUDA_HOSTDEVICE_FP8_DECL__ __nv_fp8x2_storage_t __nv_cvt_double2_to_fp8x2(const double2 x, const __nv_saturation_t saturate, const __nv_fp8_interpretation_t fp8_interpretation);
  // HIP: __FP8_HOST_DEVICE_STATIC__ __hip_fp8x2_storage_t __hip_cvt_double2_to_fp8x2(const double2 d2, const __hip_saturation_t sat, const __hip_fp8_interpretation_t type);
  // CHECK: fp8x2_storage_t = __hip_cvt_double2_to_fp8x2(d2, saturation_t, fp8_interpretation_t);
  fp8x2_storage_t = __nv_cvt_double2_to_fp8x2(d2, saturation_t, fp8_interpretation_t);

  // CUDA: __CUDA_HOSTDEVICE_FP8_DECL__ __nv_fp8_storage_t __nv_cvt_float_to_fp8(const float x, const __nv_saturation_t saturate, const __nv_fp8_interpretation_t fp8_interpretation);
  // HIP: __FP8_HOST_DEVICE_STATIC__ __hip_fp8_storage_t __hip_cvt_float_to_fp8(const float f, const __hip_saturation_t sat, const __hip_fp8_interpretation_t type);
  // CHECK: fp8_storage_t = __hip_cvt_float_to_fp8(fx, saturation_t, fp8_interpretation_t);
  fp8_storage_t = __nv_cvt_float_to_fp8(fx, saturation_t, fp8_interpretation_t);

  // CUDA: __CUDA_HOSTDEVICE_FP8_DECL__ __nv_fp8x2_storage_t __nv_cvt_float2_to_fp8x2(const float2 x, const __nv_saturation_t saturate, const __nv_fp8_interpretation_t fp8_interpretation);
  // HIP: __FP8_HOST_DEVICE_STATIC__ __hip_fp8x2_storage_t __hip_cvt_float2_to_fp8x2(const float2 f2, const __hip_saturation_t sat, const __hip_fp8_interpretation_t type);
  // CHECK: fp8x2_storage_t = __hip_cvt_float2_to_fp8x2(f2, saturation_t, fp8_interpretation_t);
  fp8x2_storage_t = __nv_cvt_float2_to_fp8x2(f2, saturation_t, fp8_interpretation_t);

  // CUDA: __CUDA_HOSTDEVICE_FP8_DECL__ __nv_fp8_storage_t __nv_cvt_halfraw_to_fp8(const __half_raw x, const __nv_saturation_t saturate, const __nv_fp8_interpretation_t fp8_interpretation);
  // HIP: __FP8_HOST_DEVICE_STATIC__ __hip_fp8_storage_t __hip_cvt_halfraw_to_fp8(const __half_raw x, const __hip_saturation_t sat, const __hip_fp8_interpretation_t type);
  // CHECK: fp8_storage_t = __hip_cvt_halfraw_to_fp8(hrx, saturation_t, fp8_interpretation_t);
  fp8_storage_t = __nv_cvt_halfraw_to_fp8(hrx, saturation_t, fp8_interpretation_t);

  // CUDA: __CUDA_HOSTDEVICE_FP8_DECL__ __nv_fp8x2_storage_t __nv_cvt_halfraw2_to_fp8x2(const __half2_raw x, const __nv_saturation_t saturate, const __nv_fp8_interpretation_t fp8_interpretation);
  // HIP: __FP8_HOST_DEVICE_STATIC__ __hip_fp8x2_storage_t __hip_cvt_halfraw2_to_fp8x2(const __half2_raw x, const __hip_saturation_t sat, const __hip_fp8_interpretation_t type);
  // CHECK: fp8x2_storage_t = __hip_cvt_halfraw2_to_fp8x2(h2rx, saturation_t, fp8_interpretation_t);
  fp8x2_storage_t = __nv_cvt_halfraw2_to_fp8x2(h2rx, saturation_t, fp8_interpretation_t);

  // CUDA: __CUDA_HOSTDEVICE_FP8_DECL__ __nv_fp8_storage_t __nv_cvt_bfloat16raw_to_fp8(const __nv_bfloat16_raw x, const __nv_saturation_t saturate, const __nv_fp8_interpretation_t fp8_interpretation);
  // HIP: __FP8_HOST_DEVICE_STATIC__ __hip_fp8_storage_t __hip_cvt_bfloat16raw_to_fp8(const __hip_bfloat16_raw hr, const __hip_saturation_t sat, const __hip_fp8_interpretation_t type);
  // CHECK: fp8_storage_t = __hip_cvt_bfloat16raw_to_fp8(bf16r, saturation_t, fp8_interpretation_t);
  fp8_storage_t = __nv_cvt_bfloat16raw_to_fp8(bf16r, saturation_t, fp8_interpretation_t);

  // CUDA: __CUDA_HOSTDEVICE_FP8_DECL__ __nv_fp8x2_storage_t __nv_cvt_bfloat16raw2_to_fp8x2(const __nv_bfloat162_raw x, const __nv_saturation_t saturate, const __nv_fp8_interpretation_t fp8_interpretation);
  // HIP: __FP8_HOST_DEVICE_STATIC__ __hip_fp8x2_storage_t __hip_cvt_bfloat16raw2_to_fp8x2(const __hip_bfloat162_raw hr, const __hip_saturation_t sat, const __hip_fp8_interpretation_t type);
  // CHECK: fp8x2_storage_t = __hip_cvt_bfloat16raw2_to_fp8x2(bf162r, saturation_t, fp8_interpretation_t);
  fp8x2_storage_t = __nv_cvt_bfloat16raw2_to_fp8x2(bf162r, saturation_t, fp8_interpretation_t);

  // CUDA: __CUDA_HOSTDEVICE_FP8_DECL__ __half_raw __nv_cvt_fp8_to_halfraw(const __nv_fp8_storage_t x, const __nv_fp8_interpretation_t fp8_interpretation);
  // HIP: __FP8_HOST_DEVICE_STATIC__ __half_raw __hip_cvt_fp8_to_halfraw(const __hip_fp8_storage_t x, const __hip_fp8_interpretation_t type);
  // CHECK: hrx = __hip_cvt_fp8_to_halfraw(fp8_storage_t, fp8_interpretation_t);
  hrx = __nv_cvt_fp8_to_halfraw(fp8_storage_t, fp8_interpretation_t);

  // CUDA: __CUDA_HOSTDEVICE_FP8_DECL__ __half2_raw __nv_cvt_fp8x2_to_halfraw2(const __nv_fp8x2_storage_t x, const __nv_fp8_interpretation_t fp8_interpretation);
  // HIP: __FP8_HOST_DEVICE_STATIC__ __half2_raw __hip_cvt_fp8x2_to_halfraw2(const __hip_fp8x2_storage_t x, const __hip_fp8_interpretation_t type);
  // CHECK: h2rx = __hip_cvt_fp8x2_to_halfraw2(fp8x2_storage_t, fp8_interpretation_t);
  h2rx = __nv_cvt_fp8x2_to_halfraw2(fp8x2_storage_t, fp8_interpretation_t);
#endif

#if CUDA_VERSION >= 12020
  // CUDA: __CUDA_HOSTDEVICE_FP16_DECL__ __half2 make_half2(const __half x, const __half y);
  // HIP: __HOST_DEVICE__ __half2 make_half2(__half x, __half y);
  // CHECK: h2 = make_half2(hx, hy);
  h2 = make_half2(hx, hy);
#endif

  return 0;
}
