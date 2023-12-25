/*
Copyright (c) 2015 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "CUDA2HIP.h"

// Maps CUDA header names to HIP header names
const std::map <llvm::StringRef, hipCounter> CUDA_INCLUDE_MAP {
  // CUDA includes
  {"cuda.h",                                                {"hip/hip_runtime.h",                                     "", CONV_INCLUDE_CUDA_MAIN_H,    API_DRIVER, 0}},
  {"cuda_runtime.h",                                        {"hip/hip_runtime.h",                                     "", CONV_INCLUDE_CUDA_MAIN_H,    API_RUNTIME, 0}},
  {"cuda_runtime_api.h",                                    {"hip/hip_runtime_api.h",                                 "", CONV_INCLUDE,                API_RUNTIME, 0}},
  {"channel_descriptor.h",                                  {"hip/channel_descriptor.h",                              "", CONV_INCLUDE,                API_RUNTIME, 0}},
  {"device_functions.h",                                    {"hip/device_functions.h",                                "", CONV_INCLUDE,                API_RUNTIME, 0}},
  {"driver_types.h",                                        {"hip/driver_types.h",                                    "", CONV_INCLUDE,                API_RUNTIME, 0}},
  {"cuda_fp16.h",                                           {"hip/hip_fp16.h",                                        "", CONV_INCLUDE,                API_RUNTIME, 0}},
  {"cuda_texture_types.h",                                  {"hip/hip_texture_types.h",                               "", CONV_INCLUDE,                API_RUNTIME, 0}},
  {"texture_fetch_functions.h",                             {"",                                                      "", CONV_INCLUDE,                API_RUNTIME, 0}},
  {"vector_types.h",                                        {"hip/hip_vector_types.h",                                "", CONV_INCLUDE,                API_RUNTIME, 0}},
  {"cuda_profiler_api.h",                                   {"hip/hip_runtime_api.h",                                 "", CONV_INCLUDE,                API_RUNTIME, 0}},
  {"cooperative_groups.h",                                  {"hip/hip_cooperative_groups.h",                          "", CONV_INCLUDE,                API_RUNTIME, 0}},
  {"library_types.h",                                       {"hip/library_types.h",                                   "", CONV_INCLUDE,                API_RUNTIME, 0}},
  // cuComplex includes
  {"cuComplex.h",                                           {"hip/hip_complex.h",                                     "", CONV_INCLUDE_CUDA_MAIN_H,    API_COMPLEX, 0}},
  // cuBLAS includes
  {"cublas.h",                                              {"hipblas.h",                                    "rocblas.h", CONV_INCLUDE_CUDA_MAIN_H,    API_BLAS, 0}},
  {"cublas_v2.h",                                           {"hipblas.h",                                    "rocblas.h", CONV_INCLUDE_CUDA_MAIN_V2_H, API_BLAS, 0}},
  {"cublas_api.h",                                          {"hipblas.h",                                    "rocblas.h", CONV_INCLUDE,                API_BLAS, 0}},
  // cuRAND includes
  {"curand.h",                                              {"hiprand/hiprand.h",                                     "", CONV_INCLUDE_CUDA_MAIN_H,    API_RAND, 0}},
  {"curand_kernel.h",                                       {"hiprand/hiprand_kernel.h",                              "", CONV_INCLUDE,                API_RAND, 0}},
  {"curand_discrete.h",                                     {"hiprand/hiprand_kernel.h",                              "", CONV_INCLUDE,                API_RAND, 0}},
  {"curand_discrete2.h",                                    {"hiprand/hiprand_kernel.h",                              "", CONV_INCLUDE,                API_RAND, 0}},
  {"curand_globals.h",                                      {"hiprand/hiprand_kernel.h",                              "", CONV_INCLUDE,                API_RAND, 0}},
  {"curand_lognormal.h",                                    {"hiprand/hiprand_kernel.h",                              "", CONV_INCLUDE,                API_RAND, 0}},
  {"curand_mrg32k3a.h",                                     {"hiprand/hiprand_kernel.h",                              "", CONV_INCLUDE,                API_RAND, 0}},
  {"curand_mtgp32.h",                                       {"hiprand/hiprand_kernel.h",                              "", CONV_INCLUDE,                API_RAND, 0}},
  {"curand_mtgp32_host.h",                                  {"hiprand/hiprand_mtgp32_host.h",                         "", CONV_INCLUDE,                API_RAND, 0}},
  {"curand_mtgp32_kernel.h",                                {"hiprand/hiprand_kernel.h",                              "", CONV_INCLUDE,                API_RAND, 0}},
  {"curand_mtgp32dc_p_11213.h",                             {"rocrand_mtgp32_11213.h",                                "", CONV_INCLUDE,                API_RAND, 0}},
  {"curand_normal.h",                                       {"hiprand/hiprand_kernel.h",                              "", CONV_INCLUDE,                API_RAND, 0}},
  {"curand_normal_static.h",                                {"hiprand/hiprand_kernel.h",                              "", CONV_INCLUDE,                API_RAND, 0}},
  {"curand_philox4x32_x.h",                                 {"hiprand/hiprand_kernel.h",                              "", CONV_INCLUDE,                API_RAND, 0}},
  {"curand_poisson.h",                                      {"hiprand/hiprand_kernel.h",                              "", CONV_INCLUDE,                API_RAND, 0}},
  {"curand_precalc.h",                                      {"hiprand/hiprand_kernel.h",                              "", CONV_INCLUDE,                API_RAND, 0}},
  {"curand_uniform.h",                                      {"hiprand/hiprand_kernel.h",                              "", CONV_INCLUDE,                API_RAND, 0}},
  // cuDNN includes
  {"cudnn.h",                                               {"hipDNN.h",                               "miopen/miopen.h", CONV_INCLUDE_CUDA_MAIN_H,    API_DNN, 0}},
  // cuFFT includes
  {"cufft.h",                                               {"hipfft/hipfft.h",                                       "", CONV_INCLUDE_CUDA_MAIN_H,    API_FFT, 0}},
  {"cufftXt.h",                                             {"hipfft/hipfftXt.h",                                     "", CONV_INCLUDE,                API_FFT, 0}},
  // cuSPARSE includes
  {"cusparse.h",                                            {"hipsparse.h",                                "rocsparse.h", CONV_INCLUDE_CUDA_MAIN_H,    API_SPARSE, 0}},
  {"cusparse_v2.h",                                         {"hipsparse.h",                                "rocsparse.h", CONV_INCLUDE_CUDA_MAIN_V2_H, API_SPARSE, 0}},
  // cuSOLVER includes
  {"cusolverDn.h",                                          {"hipsolver.h",                      "rocsolver/rocsolver.h", CONV_INCLUDE_CUDA_MAIN_H,    API_SOLVER, 0}},
  {"cusolverMg.h",                                          {"hipsolver.h",                      "rocsolver/rocsolver.h", CONV_INCLUDE_CUDA_MAIN_H,    API_SOLVER, 0}},
  {"cusolverRf.h",                                          {"hipsolver.h",                      "rocsolver/rocsolver.h", CONV_INCLUDE_CUDA_MAIN_H,    API_SOLVER, 0}},
  {"cusolverSp.h",                                          {"hipsolver.h",                      "rocsolver/rocsolver.h", CONV_INCLUDE_CUDA_MAIN_H,    API_SOLVER, 0}},
  {"cusolverSp_LOWLEVEL_PREVIEW.h",                         {"hipsolver.h",                      "rocsolver/rocsolver.h", CONV_INCLUDE_CUDA_MAIN_H,    API_SOLVER, 0}},
  {"cusolver_common.h",                                     {"hipsolver.h",                      "rocsolver/rocsolver.h", CONV_INCLUDE_CUDA_MAIN_H,    API_SOLVER, 0}},
  // CUB includes
  {"cub/cub.cuh",                                           {"hipcub/hipcub.hpp",                                     "", CONV_INCLUDE_CUDA_MAIN_H,    API_CUB, 0}},
  // CAFFE2 includes
  {"caffe2/core/common_gpu.h",                              {"caffe2/core/hip/common_gpu.h",                          "", CONV_INCLUDE,                API_CAFFE2, 0, UNSUPPORTED}},
  {"caffe2/core/context_gpu.h",                             {"caffe2/core/hip/context_gpu.h",                         "", CONV_INCLUDE,                API_CAFFE2, 0, UNSUPPORTED}},
  {"caffe2/operators/operator_fallback_gpu.h",              {"",                                                      "", CONV_INCLUDE,                API_CAFFE2, 0, UNSUPPORTED}},
  {"caffe2/operators/spatial_batch_norm_op.h",              {"caffe2/operators/hip/spatial_batch_norm_op_miopen.hip", "", CONV_INCLUDE,                API_CAFFE2, 0}},
  {"caffe2/operators/generate_proposals_op_util_nms_gpu.h", {"",                                                      "", CONV_INCLUDE,                API_CAFFE2, 0, UNSUPPORTED}},
  {"caffe2/operators/max_pool_with_index_gpu.h",            {"",                                                      "", CONV_INCLUDE,                API_CAFFE2, 0, UNSUPPORTED}},
  {"caffe2/operators/rnn/recurrent_network_executor_gpu.h", {"",                                                      "", CONV_INCLUDE,                API_CAFFE2, 0, UNSUPPORTED}},
  {"caffe2/utils/math/reduce.cuh",                          {"caffe2/utils/math/hip/reduce.cuh",                      "", CONV_INCLUDE,                API_CAFFE2, 0, UNSUPPORTED}},
  {"caffe2/operators/gather_op.cuh",                        {"caffe2/operators/math/gather_op.cuh",                   "", CONV_INCLUDE,                API_CAFFE2, 0, UNSUPPORTED}},
  {"caffe2/core/common_cudnn.h",                            {"caffe2/core/hip/common_miopen.h",                       "", CONV_INCLUDE,                API_CAFFE2, 0}},
  // RTC includes
  {"nvrtc.h",                                               {"hiprtc.h",                                              "", CONV_INCLUDE_CUDA_MAIN_H, API_RTC, 0}},
};

const std::map<llvm::StringRef, hipCounter> &CUDA_RENAMES_MAP() {
  static std::map<llvm::StringRef, hipCounter> ret;
  if (!ret.empty())
    return ret;
  // First run, so compute the union map.
  ret.insert(CUDA_DRIVER_TYPE_NAME_MAP.begin(), CUDA_DRIVER_TYPE_NAME_MAP.end());
  ret.insert(CUDA_DRIVER_FUNCTION_MAP.begin(), CUDA_DRIVER_FUNCTION_MAP.end());
  ret.insert(CUDA_RUNTIME_TYPE_NAME_MAP.begin(), CUDA_RUNTIME_TYPE_NAME_MAP.end());
  ret.insert(CUDA_RUNTIME_FUNCTION_MAP.begin(), CUDA_RUNTIME_FUNCTION_MAP.end());
  ret.insert(CUDA_COMPLEX_TYPE_NAME_MAP.begin(), CUDA_COMPLEX_TYPE_NAME_MAP.end());
  ret.insert(CUDA_COMPLEX_FUNCTION_MAP.begin(), CUDA_COMPLEX_FUNCTION_MAP.end());
  ret.insert(CUDA_BLAS_TYPE_NAME_MAP.begin(), CUDA_BLAS_TYPE_NAME_MAP.end());
  ret.insert(CUDA_BLAS_FUNCTION_MAP.begin(), CUDA_BLAS_FUNCTION_MAP.end());
  ret.insert(CUDA_RAND_TYPE_NAME_MAP.begin(), CUDA_RAND_TYPE_NAME_MAP.end());
  ret.insert(CUDA_RAND_FUNCTION_MAP.begin(), CUDA_RAND_FUNCTION_MAP.end());
  ret.insert(CUDA_DNN_TYPE_NAME_MAP.begin(), CUDA_DNN_TYPE_NAME_MAP.end());
  ret.insert(CUDA_DNN_FUNCTION_MAP.begin(), CUDA_DNN_FUNCTION_MAP.end());
  ret.insert(CUDA_FFT_TYPE_NAME_MAP.begin(), CUDA_FFT_TYPE_NAME_MAP.end());
  ret.insert(CUDA_FFT_FUNCTION_MAP.begin(), CUDA_FFT_FUNCTION_MAP.end());
  ret.insert(CUDA_SPARSE_TYPE_NAME_MAP.begin(), CUDA_SPARSE_TYPE_NAME_MAP.end());
  ret.insert(CUDA_SPARSE_FUNCTION_MAP.begin(), CUDA_SPARSE_FUNCTION_MAP.end());
  ret.insert(CUDA_CAFFE2_TYPE_NAME_MAP.begin(), CUDA_CAFFE2_TYPE_NAME_MAP.end());
  ret.insert(CUDA_CAFFE2_FUNCTION_MAP.begin(), CUDA_CAFFE2_FUNCTION_MAP.end());
  ret.insert(CUDA_CUB_TYPE_NAME_MAP.begin(), CUDA_CUB_TYPE_NAME_MAP.end());
  ret.insert(CUDA_CUB_FUNCTION_MAP.begin(), CUDA_CUB_FUNCTION_MAP.end());
  ret.insert(CUDA_RTC_TYPE_NAME_MAP.begin(), CUDA_RTC_TYPE_NAME_MAP.end());
  ret.insert(CUDA_RTC_FUNCTION_MAP.begin(), CUDA_RTC_FUNCTION_MAP.end());
  ret.insert(CUDA_DEVICE_TYPE_NAME_MAP.begin(), CUDA_DEVICE_TYPE_NAME_MAP.end());
  ret.insert(CUDA_SOLVER_TYPE_NAME_MAP.begin(), CUDA_SOLVER_TYPE_NAME_MAP.end());
  ret.insert(CUDA_SOLVER_FUNCTION_MAP.begin(), CUDA_SOLVER_FUNCTION_MAP.end());
  return ret;
};

const std::map<llvm::StringRef, cudaAPIversions> &CUDA_VERSIONS_MAP() {
  static std::map<llvm::StringRef, cudaAPIversions> ret;
  if (!ret.empty())
    return ret;
  // First run, so compute the union map.
  ret.insert(CUDA_DRIVER_TYPE_NAME_VER_MAP.begin(), CUDA_DRIVER_TYPE_NAME_VER_MAP.end());
  ret.insert(CUDA_DRIVER_FUNCTION_VER_MAP.begin(), CUDA_DRIVER_FUNCTION_VER_MAP.end());
  ret.insert(CUDA_RUNTIME_TYPE_NAME_VER_MAP.begin(), CUDA_RUNTIME_TYPE_NAME_VER_MAP.end());
  ret.insert(CUDA_RUNTIME_FUNCTION_VER_MAP.begin(), CUDA_RUNTIME_FUNCTION_VER_MAP.end());
  ret.insert(CUDA_COMPLEX_TYPE_NAME_VER_MAP.begin(), CUDA_COMPLEX_TYPE_NAME_VER_MAP.end());
  ret.insert(CUDA_COMPLEX_FUNCTION_VER_MAP.begin(), CUDA_COMPLEX_FUNCTION_VER_MAP.end());
  ret.insert(CUDA_BLAS_TYPE_NAME_VER_MAP.begin(), CUDA_BLAS_TYPE_NAME_VER_MAP.end());
  ret.insert(CUDA_BLAS_FUNCTION_VER_MAP.begin(), CUDA_BLAS_FUNCTION_VER_MAP.end());
  ret.insert(CUDA_RAND_TYPE_NAME_VER_MAP.begin(), CUDA_RAND_TYPE_NAME_VER_MAP.end());
  ret.insert(CUDA_RAND_FUNCTION_VER_MAP.begin(), CUDA_RAND_FUNCTION_VER_MAP.end());
  ret.insert(CUDA_DNN_TYPE_NAME_VER_MAP.begin(), CUDA_DNN_TYPE_NAME_VER_MAP.end());
  ret.insert(CUDA_DNN_FUNCTION_VER_MAP.begin(), CUDA_DNN_FUNCTION_VER_MAP.end());
  ret.insert(CUDA_FFT_TYPE_NAME_VER_MAP.begin(), CUDA_FFT_TYPE_NAME_VER_MAP.end());
  ret.insert(CUDA_FFT_FUNCTION_VER_MAP.begin(), CUDA_FFT_FUNCTION_VER_MAP.end());
  ret.insert(CUDA_SPARSE_TYPE_NAME_VER_MAP.begin(), CUDA_SPARSE_TYPE_NAME_VER_MAP.end());
  ret.insert(CUDA_SPARSE_FUNCTION_VER_MAP.begin(), CUDA_SPARSE_FUNCTION_VER_MAP.end());
  ret.insert(CUDA_CAFFE2_TYPE_NAME_VER_MAP.begin(), CUDA_CAFFE2_TYPE_NAME_VER_MAP.end());
  ret.insert(CUDA_CAFFE2_FUNCTION_VER_MAP.begin(), CUDA_CAFFE2_FUNCTION_VER_MAP.end());
  ret.insert(CUDA_DEVICE_TYPE_NAME_VER_MAP.begin(), CUDA_DEVICE_TYPE_NAME_VER_MAP.end());
  ret.insert(CUDA_DEVICE_FUNCTION_VER_MAP.begin(), CUDA_DEVICE_FUNCTION_VER_MAP.end());
  ret.insert(CUDA_CUB_TYPE_NAME_VER_MAP.begin(), CUDA_CUB_TYPE_NAME_VER_MAP.end());
  ret.insert(CUDA_CUB_FUNCTION_VER_MAP.begin(), CUDA_CUB_FUNCTION_VER_MAP.end());
  ret.insert(CUDA_RTC_TYPE_NAME_VER_MAP.begin(), CUDA_RTC_TYPE_NAME_VER_MAP.end());
  ret.insert(CUDA_RTC_FUNCTION_VER_MAP.begin(), CUDA_RTC_FUNCTION_VER_MAP.end());
  ret.insert(CUDA_SOLVER_TYPE_NAME_VER_MAP.begin(), CUDA_SOLVER_TYPE_NAME_VER_MAP.end());
  ret.insert(CUDA_SOLVER_FUNCTION_VER_MAP.begin(), CUDA_SOLVER_FUNCTION_VER_MAP.end());
  return ret;
}

const std::map<llvm::StringRef, hipAPIversions> &HIP_VERSIONS_MAP() {
  static std::map<llvm::StringRef, hipAPIversions> ret;
  if (!ret.empty())
    return ret;
  // First run, so compute the union map.
  ret.insert(HIP_DRIVER_TYPE_NAME_VER_MAP.begin(), HIP_DRIVER_TYPE_NAME_VER_MAP.end());
  ret.insert(HIP_DRIVER_FUNCTION_VER_MAP.begin(), HIP_DRIVER_FUNCTION_VER_MAP.end());
  ret.insert(HIP_RUNTIME_TYPE_NAME_VER_MAP.begin(), HIP_RUNTIME_TYPE_NAME_VER_MAP.end());
  ret.insert(HIP_RUNTIME_FUNCTION_VER_MAP.begin(), HIP_RUNTIME_FUNCTION_VER_MAP.end());
  ret.insert(HIP_COMPLEX_TYPE_NAME_VER_MAP.begin(), HIP_COMPLEX_TYPE_NAME_VER_MAP.end());
  ret.insert(HIP_COMPLEX_FUNCTION_VER_MAP.begin(), HIP_COMPLEX_FUNCTION_VER_MAP.end());
  ret.insert(HIP_BLAS_TYPE_NAME_VER_MAP.begin(), HIP_BLAS_TYPE_NAME_VER_MAP.end());
  ret.insert(HIP_BLAS_FUNCTION_VER_MAP.begin(), HIP_BLAS_FUNCTION_VER_MAP.end());
  ret.insert(HIP_RAND_TYPE_NAME_VER_MAP.begin(), HIP_RAND_TYPE_NAME_VER_MAP.end());
  ret.insert(HIP_RAND_FUNCTION_VER_MAP.begin(), HIP_RAND_FUNCTION_VER_MAP.end());
  ret.insert(HIP_DNN_TYPE_NAME_VER_MAP.begin(), HIP_DNN_TYPE_NAME_VER_MAP.end());
  ret.insert(HIP_DNN_FUNCTION_VER_MAP.begin(), HIP_DNN_FUNCTION_VER_MAP.end());
  ret.insert(HIP_FFT_TYPE_NAME_VER_MAP.begin(), HIP_FFT_TYPE_NAME_VER_MAP.end());
  ret.insert(HIP_FFT_FUNCTION_VER_MAP.begin(), HIP_FFT_FUNCTION_VER_MAP.end());
  ret.insert(HIP_SPARSE_TYPE_NAME_VER_MAP.begin(), HIP_SPARSE_TYPE_NAME_VER_MAP.end());
  ret.insert(HIP_SPARSE_FUNCTION_VER_MAP.begin(), HIP_SPARSE_FUNCTION_VER_MAP.end());
  ret.insert(HIP_CAFFE2_TYPE_NAME_VER_MAP.begin(), HIP_CAFFE2_TYPE_NAME_VER_MAP.end());
  ret.insert(HIP_CAFFE2_FUNCTION_VER_MAP.begin(), HIP_CAFFE2_FUNCTION_VER_MAP.end());
  ret.insert(HIP_DEVICE_TYPE_NAME_VER_MAP.begin(), HIP_DEVICE_TYPE_NAME_VER_MAP.end());
  ret.insert(HIP_DEVICE_FUNCTION_VER_MAP.begin(), HIP_DEVICE_FUNCTION_VER_MAP.end());
  ret.insert(HIP_CUB_TYPE_NAME_VER_MAP.begin(), HIP_CUB_TYPE_NAME_VER_MAP.end());
  ret.insert(HIP_CUB_FUNCTION_VER_MAP.begin(), HIP_CUB_FUNCTION_VER_MAP.end());
  ret.insert(HIP_RTC_TYPE_NAME_VER_MAP.begin(), HIP_RTC_TYPE_NAME_VER_MAP.end());
  ret.insert(HIP_RTC_FUNCTION_VER_MAP.begin(), HIP_RTC_FUNCTION_VER_MAP.end());
  ret.insert(HIP_SOLVER_TYPE_NAME_VER_MAP.begin(), HIP_SOLVER_TYPE_NAME_VER_MAP.end());
  ret.insert(HIP_SOLVER_FUNCTION_VER_MAP.begin(), HIP_SOLVER_FUNCTION_VER_MAP.end());
  return ret;
}
