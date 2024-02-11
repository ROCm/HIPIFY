// RUN: %run_test hipify "%s" "%t" %hipify_args 4 --skip-excluded-preprocessor-conditional-blocks --experimental --roc --use-hip-data-types %clang_args -ferror-limit=500

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
// CHECK: #include "hip/hip_complex.h"
#include "cuComplex.h"
#include <stdio.h>
// CHECK: #include "rocsparse.h"
#include "cusparse.h"
// CHECK-NOT: #include "rocsparse.h"

int main() {
  printf("18.1. cuSPARSE API to rocSPARSE API synthetic test\n");

  // CHECK: _rocsparse_handle *handle = nullptr;
  // CHECK-NEXT: rocsparse_handle handle_t;
  cusparseContext *handle = nullptr;
  cusparseHandle_t handle_t;

  // CHECK: rocsparse_status status_t;
  cusparseStatus_t status_t;

  // CHECK: _rocsparse_mat_descr *matDescr = nullptr;
  // CHECK-NEXT: rocsparse_mat_descr matDescr_t, matDescr_t_2, matDescr_A, matDescr_B, matDescr_C, matDescr_D;
  cusparseMatDescr *matDescr = nullptr;
  cusparseMatDescr_t matDescr_t, matDescr_t_2, matDescr_A, matDescr_B, matDescr_C, matDescr_D;

  // CHECK: rocsparse_action action_t;
  cusparseAction_t action_t;

  // CHECK: rocsparse_index_base indexBase_t;
  cusparseIndexBase_t indexBase_t;

  int m = 0;
  int n = 0;
  int innz = 0;
  int csrRowPtrA = 0;
  int csrRowPtrB = 0;
  int csrRowPtrC = 0;
  int cscRowIndA = 0;
  int csrColIndA = 0;
  int csrColIndB = 0;
  int csrColIndC = 0;
  int cscColPtrA = 0;
  size_t bufferSize = 0;
  void *pcsrVal = nullptr;
  void *pcscVal = nullptr;
  void *alpha = nullptr;
  void *beta = nullptr;
  void *tempBuffer = nullptr;

  // CHECK: rocsparse_operation opA, opB, opX;
  cusparseOperation_t opA, opB, opX;

#if CUDA_VERSION >= 8000
  // TODO: [#899] There should be rocsparse_datatype instead of hipDataType
  cudaDataType_t dataType_t;
  cudaDataType dataType;
#endif

#if CUDA_VERSION >= 10010
  // TODO: cusparseCsr2CscAlg_t has no analogue in rocSPARSE. The deletion of declaration and usage is needed to be implemented
  cusparseCsr2CscAlg_t Csr2CscAlg_t;

#if (CUDA_VERSION < 11000 && !defined(_WIN32)) || CUDA_VERSION >= 11000
  // CHECK: rocsparse_spmat_descr spMatDescr_t, spmatA, spmatB, spmatC;
  cusparseSpMatDescr_t spMatDescr_t, spmatA, spmatB, spmatC;

  // CHECK: rocsparse_dnmat_descr dnMatDescr_t, dnmatA, dnmatB, dnmatC;
  cusparseDnMatDescr_t dnMatDescr_t, dnmatA, dnmatB, dnmatC;

  // CHECK: rocsparse_spmm_alg spMMAlg_t;
  cusparseSpMMAlg_t spMMAlg_t;

#if CUDA_VERSION < 12000
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMM_bufferSize(cusparseHandle_t handle, cusparseOperation_t opA, cusparseOperation_t opB, const void* alpha, const cusparseSpMatDescr_t matA, const cusparseDnMatDescr_t matB, const void* beta, cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpMMAlg_t alg, size_t* bufferSize);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_spmm(rocsparse_handle handle, rocsparse_operation trans_A, rocsparse_operation trans_B, const void* alpha, const rocsparse_spmat_descr mat_A, const rocsparse_dnmat_descr mat_B, const void* beta, const rocsparse_dnmat_descr mat_C, rocsparse_datatype compute_type, rocsparse_spmm_alg alg, rocsparse_spmm_stage stage, size_t* buffer_size, void* temp_buffer);
  // CHECK: status_t = rocsparse_spmm(handle_t, opA, opB, alpha, spmatA, dnmatB, beta, dnmatC, dataType, spMMAlg_t, rocsparse_spmm_stage_compute, &bufferSize, nullptr);
  status_t = cusparseSpMM_bufferSize(handle_t, opA, opB, alpha, spmatA, dnmatB, beta, dnmatC, dataType, spMMAlg_t, &bufferSize);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSpMM(cusparseHandle_t handle, cusparseOperation_t transA, cusparseOperation_t transB, const void* alpha, const cusparseSpMatDescr_t matA, const cusparseDnMatDescr_t matB, const void* beta, cusparseDnMatDescr_t matC, cudaDataType computeType, cusparseSpMMAlg_t alg, void* externalBuffer);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_spmm(rocsparse_handle handle, rocsparse_operation trans_A, rocsparse_operation trans_B, const void* alpha, const rocsparse_spmat_descr mat_A, const rocsparse_dnmat_descr mat_B, const void* beta, const rocsparse_dnmat_descr mat_C, rocsparse_datatype compute_type, rocsparse_spmm_alg alg, rocsparse_spmm_stage stage, size_t* buffer_size, void* temp_buffer);
  // CHECK: status_t = rocsparse_spmm(handle_t, opA, opB, alpha, spmatA, dnmatB, beta, dnmatC, dataType, spMMAlg_t, rocsparse_spmm_stage_compute, nullptr, tempBuffer);
  status_t = cusparseSpMM(handle_t, opA, opB, alpha, spmatA, dnmatB, beta, dnmatC, dataType, spMMAlg_t, tempBuffer);
#endif
#endif
#endif

  return 0;
}
