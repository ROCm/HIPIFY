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

#if CUDA_VERSION >= 8000
  // TODO: [#899] There should be rocsparse_datatype instead of hipDataType
  cudaDataType_t dataType_t;
  cudaDataType dataType;
#endif

#if CUDA_VERSION >= 10010
  // TODO: cusparseCsr2CscAlg_t has no analogue in rocSPARSE. The deletion of declaration and usage is needed to be implemented
  cusparseCsr2CscAlg_t Csr2CscAlg_t;

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCsr2cscEx2_bufferSize(cusparseHandle_t handle, int m, int n, int nnz, const void* csrVal, const int* csrRowPtr, const int* csrColInd, void* cscVal, int* cscColPtr, int* cscRowInd, cudaDataType valType, cusparseAction_t copyValues, cusparseIndexBase_t idxBase, cusparseCsr2CscAlg_t alg, size_t* bufferSize);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_csr2csc_buffer_size(rocsparse_handle handle, rocsparse_int m, rocsparse_int n, rocsparse_int nnz, const rocsparse_int* csr_row_ptr, const rocsparse_int* csr_col_ind, rocsparse_action copy_values, size_t* buffer_size);
  // CHECK: status_t = rocsparse_csr2csc_buffer_size(handle_t, m, n, innz, &csrRowPtrA, &csrColIndA, action_t, &bufferSize);
  status_t = cusparseCsr2cscEx2_bufferSize(handle_t, m, n, innz, pcsrVal, &csrRowPtrA, &csrColIndA, pcscVal, &cscColPtrA, &cscRowIndA, dataType, action_t, indexBase_t, Csr2CscAlg_t, &bufferSize);
#endif

  return 0;
}
