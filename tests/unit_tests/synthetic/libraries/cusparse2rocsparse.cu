// RUN: %run_test hipify "%s" "%t" %hipify_args 4 --skip-excluded-preprocessor-conditional-blocks --experimental --roc --use-hip-data-types %clang_args -ferror-limit=500

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <stdio.h>
// CHECK: #include "rocsparse.h"
#include "cusparse.h"
// CHECK-NOT: #include "rocsparse.h"

int main() {
  printf("18. cuSPARSE API to rocSPARSE API synthetic test\n");

  // CHECK: _rocsparse_handle *handle = nullptr;
  // CHECK-NEXT: rocsparse_handle handle_t;
  cusparseContext *handle = nullptr;
  cusparseHandle_t handle_t;

  // CHECK: _rocsparse_mat_descr *matDescr = nullptr;
  // CHECK-NEXT: rocsparse_mat_descr matDescr_t, matDescr_t_2;
  cusparseMatDescr *matDescr = nullptr;
  cusparseMatDescr_t matDescr_t, matDescr_t_2;

  // CHECK: _rocsparse_color_info *colorInfo = nullptr;
  // CHECK-NEXT: rocsparse_color_info colorInfo_t;
  cusparseColorInfo *colorInfo = nullptr;
  cusparseColorInfo_t colorInfo_t;

  // CHECK: rocsparse_operation sparseOperation_t;
  // CHECK-NEXT: rocsparse_operation OPERATION_NON_TRANSPOSE = rocsparse_operation_none;
  // CHECK-NEXT: rocsparse_operation OPERATION_TRANSPOSE = rocsparse_operation_transpose;
  // CHECK-NEXT: rocsparse_operation OPERATION_CONJUGATE_TRANSPOSE = rocsparse_operation_conjugate_transpose;
  cusparseOperation_t sparseOperation_t;
  cusparseOperation_t OPERATION_NON_TRANSPOSE = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseOperation_t OPERATION_TRANSPOSE = CUSPARSE_OPERATION_TRANSPOSE;
  cusparseOperation_t OPERATION_CONJUGATE_TRANSPOSE = CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;

  // CHECK: rocsparse_index_base indexBase_t;
  // CHECK-NEXT: rocsparse_index_base INDEX_BASE_ZERO = rocsparse_index_base_zero;
  // CHECK-NEXT: rocsparse_index_base INDEX_BASE_ONE = rocsparse_index_base_one;
  cusparseIndexBase_t indexBase_t;
  cusparseIndexBase_t INDEX_BASE_ZERO = CUSPARSE_INDEX_BASE_ZERO;
  cusparseIndexBase_t INDEX_BASE_ONE = CUSPARSE_INDEX_BASE_ONE;

  // CHECK: rocsparse_matrix_type matrixType_t;
  // CHECK-NEXT: rocsparse_matrix_type MATRIX_TYPE_GENERAL = rocsparse_matrix_type_general;
  // CHECK-NEXT: rocsparse_matrix_type MATRIX_TYPE_SYMMETRIC = rocsparse_matrix_type_symmetric;
  // CHECK-NEXT: rocsparse_matrix_type MATRIX_TYPE_HERMITIAN = rocsparse_matrix_type_hermitian;
  // CHECK-NEXT: rocsparse_matrix_type MATRIX_TYPE_TRIANGULAR = rocsparse_matrix_type_triangular;
  cusparseMatrixType_t matrixType_t;
  cusparseMatrixType_t MATRIX_TYPE_GENERAL = CUSPARSE_MATRIX_TYPE_GENERAL;
  cusparseMatrixType_t MATRIX_TYPE_SYMMETRIC = CUSPARSE_MATRIX_TYPE_SYMMETRIC;
  cusparseMatrixType_t MATRIX_TYPE_HERMITIAN = CUSPARSE_MATRIX_TYPE_HERMITIAN;
  cusparseMatrixType_t MATRIX_TYPE_TRIANGULAR = CUSPARSE_MATRIX_TYPE_TRIANGULAR;

  // CHECK: rocsparse_diag_type diagType_t;
  // CHECK-NEXT: rocsparse_diag_type DIAG_TYPE_NON_UNIT = rocsparse_diag_type_non_unit;
  // CHECK-NEXT: rocsparse_diag_type DIAG_TYPE_UNIT = rocsparse_diag_type_unit;
  cusparseDiagType_t diagType_t;
  cusparseDiagType_t DIAG_TYPE_NON_UNIT = CUSPARSE_DIAG_TYPE_NON_UNIT;
  cusparseDiagType_t DIAG_TYPE_UNIT = CUSPARSE_DIAG_TYPE_UNIT;

  // CHECK: rocsparse_fill_mode fillMode_t;
  // CHECK-NEXT: rocsparse_fill_mode FILL_MODE_LOWER = rocsparse_fill_mode_lower;
  // CHECK-NEXT: rocsparse_fill_mode FILL_MODE_UPPER = rocsparse_fill_mode_upper;
  cusparseFillMode_t fillMode_t;
  cusparseFillMode_t FILL_MODE_LOWER = CUSPARSE_FILL_MODE_LOWER;
  cusparseFillMode_t FILL_MODE_UPPER = CUSPARSE_FILL_MODE_UPPER;

  // CHECK: rocsparse_action action_t;
  // CHECK-NEXT: rocsparse_action ACTION_SYMBOLIC = rocsparse_action_symbolic;
  // CHECK-NEXT: rocsparse_action ACTION_NUMERIC = rocsparse_action_numeric;
  cusparseAction_t action_t;
  cusparseAction_t ACTION_SYMBOLIC = CUSPARSE_ACTION_SYMBOLIC;
  cusparseAction_t ACTION_NUMERIC = CUSPARSE_ACTION_NUMERIC;

  // CHECK: rocsparse_direction direction_t;
  // CHECK-NEXT: rocsparse_direction DIRECTION_ROW = rocsparse_direction_row;
  // CHECK-NEXT: rocsparse_direction DIRECTION_COLUMN = rocsparse_direction_column;
  cusparseDirection_t direction_t;
  cusparseDirection_t DIRECTION_ROW = CUSPARSE_DIRECTION_ROW;
  cusparseDirection_t DIRECTION_COLUMN = CUSPARSE_DIRECTION_COLUMN;

  // CHECK: rocsparse_solve_policy solvePolicy_t;
  // CHECK-NEXT: rocsparse_solve_policy SOLVE_POLICY_NO_LEVEL = rocsparse_solve_policy_auto;
  // CHECK-NEXT: rocsparse_solve_policy SOLVE_POLICY_USE_LEVEL = rocsparse_solve_policy_auto;
  cusparseSolvePolicy_t solvePolicy_t;
  cusparseSolvePolicy_t SOLVE_POLICY_NO_LEVEL = CUSPARSE_SOLVE_POLICY_NO_LEVEL;
  cusparseSolvePolicy_t SOLVE_POLICY_USE_LEVEL = CUSPARSE_SOLVE_POLICY_USE_LEVEL;

  // CHECK: rocsparse_pointer_mode pointerMode_t;
  // CHECK-NEXT: rocsparse_pointer_mode POINTER_MODE_HOST = rocsparse_pointer_mode_host;
  // CHECK-NEXT: rocsparse_pointer_mode POINTER_MODE_DEVICE = rocsparse_pointer_mode_device;
  cusparsePointerMode_t pointerMode_t;
  cusparsePointerMode_t POINTER_MODE_HOST = CUSPARSE_POINTER_MODE_HOST;
  cusparsePointerMode_t POINTER_MODE_DEVICE = CUSPARSE_POINTER_MODE_DEVICE;

  // CHECK: rocsparse_status status_t;
  // CHECK-NEXT: rocsparse_status STATUS_SUCCESS = rocsparse_status_success;
  // CHECK-NEXT: rocsparse_status STATUS_NOT_INITIALIZED = rocsparse_status_not_initialized;
  // CHECK-NEXT: rocsparse_status STATUS_ALLOC_FAILED = rocsparse_status_memory_error;
  // CHECK-NEXT: rocsparse_status STATUS_INVALID_VALUE = rocsparse_status_invalid_value;
  // CHECK-NEXT: rocsparse_status STATUS_ARCH_MISMATCH = rocsparse_status_arch_mismatch;
  // CHECK-NEXT: rocsparse_status STATUS_INTERNAL_ERROR = rocsparse_status_internal_error;
  // CHECK-NEXT: rocsparse_status STATUS_ZERO_PIVOT = rocsparse_status_zero_pivot;
  cusparseStatus_t status_t;
  cusparseStatus_t STATUS_SUCCESS = CUSPARSE_STATUS_SUCCESS;
  cusparseStatus_t STATUS_NOT_INITIALIZED = CUSPARSE_STATUS_NOT_INITIALIZED;
  cusparseStatus_t STATUS_ALLOC_FAILED = CUSPARSE_STATUS_ALLOC_FAILED;
  cusparseStatus_t STATUS_INVALID_VALUE = CUSPARSE_STATUS_INVALID_VALUE;
  cusparseStatus_t STATUS_ARCH_MISMATCH = CUSPARSE_STATUS_ARCH_MISMATCH;
  cusparseStatus_t STATUS_INTERNAL_ERROR = CUSPARSE_STATUS_INTERNAL_ERROR;
  cusparseStatus_t STATUS_ZERO_PIVOT = CUSPARSE_STATUS_ZERO_PIVOT;

  // CHECK: hipStream_t stream_t;
  cudaStream_t stream_t;

  int iVal = 0;

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreate(cusparseHandle_t* handle);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_create_handle(rocsparse_handle* handle);
  // CHECK: status_t = rocsparse_create_handle(&handle_t);
  status_t = cusparseCreate(&handle_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDestroy(cusparseHandle_t handle);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_destroy_handle(rocsparse_handle handle);
  // CHECK: status_t = rocsparse_destroy_handle(handle_t);
  status_t = cusparseDestroy(handle_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSetStream(cusparseHandle_t handle, cudaStream_t streamId);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_set_stream(rocsparse_handle handle, hipStream_t stream);
  // CHECK: status_t = rocsparse_set_stream(handle_t, stream_t);
  status_t = cusparseSetStream(handle_t, stream_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseGetStream(cusparseHandle_t handle, cudaStream_t* streamId);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_get_stream(rocsparse_handle handle, hipStream_t* stream);
  // CHECK: status_t = rocsparse_get_stream(handle_t, &stream_t);
  status_t = cusparseGetStream(handle_t, &stream_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSetPointerMode(cusparseHandle_t handle, cusparsePointerMode_t mode);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_set_pointer_mode(rocsparse_handle handle, rocsparse_pointer_mode pointer_mode);
  // CHECK: status_t = rocsparse_set_pointer_mode(handle_t, pointerMode_t);
  status_t = cusparseSetPointerMode(handle_t, pointerMode_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseGetPointerMode(cusparseHandle_t handle, cusparsePointerMode_t* mode);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_get_pointer_mode(rocsparse_handle handle, rocsparse_pointer_mode* pointer_mode);
  // CHECK: status_t = rocsparse_get_pointer_mode(handle_t, &pointerMode_t);
  status_t = cusparseGetPointerMode(handle_t, &pointerMode_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseGetVersion(cusparseHandle_t handle, int* version);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_get_version(rocsparse_handle handle, int* version);
  // CHECK: status_t = rocsparse_get_version(handle_t, &iVal);
  status_t = cusparseGetVersion(handle_t, &iVal);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateMatDescr(cusparseMatDescr_t* descrA);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_create_mat_descr(rocsparse_mat_descr* descr);
  // CHECK: status_t = rocsparse_create_mat_descr(&matDescr_t);
  status_t = cusparseCreateMatDescr(&matDescr_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDestroyMatDescr(cusparseMatDescr_t descrA);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_destroy_mat_descr(rocsparse_mat_descr descr);
  // CHECK: status_t = rocsparse_destroy_mat_descr(matDescr_t);
  status_t = cusparseDestroyMatDescr(matDescr_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSetMatIndexBase(cusparseMatDescr_t descrA, cusparseIndexBase_t base);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_set_mat_index_base(rocsparse_mat_descr descr, rocsparse_index_base base);
  // CHECK: status_t = rocsparse_set_mat_index_base(matDescr_t, indexBase_t);
  status_t = cusparseSetMatIndexBase(matDescr_t, indexBase_t);

  // CUDA: cusparseIndexBase_t CUSPARSEAPI cusparseGetMatIndexBase(const cusparseMatDescr_t descrA);
  // ROC: ROCSPARSE_EXPORT rocsparse_index_base rocsparse_get_mat_index_base(const rocsparse_mat_descr descr);
  // CHECK: indexBase_t = rocsparse_get_mat_index_base(matDescr_t);
  indexBase_t = cusparseGetMatIndexBase(matDescr_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSetMatType(cusparseMatDescr_t descrA, cusparseMatrixType_t type);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_set_mat_type(rocsparse_mat_descr descr, rocsparse_matrix_type type);
  // CHECK: status_t = rocsparse_set_mat_type(matDescr_t, matrixType_t);
  status_t = cusparseSetMatType(matDescr_t, matrixType_t);

  // CUDA: cusparseMatrixType_t CUSPARSEAPI cusparseGetMatType(const cusparseMatDescr_t descrA);
  // ROC: ROCSPARSE_EXPORT rocsparse_matrix_type rocsparse_get_mat_type(const rocsparse_mat_descr descr);
  // CHECK: matrixType_t = rocsparse_get_mat_type(matDescr_t);
  matrixType_t = cusparseGetMatType(matDescr_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSetMatFillMode(cusparseMatDescr_t descrA, cusparseFillMode_t fillMode);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_set_mat_fill_mode(rocsparse_mat_descr descr, rocsparse_fill_mode fill_mode);
  // CHECK: status_t = rocsparse_set_mat_fill_mode(matDescr_t, fillMode_t);
  status_t = cusparseSetMatFillMode(matDescr_t, fillMode_t);

  // CUDA: cusparseFillMode_t CUSPARSEAPI cusparseGetMatFillMode(const cusparseMatDescr_t descrA);
  // ROC: ROCSPARSE_EXPORT rocsparse_fill_mode rocsparse_get_mat_fill_mode(const rocsparse_mat_descr descr);
  // CHECK: fillMode_t = rocsparse_get_mat_fill_mode(matDescr_t);
  fillMode_t = cusparseGetMatFillMode(matDescr_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseSetMatDiagType(cusparseMatDescr_t descrA, cusparseDiagType_t diagType);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_set_mat_diag_type(rocsparse_mat_descr descr, rocsparse_diag_type diag_type);
  // CHECK: status_t = rocsparse_set_mat_diag_type(matDescr_t, diagType_t);
  status_t = cusparseSetMatDiagType(matDescr_t, diagType_t);

  // CUDA: cusparseDiagType_t CUSPARSEAPI cusparseGetMatDiagType(const cusparseMatDescr_t descrA);
  // ROC: ROCSPARSE_EXPORT rocsparse_diag_type rocsparse_get_mat_diag_type(const rocsparse_mat_descr descr);
  // CHECK: diagType_t = rocsparse_get_mat_diag_type(matDescr_t);
  diagType_t = cusparseGetMatDiagType(matDescr_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateColorInfo(cusparseColorInfo_t* info);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_create_color_info(rocsparse_color_info* info);
  // CHECK: status_t = rocsparse_create_color_info(&colorInfo_t);
  status_t = cusparseCreateColorInfo(&colorInfo_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDestroyColorInfo(cusparseColorInfo_t info);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_destroy_color_info(rocsparse_color_info info);
  // CHECK: status_t = rocsparse_destroy_color_info(colorInfo_t);
  status_t = cusparseDestroyColorInfo(colorInfo_t);

#if CUDA_VERSION >= 8000
  // CHECK: hipDataType dataType_t;
  // CHECK-NEXT: hipDataType dataType;
  cudaDataType_t dataType_t;
  cudaDataType dataType;
#endif

#if CUDA_VERSION >= 8000 && CUDA_VERSION < 12000
  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCopyMatDescr(cusparseMatDescr_t dest, const cusparseMatDescr_t src);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_copy_mat_descr(rocsparse_mat_descr dest, const rocsparse_mat_descr src);
  // CHECK: status_t = rocsparse_copy_mat_descr(matDescr_t, matDescr_t_2);
  status_t = cusparseCopyMatDescr(matDescr_t, matDescr_t_2);
#endif

#if CUDA_VERSION >= 10010
  // CHECK: _rocsparse_spmat_descr *spMatDescr = nullptr;
  // CHECK-NEXT: rocsparse_spmat_descr spMatDescr_t;
  cusparseSpMatDescr *spMatDescr = nullptr;
  cusparseSpMatDescr_t spMatDescr_t;

  // CHECK: _rocsparse_dnmat_descr *dnMatDescr = nullptr;
  // CHECK-NEXT: rocsparse_dnmat_descr dnMatDescr_t;
  cusparseDnMatDescr *dnMatDescr = nullptr;
  cusparseDnMatDescr_t dnMatDescr_t;

  // CHECK: rocsparse_indextype indexType_t;
  // CHECK-NEXT: rocsparse_indextype INDEX_16U = rocsparse_indextype_u16;
  // CHECK-NEXT: rocsparse_indextype INDEX_32I = rocsparse_indextype_i32;
  // CHECK-NEXT: rocsparse_indextype INDEX_64I = rocsparse_indextype_i64;
  cusparseIndexType_t indexType_t;
  cusparseIndexType_t INDEX_16U = CUSPARSE_INDEX_16U;
  cusparseIndexType_t INDEX_32I = CUSPARSE_INDEX_32I;
  cusparseIndexType_t INDEX_64I = CUSPARSE_INDEX_64I;

  // CHECK: rocsparse_format format_t;
  // CHECK-NEXT: rocsparse_format FORMAT_CSR = rocsparse_format_csr;
  // CHECK-NEXT: rocsparse_format FORMAT_CSC = rocsparse_format_csc;
  // CHECK-NEXT: rocsparse_format FORMAT_CSO = rocsparse_format_coo;
  cusparseFormat_t format_t;
  cusparseFormat_t FORMAT_CSR = CUSPARSE_FORMAT_CSR;
  cusparseFormat_t FORMAT_CSC = CUSPARSE_FORMAT_CSC;
  cusparseFormat_t FORMAT_CSO = CUSPARSE_FORMAT_COO;

  // CHECK: rocsparse_order order_t;
  // CHECK-NEXT: rocsparse_order ORDER_COL = rocsparse_order_row;
  // CHECK-NEXT: rocsparse_order ORDER_ROW = rocsparse_order_column;
  cusparseOrder_t order_t;
  cusparseOrder_t ORDER_COL = CUSPARSE_ORDER_COL;
  cusparseOrder_t ORDER_ROW = CUSPARSE_ORDER_ROW;

  // CHECK: rocsparse_spmm_alg spMMAlg_t;
  cusparseSpMMAlg_t spMMAlg_t;
#endif

#if CUDA_VERSION >= 10020
  // CHECK: _rocsparse_spvec_descr *spVecDescr = nullptr;
  // CHECK-NEXT: rocsparse_spvec_descr spVecDescr_t;
  cusparseSpVecDescr *spVecDescr = nullptr;
  cusparseSpVecDescr_t spVecDescr_t;

  // CHECK: _rocsparse_dnvec_descr *dnVecDescr = nullptr;
  // CHECK-NEXT: rocsparse_dnvec_descr dnVecDescr_t;
  cusparseDnVecDescr *dnVecDescr = nullptr;
  cusparseDnVecDescr_t dnVecDescr_t;

  // CHECK: rocsparse_status STATUS_NOT_SUPPORTED = rocsparse_status_not_implemented;
  cusparseStatus_t STATUS_NOT_SUPPORTED = CUSPARSE_STATUS_NOT_SUPPORTED;

  // CHECK: rocsparse_spmv_alg spMVAlg_t;
  cusparseSpMVAlg_t spMVAlg_t;

  int64_t size = 0;
  int64_t nnz = 0;
  void* indices = nullptr;
  void* values = nullptr;

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateSpVec(cusparseSpVecDescr_t* spVecDescr, int64_t size, int64_t nnz, void* indices, void* values, cusparseIndexType_t idxType, cusparseIndexBase_t idxBase, cudaDataType valueType);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_create_spvec_descr(rocsparse_spvec_descr* descr, int64_t size, int64_t nnz, void* indices, void* values, rocsparse_indextype idx_type, rocsparse_index_base idx_base, rocsparse_datatype data_type);
  // CHECK: status_t = rocsparse_create_spvec_descr(&spVecDescr_t, size, nnz, indices, values, indexType_t, indexBase_t, dataType);
  status_t = cusparseCreateSpVec(&spVecDescr_t, size, nnz, indices, values, indexType_t, indexBase_t, dataType);
#endif

#if CUDA_VERSION >= 10020 && CUDA_VERSION < 12000
  // CHECK: rocsparse_format FORMAT_COO_AOS = rocsparse_format_coo_aos;
  cusparseFormat_t FORMAT_COO_AOS = CUSPARSE_FORMAT_COO_AOS;
#endif

#if CUDA_VERSION < 11000
  // CHECK: _rocsparse_hyb_mat *hybMat = nullptr;
  // CHECK-NEXT: rocsparse_hyb_mat hybMat_t;
  cusparseHybMat *hybMat = nullptr;
  cusparseHybMat_t hybMat_t;

  // CHECK: rocsparse_hyb_partition hybPartition_t;
  // CHECK-NEXT: rocsparse_hyb_partition HYB_PARTITION_AUTO = rocsparse_hyb_partition_auto;
  // CHECK-NEXT: rocsparse_hyb_partition HYB_PARTITION_USER = rocsparse_hyb_partition_user;
  // CHECK-NEXT: rocsparse_hyb_partition HYB_PARTITION_MAX = rocsparse_hyb_partition_max;
  cusparseHybPartition_t hybPartition_t;
  cusparseHybPartition_t HYB_PARTITION_AUTO = CUSPARSE_HYB_PARTITION_AUTO;
  cusparseHybPartition_t HYB_PARTITION_USER = CUSPARSE_HYB_PARTITION_USER;
  cusparseHybPartition_t HYB_PARTITION_MAX = CUSPARSE_HYB_PARTITION_MAX;

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseCreateHybMat(cusparseHybMat_t* hybA);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_create_hyb_mat(rocsparse_hyb_mat* hyb);
  // CHECK: status_t = rocsparse_create_hyb_mat(&hybMat_t);
  status_t = cusparseCreateHybMat(&hybMat_t);

  // CUDA: cusparseStatus_t CUSPARSEAPI cusparseDestroyHybMat(cusparseHybMat_t hybA);
  // ROC: ROCSPARSE_EXPORT rocsparse_status rocsparse_destroy_hyb_mat(rocsparse_hyb_mat hyb);
  // CHECK: status_t = rocsparse_destroy_hyb_mat(hybMat_t);
  status_t = cusparseDestroyHybMat(hybMat_t);
#endif

#if CUDA_VERSION >= 11000
  // CHECK: rocsparse_spmm_alg SPMM_ALG_DEFAULT = rocsparse_spmm_alg_default;
  // CHECK-NEXT: rocsparse_spmm_alg SPMM_COO_ALG1 = rocsparse_spmm_alg_coo_segmented;
  // CHECK-NEXT: rocsparse_spmm_alg SPMM_COO_ALG2 = rocsparse_spmm_alg_coo_atomic;
  // CHECK-NEXT: rocsparse_spmm_alg SPMM_COO_ALG3 = rocsparse_spmm_alg_coo_segmented_atomic;
  // CHECK-NEXT: rocsparse_spmm_alg SPMM_CSR_ALG1 = rocsparse_spmm_alg_csr;
  // CHECK-NEXT: rocsparse_spmm_alg SPMM_CSR_ALG2 = rocsparse_spmm_alg_csr_row_split;
  cusparseSpMMAlg_t SPMM_ALG_DEFAULT = CUSPARSE_SPMM_ALG_DEFAULT;
  cusparseSpMMAlg_t SPMM_COO_ALG1 = CUSPARSE_SPMM_COO_ALG1;
  cusparseSpMMAlg_t SPMM_COO_ALG2 = CUSPARSE_SPMM_COO_ALG2;
  cusparseSpMMAlg_t SPMM_COO_ALG3 = CUSPARSE_SPMM_COO_ALG3;
  cusparseSpMMAlg_t SPMM_CSR_ALG1 = CUSPARSE_SPMM_CSR_ALG1;
  cusparseSpMMAlg_t SPMM_CSR_ALG2 = CUSPARSE_SPMM_CSR_ALG2;

  // CHECK: rocsparse_spgemm_alg spGEMMAlg_t;
  // CHECK-NEXT: rocsparse_spgemm_alg SPGEMM_DEFAULT = rocsparse_spgemm_alg_default;
  cusparseSpGEMMAlg_t spGEMMAlg_t;
  cusparseSpGEMMAlg_t SPGEMM_DEFAULT = CUSPARSE_SPGEMM_DEFAULT;
#endif

#if CUDA_VERSION >= 11010
  // CHECK: rocsparse_sparse_to_dense_alg sparseToDenseAlg_t;
  // CHECK-NEXT: rocsparse_sparse_to_dense_alg SPARSETODENSE_ALG_DEFAULT = rocsparse_sparse_to_dense_alg_default;
  cusparseSparseToDenseAlg_t sparseToDenseAlg_t;
  cusparseSparseToDenseAlg_t SPARSETODENSE_ALG_DEFAULT = CUSPARSE_SPARSETODENSE_ALG_DEFAULT;

  // CHECK: rocsparse_dense_to_sparse_alg denseToSparseAlg_t;
  // CHECK-NEXT: rocsparse_dense_to_sparse_alg DENSETOSPARSE_ALG_DEFAULT = rocsparse_dense_to_sparse_alg_default;
  cusparseDenseToSparseAlg_t denseToSparseAlg_t;
  cusparseDenseToSparseAlg_t DENSETOSPARSE_ALG_DEFAULT = CUSPARSE_DENSETOSPARSE_ALG_DEFAULT;
#endif

#if CUDA_VERSION >= 11020
  // CHECK: rocsparse_format FORMAT_BLOCKED_ELL = rocsparse_format_bell;
  cusparseFormat_t FORMAT_BLOCKED_ELL = CUSPARSE_FORMAT_BLOCKED_ELL;

  // CHECK: rocsparse_spmv_alg SPMV_ALG_DEFAULT = rocsparse_spmv_alg_default;
  // CHECK-NEXT: rocsparse_spmv_alg SPMV_COO_ALG1 = rocsparse_spmv_alg_coo;
  // CHECK-NEXT: rocsparse_spmv_alg SPMV_COO_ALG2 = rocsparse_spmv_alg_coo_atomic;
  // CHECK-NEXT: rocsparse_spmv_alg SPMV_CSR_ALG1 = rocsparse_spmv_alg_csr_adaptive;
  // CHECK-NEXT: rocsparse_spmv_alg SPMV_CSR_ALG2 = rocsparse_spmv_alg_csr_stream;
  cusparseSpMVAlg_t SPMV_ALG_DEFAULT = CUSPARSE_SPMV_ALG_DEFAULT;
  cusparseSpMVAlg_t SPMV_COO_ALG1 = CUSPARSE_SPMV_COO_ALG1;
  cusparseSpMVAlg_t SPMV_COO_ALG2 = CUSPARSE_SPMV_COO_ALG2;
  cusparseSpMVAlg_t SPMV_CSR_ALG1 = CUSPARSE_SPMV_CSR_ALG1;
  cusparseSpMVAlg_t SPMV_CSR_ALG2 = CUSPARSE_SPMV_CSR_ALG2;

  // CHECK: rocsparse_spmm_alg SPMM_CSR_ALG3 = rocsparse_spmm_alg_csr_merge;
  // CHECK-NEXT: rocsparse_spmm_alg SPMM_BLOCKED_ELL_ALG1 = rocsparse_spmm_alg_bell;
  cusparseSpMMAlg_t SPMM_CSR_ALG3 = CUSPARSE_SPMM_CSR_ALG3;
  cusparseSpMMAlg_t SPMM_BLOCKED_ELL_ALG1 = CUSPARSE_SPMM_BLOCKED_ELL_ALG1;

  // CHECK: rocsparse_sddmm_alg sDDMMAlg_t;
  // CHECK-NEXT: rocsparse_sddmm_alg SDDMM_ALG_DEFAULT = rocsparse_sddmm_alg_default;
  cusparseSDDMMAlg_t sDDMMAlg_t;
  cusparseSDDMMAlg_t SDDMM_ALG_DEFAULT = CUSPARSE_SDDMM_ALG_DEFAULT;
#endif

#if CUDA_VERSION >= 11030
  // CHECK: rocsparse_spmat_attribute spMatAttribute_t;
  // CHECK-NEXT: rocsparse_spmat_attribute SPMAT_FILL_MODE = rocsparse_spmat_fill_mode;
  // CHECK-NEXT: rocsparse_spmat_attribute SPMAT_DIAG_TYPE = rocsparse_spmat_diag_type;
  cusparseSpMatAttribute_t spMatAttribute_t;
  cusparseSpMatAttribute_t SPMAT_FILL_MODE = CUSPARSE_SPMAT_FILL_MODE;
  cusparseSpMatAttribute_t SPMAT_DIAG_TYPE = CUSPARSE_SPMAT_DIAG_TYPE;

  // CHECK: rocsparse_spsv_alg spSVAlg_t;
  // CHECK-NEXT: rocsparse_spsv_alg SPSV_ALG_DEFAULT = rocsparse_spsv_alg_default;
  cusparseSpSVAlg_t spSVAlg_t;
  cusparseSpSVAlg_t SPSV_ALG_DEFAULT = CUSPARSE_SPSV_ALG_DEFAULT;

  // CHECK: rocsparse_spsm_alg spSMAlg_t;
  // CHECK-NEXT: rocsparse_spsm_alg SPSM_ALG_DEFAULT = rocsparse_spsm_alg_default;
  cusparseSpSMAlg_t spSMAlg_t;
  cusparseSpSMAlg_t SPSM_ALG_DEFAULT = CUSPARSE_SPSM_ALG_DEFAULT;
#endif

#if CUDA_VERSION >= 12010
  // CHECK: rocsparse_format FORMAT_BSR = rocsparse_format_bsr;
  // CHECK-NEXT: rocsparse_format FORMAT_SLICED_ELLPACK = rocsparse_format_ell;
  cusparseFormat_t FORMAT_BSR = CUSPARSE_FORMAT_BSR;
  cusparseFormat_t FORMAT_SLICED_ELLPACK = CUSPARSE_FORMAT_SLICED_ELLPACK;
#endif

#if CUDA_VERSION >= 12010 && CUSPARSE_VERSION >= 12100
  // CHECK: rocsparse_spmv_alg SPMV_SELL_ALG1 = rocsparse_spmv_alg_ell;
  cusparseSpMVAlg_t SPMV_SELL_ALG1 = CUSPARSE_SPMV_SELL_ALG1;
#endif

  return 0;
}
