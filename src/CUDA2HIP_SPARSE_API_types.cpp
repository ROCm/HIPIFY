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

// Maps the names of CUDA SPARSE API types to the corresponding HIP types
const std::map<llvm::StringRef, hipCounter> CUDA_SPARSE_TYPE_NAME_MAP = [] {
  std::map<llvm::StringRef, hipCounter> m;

  // 1. Structs
  m["cusparseContext"]                                                = {"hipsparseContext",                           "_rocsparse_handle",                                  CONV_TYPE, API_SPARSE, 4, HIP_UNSUPPORTED};
  m["cusparseHandle_t"]                                               = {"hipsparseHandle_t",                          "rocsparse_handle",                                   CONV_TYPE, API_SPARSE, 4};

  m["cusparseHybMat"]                                                 = {"hipsparseHybMat",                            "_rocsparse_hyb_mat",                                 CONV_TYPE, API_SPARSE, 4, HIP_UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cusparseHybMat_t"]                                               = {"hipsparseHybMat_t",                          "rocsparse_hyb_mat",                                  CONV_TYPE, API_SPARSE, 4, CUDA_DEPRECATED | CUDA_REMOVED};

  m["cusparseMatDescr"]                                               = {"hipsparseMatDescr",                          "_rocsparse_mat_descr",                               CONV_TYPE, API_SPARSE, 4, HIP_UNSUPPORTED};
  m["cusparseMatDescr_t"]                                             = {"hipsparseMatDescr_t",                        "rocsparse_mat_descr",                                CONV_TYPE, API_SPARSE, 4};

  m["cusparseSolveAnalysisInfo"]                                      = {"hipsparseSolveAnalysisInfo",                 "",                                                   CONV_TYPE, API_SPARSE, 4, UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["cusparseSolveAnalysisInfo_t"]                                    = {"hipsparseSolveAnalysisInfo_t",               "",                                                   CONV_TYPE, API_SPARSE, 4, UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};

  m["csrsv2Info"]                                                     = {"csrsv2Info",                                 "_rocsparse_mat_descr",                               CONV_TYPE, API_SPARSE, 4, HIP_UNSUPPORTED | CUDA_REMOVED};
  m["csrsv2Info_t"]                                                   = {"csrsv2Info_t",                               "rocsparse_mat_descr",                                CONV_TYPE, API_SPARSE, 4, CUDA_REMOVED};

  m["csrsm2Info"]                                                     = {"csrsm2Info",                                 "_rocsparse_mat_info",                                CONV_TYPE, API_SPARSE, 4, HIP_UNSUPPORTED | CUDA_REMOVED};
  m["csrsm2Info_t"]                                                   = {"csrsm2Info_t",                               "rocsparse_mat_info",                                 CONV_TYPE, API_SPARSE, 4, CUDA_REMOVED};

  m["bsrsv2Info"]                                                     = {"bsrsv2Info",                                 "_rocsparse_mat_info",                                CONV_TYPE, API_SPARSE, 4, CUDA_DEPRECATED};
  m["bsrsv2Info_t"]                                                   = {"bsrsv2Info_t",                               "rocsparse_mat_info",                                 CONV_TYPE, API_SPARSE, 4, CUDA_DEPRECATED};

  m["bsrsm2Info"]                                                     = {"bsrsm2Info",                                 "_rocsparse_mat_info",                                CONV_TYPE, API_SPARSE, 4, CUDA_DEPRECATED};
  m["bsrsm2Info_t"]                                                   = {"bsrsm2Info_t",                               "rocsparse_mat_info",                                 CONV_TYPE, API_SPARSE, 4, CUDA_DEPRECATED};

  m["bsric02Info"]                                                    = {"bsric02Info",                                "_rocsparse_mat_info",                                CONV_TYPE, API_SPARSE, 4, CUDA_DEPRECATED};
  m["bsric02Info_t"]                                                  = {"bsric02Info_t",                              "rocsparse_mat_info",                                 CONV_TYPE, API_SPARSE, 4, CUDA_DEPRECATED};

  m["csrilu02Info"]                                                   = {"csrilu02Info",                               "_rocsparse_mat_info",                                CONV_TYPE, API_SPARSE, 4, CUDA_DEPRECATED};
  m["csrilu02Info_t"]                                                 = {"csrilu02Info_t",                             "rocsparse_mat_info",                                 CONV_TYPE, API_SPARSE, 4, CUDA_DEPRECATED};

  m["bsrilu02Info"]                                                   = {"bsrilu02Info",                               "_rocsparse_mat_info",                                CONV_TYPE, API_SPARSE, 4, CUDA_DEPRECATED};
  m["bsrilu02Info_t"]                                                 = {"bsrilu02Info_t",                             "rocsparse_mat_info",                                 CONV_TYPE, API_SPARSE, 4, CUDA_DEPRECATED};

  m["csru2csrInfo"]                                                   = {"csru2csrInfo",                               "",                                                   CONV_TYPE, API_SPARSE, 4, ROC_UNSUPPORTED | CUDA_DEPRECATED};
  m["csru2csrInfo_t"]                                                 = {"csru2csrInfo_t",                             "",                                                   CONV_TYPE, API_SPARSE, 4, ROC_UNSUPPORTED | CUDA_DEPRECATED};

  m["csric02Info"]                                                    = {"csric02Info",                                "_rocsparse_mat_info",                                CONV_TYPE, API_SPARSE, 4, CUDA_DEPRECATED};
  m["csric02Info_t"]                                                  = {"csric02Info_t",                              "rocsparse_mat_info",                                 CONV_TYPE, API_SPARSE, 4, CUDA_DEPRECATED};

  m["csrgemm2Info"]                                                   = {"csrgemm2Info",                               "_rocsparse_mat_info",                                CONV_TYPE, API_SPARSE, 4, CUDA_REMOVED};
  m["csrgemm2Info_t"]                                                 = {"csrgemm2Info_t",                             "rocsparse_mat_info",                                 CONV_TYPE, API_SPARSE, 4, CUDA_REMOVED};

  m["cusparseColorInfo"]                                              = {"hipsparseColorInfo",                         "_rocsparse_color_info",                              CONV_TYPE, API_SPARSE, 4, HIP_UNSUPPORTED | CUDA_DEPRECATED};
  m["cusparseColorInfo_t"]                                            = {"hipsparseColorInfo_t",                       "rocsparse_color_info",                               CONV_TYPE, API_SPARSE, 4, CUDA_DEPRECATED};

  m["pruneInfo"]                                                      = {"pruneInfo",                                  "_rocsparse_mat_info",                                CONV_TYPE, API_SPARSE, 4, CUDA_DEPRECATED};
  m["pruneInfo_t"]                                                    = {"pruneInfo_t",                                "rocsparse_mat_info",                                 CONV_TYPE, API_SPARSE, 4, CUDA_DEPRECATED};

  m["cusparseSpMatDescr"]                                             = {"hipsparseSpMatDescr",                        "_rocsparse_spmat_descr",                             CONV_TYPE, API_SPARSE, 4, HIP_UNSUPPORTED};
  m["cusparseSpMatDescr_t"]                                           = {"hipsparseSpMatDescr_t",                      "rocsparse_spmat_descr",                              CONV_TYPE, API_SPARSE, 4};

  m["cusparseDnMatDescr"]                                             = {"hipsparseDnMatDescr",                        "_rocsparse_dnmat_descr",                             CONV_TYPE, API_SPARSE, 4, HIP_UNSUPPORTED};
  m["cusparseDnMatDescr_t"]                                           = {"hipsparseDnMatDescr_t",                      "rocsparse_dnmat_descr",                              CONV_TYPE, API_SPARSE, 4};

  m["cusparseSpVecDescr"]                                             = {"hipsparseSpVecDescr",                        "_rocsparse_spvec_descr",                             CONV_TYPE, API_SPARSE, 4, HIP_UNSUPPORTED};
  m["cusparseSpVecDescr_t"]                                           = {"hipsparseSpVecDescr_t",                      "rocsparse_spvec_descr",                              CONV_TYPE, API_SPARSE, 4};

  m["cusparseDnVecDescr"]                                             = {"hipsparseDnVecDescr",                        "_rocsparse_dnvec_descr",                             CONV_TYPE, API_SPARSE, 4, HIP_UNSUPPORTED};
  m["cusparseDnVecDescr_t"]                                           = {"hipsparseDnVecDescr_t",                      "rocsparse_dnvec_descr",                              CONV_TYPE, API_SPARSE, 4};

  m["cusparseSpGEMMDescr"]                                            = {"hipsparseSpGEMMDescr",                       "",                                                   CONV_TYPE, API_SPARSE, 4, ROC_UNSUPPORTED};
  m["cusparseSpGEMMDescr_t"]                                          = {"hipsparseSpGEMMDescr_t",                     "",                                                   CONV_TYPE, API_SPARSE, 4, ROC_UNSUPPORTED};

  m["cusparseSpSMDescr"]                                              = {"hipsparseSpSMDescr",                         "",                                                   CONV_TYPE, API_SPARSE, 4, ROC_UNSUPPORTED};
  m["cusparseSpSMDescr_t"]                                            = {"hipsparseSpSMDescr_t",                       "",                                                   CONV_TYPE, API_SPARSE, 4, ROC_UNSUPPORTED};

  m["cusparseSpSVDescr"]                                              = {"hipsparseSpSVDescr",                         "",                                                   CONV_TYPE, API_SPARSE, 4, ROC_UNSUPPORTED};
  m["cusparseSpSVDescr_t"]                                            = {"hipsparseSpSVDescr_t",                       "",                                                   CONV_TYPE, API_SPARSE, 4, ROC_UNSUPPORTED};

  m["cusparseSpMMOpPlan"]                                             = {"hipsparseSpMMOpPlan",                        "",                                                   CONV_TYPE, API_SPARSE, 4, UNSUPPORTED};
  m["cusparseSpMMOpPlan_t"]                                           = {"hipsparseSpMMOpPlan_t",                      "",                                                   CONV_TYPE, API_SPARSE, 4, UNSUPPORTED};

  // 2. Enums
  m["cusparseAction_t"]                                               = {"hipsparseAction_t",                          "rocsparse_action",                                   CONV_TYPE, API_SPARSE, 4};
  m["CUSPARSE_ACTION_SYMBOLIC"]                                       = {"HIPSPARSE_ACTION_SYMBOLIC",                  "rocsparse_action_symbolic",                          CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_ACTION_NUMERIC"]                                        = {"HIPSPARSE_ACTION_NUMERIC",                   "rocsparse_action_numeric",                           CONV_NUMERIC_LITERAL, API_SPARSE, 4};

  m["cusparseDirection_t"]                                            = {"hipsparseDirection_t",                       "rocsparse_direction",                                CONV_TYPE, API_SPARSE, 4};
  m["CUSPARSE_DIRECTION_ROW"]                                         = {"HIPSPARSE_DIRECTION_ROW",                    "rocsparse_direction_row",                            CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_DIRECTION_COLUMN"]                                      = {"HIPSPARSE_DIRECTION_COLUMN",                 "rocsparse_direction_column",                         CONV_NUMERIC_LITERAL, API_SPARSE, 4};

  m["cusparseHybPartition_t"]                                         = {"hipsparseHybPartition_t",                    "rocsparse_hyb_partition",                            CONV_TYPE, API_SPARSE, 4, CUDA_DEPRECATED | CUDA_REMOVED};
  m["CUSPARSE_HYB_PARTITION_AUTO"]                                    = {"HIPSPARSE_HYB_PARTITION_AUTO",               "rocsparse_hyb_partition_auto",                       CONV_NUMERIC_LITERAL, API_SPARSE, 4, CUDA_DEPRECATED | CUDA_REMOVED};
  m["CUSPARSE_HYB_PARTITION_USER"]                                    = {"HIPSPARSE_HYB_PARTITION_USER",               "rocsparse_hyb_partition_user",                       CONV_NUMERIC_LITERAL, API_SPARSE, 4, CUDA_DEPRECATED | CUDA_REMOVED};
  m["CUSPARSE_HYB_PARTITION_MAX"]                                     = {"HIPSPARSE_HYB_PARTITION_MAX",                "rocsparse_hyb_partition_max",                        CONV_NUMERIC_LITERAL, API_SPARSE, 4, CUDA_DEPRECATED | CUDA_REMOVED};

  m["cusparseDiagType_t"]                                             = {"hipsparseDiagType_t",                        "rocsparse_diag_type",                                CONV_TYPE, API_SPARSE, 4};
  m["CUSPARSE_DIAG_TYPE_NON_UNIT"]                                    = {"HIPSPARSE_DIAG_TYPE_NON_UNIT",               "rocsparse_diag_type_non_unit",                       CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_DIAG_TYPE_UNIT"]                                        = {"HIPSPARSE_DIAG_TYPE_UNIT",                   "rocsparse_diag_type_unit",                           CONV_NUMERIC_LITERAL, API_SPARSE, 4};

  m["cusparseFillMode_t"]                                             = {"hipsparseFillMode_t",                        "rocsparse_fill_mode",                                CONV_TYPE, API_SPARSE, 4};
  m["CUSPARSE_FILL_MODE_LOWER"]                                       = {"HIPSPARSE_FILL_MODE_LOWER",                  "rocsparse_fill_mode_lower",                          CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_FILL_MODE_UPPER"]                                       = {"HIPSPARSE_FILL_MODE_UPPER",                  "rocsparse_fill_mode_upper",                          CONV_NUMERIC_LITERAL, API_SPARSE, 4};

  m["cusparseIndexBase_t"]                                            = {"hipsparseIndexBase_t",                       "rocsparse_index_base",                               CONV_TYPE, API_SPARSE, 4};
  m["CUSPARSE_INDEX_BASE_ZERO"]                                       = {"HIPSPARSE_INDEX_BASE_ZERO",                  "rocsparse_index_base_zero",                          CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_INDEX_BASE_ONE"]                                        = {"HIPSPARSE_INDEX_BASE_ONE",                   "rocsparse_index_base_one",                           CONV_NUMERIC_LITERAL, API_SPARSE, 4};

  m["cusparseMatrixType_t"]                                           = {"hipsparseMatrixType_t",                      "rocsparse_matrix_type",                              CONV_TYPE, API_SPARSE, 4};
  m["CUSPARSE_MATRIX_TYPE_GENERAL"]                                   = {"HIPSPARSE_MATRIX_TYPE_GENERAL",              "rocsparse_matrix_type_general",                      CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_MATRIX_TYPE_SYMMETRIC"]                                 = {"HIPSPARSE_MATRIX_TYPE_SYMMETRIC",            "rocsparse_matrix_type_symmetric",                    CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_MATRIX_TYPE_HERMITIAN"]                                 = {"HIPSPARSE_MATRIX_TYPE_HERMITIAN",            "rocsparse_matrix_type_hermitian",                    CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_MATRIX_TYPE_TRIANGULAR"]                                = {"HIPSPARSE_MATRIX_TYPE_TRIANGULAR",           "rocsparse_matrix_type_triangular",                   CONV_NUMERIC_LITERAL, API_SPARSE, 4};

  m["cusparseOperation_t"]                                            = {"hipsparseOperation_t",                       "rocsparse_operation",                                CONV_TYPE, API_SPARSE, 4};
  m["CUSPARSE_OPERATION_NON_TRANSPOSE"]                               = {"HIPSPARSE_OPERATION_NON_TRANSPOSE",          "rocsparse_operation_none",                           CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_OPERATION_TRANSPOSE"]                                   = {"HIPSPARSE_OPERATION_TRANSPOSE",              "rocsparse_operation_transpose",                      CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE"]                         = {"HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE",    "rocsparse_operation_conjugate_transpose",            CONV_NUMERIC_LITERAL, API_SPARSE, 4};

  m["cusparsePointerMode_t"]                                          = {"hipsparsePointerMode_t",                     "rocsparse_pointer_mode",                             CONV_TYPE, API_SPARSE, 4};
  m["CUSPARSE_POINTER_MODE_HOST"]                                     = {"HIPSPARSE_POINTER_MODE_HOST",                "rocsparse_pointer_mode_host",                        CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_POINTER_MODE_DEVICE"]                                   = {"HIPSPARSE_POINTER_MODE_DEVICE",              "rocsparse_pointer_mode_device",                      CONV_NUMERIC_LITERAL, API_SPARSE, 4};

  m["cusparseAlgMode_t"]                                              = {"hipsparseAlgMode_t",                         "",                                                   CONV_TYPE, API_SPARSE, 4, UNSUPPORTED | CUDA_REMOVED};
  m["CUSPARSE_ALG0"]                                                  = {"CUSPARSE_ALG0",                              "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, UNSUPPORTED | CUDA_REMOVED};
  m["CUSPARSE_ALG1"]                                                  = {"CUSPARSE_ALG1",                              "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, UNSUPPORTED | CUDA_REMOVED};
  m["CUSPARSE_ALG_NAIVE"]                                             = {"CUSPARSE_ALG_NAIVE",                         "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, UNSUPPORTED | CUDA_REMOVED};
  m["CUSPARSE_ALG_MERGE_PATH"]                                        = {"CUSPARSE_ALG_MERGE_PATH",                    "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, UNSUPPORTED | CUDA_REMOVED};

  // TODO [HIPIFY] Warn about hipification to rocsparse_solve_policy_auto (automatically decide on level information)
  m["cusparseSolvePolicy_t"]                                          = {"hipsparseSolvePolicy_t",                     "rocsparse_solve_policy",                             CONV_TYPE, API_SPARSE, 4, CUDA_DEPRECATED};
  m["CUSPARSE_SOLVE_POLICY_NO_LEVEL"]                                 = {"HIPSPARSE_SOLVE_POLICY_NO_LEVEL",            "rocsparse_solve_policy_auto",                        CONV_NUMERIC_LITERAL, API_SPARSE, 4, CUDA_DEPRECATED};
  m["CUSPARSE_SOLVE_POLICY_USE_LEVEL"]                                = {"HIPSPARSE_SOLVE_POLICY_USE_LEVEL",           "rocsparse_solve_policy_auto",                        CONV_NUMERIC_LITERAL, API_SPARSE, 4, CUDA_DEPRECATED};

  m["cusparseStatus_t"]                                               = {"hipsparseStatus_t",                          "rocsparse_status",                                   CONV_TYPE, API_SPARSE, 4};
  m["CUSPARSE_STATUS_SUCCESS"]                                        = {"HIPSPARSE_STATUS_SUCCESS",                   "rocsparse_status_success",                           CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_STATUS_NOT_INITIALIZED"]                                = {"HIPSPARSE_STATUS_NOT_INITIALIZED",           "rocsparse_status_not_initialized",                   CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_STATUS_ALLOC_FAILED"]                                   = {"HIPSPARSE_STATUS_ALLOC_FAILED",              "rocsparse_status_memory_error",                      CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_STATUS_INVALID_VALUE"]                                  = {"HIPSPARSE_STATUS_INVALID_VALUE",             "rocsparse_status_invalid_value",                     CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_STATUS_ARCH_MISMATCH"]                                  = {"HIPSPARSE_STATUS_ARCH_MISMATCH",             "rocsparse_status_arch_mismatch",                     CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_STATUS_MAPPING_ERROR"]                                  = {"HIPSPARSE_STATUS_MAPPING_ERROR",             "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, ROC_UNSUPPORTED};
  m["CUSPARSE_STATUS_EXECUTION_FAILED"]                               = {"HIPSPARSE_STATUS_EXECUTION_FAILED",          "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, ROC_UNSUPPORTED};
  m["CUSPARSE_STATUS_INTERNAL_ERROR"]                                 = {"HIPSPARSE_STATUS_INTERNAL_ERROR",            "rocsparse_status_internal_error",                    CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED"]                      = {"HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED", "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, ROC_UNSUPPORTED};
  m["CUSPARSE_STATUS_ZERO_PIVOT"]                                     = {"HIPSPARSE_STATUS_ZERO_PIVOT",                "rocsparse_status_zero_pivot",                        CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_STATUS_NOT_SUPPORTED"]                                  = {"HIPSPARSE_STATUS_NOT_SUPPORTED",             "rocsparse_status_not_implemented",                   CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_STATUS_INSUFFICIENT_RESOURCES"]                         = {"HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES",    "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, ROC_UNSUPPORTED};

  m["cusparseCsr2CscAlg_t"]                                           = {"hipsparseCsr2CscAlg_t",                      "",                                                   CONV_TYPE, API_SPARSE, 4, ROC_UNSUPPORTED};
  m["CUSPARSE_CSR2CSC_ALG1"]                                          = {"HIPSPARSE_CSR2CSC_ALG1",                     "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, ROC_UNSUPPORTED};
  m["CUSPARSE_CSR2CSC_ALG2"]                                          = {"HIPSPARSE_CSR2CSC_ALG2",                     "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, ROC_UNSUPPORTED};
  m["CUSPARSE_CSR2CSC_ALG_DEFAULT"]                                   = {"HIPSPARSE_CSR2CSC_ALG_DEFAULT",              "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, ROC_UNSUPPORTED};

  m["cusparseFormat_t"]                                               = {"hipsparseFormat_t",                          "rocsparse_format",                                   CONV_TYPE, API_SPARSE, 4};
  m["CUSPARSE_FORMAT_CSR"]                                            = {"HIPSPARSE_FORMAT_CSR",                       "rocsparse_format_csr",                               CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_FORMAT_CSC"]                                            = {"HIPSPARSE_FORMAT_CSC",                       "rocsparse_format_csc",                               CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_FORMAT_COO"]                                            = {"HIPSPARSE_FORMAT_COO",                       "rocsparse_format_coo",                               CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_FORMAT_COO_AOS"]                                        = {"HIPSPARSE_FORMAT_COO_AOS",                   "rocsparse_format_coo_aos",                           CONV_NUMERIC_LITERAL, API_SPARSE, 4, CUDA_REMOVED};
  m["CUSPARSE_FORMAT_BLOCKED_ELL"]                                    = {"HIPSPARSE_FORMAT_BLOCKED_ELL",               "rocsparse_format_bell",                              CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_FORMAT_BSR"]                                            = {"HIPSPARSE_FORMAT_BSR",                       "rocsparse_format_bsr",                               CONV_NUMERIC_LITERAL, API_SPARSE, 4, HIP_UNSUPPORTED};
  m["CUSPARSE_FORMAT_SLICED_ELLPACK"]                                 = {"HIPSPARSE_FORMAT_SLICED_ELLPACK",            "rocsparse_format_ell",                               CONV_NUMERIC_LITERAL, API_SPARSE, 4, HIP_UNSUPPORTED};

  m["cusparseOrder_t"]                                                = {"hipsparseOrder_t",                           "rocsparse_order",                                    CONV_TYPE, API_SPARSE, 4};
  m["CUSPARSE_ORDER_COL"]                                             = {"HIPSPARSE_ORDER_COL",                        "rocsparse_order_row",                                CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_ORDER_ROW"]                                             = {"HIPSPARSE_ORDER_ROW",                        "rocsparse_order_column",                             CONV_NUMERIC_LITERAL, API_SPARSE, 4};

  m["cusparseSpMVAlg_t"]                                              = {"hipsparseSpMVAlg_t",                         "rocsparse_spmv_alg",                                 CONV_TYPE, API_SPARSE, 4};
  m["CUSPARSE_MV_ALG_DEFAULT"]                                        = {"HIPSPARSE_MV_ALG_DEFAULT",                   "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, ROC_UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["CUSPARSE_SPMV_ALG_DEFAULT"]                                      = {"HIPSPARSE_SPMV_ALG_DEFAULT",                 "rocsparse_spmv_alg_default",                         CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_COOMV_ALG"]                                             = {"HIPSPARSE_COOMV_ALG",                        "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, ROC_UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["CUSPARSE_SPMV_COO_ALG1"]                                         = {"HIPSPARSE_SPMV_COO_ALG1",                    "rocsparse_spmv_alg_coo",                             CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_SPMV_COO_ALG2"]                                         = {"HIPSPARSE_SPMV_COO_ALG2",                    "rocsparse_spmv_alg_coo_atomic",                      CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_CSRMV_ALG1"]                                            = {"HIPSPARSE_CSRMV_ALG1",                       "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, ROC_UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["CUSPARSE_SPMV_CSR_ALG1"]                                         = {"HIPSPARSE_SPMV_CSR_ALG1",                    "rocsparse_spmv_alg_csr_adaptive",                    CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_CSRMV_ALG2"]                                            = {"HIPSPARSE_CSRMV_ALG2",                       "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, ROC_UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["CUSPARSE_SPMV_CSR_ALG2"]                                         = {"HIPSPARSE_SPMV_CSR_ALG2",                    "rocsparse_spmv_alg_csr_stream",                      CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_SPMV_SELL_ALG1"]                                        = {"HIPSPARSE_SPMV_SELL_ALG1",                   "rocsparse_spmv_alg_ell",                             CONV_NUMERIC_LITERAL, API_SPARSE, 4, HIP_UNSUPPORTED};
  m["CUSPARSE_SPMV_BSR_ALG1"]                                         = {"HIPSPARSE_SPMV_BSR_ALG1",                    "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, UNSUPPORTED};

  m["cusparseSpMMAlg_t"]                                              = {"hipsparseSpMMAlg_t",                         "rocsparse_spmm_alg",                                 CONV_TYPE, API_SPARSE, 4};
  m["CUSPARSE_MM_ALG_DEFAULT"]                                        = {"HIPSPARSE_MM_ALG_DEFAULT",                   "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, ROC_UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["CUSPARSE_SPMM_ALG_DEFAULT"]                                      = {"HIPSPARSE_SPMM_ALG_DEFAULT",                 "rocsparse_spmm_alg_default",                         CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_COOMM_ALG1"]                                            = {"HIPSPARSE_COOMM_ALG1",                       "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, ROC_UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["CUSPARSE_SPMM_COO_ALG1"]                                         = {"HIPSPARSE_SPMM_COO_ALG1",                    "rocsparse_spmm_alg_coo_segmented",                   CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_COOMM_ALG2"]                                            = {"HIPSPARSE_COOMM_ALG2",                       "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, ROC_UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["CUSPARSE_SPMM_COO_ALG2"]                                         = {"HIPSPARSE_SPMM_COO_ALG2",                    "rocsparse_spmm_alg_coo_atomic",                      CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_COOMM_ALG3"]                                            = {"HIPSPARSE_COOMM_ALG3",                       "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, ROC_UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["CUSPARSE_SPMM_COO_ALG3"]                                         = {"HIPSPARSE_SPMM_COO_ALG3",                    "rocsparse_spmm_alg_coo_segmented_atomic",            CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_CSRMM_ALG1"]                                            = {"HIPSPARSE_CSRMM_ALG1",                       "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, ROC_UNSUPPORTED | CUDA_DEPRECATED | CUDA_REMOVED};
  m["CUSPARSE_SPMM_CSR_ALG1"]                                         = {"HIPSPARSE_SPMM_CSR_ALG1",                    "rocsparse_spmm_alg_csr",                             CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_SPMM_COO_ALG4"]                                         = {"HIPSPARSE_SPMM_COO_ALG4",                    "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, ROC_UNSUPPORTED};
  m["CUSPARSE_SPMM_CSR_ALG2"]                                         = {"HIPSPARSE_SPMM_CSR_ALG2",                    "rocsparse_spmm_alg_csr_row_split",                   CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_SPMM_CSR_ALG3"]                                         = {"HIPSPARSE_SPMM_CSR_ALG3",                    "rocsparse_spmm_alg_csr_merge",                       CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_SPMM_BLOCKED_ELL_ALG1"]                                 = {"HIPSPARSE_SPMM_BLOCKED_ELL_ALG1",            "rocsparse_spmm_alg_bell",                            CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_SPMM_BSR_ALG1"]                                         = {"HIPSPARSE_SPMM_BSR_ALG1",                    "rocsparse_spmm_alg_bell",                            CONV_NUMERIC_LITERAL, API_SPARSE, 4, UNSUPPORTED};
  m["CUSPARSE_SPMMA_PREPROCESS"]                                      = {"HIPSPARSE_SPMMA_PREPROCESS",                 "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, CUDA_REMOVED | UNSUPPORTED};
  m["CUSPARSE_SPMMA_ALG1"]                                            = {"HIPSPARSE_SPMMA_ALG1",                       "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, CUDA_REMOVED | UNSUPPORTED};
  m["CUSPARSE_SPMMA_ALG2"]                                            = {"HIPSPARSE_SPMMA_ALG2",                       "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, CUDA_REMOVED | UNSUPPORTED};
  m["CUSPARSE_SPMMA_ALG3"]                                            = {"HIPSPARSE_SPMMA_ALG3",                       "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, CUDA_REMOVED | UNSUPPORTED};
  m["CUSPARSE_SPMMA_ALG4"]                                            = {"HIPSPARSE_SPMMA_ALG4",                       "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, CUDA_REMOVED | UNSUPPORTED};

  m["cusparseIndexType_t"]                                            = {"hipsparseIndexType_t",                       "rocsparse_indextype",                                CONV_TYPE, API_SPARSE, 4};
  m["CUSPARSE_INDEX_16U"]                                             = {"HIPSPARSE_INDEX_16U",                        "rocsparse_indextype_u16",                            CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_INDEX_32I"]                                             = {"HIPSPARSE_INDEX_32I",                        "rocsparse_indextype_i32",                            CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_INDEX_64I"]                                             = {"HIPSPARSE_INDEX_64I",                        "rocsparse_indextype_i64",                            CONV_NUMERIC_LITERAL, API_SPARSE, 4};

  m["cusparseSpGEMMAlg_t"]                                            = {"hipsparseSpGEMMAlg_t",                       "rocsparse_spgemm_alg",                               CONV_TYPE, API_SPARSE, 4};
  m["CUSPARSE_SPGEMM_DEFAULT"]                                        = {"HIPSPARSE_SPGEMM_DEFAULT",                   "rocsparse_spgemm_alg_default",                       CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_SPGEMM_CSR_ALG_DETERMINITIC"]                           = {"HIPSPARSE_SPGEMM_CSR_ALG_DETERMINISTIC",     "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, ROC_UNSUPPORTED};
  m["CUSPARSE_SPGEMM_CSR_ALG_NONDETERMINITIC"]                        = {"HIPSPARSE_SPGEMM_CSR_ALG_NONDETERMINISTIC",  "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, ROC_UNSUPPORTED};
  m["CUSPARSE_SPGEMM_ALG1"]                                           = {"HIPSPARSE_SPGEMM_ALG1",                      "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, ROC_UNSUPPORTED};
  m["CUSPARSE_SPGEMM_ALG2"]                                           = {"HIPSPARSE_SPGEMM_ALG2",                      "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, ROC_UNSUPPORTED};
  m["CUSPARSE_SPGEMM_ALG3"]                                           = {"HIPSPARSE_SPGEMM_ALG3",                      "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, ROC_UNSUPPORTED};

  m["cusparseDenseToSparseAlg_t"]                                     = {"hipsparseDenseToSparseAlg_t",                "rocsparse_dense_to_sparse_alg",                      CONV_TYPE, API_SPARSE, 4};
  m["CUSPARSE_DENSETOSPARSE_ALG_DEFAULT"]                             = {"HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT",        "rocsparse_dense_to_sparse_alg_default",              CONV_NUMERIC_LITERAL, API_SPARSE, 4};

  m["cusparseSparseToDenseAlg_t"]                                     = {"hipsparseSparseToDenseAlg_t",                "rocsparse_sparse_to_dense_alg",                      CONV_TYPE, API_SPARSE, 4};
  m["CUSPARSE_SPARSETODENSE_ALG_DEFAULT"]                             = {"HIPSPARSE_SPARSETODENSE_ALG_DEFAULT",        "rocsparse_sparse_to_dense_alg_default",              CONV_NUMERIC_LITERAL, API_SPARSE, 4};

  m["cusparseSpMatAttribute_t"]                                       = {"hipsparseSpMatAttribute_t",                  "rocsparse_spmat_attribute",                          CONV_TYPE, API_SPARSE, 4};
  m["CUSPARSE_SPMAT_FILL_MODE"]                                       = {"HIPSPARSE_SPMAT_FILL_MODE",                  "rocsparse_spmat_fill_mode",                          CONV_NUMERIC_LITERAL, API_SPARSE, 4};
  m["CUSPARSE_SPMAT_DIAG_TYPE"]                                       = {"HIPSPARSE_SPMAT_DIAG_TYPE",                  "rocsparse_spmat_diag_type",                          CONV_NUMERIC_LITERAL, API_SPARSE, 4};

  m["cusparseSpSVAlg_t"]                                              = {"hipsparseSpSVAlg_t",                         "rocsparse_spsv_alg",                                 CONV_TYPE, API_SPARSE, 4};
  m["CUSPARSE_SPSV_ALG_DEFAULT"]                                      = {"HIPSPARSE_SPSV_ALG_DEFAULT",                 "rocsparse_spsv_alg_default",                         CONV_NUMERIC_LITERAL, API_SPARSE, 4};

  m["cusparseSpSMAlg_t"]                                              = {"hipsparseSpSMAlg_t",                         "rocsparse_spsm_alg",                                 CONV_TYPE, API_SPARSE, 4};
  m["CUSPARSE_SPSM_ALG_DEFAULT"]                                      = {"HIPSPARSE_SPSM_ALG_DEFAULT",                 "rocsparse_spsm_alg_default",                         CONV_NUMERIC_LITERAL, API_SPARSE, 4};

  m["cusparseSpSMUpdate_t"]                                           = {"hipsparseSpSMUpdate_t",                      "",                                                   CONV_TYPE, API_SPARSE, 4, UNSUPPORTED};
  m["CUSPARSE_SPSM_UPDATE_GENERAL"]                                   = {"HIPSPARSE_SPSM_UPDATE_GENERAL",              "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, UNSUPPORTED};
  m["CUSPARSE_SPSM_UPDATE_DIAGONAL"]                                  = {"HIPSPARSE_SPSM_UPDATE_DIAGONAL",             "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, UNSUPPORTED};

  m["cusparseSDDMMAlg_t"]                                             = {"hipsparseSDDMMAlg_t",                        "rocsparse_sddmm_alg",                                CONV_TYPE, API_SPARSE, 4};
  m["CUSPARSE_SDDMM_ALG_DEFAULT"]                                     = {"HIPSPARSE_SDDMM_ALG_DEFAULT",                "rocsparse_sddmm_alg_default",                        CONV_NUMERIC_LITERAL, API_SPARSE, 4};

  m["cusparseSideMode_t"]                                             = {"hipsparseSideMode_t",                        "",                                                   CONV_TYPE, API_SPARSE, 4, UNSUPPORTED | CUDA_REMOVED};
  m["CUSPARSE_SIDE_LEFT"]                                             = {"HIPSPARSE_SIDE_LEFT",                        "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, UNSUPPORTED | CUDA_REMOVED};
  m["CUSPARSE_SIDE_RIGHT"]                                            = {"HIPSPARSE_SIDE_RIGHT",                       "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, UNSUPPORTED | CUDA_REMOVED};

  m["cusparseSpMMOpAlg_t"]                                            = {"hipsparseSpMMOpAlg_t",                       "",                                                   CONV_TYPE, API_SPARSE, 4, UNSUPPORTED};
  m["CUSPARSE_SPMM_OP_ALG_DEFAULT"]                                   = {"HIPSPARSE_SPMM_OP_ALG_DEFAULT",              "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, UNSUPPORTED};

  m["cusparseSpSVUpdate_t"]                                           = {"hipsparseSpSVUpdate_t",                      "",                                                   CONV_TYPE, API_SPARSE, 4, UNSUPPORTED};
  m["CUSPARSE_SPSV_UPDATE_GENERAL"]                                   = {"HIPSPARSE_SPSV_UPDATE_GENERAL",              "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, UNSUPPORTED};
  m["CUSPARSE_SPSV_UPDATE_DIAGONAL"]                                  = {"HIPSPARSE_SPSV_UPDATE_DIAGONAL",             "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, UNSUPPORTED};

  m["cusparseColorAlg_t"]                                             = {"hipsparseColorAlg_t",                        "",                                                   CONV_TYPE, API_SPARSE, 4, UNSUPPORTED | CUDA_DEPRECATED};
  m["CUSPARSE_COLOR_ALG0"]                                            = {"HIPSPARSE_COLOR_ALG0",                       "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, UNSUPPORTED | CUDA_DEPRECATED};
  m["CUSPARSE_COLOR_ALG1"]                                            = {"HIPSPARSE_COLOR_ALG1",                       "",                                                   CONV_NUMERIC_LITERAL, API_SPARSE, 4, UNSUPPORTED | CUDA_DEPRECATED};

  // 3. Defines

  // 4. Typedefs
  m["cusparseLoggerCallback_t"]                                       = {"hipsparseLoggerCallback_t",                  "",                                                   CONV_TYPE, API_SPARSE, 4, UNSUPPORTED};
  m["cusparseConstSpVecDescr_t"]                                      = {"hipsparseConstSpVecDescr_t",                 "rocsparse_const_spvec_descr",                        CONV_TYPE, API_SPARSE, 4};
  m["cusparseConstDnVecDescr_t"]                                      = {"hipsparseConstDnVecDescr_t",                 "rocsparse_const_dnvec_descr",                        CONV_TYPE, API_SPARSE, 4};
  m["cusparseConstSpMatDescr_t"]                                      = {"hipsparseConstSpMatDescr_t",                 "rocsparse_const_spmat_descr",                        CONV_TYPE, API_SPARSE, 4};
  m["cusparseConstDnMatDescr_t"]                                      = {"hipsparseConstDnMatDescr_t",                 "rocsparse_const_dnmat_descr",                        CONV_TYPE, API_SPARSE, 4};

  return m;
}();

const std::map<llvm::StringRef, cudaAPIversions> CUDA_SPARSE_TYPE_NAME_VER_MAP = [] {
  std::map<llvm::StringRef, cudaAPIversions> m;

  m["cusparseHybMat"]                                                 = {CUDA_0,   CUDA_102, CUDA_110};
  m["cusparseHybMat_t"]                                               = {CUDA_0,   CUDA_102, CUDA_110};
  m["cusparseSolveAnalysisInfo"]                                      = {CUDA_0,   CUDA_102, CUDA_110};
  m["cusparseSolveAnalysisInfo_t"]                                    = {CUDA_0,   CUDA_102, CUDA_110};
  m["csrsm2Info"]                                                     = {CUDA_92,  CUDA_0,   CUDA_120};
  m["csrsm2Info_t"]                                                   = {CUDA_92,  CUDA_0,   CUDA_120};
  m["pruneInfo"]                                                      = {CUDA_90,  CUDA_122, CUDA_0  }; // CUSPARSE_VERSION 12120
  m["pruneInfo_t"]                                                    = {CUDA_90,  CUDA_122, CUDA_0  }; // CUSPARSE_VERSION 12120
  m["cusparseSpMatDescr"]                                             = {CUDA_101, CUDA_0,   CUDA_0  };
  m["cusparseSpMatDescr_t"]                                           = {CUDA_101, CUDA_0,   CUDA_0  };
  m["cusparseDnMatDescr"]                                             = {CUDA_101, CUDA_0,   CUDA_0  };
  m["cusparseDnMatDescr_t"]                                           = {CUDA_101, CUDA_0,   CUDA_0  };
  m["cusparseSpVecDescr"]                                             = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cusparseSpVecDescr_t"]                                           = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cusparseDnVecDescr"]                                             = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cusparseDnVecDescr_t"]                                           = {CUDA_102, CUDA_0,   CUDA_0  };
  m["cusparseHybPartition_t"]                                         = {CUDA_0,   CUDA_102, CUDA_110};
  m["CUSPARSE_HYB_PARTITION_AUTO"]                                    = {CUDA_0,   CUDA_102, CUDA_110};
  m["CUSPARSE_HYB_PARTITION_USER"]                                    = {CUDA_0,   CUDA_102, CUDA_110};
  m["CUSPARSE_HYB_PARTITION_MAX"]                                     = {CUDA_0,   CUDA_102, CUDA_110};
  m["cusparseAlgMode_t"]                                              = {CUDA_80,  CUDA_0,   CUDA_120};
  m["CUSPARSE_ALG0"]                                                  = {CUDA_80,  CUDA_0,   CUDA_110};
  m["CUSPARSE_ALG1"]                                                  = {CUDA_80,  CUDA_0,   CUDA_110};
  m["CUSPARSE_ALG_NAIVE"]                                             = {CUDA_92,  CUDA_0,   CUDA_110};
  m["CUSPARSE_ALG_MERGE_PATH"]                                        = {CUDA_92,  CUDA_0,   CUDA_120};
  m["CUSPARSE_STATUS_NOT_SUPPORTED"]                                  = {CUDA_102, CUDA_0,   CUDA_0  };
  m["CUSPARSE_STATUS_INSUFFICIENT_RESOURCES"]                         = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cusparseCsr2CscAlg_t"]                                           = {CUDA_101, CUDA_0,   CUDA_0  };
  m["CUSPARSE_CSR2CSC_ALG1"]                                          = {CUDA_101, CUDA_0,   CUDA_0  };
  m["CUSPARSE_CSR2CSC_ALG2"]                                          = {CUDA_101, CUDA_0,   CUDA_120};
  m["CUSPARSE_CSR2CSC_ALG_DEFAULT"]                                   = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cusparseFormat_t"]                                               = {CUDA_101, CUDA_0,   CUDA_0  };
  m["CUSPARSE_FORMAT_CSR"]                                            = {CUDA_101, CUDA_0,   CUDA_0  };
  m["CUSPARSE_FORMAT_CSC"]                                            = {CUDA_101, CUDA_0,   CUDA_0  };
  m["CUSPARSE_FORMAT_COO"]                                            = {CUDA_101, CUDA_0,   CUDA_0  };
  m["CUSPARSE_FORMAT_COO_AOS"]                                        = {CUDA_102, CUDA_0,   CUDA_120};
  m["cusparseOrder_t"]                                                = {CUDA_101, CUDA_0,   CUDA_0  };
  m["CUSPARSE_ORDER_COL"]                                             = {CUDA_101, CUDA_0,   CUDA_0  };
  m["CUSPARSE_ORDER_ROW"]                                             = {CUDA_101, CUDA_0,   CUDA_0  };
  m["cusparseSpMVAlg_t"]                                              = {CUDA_102, CUDA_0,   CUDA_0  };
  m["CUSPARSE_MV_ALG_DEFAULT"]                                        = {CUDA_102, CUDA_113, CUDA_120};
  m["CUSPARSE_SPMV_ALG_DEFAULT"]                                      = {CUDA_112, CUDA_0,   CUDA_0  };
  m["CUSPARSE_COOMV_ALG"]                                             = {CUDA_102, CUDA_112, CUDA_120};
  m["CUSPARSE_CSRMV_ALG1"]                                            = {CUDA_102, CUDA_112, CUDA_120};
  m["CUSPARSE_CSRMV_ALG2"]                                            = {CUDA_102, CUDA_112, CUDA_120};
  m["cusparseSpMMAlg_t"]                                              = {CUDA_101, CUDA_0,   CUDA_0  };
  m["CUSPARSE_MM_ALG_DEFAULT"]                                        = {CUDA_102, CUDA_110, CUDA_120};
  m["CUSPARSE_COOMM_ALG1"]                                            = {CUDA_101, CUDA_110, CUDA_120};
  m["CUSPARSE_COOMM_ALG2"]                                            = {CUDA_101, CUDA_110, CUDA_120};
  m["CUSPARSE_COOMM_ALG3"]                                            = {CUDA_101, CUDA_110, CUDA_120};
  m["CUSPARSE_CSRMM_ALG1"]                                            = {CUDA_102, CUDA_110, CUDA_120};
  m["CUSPARSE_SPMM_ALG_DEFAULT"]                                      = {CUDA_110, CUDA_0,   CUDA_0  };
  m["CUSPARSE_SPMM_COO_ALG1"]                                         = {CUDA_110, CUDA_0,   CUDA_0  };
  m["CUSPARSE_SPMM_COO_ALG2"]                                         = {CUDA_110, CUDA_0,   CUDA_0  };
  m["CUSPARSE_SPMM_COO_ALG3"]                                         = {CUDA_110, CUDA_0,   CUDA_0  };
  m["CUSPARSE_SPMM_COO_ALG4"]                                         = {CUDA_110, CUDA_0,   CUDA_0  };
  m["CUSPARSE_SPMM_CSR_ALG1"]                                         = {CUDA_110, CUDA_0,   CUDA_0  };
  m["CUSPARSE_SPMM_CSR_ALG2"]                                         = {CUDA_110, CUDA_0,   CUDA_0  };
  m["CUSPARSE_SPMM_CSR_ALG3"]                                         = {CUDA_112, CUDA_0,   CUDA_0  };
  m["CUSPARSE_SPMM_BLOCKED_ELL_ALG1"]                                 = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cusparseIndexType_t"]                                            = {CUDA_101, CUDA_0,   CUDA_0  };
  m["CUSPARSE_INDEX_16U"]                                             = {CUDA_101, CUDA_0,   CUDA_0  };
  m["CUSPARSE_INDEX_32I"]                                             = {CUDA_101, CUDA_0,   CUDA_0  };
  m["CUSPARSE_INDEX_64I"]                                             = {CUDA_101, CUDA_0,   CUDA_0  }; // CUSPARSE_VERSION 10200
  m["cusparseSpGEMMAlg_t"]                                            = {CUDA_110, CUDA_0,   CUDA_0  };
  m["CUSPARSE_SPGEMM_DEFAULT"]                                        = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cusparseSpGEMMDescr"]                                            = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cusparseSpGEMMDescr_t"]                                          = {CUDA_110, CUDA_0,   CUDA_0  };
  m["cusparseDenseToSparseAlg_t"]                                     = {CUDA_111, CUDA_0,   CUDA_0  };
  m["CUSPARSE_DENSETOSPARSE_ALG_DEFAULT"]                             = {CUDA_111, CUDA_0,   CUDA_0  };
  m["cusparseSparseToDenseAlg_t"]                                     = {CUDA_111, CUDA_0,   CUDA_0  };
  m["CUSPARSE_SPARSETODENSE_ALG_DEFAULT"]                             = {CUDA_111, CUDA_0,   CUDA_0  };
  m["CUSPARSE_SPMMA_PREPROCESS"]                                      = {CUDA_111, CUDA_0,   CUDA_112};
  m["CUSPARSE_SPMMA_ALG1"]                                            = {CUDA_111, CUDA_0,   CUDA_112};
  m["CUSPARSE_SPMMA_ALG2"]                                            = {CUDA_111, CUDA_0,   CUDA_112};
  m["CUSPARSE_SPMMA_ALG3"]                                            = {CUDA_111, CUDA_0,   CUDA_112};
  m["CUSPARSE_SPMMA_ALG4"]                                            = {CUDA_111, CUDA_0,   CUDA_112};
  m["cusparseSpMatAttribute_t"]                                       = {CUDA_113, CUDA_0,   CUDA_0  };
  m["CUSPARSE_SPMAT_FILL_MODE"]                                       = {CUDA_113, CUDA_0,   CUDA_0  };
  m["CUSPARSE_SPMAT_DIAG_TYPE"]                                       = {CUDA_113, CUDA_0,   CUDA_0  };
  m["CUSPARSE_SPMV_COO_ALG1"]                                         = {CUDA_112, CUDA_0,   CUDA_0  };
  m["CUSPARSE_SPMV_COO_ALG2"]                                         = {CUDA_112, CUDA_0,   CUDA_0  };
  m["CUSPARSE_SPMV_CSR_ALG1"]                                         = {CUDA_112, CUDA_0,   CUDA_0  };
  m["CUSPARSE_SPMV_CSR_ALG2"]                                         = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cusparseSpSVAlg_t"]                                              = {CUDA_113, CUDA_0,   CUDA_0  };
  m["CUSPARSE_SPSV_ALG_DEFAULT"]                                      = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cusparseSpSMAlg_t"]                                              = {CUDA_113, CUDA_0,   CUDA_0  };
  m["CUSPARSE_SPSM_ALG_DEFAULT"]                                      = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cusparseSpSMDescr"]                                              = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cusparseSpSMDescr_t"]                                            = {CUDA_113, CUDA_0,   CUDA_0  };
  m["CUSPARSE_SPGEMM_CSR_ALG_DETERMINITIC"]                           = {CUDA_113, CUDA_0,   CUDA_0  };
  m["CUSPARSE_SPGEMM_CSR_ALG_NONDETERMINITIC"]                        = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cusparseSDDMMAlg_t"]                                             = {CUDA_112, CUDA_0,   CUDA_0  };
  m["CUSPARSE_SDDMM_ALG_DEFAULT"]                                     = {CUDA_112, CUDA_0,   CUDA_0  };
  m["CUSPARSE_FORMAT_BLOCKED_ELL"]                                    = {CUDA_112, CUDA_0,   CUDA_0  };
  m["cusparseSpSVDescr"]                                              = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cusparseSpSVDescr_t"]                                            = {CUDA_113, CUDA_0,   CUDA_0  };
  m["cusparseSideMode_t"]                                             = {CUDA_0,   CUDA_0,   CUDA_115};
  m["CUSPARSE_SIDE_LEFT"]                                             = {CUDA_0,   CUDA_0,   CUDA_115};
  m["CUSPARSE_SIDE_RIGHT"]                                            = {CUDA_0,   CUDA_0,   CUDA_115};
  m["cusparseLoggerCallback_t"]                                       = {CUDA_115, CUDA_0,   CUDA_0  };
  m["cusparseSpMMOpPlan"]                                             = {CUDA_115, CUDA_0,   CUDA_0  };
  m["cusparseSpMMOpPlan_t"]                                           = {CUDA_115, CUDA_0,   CUDA_0  };
  m["cusparseSpMMOpAlg_t"]                                            = {CUDA_115, CUDA_0,   CUDA_0  };
  m["CUSPARSE_SPMM_OP_ALG_DEFAULT"]                                   = {CUDA_115, CUDA_0,   CUDA_0  };
  m["csrsv2Info"]                                                     = {CUDA_0,   CUDA_0,   CUDA_120};
  m["csrsv2Info_t"]                                                   = {CUDA_0,   CUDA_0,   CUDA_120};
  m["csrgemm2Info"]                                                   = {CUDA_0,   CUDA_0,   CUDA_120};
  m["csrgemm2Info_t"]                                                 = {CUDA_0,   CUDA_0,   CUDA_120};
  m["cusparseConstSpVecDescr_t"]                                      = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cusparseConstDnVecDescr_t"]                                      = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cusparseConstSpMatDescr_t"]                                      = {CUDA_120, CUDA_0,   CUDA_0  };
  m["cusparseConstDnMatDescr_t"]                                      = {CUDA_120, CUDA_0,   CUDA_0  };
  m["CUSPARSE_SPGEMM_ALG1"]                                           = {CUDA_120, CUDA_0,   CUDA_0  };
  m["CUSPARSE_SPGEMM_ALG2"]                                           = {CUDA_120, CUDA_0,   CUDA_0  };
  m["CUSPARSE_SPGEMM_ALG3"]                                           = {CUDA_120, CUDA_0,   CUDA_0  };
  m["CUSPARSE_FORMAT_BSR"]                                            = {CUDA_121, CUDA_0,   CUDA_0  }; // CUSPARSE_VERSION 12100
  m["CUSPARSE_FORMAT_SLICED_ELLPACK"]                                 = {CUDA_121, CUDA_0,   CUDA_0  }; // CUSPARSE_VERSION 12100
  m["CUSPARSE_SPMV_SELL_ALG1"]                                        = {CUDA_121, CUDA_0,   CUDA_0  }; // CUSPARSE_VERSION 12100
  m["cusparseSpSVUpdate_t"]                                           = {CUDA_121, CUDA_0,   CUDA_0  }; // CUSPARSE_VERSION 12100
  m["CUSPARSE_SPSV_UPDATE_GENERAL"]                                   = {CUDA_121, CUDA_0,   CUDA_0  }; // CUSPARSE_VERSION 12100
  m["CUSPARSE_SPSV_UPDATE_DIAGONAL"]                                  = {CUDA_121, CUDA_0,   CUDA_0  }; // CUSPARSE_VERSION 12100
  m["bsrsv2Info"]                                                     = {CUDA_0,   CUDA_122, CUDA_0  }; // CUSPARSE_VERSION 12120
  m["bsrsv2Info_t"]                                                   = {CUDA_0,   CUDA_122, CUDA_0  }; // CUSPARSE_VERSION 12120
  m["bsrsm2Info"]                                                     = {CUDA_0,   CUDA_122, CUDA_0  }; // CUSPARSE_VERSION 12120
  m["bsrsm2Info_t"]                                                   = {CUDA_0,   CUDA_122, CUDA_0  }; // CUSPARSE_VERSION 12120
  m["csric02Info"]                                                    = {CUDA_0,   CUDA_122, CUDA_0  }; // CUSPARSE_VERSION 12120
  m["csric02Info_t"]                                                  = {CUDA_0,   CUDA_122, CUDA_0  }; // CUSPARSE_VERSION 12120
  m["csrilu02Info"]                                                   = {CUDA_0,   CUDA_122, CUDA_0  }; // CUSPARSE_VERSION 12120
  m["csrilu02Info_t"]                                                 = {CUDA_0,   CUDA_122, CUDA_0  }; // CUSPARSE_VERSION 12120
  m["bsrilu02Info"]                                                   = {CUDA_0,   CUDA_122, CUDA_0  }; // CUSPARSE_VERSION 12120
  m["bsrilu02Info_t"]                                                 = {CUDA_0,   CUDA_122, CUDA_0  }; // CUSPARSE_VERSION 12120
  m["csru2csrInfo"]                                                   = {CUDA_0,   CUDA_122, CUDA_0  }; // CUSPARSE_VERSION 12120
  m["csru2csrInfo_t"]                                                 = {CUDA_0,   CUDA_122, CUDA_0  }; // CUSPARSE_VERSION 12120
  m["cusparseColorInfo"]                                              = {CUDA_0,   CUDA_122, CUDA_0  }; // CUSPARSE_VERSION 12120
  m["cusparseColorInfo_t"]                                            = {CUDA_0,   CUDA_122, CUDA_0  }; // CUSPARSE_VERSION 12120
  m["cusparseSolvePolicy_t"]                                          = {CUDA_0,   CUDA_122, CUDA_0  }; // CUSPARSE_VERSION 12120
  m["CUSPARSE_SOLVE_POLICY_NO_LEVEL"]                                 = {CUDA_0,   CUDA_122, CUDA_0  }; // CUSPARSE_VERSION 12120
  m["CUSPARSE_SOLVE_POLICY_USE_LEVEL"]                                = {CUDA_0,   CUDA_122, CUDA_0  }; // CUSPARSE_VERSION 12120
  m["cusparseColorAlg_t"]                                             = {CUDA_80,  CUDA_122, CUDA_0  }; // CUSPARSE_VERSION 12120
  m["CUSPARSE_COLOR_ALG0"]                                            = {CUDA_80,  CUDA_122, CUDA_0  }; // CUSPARSE_VERSION 12120
  m["CUSPARSE_COLOR_ALG1"]                                            = {CUDA_80,  CUDA_122, CUDA_0  }; // CUSPARSE_VERSION 12120
  m["cusparseSpSMUpdate_t"]                                           = {CUDA_124, CUDA_0,   CUDA_0  };
  m["CUSPARSE_SPSM_UPDATE_GENERAL"]                                   = {CUDA_124, CUDA_0,   CUDA_0  };
  m["CUSPARSE_SPSM_UPDATE_DIAGONAL"]                                  = {CUDA_124, CUDA_0,   CUDA_0  };
  m["CUSPARSE_SPMM_BSR_ALG1"]                                         = {CUDA_125, CUDA_0,   CUDA_0  }; // CUSPARSE_VERSION 12501
  m["CUSPARSE_SPMV_BSR_ALG1"]                                         = {CUDA_130, CUDA_0,   CUDA_0  }; // CUSPARSE_VERSION 12603 CUDA 13.0.1

  return m;
}();

const std::map<llvm::StringRef, hipAPIversions> HIP_SPARSE_TYPE_NAME_VER_MAP = [] {
  std::map<llvm::StringRef, hipAPIversions> m;

  m["hipsparseHandle_t"]                                              = {HIP_1092, HIP_0,    HIP_0   };
  m["hipsparseHybMat_t"]                                              = {HIP_1092, HIP_0,    HIP_0   };
  m["hipsparseMatDescr_t"]                                            = {HIP_1092, HIP_0,    HIP_0   };
  m["csrsv2Info_t"]                                                   = {HIP_1092, HIP_0,    HIP_0   };
  m["csrsm2Info_t"]                                                   = {HIP_3010, HIP_0,    HIP_0   };
  m["bsrsv2Info"]                                                     = {HIP_3060, HIP_0,    HIP_0   };
  m["bsrsv2Info_t"]                                                   = {HIP_3060, HIP_0,    HIP_0   };
  m["bsric02Info"]                                                    = {HIP_3080, HIP_0,    HIP_0   };
  m["bsric02Info_t"]                                                  = {HIP_3080, HIP_0,    HIP_0   };
  m["csrilu02Info"]                                                   = {HIP_1092, HIP_0,    HIP_0   };
  m["csrilu02Info_t"]                                                 = {HIP_1092, HIP_0,    HIP_0   };
  m["bsrilu02Info"]                                                   = {HIP_3090, HIP_0,    HIP_0   };
  m["bsrilu02Info_t"]                                                 = {HIP_3090, HIP_0,    HIP_0   };
  m["csrgemm2Info"]                                                   = {HIP_2080, HIP_0,    HIP_0   };
  m["csrgemm2Info_t"]                                                 = {HIP_2080, HIP_0,    HIP_0   };
  m["pruneInfo"]                                                      = {HIP_3090, HIP_0,    HIP_0   };
  m["pruneInfo_t"]                                                    = {HIP_3090, HIP_0,    HIP_0   };
  m["hipsparseAction_t"]                                              = {HIP_1092, HIP_0,    HIP_0   };
  m["HIPSPARSE_ACTION_SYMBOLIC"]                                      = {HIP_1092, HIP_0,    HIP_0   };
  m["HIPSPARSE_ACTION_NUMERIC"]                                       = {HIP_1092, HIP_0,    HIP_0   };
  m["hipsparseDirection_t"]                                           = {HIP_3020, HIP_0,    HIP_0   };
  m["HIPSPARSE_DIRECTION_ROW"]                                        = {HIP_3020, HIP_0,    HIP_0   };
  m["HIPSPARSE_DIRECTION_COLUMN"]                                     = {HIP_3020, HIP_0,    HIP_0   };
  m["hipsparseHybPartition_t"]                                        = {HIP_1092, HIP_0,    HIP_0   };
  m["HIPSPARSE_HYB_PARTITION_AUTO"]                                   = {HIP_1092, HIP_0,    HIP_0   };
  m["HIPSPARSE_HYB_PARTITION_USER"]                                   = {HIP_1092, HIP_0,    HIP_0   };
  m["HIPSPARSE_HYB_PARTITION_MAX"]                                    = {HIP_1092, HIP_0,    HIP_0   };
  m["hipsparseDiagType_t"]                                            = {HIP_1092, HIP_0,    HIP_0   };
  m["HIPSPARSE_DIAG_TYPE_NON_UNIT"]                                   = {HIP_1092, HIP_0,    HIP_0   };
  m["HIPSPARSE_DIAG_TYPE_UNIT"]                                       = {HIP_1092, HIP_0,    HIP_0   };
  m["hipsparseFillMode_t"]                                            = {HIP_1092, HIP_0,    HIP_0   };
  m["HIPSPARSE_FILL_MODE_LOWER"]                                      = {HIP_1092, HIP_0,    HIP_0   };
  m["HIPSPARSE_FILL_MODE_UPPER"]                                      = {HIP_1092, HIP_0,    HIP_0   };
  m["hipsparseIndexBase_t"]                                           = {HIP_1092, HIP_0,    HIP_0   };
  m["HIPSPARSE_INDEX_BASE_ZERO"]                                      = {HIP_1092, HIP_0,    HIP_0   };
  m["HIPSPARSE_INDEX_BASE_ONE"]                                       = {HIP_1092, HIP_0,    HIP_0   };
  m["hipsparseMatrixType_t"]                                          = {HIP_1092, HIP_0,    HIP_0   };
  m["HIPSPARSE_MATRIX_TYPE_GENERAL"]                                  = {HIP_1092, HIP_0,    HIP_0   };
  m["HIPSPARSE_MATRIX_TYPE_SYMMETRIC"]                                = {HIP_1092, HIP_0,    HIP_0   };
  m["HIPSPARSE_MATRIX_TYPE_HERMITIAN"]                                = {HIP_1092, HIP_0,    HIP_0   };
  m["HIPSPARSE_MATRIX_TYPE_TRIANGULAR"]                               = {HIP_1092, HIP_0,    HIP_0   };
  m["hipsparseOperation_t"]                                           = {HIP_1092, HIP_0,    HIP_0   };
  m["HIPSPARSE_OPERATION_NON_TRANSPOSE"]                              = {HIP_1092, HIP_0,    HIP_0   };
  m["HIPSPARSE_OPERATION_TRANSPOSE"]                                  = {HIP_1092, HIP_0,    HIP_0   };
  m["HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE"]                        = {HIP_1092, HIP_0,    HIP_0   };
  m["hipsparsePointerMode_t"]                                         = {HIP_1092, HIP_0,    HIP_0   };
  m["HIPSPARSE_POINTER_MODE_HOST"]                                    = {HIP_1092, HIP_0,    HIP_0   };
  m["HIPSPARSE_POINTER_MODE_DEVICE"]                                  = {HIP_1092, HIP_0,    HIP_0   };
  m["hipsparseSolvePolicy_t"]                                         = {HIP_1092, HIP_0,    HIP_0   };
  m["HIPSPARSE_SOLVE_POLICY_NO_LEVEL"]                                = {HIP_1092, HIP_0,    HIP_0   };
  m["HIPSPARSE_SOLVE_POLICY_USE_LEVEL"]                               = {HIP_1092, HIP_0,    HIP_0   };
  m["hipsparseStatus_t"]                                              = {HIP_1092, HIP_0,    HIP_0   };
  m["HIPSPARSE_STATUS_SUCCESS"]                                       = {HIP_1092, HIP_0,    HIP_0   };
  m["HIPSPARSE_STATUS_NOT_INITIALIZED"]                               = {HIP_1092, HIP_0,    HIP_0   };
  m["HIPSPARSE_STATUS_ALLOC_FAILED"]                                  = {HIP_1092, HIP_0,    HIP_0   };
  m["HIPSPARSE_STATUS_INVALID_VALUE"]                                 = {HIP_1092, HIP_0,    HIP_0   };
  m["HIPSPARSE_STATUS_ARCH_MISMATCH"]                                 = {HIP_1092, HIP_0,    HIP_0   };
  m["HIPSPARSE_STATUS_MAPPING_ERROR"]                                 = {HIP_1092, HIP_0,    HIP_0   };
  m["HIPSPARSE_STATUS_EXECUTION_FAILED"]                              = {HIP_1092, HIP_0,    HIP_0   };
  m["HIPSPARSE_STATUS_INTERNAL_ERROR"]                                = {HIP_1092, HIP_0,    HIP_0   };
  m["HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED"]                     = {HIP_1092, HIP_0,    HIP_0   };
  m["HIPSPARSE_STATUS_ZERO_PIVOT"]                                    = {HIP_1092, HIP_0,    HIP_0   };
  m["hipsparseSpMatDescr_t"]                                          = {HIP_4010, HIP_0,    HIP_0   };
  m["hipsparseSpVecDescr_t"]                                          = {HIP_4010, HIP_0,    HIP_0   };
  m["hipsparseDnVecDescr_t"]                                          = {HIP_4010, HIP_0,    HIP_0   };
  m["hipsparseSpGEMMDescr"]                                           = {HIP_4010, HIP_0,    HIP_0   };
  m["hipsparseSpGEMMDescr_t"]                                         = {HIP_4010, HIP_0,    HIP_0   };
  m["HIPSPARSE_STATUS_NOT_SUPPORTED"]                                 = {HIP_4010, HIP_0,    HIP_0   };
  m["HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES"]                        = {HIP_4010, HIP_0,    HIP_0   };
  m["hipsparseFormat_t"]                                              = {HIP_4010, HIP_0,    HIP_0   };
  m["HIPSPARSE_FORMAT_CSR"]                                           = {HIP_4010, HIP_0,    HIP_0   };
  m["HIPSPARSE_FORMAT_CSC"]                                           = {HIP_4010, HIP_0,    HIP_0   };
  m["HIPSPARSE_FORMAT_COO"]                                           = {HIP_4010, HIP_0,    HIP_0   };
  m["HIPSPARSE_FORMAT_COO_AOS"]                                       = {HIP_4010, HIP_0,    HIP_0   };
  m["hipsparseSpMVAlg_t"]                                             = {HIP_4010, HIP_0,    HIP_0   };
  m["HIPSPARSE_MV_ALG_DEFAULT"]                                       = {HIP_4010, HIP_0,    HIP_0   };
  m["HIPSPARSE_COOMV_ALG"]                                            = {HIP_4010, HIP_0,    HIP_0   };
  m["HIPSPARSE_CSRMV_ALG1"]                                           = {HIP_4010, HIP_0,    HIP_0   };
  m["HIPSPARSE_CSRMV_ALG2"]                                           = {HIP_4010, HIP_0,    HIP_0   };
  m["hipsparseIndexType_t"]                                           = {HIP_4010, HIP_0,    HIP_0   };
  m["HIPSPARSE_INDEX_16U"]                                            = {HIP_4010, HIP_0,    HIP_0   };
  m["HIPSPARSE_INDEX_32I"]                                            = {HIP_4010, HIP_0,    HIP_0   };
  m["HIPSPARSE_INDEX_64I"]                                            = {HIP_4010, HIP_0,    HIP_0   };
  m["hipsparseSpGEMMAlg_t"]                                           = {HIP_4010, HIP_0,    HIP_0   };
  m["HIPSPARSE_SPGEMM_DEFAULT"]                                       = {HIP_4010, HIP_0,    HIP_0   };
  m["csru2csrInfo"]                                                   = {HIP_4020, HIP_0,    HIP_0   };
  m["csru2csrInfo_t"]                                                 = {HIP_4020, HIP_0,    HIP_0   };
  m["hipsparseDnMatDescr_t"]                                          = {HIP_4020, HIP_0,    HIP_0   };
  m["hipsparseOrder_t"]                                               = {HIP_4020, HIP_0,    HIP_0   };
  m["HIPSPARSE_ORDER_COL"]                                            = {HIP_5040, HIP_0,    HIP_0   };
  m["HIPSPARSE_ORDER_ROW"]                                            = {HIP_4020, HIP_0,    HIP_0   };
  m["hipsparseSpMMAlg_t"]                                             = {HIP_4020, HIP_0,    HIP_0   };
  m["HIPSPARSE_MM_ALG_DEFAULT"]                                       = {HIP_4020, HIP_0,    HIP_0   };
  m["HIPSPARSE_SPMM_ALG_DEFAULT"]                                     = {HIP_4020, HIP_0,    HIP_0   };
  m["HIPSPARSE_COOMM_ALG1"]                                           = {HIP_4020, HIP_0,    HIP_0   };
  m["HIPSPARSE_SPMM_COO_ALG1"]                                        = {HIP_4020, HIP_0,    HIP_0   };
  m["HIPSPARSE_COOMM_ALG2"]                                           = {HIP_4020, HIP_0,    HIP_0   };
  m["HIPSPARSE_SPMM_COO_ALG2"]                                        = {HIP_4020, HIP_0,    HIP_0   };
  m["HIPSPARSE_COOMM_ALG3"]                                           = {HIP_4020, HIP_0,    HIP_0   };
  m["HIPSPARSE_SPMM_COO_ALG3"]                                        = {HIP_4020, HIP_0,    HIP_0   };
  m["HIPSPARSE_CSRMM_ALG1"]                                           = {HIP_4020, HIP_0,    HIP_0   };
  m["HIPSPARSE_SPMM_CSR_ALG1"]                                        = {HIP_4020, HIP_0,    HIP_0   };
  m["HIPSPARSE_SPMM_COO_ALG4"]                                        = {HIP_4020, HIP_0,    HIP_0   };
  m["HIPSPARSE_SPMM_CSR_ALG2"]                                        = {HIP_4020, HIP_0,    HIP_0   };
  m["hipsparseSparseToDenseAlg_t"]                                    = {HIP_4020, HIP_0,    HIP_0   };
  m["HIPSPARSE_SPARSETODENSE_ALG_DEFAULT"]                            = {HIP_4020, HIP_0,    HIP_0   };
  m["hipsparseSDDMMAlg_t"]                                            = {HIP_4030, HIP_0,    HIP_0   };
  m["HIPSPARSE_SDDMM_ALG_DEFAULT"]                                    = {HIP_4030, HIP_0,    HIP_0   };
  m["hipsparseColorInfo_t"]                                           = {HIP_4050, HIP_0,    HIP_0   };
  m["bsrsm2Info_t"]                                                   = {HIP_4050, HIP_0,    HIP_0   };
  m["bsrsm2Info"]                                                     = {HIP_4050, HIP_0,    HIP_0   };
  m["hipsparseSpSVDescr"]                                             = {HIP_4050, HIP_0,    HIP_0   };
  m["hipsparseSpSVDescr_t"]                                           = {HIP_4050, HIP_0,    HIP_0   };
  m["hipsparseSpSMDescr"]                                             = {HIP_4050, HIP_0,    HIP_0   };
  m["hipsparseSpSMDescr_t"]                                           = {HIP_4050, HIP_0,    HIP_0   };
  m["HIPSPARSE_FORMAT_BLOCKED_ELL"]                                   = {HIP_4050, HIP_0,    HIP_0   };
  m["HIPSPARSE_SPMV_ALG_DEFAULT"]                                     = {HIP_4050, HIP_0,    HIP_0   };
  m["HIPSPARSE_SPMV_COO_ALG1"]                                        = {HIP_4050, HIP_0,    HIP_0   };
  m["HIPSPARSE_SPMV_COO_ALG2"]                                        = {HIP_4050, HIP_0,    HIP_0   };
  m["HIPSPARSE_SPMV_CSR_ALG1"]                                        = {HIP_4050, HIP_0,    HIP_0   };
  m["HIPSPARSE_SPMV_CSR_ALG2"]                                        = {HIP_4050, HIP_0,    HIP_0   };
  m["HIPSPARSE_SPMM_BLOCKED_ELL_ALG1"]                                = {HIP_4050, HIP_0,    HIP_0   };
  m["HIPSPARSE_SPMM_CSR_ALG3"]                                        = {HIP_4050, HIP_0,    HIP_0   };
  m["HIPSPARSE_SPSV_ALG_DEFAULT"]                                     = {HIP_4050, HIP_0,    HIP_0   };
  m["HIPSPARSE_SPSM_ALG_DEFAULT"]                                     = {HIP_4050, HIP_0,    HIP_0   };
  m["hipsparseSpSVAlg_t"]                                             = {HIP_4050, HIP_0,    HIP_0   };
  m["hipsparseSpSMAlg_t"]                                             = {HIP_4050, HIP_0,    HIP_0   };
  m["hipsparseSpMatAttribute_t"]                                      = {HIP_4050, HIP_0,    HIP_0   };
  m["HIPSPARSE_SPMAT_FILL_MODE"]                                      = {HIP_4050, HIP_0,    HIP_0   };
  m["HIPSPARSE_SPMAT_DIAG_TYPE"]                                      = {HIP_4050, HIP_0,    HIP_0   };
  m["hipsparseCsr2CscAlg_t"]                                          = {HIP_5040, HIP_0,    HIP_0   };
  m["HIPSPARSE_CSR2CSC_ALG1"]                                         = {HIP_5040, HIP_0,    HIP_0   };
  m["HIPSPARSE_CSR2CSC_ALG2"]                                         = {HIP_5040, HIP_0,    HIP_0   };
  m["HIPSPARSE_ORDER_COLUMN"]                                         = {HIP_4020, HIP_5040, HIP_0   };
  m["hipsparseDenseToSparseAlg_t"]                                    = {HIP_4020, HIP_0,    HIP_0   };
  m["HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT"]                            = {HIP_4020, HIP_0,    HIP_0   };
  m["HIPSPARSE_CSR2CSC_ALG_DEFAULT"]                                  = {HIP_5060, HIP_0,    HIP_0   };
  m["HIPSPARSE_SPGEMM_CSR_ALG_DETERMINISTIC"]                         = {HIP_5010, HIP_0,    HIP_0   };
  m["HIPSPARSE_SPGEMM_CSR_ALG_NONDETERMINISTIC"]                      = {HIP_5010, HIP_0,    HIP_0   };
  m["HIPSPARSE_SPGEMM_ALG1"]                                          = {HIP_5060, HIP_0,    HIP_0   };
  m["HIPSPARSE_SPGEMM_ALG2"]                                          = {HIP_5060, HIP_0,    HIP_0   };
  m["HIPSPARSE_SPGEMM_ALG3"]                                          = {HIP_5060, HIP_0,    HIP_0   };
  m["hipsparseConstSpVecDescr_t"]                                     = {HIP_6000, HIP_0,    HIP_0   };
  m["hipsparseConstSpMatDescr_t"]                                     = {HIP_6000, HIP_0,    HIP_0   };
  m["hipsparseConstDnVecDescr_t"]                                     = {HIP_6000, HIP_0,    HIP_0   };
  m["hipsparseConstDnMatDescr_t"]                                     = {HIP_6000, HIP_0,    HIP_0   };
  m["csric02Info_t"]                                                  = {HIP_3010, HIP_0,    HIP_0   };
  m["csric02Info"]                                                    = {HIP_3010, HIP_0,    HIP_0   };

  m["_rocsparse_handle"]                                              = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_handle"]                                               = {HIP_1090, HIP_0,    HIP_0   };
  m["_rocsparse_hyb_mat"]                                             = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_hyb_mat"]                                              = {HIP_1090, HIP_0,    HIP_0   };
  m["_rocsparse_mat_descr"]                                           = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_mat_descr"]                                            = {HIP_1090, HIP_0,    HIP_0   };
  m["_rocsparse_spvec_descr"]                                         = {HIP_4010, HIP_0,    HIP_0   };
  m["rocsparse_spvec_descr"]                                          = {HIP_4010, HIP_0,    HIP_0   };
  m["_rocsparse_spmat_descr"]                                         = {HIP_4010, HIP_0,    HIP_0   };
  m["rocsparse_spmat_descr"]                                          = {HIP_4010, HIP_0,    HIP_0   };
  m["_rocsparse_dnvec_descr"]                                         = {HIP_4010, HIP_0,    HIP_0   };
  m["rocsparse_dnvec_descr"]                                          = {HIP_4010, HIP_0,    HIP_0   };
  m["_rocsparse_dnmat_descr"]                                         = {HIP_4010, HIP_0,    HIP_0   };
  m["rocsparse_dnmat_descr"]                                          = {HIP_4010, HIP_0,    HIP_0   };
  m["_rocsparse_color_info"]                                          = {HIP_4050, HIP_0,    HIP_0   };
  m["rocsparse_color_info"]                                           = {HIP_4050, HIP_0,    HIP_0   };
  m["rocsparse_operation"]                                            = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_operation_none"]                                       = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_operation_transpose"]                                  = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_operation_conjugate_transpose"]                        = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_index_base"]                                           = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_index_base_zero"]                                      = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_index_base_one"]                                       = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_matrix_type"]                                          = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_matrix_type_general"]                                  = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_matrix_type_symmetric"]                                = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_matrix_type_hermitian"]                                = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_matrix_type_triangular"]                               = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_diag_type"]                                            = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_diag_type_non_unit"]                                   = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_diag_type_unit"]                                       = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_fill_mode"]                                            = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_fill_mode_lower"]                                      = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_fill_mode_upper"]                                      = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_action"]                                               = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_action_symbolic"]                                      = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_action_numeric"]                                       = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_direction"]                                            = {HIP_3010, HIP_0,    HIP_0   };
  m["rocsparse_direction_row"]                                        = {HIP_3010, HIP_0,    HIP_0   };
  m["rocsparse_direction_column"]                                     = {HIP_3010, HIP_0,    HIP_0   };
  m["rocsparse_hyb_partition"]                                        = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_hyb_partition_auto"]                                   = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_hyb_partition_user"]                                   = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_hyb_partition_max"]                                    = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_solve_policy"]                                         = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_solve_policy_auto"]                                    = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_pointer_mode"]                                         = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_pointer_mode_host"]                                    = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_pointer_mode_device"]                                  = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_status"]                                               = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_status_success"]                                       = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_status_not_initialized"]                               = {HIP_4010, HIP_0,    HIP_0   };
  m["rocsparse_status_memory_error"]                                  = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_status_invalid_value"]                                 = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_status_arch_mismatch"]                                 = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_status_internal_error"]                                = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_status_zero_pivot"]                                    = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_status_not_implemented"]                               = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_indextype"]                                            = {HIP_4010, HIP_0,    HIP_0   };
  m["rocsparse_indextype_u16"]                                        = {HIP_4010, HIP_0,    HIP_0   };
  m["rocsparse_indextype_i32"]                                        = {HIP_4010, HIP_0,    HIP_0   };
  m["rocsparse_indextype_i64"]                                        = {HIP_4010, HIP_0,    HIP_0   };
  m["rocsparse_format"]                                               = {HIP_4010, HIP_0,    HIP_0   };
  m["rocsparse_format_coo"]                                           = {HIP_4010, HIP_0,    HIP_0   };
  m["rocsparse_format_coo_aos"]                                       = {HIP_4010, HIP_0,    HIP_0   };
  m["rocsparse_format_csr"]                                           = {HIP_4010, HIP_0,    HIP_0   };
  m["rocsparse_format_csc"]                                           = {HIP_4010, HIP_0,    HIP_0   };
  m["rocsparse_format_ell"]                                           = {HIP_4010, HIP_0,    HIP_0   };
  m["rocsparse_format_bell"]                                          = {HIP_4050, HIP_0,    HIP_0   };
  m["rocsparse_format_bsr"]                                           = {HIP_5030, HIP_0,    HIP_0   };
  m["rocsparse_order"]                                                = {HIP_4010, HIP_0,    HIP_0   };
  m["rocsparse_order_row"]                                            = {HIP_4010, HIP_0,    HIP_0   };
  m["rocsparse_order_column"]                                         = {HIP_4010, HIP_0,    HIP_0   };
  m["rocsparse_spmat_attribute"]                                      = {HIP_4050, HIP_0,    HIP_0   };
  m["rocsparse_spmat_fill_mode"]                                      = {HIP_4050, HIP_0,    HIP_0   };
  m["rocsparse_spmat_diag_type"]                                      = {HIP_4050, HIP_0,    HIP_0   };
  m["rocsparse_spmv_alg"]                                             = {HIP_4010, HIP_0,    HIP_0   };
  m["rocsparse_spmv_alg_default"]                                     = {HIP_4010, HIP_0,    HIP_0   };
  m["rocsparse_spmv_alg_coo"]                                         = {HIP_4010, HIP_0,    HIP_0   };
  m["rocsparse_spmv_alg_coo_atomic"]                                  = {HIP_5030, HIP_0,    HIP_0   };
  m["rocsparse_spmv_alg_csr_adaptive"]                                = {HIP_4010, HIP_0,    HIP_0   };
  m["rocsparse_spmv_alg_csr_stream"]                                  = {HIP_4010, HIP_0,    HIP_0   };
  m["rocsparse_spmv_alg_ell"]                                         = {HIP_4010, HIP_0,    HIP_0   };
  m["rocsparse_spsv_alg"]                                             = {HIP_4050, HIP_0,    HIP_0   };
  m["rocsparse_spsv_alg_default"]                                     = {HIP_4050, HIP_0,    HIP_0   };
  m["rocsparse_spsm_alg"]                                             = {HIP_4050, HIP_0,    HIP_0   };
  m["rocsparse_spsm_alg_default"]                                     = {HIP_4050, HIP_0,    HIP_0   };
  m["rocsparse_spmm_alg"]                                             = {HIP_4020, HIP_0,    HIP_0   };
  m["rocsparse_spmm_alg_default"]                                     = {HIP_4020, HIP_0,    HIP_0   };
  m["rocsparse_spmm_alg_coo_segmented"]                               = {HIP_4020, HIP_0,    HIP_0   };
  m["rocsparse_spmm_alg_coo_atomic"]                                  = {HIP_4020, HIP_0,    HIP_0   };
  m["rocsparse_spmm_alg_coo_segmented_atomic"]                        = {HIP_4050, HIP_0,    HIP_0   };
  m["rocsparse_spmm_alg_csr"]                                         = {HIP_4020, HIP_0,    HIP_0   };
  m["rocsparse_spmm_alg_csr_row_split"]                               = {HIP_4050, HIP_0,    HIP_0   };
  m["rocsparse_spmm_alg_csr_merge"]                                   = {HIP_4050, HIP_0,    HIP_0   };
  m["rocsparse_spmm_alg_bell"]                                        = {HIP_4050, HIP_0,    HIP_0   };
  m["rocsparse_sddmm_alg"]                                            = {HIP_4030, HIP_0,    HIP_0   };
  m["rocsparse_sddmm_alg_default"]                                    = {HIP_4030, HIP_0,    HIP_0   };
  m["rocsparse_sparse_to_dense_alg"]                                  = {HIP_4010, HIP_0,    HIP_0   };
  m["rocsparse_sparse_to_dense_alg_default"]                          = {HIP_4010, HIP_0,    HIP_0   };
  m["rocsparse_dense_to_sparse_alg"]                                  = {HIP_4010, HIP_0,    HIP_0   };
  m["rocsparse_dense_to_sparse_alg_default"]                          = {HIP_4010, HIP_0,    HIP_0   };
  m["rocsparse_spgemm_alg"]                                           = {HIP_4010, HIP_0,    HIP_0   };
  m["rocsparse_spgemm_alg_default"]                                   = {HIP_4010, HIP_0,    HIP_0   };
  m["_rocsparse_mat_info"]                                            = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_mat_info"]                                             = {HIP_1090, HIP_0,    HIP_0   };
  m["rocsparse_const_spvec_descr"]                                    = {HIP_6000, HIP_0,    HIP_0   };
  m["rocsparse_const_spmat_descr"]                                    = {HIP_6000, HIP_0,    HIP_0   };
  m["rocsparse_const_dnvec_descr"]                                    = {HIP_6000, HIP_0,    HIP_0   };
  m["rocsparse_const_dnmat_descr"]                                    = {HIP_6000, HIP_0,    HIP_0   };

  return m;
}();
