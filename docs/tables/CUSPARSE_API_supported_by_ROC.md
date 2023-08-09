# CUSPARSE API supported by ROC

## **4. CUSPARSE Types References**

|**CUDA**|**A**|**D**|**R**|**ROC**|**A**|**D**|**R**|**E**|
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|
|`CUSPARSE_ACTION_NUMERIC`| | | |`rocsparse_action_numeric`|1.9.0| | | |
|`CUSPARSE_ACTION_SYMBOLIC`| | | |`rocsparse_action_symbolic`|1.9.0| | | |
|`CUSPARSE_ALG0`|8.0| |11.0| | | | | |
|`CUSPARSE_ALG1`|8.0| |11.0| | | | | |
|`CUSPARSE_ALG_MERGE_PATH`|9.2| |12.0| | | | | |
|`CUSPARSE_ALG_NAIVE`|9.2| |11.0| | | | | |
|`CUSPARSE_COLOR_ALG0`|8.0|12.2| | | | | | |
|`CUSPARSE_COLOR_ALG1`|8.0|12.2| | | | | | |
|`CUSPARSE_COOMM_ALG1`|10.1|11.0|12.0| | | | | |
|`CUSPARSE_COOMM_ALG2`|10.1|11.0|12.0| | | | | |
|`CUSPARSE_COOMM_ALG3`|10.1|11.0|12.0| | | | | |
|`CUSPARSE_COOMV_ALG`|10.2|11.2|12.0| | | | | |
|`CUSPARSE_CSR2CSC_ALG1`|10.1| | | | | | | |
|`CUSPARSE_CSR2CSC_ALG2`|10.1| |12.0| | | | | |
|`CUSPARSE_CSR2CSC_ALG_DEFAULT`|12.0| | | | | | | |
|`CUSPARSE_CSRMM_ALG1`|10.2|11.0|12.0| | | | | |
|`CUSPARSE_CSRMV_ALG1`|10.2|11.2|12.0| | | | | |
|`CUSPARSE_CSRMV_ALG2`|10.2|11.2|12.0| | | | | |
|`CUSPARSE_DENSETOSPARSE_ALG_DEFAULT`|11.1| | |`rocsparse_dense_to_sparse_alg_default`|4.1.0| | | |
|`CUSPARSE_DIAG_TYPE_NON_UNIT`| | | |`rocsparse_diag_type_non_unit`|1.9.0| | | |
|`CUSPARSE_DIAG_TYPE_UNIT`| | | |`rocsparse_diag_type_unit`|1.9.0| | | |
|`CUSPARSE_DIRECTION_COLUMN`| | | |`rocsparse_direction_column`|3.1.0| | | |
|`CUSPARSE_DIRECTION_ROW`| | | |`rocsparse_direction_row`|3.1.0| | | |
|`CUSPARSE_FILL_MODE_LOWER`| | | |`rocsparse_fill_mode_lower`|1.9.0| | | |
|`CUSPARSE_FILL_MODE_UPPER`| | | |`rocsparse_fill_mode_upper`|1.9.0| | | |
|`CUSPARSE_FORMAT_BLOCKED_ELL`|11.2| | |`rocsparse_format_bell`|4.5.0| | | |
|`CUSPARSE_FORMAT_BSR`|12.1| | |`rocsparse_format_bsr`|5.3.0| | | |
|`CUSPARSE_FORMAT_COO`|10.1| | |`rocsparse_format_coo`|4.1.0| | | |
|`CUSPARSE_FORMAT_COO_AOS`|10.2| |12.0|`rocsparse_format_coo_aos`|4.1.0| | | |
|`CUSPARSE_FORMAT_CSC`|10.1| | |`rocsparse_format_csc`|4.1.0| | | |
|`CUSPARSE_FORMAT_CSR`|10.1| | |`rocsparse_format_csr`|4.1.0| | | |
|`CUSPARSE_FORMAT_SLICED_ELLPACK`|12.1| | |`rocsparse_format_ell`|4.1.0| | | |
|`CUSPARSE_HYB_PARTITION_AUTO`| |10.2|11.0|`rocsparse_hyb_partition_auto`|1.9.0| | | |
|`CUSPARSE_HYB_PARTITION_MAX`| |10.2|11.0|`rocsparse_hyb_partition_max`|1.9.0| | | |
|`CUSPARSE_HYB_PARTITION_USER`| |10.2|11.0|`rocsparse_hyb_partition_user`|1.9.0| | | |
|`CUSPARSE_INDEX_16U`|10.1| | |`rocsparse_indextype_u16`|4.1.0| | | |
|`CUSPARSE_INDEX_32I`|10.1| | |`rocsparse_indextype_i32`|4.1.0| | | |
|`CUSPARSE_INDEX_64I`|10.2| | |`rocsparse_indextype_i64`|4.1.0| | | |
|`CUSPARSE_INDEX_BASE_ONE`| | | |`rocsparse_index_base_one`|1.9.0| | | |
|`CUSPARSE_INDEX_BASE_ZERO`| | | |`rocsparse_index_base_zero`|1.9.0| | | |
|`CUSPARSE_MATRIX_TYPE_GENERAL`| | | |`rocsparse_matrix_type_general`|1.9.0| | | |
|`CUSPARSE_MATRIX_TYPE_HERMITIAN`| | | |`rocsparse_matrix_type_hermitian`|1.9.0| | | |
|`CUSPARSE_MATRIX_TYPE_SYMMETRIC`| | | |`rocsparse_matrix_type_symmetric`|1.9.0| | | |
|`CUSPARSE_MATRIX_TYPE_TRIANGULAR`| | | |`rocsparse_matrix_type_triangular`|1.9.0| | | |
|`CUSPARSE_MM_ALG_DEFAULT`|10.2|11.0|12.0| | | | | |
|`CUSPARSE_MV_ALG_DEFAULT`|10.2|11.3|12.0| | | | | |
|`CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE`| | | |`rocsparse_operation_conjugate_transpose`|1.9.0| | | |
|`CUSPARSE_OPERATION_NON_TRANSPOSE`| | | |`rocsparse_operation_none`|1.9.0| | | |
|`CUSPARSE_OPERATION_TRANSPOSE`| | | |`rocsparse_operation_transpose`|1.9.0| | | |
|`CUSPARSE_ORDER_COL`|10.1| | |`rocsparse_order_row`|4.1.0| | | |
|`CUSPARSE_ORDER_ROW`|10.1| | |`rocsparse_order_column`|4.1.0| | | |
|`CUSPARSE_POINTER_MODE_DEVICE`| | | |`rocsparse_pointer_mode_device`|1.9.0| | | |
|`CUSPARSE_POINTER_MODE_HOST`| | | |`rocsparse_pointer_mode_host`|1.9.0| | | |
|`CUSPARSE_SDDMM_ALG_DEFAULT`|11.2| | |`rocsparse_sddmm_alg_default`|4.3.0| | | |
|`CUSPARSE_SIDE_LEFT`| | |11.5| | | | | |
|`CUSPARSE_SIDE_RIGHT`| | |11.5| | | | | |
|`CUSPARSE_SOLVE_POLICY_NO_LEVEL`| |12.2| |`rocsparse_solve_policy_auto`|1.9.0| | | |
|`CUSPARSE_SOLVE_POLICY_USE_LEVEL`| |12.2| |`rocsparse_solve_policy_auto`|1.9.0| | | |
|`CUSPARSE_SPARSETODENSE_ALG_DEFAULT`|11.1| | |`rocsparse_sparse_to_dense_alg_default`|4.1.0| | | |
|`CUSPARSE_SPGEMM_ALG1`|12.0| | | | | | | |
|`CUSPARSE_SPGEMM_ALG2`|12.0| | | | | | | |
|`CUSPARSE_SPGEMM_ALG3`|12.0| | | | | | | |
|`CUSPARSE_SPGEMM_CSR_ALG_DETERMINITIC`|11.3| | | | | | | |
|`CUSPARSE_SPGEMM_CSR_ALG_NONDETERMINITIC`|11.3| | | | | | | |
|`CUSPARSE_SPGEMM_DEFAULT`|11.0| | |`rocsparse_spgemm_alg_default`|4.1.0| | | |
|`CUSPARSE_SPMAT_DIAG_TYPE`|11.3| | |`rocsparse_spmat_diag_type`|4.5.0| | | |
|`CUSPARSE_SPMAT_FILL_MODE`|11.3| | |`rocsparse_spmat_fill_mode`|4.5.0| | | |
|`CUSPARSE_SPMMA_ALG1`|11.1| |11.2| | | | | |
|`CUSPARSE_SPMMA_ALG2`|11.1| |11.2| | | | | |
|`CUSPARSE_SPMMA_ALG3`|11.1| |11.2| | | | | |
|`CUSPARSE_SPMMA_ALG4`|11.1| |11.2| | | | | |
|`CUSPARSE_SPMMA_PREPROCESS`|11.1| |11.2| | | | | |
|`CUSPARSE_SPMM_ALG_DEFAULT`|11.0| | |`rocsparse_spmm_alg_default`|4.2.0| | | |
|`CUSPARSE_SPMM_BLOCKED_ELL_ALG1`|11.2| | |`rocsparse_spmm_alg_bell`|4.5.0| | | |
|`CUSPARSE_SPMM_COO_ALG1`|11.0| | |`rocsparse_spmm_alg_coo_segmented`|4.2.0| | | |
|`CUSPARSE_SPMM_COO_ALG2`|11.0| | |`rocsparse_spmm_alg_coo_atomic`|4.2.0| | | |
|`CUSPARSE_SPMM_COO_ALG3`|11.0| | |`rocsparse_spmm_alg_coo_segmented_atomic`|4.5.0| | | |
|`CUSPARSE_SPMM_COO_ALG4`|11.0| | | | | | | |
|`CUSPARSE_SPMM_CSR_ALG1`|11.0| | |`rocsparse_spmm_alg_csr`|4.2.0| | | |
|`CUSPARSE_SPMM_CSR_ALG2`|11.0| | |`rocsparse_spmm_alg_csr_row_split`|4.5.0| | | |
|`CUSPARSE_SPMM_CSR_ALG3`|11.2| | |`rocsparse_spmm_alg_csr_merge`|4.5.0| | | |
|`CUSPARSE_SPMM_OP_ALG_DEFAULT`|11.5| | | | | | | |
|`CUSPARSE_SPMV_ALG_DEFAULT`|11.2| | |`rocsparse_spmv_alg_default`|4.1.0| | | |
|`CUSPARSE_SPMV_COO_ALG1`|11.2| | |`rocsparse_spmv_alg_coo`|4.1.0| | | |
|`CUSPARSE_SPMV_COO_ALG2`|11.2| | |`rocsparse_spmv_alg_coo_atomic`|5.3.0| | | |
|`CUSPARSE_SPMV_CSR_ALG1`|11.2| | |`rocsparse_spmv_alg_csr_adaptive`|4.1.0| | | |
|`CUSPARSE_SPMV_CSR_ALG2`|11.2| | |`rocsparse_spmv_alg_csr_stream`|4.1.0| | | |
|`CUSPARSE_SPMV_SELL_ALG1`|12.1| | |`rocsparse_spmv_alg_ell`|4.1.0| | | |
|`CUSPARSE_SPSM_ALG_DEFAULT`|11.3| | |`rocsparse_spsm_alg_default`|4.5.0| | | |
|`CUSPARSE_SPSV_ALG_DEFAULT`|11.3| | |`rocsparse_spsv_alg_default`|4.5.0| | | |
|`CUSPARSE_SPSV_UPDATE_DIAGONAL`|12.1| | | | | | | |
|`CUSPARSE_SPSV_UPDATE_GENERAL`|12.1| | | | | | | |
|`CUSPARSE_STATUS_ALLOC_FAILED`| | | |`rocsparse_status_memory_error`|1.9.0| | | |
|`CUSPARSE_STATUS_ARCH_MISMATCH`| | | |`rocsparse_status_arch_mismatch`|1.9.0| | | |
|`CUSPARSE_STATUS_EXECUTION_FAILED`| | | | | | | | |
|`CUSPARSE_STATUS_INSUFFICIENT_RESOURCES`|11.0| | | | | | | |
|`CUSPARSE_STATUS_INTERNAL_ERROR`| | | |`rocsparse_status_internal_error`|1.9.0| | | |
|`CUSPARSE_STATUS_INVALID_VALUE`| | | |`rocsparse_status_invalid_value`|1.9.0| | | |
|`CUSPARSE_STATUS_MAPPING_ERROR`| | | | | | | | |
|`CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED`| | | | | | | | |
|`CUSPARSE_STATUS_NOT_INITIALIZED`| | | |`rocsparse_status_not_initialized`|4.1.0| | | |
|`CUSPARSE_STATUS_NOT_SUPPORTED`|10.2| | |`rocsparse_status_not_implemented`|1.9.0| | | |
|`CUSPARSE_STATUS_SUCCESS`| | | |`rocsparse_status_success`|1.9.0| | | |
|`CUSPARSE_STATUS_ZERO_PIVOT`| | | |`rocsparse_status_zero_pivot`|1.9.0| | | |
|`bsric02Info`| | | | | | | | |
|`bsric02Info_t`| | | | | | | | |
|`bsrilu02Info`| |12.2| | | | | | |
|`bsrilu02Info_t`| |12.2| | | | | | |
|`bsrsm2Info`| |12.2| | | | | | |
|`bsrsm2Info_t`| |12.2| | | | | | |
|`bsrsv2Info`| |12.2| | | | | | |
|`bsrsv2Info_t`| |12.2| | | | | | |
|`csrgemm2Info`| | |12.0| | | | | |
|`csrgemm2Info_t`| | |12.0| | | | | |
|`csric02Info`| |12.2| | | | | | |
|`csric02Info_t`| |12.2| | | | | | |
|`csrilu02Info`| |12.2| | | | | | |
|`csrilu02Info_t`| |12.2| | | | | | |
|`csrsm2Info`|9.2| |12.0| | | | | |
|`csrsm2Info_t`|9.2| |12.0| | | | | |
|`csrsv2Info`| | |12.0| | | | | |
|`csrsv2Info_t`| | |12.0| | | | | |
|`csru2csrInfo`| |12.2| | | | | | |
|`csru2csrInfo_t`| |12.2| | | | | | |
|`cusparseAction_t`| | | |`rocsparse_action`|1.9.0| | | |
|`cusparseAlgMode_t`|8.0| |12.0| | | | | |
|`cusparseColorAlg_t`|8.0|12.2| | | | | | |
|`cusparseColorInfo`| |12.2| |`_rocsparse_color_info`|4.5.0| | | |
|`cusparseColorInfo_t`| |12.2| |`rocsparse_color_info`|4.5.0| | | |
|`cusparseConstDnMatDescr_t`|12.0| | | | | | | |
|`cusparseConstDnVecDescr_t`|12.0| | | | | | | |
|`cusparseConstSpMatDescr_t`|12.0| | | | | | | |
|`cusparseConstSpVecDescr_t`|12.0| | | | | | | |
|`cusparseContext`| | | |`_rocsparse_handle`|1.9.0| | | |
|`cusparseCsr2CscAlg_t`|10.1| | | | | | | |
|`cusparseDenseToSparseAlg_t`|11.1| | |`rocsparse_dense_to_sparse_alg`|4.1.0| | | |
|`cusparseDiagType_t`| | | |`rocsparse_diag_type`|1.9.0| | | |
|`cusparseDirection_t`| | | |`rocsparse_direction`|3.1.0| | | |
|`cusparseDnMatDescr`|10.1| | |`_rocsparse_dnmat_descr`|4.1.0| | | |
|`cusparseDnMatDescr_t`|10.1| | |`rocsparse_dnmat_descr`|4.1.0| | | |
|`cusparseDnVecDescr`|10.2| | |`_rocsparse_dnvec_descr`|4.1.0| | | |
|`cusparseDnVecDescr_t`|10.2| | |`rocsparse_dnvec_descr`|4.1.0| | | |
|`cusparseFillMode_t`| | | |`rocsparse_fill_mode`|1.9.0| | | |
|`cusparseFormat_t`|10.1| | |`rocsparse_format`|4.1.0| | | |
|`cusparseHandle_t`| | | |`rocsparse_handle`|1.9.0| | | |
|`cusparseHybMat`| |10.2|11.0|`_rocsparse_hyb_mat`|1.9.0| | | |
|`cusparseHybMat_t`| |10.2|11.0|`rocsparse_hyb_mat`|1.9.0| | | |
|`cusparseHybPartition_t`| |10.2|11.0|`rocsparse_hyb_partition`|1.9.0| | | |
|`cusparseIndexBase_t`| | | |`rocsparse_index_base`|1.9.0| | | |
|`cusparseIndexType_t`|10.1| | |`rocsparse_indextype`|4.1.0| | | |
|`cusparseLoggerCallback_t`|11.5| | | | | | | |
|`cusparseMatDescr`| | | |`_rocsparse_mat_descr`|1.9.0| | | |
|`cusparseMatDescr_t`| | | |`rocsparse_mat_descr`|1.9.0| | | |
|`cusparseMatrixType_t`| | | |`rocsparse_matrix_type`|1.9.0| | | |
|`cusparseOperation_t`| | | |`rocsparse_operation`|1.9.0| | | |
|`cusparseOrder_t`|10.1| | |`rocsparse_order`|4.1.0| | | |
|`cusparsePointerMode_t`| | | |`rocsparse_pointer_mode`|1.9.0| | | |
|`cusparseSDDMMAlg_t`|11.2| | |`rocsparse_sddmm_alg`|4.3.0| | | |
|`cusparseSideMode_t`| | |11.5| | | | | |
|`cusparseSolveAnalysisInfo`| |10.2|11.0| | | | | |
|`cusparseSolveAnalysisInfo_t`| |10.2|11.0| | | | | |
|`cusparseSolvePolicy_t`| |12.2| |`rocsparse_solve_policy`|1.9.0| | | |
|`cusparseSpGEMMAlg_t`|11.0| | |`rocsparse_spgemm_alg`|4.1.0| | | |
|`cusparseSpGEMMDescr`|11.0| | | | | | | |
|`cusparseSpGEMMDescr_t`|11.0| | | | | | | |
|`cusparseSpMMAlg_t`|10.1| | |`rocsparse_spmm_alg`|4.2.0| | | |
|`cusparseSpMMOpAlg_t`|11.5| | | | | | | |
|`cusparseSpMMOpPlan`|11.5| | | | | | | |
|`cusparseSpMMOpPlan_t`|11.5| | | | | | | |
|`cusparseSpMVAlg_t`|10.2| | |`rocsparse_spmv_alg`|4.1.0| | | |
|`cusparseSpMatAttribute_t`|11.3| | |`rocsparse_spmat_attribute`|4.5.0| | | |
|`cusparseSpMatDescr`|10.1| | |`_rocsparse_spmat_descr`|4.1.0| | | |
|`cusparseSpMatDescr_t`|10.1| | |`rocsparse_spmat_descr`|4.1.0| | | |
|`cusparseSpSMAlg_t`|11.3| | |`rocsparse_spsm_alg`|4.5.0| | | |
|`cusparseSpSMDescr`|11.3| | | | | | | |
|`cusparseSpSMDescr_t`|11.3| | | | | | | |
|`cusparseSpSVAlg_t`|11.3| | |`rocsparse_spsv_alg`|4.5.0| | | |
|`cusparseSpSVDescr`|11.3| | | | | | | |
|`cusparseSpSVDescr_t`|11.3| | | | | | | |
|`cusparseSpSVUpdate_t`|12.1| | | | | | | |
|`cusparseSpVecDescr`|10.2| | |`_rocsparse_spvec_descr`|4.1.0| | | |
|`cusparseSpVecDescr_t`|10.2| | |`rocsparse_spvec_descr`|4.1.0| | | |
|`cusparseSparseToDenseAlg_t`|11.1| | |`rocsparse_sparse_to_dense_alg`|4.1.0| | | |
|`cusparseStatus_t`| | | |`rocsparse_status`|1.9.0| | | |
|`pruneInfo`|9.0|12.2| |`_rocsparse_mat_info`|1.9.0| | | |
|`pruneInfo_t`|9.0|12.2| |`rocsparse_mat_info`|1.9.0| | | |

## **5. CUSPARSE Management Function Reference**

|**CUDA**|**A**|**D**|**R**|**ROC**|**A**|**D**|**R**|**E**|
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|
|`cusparseCreate`| | | |`rocsparse_create_handle`|1.9.0| | | |
|`cusparseDestroy`| | | |`rocsparse_destroy_handle`|1.9.0| | | |
|`cusparseGetPointerMode`| | | |`rocsparse_get_pointer_mode`|1.9.0| | | |
|`cusparseGetStream`| | | |`rocsparse_get_stream`|1.9.0| | | |
|`cusparseGetVersion`| | | |`rocsparse_get_version`|1.9.0| | | |
|`cusparseSetPointerMode`| | | |`rocsparse_set_pointer_mode`|1.9.0| | | |
|`cusparseSetStream`| | | |`rocsparse_set_stream`|1.9.0| | | |

## **6. CUSPARSE Logging**

|**CUDA**|**A**|**D**|**R**|**ROC**|**A**|**D**|**R**|**E**|
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|
|`cusparseLoggerForceDisable`|11.5| | | | | | | |
|`cusparseLoggerOpenFile`|11.5| | | | | | | |
|`cusparseLoggerSetCallback`|11.5| | | | | | | |
|`cusparseLoggerSetFile`|11.5| | | | | | | |
|`cusparseLoggerSetLevel`|11.5| | | | | | | |
|`cusparseLoggerSetMask`|11.5| | | | | | | |

## **7. CUSPARSE Helper Function Reference**

|**CUDA**|**A**|**D**|**R**|**ROC**|**A**|**D**|**R**|**E**|
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|
|`cusparseCopyMatDescr`|8.0| |12.0|`rocsparse_copy_mat_descr`|1.9.0| | | |
|`cusparseCreateBsric02Info`| |12.2| | | | | | |
|`cusparseCreateBsrilu02Info`| |12.2| | | | | | |
|`cusparseCreateBsrsm2Info`| |12.2| | | | | | |
|`cusparseCreateBsrsv2Info`| |12.2| | | | | | |
|`cusparseCreateColorInfo`| |12.2| |`rocsparse_create_color_info`|4.5.0| | | |
|`cusparseCreateCsrgemm2Info`| |11.0|12.0| | | | | |
|`cusparseCreateCsric02Info`| |12.2| | | | | | |
|`cusparseCreateCsrilu02Info`| |12.2| | | | | | |
|`cusparseCreateCsrsm2Info`|10.0|11.3|12.0| | | | | |
|`cusparseCreateCsrsv2Info`| |11.3|12.0| | | | | |
|`cusparseCreateHybMat`| |10.2|11.0|`rocsparse_create_hyb_mat`|1.9.0| | | |
|`cusparseCreateMatDescr`| | | |`rocsparse_create_mat_descr`|1.9.0| | | |
|`cusparseCreatePruneInfo`|9.0|12.2| | | | | | |
|`cusparseCreateSolveAnalysisInfo`| |10.2|11.0| | | | | |
|`cusparseDestroyBsric02Info`| |12.2| | | | | | |
|`cusparseDestroyBsrilu02Info`| |12.2| | | | | | |
|`cusparseDestroyBsrsm2Info`| |12.2| | | | | | |
|`cusparseDestroyBsrsv2Info`| |12.2| | | | | | |
|`cusparseDestroyColorInfo`| |12.2| |`rocsparse_destroy_color_info`|4.5.0| | | |
|`cusparseDestroyCsrgemm2Info`| |11.0|12.0| | | | | |
|`cusparseDestroyCsric02Info`| |12.2| | | | | | |
|`cusparseDestroyCsrilu02Info`| |12.2| | | | | | |
|`cusparseDestroyCsrsm2Info`|10.0|11.3|12.0| | | | | |
|`cusparseDestroyCsrsv2Info`| |11.3|12.0| | | | | |
|`cusparseDestroyHybMat`| |10.2|11.0|`rocsparse_destroy_hyb_mat`|1.9.0| | | |
|`cusparseDestroyMatDescr`| | | |`rocsparse_destroy_mat_descr`|1.9.0| | | |
|`cusparseDestroyPruneInfo`|9.0|12.2| | | | | | |
|`cusparseDestroySolveAnalysisInfo`| |10.2|11.0| | | | | |
|`cusparseGetLevelInfo`| | |11.0| | | | | |
|`cusparseGetMatDiagType`| | | |`rocsparse_get_mat_diag_type`|1.9.0| | | |
|`cusparseGetMatFillMode`| | | |`rocsparse_get_mat_fill_mode`|1.9.0| | | |
|`cusparseGetMatIndexBase`| | | |`rocsparse_get_mat_index_base`|1.9.0| | | |
|`cusparseGetMatType`| | | |`rocsparse_get_mat_type`|1.9.0| | | |
|`cusparseSetMatDiagType`| | | |`rocsparse_set_mat_diag_type`|1.9.0| | | |
|`cusparseSetMatFillMode`| | | |`rocsparse_set_mat_fill_mode`|1.9.0| | | |
|`cusparseSetMatIndexBase`| | | |`rocsparse_set_mat_index_base`|1.9.0| | | |
|`cusparseSetMatType`| | | |`rocsparse_set_mat_type`|1.9.0| | | |

## **8. CUSPARSE Level 1 Function Reference**

|**CUDA**|**A**|**D**|**R**|**ROC**|**A**|**D**|**R**|**E**|
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|
|`cusparseCaxpyi`| |11.0|12.0| | | | | |
|`cusparseCdotci`| |10.2|11.0| | | | | |
|`cusparseCdoti`| |10.2|11.0| | | | | |
|`cusparseCgthr`| |11.0|12.0| | | | | |
|`cusparseCgthrz`| |11.0|12.0| | | | | |
|`cusparseCsctr`| |11.0|12.0| | | | | |
|`cusparseDaxpyi`| |11.0|12.0| | | | | |
|`cusparseDdoti`| |10.2|11.0| | | | | |
|`cusparseDgthr`| |11.0|12.0| | | | | |
|`cusparseDgthrz`| |11.0|12.0| | | | | |
|`cusparseDroti`| |11.0|12.0| | | | | |
|`cusparseDsctr`| |11.0|12.0| | | | | |
|`cusparseSaxpyi`| |11.0|12.0| | | | | |
|`cusparseSdoti`| |10.2|11.0| | | | | |
|`cusparseSgthr`| |11.0|12.0| | | | | |
|`cusparseSgthrz`| |11.0|12.0| | | | | |
|`cusparseSroti`| |11.0|12.0| | | | | |
|`cusparseSsctr`| |11.0|12.0| | | | | |
|`cusparseZaxpyi`| |11.0|12.0| | | | | |
|`cusparseZdotci`| |10.2|11.0| | | | | |
|`cusparseZdoti`| |10.2|11.0| | | | | |
|`cusparseZgthr`| |11.0|12.0| | | | | |
|`cusparseZgthrz`| |11.0|12.0| | | | | |
|`cusparseZsctr`| |11.0|12.0| | | | | |

## **9. CUSPARSE Level 2 Function Reference**

|**CUDA**|**A**|**D**|**R**|**ROC**|**A**|**D**|**R**|**E**|
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|
|`cusparseCbsrmv`| | | | | | | | |
|`cusparseCbsrsv2_analysis`| | | | | | | | |
|`cusparseCbsrsv2_bufferSize`| | | | | | | | |
|`cusparseCbsrsv2_bufferSizeExt`| | | | | | | | |
|`cusparseCbsrsv2_solve`| | | | | | | | |
|`cusparseCbsrxmv`| | | | | | | | |
|`cusparseCcsrmv`| |10.2|11.0| | | | | |
|`cusparseCcsrmv_mp`|8.0|10.2|11.0| | | | | |
|`cusparseCcsrsv2_analysis`| |11.3|12.0| | | | | |
|`cusparseCcsrsv2_bufferSize`| |11.3|12.0| | | | | |
|`cusparseCcsrsv2_bufferSizeExt`| |11.3|12.0| | | | | |
|`cusparseCcsrsv2_solve`| |11.3|12.0| | | | | |
|`cusparseCcsrsv_analysis`| |10.2|11.0| | | | | |
|`cusparseCcsrsv_solve`| |10.2|11.0| | | | | |
|`cusparseCgemvi`|7.5| | | | | | | |
|`cusparseCgemvi_bufferSize`|7.5| | | | | | | |
|`cusparseChybmv`| |10.2|11.0| | | | | |
|`cusparseChybsv_analysis`| |10.2|11.0| | | | | |
|`cusparseChybsv_solve`| |10.2|11.0| | | | | |
|`cusparseCsrmvEx`|8.0|11.2|12.0| | | | | |
|`cusparseCsrmvEx_bufferSize`|8.0|11.2|12.0| | | | | |
|`cusparseCsrsv_analysisEx`|8.0|10.2|11.0| | | | | |
|`cusparseCsrsv_solveEx`|8.0|10.2|11.0| | | | | |
|`cusparseDbsrmv`| | | | | | | | |
|`cusparseDbsrsv2_analysis`| | | | | | | | |
|`cusparseDbsrsv2_bufferSize`| | | | | | | | |
|`cusparseDbsrsv2_bufferSizeExt`| | | | | | | | |
|`cusparseDbsrsv2_solve`| | | | | | | | |
|`cusparseDbsrxmv`| | | | | | | | |
|`cusparseDcsrmv`| |10.2|11.0| | | | | |
|`cusparseDcsrmv_mp`|8.0|10.2|11.0| | | | | |
|`cusparseDcsrsv2_analysis`| |11.3|12.0| | | | | |
|`cusparseDcsrsv2_bufferSize`| |11.3|12.0| | | | | |
|`cusparseDcsrsv2_bufferSizeExt`| |11.3|12.0| | | | | |
|`cusparseDcsrsv2_solve`| |11.3|12.0| | | | | |
|`cusparseDcsrsv_analysis`| |10.2|11.0| | | | | |
|`cusparseDcsrsv_solve`| |10.2|11.0| | | | | |
|`cusparseDgemvi`|7.5| | | | | | | |
|`cusparseDgemvi_bufferSize`|7.5| | | | | | | |
|`cusparseDhybmv`| |10.2|11.0| | | | | |
|`cusparseDhybsv_analysis`| |10.2|11.0| | | | | |
|`cusparseDhybsv_solve`| |10.2|11.0| | | | | |
|`cusparseSbsrmv`| | | | | | | | |
|`cusparseSbsrsv2_analysis`| | | | | | | | |
|`cusparseSbsrsv2_bufferSize`| | | | | | | | |
|`cusparseSbsrsv2_bufferSizeExt`| | | | | | | | |
|`cusparseSbsrsv2_solve`| | | | | | | | |
|`cusparseSbsrxmv`| | | | | | | | |
|`cusparseScsrmv`| |10.2|11.0| | | | | |
|`cusparseScsrmv_mp`|8.0|10.2|11.0| | | | | |
|`cusparseScsrsv2_analysis`| |11.3|12.0| | | | | |
|`cusparseScsrsv2_bufferSize`| |11.3|12.0| | | | | |
|`cusparseScsrsv2_bufferSizeExt`| |11.3|12.0| | | | | |
|`cusparseScsrsv2_solve`| |11.3|12.0| | | | | |
|`cusparseScsrsv_analysis`| |10.2|11.0| | | | | |
|`cusparseScsrsv_solve`| |10.2|11.0| | | | | |
|`cusparseSgemvi`|7.5| | | | | | | |
|`cusparseSgemvi_bufferSize`|7.5| | | | | | | |
|`cusparseShybmv`| |10.2|11.0| | | | | |
|`cusparseShybsv_analysis`| |10.2|11.0| | | | | |
|`cusparseShybsv_solve`| |10.2|11.0| | | | | |
|`cusparseXbsrsv2_zeroPivot`| | | | | | | | |
|`cusparseXcsrsv2_zeroPivot`| |11.3|12.0| | | | | |
|`cusparseZbsrmv`| | | | | | | | |
|`cusparseZbsrsv2_analysis`| | | | | | | | |
|`cusparseZbsrsv2_bufferSize`| | | | | | | | |
|`cusparseZbsrsv2_bufferSizeExt`| | | | | | | | |
|`cusparseZbsrsv2_solve`| | | | | | | | |
|`cusparseZbsrxmv`| | | | | | | | |
|`cusparseZcsrmv`| |10.2|11.0| | | | | |
|`cusparseZcsrmv_mp`|8.0|10.2|11.0| | | | | |
|`cusparseZcsrsv2_analysis`| |11.3|12.0| | | | | |
|`cusparseZcsrsv2_bufferSize`| |11.3|12.0| | | | | |
|`cusparseZcsrsv2_bufferSizeExt`| |11.3|12.0| | | | | |
|`cusparseZcsrsv2_solve`| |11.3|12.0| | | | | |
|`cusparseZcsrsv_analysis`| |10.2|11.0| | | | | |
|`cusparseZcsrsv_solve`| |10.2|11.0| | | | | |
|`cusparseZgemvi`|7.5| | | | | | | |
|`cusparseZgemvi_bufferSize`|7.5| | | | | | | |
|`cusparseZhybmv`| |10.2|11.0| | | | | |
|`cusparseZhybsv_analysis`| |10.2|11.0| | | | | |
|`cusparseZhybsv_solve`| |10.2|11.0| | | | | |

## **10. CUSPARSE Level 3 Function Reference**

|**CUDA**|**A**|**D**|**R**|**ROC**|**A**|**D**|**R**|**E**|
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|
|`cusparseCbsrmm`| | | | | | | | |
|`cusparseCbsrsm2_analysis`| | | | | | | | |
|`cusparseCbsrsm2_bufferSize`| | | | | | | | |
|`cusparseCbsrsm2_bufferSizeExt`| | | | | | | | |
|`cusparseCbsrsm2_solve`| | | | | | | | |
|`cusparseCcsrmm`| |10.2|11.0| | | | | |
|`cusparseCcsrmm2`| |10.2|11.0| | | | | |
|`cusparseCcsrsm2_analysis`|10.0|11.3|12.0| | | | | |
|`cusparseCcsrsm2_bufferSizeExt`|10.0|11.3|12.0| | | | | |
|`cusparseCcsrsm2_solve`|10.0|11.3|12.0| | | | | |
|`cusparseCcsrsm_analysis`| |10.2|11.0| | | | | |
|`cusparseCcsrsm_solve`| |10.2|11.0| | | | | |
|`cusparseCgemmi`|8.0|11.0|12.0| | | | | |
|`cusparseDbsrmm`| | | | | | | | |
|`cusparseDbsrsm2_analysis`| | | | | | | | |
|`cusparseDbsrsm2_bufferSize`| | | | | | | | |
|`cusparseDbsrsm2_bufferSizeExt`| | | | | | | | |
|`cusparseDbsrsm2_solve`| | | | | | | | |
|`cusparseDcsrmm`| |10.2|11.0| | | | | |
|`cusparseDcsrmm2`| |10.2|11.0| | | | | |
|`cusparseDcsrsm2_analysis`|10.0|11.3|12.0| | | | | |
|`cusparseDcsrsm2_bufferSizeExt`|10.0|11.3|12.0| | | | | |
|`cusparseDcsrsm2_solve`|10.0|11.3|12.0| | | | | |
|`cusparseDcsrsm_analysis`| |10.2|11.0| | | | | |
|`cusparseDcsrsm_solve`| |10.2|11.0| | | | | |
|`cusparseDgemmi`|8.0|11.0|12.0| | | | | |
|`cusparseSbsrmm`| | | | | | | | |
|`cusparseSbsrsm2_analysis`| | | | | | | | |
|`cusparseSbsrsm2_bufferSize`| | | | | | | | |
|`cusparseSbsrsm2_bufferSizeExt`| | | | | | | | |
|`cusparseSbsrsm2_solve`| | | | | | | | |
|`cusparseScsrmm`| |10.2|11.0| | | | | |
|`cusparseScsrmm2`| |10.2|11.0| | | | | |
|`cusparseScsrsm2_analysis`|10.0|11.3|12.0| | | | | |
|`cusparseScsrsm2_bufferSizeExt`|10.0|11.3|12.0| | | | | |
|`cusparseScsrsm2_solve`|10.0|11.3|12.0| | | | | |
|`cusparseScsrsm_analysis`| |10.2|11.0| | | | | |
|`cusparseScsrsm_solve`| |10.2|11.0| | | | | |
|`cusparseSgemmi`|8.0|11.0|12.0| | | | | |
|`cusparseXbsrsm2_zeroPivot`| | | | | | | | |
|`cusparseXcsrsm2_zeroPivot`|10.0|11.3|12.0| | | | | |
|`cusparseZbsrmm`| | | | | | | | |
|`cusparseZbsrsm2_analysis`| | | | | | | | |
|`cusparseZbsrsm2_bufferSize`| | | | | | | | |
|`cusparseZbsrsm2_bufferSizeExt`| | | | | | | | |
|`cusparseZbsrsm2_solve`| | | | | | | | |
|`cusparseZcsrmm`| |10.2|11.0| | | | | |
|`cusparseZcsrmm2`| |10.2|11.0| | | | | |
|`cusparseZcsrsm2_analysis`|10.0|11.3|12.0| | | | | |
|`cusparseZcsrsm2_bufferSizeExt`|10.0|11.3|12.0| | | | | |
|`cusparseZcsrsm2_solve`|10.0|11.3|12.0| | | | | |
|`cusparseZcsrsm_analysis`| |10.2|11.0| | | | | |
|`cusparseZcsrsm_solve`| |10.2|11.0| | | | | |
|`cusparseZgemmi`|8.0|11.0|12.0| | | | | |

## **11. CUSPARSE Extra Function Reference**

|**CUDA**|**A**|**D**|**R**|**ROC**|**A**|**D**|**R**|**E**|
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|
|`cusparseCcsrgeam`| |10.2|11.0| | | | | |
|`cusparseCcsrgeam2`|10.0| | | | | | | |
|`cusparseCcsrgeam2_bufferSizeExt`|10.0| | | | | | | |
|`cusparseCcsrgemm`| |10.2|11.0| | | | | |
|`cusparseCcsrgemm2`| |11.0|12.0| | | | | |
|`cusparseCcsrgemm2_bufferSizeExt`| |11.0|12.0| | | | | |
|`cusparseDcsrgeam`| |10.2|11.0| | | | | |
|`cusparseDcsrgeam2`|10.0| | | | | | | |
|`cusparseDcsrgeam2_bufferSizeExt`|10.0| | | | | | | |
|`cusparseDcsrgemm`| |10.2|11.0| | | | | |
|`cusparseDcsrgemm2`| |11.0|12.0| | | | | |
|`cusparseDcsrgemm2_bufferSizeExt`| |11.0|12.0| | | | | |
|`cusparseScsrgeam`| |10.2|11.0| | | | | |
|`cusparseScsrgeam2`|10.0| | | | | | | |
|`cusparseScsrgeam2_bufferSizeExt`|10.0| | | | | | | |
|`cusparseScsrgemm`| |10.2|11.0| | | | | |
|`cusparseScsrgemm2`| |11.0|12.0| | | | | |
|`cusparseScsrgemm2_bufferSizeExt`| |11.0|12.0| | | | | |
|`cusparseXcsrgeam2Nnz`|10.0| | | | | | | |
|`cusparseXcsrgeamNnz`| |10.2|11.0| | | | | |
|`cusparseXcsrgemm2Nnz`| |11.0|12.0| | | | | |
|`cusparseXcsrgemmNnz`| |10.2|11.0| | | | | |
|`cusparseZcsrgeam`| |10.2|11.0| | | | | |
|`cusparseZcsrgeam2`|10.0| | | | | | | |
|`cusparseZcsrgeam2_bufferSizeExt`|10.0| | | | | | | |
|`cusparseZcsrgemm`| |10.2|11.0| | | | | |
|`cusparseZcsrgemm2`| |11.0|12.0| | | | | |
|`cusparseZcsrgemm2_bufferSizeExt`| |11.0|12.0| | | | | |

## **12. CUSPARSE Preconditioners Reference**

|**CUDA**|**A**|**D**|**R**|**ROC**|**A**|**D**|**R**|**E**|
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|
|`cusparseCbsric02`| | | | | | | | |
|`cusparseCbsric02_analysis`| | | | | | | | |
|`cusparseCbsric02_bufferSize`| | | | | | | | |
|`cusparseCbsric02_bufferSizeExt`| | | | | | | | |
|`cusparseCbsrilu02`| | | | | | | | |
|`cusparseCbsrilu02_analysis`| | | | | | | | |
|`cusparseCbsrilu02_bufferSize`| | | | | | | | |
|`cusparseCbsrilu02_bufferSizeExt`| | | | | | | | |
|`cusparseCbsrilu02_numericBoost`| | | | | | | | |
|`cusparseCcsric0`| |10.2|11.0| | | | | |
|`cusparseCcsric02`| | | | | | | | |
|`cusparseCcsric02_analysis`| | | | | | | | |
|`cusparseCcsric02_bufferSize`| | | | | | | | |
|`cusparseCcsric02_bufferSizeExt`| | | | | | | | |
|`cusparseCcsrilu0`| |10.2|11.0| | | | | |
|`cusparseCcsrilu02`| | | | | | | | |
|`cusparseCcsrilu02_analysis`| | | | | | | | |
|`cusparseCcsrilu02_bufferSize`| | | | | | | | |
|`cusparseCcsrilu02_bufferSizeExt`| | | | | | | | |
|`cusparseCcsrilu02_numericBoost`| | | | | | | | |
|`cusparseCgpsvInterleavedBatch`|9.2| | | | | | | |
|`cusparseCgpsvInterleavedBatch_bufferSizeExt`|9.2| | | | | | | |
|`cusparseCgtsv`| |10.2|11.0| | | | | |
|`cusparseCgtsv2`|9.0| | | | | | | |
|`cusparseCgtsv2StridedBatch`| | | | | | | | |
|`cusparseCgtsv2StridedBatch_bufferSizeExt`| | | | | | | | |
|`cusparseCgtsv2_bufferSizeExt`|9.0| | | | | | | |
|`cusparseCgtsv2_nopivot`|9.0| | | | | | | |
|`cusparseCgtsv2_nopivot_bufferSizeExt`|9.0| | | | | | | |
|`cusparseCgtsvInterleavedBatch`|9.2| | | | | | | |
|`cusparseCgtsvInterleavedBatch_bufferSizeExt`|9.2| | | | | | | |
|`cusparseCgtsvStridedBatch`| |10.2|11.0| | | | | |
|`cusparseCgtsv_nopivot`| |10.2|11.0| | | | | |
|`cusparseCsrilu0Ex`|8.0|10.2|11.0| | | | | |
|`cusparseDbsric02`| | | | | | | | |
|`cusparseDbsric02_analysis`| | | | | | | | |
|`cusparseDbsric02_bufferSize`| | | | | | | | |
|`cusparseDbsric02_bufferSizeExt`| | | | | | | | |
|`cusparseDbsrilu02`| | | | | | | | |
|`cusparseDbsrilu02_analysis`| | | | | | | | |
|`cusparseDbsrilu02_bufferSize`| | | | | | | | |
|`cusparseDbsrilu02_bufferSizeExt`| | | | | | | | |
|`cusparseDbsrilu02_numericBoost`| | | | | | | | |
|`cusparseDcsric0`| |10.2|11.0| | | | | |
|`cusparseDcsric02`| | | | | | | | |
|`cusparseDcsric02_analysis`| | | | | | | | |
|`cusparseDcsric02_bufferSize`| | | | | | | | |
|`cusparseDcsric02_bufferSizeExt`| | | | | | | | |
|`cusparseDcsrilu0`| |10.2|11.0| | | | | |
|`cusparseDcsrilu02`| | | | | | | | |
|`cusparseDcsrilu02_analysis`| | | | | | | | |
|`cusparseDcsrilu02_bufferSize`| | | | | | | | |
|`cusparseDcsrilu02_bufferSizeExt`| | | | | | | | |
|`cusparseDcsrilu02_numericBoost`| | | | | | | | |
|`cusparseDgpsvInterleavedBatch`|9.2| | | | | | | |
|`cusparseDgpsvInterleavedBatch_bufferSizeExt`|9.2| | | | | | | |
|`cusparseDgtsv`| |10.2|11.0| | | | | |
|`cusparseDgtsv2`|9.0| | | | | | | |
|`cusparseDgtsv2StridedBatch`| | | | | | | | |
|`cusparseDgtsv2StridedBatch_bufferSizeExt`| | | | | | | | |
|`cusparseDgtsv2_bufferSizeExt`|9.0| | | | | | | |
|`cusparseDgtsv2_nopivot`|9.0| | | | | | | |
|`cusparseDgtsv2_nopivot_bufferSizeExt`|9.0| | | | | | | |
|`cusparseDgtsvInterleavedBatch`|9.2| | | | | | | |
|`cusparseDgtsvInterleavedBatch_bufferSizeExt`|9.2| | | | | | | |
|`cusparseDgtsvStridedBatch`| |10.2|11.0| | | | | |
|`cusparseDgtsv_nopivot`| |10.2|11.0| | | | | |
|`cusparseSbsric02`| | | | | | | | |
|`cusparseSbsric02_analysis`| | | | | | | | |
|`cusparseSbsric02_bufferSize`| | | | | | | | |
|`cusparseSbsric02_bufferSizeExt`| | | | | | | | |
|`cusparseSbsrilu02`| | | | | | | | |
|`cusparseSbsrilu02_analysis`| | | | | | | | |
|`cusparseSbsrilu02_bufferSize`| | | | | | | | |
|`cusparseSbsrilu02_bufferSizeExt`| | | | | | | | |
|`cusparseSbsrilu02_numericBoost`| | | | | | | | |
|`cusparseScsric0`| |10.2|11.0| | | | | |
|`cusparseScsric02`| | | | | | | | |
|`cusparseScsric02_analysis`| | | | | | | | |
|`cusparseScsric02_bufferSize`| | | | | | | | |
|`cusparseScsric02_bufferSizeExt`| | | | | | | | |
|`cusparseScsrilu0`| |10.2|11.0| | | | | |
|`cusparseScsrilu02`| | | | | | | | |
|`cusparseScsrilu02_analysis`| | | | | | | | |
|`cusparseScsrilu02_bufferSize`| | | | | | | | |
|`cusparseScsrilu02_bufferSizeExt`| | | | | | | | |
|`cusparseScsrilu02_numericBoost`| | | | | | | | |
|`cusparseSgpsvInterleavedBatch`|9.2| | | | | | | |
|`cusparseSgpsvInterleavedBatch_bufferSizeExt`|9.2| | | | | | | |
|`cusparseSgtsv`| |10.2|11.0| | | | | |
|`cusparseSgtsv2`|9.0| | | | | | | |
|`cusparseSgtsv2StridedBatch`|9.0| | | | | | | |
|`cusparseSgtsv2StridedBatch_bufferSizeExt`|9.0| | | | | | | |
|`cusparseSgtsv2_bufferSizeExt`|9.0| | | | | | | |
|`cusparseSgtsv2_nopivot`|9.0| | | | | | | |
|`cusparseSgtsv2_nopivot_bufferSizeExt`|9.0| | | | | | | |
|`cusparseSgtsvInterleavedBatch`|9.2| | | | | | | |
|`cusparseSgtsvInterleavedBatch_bufferSizeExt`|9.2| | | | | | | |
|`cusparseSgtsvStridedBatch`| |10.2|11.0| | | | | |
|`cusparseSgtsv_nopivot`| |10.2|11.0| | | | | |
|`cusparseXbsric02_zeroPivot`| | | | | | | | |
|`cusparseXbsrilu02_zeroPivot`| | | | | | | | |
|`cusparseXcsric02_zeroPivot`| | | | | | | | |
|`cusparseXcsrilu02_zeroPivot`| | | | | | | | |
|`cusparseZbsric02`| | | | | | | | |
|`cusparseZbsric02_analysis`| | | | | | | | |
|`cusparseZbsric02_bufferSize`| | | | | | | | |
|`cusparseZbsric02_bufferSizeExt`| | | | | | | | |
|`cusparseZbsrilu02`| | | | | | | | |
|`cusparseZbsrilu02_analysis`| | | | | | | | |
|`cusparseZbsrilu02_bufferSize`| | | | | | | | |
|`cusparseZbsrilu02_bufferSizeExt`| | | | | | | | |
|`cusparseZbsrilu02_numericBoost`| | | | | | | | |
|`cusparseZcsric0`| |10.2|11.0| | | | | |
|`cusparseZcsric02`| | | | | | | | |
|`cusparseZcsric02_analysis`| | | | | | | | |
|`cusparseZcsric02_bufferSize`| | | | | | | | |
|`cusparseZcsric02_bufferSizeExt`| | | | | | | | |
|`cusparseZcsrilu0`| |10.2|11.0| | | | | |
|`cusparseZcsrilu02`| | | | | | | | |
|`cusparseZcsrilu02_analysis`| | | | | | | | |
|`cusparseZcsrilu02_bufferSize`| | | | | | | | |
|`cusparseZcsrilu02_bufferSizeExt`| | | | | | | | |
|`cusparseZcsrilu02_numericBoost`| | | | | | | | |
|`cusparseZgpsvInterleavedBatch`|9.2| | | | | | | |
|`cusparseZgpsvInterleavedBatch_bufferSizeExt`|9.2| | | | | | | |
|`cusparseZgtsv`| |10.2|11.0| | | | | |
|`cusparseZgtsv2`|9.0| | | | | | | |
|`cusparseZgtsv2StridedBatch`| | | | | | | | |
|`cusparseZgtsv2StridedBatch_bufferSizeExt`| | | | | | | | |
|`cusparseZgtsv2_bufferSizeExt`|9.0| | | | | | | |
|`cusparseZgtsv2_nopivot`|9.0| | | | | | | |
|`cusparseZgtsv2_nopivot_bufferSizeExt`|9.0| | | | | | | |
|`cusparseZgtsvInterleavedBatch`|9.2| | | | | | | |
|`cusparseZgtsvInterleavedBatch_bufferSizeExt`|9.2| | | | | | | |
|`cusparseZgtsvStridedBatch`| |10.2|11.0| | | | | |
|`cusparseZgtsv_nopivot`| |10.2|11.0| | | | | |

## **13. CUSPARSE Reorderings Reference**

|**CUDA**|**A**|**D**|**R**|**ROC**|**A**|**D**|**R**|**E**|
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|
|`cusparseCcsrcolor`| | | |`rocsparse_ccsrcolor`|4.5.0| | | |
|`cusparseDcsrcolor`| | | |`rocsparse_dcsrcolor`|4.5.0| | | |
|`cusparseScsrcolor`| | | |`rocsparse_scsrcolor`|4.5.0| | | |
|`cusparseZcsrcolor`| | | |`rocsparse_zcsrcolor`|4.5.0| | | |

## **14. CUSPARSE Format Conversion Reference**

|**CUDA**|**A**|**D**|**R**|**ROC**|**A**|**D**|**R**|**E**|
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|
|`cusparseCbsr2csr`| | | |`rocsparse_cbsr2csr`|3.10.0| | | |
|`cusparseCcsc2dense`| |11.1|12.0| | | | | |
|`cusparseCcsc2hyb`| |10.2|11.0| | | | | |
|`cusparseCcsr2bsr`| | | |`rocsparse_ccsr2bsr`|3.5.0| | | |
|`cusparseCcsr2csc`| |10.2|11.0| | | | | |
|`cusparseCcsr2csr_compress`|8.0| | |`rocsparse_ccsr2csr_compress`|3.5.0| | | |
|`cusparseCcsr2csru`| | | | | | | | |
|`cusparseCcsr2dense`| |11.1|12.0| | | | | |
|`cusparseCcsr2gebsr`| | | |`rocsparse_ccsr2gebsr`|4.1.0| | | |
|`cusparseCcsr2gebsr_bufferSize`| | | |`rocsparse_ccsr2gebsr_buffer_size`|4.1.0| | | |
|`cusparseCcsr2gebsr_bufferSizeExt`| | | | | | | | |
|`cusparseCcsr2hyb`| |10.2|11.0|`rocsparse_ccsr2hyb`|2.10.0| | | |
|`cusparseCcsru2csr`| | | | | | | | |
|`cusparseCcsru2csr_bufferSizeExt`| | | | | | | | |
|`cusparseCdense2csc`| |11.1|12.0| | | | | |
|`cusparseCdense2csr`| |11.1|12.0| | | | | |
|`cusparseCdense2hyb`| |10.2|11.0| | | | | |
|`cusparseCgebsr2csr`| | | |`rocsparse_cgebsr2csr`|3.10.0| | | |
|`cusparseCgebsr2gebsc`| | | |`rocsparse_cgebsr2gebsc`|4.1.0| | | |
|`cusparseCgebsr2gebsc_bufferSize`| | | | | | | | |
|`cusparseCgebsr2gebsc_bufferSizeExt`| | | | | | | | |
|`cusparseCgebsr2gebsr`| | | |`rocsparse_cgebsr2gebsr`|4.1.0| | | |
|`cusparseCgebsr2gebsr_bufferSize`| | | |`rocsparse_cgebsr2gebsr_buffer_size`|4.1.0| | | |
|`cusparseCgebsr2gebsr_bufferSizeExt`| | | | | | | | |
|`cusparseChyb2csc`| |10.2|11.0| | | | | |
|`cusparseChyb2csr`| |10.2|11.0| | | | | |
|`cusparseChyb2dense`| |10.2|11.0| | | | | |
|`cusparseCnnz`| | | | | | | | |
|`cusparseCnnz_compress`|8.0| | | | | | | |
|`cusparseCreateCsru2csrInfo`| |12.2| | | | | | |
|`cusparseCreateIdentityPermutation`| | | |`rocsparse_create_identity_permutation`|1.9.0| | | |
|`cusparseCsr2cscEx`|8.0|10.2|11.0| | | | | |
|`cusparseCsr2cscEx2`|10.1| | | | | | | |
|`cusparseCsr2cscEx2_bufferSize`|10.1| | | | | | | |
|`cusparseDbsr2csr`| | | |`rocsparse_dbsr2csr`|3.10.0| | | |
|`cusparseDcsc2dense`| |11.1|12.0| | | | | |
|`cusparseDcsc2hyb`| |10.2|11.0| | | | | |
|`cusparseDcsr2bsr`| | | |`rocsparse_dcsr2bsr`|3.5.0| | | |
|`cusparseDcsr2csc`| |10.2|11.0| | | | | |
|`cusparseDcsr2csr_compress`|8.0| | |`rocsparse_dcsr2csr_compress`|3.5.0| | | |
|`cusparseDcsr2csru`| | | | | | | | |
|`cusparseDcsr2dense`| |11.1|12.0| | | | | |
|`cusparseDcsr2gebsr`| | | |`rocsparse_dcsr2gebsr`|4.1.0| | | |
|`cusparseDcsr2gebsr_bufferSize`| | | |`rocsparse_dcsr2gebsr_buffer_size`|4.1.0| | | |
|`cusparseDcsr2gebsr_bufferSizeExt`| | | | | | | | |
|`cusparseDcsr2hyb`| |10.2|11.0|`rocsparse_dcsr2hyb`|1.9.0| | | |
|`cusparseDcsru2csr`| | | | | | | | |
|`cusparseDcsru2csr_bufferSizeExt`| | | | | | | | |
|`cusparseDdense2csc`| |11.1|12.0| | | | | |
|`cusparseDdense2csr`| |11.1|12.0| | | | | |
|`cusparseDdense2hyb`| |10.2|11.0| | | | | |
|`cusparseDestroyCsru2csrInfo`| |12.2| | | | | | |
|`cusparseDgebsr2csr`| | | |`rocsparse_dgebsr2csr`|3.10.0| | | |
|`cusparseDgebsr2gebsc`| | | |`rocsparse_dgebsr2gebsc`|4.1.0| | | |
|`cusparseDgebsr2gebsc_bufferSize`| | | | | | | | |
|`cusparseDgebsr2gebsc_bufferSizeExt`| | | | | | | | |
|`cusparseDgebsr2gebsr`| | | |`rocsparse_dgebsr2gebsr`|4.1.0| | | |
|`cusparseDgebsr2gebsr_bufferSize`| | | |`rocsparse_dgebsr2gebsr_buffer_size`|4.1.0| | | |
|`cusparseDgebsr2gebsr_bufferSizeExt`| | | | | | | | |
|`cusparseDhyb2csc`| |10.2|11.0| | | | | |
|`cusparseDhyb2csr`| |10.2|11.0| | | | | |
|`cusparseDhyb2dense`| |10.2|11.0| | | | | |
|`cusparseDnnz`| | | | | | | | |
|`cusparseDnnz_compress`|8.0| | | | | | | |
|`cusparseDpruneCsr2csr`|9.0| | |`rocsparse_dprune_csr2csr`|3.9.0| | | |
|`cusparseDpruneCsr2csrByPercentage`|9.0| | |`rocsparse_dprune_csr2csr_by_percentage`|3.9.0| | | |
|`cusparseDpruneCsr2csrByPercentage_bufferSizeExt`|9.0| | |`rocsparse_dprune_csr2csr_by_percentage_buffer_size`|3.9.0| | | |
|`cusparseDpruneCsr2csrNnz`|9.0| | |`rocsparse_dprune_csr2csr_nnz`|3.9.0| | | |
|`cusparseDpruneCsr2csrNnzByPercentage`|9.0| | |`rocsparse_dprune_csr2csr_nnz_by_percentage`|3.9.0| | | |
|`cusparseDpruneCsr2csr_bufferSizeExt`|9.0| | |`rocsparse_dprune_csr2csr_buffer_size`|3.9.0| | | |
|`cusparseDpruneDense2csr`|9.0| | | | | | | |
|`cusparseDpruneDense2csrByPercentage`|9.0| | | | | | | |
|`cusparseDpruneDense2csrByPercentage_bufferSizeExt`|9.0| | | | | | | |
|`cusparseDpruneDense2csrNnz`|9.0| | | | | | | |
|`cusparseDpruneDense2csrNnzByPercentage`|9.0| | | | | | | |
|`cusparseDpruneDense2csr_bufferSizeExt`|9.0| | | | | | | |
|`cusparseHpruneCsr2csr`|9.0| | | | | | | |
|`cusparseHpruneCsr2csrByPercentage`|9.0| | | | | | | |
|`cusparseHpruneCsr2csrByPercentage_bufferSizeExt`|9.0| | | | | | | |
|`cusparseHpruneCsr2csrNnz`|9.0| | | | | | | |
|`cusparseHpruneCsr2csrNnzByPercentage`|9.0| | | | | | | |
|`cusparseHpruneCsr2csr_bufferSizeExt`|9.0| | | | | | | |
|`cusparseHpruneDense2csr`|9.0| | | | | | | |
|`cusparseHpruneDense2csrByPercentage`|9.0| | | | | | | |
|`cusparseHpruneDense2csrByPercentage_bufferSizeExt`|9.0| | | | | | | |
|`cusparseHpruneDense2csrNnz`|9.0| | | | | | | |
|`cusparseHpruneDense2csrNnzByPercentage`|9.0| | | | | | | |
|`cusparseHpruneDense2csr_bufferSizeExt`|9.0| | | | | | | |
|`cusparseSbsr2csr`| | | |`rocsparse_sbsr2csr`|3.10.0| | | |
|`cusparseScsc2dense`| |11.1|12.0| | | | | |
|`cusparseScsc2hyb`| |10.2|11.0| | | | | |
|`cusparseScsr2bsr`| | | |`rocsparse_scsr2bsr`|3.5.0| | | |
|`cusparseScsr2csc`| |10.2|11.0| | | | | |
|`cusparseScsr2csr_compress`|8.0| | |`rocsparse_scsr2csr_compress`|3.5.0| | | |
|`cusparseScsr2csru`| | | | | | | | |
|`cusparseScsr2dense`| |11.1|12.0| | | | | |
|`cusparseScsr2gebsr`| | | |`rocsparse_scsr2gebsr`|4.1.0| | | |
|`cusparseScsr2gebsr_bufferSize`| | | |`rocsparse_scsr2gebsr_buffer_size`|4.1.0| | | |
|`cusparseScsr2gebsr_bufferSizeExt`| | | | | | | | |
|`cusparseScsr2hyb`| |10.2|11.0|`rocsparse_scsr2hyb`|1.9.0| | | |
|`cusparseScsru2csr`| | | | | | | | |
|`cusparseScsru2csr_bufferSizeExt`| | | | | | | | |
|`cusparseSdense2csc`| |11.1|12.0| | | | | |
|`cusparseSdense2csr`| |11.1|12.0| | | | | |
|`cusparseSdense2hyb`| |10.2|11.0| | | | | |
|`cusparseSgebsr2csr`| | | |`rocsparse_sgebsr2csr`|3.10.0| | | |
|`cusparseSgebsr2gebsc`| | | |`rocsparse_sgebsr2gebsc`|4.1.0| | | |
|`cusparseSgebsr2gebsc_bufferSize`| | | | | | | | |
|`cusparseSgebsr2gebsc_bufferSizeExt`| | | | | | | | |
|`cusparseSgebsr2gebsr`| | | |`rocsparse_sgebsr2gebsr`|4.1.0| | | |
|`cusparseSgebsr2gebsr_bufferSize`| | | |`rocsparse_sgebsr2gebsr_buffer_size`|4.1.0| | | |
|`cusparseSgebsr2gebsr_bufferSizeExt`| | | | | | | | |
|`cusparseShyb2csc`| |10.2|11.0| | | | | |
|`cusparseShyb2csr`| |10.2|11.0| | | | | |
|`cusparseShyb2dense`| |10.2|11.0| | | | | |
|`cusparseSnnz`| | | | | | | | |
|`cusparseSnnz_compress`|8.0| | | | | | | |
|`cusparseSpruneCsr2csr`|9.0| | |`rocsparse_sprune_csr2csr`|3.9.0| | | |
|`cusparseSpruneCsr2csrByPercentage`|9.0| | |`rocsparse_sprune_csr2csr_by_percentage`|3.9.0| | | |
|`cusparseSpruneCsr2csrByPercentage_bufferSizeExt`|9.0| | |`rocsparse_sprune_csr2csr_by_percentage_buffer_size`|3.9.0| | | |
|`cusparseSpruneCsr2csrNnz`|9.0| | |`rocsparse_sprune_csr2csr_nnz`|3.9.0| | | |
|`cusparseSpruneCsr2csrNnzByPercentage`|9.0| | |`rocsparse_sprune_csr2csr_nnz_by_percentage`|3.9.0| | | |
|`cusparseSpruneCsr2csr_bufferSizeExt`|9.0| | |`rocsparse_sprune_csr2csr_buffer_size`|3.9.0| | | |
|`cusparseSpruneDense2csr`|9.0| | | | | | | |
|`cusparseSpruneDense2csrByPercentage`|9.0| | | | | | | |
|`cusparseSpruneDense2csrByPercentage_bufferSizeExt`|9.0| | | | | | | |
|`cusparseSpruneDense2csrNnz`|9.0| | | | | | | |
|`cusparseSpruneDense2csrNnzByPercentage`|9.0| | | | | | | |
|`cusparseSpruneDense2csr_bufferSizeExt`|9.0| | | | | | | |
|`cusparseXcoo2csr`| | | |`rocsparse_coo2csr`|1.9.0| | | |
|`cusparseXcoosortByColumn`| | | |`rocsparse_coosort_by_column`|1.9.0| | | |
|`cusparseXcoosortByRow`| | | |`rocsparse_coosort_by_row`|1.9.0| | | |
|`cusparseXcoosort_bufferSizeExt`| | | |`rocsparse_coosort_buffer_size`|1.9.0| | | |
|`cusparseXcscsort`| | | |`rocsparse_cscsort`|2.10.0| | | |
|`cusparseXcscsort_bufferSizeExt`| | | |`rocsparse_cscsort_buffer_size`|2.10.0| | | |
|`cusparseXcsr2bsrNnz`| | | |`rocsparse_csr2bsr_nnz`|3.5.0| | | |
|`cusparseXcsr2coo`| | | | | | | | |
|`cusparseXcsr2gebsrNnz`| | | |`rocsparse_csr2gebsr_nnz`|4.1.0| | | |
|`cusparseXcsrsort`| | | |`rocsparse_csrsort`|1.9.0| | | |
|`cusparseXcsrsort_bufferSizeExt`| | | |`rocsparse_csrsort_buffer_size`|1.9.0| | | |
|`cusparseXgebsr2csr`| | | | | | | | |
|`cusparseXgebsr2gebsrNnz`| | | |`rocsparse_gebsr2gebsr_nnz`|4.1.0| | | |
|`cusparseZbsr2csr`| | | |`rocsparse_zbsr2csr`|3.10.0| | | |
|`cusparseZcsc2dense`| |11.1|12.0| | | | | |
|`cusparseZcsc2hyb`| |10.2|11.0| | | | | |
|`cusparseZcsr2bsr`| | | |`rocsparse_zcsr2bsr`|3.5.0| | | |
|`cusparseZcsr2csc`| |10.2|11.0| | | | | |
|`cusparseZcsr2csr_compress`|8.0| | |`rocsparse_zcsr2csr_compress`|3.5.0| | | |
|`cusparseZcsr2csru`| | | | | | | | |
|`cusparseZcsr2dense`| |11.1|12.0| | | | | |
|`cusparseZcsr2gebsr`| | | |`rocsparse_zcsr2gebsr`|4.1.0| | | |
|`cusparseZcsr2gebsr_bufferSize`| | | |`rocsparse_zcsr2gebsr_buffer_size`|4.1.0| | | |
|`cusparseZcsr2gebsr_bufferSizeExt`| | | | | | | | |
|`cusparseZcsr2hyb`| |10.2|11.0|`rocsparse_zcsr2hyb`|2.10.0| | | |
|`cusparseZcsru2csr`| | | | | | | | |
|`cusparseZcsru2csr_bufferSizeExt`| | | | | | | | |
|`cusparseZdense2csc`| |11.1|12.0| | | | | |
|`cusparseZdense2csr`| |11.1|12.0| | | | | |
|`cusparseZdense2hyb`| |10.2|11.0| | | | | |
|`cusparseZgebsr2csr`| | | |`rocsparse_zgebsr2csr`|3.10.0| | | |
|`cusparseZgebsr2gebsc`| | | |`rocsparse_zgebsr2gebsc`|4.1.0| | | |
|`cusparseZgebsr2gebsc_bufferSize`| | | | | | | | |
|`cusparseZgebsr2gebsc_bufferSizeExt`| | | | | | | | |
|`cusparseZgebsr2gebsr`| | | |`rocsparse_zgebsr2gebsr`|4.1.0| | | |
|`cusparseZgebsr2gebsr_bufferSize`| | | |`rocsparse_zgebsr2gebsr_buffer_size`|4.1.0| | | |
|`cusparseZgebsr2gebsr_bufferSizeExt`| | | | | | | | |
|`cusparseZhyb2csc`| |10.2|11.0| | | | | |
|`cusparseZhyb2csr`| |10.2|11.0| | | | | |
|`cusparseZhyb2dense`| |10.2|11.0| | | | | |
|`cusparseZnnz`| | | | | | | | |
|`cusparseZnnz_compress`|8.0| | | | | | | |

## **15. CUSPARSE Generic API Reference**

|**CUDA**|**A**|**D**|**R**|**ROC**|**A**|**D**|**R**|**E**|
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|
|`cusparseAxpby`|11.0| | |`rocsparse_axpby`|4.1.0| | | |
|`cusparseBlockedEllGet`|11.2| | |`rocsparse_bell_get`|4.1.0| | | |
|`cusparseBsrSetStridedBatch`|12.1| | | | | | | |
|`cusparseConstBlockedEllGet`|12.0| | | | | | | |
|`cusparseConstCooGet`|12.0| | | | | | | |
|`cusparseConstCscGet`|12.0| | | | | | | |
|`cusparseConstCsrGet`|12.0| | | | | | | |
|`cusparseConstDnMatGet`|12.0| | | | | | | |
|`cusparseConstDnMatGetValues`|12.0| | | | | | | |
|`cusparseConstDnVecGet`|12.0| | | | | | | |
|`cusparseConstDnVecGetValues`|12.0| | | | | | | |
|`cusparseConstSpMatGetValues`|12.0| | | | | | | |
|`cusparseConstSpVecGet`|12.0| | | | | | | |
|`cusparseConstSpVecGetValues`|12.0| | | | | | | |
|`cusparseConstrainedGeMM`|10.2|11.2|12.0| | | | | |
|`cusparseConstrainedGeMM_bufferSize`|10.2|11.2|12.0| | | | | |
|`cusparseCooAoSGet`|10.2|11.2|12.0|`rocsparse_coo_aos_get`|4.1.0| | | |
|`cusparseCooGet`|10.1| | |`rocsparse_coo_get`|4.1.0| | | |
|`cusparseCooSetPointers`|11.1| | |`rocsparse_coo_set_pointers`|4.1.0| | | |
|`cusparseCooSetStridedBatch`|11.0| | |`rocsparse_coo_set_strided_batch`|5.2.0| | | |
|`cusparseCreateBlockedEll`|11.2| | |`rocsparse_create_bell_descr`|4.5.0| | | |
|`cusparseCreateBsr`|12.1| | | | | | | |
|`cusparseCreateConstBlockedEll`|12.0| | | | | | | |
|`cusparseCreateConstBsr`|12.1| | | | | | | |
|`cusparseCreateConstCoo`|12.0| | | | | | | |
|`cusparseCreateConstCsc`|12.0| | | | | | | |
|`cusparseCreateConstCsr`|12.0| | | | | | | |
|`cusparseCreateConstDnMat`|12.0| | | | | | | |
|`cusparseCreateConstDnVec`|12.0| | | | | | | |
|`cusparseCreateConstSlicedEll`|12.1| | | | | | | |
|`cusparseCreateConstSpVec`|12.0| | | | | | | |
|`cusparseCreateCoo`|10.1| | |`rocsparse_create_coo_descr`|4.1.0| | | |
|`cusparseCreateCooAoS`|10.2|11.2|12.0|`rocsparse_create_coo_aos_descr`|4.1.0| | | |
|`cusparseCreateCsc`|11.1| | |`rocsparse_create_csc_descr`|4.1.0| | | |
|`cusparseCreateCsr`|10.2| | |`rocsparse_create_csr_descr`|4.1.0| | | |
|`cusparseCreateDnMat`|10.1| | |`rocsparse_create_dnmat_descr`|4.1.0| | | |
|`cusparseCreateDnVec`|10.2| | |`rocsparse_create_dnvec_descr`|4.1.0| | | |
|`cusparseCreateSlicedEll`|12.1| | | | | | | |
|`cusparseCreateSpVec`|10.2| | |`rocsparse_create_spvec_descr`|4.1.0| | | |
|`cusparseCscGet`|11.7| | | | | | | |
|`cusparseCscSetPointers`|11.1| | |`rocsparse_csc_set_pointers`|4.1.0| | | |
|`cusparseCsrGet`|10.2| | |`rocsparse_csr_get`|4.1.0| | | |
|`cusparseCsrSetPointers`|11.0| | |`rocsparse_csr_set_pointers`|4.1.0| | | |
|`cusparseCsrSetStridedBatch`|11.0| | |`rocsparse_csr_set_strided_batch`|5.2.0| | | |
|`cusparseDenseToSparse_analysis`|11.1| | | | | | | |
|`cusparseDenseToSparse_bufferSize`|11.1| | | | | | | |
|`cusparseDenseToSparse_convert`|11.1| | | | | | | |
|`cusparseDestroyDnMat`|10.1| | |`rocsparse_destroy_dnmat_descr`|4.1.0| | | |
|`cusparseDestroyDnVec`|10.2| | |`rocsparse_destroy_dnvec_descr`|4.1.0| | | |
|`cusparseDestroySpMat`|10.1| | |`rocsparse_destroy_spmat_descr`|4.1.0| | | |
|`cusparseDestroySpVec`|10.2| | |`rocsparse_destroy_spvec_descr`|4.1.0| | | |
|`cusparseDnMatGet`|10.1| | |`rocsparse_dnmat_get`|4.1.0| | | |
|`cusparseDnMatGetStridedBatch`|10.1| | |`rocsparse_dnmat_get_strided_batch`|5.2.0| | | |
|`cusparseDnMatGetValues`|10.2| | |`rocsparse_dnmat_get_values`|4.1.0| | | |
|`cusparseDnMatSetStridedBatch`|10.1| | |`rocsparse_dnmat_set_strided_batch`|5.2.0| | | |
|`cusparseDnMatSetValues`|10.2| | |`rocsparse_dnmat_set_values`|4.1.0| | | |
|`cusparseDnVecGet`|10.2| | |`rocsparse_dnvec_get`|4.1.0| | | |
|`cusparseDnVecGetValues`|10.2| | |`rocsparse_dnvec_get_values`|4.1.0| | | |
|`cusparseDnVecSetValues`|10.2| | |`rocsparse_dnvec_set_values`|4.1.0| | | |
|`cusparseGather`|11.0| | |`rocsparse_gather`|4.1.0| | | |
|`cusparseRot`|11.0| | |`rocsparse_rot`|4.1.0| | | |
|`cusparseSDDMM`|11.2| | |`rocsparse_sddmm`|4.3.0| | | |
|`cusparseSDDMM_bufferSize`|11.2| | |`rocsparse_sddmm_buffer_size`|4.3.0| | | |
|`cusparseSDDMM_preprocess`|11.2| | |`rocsparse_sddmm_preprocess`|4.3.0| | | |
|`cusparseScatter`|11.0| | |`rocsparse_scatter`|4.1.0| | | |
|`cusparseSpGEMM_compute`|11.0| | | | | | | |
|`cusparseSpGEMM_copy`|11.0| | | | | | | |
|`cusparseSpGEMM_createDescr`|11.0| | | | | | | |
|`cusparseSpGEMM_destroyDescr`|11.0| | | | | | | |
|`cusparseSpGEMM_estimateMemory`|12.0| | | | | | | |
|`cusparseSpGEMM_getNumProducts`|12.0| | | | | | | |
|`cusparseSpGEMM_workEstimation`| | | | | | | | |
|`cusparseSpGEMMreuse_compute`|11.3| | | | | | | |
|`cusparseSpGEMMreuse_copy`|11.3| | | | | | | |
|`cusparseSpGEMMreuse_nnz`|11.3| | | | | | | |
|`cusparseSpGEMMreuse_workEstimation`|11.3| | | | | | | |
|`cusparseSpMM`|10.1| | | | | | | |
|`cusparseSpMMOp`|11.5| | | | | | | |
|`cusparseSpMMOp_createPlan`|11.5| | | | | | | |
|`cusparseSpMMOp_destroyPlan`|11.5| | | | | | | |
|`cusparseSpMM_bufferSize`|10.1| | | | | | | |
|`cusparseSpMM_preprocess`|11.2| | | | | | | |
|`cusparseSpMV`|10.2| | |`rocsparse_spmv`|4.1.0| | | |
|`cusparseSpMV_bufferSize`|10.2| | | | | | | |
|`cusparseSpMatGetAttribute`|11.3| | |`rocsparse_spmat_get_attribute`|4.5.0| | | |
|`cusparseSpMatGetFormat`|10.1| | |`rocsparse_spmat_get_format`|4.1.0| | | |
|`cusparseSpMatGetIndexBase`|10.1| | |`rocsparse_spmat_get_index_base`|4.1.0| | | |
|`cusparseSpMatGetNumBatches`|10.1| |10.2| | | | | |
|`cusparseSpMatGetSize`|11.0| | |`rocsparse_spmat_get_size`|4.1.0| | | |
|`cusparseSpMatGetStridedBatch`|10.2| | |`rocsparse_spmat_get_strided_batch`|5.2.0| | | |
|`cusparseSpMatGetValues`|10.2| | |`rocsparse_spmat_get_values`|4.1.0| | | |
|`cusparseSpMatSetAttribute`|11.3| | |`rocsparse_spmat_set_attribute`|4.5.0| | | |
|`cusparseSpMatSetNumBatches`|10.1| |10.2| | | | | |
|`cusparseSpMatSetStridedBatch`|10.2| |12.0|`rocsparse_spmat_set_strided_batch`|5.2.0| | | |
|`cusparseSpMatSetValues`|10.2| | |`rocsparse_spmat_set_values`|4.1.0| | | |
|`cusparseSpSM_analysis`|11.3| | | | | | | |
|`cusparseSpSM_bufferSize`|11.3| | | | | | | |
|`cusparseSpSM_createDescr`|11.3| | | | | | | |
|`cusparseSpSM_destroyDescr`|11.3| | | | | | | |
|`cusparseSpSM_solve`|11.3| | | | | | | |
|`cusparseSpSV_analysis`|11.3| | | | | | | |
|`cusparseSpSV_bufferSize`|11.3| | | | | | | |
|`cusparseSpSV_createDescr`|11.3| | | | | | | |
|`cusparseSpSV_destroyDescr`|11.3| | | | | | | |
|`cusparseSpSV_solve`|11.3| | | | | | | |
|`cusparseSpSV_updateMatrix`|12.1| | | | | | | |
|`cusparseSpVV`|10.2| | | | | | | |
|`cusparseSpVV_bufferSize`|10.2| | | | | | | |
|`cusparseSpVecGet`|10.2| | |`rocsparse_spvec_get`|4.1.0| | | |
|`cusparseSpVecGetIndexBase`|10.2| | |`rocsparse_spvec_get_index_base`|4.1.0| | | |
|`cusparseSpVecGetValues`|10.2| | |`rocsparse_spvec_get_values`|4.1.0| | | |
|`cusparseSpVecSetValues`|10.2| | |`rocsparse_spvec_set_values`|4.1.0| | | |
|`cusparseSparseToDense`|11.1| | | | | | | |
|`cusparseSparseToDense_bufferSize`|11.1| | | | | | | |


\*A - Added; D - Deprecated; R - Removed; E - Experimental