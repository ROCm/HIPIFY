# CUSPARSE API supported by ROC

## **4. CUSPARSE Types References**

|**CUDA**|**A**|**D**|**C**|**R**|**ROC**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`CUSPARSE_ACTION_NUMERIC`| | | | |`rocsparse_action_numeric`|1.9.0| | | | |
|`CUSPARSE_ACTION_SYMBOLIC`| | | | |`rocsparse_action_symbolic`|1.9.0| | | | |
|`CUSPARSE_ALG0`|8.0| | |11.0| | | | | | |
|`CUSPARSE_ALG1`|8.0| | |11.0| | | | | | |
|`CUSPARSE_ALG_MERGE_PATH`|9.2| | |12.0| | | | | | |
|`CUSPARSE_ALG_NAIVE`|9.2| | |11.0| | | | | | |
|`CUSPARSE_COLOR_ALG0`|8.0|12.2| | | | | | | | |
|`CUSPARSE_COLOR_ALG1`|8.0|12.2| | | | | | | | |
|`CUSPARSE_COOMM_ALG1`|10.1|11.0| |12.0| | | | | | |
|`CUSPARSE_COOMM_ALG2`|10.1|11.0| |12.0| | | | | | |
|`CUSPARSE_COOMM_ALG3`|10.1|11.0| |12.0| | | | | | |
|`CUSPARSE_COOMV_ALG`|10.2|11.2| |12.0| | | | | | |
|`CUSPARSE_CSR2CSC_ALG1`|10.1| | | | | | | | | |
|`CUSPARSE_CSR2CSC_ALG2`|10.1| | |12.0| | | | | | |
|`CUSPARSE_CSR2CSC_ALG_DEFAULT`|12.0| | | | | | | | | |
|`CUSPARSE_CSRMM_ALG1`|10.2|11.0| |12.0| | | | | | |
|`CUSPARSE_CSRMV_ALG1`|10.2|11.2| |12.0| | | | | | |
|`CUSPARSE_CSRMV_ALG2`|10.2|11.2| |12.0| | | | | | |
|`CUSPARSE_DENSETOSPARSE_ALG_DEFAULT`|11.1| | | |`rocsparse_dense_to_sparse_alg_default`|4.1.0| | | | |
|`CUSPARSE_DIAG_TYPE_NON_UNIT`| | | | |`rocsparse_diag_type_non_unit`|1.9.0| | | | |
|`CUSPARSE_DIAG_TYPE_UNIT`| | | | |`rocsparse_diag_type_unit`|1.9.0| | | | |
|`CUSPARSE_DIRECTION_COLUMN`| | | | |`rocsparse_direction_column`|3.1.0| | | | |
|`CUSPARSE_DIRECTION_ROW`| | | | |`rocsparse_direction_row`|3.1.0| | | | |
|`CUSPARSE_FILL_MODE_LOWER`| | | | |`rocsparse_fill_mode_lower`|1.9.0| | | | |
|`CUSPARSE_FILL_MODE_UPPER`| | | | |`rocsparse_fill_mode_upper`|1.9.0| | | | |
|`CUSPARSE_FORMAT_BLOCKED_ELL`|11.2| | | |`rocsparse_format_bell`|4.5.0| | | | |
|`CUSPARSE_FORMAT_BSR`|12.1| | | |`rocsparse_format_bsr`|5.3.0| | | | |
|`CUSPARSE_FORMAT_COO`|10.1| | | |`rocsparse_format_coo`|4.1.0| | | | |
|`CUSPARSE_FORMAT_COO_AOS`|10.2| | |12.0|`rocsparse_format_coo_aos`|4.1.0| | | | |
|`CUSPARSE_FORMAT_CSC`|10.1| | | |`rocsparse_format_csc`|4.1.0| | | | |
|`CUSPARSE_FORMAT_CSR`|10.1| | | |`rocsparse_format_csr`|4.1.0| | | | |
|`CUSPARSE_FORMAT_SLICED_ELLPACK`|12.1| | | |`rocsparse_format_ell`|4.1.0| | | | |
|`CUSPARSE_HYB_PARTITION_AUTO`| |10.2| |11.0|`rocsparse_hyb_partition_auto`|1.9.0| | | | |
|`CUSPARSE_HYB_PARTITION_MAX`| |10.2| |11.0|`rocsparse_hyb_partition_max`|1.9.0| | | | |
|`CUSPARSE_HYB_PARTITION_USER`| |10.2| |11.0|`rocsparse_hyb_partition_user`|1.9.0| | | | |
|`CUSPARSE_INDEX_16U`|10.1| | | |`rocsparse_indextype_u16`|4.1.0| | | | |
|`CUSPARSE_INDEX_32I`|10.1| | | |`rocsparse_indextype_i32`|4.1.0| | | | |
|`CUSPARSE_INDEX_64I`|10.1| | | |`rocsparse_indextype_i64`|4.1.0| | | | |
|`CUSPARSE_INDEX_BASE_ONE`| | | | |`rocsparse_index_base_one`|1.9.0| | | | |
|`CUSPARSE_INDEX_BASE_ZERO`| | | | |`rocsparse_index_base_zero`|1.9.0| | | | |
|`CUSPARSE_MATRIX_TYPE_GENERAL`| | | | |`rocsparse_matrix_type_general`|1.9.0| | | | |
|`CUSPARSE_MATRIX_TYPE_HERMITIAN`| | | | |`rocsparse_matrix_type_hermitian`|1.9.0| | | | |
|`CUSPARSE_MATRIX_TYPE_SYMMETRIC`| | | | |`rocsparse_matrix_type_symmetric`|1.9.0| | | | |
|`CUSPARSE_MATRIX_TYPE_TRIANGULAR`| | | | |`rocsparse_matrix_type_triangular`|1.9.0| | | | |
|`CUSPARSE_MM_ALG_DEFAULT`|10.2|11.0| |12.0| | | | | | |
|`CUSPARSE_MV_ALG_DEFAULT`|10.2|11.3| |12.0| | | | | | |
|`CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE`| | | | |`rocsparse_operation_conjugate_transpose`|1.9.0| | | | |
|`CUSPARSE_OPERATION_NON_TRANSPOSE`| | | | |`rocsparse_operation_none`|1.9.0| | | | |
|`CUSPARSE_OPERATION_TRANSPOSE`| | | | |`rocsparse_operation_transpose`|1.9.0| | | | |
|`CUSPARSE_ORDER_COL`|10.1| | | |`rocsparse_order_row`|4.1.0| | | | |
|`CUSPARSE_ORDER_ROW`|10.1| | | |`rocsparse_order_column`|4.1.0| | | | |
|`CUSPARSE_POINTER_MODE_DEVICE`| | | | |`rocsparse_pointer_mode_device`|1.9.0| | | | |
|`CUSPARSE_POINTER_MODE_HOST`| | | | |`rocsparse_pointer_mode_host`|1.9.0| | | | |
|`CUSPARSE_SDDMM_ALG_DEFAULT`|11.2| | | |`rocsparse_sddmm_alg_default`|4.3.0| | | | |
|`CUSPARSE_SIDE_LEFT`| | | |11.5| | | | | | |
|`CUSPARSE_SIDE_RIGHT`| | | |11.5| | | | | | |
|`CUSPARSE_SOLVE_POLICY_NO_LEVEL`| |12.2| | |`rocsparse_solve_policy_auto`|1.9.0| | | | |
|`CUSPARSE_SOLVE_POLICY_USE_LEVEL`| |12.2| | |`rocsparse_solve_policy_auto`|1.9.0| | | | |
|`CUSPARSE_SPARSETODENSE_ALG_DEFAULT`|11.1| | | |`rocsparse_sparse_to_dense_alg_default`|4.1.0| | | | |
|`CUSPARSE_SPGEMM_ALG1`|12.0| | | | | | | | | |
|`CUSPARSE_SPGEMM_ALG2`|12.0| | | | | | | | | |
|`CUSPARSE_SPGEMM_ALG3`|12.0| | | | | | | | | |
|`CUSPARSE_SPGEMM_CSR_ALG_DETERMINITIC`|11.3| | | | | | | | | |
|`CUSPARSE_SPGEMM_CSR_ALG_NONDETERMINITIC`|11.3| | | | | | | | | |
|`CUSPARSE_SPGEMM_DEFAULT`|11.0| | | |`rocsparse_spgemm_alg_default`|4.1.0| | | | |
|`CUSPARSE_SPMAT_DIAG_TYPE`|11.3| | | |`rocsparse_spmat_diag_type`|4.5.0| | | | |
|`CUSPARSE_SPMAT_FILL_MODE`|11.3| | | |`rocsparse_spmat_fill_mode`|4.5.0| | | | |
|`CUSPARSE_SPMMA_ALG1`|11.1| | |11.2| | | | | | |
|`CUSPARSE_SPMMA_ALG2`|11.1| | |11.2| | | | | | |
|`CUSPARSE_SPMMA_ALG3`|11.1| | |11.2| | | | | | |
|`CUSPARSE_SPMMA_ALG4`|11.1| | |11.2| | | | | | |
|`CUSPARSE_SPMMA_PREPROCESS`|11.1| | |11.2| | | | | | |
|`CUSPARSE_SPMM_ALG_DEFAULT`|11.0| | | |`rocsparse_spmm_alg_default`|4.2.0| | | | |
|`CUSPARSE_SPMM_BLOCKED_ELL_ALG1`|11.2| | | |`rocsparse_spmm_alg_bell`|4.5.0| | | | |
|`CUSPARSE_SPMM_COO_ALG1`|11.0| | | |`rocsparse_spmm_alg_coo_segmented`|4.2.0| | | | |
|`CUSPARSE_SPMM_COO_ALG2`|11.0| | | |`rocsparse_spmm_alg_coo_atomic`|4.2.0| | | | |
|`CUSPARSE_SPMM_COO_ALG3`|11.0| | | |`rocsparse_spmm_alg_coo_segmented_atomic`|4.5.0| | | | |
|`CUSPARSE_SPMM_COO_ALG4`|11.0| | | | | | | | | |
|`CUSPARSE_SPMM_CSR_ALG1`|11.0| | | |`rocsparse_spmm_alg_csr`|4.2.0| | | | |
|`CUSPARSE_SPMM_CSR_ALG2`|11.0| | | |`rocsparse_spmm_alg_csr_row_split`|4.5.0| | | | |
|`CUSPARSE_SPMM_CSR_ALG3`|11.2| | | |`rocsparse_spmm_alg_csr_merge`|4.5.0| | | | |
|`CUSPARSE_SPMM_OP_ALG_DEFAULT`|11.5| | | | | | | | | |
|`CUSPARSE_SPMV_ALG_DEFAULT`|11.2| | | |`rocsparse_spmv_alg_default`|4.1.0| | | | |
|`CUSPARSE_SPMV_COO_ALG1`|11.2| | | |`rocsparse_spmv_alg_coo`|4.1.0| | | | |
|`CUSPARSE_SPMV_COO_ALG2`|11.2| | | |`rocsparse_spmv_alg_coo_atomic`|5.3.0| | | | |
|`CUSPARSE_SPMV_CSR_ALG1`|11.2| | | |`rocsparse_spmv_alg_csr_adaptive`|4.1.0| | | | |
|`CUSPARSE_SPMV_CSR_ALG2`|11.2| | | |`rocsparse_spmv_alg_csr_stream`|4.1.0| | | | |
|`CUSPARSE_SPMV_SELL_ALG1`|12.1| | | |`rocsparse_spmv_alg_ell`|4.1.0| | | | |
|`CUSPARSE_SPSM_ALG_DEFAULT`|11.3| | | |`rocsparse_spsm_alg_default`|4.5.0| | | | |
|`CUSPARSE_SPSV_ALG_DEFAULT`|11.3| | | |`rocsparse_spsv_alg_default`|4.5.0| | | | |
|`CUSPARSE_SPSV_UPDATE_DIAGONAL`|12.1| | | | | | | | | |
|`CUSPARSE_SPSV_UPDATE_GENERAL`|12.1| | | | | | | | | |
|`CUSPARSE_STATUS_ALLOC_FAILED`| | | | |`rocsparse_status_memory_error`|1.9.0| | | | |
|`CUSPARSE_STATUS_ARCH_MISMATCH`| | | | |`rocsparse_status_arch_mismatch`|1.9.0| | | | |
|`CUSPARSE_STATUS_EXECUTION_FAILED`| | | | | | | | | | |
|`CUSPARSE_STATUS_INSUFFICIENT_RESOURCES`|11.0| | | | | | | | | |
|`CUSPARSE_STATUS_INTERNAL_ERROR`| | | | |`rocsparse_status_internal_error`|1.9.0| | | | |
|`CUSPARSE_STATUS_INVALID_VALUE`| | | | |`rocsparse_status_invalid_value`|1.9.0| | | | |
|`CUSPARSE_STATUS_MAPPING_ERROR`| | | | | | | | | | |
|`CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED`| | | | | | | | | | |
|`CUSPARSE_STATUS_NOT_INITIALIZED`| | | | |`rocsparse_status_not_initialized`|4.1.0| | | | |
|`CUSPARSE_STATUS_NOT_SUPPORTED`|10.2| | | |`rocsparse_status_not_implemented`|1.9.0| | | | |
|`CUSPARSE_STATUS_SUCCESS`| | | | |`rocsparse_status_success`|1.9.0| | | | |
|`CUSPARSE_STATUS_ZERO_PIVOT`| | | | |`rocsparse_status_zero_pivot`|1.9.0| | | | |
|`bsric02Info`| | | | |`_rocsparse_mat_info`|1.9.0| | | | |
|`bsric02Info_t`| | | | |`rocsparse_mat_info`|1.9.0| | | | |
|`bsrilu02Info`| |12.2| | |`_rocsparse_mat_info`|1.9.0| | | | |
|`bsrilu02Info_t`| |12.2| | |`rocsparse_mat_info`|1.9.0| | | | |
|`bsrsm2Info`| |12.2| | |`_rocsparse_mat_info`|1.9.0| | | | |
|`bsrsm2Info_t`| |12.2| | |`rocsparse_mat_info`|1.9.0| | | | |
|`bsrsv2Info`| |12.2| | |`_rocsparse_mat_info`|1.9.0| | | | |
|`bsrsv2Info_t`| |12.2| | |`rocsparse_mat_info`|1.9.0| | | | |
|`csrgemm2Info`| | | |12.0|`_rocsparse_mat_info`|1.9.0| | | | |
|`csrgemm2Info_t`| | | |12.0|`rocsparse_mat_info`|1.9.0| | | | |
|`csric02Info`| |12.2| | |`_rocsparse_mat_info`|1.9.0| | | | |
|`csric02Info_t`| |12.2| | |`rocsparse_mat_info`|1.9.0| | | | |
|`csrilu02Info`| |12.2| | |`_rocsparse_mat_info`|1.9.0| | | | |
|`csrilu02Info_t`| |12.2| | |`rocsparse_mat_info`|1.9.0| | | | |
|`csrsm2Info`|9.2| | |12.0|`_rocsparse_mat_info`|1.9.0| | | | |
|`csrsm2Info_t`|9.2| | |12.0|`rocsparse_mat_info`|1.9.0| | | | |
|`csrsv2Info`| | | |12.0|`_rocsparse_mat_descr`|1.9.0| | | | |
|`csrsv2Info_t`| | | |12.0|`rocsparse_mat_descr`|1.9.0| | | | |
|`csru2csrInfo`| |12.2| | | | | | | | |
|`csru2csrInfo_t`| |12.2| | | | | | | | |
|`cusparseAction_t`| | | | |`rocsparse_action`|1.9.0| | | | |
|`cusparseAlgMode_t`|8.0| | |12.0| | | | | | |
|`cusparseColorAlg_t`|8.0|12.2| | | | | | | | |
|`cusparseColorInfo`| |12.2| | |`_rocsparse_color_info`|4.5.0| | | | |
|`cusparseColorInfo_t`| |12.2| | |`rocsparse_color_info`|4.5.0| | | | |
|`cusparseConstDnMatDescr_t`|12.0| | | |`rocsparse_const_dnmat_descr`|6.0.0| | | | |
|`cusparseConstDnVecDescr_t`|12.0| | | |`rocsparse_const_dnvec_descr`|6.0.0| | | | |
|`cusparseConstSpMatDescr_t`|12.0| | | |`rocsparse_const_spmat_descr`|6.0.0| | | | |
|`cusparseConstSpVecDescr_t`|12.0| | | |`rocsparse_const_spvec_descr`|6.0.0| | | | |
|`cusparseContext`| | | | |`_rocsparse_handle`|1.9.0| | | | |
|`cusparseCsr2CscAlg_t`|10.1| | | | | | | | | |
|`cusparseDenseToSparseAlg_t`|11.1| | | |`rocsparse_dense_to_sparse_alg`|4.1.0| | | | |
|`cusparseDiagType_t`| | | | |`rocsparse_diag_type`|1.9.0| | | | |
|`cusparseDirection_t`| | | | |`rocsparse_direction`|3.1.0| | | | |
|`cusparseDnMatDescr`|10.1| | | |`_rocsparse_dnmat_descr`|4.1.0| | | | |
|`cusparseDnMatDescr_t`|10.1| | | |`rocsparse_dnmat_descr`|4.1.0| | | | |
|`cusparseDnVecDescr`|10.2| | | |`_rocsparse_dnvec_descr`|4.1.0| | | | |
|`cusparseDnVecDescr_t`|10.2| | | |`rocsparse_dnvec_descr`|4.1.0| | | | |
|`cusparseFillMode_t`| | | | |`rocsparse_fill_mode`|1.9.0| | | | |
|`cusparseFormat_t`|10.1| | | |`rocsparse_format`|4.1.0| | | | |
|`cusparseHandle_t`| | | | |`rocsparse_handle`|1.9.0| | | | |
|`cusparseHybMat`| |10.2| |11.0|`_rocsparse_hyb_mat`|1.9.0| | | | |
|`cusparseHybMat_t`| |10.2| |11.0|`rocsparse_hyb_mat`|1.9.0| | | | |
|`cusparseHybPartition_t`| |10.2| |11.0|`rocsparse_hyb_partition`|1.9.0| | | | |
|`cusparseIndexBase_t`| | | | |`rocsparse_index_base`|1.9.0| | | | |
|`cusparseIndexType_t`|10.1| | | |`rocsparse_indextype`|4.1.0| | | | |
|`cusparseLoggerCallback_t`|11.5| | | | | | | | | |
|`cusparseMatDescr`| | | | |`_rocsparse_mat_descr`|1.9.0| | | | |
|`cusparseMatDescr_t`| | | | |`rocsparse_mat_descr`|1.9.0| | | | |
|`cusparseMatrixType_t`| | | | |`rocsparse_matrix_type`|1.9.0| | | | |
|`cusparseOperation_t`| | | | |`rocsparse_operation`|1.9.0| | | | |
|`cusparseOrder_t`|10.1| | | |`rocsparse_order`|4.1.0| | | | |
|`cusparsePointerMode_t`| | | | |`rocsparse_pointer_mode`|1.9.0| | | | |
|`cusparseSDDMMAlg_t`|11.2| | | |`rocsparse_sddmm_alg`|4.3.0| | | | |
|`cusparseSideMode_t`| | | |11.5| | | | | | |
|`cusparseSolveAnalysisInfo`| |10.2| |11.0| | | | | | |
|`cusparseSolveAnalysisInfo_t`| |10.2| |11.0| | | | | | |
|`cusparseSolvePolicy_t`| |12.2| | |`rocsparse_solve_policy`|1.9.0| | | | |
|`cusparseSpGEMMAlg_t`|11.0| | | |`rocsparse_spgemm_alg`|4.1.0| | | | |
|`cusparseSpGEMMDescr`|11.0| | | | | | | | | |
|`cusparseSpGEMMDescr_t`|11.0| | | | | | | | | |
|`cusparseSpMMAlg_t`|10.1| | | |`rocsparse_spmm_alg`|4.2.0| | | | |
|`cusparseSpMMOpAlg_t`|11.5| | | | | | | | | |
|`cusparseSpMMOpPlan`|11.5| | | | | | | | | |
|`cusparseSpMMOpPlan_t`|11.5| | | | | | | | | |
|`cusparseSpMVAlg_t`|10.2| | | |`rocsparse_spmv_alg`|4.1.0| | | | |
|`cusparseSpMatAttribute_t`|11.3| | | |`rocsparse_spmat_attribute`|4.5.0| | | | |
|`cusparseSpMatDescr`|10.1| | | |`_rocsparse_spmat_descr`|4.1.0| | | | |
|`cusparseSpMatDescr_t`|10.1| | | |`rocsparse_spmat_descr`|4.1.0| | | | |
|`cusparseSpSMAlg_t`|11.3| | | |`rocsparse_spsm_alg`|4.5.0| | | | |
|`cusparseSpSMDescr`|11.3| | | | | | | | | |
|`cusparseSpSMDescr_t`|11.3| | | | | | | | | |
|`cusparseSpSVAlg_t`|11.3| | | |`rocsparse_spsv_alg`|4.5.0| | | | |
|`cusparseSpSVDescr`|11.3| | | | | | | | | |
|`cusparseSpSVDescr_t`|11.3| | | | | | | | | |
|`cusparseSpSVUpdate_t`|12.1| | | | | | | | | |
|`cusparseSpVecDescr`|10.2| | | |`_rocsparse_spvec_descr`|4.1.0| | | | |
|`cusparseSpVecDescr_t`|10.2| | | |`rocsparse_spvec_descr`|4.1.0| | | | |
|`cusparseSparseToDenseAlg_t`|11.1| | | |`rocsparse_sparse_to_dense_alg`|4.1.0| | | | |
|`cusparseStatus_t`| | | | |`rocsparse_status`|1.9.0| | | | |
|`pruneInfo`|9.0|12.2| | |`_rocsparse_mat_info`|1.9.0| | | | |
|`pruneInfo_t`|9.0|12.2| | |`rocsparse_mat_info`|1.9.0| | | | |

## **5. CUSPARSE Management Function Reference**

|**CUDA**|**A**|**D**|**C**|**R**|**ROC**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cusparseCreate`| | | | |`rocsparse_create_handle`|1.9.0| | | | |
|`cusparseDestroy`| | | | |`rocsparse_destroy_handle`|1.9.0| | | | |
|`cusparseGetErrorName`|10.2| | | |`rocsparse_get_status_name`|6.0.0| | | | |
|`cusparseGetErrorString`|10.2| | | |`rocsparse_get_status_description`|6.0.0| | | | |
|`cusparseGetPointerMode`| | | | |`rocsparse_get_pointer_mode`|1.9.0| | | | |
|`cusparseGetStream`|8.0| | | |`rocsparse_get_stream`|1.9.0| | | | |
|`cusparseGetVersion`| | | | |`rocsparse_get_version`|1.9.0| | | | |
|`cusparseSetPointerMode`| | | | |`rocsparse_set_pointer_mode`|1.9.0| | | | |
|`cusparseSetStream`| | | | |`rocsparse_set_stream`|1.9.0| | | | |

## **6. CUSPARSE Logging**

|**CUDA**|**A**|**D**|**C**|**R**|**ROC**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cusparseLoggerForceDisable`|11.5| | | | | | | | | |
|`cusparseLoggerOpenFile`|11.5| | | | | | | | | |
|`cusparseLoggerSetCallback`|11.5| | | | | | | | | |
|`cusparseLoggerSetFile`|11.5| | | | | | | | | |
|`cusparseLoggerSetLevel`|11.5| | | | | | | | | |
|`cusparseLoggerSetMask`|11.5| | | | | | | | | |

## **7. CUSPARSE Helper Function Reference**

|**CUDA**|**A**|**D**|**C**|**R**|**ROC**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cusparseCopyMatDescr`|8.0| | |12.0|`rocsparse_copy_mat_descr`|1.9.0| | | | |
|`cusparseCreateBsric02Info`| |12.2| | |`rocsparse_create_mat_info`|1.9.0| | | | |
|`cusparseCreateBsrilu02Info`| |12.2| | |`rocsparse_create_mat_info`|1.9.0| | | | |
|`cusparseCreateBsrsm2Info`| |12.2| | |`rocsparse_create_mat_info`|1.9.0| | | | |
|`cusparseCreateBsrsv2Info`| |12.2| | |`rocsparse_create_mat_info`|1.9.0| | | | |
|`cusparseCreateColorInfo`| |12.2| | |`rocsparse_create_color_info`|4.5.0| | | | |
|`cusparseCreateCsrgemm2Info`| |11.0| |12.0|`rocsparse_create_mat_info`|1.9.0| | | | |
|`cusparseCreateCsric02Info`| |12.2| | |`rocsparse_create_mat_info`|1.9.0| | | | |
|`cusparseCreateCsrilu02Info`| |12.2| | |`rocsparse_create_mat_info`|1.9.0| | | | |
|`cusparseCreateCsrsm2Info`|9.2|11.3| |12.0|`rocsparse_create_mat_info`|1.9.0| | | | |
|`cusparseCreateCsrsv2Info`| |11.3| |12.0|`rocsparse_create_mat_info`|1.9.0| | | | |
|`cusparseCreateHybMat`| |10.2| |11.0|`rocsparse_create_hyb_mat`|1.9.0| | | | |
|`cusparseCreateMatDescr`| | | | |`rocsparse_create_mat_descr`|1.9.0| | | | |
|`cusparseCreatePruneInfo`|9.0|12.2| | |`rocsparse_create_mat_info`|1.9.0| | | | |
|`cusparseCreateSolveAnalysisInfo`| |10.2| |11.0| | | | | | |
|`cusparseDestroyBsric02Info`| |12.2| | |`rocsparse_destroy_mat_info`|1.9.0| | | | |
|`cusparseDestroyBsrilu02Info`| |12.2| | |`rocsparse_destroy_mat_info`|1.9.0| | | | |
|`cusparseDestroyBsrsm2Info`| |12.2| | |`rocsparse_destroy_mat_info`|1.9.0| | | | |
|`cusparseDestroyBsrsv2Info`| |12.2| | |`rocsparse_destroy_mat_info`|1.9.0| | | | |
|`cusparseDestroyColorInfo`| |12.2| | |`rocsparse_destroy_color_info`|4.5.0| | | | |
|`cusparseDestroyCsrgemm2Info`| |11.0| |12.0|`rocsparse_destroy_mat_info`|1.9.0| | | | |
|`cusparseDestroyCsric02Info`| |12.2| | |`rocsparse_destroy_mat_info`|1.9.0| | | | |
|`cusparseDestroyCsrilu02Info`| |12.2| | |`rocsparse_destroy_mat_info`|1.9.0| | | | |
|`cusparseDestroyCsrsm2Info`|9.2|11.3| |12.0|`rocsparse_destroy_mat_info`|1.9.0| | | | |
|`cusparseDestroyCsrsv2Info`| |11.3| |12.0|`rocsparse_destroy_mat_info`|1.9.0| | | | |
|`cusparseDestroyHybMat`| |10.2| |11.0|`rocsparse_destroy_hyb_mat`|1.9.0| | | | |
|`cusparseDestroyMatDescr`| | | | |`rocsparse_destroy_mat_descr`|1.9.0| | | | |
|`cusparseDestroyPruneInfo`|9.0|12.2| | |`rocsparse_destroy_mat_info`|1.9.0| | | | |
|`cusparseDestroySolveAnalysisInfo`| |10.2| |11.0| | | | | | |
|`cusparseGetLevelInfo`| | | |11.0| | | | | | |
|`cusparseGetMatDiagType`| | | | |`rocsparse_get_mat_diag_type`|1.9.0| | | | |
|`cusparseGetMatFillMode`| | | | |`rocsparse_get_mat_fill_mode`|1.9.0| | | | |
|`cusparseGetMatIndexBase`| | | | |`rocsparse_get_mat_index_base`|1.9.0| | | | |
|`cusparseGetMatType`| | | | |`rocsparse_get_mat_type`|1.9.0| | | | |
|`cusparseSetMatDiagType`| | | | |`rocsparse_set_mat_diag_type`|1.9.0| | | | |
|`cusparseSetMatFillMode`| | | | |`rocsparse_set_mat_fill_mode`|1.9.0| | | | |
|`cusparseSetMatIndexBase`| | | | |`rocsparse_set_mat_index_base`|1.9.0| | | | |
|`cusparseSetMatType`| | | | |`rocsparse_set_mat_type`|1.9.0| | | | |

## **8. CUSPARSE Level 1 Function Reference**

|**CUDA**|**A**|**D**|**C**|**R**|**ROC**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cusparseCaxpyi`| |11.0| |12.0|`rocsparse_caxpyi`|1.9.0| | | | |
|`cusparseCdotci`| |10.2| |11.0|`rocsparse_cdotci`|3.0.0| | | | |
|`cusparseCdoti`| |10.2| |11.0|`rocsparse_cdoti`|1.9.0| | | | |
|`cusparseCgthr`| |11.0| |12.0|`rocsparse_cgthr`|1.9.0| | | | |
|`cusparseCgthrz`| |11.0| |12.0|`rocsparse_cgthrz`|1.9.0| | | | |
|`cusparseCsctr`| |11.0| |12.0|`rocsparse_csctr`|1.9.0| | | | |
|`cusparseDaxpyi`| |11.0| |12.0|`rocsparse_daxpyi`|1.9.0| | | | |
|`cusparseDdoti`| |10.2| |11.0|`rocsparse_ddoti`|1.9.0| | | | |
|`cusparseDgthr`| |11.0| |12.0|`rocsparse_dgthr`|1.9.0| | | | |
|`cusparseDgthrz`| |11.0| |12.0|`rocsparse_dgthrz`|1.9.0| | | | |
|`cusparseDroti`| |11.0| |12.0|`rocsparse_droti`|1.9.0| | | | |
|`cusparseDsctr`| |11.0| |12.0|`rocsparse_dsctr`|1.9.0| | | | |
|`cusparseSaxpyi`| |11.0| |12.0|`rocsparse_saxpyi`|1.9.0| | | | |
|`cusparseSdoti`| |10.2| |11.0|`rocsparse_sdoti`|1.9.0| | | | |
|`cusparseSgthr`| |11.0| |12.0|`rocsparse_sgthr`|1.9.0| | | | |
|`cusparseSgthrz`| |11.0| |12.0|`rocsparse_sgthrz`|1.9.0| | | | |
|`cusparseSroti`| |11.0| |12.0|`rocsparse_sroti`|1.9.0| | | | |
|`cusparseSsctr`| |11.0| |12.0|`rocsparse_ssctr`|1.9.0| | | | |
|`cusparseZaxpyi`| |11.0| |12.0|`rocsparse_zaxpyi`|1.9.0| | | | |
|`cusparseZdotci`| |10.2| |11.0|`rocsparse_zdotci`|3.0.0| | | | |
|`cusparseZdoti`| |10.2| |11.0|`rocsparse_zdoti`|1.9.0| | | | |
|`cusparseZgthr`| |11.0| |12.0|`rocsparse_zgthr`|1.9.0| | | | |
|`cusparseZgthrz`| |11.0| |12.0|`rocsparse_zgthrz`|1.9.0| | | | |
|`cusparseZsctr`| |11.0| |12.0|`rocsparse_zsctr`|1.9.0| | | | |

## **9. CUSPARSE Level 2 Function Reference**

|**CUDA**|**A**|**D**|**C**|**R**|**ROC**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cusparseCbsrmv`| | | | |`rocsparse_cbsrmv`|3.5.0|5.4.0| | | |
|`cusparseCbsrsv2_analysis`| |12.2| | |`rocsparse_cbsrsv_analysis`|3.6.0| | | | |
|`cusparseCbsrsv2_bufferSize`| |12.2| | |`rocsparse_cbsrsv_buffer_size`|3.6.0| | | | |
|`cusparseCbsrsv2_bufferSizeExt`| |12.2| | |`rocsparse_cbsrsv_buffer_size`|3.6.0| | | | |
|`cusparseCbsrsv2_solve`| |12.2| | |`rocsparse_cbsrsv_solve`|3.6.0| | | | |
|`cusparseCbsrxmv`| |12.2| | |`rocsparse_cbsrxmv`|4.5.0| | | | |
|`cusparseCcsrmv`| |10.2| |11.0|`rocsparse_ccsrmv`|1.9.0| | | | |
|`cusparseCcsrmv_mp`|8.0|10.2| |11.0| | | | | | |
|`cusparseCcsrsv2_analysis`| |11.3| |12.0|`rocsparse_ccsrsv_analysis`|2.10.0| | | | |
|`cusparseCcsrsv2_bufferSize`| |11.3| |12.0|`rocsparse_ccsrsv_buffer_size`|2.10.0| | | | |
|`cusparseCcsrsv2_bufferSizeExt`| |11.3| |12.0|`rocsparse_ccsrsv_buffer_size`|2.10.0| | | | |
|`cusparseCcsrsv2_solve`| |11.3| |12.0|`rocsparse_ccsrsv_solve`|2.10.0| | | | |
|`cusparseCcsrsv_analysis`| |10.2| |11.0| | | | | | |
|`cusparseCcsrsv_solve`| |10.2| |11.0| | | | | | |
|`cusparseCgemvi`|7.5| | | |`rocsparse_cgemvi`|4.3.0| | | | |
|`cusparseCgemvi_bufferSize`|7.5| | | |`rocsparse_cgemvi_buffer_size`|4.3.0| | | | |
|`cusparseChybmv`| |10.2| |11.0|`rocsparse_chybmv`|2.10.0| | | | |
|`cusparseChybsv_analysis`| |10.2| |11.0| | | | | | |
|`cusparseChybsv_solve`| |10.2| |11.0| | | | | | |
|`cusparseCsrmvEx`|8.0|11.2| |12.0| | | | | | |
|`cusparseCsrmvEx_bufferSize`|8.0|11.2| |12.0| | | | | | |
|`cusparseCsrsv_analysisEx`|8.0|10.2| |11.0| | | | | | |
|`cusparseCsrsv_solveEx`|8.0|10.2| |11.0| | | | | | |
|`cusparseDbsrmv`| | | | |`rocsparse_dbsrmv`|3.5.0|5.4.0| | | |
|`cusparseDbsrsv2_analysis`| |12.2| | |`rocsparse_dbsrsv_analysis`|3.6.0| | | | |
|`cusparseDbsrsv2_bufferSize`| |12.2| | |`rocsparse_dbsrsv_buffer_size`|3.6.0| | | | |
|`cusparseDbsrsv2_bufferSizeExt`| |12.2| | |`rocsparse_dbsrsv_buffer_size`|3.6.0| | | | |
|`cusparseDbsrsv2_solve`| |12.2| | |`rocsparse_dbsrsv_solve`|3.6.0| | | | |
|`cusparseDbsrxmv`| |12.2| | |`rocsparse_dbsrxmv`|4.5.0| | | | |
|`cusparseDcsrmv`| |10.2| |11.0|`rocsparse_dcsrmv`|1.9.0| | | | |
|`cusparseDcsrmv_mp`|8.0|10.2| |11.0| | | | | | |
|`cusparseDcsrsv2_analysis`| |11.3| |12.0|`rocsparse_dcsrsv_analysis`|1.9.0| | | | |
|`cusparseDcsrsv2_bufferSize`| |11.3| |12.0|`rocsparse_dcsrsv_buffer_size`|1.9.0| | | | |
|`cusparseDcsrsv2_bufferSizeExt`| |11.3| |12.0|`rocsparse_dcsrsv_buffer_size`|1.9.0| | | | |
|`cusparseDcsrsv2_solve`| |11.3| |12.0|`rocsparse_dcsrsv_solve`|1.9.0| | | | |
|`cusparseDcsrsv_analysis`| |10.2| |11.0| | | | | | |
|`cusparseDcsrsv_solve`| |10.2| |11.0| | | | | | |
|`cusparseDgemvi`|7.5| | | |`rocsparse_dgemvi`|4.3.0| | | | |
|`cusparseDgemvi_bufferSize`|7.5| | | |`rocsparse_dgemvi_buffer_size`|4.3.0| | | | |
|`cusparseDhybmv`| |10.2| |11.0|`rocsparse_dhybmv`|1.9.0| | | | |
|`cusparseDhybsv_analysis`| |10.2| |11.0| | | | | | |
|`cusparseDhybsv_solve`| |10.2| |11.0| | | | | | |
|`cusparseSbsrmv`| | | | |`rocsparse_sbsrmv`|3.5.0|5.4.0| | | |
|`cusparseSbsrsv2_analysis`| |12.2| | |`rocsparse_sbsrsv_analysis`|3.6.0| | | | |
|`cusparseSbsrsv2_bufferSize`| |12.2| | |`rocsparse_sbsrsv_buffer_size`|3.6.0| | | | |
|`cusparseSbsrsv2_bufferSizeExt`| |12.2| | |`rocsparse_sbsrsv_buffer_size`|3.6.0| | | | |
|`cusparseSbsrsv2_solve`| |12.2| | |`rocsparse_sbsrsv_solve`|3.6.0| | | | |
|`cusparseSbsrxmv`| |12.2| | |`rocsparse_sbsrxmv`|4.5.0| | | | |
|`cusparseScsrmv`| |10.2| |11.0|`rocsparse_scsrmv`|1.9.0| | | | |
|`cusparseScsrmv_mp`|8.0|10.2| |11.0| | | | | | |
|`cusparseScsrsv2_analysis`| |11.3| |12.0|`rocsparse_scsrsv_analysis`|1.9.0| | | | |
|`cusparseScsrsv2_bufferSize`| |11.3| |12.0|`rocsparse_scsrsv_buffer_size`|1.9.0| | | | |
|`cusparseScsrsv2_bufferSizeExt`| |11.3| |12.0|`rocsparse_scsrsv_buffer_size`|1.9.0| | | | |
|`cusparseScsrsv2_solve`| |11.3| |12.0|`rocsparse_scsrsv_solve`|1.9.0| | | | |
|`cusparseScsrsv_analysis`| |10.2| |11.0| | | | | | |
|`cusparseScsrsv_solve`| |10.2| |11.0| | | | | | |
|`cusparseSgemvi`|7.5| | | |`rocsparse_sgemvi`|4.3.0| | | | |
|`cusparseSgemvi_bufferSize`|7.5| | | |`rocsparse_sgemvi_buffer_size`|4.3.0| | | | |
|`cusparseShybmv`| |10.2| |11.0|`rocsparse_shybmv`|1.9.0| | | | |
|`cusparseShybsv_analysis`| |10.2| |11.0| | | | | | |
|`cusparseShybsv_solve`| |10.2| |11.0| | | | | | |
|`cusparseXbsrsv2_zeroPivot`| |12.2| | |`rocsparse_bsrsv_zero_pivot`|3.6.0| | | | |
|`cusparseXcsrsv2_zeroPivot`| |11.3| |12.0|`rocsparse_csrsv_zero_pivot`|1.9.0| | | | |
|`cusparseZbsrmv`| | | | |`rocsparse_zbsrmv`|3.5.0|5.4.0| | | |
|`cusparseZbsrsv2_analysis`| |12.2| | |`rocsparse_zbsrsv_analysis`|3.6.0| | | | |
|`cusparseZbsrsv2_bufferSize`| |12.2| | |`rocsparse_zbsrsv_buffer_size`|3.6.0| | | | |
|`cusparseZbsrsv2_bufferSizeExt`| |12.2| | |`rocsparse_zbsrsv_buffer_size`|3.6.0| | | | |
|`cusparseZbsrsv2_solve`| |12.2| | |`rocsparse_zbsrsv_solve`|3.6.0| | | | |
|`cusparseZbsrxmv`| |12.2| | |`rocsparse_zbsrxmv`|4.5.0| | | | |
|`cusparseZcsrmv`| |10.2| |11.0|`rocsparse_zcsrmv`|1.9.0| | | | |
|`cusparseZcsrmv_mp`|8.0|10.2| |11.0| | | | | | |
|`cusparseZcsrsv2_analysis`| |11.3| |12.0|`rocsparse_zcsrsv_analysis`|2.10.0| | | | |
|`cusparseZcsrsv2_bufferSize`| |11.3| |12.0|`rocsparse_zcsrsv_buffer_size`|2.10.0| | | | |
|`cusparseZcsrsv2_bufferSizeExt`| |11.3| |12.0|`rocsparse_zcsrsv_buffer_size`|2.10.0| | | | |
|`cusparseZcsrsv2_solve`| |11.3| |12.0|`rocsparse_zcsrsv_solve`|2.10.0| | | | |
|`cusparseZcsrsv_analysis`| |10.2| |11.0| | | | | | |
|`cusparseZcsrsv_solve`| |10.2| |11.0| | | | | | |
|`cusparseZgemvi`|7.5| | | |`rocsparse_zgemvi`|4.3.0| | | | |
|`cusparseZgemvi_bufferSize`|7.5| | | |`rocsparse_zgemvi_buffer_size`|4.3.0| | | | |
|`cusparseZhybmv`| |10.2| |11.0|`rocsparse_zhybmv`|2.10.0| | | | |
|`cusparseZhybsv_analysis`| |10.2| |11.0| | | | | | |
|`cusparseZhybsv_solve`| |10.2| |11.0| | | | | | |

## **10. CUSPARSE Level 3 Function Reference**

|**CUDA**|**A**|**D**|**C**|**R**|**ROC**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cusparseCbsrmm`| | | | |`rocsparse_cbsrmm`|3.7.0| | | | |
|`cusparseCbsrsm2_analysis`| |12.2| | |`rocsparse_cbsrsm_analysis`|3.6.0| | | | |
|`cusparseCbsrsm2_bufferSize`| |12.2| | |`rocsparse_cbsrsm_buffer_size`|4.5.0| | | | |
|`cusparseCbsrsm2_bufferSizeExt`| |12.2| | | | | | | | |
|`cusparseCbsrsm2_solve`| |12.2| | |`rocsparse_cbsrsm_solve`|4.5.0| | | | |
|`cusparseCcsrmm`| |10.2| |11.0|`rocsparse_ccsrmm`|1.9.0| | | | |
|`cusparseCcsrmm2`| |10.2| |11.0|`rocsparse_ccsrmm`|1.9.0| | | | |
|`cusparseCcsrsm2_analysis`|9.2|11.3| |12.0|`rocsparse_ccsrsm_analysis`|3.1.0| | | | |
|`cusparseCcsrsm2_bufferSizeExt`|9.2|11.3| |12.0|`rocsparse_ccsrsm_buffer_size`|3.1.0| | | | |
|`cusparseCcsrsm2_solve`|9.2|11.3| |12.0|`rocsparse_ccsrsm_solve`|3.1.0| | | | |
|`cusparseCcsrsm_analysis`| |10.2| |11.0| | | | | | |
|`cusparseCcsrsm_solve`| |10.2| |11.0| | | | | | |
|`cusparseCgemmi`|8.0|11.0| |12.0| | | | | | |
|`cusparseDbsrmm`| | | | |`rocsparse_dbsrmm`|3.7.0| | | | |
|`cusparseDbsrsm2_analysis`| |12.2| | |`rocsparse_dbsrsm_analysis`|3.6.0| | | | |
|`cusparseDbsrsm2_bufferSize`| |12.2| | |`rocsparse_dbsrsm_buffer_size`|4.5.0| | | | |
|`cusparseDbsrsm2_bufferSizeExt`| |12.2| | | | | | | | |
|`cusparseDbsrsm2_solve`| |12.2| | |`rocsparse_dbsrsm_solve`|4.5.0| | | | |
|`cusparseDcsrmm`| |10.2| |11.0|`rocsparse_dcsrmm`|1.9.0| | | | |
|`cusparseDcsrmm2`| |10.2| |11.0|`rocsparse_dcsrmm`|1.9.0| | | | |
|`cusparseDcsrsm2_analysis`|9.2|11.3| |12.0|`rocsparse_dcsrsm_analysis`|3.1.0| | | | |
|`cusparseDcsrsm2_bufferSizeExt`|9.2|11.3| |12.0|`rocsparse_dcsrsm_buffer_size`|3.1.0| | | | |
|`cusparseDcsrsm2_solve`|9.2|11.3| |12.0|`rocsparse_dcsrsm_solve`|3.1.0| | | | |
|`cusparseDcsrsm_analysis`| |10.2| |11.0| | | | | | |
|`cusparseDcsrsm_solve`| |10.2| |11.0| | | | | | |
|`cusparseDgemmi`|8.0|11.0| |12.0| | | | | | |
|`cusparseSbsrmm`| | | | |`rocsparse_sbsrmm`|3.7.0| | | | |
|`cusparseSbsrsm2_analysis`| |12.2| | |`rocsparse_sbsrsm_analysis`|3.6.0| | | | |
|`cusparseSbsrsm2_bufferSize`| |12.2| | |`rocsparse_sbsrsm_buffer_size`|4.5.0| | | | |
|`cusparseSbsrsm2_bufferSizeExt`| |12.2| | | | | | | | |
|`cusparseSbsrsm2_solve`| |12.2| | |`rocsparse_sbsrsm_solve`|4.5.0| | | | |
|`cusparseScsrmm`| |10.2| |11.0|`rocsparse_scsrmm`|1.9.0| | | | |
|`cusparseScsrmm2`| |10.2| |11.0|`rocsparse_scsrmm`|1.9.0| | | | |
|`cusparseScsrsm2_analysis`|9.2|11.3| |12.0|`rocsparse_scsrsm_analysis`|3.1.0| | | | |
|`cusparseScsrsm2_bufferSizeExt`|9.2|11.3| |12.0|`rocsparse_scsrsm_buffer_size`|3.1.0| | | | |
|`cusparseScsrsm2_solve`|9.2|11.3| |12.0|`rocsparse_scsrsm_solve`|3.1.0| | | | |
|`cusparseScsrsm_analysis`| |10.2| |11.0| | | | | | |
|`cusparseScsrsm_solve`| |10.2| |11.0| | | | | | |
|`cusparseSgemmi`|8.0|11.0| |12.0| | | | | | |
|`cusparseXbsrsm2_zeroPivot`| |12.2| | |`rocsparse_bsrsm_zero_pivot`|4.5.0| | | | |
|`cusparseXcsrsm2_zeroPivot`|9.2|11.3| |12.0|`rocsparse_csrsm_zero_pivot`|3.1.0| | | | |
|`cusparseZbsrmm`| | | | |`rocsparse_zbsrmm`|3.7.0| | | | |
|`cusparseZbsrsm2_analysis`| |12.2| | |`rocsparse_zbsrsm_analysis`|3.6.0| | | | |
|`cusparseZbsrsm2_bufferSize`| |12.2| | |`rocsparse_zbsrsm_buffer_size`|4.5.0| | | | |
|`cusparseZbsrsm2_bufferSizeExt`| |12.2| | | | | | | | |
|`cusparseZbsrsm2_solve`| |12.2| | |`rocsparse_zbsrsm_solve`|4.5.0| | | | |
|`cusparseZcsrmm`| |10.2| |11.0|`rocsparse_zcsrmm`|1.9.0| | | | |
|`cusparseZcsrmm2`| |10.2| |11.0|`rocsparse_zcsrmm`|1.9.0| | | | |
|`cusparseZcsrsm2_analysis`|9.2|11.3| |12.0|`rocsparse_zcsrsm_analysis`|3.1.0| | | | |
|`cusparseZcsrsm2_bufferSizeExt`|9.2|11.3| |12.0|`rocsparse_zcsrsm_buffer_size`|3.1.0| | | | |
|`cusparseZcsrsm2_solve`|9.2|11.3| |12.0|`rocsparse_zcsrsm_solve`|3.1.0| | | | |
|`cusparseZcsrsm_analysis`| |10.2| |11.0| | | | | | |
|`cusparseZcsrsm_solve`| |10.2| |11.0| | | | | | |
|`cusparseZgemmi`|8.0|11.0| |12.0| | | | | | |

## **11. CUSPARSE Extra Function Reference**

|**CUDA**|**A**|**D**|**C**|**R**|**ROC**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cusparseCcsrgeam`| |10.2| |11.0|`rocsparse_ccsrgeam`|3.5.0| | | | |
|`cusparseCcsrgeam2`|10.0| | | |`rocsparse_ccsrgeam`|3.5.0| | | | |
|`cusparseCcsrgeam2_bufferSizeExt`|10.0| | | | | | | | | |
|`cusparseCcsrgemm`| |10.2| |11.0| | | | | | |
|`cusparseCcsrgemm2`| |11.0| |12.0|`rocsparse_ccsrgemm`|2.8.0| | | | |
|`cusparseCcsrgemm2_bufferSizeExt`| |11.0| |12.0|`rocsparse_ccsrgemm_buffer_size`|2.8.0| | | | |
|`cusparseDcsrgeam`| |10.2| |11.0|`rocsparse_dcsrgeam`|3.5.0| | | | |
|`cusparseDcsrgeam2`|10.0| | | |`rocsparse_dcsrgeam`|3.5.0| | | | |
|`cusparseDcsrgeam2_bufferSizeExt`|10.0| | | | | | | | | |
|`cusparseDcsrgemm`| |10.2| |11.0| | | | | | |
|`cusparseDcsrgemm2`| |11.0| |12.0|`rocsparse_dcsrgemm`|2.8.0| | | | |
|`cusparseDcsrgemm2_bufferSizeExt`| |11.0| |12.0|`rocsparse_dcsrgemm_buffer_size`|2.8.0| | | | |
|`cusparseScsrgeam`| |10.2| |11.0|`rocsparse_scsrgeam`|3.5.0| | | | |
|`cusparseScsrgeam2`|10.0| | | |`rocsparse_scsrgeam`|3.5.0| | | | |
|`cusparseScsrgeam2_bufferSizeExt`|10.0| | | | | | | | | |
|`cusparseScsrgemm`| |10.2| |11.0| | | | | | |
|`cusparseScsrgemm2`| |11.0| |12.0|`rocsparse_scsrgemm`|2.8.0| | | | |
|`cusparseScsrgemm2_bufferSizeExt`| |11.0| |12.0|`rocsparse_scsrgemm_buffer_size`|2.8.0| | | | |
|`cusparseXcsrgeam2Nnz`|10.0| | | |`rocsparse_csrgeam_nnz`|3.5.0| | | | |
|`cusparseXcsrgeamNnz`| |10.2| |11.0|`rocsparse_csrgeam_nnz`|3.5.0| | | | |
|`cusparseXcsrgemm2Nnz`| |11.0| |12.0|`rocsparse_csrgemm_nnz`|2.8.0| | | | |
|`cusparseXcsrgemmNnz`| |10.2| |11.0| | | | | | |
|`cusparseZcsrgeam`| |10.2| |11.0|`rocsparse_zcsrgeam`|3.5.0| | | | |
|`cusparseZcsrgeam2`|10.0| | | |`rocsparse_zcsrgeam`|3.5.0| | | | |
|`cusparseZcsrgeam2_bufferSizeExt`|10.0| | | | | | | | | |
|`cusparseZcsrgemm`| |10.2| |11.0| | | | | | |
|`cusparseZcsrgemm2`| |11.0| |12.0|`rocsparse_zcsrgemm`|2.8.0| | | | |
|`cusparseZcsrgemm2_bufferSizeExt`| |11.0| |12.0|`rocsparse_zcsrgemm_buffer_size`|2.8.0| | | | |

## **12. CUSPARSE Preconditioners Reference**

|**CUDA**|**A**|**D**|**C**|**R**|**ROC**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cusparseCbsric02`| |12.2| | |`rocsparse_cbsric0`|3.8.0| | | | |
|`cusparseCbsric02_analysis`| |12.2| | |`rocsparse_cbsric0_analysis`|3.6.0| | | | |
|`cusparseCbsric02_bufferSize`| |12.2| | |`rocsparse_cbsric0_buffer_size`|3.8.0| | | | |
|`cusparseCbsric02_bufferSizeExt`| |12.2| | | | | | | | |
|`cusparseCbsrilu02`| |12.2| | |`rocsparse_cbsrilu0`|3.9.0| | | | |
|`cusparseCbsrilu02_analysis`| |12.2| | |`rocsparse_cbsrilu0_analysis`|3.6.0| | | | |
|`cusparseCbsrilu02_bufferSize`| |12.2| | |`rocsparse_cbsrilu0_buffer_size`|3.8.0| | | | |
|`cusparseCbsrilu02_bufferSizeExt`| |12.2| | | | | | | | |
|`cusparseCbsrilu02_numericBoost`| |12.2| | |`rocsparse_dcbsrilu0_numeric_boost`|4.5.0| | | | |
|`cusparseCcsric0`| |10.2| |11.0| | | | | | |
|`cusparseCcsric02`| |12.2| | |`rocsparse_ccsric0`|3.1.0| | | | |
|`cusparseCcsric02_analysis`| |12.2| | |`rocsparse_ccsric0_analysis`|3.1.0| | | | |
|`cusparseCcsric02_bufferSize`| |12.2| | |`rocsparse_ccsric0_buffer_size`|3.1.0| | | | |
|`cusparseCcsric02_bufferSizeExt`| |12.2| | |`rocsparse_ccsric0_buffer_size`|3.1.0| | | | |
|`cusparseCcsrilu0`| |10.2| |11.0| | | | | | |
|`cusparseCcsrilu02`| |12.2| | |`rocsparse_ccsrilu0`|2.10.0| | | | |
|`cusparseCcsrilu02_analysis`| |12.2| | |`rocsparse_ccsrilu0_analysis`|2.10.0| | | | |
|`cusparseCcsrilu02_bufferSize`| |12.2| | |`rocsparse_ccsrilu0_buffer_size`|2.10.0| | | | |
|`cusparseCcsrilu02_bufferSizeExt`| |12.2| | |`rocsparse_ccsrilu0_buffer_size`|2.10.0| | | | |
|`cusparseCcsrilu02_numericBoost`| |12.2| | |`rocsparse_dccsrilu0_numeric_boost`|4.5.0| | | | |
|`cusparseCgpsvInterleavedBatch`|9.2| | | |`rocsparse_cgpsv_interleaved_batch`|5.1.0| | | | |
|`cusparseCgpsvInterleavedBatch_bufferSizeExt`|9.2| | | |`rocsparse_cgpsv_interleaved_batch_buffer_size`|5.1.0| | | | |
|`cusparseCgtsv`| |10.2| |11.0| | | | | | |
|`cusparseCgtsv2`|9.0| | | |`rocsparse_cgtsv`|4.3.0| | | | |
|`cusparseCgtsv2StridedBatch`|9.0| | | |`rocsparse_cgtsv_no_pivot_strided_batch`|4.3.0| | | | |
|`cusparseCgtsv2StridedBatch_bufferSizeExt`|9.0| | | |`rocsparse_cgtsv_no_pivot_strided_batch_buffer_size`|4.3.0| | | | |
|`cusparseCgtsv2_bufferSizeExt`|9.0| | | |`rocsparse_cgtsv_buffer_size`|4.3.0| | | | |
|`cusparseCgtsv2_nopivot`|9.0| | | |`rocsparse_cgtsv_no_pivot`|4.3.0| | | | |
|`cusparseCgtsv2_nopivot_bufferSizeExt`|9.0| | | |`rocsparse_cgtsv_no_pivot_buffer_size`|4.3.0| | | | |
|`cusparseCgtsvInterleavedBatch`|9.2| | | |`rocsparse_cgtsv_interleaved_batch`|5.1.0| | | | |
|`cusparseCgtsvInterleavedBatch_bufferSizeExt`|9.2| | | |`rocsparse_cgtsv_interleaved_batch_buffer_size`|5.1.0| | | | |
|`cusparseCgtsvStridedBatch`| |10.2| |11.0| | | | | | |
|`cusparseCgtsv_nopivot`| |10.2| |11.0| | | | | | |
|`cusparseCsrilu0Ex`|8.0|10.2| |11.0| | | | | | |
|`cusparseDbsric02`| |12.2| | |`rocsparse_dbsric0`|3.8.0| | | | |
|`cusparseDbsric02_analysis`| |12.2| | |`rocsparse_dbsric0_analysis`|3.6.0| | | | |
|`cusparseDbsric02_bufferSize`| |12.2| | |`rocsparse_dbsric0_buffer_size`|3.8.0| | | | |
|`cusparseDbsric02_bufferSizeExt`| |12.2| | | | | | | | |
|`cusparseDbsrilu02`| |12.2| | |`rocsparse_dbsrilu0`|3.9.0| | | | |
|`cusparseDbsrilu02_analysis`| |12.2| | |`rocsparse_dbsrilu0_analysis`|3.6.0| | | | |
|`cusparseDbsrilu02_bufferSize`| |12.2| | |`rocsparse_dbsrilu0_buffer_size`|3.8.0| | | | |
|`cusparseDbsrilu02_bufferSizeExt`| |12.2| | | | | | | | |
|`cusparseDbsrilu02_numericBoost`| |12.2| | |`rocsparse_dbsrilu0_numeric_boost`|3.9.0| | | | |
|`cusparseDcsric0`| |10.2| |11.0| | | | | | |
|`cusparseDcsric02`| |12.2| | |`rocsparse_dcsric0`|3.1.0| | | | |
|`cusparseDcsric02_analysis`| |12.2| | |`rocsparse_dcsric0_analysis`|3.1.0| | | | |
|`cusparseDcsric02_bufferSize`| |12.2| | |`rocsparse_dcsric0_buffer_size`|3.1.0| | | | |
|`cusparseDcsric02_bufferSizeExt`| |12.2| | |`rocsparse_dcsric0_buffer_size`|3.1.0| | | | |
|`cusparseDcsrilu0`| |10.2| |11.0| | | | | | |
|`cusparseDcsrilu02`| |12.2| | |`rocsparse_dcsrilu0`|1.9.0| | | | |
|`cusparseDcsrilu02_analysis`| |12.2| | |`rocsparse_dcsrilu0_analysis`|1.9.0| | | | |
|`cusparseDcsrilu02_bufferSize`| |12.2| | |`rocsparse_dcsrilu0_buffer_size`|1.9.0| | | | |
|`cusparseDcsrilu02_bufferSizeExt`| |12.2| | |`rocsparse_dcsrilu0_buffer_size`|1.9.0| | | | |
|`cusparseDcsrilu02_numericBoost`| |12.2| | |`rocsparse_dcsrilu0_numeric_boost`|3.9.0| | | | |
|`cusparseDgpsvInterleavedBatch`|9.2| | | |`rocsparse_dgpsv_interleaved_batch`|5.1.0| | | | |
|`cusparseDgpsvInterleavedBatch_bufferSizeExt`|9.2| | | |`rocsparse_dgpsv_interleaved_batch_buffer_size`|5.1.0| | | | |
|`cusparseDgtsv`| |10.2| |11.0| | | | | | |
|`cusparseDgtsv2`|9.0| | | |`rocsparse_dgtsv`|4.3.0| | | | |
|`cusparseDgtsv2StridedBatch`|9.0| | | |`rocsparse_dgtsv_no_pivot_strided_batch`|4.3.0| | | | |
|`cusparseDgtsv2StridedBatch_bufferSizeExt`|9.0| | | |`rocsparse_dgtsv_no_pivot_strided_batch_buffer_size`|4.3.0| | | | |
|`cusparseDgtsv2_bufferSizeExt`|9.0| | | |`rocsparse_dgtsv_buffer_size`|4.3.0| | | | |
|`cusparseDgtsv2_nopivot`|9.0| | | |`rocsparse_dgtsv_no_pivot`|4.3.0| | | | |
|`cusparseDgtsv2_nopivot_bufferSizeExt`|9.0| | | |`rocsparse_dgtsv_no_pivot_buffer_size`|4.3.0| | | | |
|`cusparseDgtsvInterleavedBatch`|9.2| | | |`rocsparse_dgtsv_interleaved_batch`|5.1.0| | | | |
|`cusparseDgtsvInterleavedBatch_bufferSizeExt`|9.2| | | |`rocsparse_dgtsv_interleaved_batch_buffer_size`|5.1.0| | | | |
|`cusparseDgtsvStridedBatch`| |10.2| |11.0| | | | | | |
|`cusparseDgtsv_nopivot`| |10.2| |11.0| | | | | | |
|`cusparseSbsric02`| |12.2| | |`rocsparse_sbsric0`|3.8.0| | | | |
|`cusparseSbsric02_analysis`| |12.2| | |`rocsparse_sbsric0_analysis`|3.6.0| | | | |
|`cusparseSbsric02_bufferSize`| |12.2| | |`rocsparse_sbsric0_buffer_size`|3.8.0| | | | |
|`cusparseSbsric02_bufferSizeExt`| |12.2| | | | | | | | |
|`cusparseSbsrilu02`| |12.2| | |`rocsparse_sbsrilu0`|3.9.0| | | | |
|`cusparseSbsrilu02_analysis`| |12.2| | |`rocsparse_sbsrilu0_analysis`|3.6.0| | | | |
|`cusparseSbsrilu02_bufferSize`| |12.2| | |`rocsparse_sbsrilu0_buffer_size`|3.8.0| | | | |
|`cusparseSbsrilu02_bufferSizeExt`| |12.2| | | | | | | | |
|`cusparseSbsrilu02_numericBoost`| |12.2| | |`rocsparse_dsbsrilu0_numeric_boost`|4.5.0| | | | |
|`cusparseScsric0`| |10.2| |11.0| | | | | | |
|`cusparseScsric02`| |12.2| | |`rocsparse_scsric0`|3.1.0| | | | |
|`cusparseScsric02_analysis`| |12.2| | |`rocsparse_scsric0_analysis`|3.1.0| | | | |
|`cusparseScsric02_bufferSize`| |12.2| | |`rocsparse_scsric0_buffer_size`|3.1.0| | | | |
|`cusparseScsric02_bufferSizeExt`| |12.2| | |`rocsparse_scsric0_buffer_size`|3.1.0| | | | |
|`cusparseScsrilu0`| |10.2| |11.0| | | | | | |
|`cusparseScsrilu02`| |12.2| | |`rocsparse_scsrilu0`|1.9.0| | | | |
|`cusparseScsrilu02_analysis`| |12.2| | |`rocsparse_scsrilu0_analysis`|1.9.0| | | | |
|`cusparseScsrilu02_bufferSize`| |12.2| | |`rocsparse_scsrilu0_buffer_size`|1.9.0| | | | |
|`cusparseScsrilu02_bufferSizeExt`| |12.2| | |`rocsparse_scsrilu0_buffer_size`|1.9.0| | | | |
|`cusparseScsrilu02_numericBoost`| |12.2| | |`rocsparse_dscsrilu0_numeric_boost`|4.5.0| | | | |
|`cusparseSgpsvInterleavedBatch`|9.2| | | |`rocsparse_sgpsv_interleaved_batch`|5.1.0| | | | |
|`cusparseSgpsvInterleavedBatch_bufferSizeExt`|9.2| | | |`rocsparse_sgpsv_interleaved_batch_buffer_size`|5.1.0| | | | |
|`cusparseSgtsv`| |10.2| |11.0| | | | | | |
|`cusparseSgtsv2`|9.0| | | |`rocsparse_sgtsv`|4.3.0| | | | |
|`cusparseSgtsv2StridedBatch`|9.0| | | |`rocsparse_sgtsv_no_pivot_strided_batch`|4.3.0| | | | |
|`cusparseSgtsv2StridedBatch_bufferSizeExt`|9.0| | | |`rocsparse_sgtsv_no_pivot_strided_batch_buffer_size`|4.3.0| | | | |
|`cusparseSgtsv2_bufferSizeExt`|9.0| | | |`rocsparse_sgtsv_buffer_size`|4.3.0| | | | |
|`cusparseSgtsv2_nopivot`|9.0| | | |`rocsparse_sgtsv_no_pivot`|4.3.0| | | | |
|`cusparseSgtsv2_nopivot_bufferSizeExt`|9.0| | | |`rocsparse_sgtsv_no_pivot_buffer_size`|4.3.0| | | | |
|`cusparseSgtsvInterleavedBatch`|9.2| | | |`rocsparse_sgtsv_interleaved_batch`|5.1.0| | | | |
|`cusparseSgtsvInterleavedBatch_bufferSizeExt`|9.2| | | |`rocsparse_sgtsv_interleaved_batch_buffer_size`|5.1.0| | | | |
|`cusparseSgtsvStridedBatch`| |10.2| |11.0| | | | | | |
|`cusparseSgtsv_nopivot`| |10.2| |11.0| | | | | | |
|`cusparseXbsric02_zeroPivot`| |12.2| | |`rocsparse_bsric0_zero_pivot`|3.8.0| | | | |
|`cusparseXbsrilu02_zeroPivot`| |12.2| | |`rocsparse_bsrilu0_zero_pivot`|3.9.0| | | | |
|`cusparseXcsric02_zeroPivot`| |12.2| | |`rocsparse_csric0_zero_pivot`|3.1.0| | | | |
|`cusparseXcsrilu02_zeroPivot`| |12.2| | |`rocsparse_csrilu0_zero_pivot`|1.9.0| | | | |
|`cusparseZbsric02`| |12.2| | |`rocsparse_zbsric0`|3.8.0| | | | |
|`cusparseZbsric02_analysis`| |12.2| | |`rocsparse_zbsric0_analysis`|3.6.0| | | | |
|`cusparseZbsric02_bufferSize`| |12.2| | |`rocsparse_zbsric0_buffer_size`|3.8.0| | | | |
|`cusparseZbsric02_bufferSizeExt`| |12.2| | | | | | | | |
|`cusparseZbsrilu02`| |12.2| | |`rocsparse_zbsrilu0`|3.9.0| | | | |
|`cusparseZbsrilu02_analysis`| |12.2| | |`rocsparse_zbsrilu0_analysis`|3.6.0| | | | |
|`cusparseZbsrilu02_bufferSize`| |12.2| | |`rocsparse_zbsrilu0_buffer_size`|3.8.0| | | | |
|`cusparseZbsrilu02_bufferSizeExt`| |12.2| | | | | | | | |
|`cusparseZbsrilu02_numericBoost`| |12.2| | |`rocsparse_zbsrilu0_numeric_boost`|3.9.0| | | | |
|`cusparseZcsric0`| |10.2| |11.0| | | | | | |
|`cusparseZcsric02`| |12.2| | |`rocsparse_zcsric0`|3.1.0| | | | |
|`cusparseZcsric02_analysis`| |12.2| | |`rocsparse_zcsric0_analysis`|3.1.0| | | | |
|`cusparseZcsric02_bufferSize`| |12.2| | |`rocsparse_zcsric0_buffer_size`|3.1.0| | | | |
|`cusparseZcsric02_bufferSizeExt`| |12.2| | |`rocsparse_zcsric0_buffer_size`|3.1.0| | | | |
|`cusparseZcsrilu0`| |10.2| |11.0| | | | | | |
|`cusparseZcsrilu02`| |12.2| | |`rocsparse_zcsrilu0`|2.10.0| | | | |
|`cusparseZcsrilu02_analysis`| |12.2| | |`rocsparse_zcsrilu0_analysis`|2.10.0| | | | |
|`cusparseZcsrilu02_bufferSize`| |12.2| | |`rocsparse_zcsrilu0_buffer_size`|2.10.0| | | | |
|`cusparseZcsrilu02_bufferSizeExt`| |12.2| | |`rocsparse_zcsrilu0_buffer_size`|2.10.0| | | | |
|`cusparseZcsrilu02_numericBoost`| |12.2| | |`rocsparse_zcsrilu0_numeric_boost`|3.9.0| | | | |
|`cusparseZgpsvInterleavedBatch`|9.2| | | |`rocsparse_zgpsv_interleaved_batch`|5.1.0| | | | |
|`cusparseZgpsvInterleavedBatch_bufferSizeExt`|9.2| | | |`rocsparse_zgpsv_interleaved_batch_buffer_size`|5.1.0| | | | |
|`cusparseZgtsv`| |10.2| |11.0| | | | | | |
|`cusparseZgtsv2`|9.0| | | |`rocsparse_zgtsv`|4.3.0| | | | |
|`cusparseZgtsv2StridedBatch`|9.0| | | |`rocsparse_zgtsv_no_pivot_strided_batch`|4.3.0| | | | |
|`cusparseZgtsv2StridedBatch_bufferSizeExt`|9.0| | | |`rocsparse_zgtsv_no_pivot_strided_batch_buffer_size`|4.3.0| | | | |
|`cusparseZgtsv2_bufferSizeExt`|9.0| | | |`rocsparse_zgtsv_buffer_size`|4.3.0| | | | |
|`cusparseZgtsv2_nopivot`|9.0| | | |`rocsparse_zgtsv_no_pivot`|4.3.0| | | | |
|`cusparseZgtsv2_nopivot_bufferSizeExt`|9.0| | | |`rocsparse_zgtsv_no_pivot_buffer_size`|4.3.0| | | | |
|`cusparseZgtsvInterleavedBatch`|9.2| | | |`rocsparse_zgtsv_interleaved_batch`|5.1.0| | | | |
|`cusparseZgtsvInterleavedBatch_bufferSizeExt`|9.2| | | |`rocsparse_zgtsv_interleaved_batch_buffer_size`|5.1.0| | | | |
|`cusparseZgtsvStridedBatch`| |10.2| |11.0| | | | | | |
|`cusparseZgtsv_nopivot`| |10.2| |11.0| | | | | | |

## **13. CUSPARSE Reorderings Reference**

|**CUDA**|**A**|**D**|**C**|**R**|**ROC**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cusparseCcsrcolor`| |12.2| | |`rocsparse_ccsrcolor`|4.5.0| | | | |
|`cusparseDcsrcolor`| |12.2| | |`rocsparse_dcsrcolor`|4.5.0| | | | |
|`cusparseScsrcolor`| |12.2| | |`rocsparse_scsrcolor`|4.5.0| | | | |
|`cusparseZcsrcolor`| |12.2| | |`rocsparse_zcsrcolor`|4.5.0| | | | |

## **14. CUSPARSE Format Conversion Reference**

|**CUDA**|**A**|**D**|**C**|**R**|**ROC**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cusparseCbsr2csr`| | | | |`rocsparse_cbsr2csr`|3.10.0| | | | |
|`cusparseCcsc2dense`| |11.1| |12.0|`rocsparse_ccsc2dense`|3.5.0| | | | |
|`cusparseCcsc2hyb`| |10.2| |11.0| | | | | | |
|`cusparseCcsr2bsr`| | | | |`rocsparse_ccsr2bsr`|3.5.0| | | | |
|`cusparseCcsr2csc`| |10.2| |11.0| | | | | | |
|`cusparseCcsr2csr_compress`|8.0|12.2| | |`rocsparse_ccsr2csr_compress`|3.5.0| | | | |
|`cusparseCcsr2csru`| |12.2| | | | | | | | |
|`cusparseCcsr2dense`| |11.1| |12.0|`rocsparse_ccsr2dense`|3.5.0| | | | |
|`cusparseCcsr2gebsr`| | | | |`rocsparse_ccsr2gebsr`|4.1.0| | | | |
|`cusparseCcsr2gebsr_bufferSize`| | | | |`rocsparse_ccsr2gebsr_buffer_size`|4.1.0| | | | |
|`cusparseCcsr2gebsr_bufferSizeExt`| | | | | | | | | | |
|`cusparseCcsr2hyb`| |10.2| |11.0|`rocsparse_ccsr2hyb`|2.10.0| | | | |
|`cusparseCcsru2csr`| |12.2| | | | | | | | |
|`cusparseCcsru2csr_bufferSizeExt`| |12.2| | | | | | | | |
|`cusparseCdense2csc`| |11.1| |12.0|`rocsparse_cdense2csc`|3.2.0| | | | |
|`cusparseCdense2csr`| |11.1| |12.0|`rocsparse_cdense2csr`|3.2.0| | | | |
|`cusparseCdense2hyb`| |10.2| |11.0| | | | | | |
|`cusparseCgebsr2csr`| | | | |`rocsparse_cgebsr2csr`|3.10.0| | | | |
|`cusparseCgebsr2gebsc`| | | | |`rocsparse_cgebsr2gebsc`|4.1.0| | | | |
|`cusparseCgebsr2gebsc_bufferSize`| | | | |`rocsparse_cgebsr2gebsc_buffer_size`|4.1.0| | | | |
|`cusparseCgebsr2gebsc_bufferSizeExt`| | | | | | | | | | |
|`cusparseCgebsr2gebsr`| | | | |`rocsparse_cgebsr2gebsr`|4.1.0| | | | |
|`cusparseCgebsr2gebsr_bufferSize`| | | | |`rocsparse_cgebsr2gebsr_buffer_size`|4.1.0| | | | |
|`cusparseCgebsr2gebsr_bufferSizeExt`| | | | | | | | | | |
|`cusparseChyb2csc`| |10.2| |11.0| | | | | | |
|`cusparseChyb2csr`| |10.2| |11.0| | | | | | |
|`cusparseChyb2dense`| |10.2| |11.0| | | | | | |
|`cusparseCnnz`| | | | |`rocsparse_cnnz`|3.2.0| | | | |
|`cusparseCnnz_compress`|8.0|12.2| | |`rocsparse_cnnz_compress`|3.5.0| | | | |
|`cusparseCreateCsru2csrInfo`| |12.2| | | | | | | | |
|`cusparseCreateIdentityPermutation`| |12.2| | |`rocsparse_create_identity_permutation`|1.9.0| | | | |
|`cusparseCsr2cscEx`|8.0|10.2| |11.0| | | | | | |
|`cusparseCsr2cscEx2`|10.1| | | | | | | | | |
|`cusparseCsr2cscEx2_bufferSize`|10.1| | | |`rocsparse_csr2csc_buffer_size`|1.9.0| | | | |
|`cusparseDbsr2csr`| | | | |`rocsparse_dbsr2csr`|3.10.0| | | | |
|`cusparseDcsc2dense`| |11.1| |12.0|`rocsparse_dcsc2dense`|3.5.0| | | | |
|`cusparseDcsc2hyb`| |10.2| |11.0| | | | | | |
|`cusparseDcsr2bsr`| | | | |`rocsparse_dcsr2bsr`|3.5.0| | | | |
|`cusparseDcsr2csc`| |10.2| |11.0| | | | | | |
|`cusparseDcsr2csr_compress`|8.0|12.2| | |`rocsparse_dcsr2csr_compress`|3.5.0| | | | |
|`cusparseDcsr2csru`| |12.2| | | | | | | | |
|`cusparseDcsr2dense`| |11.1| |12.0|`rocsparse_dcsr2dense`|3.5.0| | | | |
|`cusparseDcsr2gebsr`| | | | |`rocsparse_dcsr2gebsr`|4.1.0| | | | |
|`cusparseDcsr2gebsr_bufferSize`| | | | |`rocsparse_dcsr2gebsr_buffer_size`|4.1.0| | | | |
|`cusparseDcsr2gebsr_bufferSizeExt`| | | | | | | | | | |
|`cusparseDcsr2hyb`| |10.2| |11.0|`rocsparse_dcsr2hyb`|1.9.0| | | | |
|`cusparseDcsru2csr`| |12.2| | | | | | | | |
|`cusparseDcsru2csr_bufferSizeExt`| |12.2| | | | | | | | |
|`cusparseDdense2csc`| |11.1| |12.0|`rocsparse_ddense2csc`|3.2.0| | | | |
|`cusparseDdense2csr`| |11.1| |12.0|`rocsparse_ddense2csr`|3.2.0| | | | |
|`cusparseDdense2hyb`| |10.2| |11.0| | | | | | |
|`cusparseDestroyCsru2csrInfo`| |12.2| | | | | | | | |
|`cusparseDgebsr2csr`| | | | |`rocsparse_dgebsr2csr`|3.10.0| | | | |
|`cusparseDgebsr2gebsc`| | | | |`rocsparse_dgebsr2gebsc`|4.1.0| | | | |
|`cusparseDgebsr2gebsc_bufferSize`| | | | |`rocsparse_dgebsr2gebsc_buffer_size`|4.1.0| | | | |
|`cusparseDgebsr2gebsc_bufferSizeExt`| | | | | | | | | | |
|`cusparseDgebsr2gebsr`| | | | |`rocsparse_dgebsr2gebsr`|4.1.0| | | | |
|`cusparseDgebsr2gebsr_bufferSize`| | | | |`rocsparse_dgebsr2gebsr_buffer_size`|4.1.0| | | | |
|`cusparseDgebsr2gebsr_bufferSizeExt`| | | | | | | | | | |
|`cusparseDhyb2csc`| |10.2| |11.0| | | | | | |
|`cusparseDhyb2csr`| |10.2| |11.0| | | | | | |
|`cusparseDhyb2dense`| |10.2| |11.0| | | | | | |
|`cusparseDnnz`| | | | |`rocsparse_dnnz`|3.2.0| | | | |
|`cusparseDnnz_compress`|8.0|12.2| | |`rocsparse_dnnz_compress`|3.5.0| | | | |
|`cusparseDpruneCsr2csr`|9.0|12.2| | |`rocsparse_dprune_csr2csr`|3.9.0| | | | |
|`cusparseDpruneCsr2csrByPercentage`|9.0|12.2| | |`rocsparse_dprune_csr2csr_by_percentage`|3.9.0| | | | |
|`cusparseDpruneCsr2csrByPercentage_bufferSizeExt`|9.0|12.2| | |`rocsparse_dprune_csr2csr_by_percentage_buffer_size`|3.9.0| | | | |
|`cusparseDpruneCsr2csrNnz`|9.0|12.2| | |`rocsparse_dprune_csr2csr_nnz`|3.9.0| | | | |
|`cusparseDpruneCsr2csrNnzByPercentage`|9.0|12.2| | |`rocsparse_dprune_csr2csr_nnz_by_percentage`|3.9.0| | | | |
|`cusparseDpruneCsr2csr_bufferSizeExt`|9.0|12.2| | |`rocsparse_dprune_csr2csr_buffer_size`|3.9.0| | | | |
|`cusparseDpruneDense2csr`|9.0|12.2| | |`rocsparse_dprune_dense2csr`|3.9.0| | | | |
|`cusparseDpruneDense2csrByPercentage`|9.0|12.2| | |`rocsparse_dprune_dense2csr_by_percentage`|3.9.0| | | | |
|`cusparseDpruneDense2csrByPercentage_bufferSizeExt`|9.0|12.2| | |`rocsparse_dprune_dense2csr_by_percentage_buffer_size`|3.9.0| | | | |
|`cusparseDpruneDense2csrNnz`|9.0|12.2| | |`rocsparse_dprune_dense2csr_nnz`|3.9.0| | | | |
|`cusparseDpruneDense2csrNnzByPercentage`|9.0|12.2| | |`rocsparse_dprune_dense2csr_nnz_by_percentage`|3.9.0| | | | |
|`cusparseDpruneDense2csr_bufferSizeExt`|9.0|12.2| | |`rocsparse_dprune_dense2csr_buffer_size`|3.9.0| | | | |
|`cusparseHpruneCsr2csr`|9.0|12.2| | | | | | | | |
|`cusparseHpruneCsr2csrByPercentage`|9.0|12.2| | | | | | | | |
|`cusparseHpruneCsr2csrByPercentage_bufferSizeExt`|9.0|12.2| | | | | | | | |
|`cusparseHpruneCsr2csrNnz`|9.0|12.2| | | | | | | | |
|`cusparseHpruneCsr2csrNnzByPercentage`|9.0|12.2| | | | | | | | |
|`cusparseHpruneCsr2csr_bufferSizeExt`|9.0|12.2| | | | | | | | |
|`cusparseHpruneDense2csr`|9.0|12.2| | | | | | | | |
|`cusparseHpruneDense2csrByPercentage`|9.0|12.2| | | | | | | | |
|`cusparseHpruneDense2csrByPercentage_bufferSizeExt`|9.0|12.2| | | | | | | | |
|`cusparseHpruneDense2csrNnz`|9.0|12.2| | | | | | | | |
|`cusparseHpruneDense2csrNnzByPercentage`|9.0|12.2| | | | | | | | |
|`cusparseHpruneDense2csr_bufferSizeExt`|9.0|12.2| | | | | | | | |
|`cusparseSbsr2csr`| | | | |`rocsparse_sbsr2csr`|3.10.0| | | | |
|`cusparseScsc2dense`| |11.1| |12.0|`rocsparse_scsc2dense`|3.5.0| | | | |
|`cusparseScsc2hyb`| |10.2| |11.0| | | | | | |
|`cusparseScsr2bsr`| | | | |`rocsparse_scsr2bsr`|3.5.0| | | | |
|`cusparseScsr2csc`| |10.2| |11.0| | | | | | |
|`cusparseScsr2csr_compress`|8.0|12.2| | |`rocsparse_scsr2csr_compress`|3.5.0| | | | |
|`cusparseScsr2csru`| |12.2| | | | | | | | |
|`cusparseScsr2dense`| |11.1| |12.0|`rocsparse_scsr2dense`|3.5.0| | | | |
|`cusparseScsr2gebsr`| | | | |`rocsparse_scsr2gebsr`|4.1.0| | | | |
|`cusparseScsr2gebsr_bufferSize`| | | | |`rocsparse_scsr2gebsr_buffer_size`|4.1.0| | | | |
|`cusparseScsr2gebsr_bufferSizeExt`| | | | | | | | | | |
|`cusparseScsr2hyb`| |10.2| |11.0|`rocsparse_scsr2hyb`|1.9.0| | | | |
|`cusparseScsru2csr`| |12.2| | | | | | | | |
|`cusparseScsru2csr_bufferSizeExt`| |12.2| | | | | | | | |
|`cusparseSdense2csc`| |11.1| |12.0|`rocsparse_sdense2csc`|3.2.0| | | | |
|`cusparseSdense2csr`| |11.1| |12.0|`rocsparse_sdense2csr`|3.2.0| | | | |
|`cusparseSdense2hyb`| |10.2| |11.0| | | | | | |
|`cusparseSgebsr2csr`| | | | |`rocsparse_sgebsr2csr`|3.10.0| | | | |
|`cusparseSgebsr2gebsc`| | | | |`rocsparse_sgebsr2gebsc`|4.1.0| | | | |
|`cusparseSgebsr2gebsc_bufferSize`| | | | |`rocsparse_sgebsr2gebsc_buffer_size`|4.1.0| | | | |
|`cusparseSgebsr2gebsc_bufferSizeExt`| | | | | | | | | | |
|`cusparseSgebsr2gebsr`| | | | |`rocsparse_sgebsr2gebsr`|4.1.0| | | | |
|`cusparseSgebsr2gebsr_bufferSize`| | | | |`rocsparse_sgebsr2gebsr_buffer_size`|4.1.0| | | | |
|`cusparseSgebsr2gebsr_bufferSizeExt`| | | | | | | | | | |
|`cusparseShyb2csc`| |10.2| |11.0| | | | | | |
|`cusparseShyb2csr`| |10.2| |11.0| | | | | | |
|`cusparseShyb2dense`| |10.2| |11.0| | | | | | |
|`cusparseSnnz`| | | | |`rocsparse_snnz`|3.2.0| | | | |
|`cusparseSnnz_compress`|8.0|12.2| | |`rocsparse_snnz_compress`|3.5.0| | | | |
|`cusparseSpruneCsr2csr`|9.0|12.2| | |`rocsparse_sprune_csr2csr`|3.9.0| | | | |
|`cusparseSpruneCsr2csrByPercentage`|9.0|12.2| | |`rocsparse_sprune_csr2csr_by_percentage`|3.9.0| | | | |
|`cusparseSpruneCsr2csrByPercentage_bufferSizeExt`|9.0|12.2| | |`rocsparse_sprune_csr2csr_by_percentage_buffer_size`|3.9.0| | | | |
|`cusparseSpruneCsr2csrNnz`|9.0|12.2| | |`rocsparse_sprune_csr2csr_nnz`|3.9.0| | | | |
|`cusparseSpruneCsr2csrNnzByPercentage`|9.0|12.2| | |`rocsparse_sprune_csr2csr_nnz_by_percentage`|3.9.0| | | | |
|`cusparseSpruneCsr2csr_bufferSizeExt`|9.0|12.2| | |`rocsparse_sprune_csr2csr_buffer_size`|3.9.0| | | | |
|`cusparseSpruneDense2csr`|9.0|12.2| | |`rocsparse_sprune_dense2csr`|3.9.0| | | | |
|`cusparseSpruneDense2csrByPercentage`|9.0|12.2| | |`rocsparse_sprune_dense2csr_by_percentage`|3.9.0| | | | |
|`cusparseSpruneDense2csrByPercentage_bufferSizeExt`|9.0|12.2| | |`rocsparse_sprune_dense2csr_by_percentage_buffer_size`|3.9.0| | | | |
|`cusparseSpruneDense2csrNnz`|9.0|12.2| | |`rocsparse_sprune_dense2csr_nnz`|3.9.0| | | | |
|`cusparseSpruneDense2csrNnzByPercentage`|9.0|12.2| | |`rocsparse_sprune_dense2csr_nnz_by_percentage`|3.9.0| | | | |
|`cusparseSpruneDense2csr_bufferSizeExt`|9.0|12.2| | |`rocsparse_sprune_dense2csr_buffer_size`|3.9.0| | | | |
|`cusparseXcoo2csr`| | | | |`rocsparse_coo2csr`|1.9.0| | | | |
|`cusparseXcoosortByColumn`| | | | |`rocsparse_coosort_by_column`|1.9.0| | | | |
|`cusparseXcoosortByRow`| | | | |`rocsparse_coosort_by_row`|1.9.0| | | | |
|`cusparseXcoosort_bufferSizeExt`| | | | |`rocsparse_coosort_buffer_size`|1.9.0| | | | |
|`cusparseXcscsort`| | | | |`rocsparse_cscsort`|2.10.0| | | | |
|`cusparseXcscsort_bufferSizeExt`| | | | |`rocsparse_cscsort_buffer_size`|2.10.0| | | | |
|`cusparseXcsr2bsrNnz`| | | | |`rocsparse_csr2bsr_nnz`|3.5.0| | | | |
|`cusparseXcsr2coo`| | | | |`rocsparse_csr2coo`|1.9.0| | | | |
|`cusparseXcsr2gebsrNnz`| | | | |`rocsparse_csr2gebsr_nnz`|4.1.0| | | | |
|`cusparseXcsrsort`| | | | |`rocsparse_csrsort`|1.9.0| | | | |
|`cusparseXcsrsort_bufferSizeExt`| | | | |`rocsparse_csrsort_buffer_size`|1.9.0| | | | |
|`cusparseXgebsr2csr`| | | | | | | | | | |
|`cusparseXgebsr2gebsrNnz`| | | | |`rocsparse_gebsr2gebsr_nnz`|4.1.0| | | | |
|`cusparseZbsr2csr`| | | | |`rocsparse_zbsr2csr`|3.10.0| | | | |
|`cusparseZcsc2dense`| |11.1| |12.0|`rocsparse_zcsc2dense`|3.5.0| | | | |
|`cusparseZcsc2hyb`| |10.2| |11.0| | | | | | |
|`cusparseZcsr2bsr`| | | | |`rocsparse_zcsr2bsr`|3.5.0| | | | |
|`cusparseZcsr2csc`| |10.2| |11.0| | | | | | |
|`cusparseZcsr2csr_compress`|8.0|12.2| | |`rocsparse_zcsr2csr_compress`|3.5.0| | | | |
|`cusparseZcsr2csru`| |12.2| | | | | | | | |
|`cusparseZcsr2dense`| |11.1| |12.0|`rocsparse_zcsr2dense`|3.5.0| | | | |
|`cusparseZcsr2gebsr`| | | | |`rocsparse_zcsr2gebsr`|4.1.0| | | | |
|`cusparseZcsr2gebsr_bufferSize`| | | | |`rocsparse_zcsr2gebsr_buffer_size`|4.1.0| | | | |
|`cusparseZcsr2gebsr_bufferSizeExt`| | | | | | | | | | |
|`cusparseZcsr2hyb`| |10.2| |11.0|`rocsparse_zcsr2hyb`|2.10.0| | | | |
|`cusparseZcsru2csr`| |12.2| | | | | | | | |
|`cusparseZcsru2csr_bufferSizeExt`| |12.2| | | | | | | | |
|`cusparseZdense2csc`| |11.1| |12.0|`rocsparse_zdense2csc`|3.2.0| | | | |
|`cusparseZdense2csr`| |11.1| |12.0|`rocsparse_zdense2csr`|3.2.0| | | | |
|`cusparseZdense2hyb`| |10.2| |11.0| | | | | | |
|`cusparseZgebsr2csr`| | | | |`rocsparse_zgebsr2csr`|3.10.0| | | | |
|`cusparseZgebsr2gebsc`| | | | |`rocsparse_zgebsr2gebsc`|4.1.0| | | | |
|`cusparseZgebsr2gebsc_bufferSize`| | | | |`rocsparse_zgebsr2gebsc_buffer_size`|4.1.0| | | | |
|`cusparseZgebsr2gebsc_bufferSizeExt`| | | | | | | | | | |
|`cusparseZgebsr2gebsr`| | | | |`rocsparse_zgebsr2gebsr`|4.1.0| | | | |
|`cusparseZgebsr2gebsr_bufferSize`| | | | |`rocsparse_zgebsr2gebsr_buffer_size`|4.1.0| | | | |
|`cusparseZgebsr2gebsr_bufferSizeExt`| | | | | | | | | | |
|`cusparseZhyb2csc`| |10.2| |11.0| | | | | | |
|`cusparseZhyb2csr`| |10.2| |11.0| | | | | | |
|`cusparseZhyb2dense`| |10.2| |11.0| | | | | | |
|`cusparseZnnz`| | | | |`rocsparse_znnz`|3.2.0| | | | |
|`cusparseZnnz_compress`|8.0|12.2| | |`rocsparse_znnz_compress`|3.5.0| | | | |

## **15. CUSPARSE Generic API Reference**

|**CUDA**|**A**|**D**|**C**|**R**|**ROC**|**A**|**D**|**C**|**R**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|
|`cusparseAxpby`|11.0| |12.0| |`rocsparse_axpby`|4.1.0| |6.0.0| | |
|`cusparseBlockedEllGet`|11.2| | | |`rocsparse_bell_get`|4.1.0| | | | |
|`cusparseBsrSetStridedBatch`|12.1| | | | | | | | | |
|`cusparseConstBlockedEllGet`|12.0| | | |`rocsparse_const_bell_get`|6.0.0| | | | |
|`cusparseConstCooGet`|12.0| | | |`rocsparse_const_coo_get`|6.0.0| | | | |
|`cusparseConstCscGet`|12.0| | | |`rocsparse_const_csc_get`|6.0.0| | | | |
|`cusparseConstCsrGet`|12.0| | | |`rocsparse_const_csr_get`|6.0.0| | | | |
|`cusparseConstDnMatGet`|12.0| | | |`rocsparse_const_dnmat_get`|6.0.0| | | | |
|`cusparseConstDnMatGetValues`|12.0| | | |`rocsparse_const_dnmat_get_values`|6.0.0| | | | |
|`cusparseConstDnVecGet`|12.0| | | |`rocsparse_const_dnvec_get`|6.0.0| | | | |
|`cusparseConstDnVecGetValues`|12.0| | | |`rocsparse_const_dnvec_get_values`|6.0.0| | | | |
|`cusparseConstSpMatGetValues`|12.0| | | |`rocsparse_const_spmat_get_values`|6.0.0| | | | |
|`cusparseConstSpVecGet`|12.0| | | |`rocsparse_const_spvec_get`|6.0.0| | | | |
|`cusparseConstSpVecGetValues`|12.0| | | |`rocsparse_const_spvec_get_values`|6.0.0| | | | |
|`cusparseConstrainedGeMM`|10.2|11.2| |12.0| | | | | | |
|`cusparseConstrainedGeMM_bufferSize`|10.2|11.2| |12.0| | | | | | |
|`cusparseCooAoSGet`|10.2|11.2| |12.0|`rocsparse_coo_aos_get`|4.1.0| | | | |
|`cusparseCooGet`|10.1| | | |`rocsparse_coo_get`|4.1.0| | | | |
|`cusparseCooSetPointers`|11.1| | | |`rocsparse_coo_set_pointers`|4.1.0| | | | |
|`cusparseCooSetStridedBatch`|11.0| | | |`rocsparse_coo_set_strided_batch`|5.2.0| | | | |
|`cusparseCreateBlockedEll`|11.2| | | |`rocsparse_create_bell_descr`|4.5.0| | | | |
|`cusparseCreateBsr`|12.1| | | | | | | | | |
|`cusparseCreateConstBlockedEll`|12.0| | | |`rocsparse_create_const_bell_descr`|6.0.0| | | | |
|`cusparseCreateConstBsr`|12.1| | | | | | | | | |
|`cusparseCreateConstCoo`|12.0| | | |`rocsparse_create_const_coo_descr`|6.0.0| | | | |
|`cusparseCreateConstCsc`|12.0| | | |`rocsparse_create_const_csc_descr`|6.0.0| | | | |
|`cusparseCreateConstCsr`|12.0| | | |`rocsparse_create_const_csr_descr`|6.0.0| | | | |
|`cusparseCreateConstDnMat`|12.0| | | |`rocsparse_create_const_dnmat_descr`|6.0.0| | | | |
|`cusparseCreateConstDnVec`|12.0| | | |`rocsparse_create_const_dnvec_descr`|6.0.0| | | | |
|`cusparseCreateConstSlicedEll`|12.1| | | | | | | | | |
|`cusparseCreateConstSpVec`|12.0| | | |`rocsparse_create_const_spvec_descr`|6.0.0| | | | |
|`cusparseCreateCoo`|10.1| | | |`rocsparse_create_coo_descr`|4.1.0| | | | |
|`cusparseCreateCooAoS`|10.2|11.2| |12.0|`rocsparse_create_coo_aos_descr`|4.1.0| | | | |
|`cusparseCreateCsc`|11.1| | | |`rocsparse_create_csc_descr`|4.1.0| | | | |
|`cusparseCreateCsr`|10.2| | | |`rocsparse_create_csr_descr`|4.1.0| | | | |
|`cusparseCreateDnMat`|10.1| | | |`rocsparse_create_dnmat_descr`|4.1.0| | | | |
|`cusparseCreateDnVec`|10.2| | | |`rocsparse_create_dnvec_descr`|4.1.0| | | | |
|`cusparseCreateSlicedEll`|12.1| | | | | | | | | |
|`cusparseCreateSpVec`|10.2| | | |`rocsparse_create_spvec_descr`|4.1.0| | | | |
|`cusparseCscGet`|11.7| | | |`rocsparse_csc_get`|6.1.0| | | |6.1.0|
|`cusparseCscSetPointers`|11.1| | | |`rocsparse_csc_set_pointers`|4.1.0| | | | |
|`cusparseCsrGet`|10.2| | | |`rocsparse_csr_get`|4.1.0| | | | |
|`cusparseCsrSetPointers`|11.0| | | |`rocsparse_csr_set_pointers`|4.1.0| | | | |
|`cusparseCsrSetStridedBatch`|11.0| | | |`rocsparse_csr_set_strided_batch`|5.2.0| | | | |
|`cusparseDenseToSparse_analysis`|11.1| |12.0| |`rocsparse_dense_to_sparse`|4.1.0| |6.0.0| | |
|`cusparseDenseToSparse_bufferSize`|11.1| |12.0| |`rocsparse_dense_to_sparse`|4.1.0| |6.0.0| | |
|`cusparseDenseToSparse_convert`|11.1| |12.0| | | | | | | |
|`cusparseDestroyDnMat`|10.1| |12.0| |`rocsparse_destroy_dnmat_descr`|4.1.0| |6.0.0| | |
|`cusparseDestroyDnVec`|10.2| |12.0| |`rocsparse_destroy_dnvec_descr`|4.1.0| |6.0.0| | |
|`cusparseDestroySpMat`|10.1| |12.0| |`rocsparse_destroy_spmat_descr`|4.1.0| |6.0.0| | |
|`cusparseDestroySpVec`|10.2| |12.0| |`rocsparse_destroy_spvec_descr`|4.1.0| |6.0.0| | |
|`cusparseDnMatGet`|10.1| | | |`rocsparse_dnmat_get`|4.1.0| | | | |
|`cusparseDnMatGetStridedBatch`|10.1| |12.0| |`rocsparse_dnmat_get_strided_batch`|5.2.0| |6.0.0| | |
|`cusparseDnMatGetValues`|10.2| | | |`rocsparse_dnmat_get_values`|4.1.0| | | | |
|`cusparseDnMatSetStridedBatch`|10.1| | | |`rocsparse_dnmat_set_strided_batch`|5.2.0| | | | |
|`cusparseDnMatSetValues`|10.2| | | |`rocsparse_dnmat_set_values`|4.1.0| | | | |
|`cusparseDnVecGet`|10.2| | | |`rocsparse_dnvec_get`|4.1.0| | | | |
|`cusparseDnVecGetValues`|10.2| | | |`rocsparse_dnvec_get_values`|4.1.0| | | | |
|`cusparseDnVecSetValues`|10.2| | | |`rocsparse_dnvec_set_values`|4.1.0| | | | |
|`cusparseGather`|11.0| |12.0| |`rocsparse_gather`|4.1.0| |6.0.0| | |
|`cusparseRot`|11.0|12.2| | |`rocsparse_rot`|4.1.0| | | | |
|`cusparseSDDMM`|11.2| |12.0| |`rocsparse_sddmm`|4.3.0| |6.0.0| | |
|`cusparseSDDMM_bufferSize`|11.2| |12.0| |`rocsparse_sddmm_buffer_size`|4.3.0| |6.0.0| | |
|`cusparseSDDMM_preprocess`|11.2| |12.0| |`rocsparse_sddmm_preprocess`|4.3.0| |6.0.0| | |
|`cusparseScatter`|11.0| |12.0| |`rocsparse_scatter`|4.1.0| |6.0.0| | |
|`cusparseSpGEMM_compute`|11.0| |12.0| | | | | | | |
|`cusparseSpGEMM_copy`|11.0| |12.0| | | | | | | |
|`cusparseSpGEMM_createDescr`|11.0| | | | | | | | | |
|`cusparseSpGEMM_destroyDescr`|11.0| | | | | | | | | |
|`cusparseSpGEMM_estimateMemory`|12.0| | | | | | | | | |
|`cusparseSpGEMM_getNumProducts`|12.0| | | | | | | | | |
|`cusparseSpGEMM_workEstimation`|11.0| |12.0| | | | | | | |
|`cusparseSpGEMMreuse_compute`|11.3| |12.0| | | | | | | |
|`cusparseSpGEMMreuse_copy`|11.3| |12.0| | | | | | | |
|`cusparseSpGEMMreuse_nnz`|11.3| |12.0| | | | | | | |
|`cusparseSpGEMMreuse_workEstimation`|11.3| |12.0| | | | | | | |
|`cusparseSpMM`|10.1| |12.0| |`rocsparse_spmm`|4.2.0| |6.0.0| | |
|`cusparseSpMMOp`|11.5| | | | | | | | | |
|`cusparseSpMMOp_createPlan`|11.5| | | | | | | | | |
|`cusparseSpMMOp_destroyPlan`|11.5| | | | | | | | | |
|`cusparseSpMM_bufferSize`|10.1| |12.0| |`rocsparse_spmm`|4.2.0| |6.0.0| | |
|`cusparseSpMM_preprocess`|11.2| |12.0| |`rocsparse_spmm`|4.2.0| |6.0.0| | |
|`cusparseSpMV`|10.1| |12.0| |`rocsparse_spmv`|4.1.0| |6.0.0| | |
|`cusparseSpMV_bufferSize`|10.1| |12.0| |`rocsparse_spmv`|4.1.0| |6.0.0| | |
|`cusparseSpMatGetAttribute`|11.3| |12.0| |`rocsparse_spmat_get_attribute`|4.5.0| |6.0.0| | |
|`cusparseSpMatGetFormat`|10.1| |12.0| |`rocsparse_spmat_get_format`|4.1.0| |6.0.0| | |
|`cusparseSpMatGetIndexBase`|10.1| |12.0| |`rocsparse_spmat_get_index_base`|4.1.0| |6.0.0| | |
|`cusparseSpMatGetNumBatches`|10.1| | |10.2| | | | | | |
|`cusparseSpMatGetSize`|11.0| |12.0| |`rocsparse_spmat_get_size`|4.1.0| |6.0.0| | |
|`cusparseSpMatGetStridedBatch`|10.2| |12.0| |`rocsparse_spmat_get_strided_batch`|5.2.0| |6.0.0| | |
|`cusparseSpMatGetValues`|10.2| | | |`rocsparse_spmat_get_values`|4.1.0| | | | |
|`cusparseSpMatSetAttribute`|11.3| | | |`rocsparse_spmat_set_attribute`|4.5.0| | | | |
|`cusparseSpMatSetNumBatches`|10.1| | |10.2| | | | | | |
|`cusparseSpMatSetStridedBatch`|10.2| | |12.0|`rocsparse_spmat_set_strided_batch`|5.2.0| | | | |
|`cusparseSpMatSetValues`|10.2| | | |`rocsparse_spmat_set_values`|4.1.0| | | | |
|`cusparseSpSM_analysis`|11.3| |12.0| |`rocsparse_spsm`|4.5.0| |6.0.0| | |
|`cusparseSpSM_bufferSize`|11.3| |12.0| | | | | | | |
|`cusparseSpSM_createDescr`|11.3| | | | | | | | | |
|`cusparseSpSM_destroyDescr`|11.3| | | | | | | | | |
|`cusparseSpSM_solve`|11.3| |12.0| |`rocsparse_spsm`|4.5.0| |6.0.0| | |
|`cusparseSpSV_analysis`|11.3| |12.0| | | | | | | |
|`cusparseSpSV_bufferSize`|11.3| |12.0| |`rocsparse_spsv`|4.5.0| |6.0.0| | |
|`cusparseSpSV_createDescr`|11.3| | | | | | | | | |
|`cusparseSpSV_destroyDescr`|11.3| | | | | | | | | |
|`cusparseSpSV_solve`|11.3| |12.0| | | | | | | |
|`cusparseSpSV_updateMatrix`|12.1| | | | | | | | | |
|`cusparseSpVV`|10.1| |12.0| |`rocsparse_spvv`|4.1.0| |6.0.0| | |
|`cusparseSpVV_bufferSize`|10.1| |12.0| |`rocsparse_spvv`|4.1.0| |6.0.0| | |
|`cusparseSpVecGet`|10.2| | | |`rocsparse_spvec_get`|4.1.0| | | | |
|`cusparseSpVecGetIndexBase`|10.2| |12.0| |`rocsparse_spvec_get_index_base`|4.1.0| |6.0.0| | |
|`cusparseSpVecGetValues`|10.2| | | |`rocsparse_spvec_get_values`|4.1.0| | | | |
|`cusparseSpVecSetValues`|10.2| | | |`rocsparse_spvec_set_values`|4.1.0| | | | |
|`cusparseSparseToDense`|11.1| |12.0| |`rocsparse_sparse_to_dense`|4.1.0| |6.0.0| | |
|`cusparseSparseToDense_bufferSize`|11.1| |12.0| |`rocsparse_sparse_to_dense`|4.1.0| |6.0.0| | |


\*A - Added; D - Deprecated; C - Changed; R - Removed; E - Experimental