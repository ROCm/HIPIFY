# CUSPARSE API supported by HIP

## **4. CUSPARSE Types References**

|**CUDA**|**A**|**D**|**R**|**HIP**|**A**|**D**|**R**|
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`CUSPARSE_ACTION_NUMERIC`| | | |`HIPSPARSE_ACTION_NUMERIC`|1.9.2| | |
|`CUSPARSE_ACTION_SYMBOLIC`| | | |`HIPSPARSE_ACTION_SYMBOLIC`|1.9.2| | |
|`CUSPARSE_ALG0`|8.0| |11.0| | | | |
|`CUSPARSE_ALG1`|8.0| |11.0| | | | |
|`CUSPARSE_ALG_MERGE_PATH`|9.2| | | | | | |
|`CUSPARSE_ALG_NAIVE`|9.2| |11.0| | | | |
|`CUSPARSE_COOMM_ALG1`|10.1|11.0| |`HIPSPARSE_COOMM_ALG1`|4.2.0| | |
|`CUSPARSE_COOMM_ALG2`|10.1|11.0| |`HIPSPARSE_COOMM_ALG2`|4.2.0| | |
|`CUSPARSE_COOMM_ALG3`|10.1|11.0| |`HIPSPARSE_COOMM_ALG3`|4.2.0| | |
|`CUSPARSE_COOMV_ALG`|10.2|11.2| |`HIPSPARSE_COOMV_ALG`|4.1.0| | |
|`CUSPARSE_CSR2CSC_ALG1`|10.1| | | | | | |
|`CUSPARSE_CSR2CSC_ALG2`|10.1| | | | | | |
|`CUSPARSE_CSRMM_ALG1`|10.2|11.0| |`HIPSPARSE_CSRMM_ALG1`|4.2.0| | |
|`CUSPARSE_CSRMV_ALG1`|10.2|11.2| |`HIPSPARSE_CSRMV_ALG1`|4.1.0| | |
|`CUSPARSE_CSRMV_ALG2`|10.2|11.2| |`HIPSPARSE_CSRMV_ALG2`|4.1.0| | |
|`CUSPARSE_DENSETOSPARSE_ALG_DEFAULT`|11.1| | | | | | |
|`CUSPARSE_DIAG_TYPE_NON_UNIT`| | | |`HIPSPARSE_DIAG_TYPE_NON_UNIT`|1.9.2| | |
|`CUSPARSE_DIAG_TYPE_UNIT`| | | |`HIPSPARSE_DIAG_TYPE_UNIT`|1.9.2| | |
|`CUSPARSE_DIRECTION_COLUMN`| | | |`HIPSPARSE_DIRECTION_COLUMN`|3.2.0| | |
|`CUSPARSE_DIRECTION_ROW`| | | |`HIPSPARSE_DIRECTION_ROW`|3.2.0| | |
|`CUSPARSE_FILL_MODE_LOWER`| | | |`HIPSPARSE_FILL_MODE_LOWER`|1.9.2| | |
|`CUSPARSE_FILL_MODE_UPPER`| | | |`HIPSPARSE_FILL_MODE_UPPER`|1.9.2| | |
|`CUSPARSE_FORMAT_COO`|10.1| | |`HIPSPARSE_FORMAT_COO`|4.1.0| | |
|`CUSPARSE_FORMAT_COO_AOS`|10.2| | |`HIPSPARSE_FORMAT_COO_AOS`|4.1.0| | |
|`CUSPARSE_FORMAT_CSC`|10.1| | |`HIPSPARSE_FORMAT_CSC`|4.1.0| | |
|`CUSPARSE_FORMAT_CSR`|10.1| | |`HIPSPARSE_FORMAT_CSR`|4.1.0| | |
|`CUSPARSE_HYB_PARTITION_AUTO`| |10.2|11.0|`HIPSPARSE_HYB_PARTITION_AUTO`|1.9.2| | |
|`CUSPARSE_HYB_PARTITION_MAX`| |10.2|11.0|`HIPSPARSE_HYB_PARTITION_MAX`|1.9.2| | |
|`CUSPARSE_HYB_PARTITION_USER`| |10.2|11.0|`HIPSPARSE_HYB_PARTITION_USER`|1.9.2| | |
|`CUSPARSE_INDEX_16U`|10.1| | |`HIPSPARSE_INDEX_16U`|4.1.0| | |
|`CUSPARSE_INDEX_32I`|10.1| | |`HIPSPARSE_INDEX_32I`|4.1.0| | |
|`CUSPARSE_INDEX_64I`|10.2| | |`HIPSPARSE_INDEX_64I`|4.1.0| | |
|`CUSPARSE_INDEX_BASE_ONE`| | | |`HIPSPARSE_INDEX_BASE_ONE`|1.9.2| | |
|`CUSPARSE_INDEX_BASE_ZERO`| | | |`HIPSPARSE_INDEX_BASE_ZERO`|1.9.2| | |
|`CUSPARSE_MATRIX_TYPE_GENERAL`| | | |`HIPSPARSE_MATRIX_TYPE_GENERAL`|1.9.2| | |
|`CUSPARSE_MATRIX_TYPE_HERMITIAN`| | | |`HIPSPARSE_MATRIX_TYPE_HERMITIAN`|1.9.2| | |
|`CUSPARSE_MATRIX_TYPE_SYMMETRIC`| | | |`HIPSPARSE_MATRIX_TYPE_SYMMETRIC`|1.9.2| | |
|`CUSPARSE_MATRIX_TYPE_TRIANGULAR`| | | |`HIPSPARSE_MATRIX_TYPE_TRIANGULAR`|1.9.2| | |
|`CUSPARSE_MM_ALG_DEFAULT`|10.2|11.0| |`HIPSPARSE_MM_ALG_DEFAULT`|4.2.0| | |
|`CUSPARSE_MV_ALG_DEFAULT`|10.2|11.3| |`HIPSPARSE_MV_ALG_DEFAULT`|4.1.0| | |
|`CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE`| | | |`HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE`|1.9.2| | |
|`CUSPARSE_OPERATION_NON_TRANSPOSE`| | | |`HIPSPARSE_OPERATION_NON_TRANSPOSE`|1.9.2| | |
|`CUSPARSE_OPERATION_TRANSPOSE`| | | |`HIPSPARSE_OPERATION_TRANSPOSE`|1.9.2| | |
|`CUSPARSE_ORDER_COL`|10.1| | |`HIPSPARSE_ORDER_COL`|4.2.0| | |
|`CUSPARSE_ORDER_ROW`|10.1| | |`HIPSPARSE_ORDER_ROW`|4.2.0| | |
|`CUSPARSE_POINTER_MODE_DEVICE`| | | |`HIPSPARSE_POINTER_MODE_DEVICE`|1.9.2| | |
|`CUSPARSE_POINTER_MODE_HOST`| | | |`HIPSPARSE_POINTER_MODE_HOST`|1.9.2| | |
|`CUSPARSE_SOLVE_POLICY_NO_LEVEL`| | | |`HIPSPARSE_SOLVE_POLICY_NO_LEVEL`|1.9.2| | |
|`CUSPARSE_SOLVE_POLICY_USE_LEVEL`| | | |`HIPSPARSE_SOLVE_POLICY_USE_LEVEL`|1.9.2| | |
|`CUSPARSE_SPARSETODENSE_ALG_DEFAULT`|11.1| | |`HIPSPARSE_SPARSETODENSE_ALG_DEFAULT`|4.2.0| | |
|`CUSPARSE_SPGEMM_CSR_ALG_DETERMINITIC`|11.3| | | | | | |
|`CUSPARSE_SPGEMM_CSR_ALG_NONDETERMINITIC`|11.3| | | | | | |
|`CUSPARSE_SPGEMM_DEFAULT`|11.0| | |`HIPSPARSE_SPGEMM_DEFAULT`|4.1.0| | |
|`CUSPARSE_SPMAT_DIAG_TYPE`|11.3| | | | | | |
|`CUSPARSE_SPMAT_FILL_MODE`|11.3| | | | | | |
|`CUSPARSE_SPMMA_ALG1`|11.1| | | | | | |
|`CUSPARSE_SPMMA_ALG2`|11.1| | | | | | |
|`CUSPARSE_SPMMA_ALG3`|11.1| | | | | | |
|`CUSPARSE_SPMMA_ALG4`|11.1| | | | | | |
|`CUSPARSE_SPMMA_PREPROCESS`|11.1| | | | | | |
|`CUSPARSE_SPMM_ALG_DEFAULT`|11.0| | |`HIPSPARSE_SPMM_ALG_DEFAULT`|4.2.0| | |
|`CUSPARSE_SPMM_COO_ALG1`|11.0| | |`HIPSPARSE_SPMM_COO_ALG1`|4.2.0| | |
|`CUSPARSE_SPMM_COO_ALG2`|11.0| | |`HIPSPARSE_SPMM_COO_ALG2`|4.2.0| | |
|`CUSPARSE_SPMM_COO_ALG3`|11.0| | |`HIPSPARSE_SPMM_COO_ALG3`|4.2.0| | |
|`CUSPARSE_SPMM_COO_ALG4`|11.0| | |`HIPSPARSE_SPMM_COO_ALG4`|4.2.0| | |
|`CUSPARSE_SPMM_CSR_ALG1`|11.0| | |`HIPSPARSE_SPMM_CSR_ALG1`|4.2.0| | |
|`CUSPARSE_SPMM_CSR_ALG2`|11.0| | |`HIPSPARSE_SPMM_CSR_ALG2`|4.2.0| | |
|`CUSPARSE_SPMV_ALG_DEFAULT`| | | |`HIPSPARSE_MV_ALG_DEFAULT`|4.1.0| | |
|`CUSPARSE_SPMV_COO_ALG1`|11.2| | |`HIPSPARSE_COOMV_ALG`|4.1.0| | |
|`CUSPARSE_SPMV_CSR_ALG1`|11.2| | |`HIPSPARSE_CSRMV_ALG1`|4.1.0| | |
|`CUSPARSE_SPMV_CSR_ALG2`|11.2| | |`HIPSPARSE_CSRMV_ALG2`|4.1.0| | |
|`CUSPARSE_SPSM_ALG_DEFAULT`|11.3| | | | | | |
|`CUSPARSE_SPSV_ALG_DEFAULT`|11.3| | | | | | |
|`CUSPARSE_STATUS_ALLOC_FAILED`| | | |`HIPSPARSE_STATUS_ALLOC_FAILED`|1.9.2| | |
|`CUSPARSE_STATUS_ARCH_MISMATCH`| | | |`HIPSPARSE_STATUS_ARCH_MISMATCH`|1.9.2| | |
|`CUSPARSE_STATUS_EXECUTION_FAILED`| | | |`HIPSPARSE_STATUS_EXECUTION_FAILED`|1.9.2| | |
|`CUSPARSE_STATUS_INSUFFICIENT_RESOURCES`|11.0| | |`HIPSPARSE_STATUS_INSUFFICIENT_RESOURCES`|4.1.0| | |
|`CUSPARSE_STATUS_INTERNAL_ERROR`| | | |`HIPSPARSE_STATUS_INTERNAL_ERROR`|1.9.2| | |
|`CUSPARSE_STATUS_INVALID_VALUE`| | | |`HIPSPARSE_STATUS_INVALID_VALUE`|1.9.2| | |
|`CUSPARSE_STATUS_MAPPING_ERROR`| | | |`HIPSPARSE_STATUS_MAPPING_ERROR`|1.9.2| | |
|`CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED`| | | |`HIPSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED`|1.9.2| | |
|`CUSPARSE_STATUS_NOT_INITIALIZED`| | | |`HIPSPARSE_STATUS_NOT_INITIALIZED`|1.9.2| | |
|`CUSPARSE_STATUS_NOT_SUPPORTED`|10.2| | |`HIPSPARSE_STATUS_NOT_SUPPORTED`|4.1.0| | |
|`CUSPARSE_STATUS_SUCCESS`| | | |`HIPSPARSE_STATUS_SUCCESS`|1.9.2| | |
|`CUSPARSE_STATUS_ZERO_PIVOT`| | | |`HIPSPARSE_STATUS_ZERO_PIVOT`|1.9.2| | |
|`CUSPARSE_VERSION`|10.2| | | | | | |
|`CUSPARSE_VER_BUILD`|10.2| | | | | | |
|`CUSPARSE_VER_MAJOR`|10.2| | | | | | |
|`CUSPARSE_VER_MINOR`|10.2| | | | | | |
|`CUSPARSE_VER_PATCH`|10.2| | | | | | |
|`bsric02Info`| | | | | | | |
|`bsric02Info_t`| | | |`bsric02Info_t`|3.8.0| | |
|`bsrilu02Info`| | | | | | | |
|`bsrilu02Info_t`| | | |`bsrilu02Info_t`|3.9.0| | |
|`bsrsm2Info`| | | | | | | |
|`bsrsm2Info_t`| | | | | | | |
|`bsrsv2Info`| | | | | | | |
|`bsrsv2Info_t`| | | |`bsrsv2Info_t`|3.6.0| | |
|`csrgemm2Info`| | | | | | | |
|`csrgemm2Info_t`| | | |`csrgemm2Info_t`|2.8.0| | |
|`csrilu02Info`| | | | | | | |
|`csrilu02Info_t`| | | |`csrilu02Info_t`|1.9.2| | |
|`csrsm2Info`|9.2| | | | | | |
|`csrsm2Info_t`|9.2| | |`csrsm2Info_t`|3.1.0| | |
|`csrsv2Info`| | | | | | | |
|`csrsv2Info_t`| | | |`csrsv2Info_t`|1.9.2| | |
|`csru2csrInfo`| | | |`csru2csrInfo`|4.2.0| | |
|`csru2csrInfo_t`| | | |`csru2csrInfo_t`|4.2.0| | |
|`cusparseAction_t`| | | |`hipsparseAction_t`|1.9.2| | |
|`cusparseAlgMode_t`|8.0| | | | | | |
|`cusparseColorInfo`| | | | | | | |
|`cusparseColorInfo_t`| | | | | | | |
|`cusparseContext`| | | | | | | |
|`cusparseCsr2CscAlg_t`|10.1| | | | | | |
|`cusparseDenseToSparseAlg_t`|11.1| | | | | | |
|`cusparseDiagType_t`| | | |`hipsparseDiagType_t`|1.9.2| | |
|`cusparseDirection_t`| | | |`hipsparseDirection_t`|3.2.0| | |
|`cusparseDnMatDescr`|10.1| | |`hipsparseDnMatDescr`|4.2.0| | |
|`cusparseDnMatDescr_t`|10.1| | |`hipsparseDnMatDescr_t`|4.2.0| | |
|`cusparseDnVecDescr`|10.2| | | | | | |
|`cusparseDnVecDescr_t`|10.2| | |`hipsparseDnVecDescr_t`|4.1.0| | |
|`cusparseFillMode_t`| | | |`hipsparseFillMode_t`|1.9.2| | |
|`cusparseFormat_t`|10.1| | |`hipsparseFormat_t`|4.1.0| | |
|`cusparseHandle_t`| | | |`hipsparseHandle_t`|1.9.2| | |
|`cusparseHybMat`| |10.2|11.0| | | | |
|`cusparseHybMat_t`| |10.2|11.0|`hipsparseHybMat_t`|1.9.2| | |
|`cusparseHybPartition_t`| |10.2|11.0|`hipsparseHybPartition_t`|1.9.2| | |
|`cusparseIndexBase_t`| | | |`hipsparseIndexBase_t`|1.9.2| | |
|`cusparseIndexType_t`|10.1| | |`hipsparseIndexType_t`|4.1.0| | |
|`cusparseMatDescr`| | | | | | | |
|`cusparseMatDescr_t`| | | |`hipsparseMatDescr_t`|1.9.2| | |
|`cusparseMatrixType_t`| | | |`hipsparseMatrixType_t`|1.9.2| | |
|`cusparseOperation_t`| | | |`hipsparseOperation_t`|1.9.2| | |
|`cusparseOrder_t`|10.1| | |`hipsparseOrder_t`|4.2.0| | |
|`cusparsePointerMode_t`| | | |`hipsparsePointerMode_t`|1.9.2| | |
|`cusparseSolveAnalysisInfo`| |10.2|11.0| | | | |
|`cusparseSolveAnalysisInfo_t`| |10.2|11.0| | | | |
|`cusparseSolvePolicy_t`| | | |`hipsparseSolvePolicy_t`|1.9.2| | |
|`cusparseSpGEMMAlg_t`|11.0| | |`hipsparseSpGEMMAlg_t`|4.1.0| | |
|`cusparseSpGEMMDescr`|11.0| | |`hipsparseSpGEMMDescr`|4.1.0| | |
|`cusparseSpGEMMDescr_t`|11.0| | |`hipsparseSpGEMMDescr_t`|4.1.0| | |
|`cusparseSpMMAlg_t`|10.1| | |`hipsparseSpMMAlg_t`|4.2.0| | |
|`cusparseSpMVAlg_t`|10.2| | |`hipsparseSpMVAlg_t`|4.1.0| | |
|`cusparseSpMatAttribute_t`|11.3| | | | | | |
|`cusparseSpMatDescr`|10.1| | | | | | |
|`cusparseSpMatDescr_t`|10.1| | |`hipsparseSpMatDescr_t`|4.1.0| | |
|`cusparseSpSMAlg_t`|11.3| | | | | | |
|`cusparseSpSMDescr`|11.3| | | | | | |
|`cusparseSpSMDescr_t`|11.3| | | | | | |
|`cusparseSpSVAlg_t`|11.3| | | | | | |
|`cusparseSpVecDescr`|10.2| | | | | | |
|`cusparseSpVecDescr_t`|10.2| | |`hipsparseSpVecDescr_t`|4.1.0| | |
|`cusparseSparseToDenseAlg_t`|11.1| | |`hipsparseSparseToDenseAlg_t`|4.2.0| | |
|`cusparseStatus_t`| | | |`hipsparseStatus_t`|1.9.2| | |
|`pruneInfo`|9.0| | | | | | |
|`pruneInfo_t`|9.0| | |`pruneInfo_t`|3.9.0| | |

## **5. CUSPARSE Management Function Reference**

|**CUDA**|**A**|**D**|**R**|**HIP**|**A**|**D**|**R**|
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cusparseCreate`| | | |`hipsparseCreate`|1.9.2| | |
|`cusparseDestroy`| | | |`hipsparseDestroy`|1.9.2| | |
|`cusparseGetPointerMode`| | | |`hipsparseGetPointerMode`|1.9.2| | |
|`cusparseGetStream`| | | |`hipsparseGetStream`|1.9.2| | |
|`cusparseGetVersion`| | | |`hipsparseGetVersion`|1.9.2| | |
|`cusparseSetPointerMode`| | | |`hipsparseSetPointerMode`|1.9.2| | |
|`cusparseSetStream`| | | |`hipsparseSetStream`|1.9.2| | |

## **6. CUSPARSE Helper Function Reference**

|**CUDA**|**A**|**D**|**R**|**HIP**|**A**|**D**|**R**|
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cusparseCreateBsric02Info`| | | |`hipsparseCreateBsric02Info`|3.8.0| | |
|`cusparseCreateBsrilu02Info`| | | |`hipsparseCreateBsrilu02Info`|3.9.0| | |
|`cusparseCreateBsrsm2Info`| | | | | | | |
|`cusparseCreateBsrsv2Info`| | | |`hipsparseCreateBsrsv2Info`|3.6.0| | |
|`cusparseCreateCsrgemm2Info`| |11.0| |`hipsparseCreateCsrgemm2Info`|2.8.0| | |
|`cusparseCreateCsric02Info`| | | |`hipsparseCreateCsric02Info`|3.1.0| | |
|`cusparseCreateCsrilu02Info`| | | |`hipsparseCreateCsrilu02Info`|1.9.2| | |
|`cusparseCreateCsrsm2Info`|10.0|11.3| |`hipsparseCreateCsrsm2Info`|3.1.0| | |
|`cusparseCreateCsrsv2Info`| |11.3| |`hipsparseCreateCsrsv2Info`|1.9.2| | |
|`cusparseCreateHybMat`| |10.2|11.0|`hipsparseCreateHybMat`|1.9.2| | |
|`cusparseCreateMatDescr`| | | |`hipsparseCreateMatDescr`|1.9.2| | |
|`cusparseCreatePruneInfo`|9.0| | |`hipsparseCreatePruneInfo`|3.9.0| | |
|`cusparseCreateSolveAnalysisInfo`| |10.2|11.0| | | | |
|`cusparseDestroyBsric02Info`| | | |`hipsparseDestroyBsric02Info`|3.8.0| | |
|`cusparseDestroyBsrilu02Info`| | | |`hipsparseDestroyBsrilu02Info`|3.9.0| | |
|`cusparseDestroyBsrsm2Info`| | | | | | | |
|`cusparseDestroyBsrsv2Info`| | | |`hipsparseDestroyBsrsv2Info`|3.6.0| | |
|`cusparseDestroyCsrgemm2Info`| |11.0| |`hipsparseDestroyCsrgemm2Info`|2.8.0| | |
|`cusparseDestroyCsric02Info`| | | |`hipsparseDestroyCsric02Info`|3.1.0| | |
|`cusparseDestroyCsrilu02Info`| | | |`hipsparseDestroyCsrilu02Info`|1.9.2| | |
|`cusparseDestroyCsrsm2Info`|10.0|11.3| |`hipsparseDestroyCsrsm2Info`|3.1.0| | |
|`cusparseDestroyCsrsv2Info`| |11.3| |`hipsparseDestroyCsrsv2Info`|1.9.2| | |
|`cusparseDestroyHybMat`| |10.2|11.0|`hipsparseDestroyHybMat`|1.9.2| | |
|`cusparseDestroyMatDescr`| | | |`hipsparseDestroyMatDescr`|1.9.2| | |
|`cusparseDestroyPruneInfo`|9.0| | |`hipsparseDestroyPruneInfo`|3.9.0| | |
|`cusparseDestroySolveAnalysisInfo`| |10.2|11.0| | | | |
|`cusparseGetLevelInfo`| | |11.0| | | | |
|`cusparseGetMatDiagType`| | | |`hipsparseGetMatDiagType`|1.9.2| | |
|`cusparseGetMatFillMode`| | | |`hipsparseGetMatFillMode`|1.9.2| | |
|`cusparseGetMatIndexBase`| | | |`hipsparseGetMatIndexBase`|1.9.2| | |
|`cusparseGetMatType`| | | |`hipsparseGetMatType`|1.9.2| | |
|`cusparseSetMatDiagType`| | | |`hipsparseSetMatDiagType`|1.9.2| | |
|`cusparseSetMatFillMode`| | | |`hipsparseSetMatFillMode`|1.9.2| | |
|`cusparseSetMatIndexBase`| | | |`hipsparseSetMatIndexBase`|1.9.2| | |
|`cusparseSetMatType`| | | |`hipsparseSetMatType`|1.9.2| | |

## **7. CUSPARSE Level 1 Function Reference**

|**CUDA**|**A**|**D**|**R**|**HIP**|**A**|**D**|**R**|
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cusparseCaxpyi`| |11.0| |`hipsparseCaxpyi`|3.1.0| | |
|`cusparseCdotci`| |10.2|11.0|`hipsparseCdotci`|3.1.0| | |
|`cusparseCdoti`| |10.2|11.0|`hipsparseCdoti`|3.1.0| | |
|`cusparseCgthr`| |11.0| |`hipsparseCgthr`|3.1.0| | |
|`cusparseCgthrz`| |11.0| |`hipsparseCgthrz`|3.1.0| | |
|`cusparseCsctr`| |11.0| |`hipsparseCsctr`|3.1.0| | |
|`cusparseDaxpyi`| |11.0| |`hipsparseDaxpyi`|1.9.2| | |
|`cusparseDdoti`| |10.2|11.0|`hipsparseDdoti`|1.9.2| | |
|`cusparseDgthr`| |11.0| |`hipsparseDgthr`|1.9.2| | |
|`cusparseDgthrz`| |11.0| |`hipsparseDgthrz`|1.9.2| | |
|`cusparseDroti`| |11.0| |`hipsparseDroti`|1.9.2| | |
|`cusparseDsctr`| |11.0| |`hipsparseDsctr`|1.9.2| | |
|`cusparseSaxpyi`| |11.0| |`hipsparseSaxpyi`|1.9.2| | |
|`cusparseSdoti`| |10.2|11.0|`hipsparseSdoti`|1.9.2| | |
|`cusparseSgthr`| |11.0| |`hipsparseSgthr`|1.9.2| | |
|`cusparseSgthrz`| |11.0| |`hipsparseSgthrz`|1.9.2| | |
|`cusparseSroti`| |11.0| |`hipsparseSroti`|1.9.2| | |
|`cusparseSsctr`| |11.0| |`hipsparseSsctr`|1.9.2| | |
|`cusparseZaxpyi`| |11.0| |`hipsparseZaxpyi`|3.1.0| | |
|`cusparseZdotci`| |10.2|11.0|`hipsparseZdotci`|3.1.0| | |
|`cusparseZdoti`| |10.2|11.0|`hipsparseZdoti`|3.1.0| | |
|`cusparseZgthr`| |11.0| |`hipsparseZgthr`|3.1.0| | |
|`cusparseZgthrz`| |11.0| |`hipsparseZgthrz`|3.1.0| | |
|`cusparseZsctr`| |11.0| |`hipsparseZsctr`|3.1.0| | |

## **8. CUSPARSE Level 2 Function Reference**

|**CUDA**|**A**|**D**|**R**|**HIP**|**A**|**D**|**R**|
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cusparseCbsrmv`| | | |`hipsparseCbsrmv`|3.5.0| | |
|`cusparseCbsrsv2_analysis`| | | |`hipsparseCbsrsv2_analysis`|3.6.0| | |
|`cusparseCbsrsv2_bufferSize`| | | |`hipsparseCbsrsv2_bufferSize`|3.6.0| | |
|`cusparseCbsrsv2_bufferSizeExt`| | | |`hipsparseCbsrsv2_bufferSizeExt`|3.6.0| | |
|`cusparseCbsrsv2_solve`| | | |`hipsparseCbsrsv2_solve`|3.6.0| | |
|`cusparseCbsrxmv`| | | | | | | |
|`cusparseCcsrmv`| |10.2|11.0|`hipsparseCcsrmv`|3.1.0| | |
|`cusparseCcsrmv_mp`|8.0|10.2|11.0| | | | |
|`cusparseCcsrsv2_analysis`| |11.3| |`hipsparseCcsrsv2_analysis`|3.1.0| | |
|`cusparseCcsrsv2_bufferSize`| |11.3| |`hipsparseCcsrsv2_bufferSize`|3.1.0| | |
|`cusparseCcsrsv2_bufferSizeExt`| |11.3| |`hipsparseCcsrsv2_bufferSizeExt`|3.1.0| | |
|`cusparseCcsrsv2_solve`| |11.3| |`hipsparseCcsrsv2_solve`|3.1.0| | |
|`cusparseCcsrsv_analysis`| |10.2|11.0| | | | |
|`cusparseCcsrsv_solve`| |10.2|11.0| | | | |
|`cusparseCgemvi`|7.5| | | | | | |
|`cusparseCgemvi_bufferSize`|7.5| | | | | | |
|`cusparseChybmv`| |10.2|11.0|`hipsparseChybmv`|3.1.0| | |
|`cusparseChybsv_analysis`| |10.2|11.0| | | | |
|`cusparseChybsv_solve`| |10.2|11.0| | | | |
|`cusparseCsrmvEx`|8.0| | | | | | |
|`cusparseCsrmvEx_bufferSize`|8.0| | | | | | |
|`cusparseCsrsv_analysisEx`|8.0|10.2|11.0| | | | |
|`cusparseCsrsv_solveEx`|8.0|10.2|11.0| | | | |
|`cusparseDbsrmv`| | | |`hipsparseDbsrmv`|3.5.0| | |
|`cusparseDbsrsv2_analysis`| | | |`hipsparseDbsrsv2_analysis`|3.6.0| | |
|`cusparseDbsrsv2_bufferSize`| | | |`hipsparseDbsrsv2_bufferSize`|3.6.0| | |
|`cusparseDbsrsv2_bufferSizeExt`| | | |`hipsparseDbsrsv2_bufferSizeExt`|3.6.0| | |
|`cusparseDbsrsv2_solve`| | | |`hipsparseDbsrsv2_solve`|3.6.0| | |
|`cusparseDbsrxmv`| | | | | | | |
|`cusparseDcsrmv`| |10.2|11.0|`hipsparseDcsrmv`|1.9.2| | |
|`cusparseDcsrmv_mp`|8.0|10.2|11.0| | | | |
|`cusparseDcsrsv2_analysis`| |11.3| |`hipsparseDcsrsv2_analysis`|1.9.2| | |
|`cusparseDcsrsv2_bufferSize`| | | |`hipsparseDcsrsv2_bufferSize`|1.9.2| | |
|`cusparseDcsrsv2_bufferSizeExt`| |11.3| |`hipsparseDcsrsv2_bufferSizeExt`|1.9.2| | |
|`cusparseDcsrsv2_solve`| |11.3| |`hipsparseDcsrsv2_solve`|1.9.2| | |
|`cusparseDcsrsv_analysis`| |10.2|11.0| | | | |
|`cusparseDcsrsv_solve`| |10.2|11.0| | | | |
|`cusparseDgemvi`|7.5| | | | | | |
|`cusparseDgemvi_bufferSize`|7.5| | | | | | |
|`cusparseDhybmv`| |10.2|11.0|`hipsparseDhybmv`|1.9.2| | |
|`cusparseDhybsv_analysis`| |10.2|11.0| | | | |
|`cusparseDhybsv_solve`| |10.2|11.0| | | | |
|`cusparseSbsrmv`| | | |`hipsparseSbsrmv`|3.5.0| | |
|`cusparseSbsrsv2_analysis`| | | |`hipsparseSbsrsv2_analysis`|3.6.0| | |
|`cusparseSbsrsv2_bufferSize`| | | |`hipsparseSbsrsv2_bufferSize`|3.6.0| | |
|`cusparseSbsrsv2_bufferSizeExt`| | | |`hipsparseSbsrsv2_bufferSizeExt`|3.6.0| | |
|`cusparseSbsrsv2_solve`| | | |`hipsparseSbsrsv2_solve`|3.6.0| | |
|`cusparseSbsrxmv`| | | | | | | |
|`cusparseScsrmv`| |10.2|11.0|`hipsparseScsrmv`|1.9.2| | |
|`cusparseScsrmv_mp`|8.0|10.2|11.0| | | | |
|`cusparseScsrsv2_analysis`| |11.3| |`hipsparseScsrsv2_analysis`|1.9.2| | |
|`cusparseScsrsv2_bufferSize`| |11.3| |`hipsparseScsrsv2_bufferSize`|1.9.2| | |
|`cusparseScsrsv2_bufferSizeExt`| |11.3| |`hipsparseScsrsv2_bufferSizeExt`|1.9.2| | |
|`cusparseScsrsv2_solve`| |11.3| |`hipsparseScsrsv2_solve`|1.9.2| | |
|`cusparseScsrsv_analysis`| |10.2|11.0| | | | |
|`cusparseScsrsv_solve`| |10.2|11.0| | | | |
|`cusparseSgemvi`|7.5| | | | | | |
|`cusparseSgemvi_bufferSize`|7.5| | | | | | |
|`cusparseShybmv`| |10.2|11.0|`hipsparseShybmv`|1.9.2| | |
|`cusparseShybsv_analysis`| |10.2|11.0| | | | |
|`cusparseShybsv_solve`| |10.2|11.0| | | | |
|`cusparseXbsrsv2_zeroPivot`| | | |`hipsparseXbsrsv2_zeroPivot`|3.6.0| | |
|`cusparseXcsrsv2_zeroPivot`| |11.3| |`hipsparseXcsrsv2_zeroPivot`|1.9.2| | |
|`cusparseZbsrmv`| | | |`hipsparseZbsrmv`|3.5.0| | |
|`cusparseZbsrsv2_analysis`| | | |`hipsparseZbsrsv2_analysis`|3.6.0| | |
|`cusparseZbsrsv2_bufferSize`| | | |`hipsparseZbsrsv2_bufferSize`|3.6.0| | |
|`cusparseZbsrsv2_bufferSizeExt`| | | |`hipsparseZbsrsv2_bufferSizeExt`|3.6.0| | |
|`cusparseZbsrsv2_solve`| | | |`hipsparseZbsrsv2_solve`|3.6.0| | |
|`cusparseZbsrxmv`| | | | | | | |
|`cusparseZcsrmv`| |10.2|11.0|`hipsparseZcsrmv`|3.1.0| | |
|`cusparseZcsrmv_mp`|8.0|10.2|11.0| | | | |
|`cusparseZcsrsv2_analysis`| |11.3| |`hipsparseZcsrsv2_analysis`|3.1.0| | |
|`cusparseZcsrsv2_bufferSize`| |11.3| |`hipsparseZcsrsv2_bufferSize`|3.1.0| | |
|`cusparseZcsrsv2_bufferSizeExt`| |11.3| |`hipsparseZcsrsv2_bufferSizeExt`|3.1.0| | |
|`cusparseZcsrsv2_solve`| |11.3| |`hipsparseZcsrsv2_solve`|3.1.0| | |
|`cusparseZcsrsv_analysis`| |10.2|11.0| | | | |
|`cusparseZcsrsv_solve`| |10.2|11.0| | | | |
|`cusparseZgemvi`|7.5| | | | | | |
|`cusparseZgemvi_bufferSize`|7.5| | | | | | |
|`cusparseZhybmv`| |10.2|11.0|`hipsparseZhybmv`|3.1.0| | |
|`cusparseZhybsv_analysis`| |10.2|11.0| | | | |
|`cusparseZhybsv_solve`| |10.2|11.0| | | | |

## **9. CUSPARSE Level 3 Function Reference**

|**CUDA**|**A**|**D**|**R**|**HIP**|**A**|**D**|**R**|
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cusparseCbsrmm`| | | |`hipsparseCbsrmm`|3.7.0| | |
|`cusparseCbsrsm2_analysis`| | | | | | | |
|`cusparseCbsrsm2_bufferSize`| | | | | | | |
|`cusparseCbsrsm2_bufferSizeExt`| | | | | | | |
|`cusparseCbsrsm2_solve`| | | | | | | |
|`cusparseCcsrmm`| |10.2|11.0|`hipsparseCcsrmm`|3.1.0| | |
|`cusparseCcsrmm2`| |10.2|11.0|`hipsparseCcsrmm2`|3.1.0| | |
|`cusparseCcsrsm2_analysis`|10.0|11.3| |`hipsparseCcsrsm2_analysis`|3.1.0| | |
|`cusparseCcsrsm2_bufferSizeExt`|10.0|11.3| |`hipsparseCcsrsm2_bufferSizeExt`|3.1.0| | |
|`cusparseCcsrsm2_solve`|10.0|11.3| |`hipsparseCcsrsm2_solve`|3.1.0| | |
|`cusparseCcsrsm_analysis`| |10.2|11.0| | | | |
|`cusparseCcsrsm_solve`| |10.2|11.0| | | | |
|`cusparseCgemmi`|8.0|11.0| |`hipsparseCgemmi`|3.7.0| | |
|`cusparseDbsrmm`| | | |`hipsparseDbsrmm`|3.7.0| | |
|`cusparseDbsrsm2_analysis`| | | | | | | |
|`cusparseDbsrsm2_bufferSize`| | | | | | | |
|`cusparseDbsrsm2_bufferSizeExt`| | | | | | | |
|`cusparseDbsrsm2_solve`| | | | | | | |
|`cusparseDcsrmm`| |10.2|11.0|`hipsparseDcsrmm`|1.9.2| | |
|`cusparseDcsrmm2`| |10.2|11.0|`hipsparseDcsrmm2`|1.9.2| | |
|`cusparseDcsrsm2_analysis`|10.0|11.3| |`hipsparseDcsrsm2_analysis`|3.1.0| | |
|`cusparseDcsrsm2_bufferSizeExt`|10.0|11.3| |`hipsparseDcsrsm2_bufferSizeExt`|3.1.0| | |
|`cusparseDcsrsm2_solve`|10.0|11.3| |`hipsparseDcsrsm2_solve`|3.1.0| | |
|`cusparseDcsrsm_analysis`| |10.2|11.0| | | | |
|`cusparseDcsrsm_solve`| |10.2|11.0| | | | |
|`cusparseDgemmi`|8.0|11.0| |`hipsparseDgemmi`|3.7.0| | |
|`cusparseSbsrmm`| | | |`hipsparseSbsrmm`|3.7.0| | |
|`cusparseSbsrsm2_analysis`| | | | | | | |
|`cusparseSbsrsm2_bufferSize`| | | | | | | |
|`cusparseSbsrsm2_bufferSizeExt`| | | | | | | |
|`cusparseSbsrsm2_solve`| | | | | | | |
|`cusparseScsrmm`| |10.2|11.0|`hipsparseScsrmm`|1.9.2| | |
|`cusparseScsrmm2`| |10.2|11.0|`hipsparseScsrmm2`|1.9.2| | |
|`cusparseScsrsm2_analysis`|10.0|11.3| |`hipsparseScsrsm2_analysis`|3.1.0| | |
|`cusparseScsrsm2_bufferSizeExt`|10.0|11.3| |`hipsparseScsrsm2_bufferSizeExt`|3.1.0| | |
|`cusparseScsrsm2_solve`|10.0|11.3| |`hipsparseScsrsm2_solve`|3.1.0| | |
|`cusparseScsrsm_analysis`| |10.2|11.0| | | | |
|`cusparseScsrsm_solve`| |10.2|11.0| | | | |
|`cusparseSgemmi`|8.0|11.0| |`hipsparseSgemmi`|3.7.0| | |
|`cusparseXbsrsm2_zeroPivot`| | | | | | | |
|`cusparseXcsrsm2_zeroPivot`|10.0|11.3| |`hipsparseXcsrsm2_zeroPivot`|3.1.0| | |
|`cusparseZbsrmm`| | | |`hipsparseZbsrmm`|3.7.0| | |
|`cusparseZbsrsm2_analysis`| | | | | | | |
|`cusparseZbsrsm2_bufferSize`| | | | | | | |
|`cusparseZbsrsm2_bufferSizeExt`| | | | | | | |
|`cusparseZbsrsm2_solve`| | | | | | | |
|`cusparseZcsrmm`| |10.2|11.0|`hipsparseZcsrmm`|3.1.0| | |
|`cusparseZcsrmm2`| |10.2|11.0|`hipsparseZcsrmm2`|3.1.0| | |
|`cusparseZcsrsm2_analysis`|10.0|11.3| |`hipsparseZcsrsm2_analysis`|3.1.0| | |
|`cusparseZcsrsm2_bufferSizeExt`|10.0|11.3| |`hipsparseZcsrsm2_bufferSizeExt`|3.1.0| | |
|`cusparseZcsrsm2_solve`|10.0|11.3| |`hipsparseZcsrsm2_solve`|3.1.0| | |
|`cusparseZcsrsm_analysis`| |10.2|11.0| | | | |
|`cusparseZcsrsm_solve`| |10.2|11.0| | | | |
|`cusparseZgemmi`|8.0|11.0| |`hipsparseZgemmi`|3.7.0| | |

## **10. CUSPARSE Extra Function Reference**

|**CUDA**|**A**|**D**|**R**|**HIP**|**A**|**D**|**R**|
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cusparseCcsrgeam`| |10.2|11.0|`hipsparseCcsrgeam`|3.5.0| | |
|`cusparseCcsrgeam2`|10.0| | |`hipsparseCcsrgeam2`|3.5.0| | |
|`cusparseCcsrgeam2_bufferSizeExt`|10.0| | |`hipsparseCcsrgeam2_bufferSizeExt`|3.5.0| | |
|`cusparseCcsrgemm`| |10.2|11.0|`hipsparseCcsrgemm`|3.1.0| | |
|`cusparseCcsrgemm2`| |11.0| |`hipsparseCcsrgemm2`|3.1.0| | |
|`cusparseCcsrgemm2_bufferSizeExt`| |11.0| |`hipsparseCcsrgemm2_bufferSizeExt`|3.1.0| | |
|`cusparseDcsrgeam`| |10.2|11.0|`hipsparseDcsrgeam`|3.5.0| | |
|`cusparseDcsrgeam2`|10.0| | |`hipsparseDcsrgeam2`|3.5.0| | |
|`cusparseDcsrgeam2_bufferSizeExt`|10.0| | |`hipsparseDcsrgeam2_bufferSizeExt`|3.5.0| | |
|`cusparseDcsrgemm`| |10.2|11.0|`hipsparseDcsrgemm`|2.8.0| | |
|`cusparseDcsrgemm2`| |11.0| |`hipsparseDcsrgemm2`|2.8.0| | |
|`cusparseDcsrgemm2_bufferSizeExt`| |11.0| |`hipsparseDcsrgemm2_bufferSizeExt`|2.8.0| | |
|`cusparseScsrgeam`| |10.2|11.0|`hipsparseScsrgeam`|3.5.0| | |
|`cusparseScsrgeam2`|10.0| | |`hipsparseScsrgeam2`|3.5.0| | |
|`cusparseScsrgeam2_bufferSizeExt`|10.0| | |`hipsparseScsrgeam2_bufferSizeExt`|3.5.0| | |
|`cusparseScsrgemm`| |10.2|11.0|`hipsparseScsrgemm`|2.8.0| | |
|`cusparseScsrgemm2`| |11.0| |`hipsparseScsrgemm2`|2.8.0| | |
|`cusparseScsrgemm2_bufferSizeExt`| |11.0| |`hipsparseScsrgemm2_bufferSizeExt`|2.8.0| | |
|`cusparseXcsrgeam2Nnz`|10.0| | |`hipsparseXcsrgeam2Nnz`|3.5.0| | |
|`cusparseXcsrgeamNnz`| |10.2|11.0|`hipsparseXcsrgeamNnz`|3.5.0| | |
|`cusparseXcsrgemm2Nnz`| |11.0| |`hipsparseXcsrgemm2Nnz`|2.8.0| | |
|`cusparseXcsrgemmNnz`| |10.2|11.0|`hipsparseXcsrgemmNnz`|2.8.0| | |
|`cusparseZcsrgeam`| |10.2|11.0|`hipsparseZcsrgeam`|3.5.0| | |
|`cusparseZcsrgeam2`|10.0| | |`hipsparseZcsrgeam2`|3.5.0| | |
|`cusparseZcsrgeam2_bufferSizeExt`|10.0| | |`hipsparseZcsrgeam2_bufferSizeExt`|3.5.0| | |
|`cusparseZcsrgemm`| |10.2|11.0|`hipsparseZcsrgemm`|3.1.0| | |
|`cusparseZcsrgemm2`| |11.0| |`hipsparseZcsrgemm2`|3.1.0| | |
|`cusparseZcsrgemm2_bufferSizeExt`| |11.0| |`hipsparseZcsrgemm2_bufferSizeExt`|3.1.0| | |

## **11. CUSPARSE Preconditioners Reference**

|**CUDA**|**A**|**D**|**R**|**HIP**|**A**|**D**|**R**|
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cusparseCbsric02`| | | |`hipsparseCbsric02`|3.8.0| | |
|`cusparseCbsric02_analysis`| | | |`hipsparseCbsric02_analysis`|3.8.0| | |
|`cusparseCbsric02_bufferSize`| | | |`hipsparseCbsric02_bufferSize`|3.8.0| | |
|`cusparseCbsric02_bufferSizeExt`| | | | | | | |
|`cusparseCbsrilu02`| | | |`hipsparseCbsrilu02`|3.9.0| | |
|`cusparseCbsrilu02_analysis`| | | |`hipsparseCbsrilu02_analysis`| | | |
|`cusparseCbsrilu02_bufferSize`| | | |`hipsparseCbsrilu02_bufferSize`|3.9.0| | |
|`cusparseCbsrilu02_bufferSizeExt`| | | | | | | |
|`cusparseCbsrilu02_numericBoost`| | | |`hipsparseCbsrilu02_numericBoost`|3.9.0| | |
|`cusparseCcsric0`| |10.2|11.0| | | | |
|`cusparseCcsric02`| | | |`hipsparseCcsric02`|3.1.0| | |
|`cusparseCcsric02_analysis`| | | |`hipsparseCcsric02_analysis`|3.1.0| | |
|`cusparseCcsric02_bufferSize`| | | |`hipsparseCcsric02_bufferSize`|3.1.0| | |
|`cusparseCcsric02_bufferSizeExt`| | | |`hipsparseCcsric02_bufferSizeExt`|3.1.0| | |
|`cusparseCcsrilu0`| |10.2|11.0| | | | |
|`cusparseCcsrilu02`| | | |`hipsparseCcsrilu02`|3.1.0| | |
|`cusparseCcsrilu02_analysis`| | | |`hipsparseCcsrilu02_analysis`|3.1.0| | |
|`cusparseCcsrilu02_bufferSize`| | | |`hipsparseCcsrilu02_bufferSize`|3.1.0| | |
|`cusparseCcsrilu02_bufferSizeExt`| | | |`hipsparseCcsrilu02_bufferSizeExt`|3.1.0| | |
|`cusparseCcsrilu02_numericBoost`| | | |`hipsparseCcsrilu02_numericBoost`|3.10.0| | |
|`cusparseCgpsvInterleavedBatch`|9.2| | | | | | |
|`cusparseCgpsvInterleavedBatch_bufferSizeExt`|9.2| | | | | | |
|`cusparseCgtsv`| |10.2|11.0| | | | |
|`cusparseCgtsv2`|9.0| | | | | | |
|`cusparseCgtsv2StridedBatch`| | | | | | | |
|`cusparseCgtsv2StridedBatch_bufferSizeExt`| | | | | | | |
|`cusparseCgtsv2_bufferSizeExt`|9.0| | | | | | |
|`cusparseCgtsv2_nopivot`|9.0| | | | | | |
|`cusparseCgtsv2_nopivot_bufferSizeExt`|9.0| | | | | | |
|`cusparseCgtsvInterleavedBatch`|9.2| | | | | | |
|`cusparseCgtsvInterleavedBatch_bufferSizeExt`|9.2| | | | | | |
|`cusparseCgtsvStridedBatch`| |10.2|11.0| | | | |
|`cusparseCgtsv_nopivot`| |10.2|11.0| | | | |
|`cusparseCsrilu0Ex`|8.0|10.2|11.0| | | | |
|`cusparseDbsric02`| | | |`hipsparseDbsric02`|3.8.0| | |
|`cusparseDbsric02_analysis`| | | |`hipsparseDbsric02_analysis`|3.8.0| | |
|`cusparseDbsric02_bufferSize`| | | |`hipsparseDbsric02_bufferSize`|3.8.0| | |
|`cusparseDbsric02_bufferSizeExt`| | | | | | | |
|`cusparseDbsrilu02`| | | |`hipsparseDbsrilu02`|3.9.0| | |
|`cusparseDbsrilu02_analysis`| | | |`hipsparseDbsrilu02_analysis`|3.9.0| | |
|`cusparseDbsrilu02_bufferSize`| | | |`hipsparseDbsrilu02_bufferSize`|3.9.0| | |
|`cusparseDbsrilu02_bufferSizeExt`| | | | | | | |
|`cusparseDbsrilu02_numericBoost`| | | |`hipsparseDbsrilu02_numericBoost`|3.9.0| | |
|`cusparseDcsric0`| |10.2|11.0| | | | |
|`cusparseDcsric02`| | | |`hipsparseDcsric02`|3.1.0| | |
|`cusparseDcsric02_analysis`| | | |`hipsparseDcsric02_analysis`|3.1.0| | |
|`cusparseDcsric02_bufferSize`| | | |`hipsparseDcsric02_bufferSize`|3.1.0| | |
|`cusparseDcsric02_bufferSizeExt`| | | |`hipsparseDcsric02_bufferSizeExt`|3.1.0| | |
|`cusparseDcsrilu0`| |10.2|11.0| | | | |
|`cusparseDcsrilu02`| | | |`hipsparseDcsrilu02`|1.9.2| | |
|`cusparseDcsrilu02_analysis`| | | |`hipsparseDcsrilu02_analysis`|1.9.2| | |
|`cusparseDcsrilu02_bufferSize`| | | |`hipsparseDcsrilu02_bufferSize`|1.9.2| | |
|`cusparseDcsrilu02_bufferSizeExt`| | | |`hipsparseDcsrilu02_bufferSizeExt`|1.9.2| | |
|`cusparseDcsrilu02_numericBoost`| | | |`hipsparseDcsrilu02_numericBoost`|3.10.0| | |
|`cusparseDgpsvInterleavedBatch`|9.2| | | | | | |
|`cusparseDgpsvInterleavedBatch_bufferSizeExt`|9.2| | | | | | |
|`cusparseDgtsv`| |10.2|11.0| | | | |
|`cusparseDgtsv2`|9.0| | | | | | |
|`cusparseDgtsv2StridedBatch`| | | | | | | |
|`cusparseDgtsv2StridedBatch_bufferSizeExt`| | | | | | | |
|`cusparseDgtsv2_bufferSizeExt`|9.0| | | | | | |
|`cusparseDgtsv2_nopivot`|9.0| | | | | | |
|`cusparseDgtsv2_nopivot_bufferSizeExt`|9.0| | | | | | |
|`cusparseDgtsvInterleavedBatch`|9.2| | | | | | |
|`cusparseDgtsvInterleavedBatch_bufferSizeExt`|9.2| | | | | | |
|`cusparseDgtsvStridedBatch`| |10.2|11.0| | | | |
|`cusparseDgtsv_nopivot`| |10.2|11.0| | | | |
|`cusparseSbsric02`| | | |`hipsparseSbsric02`|3.8.0| | |
|`cusparseSbsric02_analysis`| | | |`hipsparseSbsric02_analysis`|3.8.0| | |
|`cusparseSbsric02_bufferSize`| | | |`hipsparseSbsric02_bufferSize`|3.8.0| | |
|`cusparseSbsric02_bufferSizeExt`| | | | | | | |
|`cusparseSbsrilu02`| | | |`hipsparseSbsrilu02`|3.9.0| | |
|`cusparseSbsrilu02_analysis`| | | |`hipsparseSbsrilu02_analysis`|3.9.0| | |
|`cusparseSbsrilu02_bufferSize`| | | |`hipsparseSbsrilu02_bufferSize`|3.9.0| | |
|`cusparseSbsrilu02_bufferSizeExt`| | | | | | | |
|`cusparseSbsrilu02_numericBoost`| | | |`hipsparseSbsrilu02_numericBoost`|3.9.0| | |
|`cusparseScsric0`| |10.2|11.0| | | | |
|`cusparseScsric02`| | | |`hipsparseScsric02`|3.1.0| | |
|`cusparseScsric02_analysis`| | | |`hipsparseScsric02_analysis`|3.1.0| | |
|`cusparseScsric02_bufferSize`| | | |`hipsparseScsric02_bufferSize`|3.1.0| | |
|`cusparseScsric02_bufferSizeExt`| | | |`hipsparseScsric02_bufferSizeExt`|3.1.0| | |
|`cusparseScsrilu0`| |10.2|11.0| | | | |
|`cusparseScsrilu02`| | | |`hipsparseScsrilu02`|1.9.2| | |
|`cusparseScsrilu02_analysis`| | | |`hipsparseScsrilu02_analysis`|1.9.2| | |
|`cusparseScsrilu02_bufferSize`| | | |`hipsparseScsrilu02_bufferSize`|1.9.2| | |
|`cusparseScsrilu02_bufferSizeExt`| | | |`hipsparseScsrilu02_bufferSizeExt`|1.9.2| | |
|`cusparseScsrilu02_numericBoost`| | | |`hipsparseScsrilu02_numericBoost`|3.10.0| | |
|`cusparseSgpsvInterleavedBatch`|9.2| | | | | | |
|`cusparseSgpsvInterleavedBatch_bufferSizeExt`|9.2| | | | | | |
|`cusparseSgtsv`| |10.2|11.0| | | | |
|`cusparseSgtsv2`|9.0| | | | | | |
|`cusparseSgtsv2StridedBatch`|9.0| | | | | | |
|`cusparseSgtsv2StridedBatch_bufferSizeExt`|9.0| | | | | | |
|`cusparseSgtsv2_bufferSizeExt`|9.0| | | | | | |
|`cusparseSgtsv2_nopivot`|9.0| | | | | | |
|`cusparseSgtsv2_nopivot_bufferSizeExt`|9.0| | | | | | |
|`cusparseSgtsvInterleavedBatch`|9.2| | | | | | |
|`cusparseSgtsvInterleavedBatch_bufferSizeExt`|9.2| | | | | | |
|`cusparseSgtsvStridedBatch`| |10.2|11.0| | | | |
|`cusparseSgtsv_nopivot`| |10.2|11.0| | | | |
|`cusparseXbsric02_zeroPivot`| | | |`hipsparseXbsric02_zeroPivot`|3.8.0| | |
|`cusparseXbsrilu02_zeroPivot`| | | |`hipsparseXbsrilu02_zeroPivot`|3.9.0| | |
|`cusparseXcsric02_zeroPivot`| | | |`hipsparseXcsric02_zeroPivot`|3.1.0| | |
|`cusparseXcsrilu02_zeroPivot`| | | |`hipsparseXcsrilu02_zeroPivot`|1.9.2| | |
|`cusparseZbsric02`| | | |`hipsparseZbsric02`|3.8.0| | |
|`cusparseZbsric02_analysis`| | | |`hipsparseZbsric02_analysis`|3.8.0| | |
|`cusparseZbsric02_bufferSize`| | | |`hipsparseZbsric02_bufferSize`|3.8.0| | |
|`cusparseZbsric02_bufferSizeExt`| | | | | | | |
|`cusparseZbsrilu02`| | | |`hipsparseZbsrilu02`|3.9.0| | |
|`cusparseZbsrilu02_analysis`| | | |`hipsparseZbsrilu02_analysis`|3.9.0| | |
|`cusparseZbsrilu02_bufferSize`| | | |`hipsparseZbsrilu02_bufferSize`|3.9.0| | |
|`cusparseZbsrilu02_bufferSizeExt`| | | | | | | |
|`cusparseZbsrilu02_numericBoost`| | | |`hipsparseZbsrilu02_numericBoost`|3.9.0| | |
|`cusparseZcsric0`| |10.2|11.0| | | | |
|`cusparseZcsric02`| | | |`hipsparseZcsric02`|3.1.0| | |
|`cusparseZcsric02_analysis`| | | |`hipsparseZcsric02_analysis`|3.1.0| | |
|`cusparseZcsric02_bufferSize`| | | |`hipsparseZcsric02_bufferSize`|3.1.0| | |
|`cusparseZcsric02_bufferSizeExt`| | | |`hipsparseZcsric02_bufferSizeExt`|3.1.0| | |
|`cusparseZcsrilu0`| |10.2|11.0| | | | |
|`cusparseZcsrilu02`| | | |`hipsparseZcsrilu02`|3.1.0| | |
|`cusparseZcsrilu02_analysis`| | | |`hipsparseZcsrilu02_analysis`|3.1.0| | |
|`cusparseZcsrilu02_bufferSize`| | | |`hipsparseZcsrilu02_bufferSize`|3.1.0| | |
|`cusparseZcsrilu02_bufferSizeExt`| | | |`hipsparseZcsrilu02_bufferSizeExt`|3.1.0| | |
|`cusparseZcsrilu02_numericBoost`| | | |`hipsparseZcsrilu02_numericBoost`|3.10.0| | |
|`cusparseZgpsvInterleavedBatch`|9.2| | | | | | |
|`cusparseZgpsvInterleavedBatch_bufferSizeExt`|9.2| | | | | | |
|`cusparseZgtsv`| |10.2|11.0| | | | |
|`cusparseZgtsv2`|9.0| | | | | | |
|`cusparseZgtsv2StridedBatch`| | | | | | | |
|`cusparseZgtsv2StridedBatch_bufferSizeExt`| | | | | | | |
|`cusparseZgtsv2_bufferSizeExt`|9.0| | | | | | |
|`cusparseZgtsv2_nopivot`|9.0| | | | | | |
|`cusparseZgtsv2_nopivot_bufferSizeExt`|9.0| | | | | | |
|`cusparseZgtsvInterleavedBatch`|9.2| | | | | | |
|`cusparseZgtsvInterleavedBatch_bufferSizeExt`|9.2| | | | | | |
|`cusparseZgtsvStridedBatch`| |10.2|11.0| | | | |
|`cusparseZgtsv_nopivot`| |10.2|11.0| | | | |

## **12. CUSPARSE Reorderings Reference**

|**CUDA**|**A**|**D**|**R**|**HIP**|**A**|**D**|**R**|
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cusparseCcsrcolor`| | | | | | | |
|`cusparseDcsrcolor`| | | | | | | |
|`cusparseScsrcolor`| | | | | | | |
|`cusparseZcsrcolor`| | | | | | | |

## **13. CUSPARSE Format Conversion Reference**

|**CUDA**|**A**|**D**|**R**|**HIP**|**A**|**D**|**R**|
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cusparseCbsr2csr`| | | |`hipsparseCbsr2csr`|3.5.0| | |
|`cusparseCcsc2dense`| |11.1| |`hipsparseCcsc2dense`|3.5.0| | |
|`cusparseCcsc2hyb`| |10.2|11.0| | | | |
|`cusparseCcsr2bsr`| | | |`hipsparseCcsr2bsr`|3.5.0| | |
|`cusparseCcsr2csc`| |10.2|11.0|`hipsparseCcsr2csc`|3.1.0| | |
|`cusparseCcsr2csr_compress`|8.0| | |`hipsparseCcsr2csr_compress`|3.5.0| | |
|`cusparseCcsr2csru`| | | |`hipsparseCcsr2csru`|4.2.0| | |
|`cusparseCcsr2dense`| |11.1| |`hipsparseCcsr2dense`|3.5.0| | |
|`cusparseCcsr2gebsr`| | | |`hipsparseCcsr2gebsr`|4.1.0| | |
|`cusparseCcsr2gebsr_bufferSize`| | | |`hipsparseCcsr2gebsr_bufferSize`|4.1.0| | |
|`cusparseCcsr2gebsr_bufferSizeExt`| | | | | | | |
|`cusparseCcsr2hyb`| |10.2|11.0|`hipsparseCcsr2hyb`|3.1.0| | |
|`cusparseCcsru2csr`| | | |`hipsparseCcsru2csr`|4.2.0| | |
|`cusparseCcsru2csr_bufferSizeExt`| | | |`hipsparseCcsru2csr_bufferSizeExt`|4.2.0| | |
|`cusparseCdense2csc`| |11.1| |`hipsparseCdense2csc`|3.5.0| | |
|`cusparseCdense2csr`| |11.1| |`hipsparseCdense2csr`|3.5.0| | |
|`cusparseCdense2hyb`| |10.2|11.0| | | | |
|`cusparseCgebsr2csr`| | | |`hipsparseCgebsr2csr`|4.1.0| | |
|`cusparseCgebsr2gebsc`| | | |`hipsparseCgebsr2gebsc`|4.1.0| | |
|`cusparseCgebsr2gebsc_bufferSize`| | | |`hipsparseCgebsr2gebsc_bufferSize`|4.1.0| | |
|`cusparseCgebsr2gebsc_bufferSizeExt`| | | | | | | |
|`cusparseCgebsr2gebsr`| | | |`hipsparseCgebsr2gebsr`|4.1.0| | |
|`cusparseCgebsr2gebsr_bufferSize`| | | |`hipsparseCgebsr2gebsr_bufferSize`|4.1.0| | |
|`cusparseCgebsr2gebsr_bufferSizeExt`| | | | | | | |
|`cusparseChyb2csc`| |10.2|11.0| | | | |
|`cusparseChyb2csr`| |10.2|11.0|`hipsparseChyb2csr`|3.1.0| | |
|`cusparseChyb2dense`| |10.2|11.0| | | | |
|`cusparseCnnz`| | | |`hipsparseCnnz`|3.2.0| | |
|`cusparseCnnz_compress`|8.0| | |`hipsparseCnnz_compress`|3.5.0| | |
|`cusparseCreateCsru2csrInfo`| | | |`hipsparseCreateCsru2csrInfo`|4.2.0| | |
|`cusparseCreateIdentityPermutation`| | | |`hipsparseCreateIdentityPermutation`|1.9.2| | |
|`cusparseCsr2cscEx`|8.0|10.2|11.0| | | | |
|`cusparseCsr2cscEx2`|10.1| | | | | | |
|`cusparseCsr2cscEx2_bufferSize`|10.1| | | | | | |
|`cusparseDbsr2csr`| | | |`hipsparseDbsr2csr`|3.5.0| | |
|`cusparseDcsc2dense`| |11.1| |`hipsparseDcsc2dense`|3.5.0| | |
|`cusparseDcsc2hyb`| |10.2|11.0| | | | |
|`cusparseDcsr2bsr`| | | |`hipsparseDcsr2bsr`|3.5.0| | |
|`cusparseDcsr2csc`| |10.2|11.0|`hipsparseDcsr2csc`|1.9.2| | |
|`cusparseDcsr2csr_compress`|8.0| | |`hipsparseDcsr2csr_compress`|3.5.0| | |
|`cusparseDcsr2csru`| | | |`hipsparseDcsr2csru`|4.2.0| | |
|`cusparseDcsr2dense`| |11.1| |`hipsparseDcsr2dense`|3.5.0| | |
|`cusparseDcsr2gebsr`| | | |`hipsparseDcsr2gebsr`|4.1.0| | |
|`cusparseDcsr2gebsr_bufferSize`| | | |`hipsparseDcsr2gebsr_bufferSize`|4.1.0| | |
|`cusparseDcsr2gebsr_bufferSizeExt`| | | | | | | |
|`cusparseDcsr2hyb`| |10.2|11.0|`hipsparseDcsr2hyb`|1.9.2| | |
|`cusparseDcsru2csr`| | | |`hipsparseDcsru2csr`|4.2.0| | |
|`cusparseDcsru2csr_bufferSizeExt`| | | |`hipsparseDcsru2csr_bufferSizeExt`|4.2.0| | |
|`cusparseDdense2csc`| |11.1| |`hipsparseDdense2csc`|3.5.0| | |
|`cusparseDdense2csr`| |11.1| |`hipsparseDdense2csr`|3.5.0| | |
|`cusparseDdense2hyb`| |10.2|11.0| | | | |
|`cusparseDestroyCsru2csrInfo`| | | |`hipsparseDestroyCsru2csrInfo`|4.2.0| | |
|`cusparseDgebsr2csr`| | | |`hipsparseDgebsr2csr`|4.1.0| | |
|`cusparseDgebsr2gebsc`| | | |`hipsparseDgebsr2gebsc`|4.1.0| | |
|`cusparseDgebsr2gebsc_bufferSize`| | | |`hipsparseDgebsr2gebsc_bufferSize`|4.1.0| | |
|`cusparseDgebsr2gebsc_bufferSizeExt`| | | | | | | |
|`cusparseDgebsr2gebsr`| | | |`hipsparseDgebsr2gebsr`|4.1.0| | |
|`cusparseDgebsr2gebsr_bufferSize`| | | |`hipsparseDgebsr2gebsr_bufferSize`|4.1.0| | |
|`cusparseDgebsr2gebsr_bufferSizeExt`| | | | | | | |
|`cusparseDhyb2csc`| |10.2|11.0| | | | |
|`cusparseDhyb2csr`| |10.2|11.0|`hipsparseDhyb2csr`|3.1.0| | |
|`cusparseDhyb2dense`| |10.2|11.0| | | | |
|`cusparseDnnz`| | | |`hipsparseDnnz`|3.2.0| | |
|`cusparseDnnz_compress`|8.0| | |`hipsparseDnnz_compress`|3.5.0| | |
|`cusparseDpruneCsr2csr`|9.0| | |`hipsparseDpruneCsr2csr`|3.9.0| | |
|`cusparseDpruneCsr2csrByPercentage`|9.0| | |`hipsparseDpruneCsr2csrByPercentage`|3.9.0| | |
|`cusparseDpruneCsr2csrByPercentage_bufferSizeExt`|9.0| | |`hipsparseDpruneCsr2csrByPercentage_bufferSizeExt`|3.9.0| | |
|`cusparseDpruneCsr2csrNnz`|9.0| | |`hipsparseDpruneCsr2csrNnz`|3.9.0| | |
|`cusparseDpruneCsr2csrNnzByPercentage`|9.0| | |`hipsparseDpruneCsr2csrNnzByPercentage`|3.9.0| | |
|`cusparseDpruneCsr2csr_bufferSizeExt`|9.0| | |`hipsparseDpruneCsr2csr_bufferSizeExt`|3.9.0| | |
|`cusparseDpruneDense2csr`|9.0| | |`hipsparseDpruneDense2csr`|3.9.0| | |
|`cusparseDpruneDense2csrByPercentage`|9.0| | |`hipsparseDpruneDense2csrByPercentage`|3.9.0| | |
|`cusparseDpruneDense2csrByPercentage_bufferSizeExt`|9.0| | |`hipsparseDpruneDense2csrByPercentage_bufferSizeExt`|3.9.0| | |
|`cusparseDpruneDense2csrNnz`|9.0| | |`hipsparseDpruneDense2csrNnz`|3.9.0| | |
|`cusparseDpruneDense2csrNnzByPercentage`|9.0| | |`hipsparseDpruneDense2csrNnzByPercentage`|3.9.0| | |
|`cusparseDpruneDense2csr_bufferSizeExt`|9.0| | |`hipsparseDpruneDense2csr_bufferSizeExt`|3.9.0| | |
|`cusparseHpruneCsr2csr`|9.0| | | | | | |
|`cusparseHpruneCsr2csrByPercentage`|9.0| | | | | | |
|`cusparseHpruneCsr2csrByPercentage_bufferSizeExt`|9.0| | | | | | |
|`cusparseHpruneCsr2csrNnz`|9.0| | | | | | |
|`cusparseHpruneCsr2csrNnzByPercentage`|9.0| | | | | | |
|`cusparseHpruneCsr2csr_bufferSizeExt`|9.0| | | | | | |
|`cusparseHpruneDense2csr`|9.0| | | | | | |
|`cusparseHpruneDense2csrByPercentage`|9.0| | | | | | |
|`cusparseHpruneDense2csrByPercentage_bufferSizeExt`|9.0| | | | | | |
|`cusparseHpruneDense2csrNnz`|9.0| | | | | | |
|`cusparseHpruneDense2csrNnzByPercentage`|9.0| | | | | | |
|`cusparseHpruneDense2csr_bufferSizeExt`|9.0| | | | | | |
|`cusparseSbsr2csr`| | | |`hipsparseSbsr2csr`|3.5.0| | |
|`cusparseScsc2dense`| |11.1| |`hipsparseScsc2dense`|3.5.0| | |
|`cusparseScsc2hyb`| |10.2|11.0| | | | |
|`cusparseScsr2bsr`| | | |`hipsparseScsr2bsr`|3.5.0| | |
|`cusparseScsr2csc`| |10.2|11.0|`hipsparseScsr2csc`|1.9.2| | |
|`cusparseScsr2csr_compress`|8.0| | |`hipsparseScsr2csr_compress`|3.5.0| | |
|`cusparseScsr2csru`| | | |`hipsparseScsr2csru`|4.2.0| | |
|`cusparseScsr2dense`| |11.1| |`hipsparseScsr2dense`|3.5.0| | |
|`cusparseScsr2gebsr`| | | |`hipsparseScsr2gebsr`|4.1.0| | |
|`cusparseScsr2gebsr_bufferSize`| | | |`hipsparseScsr2gebsr_bufferSize`|4.1.0| | |
|`cusparseScsr2gebsr_bufferSizeExt`| | | | | | | |
|`cusparseScsr2hyb`| |10.2|11.0|`hipsparseScsr2hyb`|1.9.2| | |
|`cusparseScsru2csr`| | | |`hipsparseScsru2csr`|4.2.0| | |
|`cusparseScsru2csr_bufferSizeExt`| | | |`hipsparseScsru2csr_bufferSizeExt`|4.2.0| | |
|`cusparseSdense2csc`| |11.1| |`hipsparseSdense2csc`|3.5.0| | |
|`cusparseSdense2csr`| |11.1| |`hipsparseSdense2csr`|3.5.0| | |
|`cusparseSdense2hyb`| |10.2|11.0| | | | |
|`cusparseSgebsr2csr`| | | |`hipsparseSgebsr2csr`|4.1.0| | |
|`cusparseSgebsr2gebsc`| | | |`hipsparseSgebsr2gebsc`|4.1.0| | |
|`cusparseSgebsr2gebsc_bufferSize`| | | |`hipsparseSgebsr2gebsc_bufferSize`|4.1.0| | |
|`cusparseSgebsr2gebsc_bufferSizeExt`| | | | | | | |
|`cusparseSgebsr2gebsr`| | | |`hipsparseSgebsr2gebsr`|4.1.0| | |
|`cusparseSgebsr2gebsr_bufferSize`| | | |`hipsparseSgebsr2gebsr_bufferSize`|4.1.0| | |
|`cusparseSgebsr2gebsr_bufferSizeExt`| | | | | | | |
|`cusparseShyb2csc`| |10.2|11.0| | | | |
|`cusparseShyb2csr`| |10.2|11.0|`hipsparseShyb2csr`|3.1.0| | |
|`cusparseShyb2dense`| |10.2|11.0| | | | |
|`cusparseSnnz`| | | |`hipsparseSnnz`|3.2.0| | |
|`cusparseSnnz_compress`|8.0| | |`hipsparseSnnz_compress`|3.5.0| | |
|`cusparseSpruneCsr2csr`|9.0| | |`hipsparseSpruneCsr2csr`|3.9.0| | |
|`cusparseSpruneCsr2csrByPercentage`|9.0| | |`hipsparseSpruneCsr2csrByPercentage`|3.9.0| | |
|`cusparseSpruneCsr2csrByPercentage_bufferSizeExt`|9.0| | |`hipsparseSpruneCsr2csrByPercentage_bufferSizeExt`|3.9.0| | |
|`cusparseSpruneCsr2csrNnz`|9.0| | |`hipsparseSpruneCsr2csrNnz`|3.9.0| | |
|`cusparseSpruneCsr2csrNnzByPercentage`|9.0| | |`hipsparseSpruneCsr2csrNnzByPercentage`|3.9.0| | |
|`cusparseSpruneCsr2csr_bufferSizeExt`|9.0| | |`hipsparseSpruneCsr2csr_bufferSizeExt`|3.9.0| | |
|`cusparseSpruneDense2csr`|9.0| | |`hipsparseSpruneDense2csr`|3.9.0| | |
|`cusparseSpruneDense2csrByPercentage`|9.0| | |`hipsparseSpruneDense2csrByPercentage`|3.9.0| | |
|`cusparseSpruneDense2csrByPercentage_bufferSizeExt`|9.0| | |`hipsparseSpruneDense2csrByPercentage_bufferSizeExt`|3.9.0| | |
|`cusparseSpruneDense2csrNnz`|9.0| | |`hipsparseSpruneDense2csrNnz`|3.9.0| | |
|`cusparseSpruneDense2csrNnzByPercentage`|9.0| | |`hipsparseSpruneDense2csrNnzByPercentage`|3.9.0| | |
|`cusparseSpruneDense2csr_bufferSizeExt`|9.0| | |`hipsparseSpruneDense2csr_bufferSizeExt`|3.9.0| | |
|`cusparseXcoo2csr`| | | |`hipsparseXcoo2csr`|1.9.2| | |
|`cusparseXcoosortByColumn`| | | |`hipsparseXcoosortByColumn`|1.9.2| | |
|`cusparseXcoosortByRow`| | | |`hipsparseXcoosortByRow`|1.9.2| | |
|`cusparseXcoosort_bufferSizeExt`| | | |`hipsparseXcoosort_bufferSizeExt`|1.9.2| | |
|`cusparseXcscsort`| | | |`hipsparseXcscsort`|2.10.0| | |
|`cusparseXcscsort_bufferSizeExt`| | | |`hipsparseXcscsort_bufferSizeExt`|2.10.0| | |
|`cusparseXcsr2bsrNnz`| | | |`hipsparseXcsr2bsrNnz`|3.5.0| | |
|`cusparseXcsr2coo`| | | |`hipsparseXcsr2coo`|1.9.2| | |
|`cusparseXcsr2gebsrNnz`| | | |`hipsparseXcsr2gebsrNnz`|4.1.0| | |
|`cusparseXcsrsort`| | | |`hipsparseXcsrsort`|1.9.2| | |
|`cusparseXcsrsort_bufferSizeExt`| | | |`hipsparseXcsrsort_bufferSizeExt`|1.9.2| | |
|`cusparseXgebsr2csr`| | | | | | | |
|`cusparseXgebsr2gebsrNnz`| | | |`hipsparseXgebsr2gebsrNnz`|4.1.0| | |
|`cusparseZbsr2csr`| | | |`hipsparseZbsr2csr`|3.5.0| | |
|`cusparseZcsc2dense`| |11.1| |`hipsparseZcsc2dense`|3.5.0| | |
|`cusparseZcsc2hyb`| |10.2|11.0| | | | |
|`cusparseZcsr2bsr`| | | |`hipsparseZcsr2bsr`|3.5.0| | |
|`cusparseZcsr2csc`| |10.2|11.0|`hipsparseZcsr2csc`|3.1.0| | |
|`cusparseZcsr2csr_compress`|8.0| | |`hipsparseZcsr2csr_compress`|3.5.0| | |
|`cusparseZcsr2csru`| | | |`hipsparseZcsr2csru`|4.2.0| | |
|`cusparseZcsr2dense`| |11.1| |`hipsparseZcsr2dense`|3.5.0| | |
|`cusparseZcsr2gebsr`| | | |`hipsparseZcsr2gebsr`|4.1.0| | |
|`cusparseZcsr2gebsr_bufferSize`| | | |`hipsparseZcsr2gebsr_bufferSize`|4.1.0| | |
|`cusparseZcsr2gebsr_bufferSizeExt`| | | | | | | |
|`cusparseZcsr2hyb`| |10.2|11.0|`hipsparseZcsr2hyb`|3.1.0| | |
|`cusparseZcsru2csr`| | | |`hipsparseZcsru2csr`|4.2.0| | |
|`cusparseZcsru2csr_bufferSizeExt`| | | |`hipsparseZcsru2csr_bufferSizeExt`|4.2.0| | |
|`cusparseZdense2csc`| |11.1| |`hipsparseZdense2csc`|3.5.0| | |
|`cusparseZdense2csr`| |11.1| |`hipsparseZdense2csr`|3.5.0| | |
|`cusparseZdense2hyb`| |10.2|11.0| | | | |
|`cusparseZgebsr2csr`| | | |`hipsparseZgebsr2csr`|4.1.0| | |
|`cusparseZgebsr2gebsc`| | | |`hipsparseZgebsr2gebsc`|4.1.0| | |
|`cusparseZgebsr2gebsc_bufferSize`| | | |`hipsparseZgebsr2gebsc_bufferSize`|4.1.0| | |
|`cusparseZgebsr2gebsc_bufferSizeExt`| | | | | | | |
|`cusparseZgebsr2gebsr`| | | |`hipsparseZgebsr2gebsr`|4.1.0| | |
|`cusparseZgebsr2gebsr_bufferSize`| | | |`hipsparseZgebsr2gebsr_bufferSize`|4.1.0| | |
|`cusparseZgebsr2gebsr_bufferSizeExt`| | | | | | | |
|`cusparseZhyb2csc`| |10.2|11.0| | | | |
|`cusparseZhyb2csr`| |10.2|11.0|`hipsparseZhyb2csr`|3.1.0| | |
|`cusparseZhyb2dense`| |10.2|11.0| | | | |
|`cusparseZnnz`| | | |`hipsparseZnnz`|3.2.0| | |
|`cusparseZnnz_compress`|8.0| | |`hipsparseZnnz_compress`|3.5.0| | |

## **14. CUSPARSE Generic API Reference**

|**CUDA**|**A**|**D**|**R**|**HIP**|**A**|**D**|**R**|
|:--|:-:|:-:|:-:|:--|:-:|:-:|:-:|
|`cusparseAxpby`|11.0| | |`hipsparseAxpby`|4.1.0| | |
|`cusparseConstrainedGeMM`|10.2| | | | | | |
|`cusparseConstrainedGeMM_bufferSize`|10.2| | | | | | |
|`cusparseCooAoSGet`|10.2| | |`hipsparseCooAoSGet`|4.1.0| | |
|`cusparseCooGet`|10.1| | |`hipsparseCooGet`|4.1.0| | |
|`cusparseCooSetPointers`|11.1| | |`hipsparseCooSetPointers`|4.2.0| | |
|`cusparseCooSetStridedBatch`|11.0| | | | | | |
|`cusparseCreateCoo`|10.1| | |`hipsparseCreateCoo`|4.1.0| | |
|`cusparseCreateCooAoS`|10.2| | |`hipsparseCreateCooAoS`|4.1.0| | |
|`cusparseCreateCsc`|11.1| | |`hipsparseCreateCsc`|4.2.0| | |
|`cusparseCreateCsr`|10.2| | |`hipsparseCreateCsr`|4.1.0| | |
|`cusparseCreateDnMat`|10.1| | |`hipsparseCreateDnMat`|4.2.0| | |
|`cusparseCreateDnVec`|10.2| | |`hipsparseCreateDnVec`|4.1.0| | |
|`cusparseCreateSpVec`|10.2| | |`hipsparseCreateSpVec`|4.1.0| | |
|`cusparseCscSetPointers`|11.1| | |`hipsparseCscSetPointers`|4.2.0| | |
|`cusparseCsrGet`|10.2| | |`hipsparseCsrGet`|4.1.0| | |
|`cusparseCsrSetPointers`|11.0| | |`hipsparseCsrSetPointers`|4.1.0| | |
|`cusparseCsrSetStridedBatch`|11.0| | | | | | |
|`cusparseDenseToSparse_analysis`|11.1| | |`hipsparseDenseToSparse_analysis`|4.2.0| | |
|`cusparseDenseToSparse_bufferSize`|11.1| | |`hipsparseDenseToSparse_bufferSize`|4.2.0| | |
|`cusparseDenseToSparse_convert`|11.1| | |`hipsparseDenseToSparse_convert`|4.2.0| | |
|`cusparseDestroyDnMat`|10.1| | |`hipsparseDestroyDnMat`|4.2.0| | |
|`cusparseDestroyDnVec`|10.2| | |`hipsparseDestroyDnVec`|4.1.0| | |
|`cusparseDestroySpMat`|10.1| | |`hipsparseDestroySpMat`|4.1.0| | |
|`cusparseDestroySpVec`|10.2| | |`hipsparseDestroySpVec`|4.1.0| | |
|`cusparseDnMatGet`|10.1| | |`hipsparseDnMatGet`|4.2.0| | |
|`cusparseDnMatGetStridedBatch`|10.1| | | | | | |
|`cusparseDnMatGetValues`|10.2| | |`hipsparseDnMatGetValues`|4.2.0| | |
|`cusparseDnMatSetStridedBatch`|10.1| | | | | | |
|`cusparseDnMatSetValues`|10.2| | |`hipsparseDnMatSetValues`|4.2.0| | |
|`cusparseDnVecGet`|10.2| | |`hipsparseDnVecGet`|4.1.0| | |
|`cusparseDnVecGetValues`|10.2| | |`hipsparseDnVecGetValues`|4.1.0| | |
|`cusparseDnVecSetValues`|10.2| | |`hipsparseDnVecSetValues`|4.1.0| | |
|`cusparseGather`|11.0| | |`hipsparseGather`|4.1.0| | |
|`cusparseRot`|11.0| | |`hipsparseRot`|4.1.0| | |
|`cusparseScatter`|11.0| | |`hipsparseScatter`|4.1.0| | |
|`cusparseSpGEMM_compute`|11.0| | |`hipsparseSpGEMM_compute`|4.1.0| | |
|`cusparseSpGEMM_copy`|11.0| | |`hipsparseSpGEMM_copy`|4.1.0| | |
|`cusparseSpGEMM_createDescr`|11.0| | |`hipsparseSpGEMM_createDescr`|4.1.0| | |
|`cusparseSpGEMM_destroyDescr`|11.0| | |`hipsparseSpGEMM_destroyDescr`|4.1.0| | |
|`cusparseSpGEMM_workEstimation`| | | |`hipsparseSpGEMM_workEstimation`|4.1.0| | |
|`cusparseSpGEMMreuse_compute`|11.3| | | | | | |
|`cusparseSpGEMMreuse_copy`|11.3| | | | | | |
|`cusparseSpGEMMreuse_nnz`|11.3| | | | | | |
|`cusparseSpGEMMreuse_workEstimation`|11.3| | | | | | |
|`cusparseSpMM`|10.1| | |`hipsparseSpMM`|4.2.0| | |
|`cusparseSpMM_bufferSize`|10.1| | |`hipsparseSpMM_bufferSize`|4.2.0| | |
|`cusparseSpMV`|10.2| | |`hipsparseSpMV`|4.1.0| | |
|`cusparseSpMV_bufferSize`|10.2| | |`hipsparseSpMV_bufferSize`|4.1.0| | |
|`cusparseSpMatGetAttribute`|11.3| | | | | | |
|`cusparseSpMatGetFormat`|10.1| | |`hipsparseSpMatGetFormat`|4.1.0| | |
|`cusparseSpMatGetIndexBase`|10.1| | |`hipsparseSpMatGetIndexBase`|4.1.0| | |
|`cusparseSpMatGetNumBatches`|10.1| |10.2| | | | |
|`cusparseSpMatGetSize`|11.0| | |`hipsparseSpMatGetSize`|4.1.0| | |
|`cusparseSpMatGetStridedBatch`|10.2| | | | | | |
|`cusparseSpMatGetValues`|10.2| | |`hipsparseSpMatGetValues`|4.1.0| | |
|`cusparseSpMatSetAttribute`|11.3| | | | | | |
|`cusparseSpMatSetNumBatches`|10.1| |10.2| | | | |
|`cusparseSpMatSetStridedBatch`|10.2| | | | | | |
|`cusparseSpMatSetValues`|10.2| | |`hipsparseSpMatSetValues`|4.1.0| | |
|`cusparseSpSM_analysis`|11.3| | | | | | |
|`cusparseSpSM_bufferSize`|11.3| | | | | | |
|`cusparseSpSM_createDescr`|11.3| | | | | | |
|`cusparseSpSM_destroyDescr`|11.3| | | | | | |
|`cusparseSpSM_solve`|11.3| | | | | | |
|`cusparseSpSV_analysis`|11.3| | | | | | |
|`cusparseSpSV_bufferSize`|11.3| | | | | | |
|`cusparseSpSV_createDescr`|11.3| | | | | | |
|`cusparseSpSV_destroyDescr`|11.3| | | | | | |
|`cusparseSpSV_solve`|11.3| | | | | | |
|`cusparseSpVV`|10.2| | |`hipsparseSpVV`|4.1.0| | |
|`cusparseSpVV_bufferSize`|10.2| | |`hipsparseSpVV_bufferSize`|4.1.0| | |
|`cusparseSpVecGet`|10.2| | |`hipsparseSpVecGet`|4.1.0| | |
|`cusparseSpVecGetIndexBase`|10.2| | |`hipsparseSpVecGetIndexBase`|4.1.0| | |
|`cusparseSpVecGetValues`|10.2| | |`hipsparseSpVecGetValues`|4.1.0| | |
|`cusparseSpVecSetValues`|10.2| | |`hipsparseSpVecSetValues`|4.1.0| | |
|`cusparseSparseToDense`|11.1| | |`hipsparseSparseToDense`|4.2.0| | |
|`cusparseSparseToDense_bufferSize`|11.1| | |`hipsparseSparseToDense_bufferSize`|4.2.0| | |


\*A - Added; D - Deprecated; R - Removed