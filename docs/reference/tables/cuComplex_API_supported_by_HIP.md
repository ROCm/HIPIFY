<head>
    <meta charset="UTF-8">
    <meta name="description" content="CUDA APIs supported by HIPIFY">
    <meta name="keywords" content="HIPIFY, HIP, ROCm, CUDA, CUDA2HIP, hipification, hipify-clang, hipify-perl, Runtime API, Complex">
</head>

# CUCOMPLEX API supported by HIP


**Note\:** In the tables that follow the columns marked `A`, `D`, `C`, `R`, `U`, and `E` mean the following:
**A** - Added; **D** - Deprecated; **C** - Changed; **R** - Removed; **U** - Unsupported for CUDA version(s); **E** - Experimental

## **1. cuComplex Data types**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**U**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|:-:|
|`cuComplex`| | | | |`hipComplex`|1.6.0| | | | | |
|`cuDoubleComplex`| | | | |`hipDoubleComplex`|1.6.0| | | | | |
|`cuFloatComplex`| | | | |`hipFloatComplex`|1.6.0| | | | | |

## **2. cuComplex API functions**

|**CUDA**|**A**|**D**|**C**|**R**|**HIP**|**A**|**D**|**C**|**R**|**U**|**E**|
|:--|:-:|:-:|:-:|:-:|:--|:-:|:-:|:-:|:-:|:-:|:-:|
|`cuCabs`| | | | |`hipCabs`|1.6.0| | | | | |
|`cuCabsf`| | | | |`hipCabsf`|1.6.0| | | | | |
|`cuCadd`| | | | |`hipCadd`|1.6.0| | | | | |
|`cuCaddf`| | | | |`hipCaddf`|1.6.0| | | | | |
|`cuCdiv`| | | | |`hipCdiv`|1.6.0| | | | | |
|`cuCdivf`| | | | |`hipCdivf`|1.6.0| | | | | |
|`cuCfma`| | | | |`hipCfma`|1.6.0| | | | | |
|`cuCfmaf`| | | | |`hipCfmaf`|1.6.0| | | | | |
|`cuCimag`| | | | |`hipCimag`|1.6.0| | | | | |
|`cuCimagf`| | | | |`hipCimagf`|1.6.0| | | | | |
|`cuCmul`| | | | |`hipCmul`|1.6.0| | | | | |
|`cuCmulf`| | | | |`hipCmulf`|1.6.0| | | | | |
|`cuComplexDoubleToFloat`| | | | |`hipComplexDoubleToFloat`|1.6.0| | | | | |
|`cuComplexFloatToDouble`| | | | |`hipComplexFloatToDouble`|1.6.0| | | | | |
|`cuConj`| | | | |`hipConj`|1.6.0| | | | | |
|`cuConjf`| | | | |`hipConjf`|1.6.0| | | | | |
|`cuCreal`| | | | |`hipCreal`|1.6.0| | | | | |
|`cuCrealf`| | | | |`hipCrealf`|1.6.0| | | | | |
|`cuCsub`| | | | |`hipCsub`|1.6.0| | | | | |
|`cuCsubf`| | | | |`hipCsubf`|1.6.0| | | | | |
|`make_cuComplex`| | | | |`make_hipComplex`|1.6.0| | | | | |
|`make_cuDoubleComplex`| | | | |`make_hipDoubleComplex`|1.6.0| | | | | |
|`make_cuFloatComplex`| | | | |`make_hipFloatComplex`|1.6.0| | | | | |

