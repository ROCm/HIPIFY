// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --amap --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args -D__CUDA_API_VERSION_INTERNAL

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime_api.h>
#include <stdio.h>
#if defined(_WIN32)
#include "windows.h"
#endif
#include "cuda_gl_interop.h"

int main() {
  printf("06. CUDA Runtime API Enums synthetic test\n");

  // CHECK: hipChannelFormatKind ChannelFormatKind;
  // CHECK-NEXT: hipChannelFormatKind ChannelFormatKindSigned = hipChannelFormatKindSigned;
  // CHECK-NEXT: hipChannelFormatKind ChannelFormatKindUnsigned = hipChannelFormatKindUnsigned;
  // CHECK-NEXT: hipChannelFormatKind ChannelFormatKindFloat = hipChannelFormatKindFloat;
  // CHECK-NEXT: hipChannelFormatKind ChannelFormatKindNone = hipChannelFormatKindNone;
  cudaChannelFormatKind ChannelFormatKind;
  cudaChannelFormatKind ChannelFormatKindSigned = cudaChannelFormatKindSigned;
  cudaChannelFormatKind ChannelFormatKindUnsigned = cudaChannelFormatKindUnsigned;
  cudaChannelFormatKind ChannelFormatKindFloat = cudaChannelFormatKindFloat;
  cudaChannelFormatKind ChannelFormatKindNone = cudaChannelFormatKindNone;

  // CHECK: hipComputeMode ComputeMode;
  // CHECK-NEXT: hipComputeMode ComputeModeDefault = hipComputeModeDefault;
  // CHECK-NEXT: hipComputeMode ComputeModeExclusive = hipComputeModeExclusive;
  // CHECK-NEXT: hipComputeMode ComputeModeProhibited = hipComputeModeProhibited;
  // CHECK-NEXT: hipComputeMode ComputeModeExclusiveProcess = hipComputeModeExclusiveProcess;
  cudaComputeMode ComputeMode;
  cudaComputeMode ComputeModeDefault = cudaComputeModeDefault;
  cudaComputeMode ComputeModeExclusive = cudaComputeModeExclusive;
  cudaComputeMode ComputeModeProhibited = cudaComputeModeProhibited;
  cudaComputeMode ComputeModeExclusiveProcess = cudaComputeModeExclusiveProcess;

  // CHECK: hipDeviceAttribute_t DeviceAttr;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxThreadsPerBlock = hipDeviceAttributeMaxThreadsPerBlock;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxBlockDimX = hipDeviceAttributeMaxBlockDimX;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxBlockDimY = hipDeviceAttributeMaxBlockDimY;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxBlockDimZ = hipDeviceAttributeMaxBlockDimZ;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxGridDimX = hipDeviceAttributeMaxGridDimX;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxGridDimY = hipDeviceAttributeMaxGridDimY;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxGridDimZ = hipDeviceAttributeMaxGridDimZ;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxSharedMemoryPerBlock = hipDeviceAttributeMaxSharedMemoryPerBlock;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrTotalConstantMemory = hipDeviceAttributeTotalConstantMemory;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrWarpSize = hipDeviceAttributeWarpSize;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxPitch = hipDeviceAttributeMaxPitch;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxRegistersPerBlock = hipDeviceAttributeMaxRegistersPerBlock;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrClockRate = hipDeviceAttributeClockRate;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrTextureAlignment = hipDeviceAttributeTextureAlignment;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrGpuOverlap = hipDeviceAttributeAsyncEngineCount;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMultiProcessorCount = hipDeviceAttributeMultiprocessorCount;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrKernelExecTimeout = hipDeviceAttributeKernelExecTimeout;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrIntegrated = hipDeviceAttributeIntegrated;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrCanMapHostMemory = hipDeviceAttributeCanMapHostMemory;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrComputeMode = hipDeviceAttributeComputeMode;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxTexture1DWidth = hipDeviceAttributeMaxTexture1DWidth;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxTexture2DWidth = hipDeviceAttributeMaxTexture2DWidth;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxTexture2DHeight = hipDeviceAttributeMaxTexture2DHeight;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxTexture3DWidth = hipDeviceAttributeMaxTexture3DWidth;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxTexture3DHeight = hipDeviceAttributeMaxTexture3DHeight;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxTexture3DDepth = hipDeviceAttributeMaxTexture3DDepth;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxTexture2DLayeredWidth = hipDeviceAttributeMaxTexture2DLayered;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxTexture2DLayeredHeight = hipDeviceAttributeMaxTexture2DLayered;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrSurfaceAlignment = hipDeviceAttributeSurfaceAlignment;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrConcurrentKernels = hipDeviceAttributeConcurrentKernels;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrEccEnabled = hipDeviceAttributeEccEnabled;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrPciBusId = hipDeviceAttributePciBusId;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrPciDeviceId = hipDeviceAttributePciDeviceId;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrTccDriver = hipDeviceAttributeTccDriver;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMemoryClockRate = hipDeviceAttributeMemoryClockRate;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrGlobalMemoryBusWidth = hipDeviceAttributeMemoryBusWidth;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrL2CacheSize = hipDeviceAttributeL2CacheSize;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxThreadsPerMultiProcessor = hipDeviceAttributeMaxThreadsPerMultiProcessor;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrAsyncEngineCount = hipDeviceAttributeAsyncEngineCount;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrUnifiedAddressing = hipDeviceAttributeUnifiedAddressing;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxTexture1DLayeredWidth = hipDeviceAttributeMaxTexture1DLayered;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxTexture2DGatherWidth = hipDeviceAttributeMaxTexture2DGather;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxTexture2DGatherHeight = hipDeviceAttributeMaxTexture2DGather;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxTexture3DWidthAlt = hipDeviceAttributeMaxTexture3DAlt;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxTexture3DHeightAlt = hipDeviceAttributeMaxTexture3DAlt;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxTexture3DDepthAlt = hipDeviceAttributeMaxTexture3DAlt;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrPciDomainId = hipDeviceAttributePciDomainID;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrTexturePitchAlignment = hipDeviceAttributeTexturePitchAlignment;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxTextureCubemapWidth = hipDeviceAttributeMaxTextureCubemap;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxTextureCubemapLayeredWidth = hipDeviceAttributeMaxTextureCubemapLayered;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxSurface1DWidth = hipDeviceAttributeMaxSurface1D;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxSurface2DWidth = hipDeviceAttributeMaxSurface2D;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxSurface2DHeight = hipDeviceAttributeMaxSurface2D;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxSurface3DWidth = hipDeviceAttributeMaxSurface3D;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxSurface3DHeight = hipDeviceAttributeMaxSurface3D;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxSurface3DDepth = hipDeviceAttributeMaxSurface3D;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxSurface1DLayeredWidth = hipDeviceAttributeMaxSurface1DLayered;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxSurface2DLayeredWidth = hipDeviceAttributeMaxSurface2DLayered;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxSurface2DLayeredHeight = hipDeviceAttributeMaxSurface2DLayered;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxSurfaceCubemapWidth = hipDeviceAttributeMaxSurfaceCubemap;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxSurfaceCubemapLayeredWidth = hipDeviceAttributeMaxSurfaceCubemapLayered;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxTexture1DLinearWidth = hipDeviceAttributeMaxTexture1DLinear;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxTexture2DLinearWidth = hipDeviceAttributeMaxTexture2DLinear;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxTexture2DLinearHeight = hipDeviceAttributeMaxTexture2DLinear;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxTexture2DLinearPitch = hipDeviceAttributeMaxTexture2DLinear;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxTexture2DMipmappedWidth = hipDeviceAttributeMaxTexture2DMipmap;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxTexture2DMipmappedHeight = hipDeviceAttributeMaxTexture2DMipmap;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrComputeCapabilityMajor = hipDeviceAttributeComputeCapabilityMajor;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrComputeCapabilityMinor = hipDeviceAttributeComputeCapabilityMinor;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxTexture1DMipmappedWidth = hipDeviceAttributeMaxTexture1DMipmap;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrStreamPrioritiesSupported = hipDeviceAttributeStreamPrioritiesSupported;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrGlobalL1CacheSupported = hipDeviceAttributeGlobalL1CacheSupported;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrLocalL1CacheSupported = hipDeviceAttributeLocalL1CacheSupported;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxSharedMemoryPerMultiprocessor = hipDeviceAttributeMaxSharedMemoryPerMultiprocessor;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxRegistersPerMultiprocessor = hipDeviceAttributeMaxRegistersPerMultiprocessor;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrManagedMemory = hipDeviceAttributeManagedMemory;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrIsMultiGpuBoard = hipDeviceAttributeIsMultiGpuBoard;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMultiGpuBoardGroupID = hipDeviceAttributeMultiGpuBoardGroupID;
  cudaDeviceAttr DeviceAttr;
  cudaDeviceAttr DevAttrMaxThreadsPerBlock = cudaDevAttrMaxThreadsPerBlock;
  cudaDeviceAttr DevAttrMaxBlockDimX = cudaDevAttrMaxBlockDimX;
  cudaDeviceAttr DevAttrMaxBlockDimY = cudaDevAttrMaxBlockDimY;
  cudaDeviceAttr DevAttrMaxBlockDimZ = cudaDevAttrMaxBlockDimZ;
  cudaDeviceAttr DevAttrMaxGridDimX = cudaDevAttrMaxGridDimX;
  cudaDeviceAttr DevAttrMaxGridDimY = cudaDevAttrMaxGridDimY;
  cudaDeviceAttr DevAttrMaxGridDimZ = cudaDevAttrMaxGridDimZ;
  cudaDeviceAttr DevAttrMaxSharedMemoryPerBlock = cudaDevAttrMaxSharedMemoryPerBlock;
  cudaDeviceAttr DevAttrTotalConstantMemory = cudaDevAttrTotalConstantMemory;
  cudaDeviceAttr DevAttrWarpSize = cudaDevAttrWarpSize;
  cudaDeviceAttr DevAttrMaxPitch = cudaDevAttrMaxPitch;
  cudaDeviceAttr DevAttrMaxRegistersPerBlock = cudaDevAttrMaxRegistersPerBlock;
  cudaDeviceAttr DevAttrClockRate = cudaDevAttrClockRate;
  cudaDeviceAttr DevAttrTextureAlignment = cudaDevAttrTextureAlignment;
  cudaDeviceAttr DevAttrGpuOverlap = cudaDevAttrGpuOverlap;
  cudaDeviceAttr DevAttrMultiProcessorCount = cudaDevAttrMultiProcessorCount;
  cudaDeviceAttr DevAttrKernelExecTimeout = cudaDevAttrKernelExecTimeout;
  cudaDeviceAttr DevAttrIntegrated = cudaDevAttrIntegrated;
  cudaDeviceAttr DevAttrCanMapHostMemory = cudaDevAttrCanMapHostMemory;
  cudaDeviceAttr DevAttrComputeMode = cudaDevAttrComputeMode;
  cudaDeviceAttr DevAttrMaxTexture1DWidth = cudaDevAttrMaxTexture1DWidth;
  cudaDeviceAttr DevAttrMaxTexture2DWidth = cudaDevAttrMaxTexture2DWidth;
  cudaDeviceAttr DevAttrMaxTexture2DHeight = cudaDevAttrMaxTexture2DHeight;
  cudaDeviceAttr DevAttrMaxTexture3DWidth = cudaDevAttrMaxTexture3DWidth;
  cudaDeviceAttr DevAttrMaxTexture3DHeight = cudaDevAttrMaxTexture3DHeight;
  cudaDeviceAttr DevAttrMaxTexture3DDepth = cudaDevAttrMaxTexture3DDepth;
  cudaDeviceAttr DevAttrMaxTexture2DLayeredWidth = cudaDevAttrMaxTexture2DLayeredWidth;
  cudaDeviceAttr DevAttrMaxTexture2DLayeredHeight = cudaDevAttrMaxTexture2DLayeredHeight;
  cudaDeviceAttr DevAttrSurfaceAlignment = cudaDevAttrSurfaceAlignment;
  cudaDeviceAttr DevAttrConcurrentKernels = cudaDevAttrConcurrentKernels;
  cudaDeviceAttr DevAttrEccEnabled = cudaDevAttrEccEnabled;
  cudaDeviceAttr DevAttrPciBusId = cudaDevAttrPciBusId;
  cudaDeviceAttr DevAttrPciDeviceId = cudaDevAttrPciDeviceId;
  cudaDeviceAttr DevAttrTccDriver = cudaDevAttrTccDriver;
  cudaDeviceAttr DevAttrMemoryClockRate = cudaDevAttrMemoryClockRate;
  cudaDeviceAttr DevAttrGlobalMemoryBusWidth = cudaDevAttrGlobalMemoryBusWidth;
  cudaDeviceAttr DevAttrL2CacheSize = cudaDevAttrL2CacheSize;
  cudaDeviceAttr DevAttrMaxThreadsPerMultiProcessor = cudaDevAttrMaxThreadsPerMultiProcessor;
  cudaDeviceAttr DevAttrAsyncEngineCount = cudaDevAttrAsyncEngineCount;
  cudaDeviceAttr DevAttrUnifiedAddressing = cudaDevAttrUnifiedAddressing;
  cudaDeviceAttr DevAttrMaxTexture1DLayeredWidth = cudaDevAttrMaxTexture1DLayeredWidth;
  cudaDeviceAttr DevAttrMaxTexture2DGatherWidth = cudaDevAttrMaxTexture2DGatherWidth;
  cudaDeviceAttr DevAttrMaxTexture2DGatherHeight = cudaDevAttrMaxTexture2DGatherHeight;
  cudaDeviceAttr DevAttrMaxTexture3DWidthAlt = cudaDevAttrMaxTexture3DWidthAlt;
  cudaDeviceAttr DevAttrMaxTexture3DHeightAlt = cudaDevAttrMaxTexture3DHeightAlt;
  cudaDeviceAttr DevAttrMaxTexture3DDepthAlt = cudaDevAttrMaxTexture3DDepthAlt;
  cudaDeviceAttr DevAttrPciDomainId = cudaDevAttrPciDomainId;
  cudaDeviceAttr DevAttrTexturePitchAlignment = cudaDevAttrTexturePitchAlignment;
  cudaDeviceAttr DevAttrMaxTextureCubemapWidth = cudaDevAttrMaxTextureCubemapWidth;
  cudaDeviceAttr DevAttrMaxTextureCubemapLayeredWidth = cudaDevAttrMaxTextureCubemapLayeredWidth;
  cudaDeviceAttr DevAttrMaxSurface1DWidth = cudaDevAttrMaxSurface1DWidth;
  cudaDeviceAttr DevAttrMaxSurface2DWidth = cudaDevAttrMaxSurface2DWidth;
  cudaDeviceAttr DevAttrMaxSurface2DHeight = cudaDevAttrMaxSurface2DHeight;
  cudaDeviceAttr DevAttrMaxSurface3DWidth = cudaDevAttrMaxSurface3DWidth;
  cudaDeviceAttr DevAttrMaxSurface3DHeight = cudaDevAttrMaxSurface3DHeight;
  cudaDeviceAttr DevAttrMaxSurface3DDepth = cudaDevAttrMaxSurface3DDepth;
  cudaDeviceAttr DevAttrMaxSurface1DLayeredWidth = cudaDevAttrMaxSurface1DLayeredWidth;
  cudaDeviceAttr DevAttrMaxSurface2DLayeredWidth = cudaDevAttrMaxSurface2DLayeredWidth;
  cudaDeviceAttr DevAttrMaxSurface2DLayeredHeight = cudaDevAttrMaxSurface2DLayeredHeight;
  cudaDeviceAttr DevAttrMaxSurfaceCubemapWidth = cudaDevAttrMaxSurfaceCubemapWidth;
  cudaDeviceAttr DevAttrMaxSurfaceCubemapLayeredWidth = cudaDevAttrMaxSurfaceCubemapLayeredWidth;
  cudaDeviceAttr DevAttrMaxTexture1DLinearWidth = cudaDevAttrMaxTexture1DLinearWidth;
  cudaDeviceAttr DevAttrMaxTexture2DLinearWidth = cudaDevAttrMaxTexture2DLinearWidth;
  cudaDeviceAttr DevAttrMaxTexture2DLinearHeight = cudaDevAttrMaxTexture2DLinearHeight;
  cudaDeviceAttr DevAttrMaxTexture2DLinearPitch = cudaDevAttrMaxTexture2DLinearPitch;
  cudaDeviceAttr DevAttrMaxTexture2DMipmappedWidth = cudaDevAttrMaxTexture2DMipmappedWidth;
  cudaDeviceAttr DevAttrMaxTexture2DMipmappedHeight = cudaDevAttrMaxTexture2DMipmappedHeight;
  cudaDeviceAttr DevAttrComputeCapabilityMajor = cudaDevAttrComputeCapabilityMajor;
  cudaDeviceAttr DevAttrComputeCapabilityMinor = cudaDevAttrComputeCapabilityMinor;
  cudaDeviceAttr DevAttrMaxTexture1DMipmappedWidth = cudaDevAttrMaxTexture1DMipmappedWidth;
  cudaDeviceAttr DevAttrStreamPrioritiesSupported = cudaDevAttrStreamPrioritiesSupported;
  cudaDeviceAttr DevAttrGlobalL1CacheSupported = cudaDevAttrGlobalL1CacheSupported;
  cudaDeviceAttr DevAttrLocalL1CacheSupported = cudaDevAttrLocalL1CacheSupported;
  cudaDeviceAttr DevAttrMaxSharedMemoryPerMultiprocessor = cudaDevAttrMaxSharedMemoryPerMultiprocessor;
  cudaDeviceAttr DevAttrMaxRegistersPerMultiprocessor = cudaDevAttrMaxRegistersPerMultiprocessor;
  cudaDeviceAttr DevAttrManagedMemory = cudaDevAttrManagedMemory;
  cudaDeviceAttr DevAttrIsMultiGpuBoard = cudaDevAttrIsMultiGpuBoard;
  cudaDeviceAttr DevAttrMultiGpuBoardGroupID = cudaDevAttrMultiGpuBoardGroupID;

  // CHECK: hipError_t Error;
  // CHECK-NEXT: hipError_t Error_t;
  // CHECK-NEXT: hipError_t Success = hipSuccess;
  // CHECK-NEXT: hipError_t ErrorInvalidValue = hipErrorInvalidValue;
  // CHECK-NEXT: hipError_t ErrorMemoryAllocation = hipErrorOutOfMemory;
  // CHECK-NEXT: hipError_t ErrorInitializationError = hipErrorNotInitialized;
  // CHECK-NEXT: hipError_t ErrorCudartUnloading = hipErrorDeinitialized;
  // CHECK-NEXT: hipError_t ErrorProfilerDisabled = hipErrorProfilerDisabled;
  // CHECK-NEXT: hipError_t ErrorProfilerNotInitialized = hipErrorProfilerNotInitialized;
  // CHECK-NEXT: hipError_t ErrorProfilerAlreadyStarted = hipErrorProfilerAlreadyStarted;
  // CHECK-NEXT: hipError_t ErrorProfilerAlreadyStopped = hipErrorProfilerAlreadyStopped;
  // CHECK-NEXT: hipError_t ErrorInvalidConfiguration = hipErrorInvalidConfiguration;
  // CHECK-NEXT: hipError_t ErrorInvalidPitchValue = hipErrorInvalidPitchValue;
  // CHECK-NEXT: hipError_t ErrorInvalidSymbol = hipErrorInvalidSymbol;
  // CHECK-NEXT: hipError_t ErrorInvalidDevicePointer = hipErrorInvalidDevicePointer;
  // CHECK-NEXT: hipError_t ErrorInvalidMemcpyDirection = hipErrorInvalidMemcpyDirection;
  // CHECK-NEXT: hipError_t ErrorInsufficientDriver = hipErrorInsufficientDriver;
  // CHECK-NEXT: hipError_t ErrorMissingConfiguration = hipErrorMissingConfiguration;
  // CHECK-NEXT: hipError_t ErrorPriorLaunchFailure = hipErrorPriorLaunchFailure;
  // CHECK-NEXT: hipError_t ErrorInvalidDeviceFunction = hipErrorInvalidDeviceFunction;
  // CHECK-NEXT: hipError_t ErrorNoDevice = hipErrorNoDevice;
  // CHECK-NEXT: hipError_t ErrorInvalidDevice = hipErrorInvalidDevice;
  // CHECK-NEXT: hipError_t ErrorInvalidKernelImage = hipErrorInvalidImage;
  // CHECK-NEXT: hipError_t ErrorMapBufferObjectFailed = hipErrorMapFailed;
  // CHECK-NEXT: hipError_t ErrorUnmapBufferObjectFailed = hipErrorUnmapFailed;
  // CHECK-NEXT: hipError_t ErrorNoKernelImageForDevice = hipErrorNoBinaryForGpu;
  // CHECK-NEXT: hipError_t ErrorECCUncorrectable = hipErrorECCNotCorrectable;
  // CHECK-NEXT: hipError_t ErrorUnsupportedLimit = hipErrorUnsupportedLimit;
  // CHECK-NEXT: hipError_t ErrorDeviceAlreadyInUse = hipErrorContextAlreadyInUse;
  // CHECK-NEXT: hipError_t ErrorPeerAccessUnsupported = hipErrorPeerAccessUnsupported;
  // CHECK-NEXT: hipError_t ErrorInvalidPtx = hipErrorInvalidKernelFile;
  // CHECK-NEXT: hipError_t ErrorInvalidGraphicsContext = hipErrorInvalidGraphicsContext;
  // CHECK-NEXT: hipError_t ErrorSharedObjectSymbolNotFound = hipErrorSharedObjectSymbolNotFound;
  // CHECK-NEXT: hipError_t ErrorSharedObjectInitFailed = hipErrorSharedObjectInitFailed;
  // CHECK-NEXT: hipError_t ErrorOperatingSystem = hipErrorOperatingSystem;
  // CHECK-NEXT: hipError_t ErrorInvalidResourceHandle = hipErrorInvalidHandle;
  // CHECK-NEXT: hipError_t ErrorNotReady = hipErrorNotReady;
  // CHECK-NEXT: hipError_t ErrorIllegalAddress = hipErrorIllegalAddress;
  // CHECK-NEXT: hipError_t ErrorLaunchOutOfResources = hipErrorLaunchOutOfResources;
  // CHECK-NEXT: hipError_t ErrorLaunchTimeout = hipErrorLaunchTimeOut;
  // CHECK-NEXT: hipError_t ErrorPeerAccessAlreadyEnabled = hipErrorPeerAccessAlreadyEnabled;
  // CHECK-NEXT: hipError_t ErrorPeerAccessNotEnabled = hipErrorPeerAccessNotEnabled;
  // CHECK-NEXT: hipError_t ErrorSetOnActiveProcess = hipErrorSetOnActiveProcess;
  // CHECK-NEXT: hipError_t ErrorAssert = hipErrorAssert;
  // CHECK-NEXT: hipError_t ErrorHostMemoryAlreadyRegistered = hipErrorHostMemoryAlreadyRegistered;
  // CHECK-NEXT: hipError_t ErrorHostMemoryNotRegistered = hipErrorHostMemoryNotRegistered;
  // CHECK-NEXT: hipError_t ErrorLaunchFailure = hipErrorLaunchFailure;
  // CHECK-NEXT: hipError_t ErrorNotSupported = hipErrorNotSupported;
  cudaError Error;
  cudaError_t Error_t;
  cudaError_t Success = cudaSuccess;
  cudaError_t ErrorInvalidValue = cudaErrorInvalidValue;
  cudaError_t ErrorMemoryAllocation = cudaErrorMemoryAllocation;
  cudaError_t ErrorInitializationError = cudaErrorInitializationError;
  cudaError_t ErrorCudartUnloading = cudaErrorCudartUnloading;
  cudaError_t ErrorProfilerDisabled = cudaErrorProfilerDisabled;
  cudaError_t ErrorProfilerNotInitialized = cudaErrorProfilerNotInitialized;
  cudaError_t ErrorProfilerAlreadyStarted = cudaErrorProfilerAlreadyStarted;
  cudaError_t ErrorProfilerAlreadyStopped = cudaErrorProfilerAlreadyStopped;
  cudaError_t ErrorInvalidConfiguration = cudaErrorInvalidConfiguration;
  cudaError_t ErrorInvalidPitchValue = cudaErrorInvalidPitchValue;
  cudaError_t ErrorInvalidSymbol = cudaErrorInvalidSymbol;
  cudaError_t ErrorInvalidDevicePointer = cudaErrorInvalidDevicePointer;
  cudaError_t ErrorInvalidMemcpyDirection = cudaErrorInvalidMemcpyDirection;
  cudaError_t ErrorInsufficientDriver = cudaErrorInsufficientDriver;
  cudaError_t ErrorMissingConfiguration = cudaErrorMissingConfiguration;
  cudaError_t ErrorPriorLaunchFailure = cudaErrorPriorLaunchFailure;
  cudaError_t ErrorInvalidDeviceFunction = cudaErrorInvalidDeviceFunction;
  cudaError_t ErrorNoDevice = cudaErrorNoDevice;
  cudaError_t ErrorInvalidDevice = cudaErrorInvalidDevice;
  cudaError_t ErrorInvalidKernelImage = cudaErrorInvalidKernelImage;
  cudaError_t ErrorMapBufferObjectFailed = cudaErrorMapBufferObjectFailed;
  cudaError_t ErrorUnmapBufferObjectFailed = cudaErrorUnmapBufferObjectFailed;
  cudaError_t ErrorNoKernelImageForDevice = cudaErrorNoKernelImageForDevice;
  cudaError_t ErrorECCUncorrectable = cudaErrorECCUncorrectable;
  cudaError_t ErrorUnsupportedLimit = cudaErrorUnsupportedLimit;
  cudaError_t ErrorDeviceAlreadyInUse = cudaErrorDeviceAlreadyInUse;
  cudaError_t ErrorPeerAccessUnsupported = cudaErrorPeerAccessUnsupported;
  cudaError_t ErrorInvalidPtx = cudaErrorInvalidPtx;
  cudaError_t ErrorInvalidGraphicsContext = cudaErrorInvalidGraphicsContext;
  cudaError_t ErrorSharedObjectSymbolNotFound = cudaErrorSharedObjectSymbolNotFound;
  cudaError_t ErrorSharedObjectInitFailed = cudaErrorSharedObjectInitFailed;
  cudaError_t ErrorOperatingSystem = cudaErrorOperatingSystem;
  cudaError_t ErrorInvalidResourceHandle = cudaErrorInvalidResourceHandle;
  cudaError_t ErrorNotReady = cudaErrorNotReady;
  cudaError_t ErrorIllegalAddress = cudaErrorIllegalAddress;
  cudaError_t ErrorLaunchOutOfResources = cudaErrorLaunchOutOfResources;
  cudaError_t ErrorLaunchTimeout = cudaErrorLaunchTimeout;
  cudaError_t ErrorPeerAccessAlreadyEnabled = cudaErrorPeerAccessAlreadyEnabled;
  cudaError_t ErrorPeerAccessNotEnabled = cudaErrorPeerAccessNotEnabled;
  cudaError_t ErrorSetOnActiveProcess = cudaErrorSetOnActiveProcess;
  cudaError_t ErrorAssert = cudaErrorAssert;
  cudaError_t ErrorHostMemoryAlreadyRegistered = cudaErrorHostMemoryAlreadyRegistered;
  cudaError_t ErrorHostMemoryNotRegistered = cudaErrorHostMemoryNotRegistered;
  cudaError_t ErrorLaunchFailure = cudaErrorLaunchFailure;
  cudaError_t ErrorNotSupported = cudaErrorNotSupported;

  // CHECK: hipError_t ErrorUnknown = hipErrorUnknown;
  cudaError_t ErrorUnknown = cudaErrorUnknown;

  // CHECK: hipFuncCache_t FuncCache;
  // CHECK-NEXT: hipFuncCache_t FuncCachePreferNone = hipFuncCachePreferNone;
  // CHECK-NEXT: hipFuncCache_t FuncCachePreferShared = hipFuncCachePreferShared;
  // CHECK-NEXT: hipFuncCache_t FuncCachePreferL1 = hipFuncCachePreferL1;
  // CHECK-NEXT: hipFuncCache_t FuncCachePreferEqual = hipFuncCachePreferEqual;
  cudaFuncCache FuncCache;
  cudaFuncCache FuncCachePreferNone = cudaFuncCachePreferNone;
  cudaFuncCache FuncCachePreferShared = cudaFuncCachePreferShared;
  cudaFuncCache FuncCachePreferL1 = cudaFuncCachePreferL1;
  cudaFuncCache FuncCachePreferEqual = cudaFuncCachePreferEqual;

  // CHECK: hipGraphicsRegisterFlags GraphicsRegisterFlags;
  // CHECK-NEXT: hipGraphicsRegisterFlags GraphicsRegisterFlagsNone = hipGraphicsRegisterFlagsNone;
  // CHECK-NEXT: hipGraphicsRegisterFlags GraphicsRegisterFlagsReadOnly = hipGraphicsRegisterFlagsReadOnly;
  // CHECK-NEXT: hipGraphicsRegisterFlags GraphicsRegisterFlagsWriteDiscard = hipGraphicsRegisterFlagsWriteDiscard;
  // CHECK-NEXT: hipGraphicsRegisterFlags GraphicsRegisterFlagsSurfaceLoadStore = hipGraphicsRegisterFlagsSurfaceLoadStore;
  // CHECK-NEXT: hipGraphicsRegisterFlags GraphicsRegisterFlagsTextureGather = hipGraphicsRegisterFlagsTextureGather;
  cudaGraphicsRegisterFlags GraphicsRegisterFlags;
  cudaGraphicsRegisterFlags GraphicsRegisterFlagsNone = cudaGraphicsRegisterFlagsNone;
  cudaGraphicsRegisterFlags GraphicsRegisterFlagsReadOnly = cudaGraphicsRegisterFlagsReadOnly;
  cudaGraphicsRegisterFlags GraphicsRegisterFlagsWriteDiscard = cudaGraphicsRegisterFlagsWriteDiscard;
  cudaGraphicsRegisterFlags GraphicsRegisterFlagsSurfaceLoadStore = cudaGraphicsRegisterFlagsSurfaceLoadStore;
  cudaGraphicsRegisterFlags GraphicsRegisterFlagsTextureGather = cudaGraphicsRegisterFlagsTextureGather;

  // CHECK: hipLimit_t Limit;
  // CHECK-NEXT: hipLimit_t LimitStackSize = hipLimitStackSize;
  // CHECK-NEXT: hipLimit_t LimitPrintfFifoSize = hipLimitPrintfFifoSize;
  // CHECK-NEXT: hipLimit_t LimitMallocHeapSize = hipLimitMallocHeapSize;
  cudaLimit Limit;
  cudaLimit LimitStackSize = cudaLimitStackSize;
  cudaLimit LimitPrintfFifoSize = cudaLimitPrintfFifoSize;
  cudaLimit LimitMallocHeapSize = cudaLimitMallocHeapSize;

  // CHECK: hipMemcpyKind MemcpyKind;
  // CHECK-NEXT: hipMemcpyKind MemcpyHostToHost = hipMemcpyHostToHost;
  // CHECK-NEXT: hipMemcpyKind MemcpyHostToDevice = hipMemcpyHostToDevice;
  // CHECK-NEXT: hipMemcpyKind MemcpyDeviceToHost = hipMemcpyDeviceToHost;
  // CHECK-NEXT: hipMemcpyKind MemcpyDeviceToDevice = hipMemcpyDeviceToDevice;
  // CHECK-NEXT: hipMemcpyKind MemcpyDefault = hipMemcpyDefault;
  cudaMemcpyKind MemcpyKind;
  cudaMemcpyKind MemcpyHostToHost = cudaMemcpyHostToHost;
  cudaMemcpyKind MemcpyHostToDevice = cudaMemcpyHostToDevice;
  cudaMemcpyKind MemcpyDeviceToHost = cudaMemcpyDeviceToHost;
  cudaMemcpyKind MemcpyDeviceToDevice = cudaMemcpyDeviceToDevice;
  cudaMemcpyKind MemcpyDefault = cudaMemcpyDefault;

  // CHECK: hipMemoryType MemoryType;
  // CHECK-NEXT: hipMemoryType MemoryTypeHost = hipMemoryTypeHost;
  // CHECK-NEXT: hipMemoryType MemoryTypeDevice = hipMemoryTypeDevice;
  cudaMemoryType MemoryType;
  cudaMemoryType MemoryTypeHost = cudaMemoryTypeHost;
  cudaMemoryType MemoryTypeDevice = cudaMemoryTypeDevice;

  // CHECK: hipResourceType ResourceType;
  // CHECK-NEXT: hipResourceType ResourceTypeArray = hipResourceTypeArray;
  // CHECK-NEXT: hipResourceType ResourceTypeMipmappedArray = hipResourceTypeMipmappedArray;
  // CHECK-NEXT: hipResourceType ResourceTypeLinear = hipResourceTypeLinear;
  // CHECK-NEXT: hipResourceType ResourceTypePitch2D = hipResourceTypePitch2D;
  cudaResourceType ResourceType;
  cudaResourceType ResourceTypeArray = cudaResourceTypeArray;
  cudaResourceType ResourceTypeMipmappedArray = cudaResourceTypeMipmappedArray;
  cudaResourceType ResourceTypeLinear = cudaResourceTypeLinear;
  cudaResourceType ResourceTypePitch2D = cudaResourceTypePitch2D;

  // CHECK: hipResourceViewFormat ResourceViewFormat;
  // CHECK-NEXT: hipResourceViewFormat ResViewFormatNone = hipResViewFormatNone;
  // CHECK-NEXT: hipResourceViewFormat ResViewFormatUnsignedChar1 = hipResViewFormatUnsignedChar1;
  // CHECK-NEXT: hipResourceViewFormat ResViewFormatUnsignedChar2 = hipResViewFormatUnsignedChar2;
  // CHECK-NEXT: hipResourceViewFormat ResViewFormatUnsignedChar4 = hipResViewFormatUnsignedChar4;
  // CHECK-NEXT: hipResourceViewFormat ResViewFormatSignedChar1 = hipResViewFormatSignedChar1;
  // CHECK-NEXT: hipResourceViewFormat ResViewFormatSignedChar2 = hipResViewFormatSignedChar2;
  // CHECK-NEXT: hipResourceViewFormat ResViewFormatSignedChar4 = hipResViewFormatSignedChar4;
  // CHECK-NEXT: hipResourceViewFormat ResViewFormatUnsignedShort1 = hipResViewFormatUnsignedShort1;
  // CHECK-NEXT: hipResourceViewFormat ResViewFormatUnsignedShort2 = hipResViewFormatUnsignedShort2;
  // CHECK-NEXT: hipResourceViewFormat ResViewFormatUnsignedShort4 = hipResViewFormatUnsignedShort4;
  // CHECK-NEXT: hipResourceViewFormat ResViewFormatSignedShort1 = hipResViewFormatSignedShort1;
  // CHECK-NEXT: hipResourceViewFormat ResViewFormatSignedShort2 = hipResViewFormatSignedShort2;
  // CHECK-NEXT: hipResourceViewFormat ResViewFormatSignedShort4 = hipResViewFormatSignedShort4;
  // CHECK-NEXT: hipResourceViewFormat ResViewFormatUnsignedInt1 = hipResViewFormatUnsignedInt1;
  // CHECK-NEXT: hipResourceViewFormat ResViewFormatUnsignedInt2 = hipResViewFormatUnsignedInt2;
  // CHECK-NEXT: hipResourceViewFormat ResViewFormatUnsignedInt4 = hipResViewFormatUnsignedInt4;
  // CHECK-NEXT: hipResourceViewFormat ResViewFormatSignedInt1 = hipResViewFormatSignedInt1;
  // CHECK-NEXT: hipResourceViewFormat ResViewFormatSignedInt2 = hipResViewFormatSignedInt2;
  // CHECK-NEXT: hipResourceViewFormat ResViewFormatSignedInt4 = hipResViewFormatSignedInt4;
  // CHECK-NEXT: hipResourceViewFormat ResViewFormatHalf1 = hipResViewFormatHalf1;
  // CHECK-NEXT: hipResourceViewFormat ResViewFormatHalf2 = hipResViewFormatHalf2;
  // CHECK-NEXT: hipResourceViewFormat ResViewFormatHalf4 = hipResViewFormatHalf4;
  // CHECK-NEXT: hipResourceViewFormat ResViewFormatFloat1 = hipResViewFormatFloat1;
  // CHECK-NEXT: hipResourceViewFormat ResViewFormatFloat2 = hipResViewFormatFloat2;
  // CHECK-NEXT: hipResourceViewFormat ResViewFormatFloat4 = hipResViewFormatFloat4;
  // CHECK-NEXT: hipResourceViewFormat ResViewFormatUnsignedBlockCompressed1 = hipResViewFormatUnsignedBlockCompressed1;
  // CHECK-NEXT: hipResourceViewFormat ResViewFormatUnsignedBlockCompressed2 = hipResViewFormatUnsignedBlockCompressed2;
  // CHECK-NEXT: hipResourceViewFormat ResViewFormatUnsignedBlockCompressed3 = hipResViewFormatUnsignedBlockCompressed3;
  // CHECK-NEXT: hipResourceViewFormat ResViewFormatUnsignedBlockCompressed4 = hipResViewFormatUnsignedBlockCompressed4;
  // CHECK-NEXT: hipResourceViewFormat ResViewFormatSignedBlockCompressed4 = hipResViewFormatSignedBlockCompressed4;
  // CHECK-NEXT: hipResourceViewFormat ResViewFormatUnsignedBlockCompressed5 = hipResViewFormatUnsignedBlockCompressed5;
  // CHECK-NEXT: hipResourceViewFormat ResViewFormatSignedBlockCompressed5 = hipResViewFormatSignedBlockCompressed5;
  // CHECK-NEXT: hipResourceViewFormat ResViewFormatUnsignedBlockCompressed6H = hipResViewFormatUnsignedBlockCompressed6H;
  // CHECK-NEXT: hipResourceViewFormat ResViewFormatSignedBlockCompressed6H = hipResViewFormatSignedBlockCompressed6H;
  // CHECK-NEXT: hipResourceViewFormat ResViewFormatUnsignedBlockCompressed7 = hipResViewFormatUnsignedBlockCompressed7;
  cudaResourceViewFormat ResourceViewFormat;
  cudaResourceViewFormat ResViewFormatNone = cudaResViewFormatNone;
  cudaResourceViewFormat ResViewFormatUnsignedChar1 = cudaResViewFormatUnsignedChar1;
  cudaResourceViewFormat ResViewFormatUnsignedChar2 = cudaResViewFormatUnsignedChar2;
  cudaResourceViewFormat ResViewFormatUnsignedChar4 = cudaResViewFormatUnsignedChar4;
  cudaResourceViewFormat ResViewFormatSignedChar1 = cudaResViewFormatSignedChar1;
  cudaResourceViewFormat ResViewFormatSignedChar2 = cudaResViewFormatSignedChar2;
  cudaResourceViewFormat ResViewFormatSignedChar4 = cudaResViewFormatSignedChar4;
  cudaResourceViewFormat ResViewFormatUnsignedShort1 = cudaResViewFormatUnsignedShort1;
  cudaResourceViewFormat ResViewFormatUnsignedShort2 = cudaResViewFormatUnsignedShort2;
  cudaResourceViewFormat ResViewFormatUnsignedShort4 = cudaResViewFormatUnsignedShort4;
  cudaResourceViewFormat ResViewFormatSignedShort1 = cudaResViewFormatSignedShort1;
  cudaResourceViewFormat ResViewFormatSignedShort2 = cudaResViewFormatSignedShort2;
  cudaResourceViewFormat ResViewFormatSignedShort4 = cudaResViewFormatSignedShort4;
  cudaResourceViewFormat ResViewFormatUnsignedInt1 = cudaResViewFormatUnsignedInt1;
  cudaResourceViewFormat ResViewFormatUnsignedInt2 = cudaResViewFormatUnsignedInt2;
  cudaResourceViewFormat ResViewFormatUnsignedInt4 = cudaResViewFormatUnsignedInt4;
  cudaResourceViewFormat ResViewFormatSignedInt1 = cudaResViewFormatSignedInt1;
  cudaResourceViewFormat ResViewFormatSignedInt2 = cudaResViewFormatSignedInt2;
  cudaResourceViewFormat ResViewFormatSignedInt4 = cudaResViewFormatSignedInt4;
  cudaResourceViewFormat ResViewFormatHalf1 = cudaResViewFormatHalf1;
  cudaResourceViewFormat ResViewFormatHalf2 = cudaResViewFormatHalf2;
  cudaResourceViewFormat ResViewFormatHalf4 = cudaResViewFormatHalf4;
  cudaResourceViewFormat ResViewFormatFloat1 = cudaResViewFormatFloat1;
  cudaResourceViewFormat ResViewFormatFloat2 = cudaResViewFormatFloat2;
  cudaResourceViewFormat ResViewFormatFloat4 = cudaResViewFormatFloat4;
  cudaResourceViewFormat ResViewFormatUnsignedBlockCompressed1 = cudaResViewFormatUnsignedBlockCompressed1;
  cudaResourceViewFormat ResViewFormatUnsignedBlockCompressed2 = cudaResViewFormatUnsignedBlockCompressed2;
  cudaResourceViewFormat ResViewFormatUnsignedBlockCompressed3 = cudaResViewFormatUnsignedBlockCompressed3;
  cudaResourceViewFormat ResViewFormatUnsignedBlockCompressed4 = cudaResViewFormatUnsignedBlockCompressed4;
  cudaResourceViewFormat ResViewFormatSignedBlockCompressed4 = cudaResViewFormatSignedBlockCompressed4;
  cudaResourceViewFormat ResViewFormatUnsignedBlockCompressed5 = cudaResViewFormatUnsignedBlockCompressed5;
  cudaResourceViewFormat ResViewFormatSignedBlockCompressed5 = cudaResViewFormatSignedBlockCompressed5;
  cudaResourceViewFormat ResViewFormatUnsignedBlockCompressed6H = cudaResViewFormatUnsignedBlockCompressed6H;
  cudaResourceViewFormat ResViewFormatSignedBlockCompressed6H = cudaResViewFormatSignedBlockCompressed6H;
  cudaResourceViewFormat ResViewFormatUnsignedBlockCompressed7 = cudaResViewFormatUnsignedBlockCompressed7;

  // CHECK: hipSharedMemConfig SharedMemConfig;
  // CHECK-NEXT: hipSharedMemConfig SharedMemBankSizeDefault = hipSharedMemBankSizeDefault;
  // CHECK-NEXT: hipSharedMemConfig SharedMemBankSizeFourByte = hipSharedMemBankSizeFourByte;
  // CHECK-NEXT: hipSharedMemConfig SharedMemBankSizeEightByte = hipSharedMemBankSizeEightByte;
  cudaSharedMemConfig SharedMemConfig;
  cudaSharedMemConfig SharedMemBankSizeDefault = cudaSharedMemBankSizeDefault;
  cudaSharedMemConfig SharedMemBankSizeFourByte = cudaSharedMemBankSizeFourByte;
  cudaSharedMemConfig SharedMemBankSizeEightByte = cudaSharedMemBankSizeEightByte;

  // CHECK: hipSurfaceBoundaryMode SurfaceBoundaryMode;
  // CHECK-NEXT: hipSurfaceBoundaryMode BoundaryModeZero = hipBoundaryModeZero;
  // CHECK-NEXT: hipSurfaceBoundaryMode BoundaryModeClamp = hipBoundaryModeClamp;
  // CHECK-NEXT: hipSurfaceBoundaryMode BoundaryModeTrap = hipBoundaryModeTrap;
  cudaSurfaceBoundaryMode SurfaceBoundaryMode;
  cudaSurfaceBoundaryMode BoundaryModeZero = cudaBoundaryModeZero;
  cudaSurfaceBoundaryMode BoundaryModeClamp = cudaBoundaryModeClamp;
  cudaSurfaceBoundaryMode BoundaryModeTrap = cudaBoundaryModeTrap;

  // CHECK: hipTextureAddressMode TextureAddressMode;
  // CHECK-NEXT: hipTextureAddressMode AddressModeWrap = hipAddressModeWrap;
  // CHECK-NEXT: hipTextureAddressMode AddressModeClamp = hipAddressModeClamp;
  // CHECK-NEXT: hipTextureAddressMode AddressModeMirror = hipAddressModeMirror;
  // CHECK-NEXT: hipTextureAddressMode AddressModeBorder = hipAddressModeBorder;
  cudaTextureAddressMode TextureAddressMode;
  cudaTextureAddressMode AddressModeWrap = cudaAddressModeWrap;
  cudaTextureAddressMode AddressModeClamp = cudaAddressModeClamp;
  cudaTextureAddressMode AddressModeMirror = cudaAddressModeMirror;
  cudaTextureAddressMode AddressModeBorder = cudaAddressModeBorder;

  // CHECK: hipTextureFilterMode TextureFilterMode;
  // CHECK-NEXT: hipTextureFilterMode FilterModePoint = hipFilterModePoint;
  // CHECK-NEXT: hipTextureFilterMode FilterModeLinear = hipFilterModeLinear;
  cudaTextureFilterMode TextureFilterMode;
  cudaTextureFilterMode FilterModePoint = cudaFilterModePoint;
  cudaTextureFilterMode FilterModeLinear = cudaFilterModeLinear;

  // CHECK: hipTextureReadMode TextureReadMode;
  // CHECK-NEXT: hipTextureReadMode ReadModeElementType = hipReadModeElementType;
  // CHECK-NEXT: hipTextureReadMode ReadModeNormalizedFloat = hipReadModeNormalizedFloat;
  cudaTextureReadMode TextureReadMode;
  cudaTextureReadMode ReadModeElementType = cudaReadModeElementType;
  cudaTextureReadMode ReadModeNormalizedFloat = cudaReadModeNormalizedFloat;

  // CHECK: hipGLDeviceList GLDeviceList;
  // CHECK-NEXT: hipGLDeviceList GLDeviceListAll = hipGLDeviceListAll;
  // CHECK-NEXT: hipGLDeviceList GLDeviceListCurrentFrame = hipGLDeviceListCurrentFrame;
  // CHECK-NEXT: hipGLDeviceList GLDeviceListNextFrame = hipGLDeviceListNextFrame;
  cudaGLDeviceList GLDeviceList;
  cudaGLDeviceList GLDeviceListAll = cudaGLDeviceListAll;
  cudaGLDeviceList GLDeviceListCurrentFrame = cudaGLDeviceListCurrentFrame;
  cudaGLDeviceList GLDeviceListNextFrame = cudaGLDeviceListNextFrame;

#if CUDA_VERSION >= 8000
  // CHECK: hipDeviceAttribute_t DevAttrHostNativeAtomicSupported = hipDeviceAttributeHostNativeAtomicSupported;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrSingleToDoublePrecisionPerfRatio = hipDeviceAttributeSingleToDoublePrecisionPerfRatio;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrPageableMemoryAccess = hipDeviceAttributePageableMemoryAccess;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrConcurrentManagedAccess = hipDeviceAttributeConcurrentManagedAccess;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrComputePreemptionSupported = hipDeviceAttributeComputePreemptionSupported;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrCanUseHostPointerForRegisteredMem = hipDeviceAttributeCanUseHostPointerForRegisteredMem;
  cudaDeviceAttr DevAttrHostNativeAtomicSupported = cudaDevAttrHostNativeAtomicSupported;
  cudaDeviceAttr DevAttrSingleToDoublePrecisionPerfRatio = cudaDevAttrSingleToDoublePrecisionPerfRatio;
  cudaDeviceAttr DevAttrPageableMemoryAccess = cudaDevAttrPageableMemoryAccess;
  cudaDeviceAttr DevAttrConcurrentManagedAccess = cudaDevAttrConcurrentManagedAccess;
  cudaDeviceAttr DevAttrComputePreemptionSupported = cudaDevAttrComputePreemptionSupported;
  cudaDeviceAttr DevAttrCanUseHostPointerForRegisteredMem = cudaDevAttrCanUseHostPointerForRegisteredMem;

  // CHECK: hipDeviceP2PAttr DeviceP2PAttr;
  // CHECK-NEXT: hipDeviceP2PAttr DevP2PAttrPerformanceRank = hipDevP2PAttrPerformanceRank;
  // CHECK-NEXT: hipDeviceP2PAttr DevP2PAttrAccessSupported = hipDevP2PAttrAccessSupported;
  // CHECK-NEXT: hipDeviceP2PAttr DevP2PAttrNativeAtomicSupported = hipDevP2PAttrNativeAtomicSupported;
  cudaDeviceP2PAttr DeviceP2PAttr;
  cudaDeviceP2PAttr DevP2PAttrPerformanceRank = cudaDevP2PAttrPerformanceRank;
  cudaDeviceP2PAttr DevP2PAttrAccessSupported = cudaDevP2PAttrAccessSupported;
  cudaDeviceP2PAttr DevP2PAttrNativeAtomicSupported = cudaDevP2PAttrNativeAtomicSupported;

  // CHECK: hipMemoryAdvise MemoryAdvise;
  // CHECK-NEXT: hipMemoryAdvise MemAdviseSetReadMostly = hipMemAdviseSetReadMostly;
  // CHECK-NEXT: hipMemoryAdvise MemAdviseUnsetReadMostly = hipMemAdviseUnsetReadMostly;
  // CHECK-NEXT: hipMemoryAdvise MemAdviseSetPreferredLocation = hipMemAdviseSetPreferredLocation;
  // CHECK-NEXT: hipMemoryAdvise MemAdviseUnsetPreferredLocation = hipMemAdviseUnsetPreferredLocation;
  // CHECK-NEXT: hipMemoryAdvise MemAdviseSetAccessedBy = hipMemAdviseSetAccessedBy;
  // CHECK-NEXT: hipMemoryAdvise MemAdviseUnsetAccessedBy = hipMemAdviseUnsetAccessedBy;
  cudaMemoryAdvise MemoryAdvise;
  cudaMemoryAdvise MemAdviseSetReadMostly = cudaMemAdviseSetReadMostly;
  cudaMemoryAdvise MemAdviseUnsetReadMostly = cudaMemAdviseUnsetReadMostly;
  cudaMemoryAdvise MemAdviseSetPreferredLocation = cudaMemAdviseSetPreferredLocation;
  cudaMemoryAdvise MemAdviseUnsetPreferredLocation = cudaMemAdviseUnsetPreferredLocation;
  cudaMemoryAdvise MemAdviseSetAccessedBy = cudaMemAdviseSetAccessedBy;
  cudaMemoryAdvise MemAdviseUnsetAccessedBy = cudaMemAdviseUnsetAccessedBy;

  // CHECK: hipMemRangeAttribute MemRangeAttribute;
  // CHECK-NEXT: hipMemRangeAttribute MemRangeAttributeReadMostly = hipMemRangeAttributeReadMostly;
  // CHECK-NEXT: hipMemRangeAttribute MemRangeAttributePreferredLocation = hipMemRangeAttributePreferredLocation;
  // CHECK-NEXT: hipMemRangeAttribute MemRangeAttributeAccessedBy = hipMemRangeAttributeAccessedBy;
  // CHECK-NEXT: hipMemRangeAttribute MemRangeAttributeLastPrefetchLocation = hipMemRangeAttributeLastPrefetchLocation;
  cudaMemRangeAttribute MemRangeAttribute;
  cudaMemRangeAttribute MemRangeAttributeReadMostly = cudaMemRangeAttributeReadMostly;
  cudaMemRangeAttribute MemRangeAttributePreferredLocation = cudaMemRangeAttributePreferredLocation;
  cudaMemRangeAttribute MemRangeAttributeAccessedBy = cudaMemRangeAttributeAccessedBy;
  cudaMemRangeAttribute MemRangeAttributeLastPrefetchLocation = cudaMemRangeAttributeLastPrefetchLocation;
#endif

#if CUDA_VERSION >= 9000
  // CHECK: hipDeviceAttribute_t DevAttrReserved94 = hipDeviceAttributeCanUseStreamWaitValue;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrCooperativeLaunch = hipDeviceAttributeCooperativeLaunch;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrCooperativeMultiDeviceLaunch = hipDeviceAttributeCooperativeMultiDeviceLaunch;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrMaxSharedMemoryPerBlockOptin = hipDeviceAttributeSharedMemPerBlockOptin;
  cudaDeviceAttr DevAttrReserved94 = cudaDevAttrReserved94;
  cudaDeviceAttr DevAttrCooperativeLaunch = cudaDevAttrCooperativeLaunch;
  cudaDeviceAttr DevAttrCooperativeMultiDeviceLaunch = cudaDevAttrCooperativeMultiDeviceLaunch;
  cudaDeviceAttr DevAttrMaxSharedMemoryPerBlockOptin = cudaDevAttrMaxSharedMemoryPerBlockOptin;

  // CHECK: hipError_t ErrorCooperativeLaunchTooLarge = hipErrorCooperativeLaunchTooLarge;
  cudaError_t ErrorCooperativeLaunchTooLarge = cudaErrorCooperativeLaunchTooLarge;

  // CHECK: hipFuncAttribute FuncAttribute;
  // CHECK-NEXT: hipFuncAttribute FuncAttributeMaxDynamicSharedMemorySize = hipFuncAttributeMaxDynamicSharedMemorySize;
  // CHECK-NEXT: hipFuncAttribute FuncAttributePreferredSharedMemoryCarveout = hipFuncAttributePreferredSharedMemoryCarveout;
  // CHECK-NEXT: hipFuncAttribute FuncAttributeMax = hipFuncAttributeMax;
  cudaFuncAttribute FuncAttribute;
  cudaFuncAttribute FuncAttributeMaxDynamicSharedMemorySize = cudaFuncAttributeMaxDynamicSharedMemorySize;
  cudaFuncAttribute FuncAttributePreferredSharedMemoryCarveout = cudaFuncAttributePreferredSharedMemoryCarveout;
  cudaFuncAttribute FuncAttributeMax = cudaFuncAttributeMax;
#endif

#if CUDA_VERSION >= 9020
  // CHECK: hipDeviceAttribute_t DevAttrPageableMemoryAccessUsesHostPageTables = hipDeviceAttributePageableMemoryAccessUsesHostPageTables;
  // CHECK-NEXT: hipDeviceAttribute_t DevAttrDirectManagedMemAccessFromHost = hipDeviceAttributeDirectManagedMemAccessFromHost;
  cudaDeviceAttr DevAttrPageableMemoryAccessUsesHostPageTables = cudaDevAttrPageableMemoryAccessUsesHostPageTables;
  cudaDeviceAttr DevAttrDirectManagedMemAccessFromHost = cudaDevAttrDirectManagedMemAccessFromHost;

  // CHECK: hipDeviceP2PAttr DevP2PAttrCudaArrayAccessSupported = hipDevP2PAttrHipArrayAccessSupported;
  cudaDeviceP2PAttr DevP2PAttrCudaArrayAccessSupported = cudaDevP2PAttrCudaArrayAccessSupported;

  // CHECK: hipDeviceAttribute_t DevAttrHostRegisterSupported = hipDeviceAttributeHostRegisterSupported;
  cudaDeviceAttr DevAttrHostRegisterSupported = cudaDevAttrHostRegisterSupported;
#endif

#if CUDA_VERSION >= 10000
  // CHECK: hipError_t ErrorStreamCaptureUnsupported = hipErrorStreamCaptureUnsupported;
  // CHECK-NEXT: hipError_t ErrorStreamCaptureInvalidated = hipErrorStreamCaptureInvalidated;
  // CHECK-NEXT: hipError_t ErrorStreamCaptureMerge = hipErrorStreamCaptureMerge;
  // CHECK-NEXT: hipError_t ErrorStreamCaptureUnmatched = hipErrorStreamCaptureUnmatched;
  // CHECK-NEXT: hipError_t ErrorStreamCaptureUnjoined = hipErrorStreamCaptureUnjoined;
  // CHECK-NEXT: hipError_t ErrorStreamCaptureIsolation = hipErrorStreamCaptureIsolation;
  // CHECK-NEXT: hipError_t ErrorStreamCaptureImplicit = hipErrorStreamCaptureImplicit;
  // CHECK-NEXT: hipError_t ErrorCapturedEvent = hipErrorCapturedEvent;
  // CHECK-NEXT: hipError_t ErrorIllegalState = hipErrorIllegalState;
  cudaError_t ErrorStreamCaptureUnsupported = cudaErrorStreamCaptureUnsupported;
  cudaError_t ErrorStreamCaptureInvalidated = cudaErrorStreamCaptureInvalidated;
  cudaError_t ErrorStreamCaptureMerge = cudaErrorStreamCaptureMerge;
  cudaError_t ErrorStreamCaptureUnmatched = cudaErrorStreamCaptureUnmatched;
  cudaError_t ErrorStreamCaptureUnjoined = cudaErrorStreamCaptureUnjoined;
  cudaError_t ErrorStreamCaptureIsolation = cudaErrorStreamCaptureIsolation;
  cudaError_t ErrorStreamCaptureImplicit = cudaErrorStreamCaptureImplicit;
  cudaError_t ErrorCapturedEvent = cudaErrorCapturedEvent;
  cudaError_t ErrorIllegalState = cudaErrorIllegalState;

  // CHECK: hipExternalMemoryHandleType ExternalMemoryHandleType;
  // CHECK-NEXT: hipExternalMemoryHandleType ExternalMemoryHandleTypeOpaqueFd = hipExternalMemoryHandleTypeOpaqueFd;
  // CHECK-NEXT: hipExternalMemoryHandleType ExternalMemoryHandleTypeOpaqueWin32 = hipExternalMemoryHandleTypeOpaqueWin32;
  // CHECK-NEXT: hipExternalMemoryHandleType ExternalMemoryHandleTypeOpaqueWin32Kmt = hipExternalMemoryHandleTypeOpaqueWin32Kmt;
  // CHECK-NEXT: hipExternalMemoryHandleType ExternalMemoryHandleTypeD3D12Heap = hipExternalMemoryHandleTypeD3D12Heap;
  // CHECK-NEXT: hipExternalMemoryHandleType ExternalMemoryHandleTypeD3D12Resource = hipExternalMemoryHandleTypeD3D12Resource;
  cudaExternalMemoryHandleType ExternalMemoryHandleType;
  cudaExternalMemoryHandleType ExternalMemoryHandleTypeOpaqueFd = cudaExternalMemoryHandleTypeOpaqueFd;
  cudaExternalMemoryHandleType ExternalMemoryHandleTypeOpaqueWin32 = cudaExternalMemoryHandleTypeOpaqueWin32;
  cudaExternalMemoryHandleType ExternalMemoryHandleTypeOpaqueWin32Kmt = cudaExternalMemoryHandleTypeOpaqueWin32Kmt;
  cudaExternalMemoryHandleType ExternalMemoryHandleTypeD3D12Heap = cudaExternalMemoryHandleTypeD3D12Heap;
  cudaExternalMemoryHandleType ExternalMemoryHandleTypeD3D12Resource = cudaExternalMemoryHandleTypeD3D12Resource;

  // CHECK: hipExternalSemaphoreHandleType ExternalSemaphoreHandleType;
  // CHECK-NEXT: hipExternalSemaphoreHandleType ExternalSemaphoreHandleTypeOpaqueFd = hipExternalSemaphoreHandleTypeOpaqueFd;
  // CHECK-NEXT: hipExternalSemaphoreHandleType ExternalSemaphoreHandleTypeOpaqueWin32 = hipExternalSemaphoreHandleTypeOpaqueWin32;
  // CHECK-NEXT: hipExternalSemaphoreHandleType ExternalSemaphoreHandleTypeOpaqueWin32Kmt = hipExternalSemaphoreHandleTypeOpaqueWin32Kmt;
  // CHECK-NEXT: hipExternalSemaphoreHandleType ExternalSemaphoreHandleTypeD3D12Fence = hipExternalSemaphoreHandleTypeD3D12Fence;
  cudaExternalSemaphoreHandleType ExternalSemaphoreHandleType;
  cudaExternalSemaphoreHandleType ExternalSemaphoreHandleTypeOpaqueFd = cudaExternalSemaphoreHandleTypeOpaqueFd;
  cudaExternalSemaphoreHandleType ExternalSemaphoreHandleTypeOpaqueWin32 = cudaExternalSemaphoreHandleTypeOpaqueWin32;
  cudaExternalSemaphoreHandleType ExternalSemaphoreHandleTypeOpaqueWin32Kmt = cudaExternalSemaphoreHandleTypeOpaqueWin32Kmt;
  cudaExternalSemaphoreHandleType ExternalSemaphoreHandleTypeD3D12Fence = cudaExternalSemaphoreHandleTypeD3D12Fence;

  // CHECK: hipGraphNodeType GraphNodeType;
  // CHECK-NEXT: hipGraphNodeType GraphNodeTypeKernel = hipGraphNodeTypeKernel;
  // CHECK-NEXT: hipGraphNodeType GraphNodeTypeMemcpy = hipGraphNodeTypeMemcpy;
  // CHECK-NEXT: hipGraphNodeType GraphNodeTypeMemset = hipGraphNodeTypeMemset;
  // CHECK-NEXT: hipGraphNodeType GraphNodeTypeHost = hipGraphNodeTypeHost;
  // CHECK-NEXT: hipGraphNodeType GraphNodeTypeGraph = hipGraphNodeTypeGraph;
  // CHECK-NEXT: hipGraphNodeType GraphNodeTypeEmpty = hipGraphNodeTypeEmpty;
  cudaGraphNodeType GraphNodeType;
  cudaGraphNodeType GraphNodeTypeKernel = cudaGraphNodeTypeKernel;
  cudaGraphNodeType GraphNodeTypeMemcpy = cudaGraphNodeTypeMemcpy;
  cudaGraphNodeType GraphNodeTypeMemset = cudaGraphNodeTypeMemset;
  cudaGraphNodeType GraphNodeTypeHost = cudaGraphNodeTypeHost;
  cudaGraphNodeType GraphNodeTypeGraph = cudaGraphNodeTypeGraph;
  cudaGraphNodeType GraphNodeTypeEmpty = cudaGraphNodeTypeEmpty;

  // CHECK: hipGraphNodeType GraphNodeTypeCount = hipGraphNodeTypeCount;
  cudaGraphNodeType GraphNodeTypeCount = cudaGraphNodeTypeCount;

  // CHECK: hipMemoryType MemoryTypeManaged = hipMemoryTypeManaged;
  cudaMemoryType MemoryTypeManaged = cudaMemoryTypeManaged;

  // CHECK: hipStreamCaptureStatus StreamCaptureStatus;
  // CHECK-NEXT: hipStreamCaptureStatus StreamCaptureStatusNone = hipStreamCaptureStatusNone;
  // CHECK-NEXT: hipStreamCaptureStatus StreamCaptureStatusActive = hipStreamCaptureStatusActive;
  // CHECK-NEXT: hipStreamCaptureStatus StreamCaptureStatusInvalidated = hipStreamCaptureStatusInvalidated;
  cudaStreamCaptureStatus StreamCaptureStatus;
  cudaStreamCaptureStatus StreamCaptureStatusNone = cudaStreamCaptureStatusNone;
  cudaStreamCaptureStatus StreamCaptureStatusActive = cudaStreamCaptureStatusActive;
  cudaStreamCaptureStatus StreamCaptureStatusInvalidated = cudaStreamCaptureStatusInvalidated;
#endif

#if CUDA_VERSION >= 10010
  // CHECK: hipError_t ErrorArrayIsMapped = hipErrorArrayIsMapped;
  // CHECK-NEXT: hipError_t ErrorAlreadyMapped = hipErrorAlreadyMapped;
  // CHECK-NEXT: hipError_t ErrorAlreadyAcquired = hipErrorAlreadyAcquired;
  // CHECK-NEXT: hipError_t ErrorNotMapped = hipErrorNotMapped;
  // CHECK-NEXT: hipError_t ErrorNotMappedAsArray = hipErrorNotMappedAsArray;
  // CHECK-NEXT: hipError_t ErrorNotMappedAsPointer = hipErrorNotMappedAsPointer;
  // CHECK-NEXT: hipError_t ErrorInvalidSource = hipErrorInvalidSource;
  // CHECK-NEXT: hipError_t ErrorFileNotFound = hipErrorFileNotFound;
  // CHECK-NEXT: hipError_t ErrorSymbolNotFound = hipErrorNotFound;
  // CHECK-NEXT: hipError_t ErrorContextIsDestroyed = hipErrorContextIsDestroyed;
  // CHECK-NEXT: hipError_t ErrorStreamCaptureWrongThread = hipErrorStreamCaptureWrongThread;
  cudaError_t ErrorArrayIsMapped = cudaErrorArrayIsMapped;
  cudaError_t ErrorAlreadyMapped = cudaErrorAlreadyMapped;
  cudaError_t ErrorAlreadyAcquired = cudaErrorAlreadyAcquired;
  cudaError_t ErrorNotMapped = cudaErrorNotMapped;
  cudaError_t ErrorNotMappedAsArray = cudaErrorNotMappedAsArray;
  cudaError_t ErrorNotMappedAsPointer = cudaErrorNotMappedAsPointer;
  cudaError_t ErrorInvalidSource = cudaErrorInvalidSource;
  cudaError_t ErrorFileNotFound = cudaErrorFileNotFound;
  cudaError_t ErrorSymbolNotFound = cudaErrorSymbolNotFound;
  cudaError_t ErrorContextIsDestroyed = cudaErrorContextIsDestroyed;
  cudaError_t ErrorStreamCaptureWrongThread = cudaErrorStreamCaptureWrongThread;

  // CHECK: hipStreamCaptureMode StreamCaptureMode;
  // CHECK-NEXT: hipStreamCaptureMode StreamCaptureModeGlobal = hipStreamCaptureModeGlobal;
  // CHECK-NEXT: hipStreamCaptureMode StreamCaptureModeThreadLocal = hipStreamCaptureModeThreadLocal;
  // CHECK-NEXT: hipStreamCaptureMode StreamCaptureModeRelaxed = hipStreamCaptureModeRelaxed;
  cudaStreamCaptureMode StreamCaptureMode;
  cudaStreamCaptureMode StreamCaptureModeGlobal = cudaStreamCaptureModeGlobal;
  cudaStreamCaptureMode StreamCaptureModeThreadLocal = cudaStreamCaptureModeThreadLocal;
  cudaStreamCaptureMode StreamCaptureModeRelaxed = cudaStreamCaptureModeRelaxed;
#endif

#if CUDA_VERSION >= 10020
  // CHECK: hipError_t ErrorDeviceUninitialized = hipErrorInvalidContext;
  // CHECK: hipError_t ErrorGraphExecUpdateFailure = hipErrorGraphExecUpdateFailure;
  cudaError_t ErrorDeviceUninitialized = cudaErrorDeviceUninitialized;
  cudaError_t ErrorGraphExecUpdateFailure = cudaErrorGraphExecUpdateFailure;

  // CHECK: hipExternalMemoryHandleType ExternalMemoryHandleTypeD3D11Resource = hipExternalMemoryHandleTypeD3D11Resource;
  // CHECK-NEXT: hipExternalMemoryHandleType ExternalMemoryHandleTypeD3D11ResourceKmt = hipExternalMemoryHandleTypeD3D11ResourceKmt;
  cudaExternalMemoryHandleType ExternalMemoryHandleTypeD3D11Resource = cudaExternalMemoryHandleTypeD3D11Resource;
  cudaExternalMemoryHandleType ExternalMemoryHandleTypeD3D11ResourceKmt = cudaExternalMemoryHandleTypeD3D11ResourceKmt;

  // CHECK: hipGraphExecUpdateResult GraphExecUpdateResult;
  // CHECK-NEXT: hipGraphExecUpdateResult GraphExecUpdateSuccess = hipGraphExecUpdateSuccess;
  // CHECK-NEXT: hipGraphExecUpdateResult GraphExecUpdateError = hipGraphExecUpdateError;
  // CHECK-NEXT: hipGraphExecUpdateResult GraphExecUpdateErrorTopologyChanged = hipGraphExecUpdateErrorTopologyChanged;
  // CHECK-NEXT: hipGraphExecUpdateResult GraphExecUpdateErrorNodeTypeChanged = hipGraphExecUpdateErrorNodeTypeChanged;
  // CHECK-NEXT: hipGraphExecUpdateResult GraphExecUpdateErrorFunctionChanged = hipGraphExecUpdateErrorFunctionChanged;
  // CHECK-NEXT: hipGraphExecUpdateResult GraphExecUpdateErrorParametersChanged = hipGraphExecUpdateErrorParametersChanged;
  // CHECK-NEXT: hipGraphExecUpdateResult GraphExecUpdateErrorNotSupported = hipGraphExecUpdateErrorNotSupported;
  cudaGraphExecUpdateResult GraphExecUpdateResult;
  cudaGraphExecUpdateResult GraphExecUpdateSuccess = cudaGraphExecUpdateSuccess;
  cudaGraphExecUpdateResult GraphExecUpdateError = cudaGraphExecUpdateError;
  cudaGraphExecUpdateResult GraphExecUpdateErrorTopologyChanged = cudaGraphExecUpdateErrorTopologyChanged;
  cudaGraphExecUpdateResult GraphExecUpdateErrorNodeTypeChanged = cudaGraphExecUpdateErrorNodeTypeChanged;
  cudaGraphExecUpdateResult GraphExecUpdateErrorFunctionChanged = cudaGraphExecUpdateErrorFunctionChanged;
  cudaGraphExecUpdateResult GraphExecUpdateErrorParametersChanged = cudaGraphExecUpdateErrorParametersChanged;
  cudaGraphExecUpdateResult GraphExecUpdateErrorNotSupported = cudaGraphExecUpdateErrorNotSupported;
#endif

#if CUDA_VERSION >= 11000
  // CHECK: hipDeviceAttribute_t DevAttrMaxBlocksPerMultiprocessor = hipDeviceAttributeMaxBlocksPerMultiprocessor;
  cudaDeviceAttr DevAttrMaxBlocksPerMultiprocessor = cudaDevAttrMaxBlocksPerMultiprocessor;

  // CHECK: hipKernelNodeAttrID kernelNodeAttrID;
  // CHECK-NEXT: hipKernelNodeAttrID KernelNodeAttributeAccessPolicyWindow = hipKernelNodeAttributeAccessPolicyWindow;
  // CHECK-NEXT: hipKernelNodeAttrID KernelNodeAttributeCooperative = hipKernelNodeAttributeCooperative;
  cudaKernelNodeAttrID kernelNodeAttrID;
  cudaKernelNodeAttrID KernelNodeAttributeAccessPolicyWindow = cudaKernelNodeAttributeAccessPolicyWindow;
  cudaKernelNodeAttrID KernelNodeAttributeCooperative = cudaKernelNodeAttributeCooperative;

  // CHECK: hipAccessProperty accessProperty;
  // CHECK-NEXT: hipAccessProperty AccessPropertyNormal = hipAccessPropertyNormal;
  // CHECK-NEXT: hipAccessProperty AccessPropertyStreaming = hipAccessPropertyStreaming;
  // CHECK-NEXT: hipAccessProperty AccessPropertyPersisting = hipAccessPropertyPersisting;
  cudaAccessProperty accessProperty;
  cudaAccessProperty AccessPropertyNormal = cudaAccessPropertyNormal;
  cudaAccessProperty AccessPropertyStreaming = cudaAccessPropertyStreaming;
  cudaAccessProperty AccessPropertyPersisting = cudaAccessPropertyPersisting;
#endif

#if CUDA_VERSION >= 11010
  // CHECK: hipGraphNodeType GraphNodeTypeWaitEvent = hipGraphNodeTypeWaitEvent;
  // CHECK-NEXT: hipGraphNodeType GraphNodeTypeEventRecord = hipGraphNodeTypeEventRecord;
  cudaGraphNodeType GraphNodeTypeWaitEvent = cudaGraphNodeTypeWaitEvent;
  cudaGraphNodeType GraphNodeTypeEventRecord = cudaGraphNodeTypeEventRecord;
#endif

#if CUDA_VERSION >= 11020
  // CHECK: hipDeviceAttribute_t DevAttrMemoryPoolsSupported = hipDeviceAttributeMemoryPoolsSupported;
  cudaDeviceAttr DevAttrMemoryPoolsSupported = cudaDevAttrMemoryPoolsSupported;

  // CHECK: hipGraphExecUpdateResult GraphExecUpdateErrorUnsupportedFunctionChange = hipGraphExecUpdateErrorUnsupportedFunctionChange;
  cudaGraphExecUpdateResult GraphExecUpdateErrorUnsupportedFunctionChange = cudaGraphExecUpdateErrorUnsupportedFunctionChange;

  // CHECK: hipMemPoolAttr MemPoolAttr;
  // CHECK-NEXT: hipMemPoolAttr MemPoolReuseFollowEventDependencies = hipMemPoolReuseFollowEventDependencies;
  // CHECK-NEXT: hipMemPoolAttr MemPoolReuseAllowOpportunistic = hipMemPoolReuseAllowOpportunistic;
  // CHECK-NEXT: hipMemPoolAttr MemPoolReuseAllowInternalDependencies = hipMemPoolReuseAllowInternalDependencies;
  // CHECK-NEXT: hipMemPoolAttr MemPoolAttrReleaseThreshold = hipMemPoolAttrReleaseThreshold;
  cudaMemPoolAttr MemPoolAttr;
  cudaMemPoolAttr MemPoolReuseFollowEventDependencies = cudaMemPoolReuseFollowEventDependencies;
  cudaMemPoolAttr MemPoolReuseAllowOpportunistic = cudaMemPoolReuseAllowOpportunistic;
  cudaMemPoolAttr MemPoolReuseAllowInternalDependencies = cudaMemPoolReuseAllowInternalDependencies;
  cudaMemPoolAttr MemPoolAttrReleaseThreshold = cudaMemPoolAttrReleaseThreshold;

  // CHECK: hipMemLocationType memLocationType;
  // CHECK-NEXT: hipMemLocationType MemLocationTypeInvalid = hipMemLocationTypeInvalid;
  // CHECK-NEXT: hipMemLocationType MemLocationTypeDevice = hipMemLocationTypeDevice;
  cudaMemLocationType memLocationType;
  cudaMemLocationType MemLocationTypeInvalid = cudaMemLocationTypeInvalid;
  cudaMemLocationType MemLocationTypeDevice = cudaMemLocationTypeDevice;

  // CHECK: hipMemAccessFlags MemAccessFlags;
  // CHECK-NEXT: hipMemAccessFlags MemAccessFlagsProtNone = hipMemAccessFlagsProtNone;
  // CHECK-NEXT: hipMemAccessFlags MemAccessFlagsProtRead = hipMemAccessFlagsProtRead;
  // CHECK-NEXT: hipMemAccessFlags MemAccessFlagsProtReadWrite = hipMemAccessFlagsProtReadWrite;
  cudaMemAccessFlags MemAccessFlags;
  cudaMemAccessFlags MemAccessFlagsProtNone = cudaMemAccessFlagsProtNone;
  cudaMemAccessFlags MemAccessFlagsProtRead = cudaMemAccessFlagsProtRead;
  cudaMemAccessFlags MemAccessFlagsProtReadWrite = cudaMemAccessFlagsProtReadWrite;

  // CHECK: hipMemAllocationType memAllocationType;
  // CHECK-NEXT: hipMemAllocationType MemAllocationTypeInvalid = hipMemAllocationTypeInvalid;
  // CHECK-NEXT: hipMemAllocationType MemAllocationTypePinned = hipMemAllocationTypePinned;
  // CHECK-NEXT: hipMemAllocationType MemAllocationTypeMax = hipMemAllocationTypeMax;
  cudaMemAllocationType memAllocationType;
  cudaMemAllocationType MemAllocationTypeInvalid = cudaMemAllocationTypeInvalid;
  cudaMemAllocationType MemAllocationTypePinned = cudaMemAllocationTypePinned;
  cudaMemAllocationType MemAllocationTypeMax = cudaMemAllocationTypeMax;

  // CHECK: hipMemAllocationHandleType memAllocationHandleType;
  // CHECK-NEXT: hipMemAllocationHandleType MEM_HANDLE_TYPE_NONE = hipMemHandleTypeNone;
  // CHECK-NEXT: hipMemAllocationHandleType MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = hipMemHandleTypePosixFileDescriptor;
  // CHECK-NEXT: hipMemAllocationHandleType MEM_HANDLE_TYPE_WIN32 = hipMemHandleTypeWin32;
  // CHECK-NEXT: hipMemAllocationHandleType MEM_HANDLE_TYPE_WIN32_KMT = hipMemHandleTypeWin32Kmt;
  cudaMemAllocationHandleType memAllocationHandleType;
  cudaMemAllocationHandleType MEM_HANDLE_TYPE_NONE = cudaMemHandleTypeNone;
  cudaMemAllocationHandleType MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR = cudaMemHandleTypePosixFileDescriptor;
  cudaMemAllocationHandleType MEM_HANDLE_TYPE_WIN32 = cudaMemHandleTypeWin32;
  cudaMemAllocationHandleType MEM_HANDLE_TYPE_WIN32_KMT = cudaMemHandleTypeWin32Kmt;
#endif

#if CUDA_VERSION >= 11030
  // CHECK: hipStreamUpdateCaptureDependenciesFlags StreamUpdateCaptureDependenciesFlags;
  // CHECK-NEXT: hipStreamUpdateCaptureDependenciesFlags StreamAddCaptureDependencies = hipStreamAddCaptureDependencies;
  // CHECK-NEXT: hipStreamUpdateCaptureDependenciesFlags StreamSetCaptureDependencies = hipStreamSetCaptureDependencies;
  cudaStreamUpdateCaptureDependenciesFlags StreamUpdateCaptureDependenciesFlags;
  cudaStreamUpdateCaptureDependenciesFlags StreamAddCaptureDependencies = cudaStreamAddCaptureDependencies;
  cudaStreamUpdateCaptureDependenciesFlags StreamSetCaptureDependencies = cudaStreamSetCaptureDependencies;

  // CHECK: hipMemPoolAttr MemPoolAttrReservedMemCurrent = hipMemPoolAttrReservedMemCurrent;
  // CHECK-NEXT: hipMemPoolAttr MemPoolAttrReservedMemHigh = hipMemPoolAttrReservedMemHigh;
  // CHECK-NEXT: hipMemPoolAttr MemPoolAttrUsedMemCurrent = hipMemPoolAttrUsedMemCurrent;
  // CHECK-NEXT: hipMemPoolAttr MemPoolAttrUsedMemHigh = hipMemPoolAttrUsedMemHigh;
  cudaMemPoolAttr MemPoolAttrReservedMemCurrent = cudaMemPoolAttrReservedMemCurrent;
  cudaMemPoolAttr MemPoolAttrReservedMemHigh = cudaMemPoolAttrReservedMemHigh;
  cudaMemPoolAttr MemPoolAttrUsedMemCurrent = cudaMemPoolAttrUsedMemCurrent;
  cudaMemPoolAttr MemPoolAttrUsedMemHigh = cudaMemPoolAttrUsedMemHigh;

  // CHECK: hipUserObjectFlags UserObjectFlags;
  // CHECK-NEXT: hipUserObjectFlags UserObjectNoDestructorSync = hipUserObjectNoDestructorSync;
  cudaUserObjectFlags UserObjectFlags;
  cudaUserObjectFlags UserObjectNoDestructorSync = cudaUserObjectNoDestructorSync;

  // CHECK: hipUserObjectRetainFlags UserObjectRetainFlags;
  // CHECK-NEXT: hipUserObjectRetainFlags GraphUserObjectMove = hipGraphUserObjectMove;
  cudaUserObjectRetainFlags UserObjectRetainFlags;
  cudaUserObjectRetainFlags GraphUserObjectMove = cudaGraphUserObjectMove;

  // CHECK: hipGraphDebugDotFlags graphDebugDot_flags;
  // CHECK-NEXT: hipGraphDebugDotFlags graphDebugDot_flags_enum;
  // CHECK-NEXT: hipGraphDebugDotFlags GRAPH_DEBUG_DOT_FLAGS_VERBOSE = hipGraphDebugDotFlagsVerbose;
  // CHECK-NEXT: hipGraphDebugDotFlags GRAPH_DEBUG_DOT_FLAGS_RUNTIME_TYPES = hipGraphDebugDotFlagsRuntimeTypes;
  // CHECK-NEXT: hipGraphDebugDotFlags GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS = hipGraphDebugDotFlagsKernelNodeParams;
  // CHECK-NEXT: hipGraphDebugDotFlags GRAPH_DEBUG_DOT_FLAGS_MEMCPY_NODE_PARAMS = hipGraphDebugDotFlagsMemcpyNodeParams;
  // CHECK-NEXT: hipGraphDebugDotFlags GRAPH_DEBUG_DOT_FLAGS_MEMSET_NODE_PARAMS = hipGraphDebugDotFlagsMemsetNodeParams;
  // CHECK-NEXT: hipGraphDebugDotFlags GRAPH_DEBUG_DOT_FLAGS_HOST_NODE_PARAMS = hipGraphDebugDotFlagsHostNodeParams;
  // CHECK-NEXT: hipGraphDebugDotFlags GRAPH_DEBUG_DOT_FLAGS_EVENT_NODE_PARAMS = hipGraphDebugDotFlagsEventNodeParams;
  // CHECK-NEXT: hipGraphDebugDotFlags GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_SIGNAL_NODE_PARAMS = hipGraphDebugDotFlagsExtSemasSignalNodeParams;
  // CHECK-NEXT: hipGraphDebugDotFlags GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_WAIT_NODE_PARAMS = hipGraphDebugDotFlagsExtSemasWaitNodeParams;
  // CHECK-NEXT: hipGraphDebugDotFlags GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_ATTRIBUTES = hipGraphDebugDotFlagsKernelNodeAttributes;
  // CHECK-NEXT: hipGraphDebugDotFlags GRAPH_DEBUG_DOT_FLAGS_HANDLES = hipGraphDebugDotFlagsHandles;
  CUgraphDebugDot_flags graphDebugDot_flags;
  CUgraphDebugDot_flags_enum graphDebugDot_flags_enum;
  CUgraphDebugDot_flags GRAPH_DEBUG_DOT_FLAGS_VERBOSE = CU_GRAPH_DEBUG_DOT_FLAGS_VERBOSE;
  CUgraphDebugDot_flags GRAPH_DEBUG_DOT_FLAGS_RUNTIME_TYPES = CU_GRAPH_DEBUG_DOT_FLAGS_RUNTIME_TYPES;
  CUgraphDebugDot_flags GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS = CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_PARAMS;
  CUgraphDebugDot_flags GRAPH_DEBUG_DOT_FLAGS_MEMCPY_NODE_PARAMS = CU_GRAPH_DEBUG_DOT_FLAGS_MEMCPY_NODE_PARAMS;
  CUgraphDebugDot_flags GRAPH_DEBUG_DOT_FLAGS_MEMSET_NODE_PARAMS = CU_GRAPH_DEBUG_DOT_FLAGS_MEMSET_NODE_PARAMS;
  CUgraphDebugDot_flags GRAPH_DEBUG_DOT_FLAGS_HOST_NODE_PARAMS = CU_GRAPH_DEBUG_DOT_FLAGS_HOST_NODE_PARAMS;
  CUgraphDebugDot_flags GRAPH_DEBUG_DOT_FLAGS_EVENT_NODE_PARAMS = CU_GRAPH_DEBUG_DOT_FLAGS_EVENT_NODE_PARAMS;
  CUgraphDebugDot_flags GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_SIGNAL_NODE_PARAMS = CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_SIGNAL_NODE_PARAMS;
  CUgraphDebugDot_flags GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_WAIT_NODE_PARAMS = CU_GRAPH_DEBUG_DOT_FLAGS_EXT_SEMAS_WAIT_NODE_PARAMS;
  CUgraphDebugDot_flags GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_ATTRIBUTES = CU_GRAPH_DEBUG_DOT_FLAGS_KERNEL_NODE_ATTRIBUTES;
  CUgraphDebugDot_flags GRAPH_DEBUG_DOT_FLAGS_HANDLES = CU_GRAPH_DEBUG_DOT_FLAGS_HANDLES;

  // CHECK: hipFlushGPUDirectRDMAWritesOptions flushGPUDirectRDMAWritesOptions;
  // CHECK-NEXT: hipFlushGPUDirectRDMAWritesOptions FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST = hipFlushGPUDirectRDMAWritesOptionHost;
  // CHECK-NEXT: hipFlushGPUDirectRDMAWritesOptions FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_MEMOPS = hipFlushGPUDirectRDMAWritesOptionMemOps;
  cudaFlushGPUDirectRDMAWritesOptions flushGPUDirectRDMAWritesOptions;
  cudaFlushGPUDirectRDMAWritesOptions FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST = cudaFlushGPUDirectRDMAWritesOptionHost;
  cudaFlushGPUDirectRDMAWritesOptions FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_MEMOPS = cudaFlushGPUDirectRDMAWritesOptionMemOps;

  // CHECK: hipGPUDirectRDMAWritesOrdering GPUDirectRDMAWritesOrdering;
  // CHECK-NEXT: hipGPUDirectRDMAWritesOrdering GPU_DIRECT_RDMA_WRITES_ORDERING_NONE = hipGPUDirectRDMAWritesOrderingNone;
  // CHECK-NEXT: hipGPUDirectRDMAWritesOrdering GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER = hipGPUDirectRDMAWritesOrderingOwner;
  // CHECK-NEXT: hipGPUDirectRDMAWritesOrdering GPU_DIRECT_RDMA_WRITES_ORDERING_ALL_DEVICES = hipGPUDirectRDMAWritesOrderingAllDevices;
  cudaGPUDirectRDMAWritesOrdering GPUDirectRDMAWritesOrdering;
  cudaGPUDirectRDMAWritesOrdering GPU_DIRECT_RDMA_WRITES_ORDERING_NONE = cudaGPUDirectRDMAWritesOrderingNone;
  cudaGPUDirectRDMAWritesOrdering GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER = cudaGPUDirectRDMAWritesOrderingOwner;
  cudaGPUDirectRDMAWritesOrdering GPU_DIRECT_RDMA_WRITES_ORDERING_ALL_DEVICES = cudaGPUDirectRDMAWritesOrderingAllDevices;
#endif

#if CUDA_VERSION >= 11040
  // CHECK: hipGraphInstantiateFlags GraphInstantiateFlags;
  // CHECK-NEXT: hipGraphInstantiateFlags GraphInstantiateFlagAutoFreeOnLaunch = hipGraphInstantiateFlagAutoFreeOnLaunch;
  cudaGraphInstantiateFlags GraphInstantiateFlags;
  cudaGraphInstantiateFlags GraphInstantiateFlagAutoFreeOnLaunch = cudaGraphInstantiateFlagAutoFreeOnLaunch;

  // CHECK: hipGraphMemAttributeType GraphMemAttributeType;
  // CHECK-NEXT: hipGraphMemAttributeType GraphMemAttrUsedMemCurrent = hipGraphMemAttrUsedMemCurrent;
  // CHECK-NEXT: hipGraphMemAttributeType GraphMemAttrUsedMemHigh = hipGraphMemAttrUsedMemHigh;
  // CHECK-NEXT: hipGraphMemAttributeType GraphMemAttrReservedMemCurrent = hipGraphMemAttrReservedMemCurrent;
  // CHECK-NEXT: hipGraphMemAttributeType GraphMemAttrReservedMemHigh = hipGraphMemAttrReservedMemHigh;
  cudaGraphMemAttributeType GraphMemAttributeType;
  cudaGraphMemAttributeType GraphMemAttrUsedMemCurrent = cudaGraphMemAttrUsedMemCurrent;
  cudaGraphMemAttributeType GraphMemAttrUsedMemHigh = cudaGraphMemAttrUsedMemHigh;
  cudaGraphMemAttributeType GraphMemAttrReservedMemCurrent = cudaGraphMemAttrReservedMemCurrent;
  cudaGraphMemAttributeType GraphMemAttrReservedMemHigh = cudaGraphMemAttrReservedMemHigh;

  // CHECK: hipGraphNodeType GraphNodeTypeExtSemaphoreSignal = hipGraphNodeTypeExtSemaphoreSignal;
  // CHECK-NEXT: hipGraphNodeType GraphNodeTypeExtSemaphoreWait = hipGraphNodeTypeExtSemaphoreWait;
  // CHECK-NEXT: hipGraphNodeType GraphNodeTypeMemAlloc = hipGraphNodeTypeMemAlloc;
  // CHECK-NEXT: hipGraphNodeType GraphNodeTypeMemFree = hipGraphNodeTypeMemFree;
  cudaGraphNodeType GraphNodeTypeExtSemaphoreSignal = cudaGraphNodeTypeExtSemaphoreSignal;
  cudaGraphNodeType GraphNodeTypeExtSemaphoreWait = cudaGraphNodeTypeExtSemaphoreWait;
  cudaGraphNodeType GraphNodeTypeMemAlloc = cudaGraphNodeTypeMemAlloc;
  cudaGraphNodeType GraphNodeTypeMemFree = cudaGraphNodeTypeMemFree;
#endif

#if CUDA_VERSION >= 11070
  // CHECK: hipGraphInstantiateFlags GraphInstantiateFlagUseNodePriority = hipGraphInstantiateFlagUseNodePriority;
  cudaGraphInstantiateFlags GraphInstantiateFlagUseNodePriority = cudaGraphInstantiateFlagUseNodePriority;
#endif

#if CUDA_VERSION >= 12000
  // CHECK: hipGraphInstantiateFlags GraphInstantiateFlagUpload = hipGraphInstantiateFlagUpload;
  // CHECK-NEXT: hipGraphInstantiateFlags GraphInstantiateFlagDeviceLaunch = hipGraphInstantiateFlagDeviceLaunch;
  cudaGraphInstantiateFlags GraphInstantiateFlagUpload = cudaGraphInstantiateFlagUpload;
  cudaGraphInstantiateFlags GraphInstantiateFlagDeviceLaunch = cudaGraphInstantiateFlagDeviceLaunch;

  // CHECK: hipGraphInstantiateResult graphInstantiateResult;
  // CHECK-NEXT: hipGraphInstantiateResult GRAPH_INSTANTIATE_SUCCESS = hipGraphInstantiateSuccess;
  // CHECK-NEXT: hipGraphInstantiateResult GRAPH_INSTANTIATE_ERROR = hipGraphInstantiateError;
  // CHECK-NEXT: hipGraphInstantiateResult GRAPH_INSTANTIATE_INVALID_STRUCTURE = hipGraphInstantiateInvalidStructure;
  // CHECK-NEXT: hipGraphInstantiateResult GRAPH_INSTANTIATE_NODE_OPERATION_NOT_SUPPORTED = hipGraphInstantiateNodeOperationNotSupported;
  // CHECK-NEXT: hipGraphInstantiateResult GRAPH_INSTANTIATE_MULTIPLE_CTXS_NOT_SUPPORTED = hipGraphInstantiateMultipleDevicesNotSupported;
  cudaGraphInstantiateResult graphInstantiateResult;
  cudaGraphInstantiateResult GRAPH_INSTANTIATE_SUCCESS = cudaGraphInstantiateSuccess;
  cudaGraphInstantiateResult GRAPH_INSTANTIATE_ERROR = cudaGraphInstantiateError;
  cudaGraphInstantiateResult GRAPH_INSTANTIATE_INVALID_STRUCTURE = cudaGraphInstantiateInvalidStructure;
  cudaGraphInstantiateResult GRAPH_INSTANTIATE_NODE_OPERATION_NOT_SUPPORTED = cudaGraphInstantiateNodeOperationNotSupported;
  cudaGraphInstantiateResult GRAPH_INSTANTIATE_MULTIPLE_CTXS_NOT_SUPPORTED = cudaGraphInstantiateMultipleDevicesNotSupported;
#endif

  return 0;
}
