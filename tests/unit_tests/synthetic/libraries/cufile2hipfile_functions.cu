// RUN: %run_test hipify "%s" "%t" %hipify_args 3 --amap --skip-excluded-preprocessor-conditional-blocks --experimental %clang_args -D__CUDA_API_VERSION_INTERNAL -ferror-limit=500

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <stdio.h>
// CHECK: #include "hipfile.h"
#include "cufile.h"
// CHECK-NOT: #include "hipfile.h"

int main() {
  printf("28. cuFile Functions API to hipFile API synthetic test\n");

  // CHECK: hipFileHandle_t fileHandle;
  CUfileHandle_t fileHandle;

  // CHECK: hipFileDescr_t fileDescr;
  CUfileDescr_t fileDescr;

  // CHECK: hipFileError_t fileError;
  CUfileError_t fileError;

  // CHECK: hipFileDriverProps_t driverProps;
  CUfileDrvProps_t driverProps;

  // CHECK: hipFileOpError_t opError;
  CUfileOpError opError;

  void* bufPtr = nullptr;
  const void* constBufPtr = nullptr;
  size_t bufSize = 4096;
  off_t fileOffset = 0;
  off_t bufOffset = 0;
  ssize_t bytesTransferred = 0;
  int fd = -1;
  int flags = 0;

  // CUDA: static inline const char *cufileop_status_error(CUfileOpError status)
  // HIP: const char *hipFileOpStatusError(hipFileOpError_t status)
  // CHECK: const char* errorStr = hipFileOpStatusError(opError);
  const char* errorStr = cufileop_status_error(opError);

  // CUDA: CUfileError_t cuFileHandleRegister(CUfileHandle_t *fh, CUfileDescr_t *descr);
  // HIP: hipFileError_t hipFileHandleRegister(hipFileHandle_t *fh, hipFileDescr_t *descr);
  // CHECK: fileError = hipFileHandleRegister(&fileHandle, &fileDescr);
  fileError = cuFileHandleRegister(&fileHandle, &fileDescr);

  // CUDA: void cuFileHandleDeregister(CUfileHandle_t fh);
  // HIP: void hipFileHandleDeregister(hipFileHandle_t fh);
  // CHECK: hipFileHandleDeregister(fileHandle);
  cuFileHandleDeregister(fileHandle);

  // CUDA: CUfileError_t cuFileBufRegister(const void *bufPtr_base, size_t length, int flags);
  // HIP: hipFileError_t hipFileBufRegister(const void *bufPtr_base, size_t length, int flags);
  // CHECK: fileError = hipFileBufRegister(constBufPtr, bufSize, flags);
  fileError = cuFileBufRegister(constBufPtr, bufSize, flags);

  // CUDA: CUfileError_t cuFileBufDeregister(const void *bufPtr_base);
  // HIP: hipFileError_t hipFileBufDeregister(const void *bufPtr_base);
  // CHECK: fileError = hipFileBufDeregister(constBufPtr);
  fileError = cuFileBufDeregister(constBufPtr);

  // CUDA: ssize_t cuFileRead(CUfileHandle_t fh, void *bufPtr_base, size_t size, off_t file_offset, off_t bufPtr_offset);
  // HIP: ssize_t hipFileRead(hipFileHandle_t fh, void *bufPtr_base, size_t size, off_t file_offset, off_t bufPtr_offset);
  // CHECK: bytesTransferred = hipFileRead(fileHandle, bufPtr, bufSize, fileOffset, bufOffset);
  bytesTransferred = cuFileRead(fileHandle, bufPtr, bufSize, fileOffset, bufOffset);

  // CUDA: ssize_t cuFileWrite(CUfileHandle_t fh, const void *bufPtr_base, size_t size, off_t file_offset, off_t bufPtr_offset);
  // HIP: ssize_t hipFileWrite(hipFileHandle_t fh, const void *bufPtr_base, size_t size, off_t file_offset, off_t bufPtr_offset);
  // CHECK: bytesTransferred = hipFileWrite(fileHandle, constBufPtr, bufSize, fileOffset, bufOffset);
  bytesTransferred = cuFileWrite(fileHandle, constBufPtr, bufSize, fileOffset, bufOffset);

  // CUDA: CUfileError_t cuFileDriverOpen(void);
  // HIP: hipFileError_t hipFileDriverOpen(void);
  // CHECK: fileError = hipFileDriverOpen();
  fileError = cuFileDriverOpen();

  // CUDA: CUfileError_t cuFileDriverClose(void);
  // HIP: hipFileError_t hipFileDriverClose(void);
  // CHECK: fileError = hipFileDriverClose();
  fileError = cuFileDriverClose();

  // CUDA: CUfileError_t cuFileDriverGetProperties(CUfileDrvProps_t *props);
  // HIP: hipFileError_t hipFileDriverGetProperties(hipFileDriverProps_t *props);
  // CHECK: fileError = hipFileDriverGetProperties(&driverProps);
  fileError = cuFileDriverGetProperties(&driverProps);

  // CUDA: CUfileError_t cuFileDriverSetPollMode(bool poll, size_t poll_threshold_size);
  // HIP: hipFileError_t hipFileDriverSetPollMode(bool poll, size_t poll_threshold_size);
  // CHECK: fileError = hipFileDriverSetPollMode(true, 4096);
  fileError = cuFileDriverSetPollMode(true, 4096);

  // CUDA: CUfileError_t cuFileDriverSetMaxDirectIOSize(size_t max_direct_io_size);
  // HIP: hipFileError_t hipFileDriverSetMaxDirectIOSize(size_t max_direct_io_size);
  // CHECK: fileError = hipFileDriverSetMaxDirectIOSize(1048576);
  fileError = cuFileDriverSetMaxDirectIOSize(1048576);

  // CUDA: CUfileError_t cuFileDriverSetMaxCacheSize(size_t max_cache_size);
  // HIP: hipFileError_t hipFileDriverSetMaxCacheSize(size_t max_cache_size);
  // CHECK: fileError = hipFileDriverSetMaxCacheSize(2097152);
  fileError = cuFileDriverSetMaxCacheSize(2097152);

  // CUDA: CUfileError_t cuFileDriverSetMaxPinnedMemSize(size_t max_pinned_size);
  // HIP: hipFileError_t hipFileDriverSetMaxPinnedMemSize(size_t max_pinned_size);
  // CHECK: fileError = hipFileDriverSetMaxPinnedMemSize(4194304);
  fileError = cuFileDriverSetMaxPinnedMemSize(4194304);

#if CUDA_VERSION >= 11060
  // CHECK: hipFileBatchHandle_t batchHandle;
  CUfileBatchHandle_t batchHandle;

  // CHECK: hipFileIOParams_t ioParams;
  CUfileIOParams_t ioParams;

  // CHECK: hipFileIOEvents_t ioEvents;
  CUfileIOEvents_t ioEvents;

  unsigned int numRequests = 1;
  unsigned int completedRequests = 0;

  // CUDA: CUfileError_t cuFileBatchIOSetUp(CUfileBatchHandle_t *batch_idp, unsigned nr);
  // HIP: hipFileError_t hipFileBatchIOSetUp(hipFileBatchHandle_t *batch_idp, unsigned nr);
  // CHECK: fileError = hipFileBatchIOSetUp(&batchHandle, numRequests);
  fileError = cuFileBatchIOSetUp(&batchHandle, numRequests);

  // CUDA: CUfileError_t cuFileBatchIOSubmit(CUfileBatchHandle_t batch_idp, unsigned nr, CUfileIOParams_t *iocbp, unsigned int flags);
  // HIP: hipFileError_t hipFileBatchIOSubmit(hipFileBatchHandle_t batch_idp, unsigned nr, hipFileIOParams_t *iocbp, unsigned int flags);
  // CHECK: fileError = hipFileBatchIOSubmit(batchHandle, numRequests, &ioParams, flags);
  fileError = cuFileBatchIOSubmit(batchHandle, numRequests, &ioParams, flags);

  // CUDA: CUfileError_t cuFileBatchIOGetStatus(CUfileBatchHandle_t batch_idp, unsigned min_nr, unsigned* nr, CUfileIOEvents_t *iocbp, struct timespec* timeout);
  // HIP: hipFileError_t hipFileBatchIOGetStatus(hipFileBatchHandle_t batch_idp, unsigned min_nr, unsigned* nr, hipFileIOEvents_t *iocbp, struct timespec* timeout);
  // CHECK: fileError = hipFileBatchIOGetStatus(batchHandle, 1, &completedRequests, &ioEvents, NULL);
  fileError = cuFileBatchIOGetStatus(batchHandle, 1, &completedRequests, &ioEvents, NULL);

  // CUDA: CUfileError_t cuFileBatchIOCancel(CUfileBatchHandle_t batch_idp);
  // HIP: hipFileError_t hipFileBatchIOCancel(hipFileBatchHandle_t batch_idp);
  // CHECK: fileError = hipFileBatchIOCancel(batchHandle);
  fileError = cuFileBatchIOCancel(batchHandle);

  // CUDA: void cuFileBatchIODestroy(CUfileBatchHandle_t batch_idp);
  // HIP: void hipFileBatchIODestroy(hipFileBatchHandle_t batch_idp);
  // CHECK: hipFileBatchIODestroy(batchHandle);
  cuFileBatchIODestroy(batchHandle);
#endif

#if CUDA_VERSION >= 11080
  // CUDA: CUfileError_t cuFileDriverClose_v2(void);
  // HIP: hipFileError_t hipFileDriverClose(void);
  // CHECK: fileError = hipFileDriverClose();
  fileError = cuFileDriverClose_v2();

  // CUDA: long cuFileUseCount(void);
  // HIP: long hipFileUseCount(void);
  // CHECK: long useCount = hipFileUseCount();
  long useCount = cuFileUseCount();
#endif

#if CUDA_VERSION >= 12020
  // CHECK: hipStream_t stream;
  cudaStream_t stream;

  size_t readSize = 4096;
  size_t writeSize = 4096;
  off_t readFileOffset = 0;
  off_t writeFileOffset = 0;
  off_t readBufOffset = 0;
  off_t writeBufOffset = 0;
  ssize_t bytesRead = 0;
  ssize_t bytesWritten = 0;

  // CUDA: CUfileError_t cuFileStreamRegister(CUstream stream, unsigned flags);
  // HIP: hipFileError_t hipFileStreamRegister(hipStream_t stream, unsigned flags);
  // CHECK: fileError = hipFileStreamRegister(stream, flags);
  fileError = cuFileStreamRegister(stream, flags);

  // CUDA: CUfileError_t cuFileStreamDeregister(CUstream stream);
  // HIP: hipFileError_t hipFileStreamDeregister(hipStream_t stream);
  // CHECK: fileError = hipFileStreamDeregister(stream);
  fileError = cuFileStreamDeregister(stream);

  // CUDA: CUfileError_t cuFileReadAsync(CUfileHandle_t fh, void *bufPtr_base, size_t *size_p, off_t *file_offset_p, off_t *bufPtr_offset_p, ssize_t *bytes_read_p, CUstream stream);
  // HIP: hipFileError_t hipFileReadAsync(hipFileHandle_t fh, void *bufPtr_base, size_t *size_p, off_t *file_offset_p, off_t *bufPtr_offset_p, ssize_t *bytes_read_p, hipStream_t stream);
  // CHECK: fileError = hipFileReadAsync(fileHandle, bufPtr, &readSize, &readFileOffset, &readBufOffset, &bytesRead, stream);
  fileError = cuFileReadAsync(fileHandle, bufPtr, &readSize, &readFileOffset, &readBufOffset, &bytesRead, stream);

  // CUDA: CUfileError_t cuFileWriteAsync(CUfileHandle_t fh, void *bufPtr_base, size_t *size_p, off_t *file_offset_p, off_t *bufPtr_offset_p, ssize_t *bytes_written_p, CUstream stream);
  // HIP: hipFileError_t hipFileWriteAsync(hipFileHandle_t fh, void *bufPtr_base, size_t *size_p, off_t *file_offset_p, off_t *bufPtr_offset_p, ssize_t *bytes_written_p, hipStream_t stream);
  // CHECK: fileError = hipFileWriteAsync(fileHandle, bufPtr, &writeSize, &writeFileOffset, &writeBufOffset, &bytesWritten, stream);
  fileError = cuFileWriteAsync(fileHandle, bufPtr, &writeSize, &writeFileOffset, &writeBufOffset, &bytesWritten, stream);
#endif

#if CUDA_VERSION >= 12050
  // CHECK: hipFileSizeTConfigParameter_t sizeTParam;
  CUFileSizeTConfigParameter_t sizeTParam;

  // CHECK: hipFileBoolConfigParameter_t boolParam;
  CUFileBoolConfigParameter_t boolParam;

  // CHECK: hipFileStringConfigParameter_t stringParam;
  CUFileStringConfigParameter_t stringParam;

  size_t sizeTValue = 0;
  bool boolValue = false;
  char stringValue[256];

  // CUDA: CUfileError_t cuFileGetParameterSizeT(CUFileSizeTConfigParameter_t param, size_t *value);
  // HIP: hipFileError_t hipFileGetParameterSizeT(hipFileSizeTConfigParameter_t param, size_t *value);
  // CHECK: fileError = hipFileGetParameterSizeT(sizeTParam, &sizeTValue);
  fileError = cuFileGetParameterSizeT(sizeTParam, &sizeTValue);

  // CUDA: CUfileError_t cuFileGetParameterBool(CUFileBoolConfigParameter_t param, bool *value);
  // HIP: hipFileError_t hipFileGetParameterBool(hipFileBoolConfigParameter_t param, bool *value);
  // CHECK: fileError = hipFileGetParameterBool(boolParam, &boolValue);
  fileError = cuFileGetParameterBool(boolParam, &boolValue);

  // CUDA: CUfileError_t cuFileGetParameterString(CUFileStringConfigParameter_t param, char *desc_str, int len);
  // HIP: hipFileError_t hipFileGetParameterString(hipFileStringConfigParameter_t param, char *desc_str, int len);
  // CHECK: fileError = hipFileGetParameterString(stringParam, stringValue, 256);
  fileError = cuFileGetParameterString(stringParam, stringValue, 256);

  // CUDA: CUfileError_t cuFileSetParameterSizeT(CUFileSizeTConfigParameter_t param, size_t value);
  // HIP: hipFileError_t hipFileSetParameterSizeT(hipFileSizeTConfigParameter_t param, size_t value);
  // CHECK: fileError = hipFileSetParameterSizeT(sizeTParam, sizeTValue);
  fileError = cuFileSetParameterSizeT(sizeTParam, sizeTValue);

  // CUDA: CUfileError_t cuFileSetParameterBool(CUFileBoolConfigParameter_t param, bool value);
  // HIP: hipFileError_t hipFileSetParameterBool(hipFileBoolConfigParameter_t param, bool value);
  // CHECK: fileError = hipFileSetParameterBool(boolParam, boolValue);
  fileError = cuFileSetParameterBool(boolParam, boolValue);

  // CUDA: CUfileError_t cuFileSetParameterString(CUFileStringConfigParameter_t param, const char* desc_str);
  // HIP: hipFileError_t hipFileSetParameterString(hipFileStringConfigParameter_t param, const char* desc_str);
  // CHECK: fileError = hipFileSetParameterString(stringParam, stringValue);
  fileError = cuFileSetParameterString(stringParam, stringValue);
#endif

  return 0;
}
