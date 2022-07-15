// RUN: %run_test hipify "%s" "%t" %hipify_args %clang_args

/*
Copyright (c) 2015-present Advanced Micro Devices, Inc. All rights reserved.

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

// CHECK: #include <hip/hip_runtime.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Random predefiend 32 and 64 bit values
constexpr uint32_t value32 = 0x70F0F0FF;
constexpr uint64_t value64 = 0x7FFF0000FFFF0000;
constexpr unsigned int writeFlag = 0;

__global__
void fn(float* px, float* py) {
  bool a[42];
  __shared__ double b[69];
  for (auto&& x : b) x = *py++;
  for (auto&& x : a) x = *px++ > 0.0;
  for (auto&& x : a) if (x) *--py = *--px;
}

void testWrite() {
  int64_t* signalPtr;
  // CHECK: hipStream_t stream;
  cudaStream_t stream;
  // CHECK: hipStreamCreate(&stream);
  cudaStreamCreate(&stream);
  int64_t* host_ptr64 = (int64_t*)malloc(sizeof(int64_t));
  int32_t* host_ptr32 = (int32_t*)malloc(sizeof(int32_t));
  //  hipExtMallocWithFlags((void**)&signalPtr, 8, hipMallocSignalMemory);
  void* device_ptr64;
  void* device_ptr32;
  *host_ptr64 = 0x0;
  *host_ptr32 = 0x0;
  *signalPtr = 0x0;
  // CHECK: hipHostRegister(host_ptr64, sizeof(int64_t), 0);
  cudaHostRegister(host_ptr64, sizeof(int64_t), 0);
  // CHECK: hipHostRegister(host_ptr32, sizeof(int32_t), 0);
  cudaHostRegister(host_ptr32, sizeof(int32_t), 0);
  // CHECK: hipStreamWriteValue64(stream, hipDeviceptr_t(host_ptr64), value64, writeFlag);
  cuStreamWriteValue64(stream, CUdeviceptr(host_ptr64), value64, writeFlag);
  // CHECK: hipStreamWriteValue32(stream, hipDeviceptr_t(host_ptr32), value32, writeFlag);
  cuStreamWriteValue32(stream, CUdeviceptr(host_ptr32), value32, writeFlag);
  // CHECK: hipStreamSynchronize(stream);
  cudaStreamSynchronize(stream);
  // CHECK: hipHostGetDevicePointer((void**)&device_ptr64, host_ptr64, 0);
  cudaHostGetDevicePointer((void**)&device_ptr64, host_ptr64, 0);
  // CHECK: hipHostGetDevicePointer((void**)&device_ptr32, host_ptr32, 0);
  cudaHostGetDevicePointer((void**)&device_ptr32, host_ptr32, 0);
  // Reset values
  *host_ptr64 = 0x0;
  *host_ptr32 = 0x0;
  // CHECK: hipStreamWriteValue64(stream, hipDeviceptr_t(device_ptr64), value64, writeFlag);
  cuStreamWriteValue64(stream, CUdeviceptr(device_ptr64), value64, writeFlag);
  // CHECK: hipStreamWriteValue32(stream, hipDeviceptr_t(device_ptr32), value32, writeFlag);
  cuStreamWriteValue32(stream, CUdeviceptr(device_ptr32), value32, writeFlag);
  // CHECK: hipStreamSynchronize(stream);
  cudaStreamSynchronize(stream);
  // Test Writing to Signal Memory
  // CHECK: hipStreamWriteValue64(stream, hipDeviceptr_t(signalPtr), value64, writeFlag);
  cuStreamWriteValue64(stream, CUdeviceptr(signalPtr), value64, writeFlag);
  // CHECK: hipStreamSynchronize(stream);
  cudaStreamSynchronize(stream);
  // Cleanup
  // CHECK: hipStreamDestroy(stream);
  cudaStreamDestroy(stream);
  // CHECK: hipHostUnregister(host_ptr64);
  cudaHostUnregister(host_ptr64);
  // CHECK: hipHostUnregister(host_ptr32);
  cudaHostUnregister(host_ptr32);
  // CHECK: hipFree(signalPtr);
  cudaFree(signalPtr);
  free(host_ptr32);
  free(host_ptr64);
}

void testWait() {
  int64_t* signalPtr;
  // random data values
  int32_t DATA_INIT = 0x1234;
  int32_t DATA_UPDATE = 0X4321;

  struct TEST_WAIT {
    int compareOp;
    uint64_t mask;
    int64_t waitValue;
    int64_t signalValueFail;
    int64_t signalValuePass;
  };

  TEST_WAIT testCases[] = {
    {
      // mask will ignore few MSB bits
      // CHECK: hipStreamWaitValueGte,
      CU_STREAM_WAIT_VALUE_GEQ,
      0x0000FFFFFFFFFFFF,
      0x000000007FFF0001,
      0x7FFF00007FFF0000,
      0x000000007FFF0001
    },
    {
      // CHECK: hipStreamWaitValueGte,
      CU_STREAM_WAIT_VALUE_GEQ,
      0xF,
      0x4,
      0x3,
      0x6
    },
    {
      // mask will ignore few MSB bits
      // CHECK: hipStreamWaitValueEq,
      CU_STREAM_WAIT_VALUE_EQ,
      0x0000FFFFFFFFFFFF,
      0x000000000FFF0001,
      0x7FFF00000FFF0000,
      0x7F0000000FFF0001
    },
    {
      // CHECK: hipStreamWaitValueEq,
      CU_STREAM_WAIT_VALUE_EQ,
      0xFF,
      0x11,
      0x25,
      0x11
    },
    {
      // mask will discard bits 8 to 11
      // CHECK: hipStreamWaitValueAnd,
      CU_STREAM_WAIT_VALUE_AND,
      0xFF,
      0xF4A,
      0xF35,
      0X02
    },
    {
      // mask is set to ignore the sign bit.
      // CHECK: hipStreamWaitValueNor,
      CU_STREAM_WAIT_VALUE_NOR,
      0x7FFFFFFFFFFFFFFF,
      0x7FFFFFFFFFFFF247,
      0x7FFFFFFFFFFFFdbd,
      0x7FFFFFFFFFFFFdb5
    },
    {
      // mask is set to apply NOR for bits 0 to 3.
      // CHECK: hipStreamWaitValueNor,
      CU_STREAM_WAIT_VALUE_NOR,
      0xF,
      0x7E,
      0x7D,
      0x76
    }
  };

  struct TEST_WAIT32_NO_MASK {
    int compareOp;
    int32_t waitValue;
    int32_t signalValueFail;
    int32_t signalValuePass;
  };

  // default mask 0xFFFFFFFF will be used.
  TEST_WAIT32_NO_MASK testCasesNoMask32[] = {
    {
      // CHECK: hipStreamWaitValueGte,
      CU_STREAM_WAIT_VALUE_GEQ,
      0x7FFF0001,
      0x7FFF0000,
      0x7FFF0010
    },
    {
      // CHECK: hipStreamWaitValueEq,
      CU_STREAM_WAIT_VALUE_EQ,
      0x7FFFFFFF,
      0x7FFF0000,
      0x7FFFFFFF
    },
    {
      // CHECK: hipStreamWaitValueAnd,
      CU_STREAM_WAIT_VALUE_AND,
      0x70F0F0F0,
      0x0F0F0F0F,
      0X1F0F0F0F
    },
    {
      // CHECK: hipStreamWaitValueNor,
      CU_STREAM_WAIT_VALUE_NOR,
      0x7AAAAAAA,
      static_cast<int32_t>(0x85555555),
      static_cast<int32_t>(0x9AAAAAAA)
    }
  };

  struct TEST_WAIT64_NO_MASK {
    int compareOp;
    int64_t waitValue;
    int64_t signalValueFail;
    int64_t signalValuePass;
  };

  // default mask 0xFFFFFFFFFFFFFFFF will be used.
  TEST_WAIT64_NO_MASK testCasesNoMask64[] = {
    {
      // CHECK: hipStreamWaitValueGte,
      CU_STREAM_WAIT_VALUE_GEQ,
      0x7FFFFFFFFFFF0001,
      0x7FFFFFFFFFFF0000,
      0x7FFFFFFFFFFF0001
    },
    {
      // CHECK: hipStreamWaitValueEq,
      CU_STREAM_WAIT_VALUE_EQ,
      0x7FFFFFFFFFFFFFFF,
      0x7FFFFFFF0FFF0000,
      0x7FFFFFFFFFFFFFFF
    },
    {
      // CHECK: hipStreamWaitValueAnd,
      CU_STREAM_WAIT_VALUE_AND,
      0x70F0F0F0F0F0F0F0,
      0x0F0F0F0F0F0F0F0F,
      0X1F0F0F0F0F0F0F0F
    },
    {
      // CHECK: hipStreamWaitValueNor,
      CU_STREAM_WAIT_VALUE_NOR,
      0x4724724747247247,
      static_cast<int64_t>(0xbddbddbdbddbddbd),
      static_cast<int64_t>(0xbddbddbdbddbddb3)
    }
  };

  // CHECK: hipStream_t stream;
  cudaStream_t stream;
  // CHECK: hipStreamCreate(&stream);
  cudaStreamCreate(&stream);
  // hipExtMallocWithFlags((void**)&signalPtr, 8, hipMallocSignalMemory);
  int64_t* dataPtr64 = (int64_t*)malloc(sizeof(int64_t));
  int32_t* dataPtr32 = (int32_t*)malloc(sizeof(int32_t));
  // hipHostRegister(dataPtr64, sizeof(int64_t), 0);
  cudaHostRegister(dataPtr64, sizeof(int64_t), 0);
  // CHECK: hipHostRegister(dataPtr32, sizeof(int32_t), 0);
  cudaHostRegister(dataPtr32, sizeof(int32_t), 0);
  // Run-1: streamWait is blocking (wait conditions is false)
  // Run-2: streamWait is non-blocking (wait condition is true)
  for (int run = 0; run < 2; run++) {
    bool isBlocking = run == 0;
    for (const auto & tc : testCases) {
      *signalPtr = isBlocking ? tc.signalValueFail : tc.signalValuePass;
      *dataPtr64 = DATA_INIT;
      // CHECK: hipStreamWaitValue64(stream, hipDeviceptr_t(signalPtr), tc.waitValue, tc.compareOp);
      cuStreamWaitValue64(stream, CUdeviceptr(signalPtr), tc.waitValue, tc.compareOp);
      // CHECK: hipStreamWriteValue64(stream, hipDeviceptr_t(dataPtr64), DATA_UPDATE, writeFlag);
      cuStreamWriteValue64(stream, CUdeviceptr(dataPtr64), DATA_UPDATE, writeFlag);
      if (isBlocking) {
        // Trigger an implict flush and verify stream has pending work.
        // CHECK: if (hipStreamQuery(stream) != hipErrorNotReady) {}
        if (cudaStreamQuery(stream) != cudaErrorNotReady) {}
        // update signal to unblock the wait.
        *signalPtr = tc.signalValuePass;
      }
      // CHECK: if (hipStreamQuery(stream) != hipSuccess) {}
      if (cudaStreamQuery(stream) != cudaSuccess) {}
      // CHECK: hipStreamSynchronize(stream);
      cudaStreamSynchronize(stream);
      if (*dataPtr64 != DATA_UPDATE) {}
      // 32-bit API
      *signalPtr = isBlocking ? tc.signalValueFail : tc.signalValuePass;
      *dataPtr32 = DATA_INIT;
      // CHECK: hipStreamWaitValue32(stream, hipDeviceptr_t(signalPtr), tc.waitValue, tc.compareOp);
      cuStreamWaitValue32(stream, CUdeviceptr(signalPtr), tc.waitValue, tc.compareOp);
      // CHECK: hipStreamWriteValue32(stream, hipDeviceptr_t(dataPtr32), DATA_UPDATE, writeFlag);
      cuStreamWriteValue32(stream, CUdeviceptr(dataPtr32), DATA_UPDATE, writeFlag);
      if (isBlocking) {
        // Trigger an implict flush and verify stream has pending work.
        // CHECK: if (hipStreamQuery(stream) != hipErrorNotReady) {}
        if (cudaStreamQuery(stream) != cudaErrorNotReady) {}
        // update signal to unblock the wait.
        *signalPtr = static_cast<int32_t>(tc.signalValuePass);
      }
      // CHECK: hipStreamSynchronize(stream);
      cudaStreamSynchronize(stream);
      if (*dataPtr32 != DATA_UPDATE) {}
    }
  }
  // Run-1: streamWait is blocking (wait conditions is false)
  // Run-2: streamWait is non-blocking (wait condition is true)
  for (int run = 0; run < 2; run++) {
    bool isBlocking = run == 0;
    for (const auto& tc : testCasesNoMask32) {
      *signalPtr = isBlocking ? tc.signalValueFail : tc.signalValuePass;
      *dataPtr32 = DATA_INIT;
      // CHECK: hipStreamWaitValue32(stream, hipDeviceptr_t(signalPtr), tc.waitValue, tc.compareOp);
      cuStreamWaitValue32(stream, CUdeviceptr(signalPtr), tc.waitValue, tc.compareOp);
      // CHECK: hipStreamWriteValue32(stream, hipDeviceptr_t(dataPtr32), DATA_UPDATE, writeFlag);
      cuStreamWriteValue32(stream, CUdeviceptr(dataPtr32), DATA_UPDATE, writeFlag);
      if (isBlocking) {
        // Trigger an implict flush and verify stream has pending work.
        // CHECK: if (hipStreamQuery(stream) != hipErrorNotReady) {}
        if (cudaStreamQuery(stream) != cudaErrorNotReady) {}
        // update signal to unblock the wait.
        *signalPtr = tc.signalValuePass;
      }
      // CHECK: hipStreamSynchronize(stream);
      cudaStreamSynchronize(stream);
      if (*dataPtr32 != DATA_UPDATE) {}
    }
  }
  // Run-1: streamWait is blocking (wait conditions is false)
  // Run-2: streamWait is non-blocking (wait condition is true)
  for (int run = 0; run < 2; run++) {
    bool isBlocking = run == 0;
    for (const auto& tc : testCasesNoMask64) {
      *signalPtr = isBlocking ? tc.signalValueFail : tc.signalValuePass;
      *dataPtr64 = DATA_INIT;
      // CHECK: hipStreamWaitValue64(stream, hipDeviceptr_t(signalPtr), tc.waitValue, tc.compareOp);
      cuStreamWaitValue64(stream, CUdeviceptr(signalPtr), tc.waitValue, tc.compareOp);
      // CHECK: hipStreamWriteValue64(stream, hipDeviceptr_t(dataPtr64), DATA_UPDATE, writeFlag);
      cuStreamWriteValue64(stream, CUdeviceptr(dataPtr64), DATA_UPDATE, writeFlag);
      if (isBlocking) {
        // Trigger an implict flush and verify stream has pending work.
        // CHECK: if (hipStreamQuery(stream) != hipErrorNotReady) {}
        if (cudaStreamQuery(stream) != cudaErrorNotReady) {}
        // update signal to unblock the wait.
        *signalPtr = tc.signalValuePass;
      }
      // CHECK: hipStreamSynchronize(stream);
      cudaStreamSynchronize(stream);
      if (*dataPtr64 != DATA_UPDATE) {}
    }
  }
  // Cleanup
  // CHECK: hipFree(signalPtr);
  cudaFree(signalPtr);
  // CHECK: hipHostUnregister(dataPtr64);
  cudaHostUnregister(dataPtr64);
  // CHECK: hipHostUnregister(dataPtr32);
  cudaHostUnregister(dataPtr32);
  free(dataPtr64);
  free(dataPtr32);
  // CHECK: hipStreamDestroy(stream);
  cudaStreamDestroy(stream);
}

int main() {
  // CHECK: hipFuncCache_t cacheConfig;
  cudaFuncCache cacheConfig;
  void* func;
  // CHECK: hipFuncSetCacheConfig(func, cacheConfig);
  cudaFuncSetCacheConfig(func, cacheConfig);
  // CHECK: hipFuncAttributes attr{};
  cudaFuncAttributes attr{};
  // CHECK: auto r = hipFuncGetAttributes(&attr, &fn);
  auto r = cudaFuncGetAttributes(&attr, &fn);
  // CHECK: if (r != hipSuccess || attr.maxThreadsPerBlock == 0) {
  if (r != cudaSuccess || attr.maxThreadsPerBlock == 0) {
    return 1;
  }
  testWrite();
  testWait();
  return 0;
}
