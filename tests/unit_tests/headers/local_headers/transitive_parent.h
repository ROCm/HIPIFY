#ifndef TRANSITIVE_PARENT_H
#define TRANSITIVE_PARENT_H

// CHECK: #include <hip/hip_runtime.h>
// CHECK-NOT: #include <cuda_runtime.h>
// CHECK: #include "transitive_child.h"
#include <cuda_runtime.h>
#include "transitive_child.h"

inline void parent_sync() {
    // CHECK: hipDeviceSynchronize();
    cudaDeviceSynchronize();
}

#endif
