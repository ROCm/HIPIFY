#ifndef DIAMOND_LEFT_H
#define DIAMOND_LEFT_H

#include "diamond_shared.h"

inline void left_sync() {
    // CHECK: hipDeviceSynchronize();
    cudaDeviceSynchronize();
}

#endif
