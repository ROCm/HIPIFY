#ifndef DIAMOND_RIGHT_H
#define DIAMOND_RIGHT_H

#include "diamond_shared.h"

inline void right_free(void *p) {
    // CHECK: hipFree(p);
    cudaFree(p);
}

#endif
