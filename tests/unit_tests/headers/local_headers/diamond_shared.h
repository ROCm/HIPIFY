#ifndef DIAMOND_SHARED_H
#define DIAMOND_SHARED_H

inline void shared_alloc(void **p, size_t size) {
    // CHECK: hipMalloc(p, size);
    cudaMalloc(p, size);
}

#endif
