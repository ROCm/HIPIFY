#ifndef TRANSITIVE_CHILD_H
#define TRANSITIVE_CHILD_H

#include <algorithm>

inline void child_sort_alloc(int *data, int n, void **p) {
    std::sort(data, data + n);
    // CHECK: hipMalloc(p, n * sizeof(int));
    cudaMalloc(p, n * sizeof(int));
}

#endif
