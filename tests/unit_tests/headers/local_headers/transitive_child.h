#ifndef TRANSITIVE_CHILD_H
#define TRANSITIVE_CHILD_H

inline void child_sort_alloc(int *data, int n, void **p) {
    std::sort(data, data + n);
    cudaMalloc(p, n * sizeof(int));
}

#endif
