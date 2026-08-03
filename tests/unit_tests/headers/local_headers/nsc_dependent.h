#ifndef NSC_DEPENDENT_H
#define NSC_DEPENDENT_H

// Deliberately not self-contained: NscVec comes from the includer. Driven by
// non_self_contained.cu, and excluded from standalone testing in tests/lit.cfg.

inline cudaError_t nsc_fill(NscVec *v, int count) {
    v->status = cudaMemset(&v->data, 0, count * sizeof(float3));
    return v->status;
}

#endif
