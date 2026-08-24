#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "bench_harness.cuh"

int main() {
    int M = 1024, N = 1024, K = 1024;

    cublasHandle_t handle;
    cublasCreate(&handle);

    // cuBLAS is column-major; our A/B/C buffers are row-major.
    // Row-major C(MxN) = A(MxK)*B(KxN) is equivalent to column-major
    // C^T(NxM) = B^T(NxK) * A^T(KxM), which is what this call computes
    // by swapping A<->B and M<->N (no actual transpose op needed).
    runBenchmark("cuBLAS", M, N, K, [=](float* dA, float* dB, float* dC, float alpha, float beta) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    dB, N,
                    dA, K,
                    &beta,
                    dC, N);
    });

    cublasDestroy(handle);
    return 0;
}
