#include <cuda_runtime.h>
#include "src/kernels/register_blocked.cuh"
#include "bench_harness.cuh"

int main() {
    int M = 1024, N = 1024, K = 1024;

    dim3 block(BM * BN / TM);
    dim3 grid(N / BN, M / BM);

    runBenchmark("register_blocked", M, N, K, [=](float* dA, float* dB, float* dC, float alpha, float beta) {
        matMult<<<grid, block>>>(dA, dB, dC, alpha, beta, M, N, K);
    });

    return 0;
}
