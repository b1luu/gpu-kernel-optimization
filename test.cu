#include <cuda_runtime.h>
#include "src/kernels/coalesced.cuh"
#include "bench_harness.cuh"

int main() {
    int M = 1024, N = 1024, K = 1024;

    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    runBenchmark("coalesced", M, N, K, [=](float* dA, float* dB, float* dC, float alpha, float beta) {
        matMult<<<grid, block>>>(dA, dB, dC, alpha, beta, M, N, K);
    });

    return 0;
}
