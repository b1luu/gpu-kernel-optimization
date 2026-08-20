#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include "kernels/shared_mem.cuh"   

int main() {
    // ---- problem size ----
    int M = 512, K = 512, N = 512;
    float alpha = 1.0f, beta = 0.0f;

    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    // ---- host allocations ----
    float *hA = (float*)malloc(bytesA);
    float *hB = (float*)malloc(bytesB);
    float *hC = (float*)malloc(bytesC);

    // fill A and B with something (e.g. 1.0f or random) so the result is checkable
    for (int i = 0; i < M * K; i++) hA[i] = 1.0f;
    for (int i = 0; i < K * N; i++) hB[i] = 1.0f;

    // ---- device allocations ----
    float *dA, *dB, *dC;
    cudaMalloc(&dA, bytesA);
    cudaMalloc(&dB, bytesB);
    cudaMalloc(&dC, bytesC);

    // ---- copy inputs host -> device ----
    cudaMemcpy(dA, hA, bytesA, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, bytesB, cudaMemcpyHostToDevice);
    // (with beta=0 you don't strictly need to copy C in, but you must
    //  either copy or cudaMemset dC so it isn't garbage if beta != 0)

    // ---- launch config ----  
    dim3 block(16, 16);
    dim3 grid ( (N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    matMult<<<grid, block>>>(dA, dB, dC, alpha, beta, M, N, K);

    cudaDeviceSynchronize();   // wait for kernel to finish; also surfaces launch errors

    // ---- copy result device -> host ----
    cudaMemcpy(hC, dC, bytesC, cudaMemcpyDeviceToHost);

    // ---- quick check (A=B=1 → every C entry should equal K) ----
    printf("C[0] = %f (expected %d)\n", hC[0], K);

    // ---- cleanup ----
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB); free(hC);
    return 0;
}