#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include "src/kernels/naive.cuh"

int main() {
    // Larger matrix sizes for performance comparison.
    // Set VERIFY to 0 to skip the O(M*N*K) CPU reference check on big sizes.
    int M = 1024, N = 1024, K = 1024;
    const bool VERIFY = (M <= 1024 && N <= 1024 && K <= 1024);

    float *h_A = (float*)malloc(M * K * sizeof(float));
    float *h_B = (float*)malloc(K * N * sizeof(float));
    float *h_C = (float*)malloc(M * N * sizeof(float));

    for (int i = 0; i < M * K; i++) h_A[i] = (float)(rand() % 10);
    for (int i = 0; i < K * N; i++) h_B[i] = (float)(rand() % 10);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, M * K * sizeof(float));
    cudaMalloc(&d_B, K * N * sizeof(float));
    cudaMalloc(&d_C, M * N * sizeof(float));

    // --- copy inputs host -> device ---
    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    // --- launch configuration ---
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);

    // --- launch the kernel (timed) ---
    float alpha = 1.0f, beta = 0.0f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    matMult<<<grid, block>>>(d_A, d_B, d_C, alpha, beta, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    double gflops = (2.0 * M * N * K) / (ms * 1e6);
    printf("M=%d N=%d K=%d  time=%.3f ms  %.2f GFLOPS\n", M, N, K, ms, gflops);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // --- copy result device -> host ---
    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

    // --- print result (small sizes only) ---
    if (M <= 8 && N <= 8) {
        printf("C = A * B:\n");
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                printf("%6.1f ", h_C[i * N + j]);
            }
            printf("\n");
        }
    }

    // --- CPU reference + compare (skipped for large sizes) ---
    if (VERIFY) {
        float *ref = (float*)malloc(M * N * sizeof(float));
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < N; j++) {
                float acc = 0.0f;
                for (int k = 0; k < K; k++) {
                    acc += h_A[i * K + k] * h_B[k * N + j];
                }
                ref[i * N + j] = acc;
            }
        }
        bool ok = true;
        for (int i = 0; i < M * N; i++) {
            if (fabs(h_C[i] - ref[i]) > 1e-1) {
                ok = false;
                printf("Mismatch at %d: GPU=%.1f CPU=%.1f\n", i, h_C[i], ref[i]);
            }
        }
        printf(ok ? "PASS\n" : "FAIL\n");
        free(ref);
    } else {
        printf("Skipping CPU verification (matrix too large)\n");
    }

    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    return 0;
}
