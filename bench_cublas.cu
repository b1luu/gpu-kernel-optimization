#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main() {
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

    cudaMemcpy(d_A, h_A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, K * N * sizeof(float), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0f, beta = 0.0f;
    const int WARMUP_RUNS = 3;
    const int TIMED_RUNS = 10;

    // cuBLAS is column-major; our A/B/C buffers are row-major.
    // Row-major C(MxN) = A(MxK)*B(KxN) is equivalent to column-major
    // C^T(NxM) = B^T(NxK) * A^T(KxM), which is what this call computes
    // by swapping A<->B and M<->N (no actual transpose op needed).
    for (int i = 0; i < WARMUP_RUNS; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K,
                    &alpha,
                    d_B, N,
                    d_A, K,
                    &beta,
                    d_C, N);
    }
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("cuBLAS launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Multiple independent trials, not just one: GPU boost-clock state can
    // vary between process launches (idle-ramp vs. already-busy), so a
    // single trial's average can be misleadingly high or low. Report
    // min/median across trials instead of trusting any one number.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    const int TRIALS = 5;
    float trialMs[TRIALS];
    for (int t = 0; t < TRIALS; t++) {
        cudaEventRecord(start);
        for (int i = 0; i < TIMED_RUNS; i++) {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                        N, M, K,
                        &alpha,
                        d_B, N,
                        d_A, K,
                        &beta,
                        d_C, N);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float total_ms = 0.0f;
        cudaEventElapsedTime(&total_ms, start, stop);
        trialMs[t] = total_ms / TIMED_RUNS;
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    float sortedMs[TRIALS];
    for (int t = 0; t < TRIALS; t++) sortedMs[t] = trialMs[t];
    for (int i = 1; i < TRIALS; i++) {
        float key = sortedMs[i];
        int j = i - 1;
        while (j >= 0 && sortedMs[j] > key) { sortedMs[j + 1] = sortedMs[j]; j--; }
        sortedMs[j + 1] = key;
    }
    float minMs = sortedMs[0];
    float medianMs = sortedMs[TRIALS / 2];
    double gflopsMin = (2.0 * M * N * K) / (minMs * 1e6);
    double gflopsMedian = (2.0 * M * N * K) / (medianMs * 1e6);

    printf("M=%d N=%d K=%d  trials(ms)=[", M, N, K);
    for (int t = 0; t < TRIALS; t++) printf("%.3f%s", trialMs[t], t < TRIALS - 1 ? ", " : "");
    printf("]  min=%.3f ms (%.2f GFLOPS)  median=%.3f ms (%.2f GFLOPS)\n",
           minMs, gflopsMin, medianMs, gflopsMedian);

    cudaMemcpy(h_C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

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
    cublasDestroy(handle);
    return 0;
}
