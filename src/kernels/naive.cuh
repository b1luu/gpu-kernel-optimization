/**
 * CUDA kernel for General Matrix Multiplication (GEMM):
 *   C = alpha * (A * B) + beta * C
 *
 * Matrix dimensions (row-major order):
 *   A: M x K
 *   B: K x N
 *   C: M x N
 */
__global__ void matMult(const float* A, const float* B, float* C, 
                        float alpha, float beta, int M, int N, int K) { 
    // Map the y-dimension to columns and the x-dimension to rows.
    // Note: Because consecutive threads in a warp vary along threadIdx.x,
    // mapping `row` to x causes strided (non-coalesced) global memory
    // accesses for A and C in row-major order.
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int row = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard against out-of-bounds threads when matrix dimensions
    // are not exact multiples of the thread block dimensions
    if (row >= M || col >= N) return;

    // Accumulate the dot product of row `row` from A and column `col` from B
    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        acc += A[row * K + k] * B[k * N + col];
    }

    // Scale the result by alpha, add the scaled existing value of C, and write back
    C[row * N + col] = alpha * acc + beta * C[row * N + col];
}