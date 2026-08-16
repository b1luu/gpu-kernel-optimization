__global__ void matMult(const float* A, const float*B, float*C, float alpha, float beta, int M, int N, int K) { 
    int row  = blockIdx.y * blockDim.y + threadIdx.y;
    int col  = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; k++){
        acc += A[row * K + k] * B[k* N + col];
    }
    C[row * N + col] = alpha * acc + beta * C[row * N + col];
}

