#define TILE 16

__global__ void matMult(const float* A, const float* B, float* C, float alpha, float beta, int M, int N, int K){
    __shared__ float As[TILE][TILE];   // shared tile for A (reused each t)
    __shared__ float Bs[TILE][TILE];   // shared tile for B (reused each t)

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockIdx.y * blockDim.y + ty;   // this thread's output row (.y = down)
    int col = blockIdx.x * blockDim.x + tx;   // this thread's output col (.x = across)

    float dot_prod = 0.0f;                     // per-thread accumulator, survives all t

    // slide a 16-wide window across K, one chunk per iteration
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {
        int aCol = t*TILE + tx;   // A column this thread grabs (slides right)
        int bRow = t*TILE + ty;   // B row this thread grabs (slides down)

        // guarded load: real element if in-bounds, else 0.0f padding
        if (row < M && aCol < K)
            As[ty][tx] = A[row * K + aCol];
        else
            As[ty][tx] = 0.0f;

        if (bRow < K && col < N)
            Bs[ty][tx] = B[bRow * N + col];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();   // barrier 1: tile fully loaded before anyone reads
        for (int k = 0; k < TILE; k++) {
            dot_prod += As[ty][k] * Bs[k][tx];
        }
        __syncthreads();

    }
    if (row < M && col < N)
        C[row * N + col] = alpha * dot_prod + beta * C[row * N + col];
    
}