#define TILE 16

__global__ void matMult(const float* A, const float* B, float* C, int M, int K, int N, float alpha, float beta) {

    // Shared-memory tiles: one 16x16 patch of A and one of B,
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    // This thread's coordinates within its 16x16 block: each ranges 0..15.
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    // The output element C[row][col] that THIS thread is responsible for.
    // Fixed for the thread's entire life.
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    // Running dot-product accumulator for C[row][col].
    float dot_prod = 0.0f;

    // Walk across the K dimension one TILE-wide chunk at a time.
    // (K + TILE - 1) / TILE is a ceiling divide so we cover a K
    // that isn't a clean multiple of 16.
    for (int t = 0; t < (K + TILE  - 1) / TILE; t++) {

        // Which column of A / row of B this thread grabs THIS chunk.
        // aCol slides right, bRow slides down, as t advances.
        int aCol = t * TILE + tx;
        int bRow = t * TILE + ty;

        // Load this thread's one element of each matrix into shared memory.
        // Flattened index = rowIndex * width + colIndex.
        As[ty][tx] = A[row * K + aCol];
        Bs[ty][tx] = B[bRow * N + col];
    }
    __syncthreads();
}
 