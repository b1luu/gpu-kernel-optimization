#define BM 64   // block tile: rows of C this block computes
#define BN 64   // block tile: cols of C this block computes
#define BK 8    // depth of each K-slice staged into shared memory
#define TM 8    // outputs each thread computes, stacked down the M dimension
// threads per block = (BM * BN) / TM = 512
// chosen so BM*BK == BK*BN == threads-per-block: each thread loads exactly
// one element of As and one element of Bs, no stride loop needed.
// Assumes M, N, K are exact multiples of BM/BN/BK (true for 1024/1024/1024) —
// no boundary guards below, unlike shared_mem.cuh.

__global__ void matMult(const float* A, const float* B, float* C, float alpha, float beta, int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // ---- top-left corner of the BM x BN output tile this block owns ----
    int blockRow = blockIdx.y * BM;
    int blockCol = blockIdx.x * BN;

    // ---- this thread's position within the tile ----
    // BN threads span the tile's width; BM/TM threads span its height.
    // threadIdx.x runs 0..511 — split it into (threadRow, threadCol).
    int threadRow = threadIdx.x / BN;
    int threadCol = threadIdx.x % BN;

    // ---- cooperative load indices (independent of threadRow/threadCol —
    //      As/Bs have different shapes than the output tile) ----
    int innerRowA = threadIdx.x / BK;  
    int innerColA = threadIdx.x % BK;   
    int innerRowB = threadIdx.x / BN;
    int innerColB = threadIdx.x % BN;   

    float threadResults[TM] = {0.0f};

    for (int tileIdx = 0; tileIdx < K; tileIdx += BK) {
        As[innerRowA][innerColA] = A[(blockRow + innerRowA) * K + tileIdx + innerColA];  // one element of A, from global memory
        Bs[innerRowB][innerColB] = B[(tileIdx + innerRowB) * N + blockCol + innerColB];  // one element of B, from global memory
        __syncthreads();

        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            float tmpB = Bs[dotIdx][threadCol];   // one shared-memory load...
            for (int resIdx = 0; resIdx < TM; resIdx++) {
                // ...reused TM times here instead of once. This is the fix
                // for the MIO throttle stalls the profiling turned up:
                // one Bs load now feeds TM FMAs instead of one.
                threadResults[resIdx] += As[threadRow * TM + resIdx][dotIdx] * tmpB;
            }
        }
        __syncthreads();
    }

    for (int resIdx = 0; resIdx < TM; resIdx++) {
        int row = blockRow + threadRow * TM + resIdx;
        int col = blockCol + threadCol;
        C[row * N + col] = alpha * threadResults[resIdx] + beta * C[row * N + col];
    }
}

