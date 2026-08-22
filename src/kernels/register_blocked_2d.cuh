#define BM 64   // block tile: rows of C this block computes
#define BN 64   // block tile: cols of C this block computes
#define BK 8    // depth of each K-slice staged into shared memory
#define TM 4    // outputs each thread computes down the M dimension
#define TN 4    // outputs each thread computes across the N dimension
// threads per block = (BM * BN) / (TM * TN) = 256
// Unlike register_blocked.cuh, BM*BK (512) and BK*BN (512) no longer equal
// threads-per-block (256) — each thread loads 2 elements per array via a
// strided loop, not 1.
// Assumes M, N, K are exact multiples of BM/BN/BK (true for 1024/1024/1024).

__global__ void matMult(const float* A, const float* B, float* C, float alpha, float beta, int M, int N, int K) {
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // ---- top-left corner of the BM x BN output tile this block owns ----
    int blockRow = blockIdx.y * BM;
    int blockCol = blockIdx.x * BN;

    // ---- this thread's position in the (BM/TM) x (BN/TN) grid of TMxTN
    //      output tiles that cover this block's BM x BN region ----
    int threadRow = threadIdx.x / (BN/TN);  // 0 .. (BM/TM - 1)
    int threadCol = threadIdx.x % (BN/TN);   // 0 .. (BN/TN - 1)

    // ---- cooperative load indices + strides (independent of threadRow/Col) ----
    // As is BM x BK; this block has (BM*BN)/(TM*TN) threads, which is less
    // than BM*BK, so each thread loads BM*BK / numThreads elements by
    // striding down the BM dimension in a loop.
    int innerRowA = threadIdx.x / BK;
    int innerColA = threadIdx.x % BK;
    int strideA = ((BM * BN) / (TM * TN)) / BK;  // = numThreads / BK

    // Bs is BK x BN; same idea, striding down the BK dimension.
    int innerRowB = threadIdx.x / BN;
    int innerColB = threadIdx.x % BN;
    int strideB = ((BM * BN) / (TM * TN)) / BN;  // = numThreads / BN

    float threadResults[TM * TN] = {0.0f};
    float regM[TM];
    float regN[TN];

    for (int tileIdx = 0; tileIdx < K; tileIdx += BK) {
        for (int loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            As[innerRowA + loadOffset][innerColA] =
                A[(blockRow + innerRowA + loadOffset) * K + (innerColA + tileIdx)];  // one element of A, from global memory
        }
        for (int loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            Bs[innerRowB + loadOffset][innerColB] =
                B[(tileIdx + innerRowB + loadOffset) * N + blockCol + innerColB];  // one element of B, from global memory
        }
        __syncthreads();

        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            // cache this dotIdx's slice of As and Bs into registers once...
            for (int i = 0; i < TM; i++) regM[i] = /* ? */;
            for (int j = 0; j < TN; j++) regN[j] = /* ? */;

            // ...then reuse them for an outer product: TM*TN FMAs from
            // only TM+TN shared-memory reads (vs register_blocked.cuh's
            // TM FMAs from TM+1 reads — this is the whole point of 2D).
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    threadResults[i * TN + j] += /* ? */;
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            int row = /* ? */;
            int col = /* ? */;
            C[row * N + col] = alpha * threadResults[i * TN + j] + beta * C[row * N + col];
        }
    }
}

