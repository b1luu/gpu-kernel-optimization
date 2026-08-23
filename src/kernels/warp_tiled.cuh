#define BM 64   // block tile
#define BN 64
#define BK 8
#define WARPS_M 2   // 2x2 warps per block = 4 warps = 128 threads/block
#define WARPS_N 2
#define WM (BM / WARPS_M)   // 32 — warp tile: the M x N region ONE warp owns
#define WN (BN / WARPS_N)   // 32
#define TM 8    // per-thread tile within the warp's WM x WN region
#define TN 4    // = float4 width, deliberately: makes adjacent lanes'
                // stores exactly adjacent 16-byte chunks (full coalescing,
                // no padding needed) and keeps regN a single float4 read
// threads per block = BM*BN/(TM*TN) = 4096/32 = 128 = 4 warps * 32 lanes
// lanes per warp arranged as (WM/TM) x (WN/TN) = 4 x 8 = 32
//
// Why this avoids both problems vectorized_padded.cuh only half-fixed:
// - WN=32 means one warp's slice of a Bs row is exactly one 32-bank
//   period wide. regN's TN=4 stride across 8 lanes tiles all 32 banks
//   exactly once per warp-instruction — no repeat, no conflict. (The old
//   vectorized.cuh's BN=128 meant a warp's read pattern repeated the
//   32-bank cycle 4x within one row, which IS what caused conflicts.)
// - TN=4 exactly matches float4, so adjacent lanes (laneCol, laneCol+1)
//   write starting addresses exactly 16 bytes apart — two lanes' float4s
//   fill one 32-byte sector completely, vs. the old TN=8 leaving a
//   16-byte gap between every pair of adjacent lanes' writes.
//
// Assumes M, N, K are exact multiples of BM/BN/BK (true for 1024^3).

__global__ void matMult(const float* A, const float* B, float* C, float alpha, float beta, int M, int N, int K){
    __shared__ float As[BK][BM];   // transposed, same reasoning as vectorized.cuh
    __shared__ float Bs[BK][BN];

    int blockRow = blockIdx.y * BM;
    int blockCol = blockIdx.x * BN;

    int warpId = threadIdx.x / 32;
    int lane   = threadIdx.x % 32;

    int warpRow = warpId / WARPS_N;   // 0..(WARPS_M-1)
    int warpCol = warpId % WARPS_N;   // 0..(WARPS_N-1)
    int warpRowStart = warpRow * WM;  // this warp's offset within the block tile
    int warpColStart = warpCol * WN;

    int laneRow = lane / (WN / TN);   // 0..(WM/TM - 1) = 0..3
    int laneCol = lane % (WN / TN);   // 0..(WN/TN - 1) = 0..7

    int rowInWarp = warpRowStart + laneRow * TM;   // this thread's tile origin
    int colInWarp = warpColStart + laneCol * TN;   // within the block tile

    // ---- cooperative load indices (block-level, all 128 threads, no
    //      stride loop needed: BM*BK/4 = BK*BN/4 = 128 = thread count) ----
    int innerRowA = threadIdx.x / (BK / 4);
    int innerColA = threadIdx.x % (BK / 4);

    int innerRowB = threadIdx.x / (BN / 4);
    int innerColB = threadIdx.x % (BN / 4);

    float threadResults[TM * TN] = {0.0f};
    float regM[TM];
    float regN[TN];

    for (int tileIdx = 0; tileIdx < K; tileIdx += BK) {
        float4 tmpA = reinterpret_cast<const float4*>(
            &A[(blockRow + innerRowA) * K + tileIdx + innerColA * 4])[0];
        As[innerColA * 4 + 0][innerRowA] = tmpA.x;
        As[innerColA * 4 + 1][innerRowA] = tmpA.y;
        As[innerColA * 4 + 2][innerRowA] = tmpA.z;
        As[innerColA * 4 + 3][innerRowA] = tmpA.w;

        reinterpret_cast<float4*>(&Bs[innerRowB][innerColB * 4])[0] =
            reinterpret_cast<const float4*>(
                &B[(tileIdx + innerRowB) * N + blockCol + innerColB * 4])[0];

        __syncthreads();

        for (int dotIdx = 0; dotIdx < BK; dotIdx++) {
            for (int v = 0; v < TM / 4; v++) {
                reinterpret_cast<float4*>(regM)[v] =
                    reinterpret_cast<const float4*>(&As[dotIdx][rowInWarp + v * 4])[0];
            }
            reinterpret_cast<float4*>(regN)[0] =
                reinterpret_cast<const float4*>(&Bs[dotIdx][colInWarp])[0];

            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    threadResults[i * TN + j] += regM[i] * regN[j];
                }
            }
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; i++) {
        int row = blockRow + rowInWarp + i;
        int col = blockCol + colInWarp;
        float4 out;
        out.x = alpha * threadResults[i * TN + 0] + beta * C[row * N + col + 0];
        out.y = alpha * threadResults[i * TN + 1] + beta * C[row * N + col + 1];
        out.z = alpha * threadResults[i * TN + 2] + beta * C[row * N + col + 2];
        out.w = alpha * threadResults[i * TN + 3] + beta * C[row * N + col + 3];
        reinterpret_cast<float4*>(&C[row * N + col])[0] = out;
    }
}
