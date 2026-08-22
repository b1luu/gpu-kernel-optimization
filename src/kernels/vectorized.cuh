#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

// threads per block = (BM * BN) / (TM * TN) = 256
//
// As tile = BM*BK = 1024 scalars = 256 float4s — exactly 1 per thread.
// Bs tile = BK*BN = 1024 scalars = 256 float4s — exactly 1 per thread.
// (This is why these specific numbers, not the 64/64/4/4 config: they're
// chosen so the float4 tile loads divide evenly across threads with no
// stride loop needed — unlike register_blocked_2d.cuh's scalar loads.)
//
// As is stored TRANSPOSED: As[BK][BM], not As[BM][BK] — same reasoning
// as before, so regM's TM=8 consecutive M-values are contiguous.
//
// TM=TN=8 but a float4 only holds 4 floats, so regM/regN/the output
// store each need TM/4 = TN/4 = 2 float4 operations, not 1.

__global__ void matMult(const float* A, const float* B, float* C, float alpha, float beta, int M, int N, int K){
    __shared__ float As[BK][BM]; // Transpose As
    __shared__ float Bs[BK][BN];

    int blockRow = blockIdx.y * BM;     
    int blockCol = blockIdx.x * BN;

    int threadRow = threadIdx.x / (BN/TN); 
    int threadCol = threadIdx.x % (BN/TN); 

    // ---- cooperative load indices ----
    // No stride loop this time: (BM*BK)/4 and (BK*BN)/4 both equal the
    // thread count exactly, so each thread loads exactly one float4 of
    // each tile.

    int innerRowA = threadIdx.x / (BK/4);
    int innerColA = threadIdx.x % (BK/4);

    int innerRowB = threadIdx.x / (BN/4);
    int innerColB = threadIdx.x % (BN/4);

    float threadResults[TM * TN] = {0.0f};

    float regM[TM];
    float regN[TN];

    for (int tileIdx = 0; tileIdx < K; tileIdx += BK) {
        float4 tmpA = reinterpret_cast<const float4*>
        (&A[(blockRow + innerRowA) * K + tileIdx + innerColA * 4])[0];
        As[innerColA*4 + 0][innerRowA] = tmpA.x;
        As[innerColA*4 + 1][innerRowA] = tmpA.y;
        As[innerColA*4 + 2][innerRowA] = tmpA.z;
        As[innerColA*4 + 3][innerRowA] = tmpA.w;
        
        reinterpret_cast<float4*>(&Bs[innerRowB][innerColB*4])[0] =
            reinterpret_cast<const float4*>(
                &B[(tileIdx + innerRowB) * N + blockCol + innerColB*4])[0];

        __syncthreads();

        for (int dotIdx = 0; dotIdx < BK; dotIdx ++){
            for (int v = 0; v < TM / 4; v++) {
                reinterpret_cast<float4*>(regM)[v] = 
                    reinterpret_cast<const float4*>(&As[dotIdx][threadRow*TM + v*4])[0];
            }

            for (int v = 0; v < TN / 4; v++) {
                reinterpret_cast<float4*>(regN)[v] =
                    reinterpret_cast<const float4*>(&Bs[dotIdx][threadCol*TN + v*4])[0];
            }

            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    threadResults[i * TN + j] += regM[i] * regN[j];
                }
            }
        }
        __syncthreads();
    }   

    // ---- vectorized STORE to C — 2 float4 stores per row (TN/4), not 1 ----
    for (int i = 0; i < TM; i++) {
        int row = blockRow + threadRow * TM + i;
        for (int v = 0; v < TN/4; v++) {
            int col = blockCol + threadCol * TN + v*4;   // start of this float4 group
            float4 out;
            out.x = alpha * threadResults[i*TN + v*4 + 0] + beta * C[row*N + col + 0];
            out.y = alpha * threadResults[i*TN + v*4 + 1] + beta * C[row*N + col + 1];
            out.z = alpha * threadResults[i*TN + v*4 + 2] + beta * C[row*N + col + 2];
            out.w = alpha * threadResults[i*TN + v*4 + 3] + beta * C[row*N + col + 3];
            reinterpret_cast<float4*>(&C[row*N + col])[0] = out;
        }
    }
}