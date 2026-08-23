#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

// Identical to vectorized.cuh except As/Bs are padded by 4 columns (not
// 1): As[BK][BM+4], Bs[BK][BN+4]. Padding by exactly 1 float — the usual
// bank-conflict trick — would break 16-byte alignment for the float4
// reads/writes below (row stride 129*4=516 bytes isn't a multiple of
// 16), which is unsafe for reinterpret_cast<float4*> on shared memory.
// Padding by 4 floats (16 bytes) still breaks BM/BN's stride-32-element
// periodicity (128 -> 132, no longer a multiple of 32) while keeping
// every row's start float4-aligned.
//
// This targets INTER-row bank conflicts (different rows landing on the
// same bank because row length is a multiple of 32). vectorized.cuh's
// profile flagged conflicts on shared loads/stores, but that doesn't
// guarantee this is the right fix: the regN load conflict looks like
// it's actually INTRA-row (same dotIdx row, threadCol*TN stride within
// that one row), which row padding doesn't change at all. Built to test
// that empirically rather than assume either way.
//
// Safe to pad: every vectorized (float4) read/write in the loop below
// stays within a single row's BM or BN valid columns and never crosses
// into the new padding columns, so no other indexing changes needed.

__global__ void matMult(const float* A, const float* B, float* C, float alpha, float beta, int M, int N, int K){
    __shared__ float As[BK][BM + 4]; // padded (must be a multiple of 4 for float4 alignment)
    __shared__ float Bs[BK][BN + 4]; // padded (must be a multiple of 4 for float4 alignment)

    int blockRow = blockIdx.y * BM;
    int blockCol = blockIdx.x * BN;

    int threadRow = threadIdx.x / (BN/TN);
    int threadCol = threadIdx.x % (BN/TN);

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

        for (int dotIdx = 0; dotIdx < BK; dotIdx++){
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

    for (int i = 0; i < TM; i++) {
        int row = blockRow + threadRow * TM + i;
        for (int v = 0; v < TN/4; v++) {
            int col = blockCol + threadCol * TN + v*4;
            float4 out;
            out.x = alpha * threadResults[i*TN + v*4 + 0] + beta * C[row*N + col + 0];
            out.y = alpha * threadResults[i*TN + v*4 + 1] + beta * C[row*N + col + 1];
            out.z = alpha * threadResults[i*TN + v*4 + 2] + beta * C[row*N + col + 2];
            out.w = alpha * threadResults[i*TN + v*4 + 3] + beta * C[row*N + col + 3];
            reinterpret_cast<float4*>(&C[row*N + col])[0] = out;
        }
    }
}
