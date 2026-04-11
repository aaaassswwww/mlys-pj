#include <cstdio>
#include <cuda_runtime.h>

__global__ void bank_conflict_test(unsigned long long* out) {
    __shared__ int s[32 * 32];
    int tid = threadIdx.x;
    unsigned long long t0 = clock64();
    int v0 = 0;
    #pragma unroll 64
    for (int i = 0; i < 4096; ++i) {
        v0 += s[(tid % 32) * 32];  // conflict-heavy pattern
    }
    unsigned long long t1 = clock64();

    unsigned long long t2 = clock64();
    int v1 = 0;
    #pragma unroll 64
    for (int i = 0; i < 4096; ++i) {
        v1 += s[tid % 32];  // conflict-light pattern
    }
    unsigned long long t3 = clock64();

    if (tid == 0) {
        unsigned long long conflict = t1 - t0;
        unsigned long long no_conflict = t3 - t2;
        out[0] = (conflict > no_conflict) ? (conflict - no_conflict) / 4096ULL : 0ULL;
        if ((v0 + v1) == -1) out[0] = 0;
    }
}

int main() {
    unsigned long long* d_out = nullptr;
    unsigned long long h_penalty = 0;
    if (cudaMalloc(&d_out, sizeof(unsigned long long)) != cudaSuccess) {
        std::printf("metric=shmem_bank_conflict_penalty_cycles value=0 samples=0 median=0 best=0 std=0\n");
        return 1;
    }
    bank_conflict_test<<<1, 256>>>(d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_penalty, d_out, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    std::printf(
        "metric=shmem_bank_conflict_penalty_cycles value=%llu samples=1 median=%llu best=%llu std=0\n",
        h_penalty,
        h_penalty,
        h_penalty
    );
    return 0;
}
