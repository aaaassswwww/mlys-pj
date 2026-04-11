#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

__global__ void pointer_chase(const int* next_idx, int iters, unsigned long long* out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    int idx = 0;
    unsigned long long t0 = clock64();
    for (int i = 0; i < iters; ++i) {
        idx = next_idx[idx];
    }
    unsigned long long t1 = clock64();
    out[0] = (t1 - t0) / static_cast<unsigned long long>(iters);
    if (idx == -1) {
        out[0] = 0;
    }
}

int main() {
    constexpr int kElems = 1 << 18;  // ~1MB, usually around L2 region
    constexpr int kIters = 1 << 20;
    std::vector<int> host(kElems);
    for (int i = 0; i < kElems - 1; ++i) host[i] = i + 1;
    host[kElems - 1] = 0;

    int* d_next = nullptr;
    unsigned long long* d_out = nullptr;
    unsigned long long h_out = 0;
    if (cudaMalloc(&d_next, sizeof(int) * kElems) != cudaSuccess) {
        std::printf("metric=l2_latency_cycles value=0 samples=0 median=0 best=0 std=0\n");
        return 1;
    }
    if (cudaMalloc(&d_out, sizeof(unsigned long long)) != cudaSuccess) {
        cudaFree(d_next);
        std::printf("metric=l2_latency_cycles value=0 samples=0 median=0 best=0 std=0\n");
        return 1;
    }
    cudaMemcpy(d_next, host.data(), sizeof(int) * kElems, cudaMemcpyHostToDevice);
    pointer_chase<<<1, 1>>>(d_next, kIters, d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_out, d_out, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaFree(d_next);
    cudaFree(d_out);
    std::printf("metric=l2_latency_cycles value=%llu samples=1 median=%llu best=%llu std=0\n", h_out, h_out, h_out);
    return 0;
}
