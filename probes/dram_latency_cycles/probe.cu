#include <cstdint>
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

__global__ void measure_latency(const int* __restrict__ next_idx, int start, int iters, unsigned long long* out_cycles) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    int idx = start;
    unsigned long long t0 = clock64();
    for (int i = 0; i < iters; ++i) {
        idx = next_idx[idx];
    }
    unsigned long long t1 = clock64();
    out_cycles[0] = (t1 - t0) / static_cast<unsigned long long>(iters);

    // keep a visible side effect
    if (idx == -1) {
        out_cycles[0] = 0;
    }
}

int main() {
    constexpr int kElems = 1 << 22;  // ~16MB working set
    constexpr int kIters = 1 << 20;

    std::vector<int> host(kElems);
    for (int i = 0; i < kElems - 1; ++i) {
        host[i] = i + 1;
    }
    host[kElems - 1] = 0;

    int* d_next = nullptr;
    unsigned long long* d_cycles = nullptr;
    unsigned long long h_cycles = 0;

    if (cudaMalloc(&d_next, sizeof(int) * kElems) != cudaSuccess) {
        std::printf("metric=dram_latency_cycles value=0 samples=0 median=0 best=0 std=0\n");
        return 1;
    }
    if (cudaMalloc(&d_cycles, sizeof(unsigned long long)) != cudaSuccess) {
        cudaFree(d_next);
        std::printf("metric=dram_latency_cycles value=0 samples=0 median=0 best=0 std=0\n");
        return 1;
    }
    cudaMemcpy(d_next, host.data(), sizeof(int) * kElems, cudaMemcpyHostToDevice);

    // warmup
    measure_latency<<<1, 1>>>(d_next, 0, 1024, d_cycles);
    cudaDeviceSynchronize();

    // measured pass
    measure_latency<<<1, 1>>>(d_next, 0, kIters, d_cycles);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    cudaFree(d_next);
    cudaFree(d_cycles);

    std::printf(
        "metric=dram_latency_cycles value=%llu samples=1 median=%llu best=%llu std=0\n",
        h_cycles,
        h_cycles,
        h_cycles
    );
    return 0;
}
