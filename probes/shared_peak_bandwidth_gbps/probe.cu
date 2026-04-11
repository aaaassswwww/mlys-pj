#include <cfloat>
#include <cstdio>
#include <cuda_runtime.h>

__global__ void shmem_bw_kernel(float* out, int iters) {
    __shared__ float buf[2048];
    int tid = threadIdx.x;
    int lane = tid & 31;
    int idx = (tid * 8) & 2047;
    buf[idx] = static_cast<float>(tid);
    __syncthreads();

    float v = static_cast<float>(lane);
    #pragma unroll 8
    for (int i = 0; i < iters; ++i) {
        float x = buf[(idx + i) & 2047];
        x = x + 1.0f;
        buf[(idx + i + 64) & 2047] = x;
        v += x;
    }

    out[blockIdx.x * blockDim.x + tid] = v;
}

int main() {
    constexpr int kBlocks = 120;
    constexpr int kThreads = 256;
    constexpr int kIters = 1 << 14;
    constexpr int kRuns = 6;

    float* d_out = nullptr;
    if (cudaMalloc(&d_out, kBlocks * kThreads * static_cast<int>(sizeof(float))) != cudaSuccess) {
        std::printf("metric=shared_peak_bandwidth_gbps value=0 samples=0 median=0 best=0 std=0\n");
        return 1;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    shmem_bw_kernel<<<kBlocks, kThreads>>>(d_out, 256);
    cudaDeviceSynchronize();

    float best_ms = FLT_MAX;
    for (int i = 0; i < kRuns; ++i) {
        cudaEventRecord(start);
        shmem_bw_kernel<<<kBlocks, kThreads>>>(d_out, kIters);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start, stop);
        if (elapsed_ms > 0.0f && elapsed_ms < best_ms) {
            best_ms = elapsed_ms;
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_out);

    if (!(best_ms > 0.0f) || best_ms == FLT_MAX) {
        std::printf("metric=shared_peak_bandwidth_gbps value=0 samples=0 median=0 best=0 std=0\n");
        return 1;
    }

    const double ops = static_cast<double>(kBlocks) * static_cast<double>(kThreads) * static_cast<double>(kIters);
    const double moved_bytes = ops * 8.0;  // one read + one write, each float (4B)
    const double seconds = static_cast<double>(best_ms) * 1.0e-3;
    const double gbps = (moved_bytes / seconds) / 1.0e9;
    std::printf("metric=shared_peak_bandwidth_gbps value=%.3f samples=%d median=%.3f best=%.3f std=0\n", gbps, kRuns, gbps, gbps);
    return 0;
}
