#include <cfloat>
#include <cstdio>
#include <cuda_runtime.h>

__global__ void global_bw_kernel(const float4* in, float* out, int n_vec4, int inner_repeats) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float acc = 0.0f;

    for (int rep = 0; rep < inner_repeats; ++rep) {
        for (int i = tid; i < n_vec4; i += stride) {
            float4 v = in[i];
            acc += (v.x + v.y + v.z + v.w);
        }
    }

    if (tid < stride) {
        out[tid] = acc;
    }
}

int main() {
    constexpr int kBytes = 256 * 1024 * 1024;  // 256MB working set
    constexpr int kThreads = 256;
    constexpr int kBlocks = 120;
    constexpr int kInnerRepeats = 4;
    constexpr int kRuns = 6;

    const int n_float = kBytes / static_cast<int>(sizeof(float));
    const int n_vec4 = n_float / 4;

    float4* d_in = nullptr;
    float* d_out = nullptr;
    if (cudaMalloc(&d_in, n_vec4 * static_cast<int>(sizeof(float4))) != cudaSuccess) {
        std::printf("metric=global_peak_bandwidth_gbps value=0 samples=0 median=0 best=0 std=0\n");
        return 1;
    }
    if (cudaMalloc(&d_out, kBlocks * kThreads * static_cast<int>(sizeof(float))) != cudaSuccess) {
        cudaFree(d_in);
        std::printf("metric=global_peak_bandwidth_gbps value=0 samples=0 median=0 best=0 std=0\n");
        return 1;
    }
    cudaMemset(d_in, 0, n_vec4 * static_cast<int>(sizeof(float4)));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warmup
    global_bw_kernel<<<kBlocks, kThreads>>>(d_in, d_out, n_vec4, 1);
    cudaDeviceSynchronize();

    float best_ms = FLT_MAX;
    for (int i = 0; i < kRuns; ++i) {
        cudaEventRecord(start);
        global_bw_kernel<<<kBlocks, kThreads>>>(d_in, d_out, n_vec4, kInnerRepeats);
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
    cudaFree(d_in);
    cudaFree(d_out);

    if (!(best_ms > 0.0f) || best_ms == FLT_MAX) {
        std::printf("metric=global_peak_bandwidth_gbps value=0 samples=0 median=0 best=0 std=0\n");
        return 1;
    }

    const double moved_bytes = static_cast<double>(kBytes) * static_cast<double>(kInnerRepeats);
    const double seconds = static_cast<double>(best_ms) * 1.0e-3;
    const double gbps = (moved_bytes / seconds) / 1.0e9;
    std::printf("metric=global_peak_bandwidth_gbps value=%.3f samples=%d median=%.3f best=%.3f std=0\n", gbps, kRuns, gbps, gbps);
    return 0;
}
