#include <cstdio>
#include <cuda_runtime.h>

__global__ void measure_l1(unsigned long long* out) {
    __shared__ int data[128];
    int idx = threadIdx.x & 127;
    data[idx] = idx;
    __syncthreads();

    unsigned long long t0 = clock64();
    int v = idx;
    #pragma unroll 64
    for (int i = 0; i < 1024; ++i) {
        v = data[(v + 1) & 127];
    }
    unsigned long long t1 = clock64();
    if (threadIdx.x == 0) {
        out[0] = (t1 - t0) / 1024ULL;
    }
}

int main() {
    unsigned long long* d_out = nullptr;
    unsigned long long h_out = 0;
    if (cudaMalloc(&d_out, sizeof(unsigned long long)) != cudaSuccess) {
        std::printf("metric=l1_latency_cycles value=0 samples=0 median=0 best=0 std=0\n");
        return 1;
    }
    measure_l1<<<1, 128>>>(d_out);
    cudaDeviceSynchronize();
    cudaMemcpy(&h_out, d_out, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    cudaFree(d_out);
    std::printf("metric=l1_latency_cycles value=%llu samples=1 median=%llu best=%llu std=0\n", h_out, h_out, h_out);
    return 0;
}
