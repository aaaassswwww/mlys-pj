#include <cstdio>
#include <cuda_runtime.h>

__global__ void touch_smem() {
    extern __shared__ unsigned char smem[];
    if (threadIdx.x == 0) {
        smem[0] = 1;
    }
}

int main() {
    int low = 0;
    int high = 256 * 1024;  // conservative upper bound in bytes
    int best = 0;

    while (low <= high) {
        int mid = low + (high - low) / 2;
        touch_smem<<<1, 32, static_cast<size_t>(mid)>>>();
        cudaError_t err = cudaDeviceSynchronize();
        if (err == cudaSuccess) {
            best = mid;
            low = mid + 1;
        } else {
            cudaGetLastError();  // clear sticky error
            high = mid - 1;
        }
    }

    double kb = static_cast<double>(best) / 1024.0;
    std::printf("metric=max_shmem_per_block_kb value=%.3f samples=1 median=%.3f best=%.3f std=0\n", kb, kb, kb);
    return 0;
}
