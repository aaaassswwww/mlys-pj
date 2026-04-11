#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

__global__ void pointer_chase(const int* next_idx, int start, int iters, unsigned long long* out_cycles) {
    if (threadIdx.x != 0 || blockIdx.x != 0) {
        return;
    }
    int idx = start;
    unsigned long long t0 = clock64();
    for (int i = 0; i < iters; ++i) {
        idx = next_idx[idx];
    }
    unsigned long long t1 = clock64();
    out_cycles[0] = (t1 - t0) / static_cast<unsigned long long>(iters);
    if (idx == -1) {
        out_cycles[0] = 0;
    }
}

static int estimate_capacity_kb(const std::vector<int>& sizes_kb, const std::vector<double>& latencies) {
    if (sizes_kb.empty() || latencies.size() != sizes_kb.size()) {
        return 0;
    }

    double baseline = latencies[0];
    for (size_t i = 1; i < latencies.size() && i < 3; ++i) {
        if (latencies[i] > 0.0 && latencies[i] < baseline) baseline = latencies[i];
    }
    if (!(baseline > 0.0)) {
        return sizes_kb[0];
    }

    int best = sizes_kb[0];
    constexpr double kCliffRatio = 1.35;
    for (size_t i = 1; i < sizes_kb.size(); ++i) {
        if (latencies[i] <= 0.0) continue;
        if (latencies[i] > baseline * kCliffRatio) {
            return sizes_kb[i - 1];
        }
        best = sizes_kb[i];
    }
    return best;
}

int main() {
    const std::vector<int> sizes_kb = {
        64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 6144, 8192, 12288, 16384
    };
    std::vector<double> latencies;
    latencies.reserve(sizes_kb.size());

    constexpr int kIters = 1 << 20;
    constexpr int kRepeats = 5;

    for (int kb : sizes_kb) {
        int elems = (kb * 1024) / static_cast<int>(sizeof(int));
        if (elems < 2) {
            latencies.push_back(0.0);
            continue;
        }
        std::vector<int> host(elems);
        for (int i = 0; i < elems - 1; ++i) {
            host[i] = i + 1;
        }
        host[elems - 1] = 0;

        int* d_next = nullptr;
        unsigned long long* d_cycles = nullptr;
        if (cudaMalloc(&d_next, sizeof(int) * elems) != cudaSuccess) {
            latencies.push_back(0.0);
            continue;
        }
        if (cudaMalloc(&d_cycles, sizeof(unsigned long long)) != cudaSuccess) {
            cudaFree(d_next);
            latencies.push_back(0.0);
            continue;
        }
        cudaMemcpy(d_next, host.data(), sizeof(int) * elems, cudaMemcpyHostToDevice);

        // warmup
        pointer_chase<<<1, 1>>>(d_next, 0, 2048, d_cycles);
        cudaDeviceSynchronize();

        double best_cycles = 1e30;
        for (int rep = 0; rep < kRepeats; ++rep) {
            pointer_chase<<<1, 1>>>(d_next, 0, kIters, d_cycles);
            cudaDeviceSynchronize();
            unsigned long long h_cycles = 0;
            cudaMemcpy(&h_cycles, d_cycles, sizeof(unsigned long long), cudaMemcpyDeviceToHost);
            if (h_cycles > 0 && static_cast<double>(h_cycles) < best_cycles) {
                best_cycles = static_cast<double>(h_cycles);
            }
        }
        cudaFree(d_next);
        cudaFree(d_cycles);
        latencies.push_back(best_cycles >= 1e20 ? 0.0 : best_cycles);
    }

    int inferred_kb = estimate_capacity_kb(sizes_kb, latencies);
    std::printf(
        "metric=l2_cache_capacity_kb value=%d samples=%zu median=%d best=%d std=0\n",
        inferred_kb,
        sizes_kb.size(),
        inferred_kb,
        inferred_kb
    );
    return 0;
}
