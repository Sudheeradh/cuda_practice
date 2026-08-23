#include <cuda_runtime.h>

__global__ void final_sum(const float* d_partials, float* output, int N) {
    const int tid = threadIdx.x;
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int block_dim = blockDim.x;
    const int block_idx = blockIdx.x;

    __shared__ float smem[256];

    smem[tid] = 0;

    for (int i = 0; i < N; i += block_dim) {
        if (tid + i < N) smem[tid] += d_partials[tid + i];
    }
    __syncthreads();

    for (int stride = block_dim / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) *output = smem[0];

}

__global__ void reduce(const float* input, float* d_partials, int N) {
    const int tid = threadIdx.x;
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int block_dim = blockDim.x;
    const int block_idx = blockIdx.x;

    __shared__ float smem[256];

    if (idx < N) {
        smem[tid] = input[idx];
    } else {
        smem[tid] = 0;
    }

    __syncthreads();

    for (int stride = block_dim / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            smem[tid] += smem[tid + stride];
        }
        __syncthreads();
    }
    d_partials[block_idx] = smem[0];
}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    float* d_partials = nullptr;
    cudaMalloc(&d_partials, blocksPerGrid * sizeof(float));

    reduce<<<blocksPerGrid, threadsPerBlock>>>(input, d_partials, N);
    final_sum<<<1, threadsPerBlock>>>(d_partials, output, blocksPerGrid);
}