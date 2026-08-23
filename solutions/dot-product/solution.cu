#include <cuda_runtime.h>

__global__ void final_sum(const float* partials, float* output, int N) {
    const int tid = threadIdx.x;
    const int block_dim = blockDim.x;
    const int block_idx = blockIdx.x;

    const int idx = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float stemp[256];

    stemp[tid] = 0;

    for (int i = 0; i < N; i += block_dim) {
        if (tid + i < N) stemp[tid] += partials[tid + i];
    }

    __syncthreads();

    for (int stride = block_dim / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            stemp[tid] += stemp[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) *output = stemp[0];
}

__global__ void dot(const float* A, const float* B, float* d_partials, int N) {
    const int tid = threadIdx.x;
    const int block_dim = blockDim.x;
    const int block_idx = blockIdx.x;

    const int idx = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float stemp[256];

    if (idx < N) {
        stemp[tid] = A[idx] * B[idx];
    } else {
        stemp[tid] = 0;
    }

    __syncthreads();

    for (int stride = block_dim / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            stemp[tid] += stemp[tid + stride];
        }
        __syncthreads();
    }

    d_partials[block_idx] = stemp[0];

}

// A, B, result are device pointers
extern "C" void solve(const float* A, const float* B, float* result, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    float* d_partials = nullptr;
    cudaMalloc(&d_partials, blocksPerGrid * sizeof(float));

    dot<<<blocksPerGrid, threadsPerBlock>>>(A, B, d_partials, N);
    final_sum<<<1, threadsPerBlock>>>(d_partials, result, blocksPerGrid);

    cudaFree(d_partials);
}
