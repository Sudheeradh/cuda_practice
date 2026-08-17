#include <cuda_runtime.h>

__global__ void final_sum(float* partials, float* output, int N){

    const int tid = threadIdx.x;
    const int block_dim = blockDim.x;

    __shared__ float stemp[256];
    stemp[tid] = 0;
    int idx;

    for (int i = 0; i < N; i += block_dim) {
        idx = i + tid;
        if (idx < N) stemp[tid] += partials[idx];
    }
    __syncthreads();

    for (int stride = block_dim / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            stemp[tid] += stemp[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *output = stemp[tid];
    }
}

__global__ void sum_partials(const float* input, float* partials, int N){
    const int tid = threadIdx.x;
    const int block_dim = blockDim.x;
    const int block_idx = blockIdx.x;

    const int idx = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float stemp[256];
    if (idx < N){
        stemp[tid] = input[idx];
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

    partials[block_idx] = stemp[0];

}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    float* d_partials = nullptr;
    cudaMalloc(&d_partials, blocksPerGrid * sizeof(float));

    sum_partials<<<blocksPerGrid, threadsPerBlock>>>(input, d_partials, N);
    final_sum<<<1, threadsPerBlock>>>(d_partials, output, blocksPerGrid);

    cudaFree(d_partials);
}

